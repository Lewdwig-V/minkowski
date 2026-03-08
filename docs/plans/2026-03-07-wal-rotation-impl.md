# Segmented WAL Rotation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-file WAL with a directory of byte-size-capped segment files, enabling truncation of old entries via `delete_segments_before(seq)`.

**Architecture:** Each WAL becomes a directory of `wal-seq{:06}.seg` segment files. Each segment is self-describing (schema preamble). `append()` rolls to a new segment when the byte threshold is exceeded. `WalCursor` lazily advances across segments. Truncation deletes whole segment files.

**Tech Stack:** Rust, rkyv (existing), std::fs for directory/file management, tempfile (dev)

---

### Task 1: WalConfig + segment filename helpers

Add the configuration struct and pure helper functions for segment file naming and discovery. No behavior changes yet.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs` (top of file, after existing imports and before `Wal` struct)
- Modify: `crates/minkowski-persist/src/lib.rs` (add re-export)

**Step 1: Write the failing tests**

Add these tests at the bottom of the existing `#[cfg(test)] mod tests` block in `wal.rs`:

```rust
#[test]
fn segment_filename_format() {
    assert_eq!(segment_filename(0), "wal-seq000000.seg");
    assert_eq!(segment_filename(47), "wal-seq000047.seg");
    assert_eq!(segment_filename(123456), "wal-seq123456.seg");
}

#[test]
fn parse_segment_start_seq_valid() {
    assert_eq!(parse_segment_start_seq("wal-seq000000.seg"), Some(0));
    assert_eq!(parse_segment_start_seq("wal-seq000047.seg"), Some(47));
    assert_eq!(parse_segment_start_seq("wal-seq123456.seg"), Some(123456));
}

#[test]
fn parse_segment_start_seq_invalid() {
    assert_eq!(parse_segment_start_seq("not-a-segment.txt"), None);
    assert_eq!(parse_segment_start_seq("wal-seq.seg"), None);
    assert_eq!(parse_segment_start_seq("wal-seqABCDEF.seg"), None);
}

#[test]
fn list_segments_sorted() {
    let dir = tempfile::tempdir().unwrap();
    // Create files out of order
    std::fs::write(dir.path().join("wal-seq000100.seg"), b"").unwrap();
    std::fs::write(dir.path().join("wal-seq000000.seg"), b"").unwrap();
    std::fs::write(dir.path().join("wal-seq000050.seg"), b"").unwrap();
    std::fs::write(dir.path().join("not-a-segment.txt"), b"").unwrap();

    let segments = list_segments(dir.path()).unwrap();
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0], (0, dir.path().join("wal-seq000000.seg")));
    assert_eq!(segments[1], (50, dir.path().join("wal-seq000050.seg")));
    assert_eq!(segments[2], (100, dir.path().join("wal-seq000100.seg")));
}

#[test]
fn list_segments_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let segments = list_segments(dir.path()).unwrap();
    assert!(segments.is_empty());
}

#[test]
fn wal_config_default() {
    let config = WalConfig::default();
    assert_eq!(config.max_segment_bytes, 64 * 1024 * 1024);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- segment_filename list_segments wal_config_default parse_segment`
Expected: compilation errors (types/functions don't exist)

**Step 3: Write minimal implementation**

Add to `wal.rs` after the `MAX_FRAME_SIZE` constant (before the `Wal` struct):

```rust
use std::path::PathBuf;

/// Configuration for segmented WAL.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum bytes per segment file before rolling to a new segment.
    /// Default: 64 MB.
    pub max_segment_bytes: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Generate the filename for a segment starting at `start_seq`.
fn segment_filename(start_seq: u64) -> String {
    format!("wal-seq{:06}.seg", start_seq)
}

/// Parse the start-seq from a segment filename. Returns `None` if the
/// filename doesn't match the expected pattern.
fn parse_segment_start_seq(filename: &str) -> Option<u64> {
    let rest = filename.strip_prefix("wal-seq")?.strip_suffix(".seg")?;
    rest.parse().ok()
}

/// List all segment files in a directory, sorted by start-seq ascending.
/// Returns `(start_seq, full_path)` pairs.
fn list_segments(dir: &Path) -> Result<Vec<(u64, PathBuf)>, WalError> {
    let mut segments = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(seq) = parse_segment_start_seq(&name_str) {
            segments.push((seq, entry.path()));
        }
    }
    segments.sort_by_key(|(seq, _)| *seq);
    Ok(segments)
}
```

Add to `lib.rs` re-exports (alongside existing `Wal` export):

```rust
pub use wal::{Wal, WalConfig, WalError};
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist -- segment_filename list_segments wal_config_default parse_segment`
Expected: all 6 tests PASS

**Step 5: Run clippy**

Run: `cargo clippy -p minkowski-persist -- -D warnings`
Expected: clean

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs crates/minkowski-persist/src/lib.rs
git commit -m "feat(persist): add WalConfig + segment filename helpers"
```

---

### Task 2: Refactor Wal struct + create() for segmented directory

Change the `Wal` struct to hold directory-level state and rewrite `create()` to produce a segment directory with one initial segment.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs` (Wal struct + create method)

**Step 1: Write the failing test**

Add to the test module in `wal.rs`:

```rust
#[test]
fn create_segmented_wal() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig { max_segment_bytes: 4096 };
    let wal = Wal::create(&wal_dir, &codecs, config).unwrap();
    assert_eq!(wal.next_seq(), 0);
    assert_eq!(wal.segment_count(), 1);

    // Verify the directory and segment file exist
    assert!(wal_dir.is_dir());
    let segments = list_segments(&wal_dir).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].0, 0); // start_seq = 0
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- create_segmented_wal`
Expected: compilation error (Wal::create signature changed, segment_count doesn't exist)

**Step 3: Rewrite Wal struct and create()**

Replace the existing `Wal` struct and `create` method. Keep all internal helpers (`write_schema_preamble`, `changeset_to_record`, `serialize_mutation`, `read_next_entry`). The key changes:

```rust
pub struct Wal {
    dir: PathBuf,
    active_file: File,
    active_start_seq: u64,
    active_bytes: u64,
    next_seq: u64,
    config: WalConfig,
    schema: WalSchema,
}

impl Wal {
    /// Create a new segmented WAL directory with the first segment.
    pub fn create(dir: &Path, codecs: &CodecRegistry, config: WalConfig) -> Result<Self, WalError> {
        std::fs::create_dir_all(dir)?;
        let schema = Self::build_schema(codecs);
        let seg_path = dir.join(segment_filename(0));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&seg_path)?;
        let mut wal = Self {
            dir: dir.to_path_buf(),
            active_file: file,
            active_start_seq: 0,
            active_bytes: 0,
            next_seq: 0,
            config,
            schema,
        };
        wal.active_bytes = wal.write_schema_preamble()?;
        Ok(wal)
    }

    /// Number of segment files in the WAL directory.
    pub fn segment_count(&self) -> usize {
        list_segments(&self.dir).map(|s| s.len()).unwrap_or(0)
    }

    fn build_schema(codecs: &CodecRegistry) -> WalSchema {
        let mut components = Vec::new();
        for &id in &codecs.registered_ids() {
            let name = codecs.stable_name(id).unwrap().to_string();
            let layout = codecs.layout(id).unwrap();
            components.push(ComponentSchema {
                id,
                name,
                size: layout.size(),
                align: layout.align(),
            });
        }
        WalSchema { components }
    }
}
```

Update `write_schema_preamble` to use `self.schema` instead of building from codecs, and return the number of bytes written:

```rust
fn write_schema_preamble(&mut self) -> Result<u64, WalError> {
    let entry = WalEntry::Schema(self.schema.clone());
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
        .map_err(|e| WalError::Format(e.to_string()))?;
    let mut writer = BufWriter::new(&self.active_file);
    let len: u32 = payload
        .len()
        .try_into()
        .map_err(|_| WalError::Format("schema preamble too large".into()))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(4 + payload.len() as u64)
}
```

**Step 4: Fix all existing tests**

All existing tests that call `Wal::create(path, &codecs)` need to change to `Wal::create(dir, &codecs, WalConfig::default())` — but the path was a file, now it's a directory. Update each existing test's path logic:
- Change `dir.path().join("test.wal")` file paths to directories
- Add `WalConfig::default()` (or a small config for tests) as third arg

Similarly, tests that call `Wal::open(path, &codecs)` will fail — we'll fix `open` in Task 3. For now, comment out or `#[ignore]` the tests that use `open` so the `create`-only tests pass.

**Step 5: Run tests**

Run: `cargo test -p minkowski-persist -- create_segmented`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "refactor(persist): Wal struct + create() for segmented directory"
```

---

### Task 3: Refactor Wal::open() for segmented directory

Rewrite `open()` to scan a directory of segments, open the last one for appending, and recover `next_seq`.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn open_segmented_wal() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Health>("health", &mut world);

    let config = WalConfig { max_segment_bytes: 4096 };
    {
        let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();
        for _ in 0..3 {
            let cs = EnumChangeSet::new();
            wal.append(&cs, &codecs).unwrap();
        }
    }

    let wal2 = Wal::open(&wal_dir, &codecs, config).unwrap();
    assert_eq!(wal2.next_seq(), 3);
}

#[test]
fn open_empty_dir_errors() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("empty.wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let codecs = CodecRegistry::new();
    let config = WalConfig::default();
    let result = Wal::open(&wal_dir, &codecs, config);
    assert!(result.is_err());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- open_segmented open_empty_dir`
Expected: compilation error or test failure

**Step 3: Implement open()**

```rust
/// Open an existing segmented WAL directory.
/// Scans for segments, opens the last one for appending, recovers `next_seq`
/// from the final valid record. Config governs future segment rollover.
pub fn open(dir: &Path, codecs: &CodecRegistry, config: WalConfig) -> Result<Self, WalError> {
    let segments = list_segments(dir)?;
    if segments.is_empty() {
        return Err(WalError::Format("no WAL segments found in directory".into()));
    }

    let (last_start_seq, last_path) = segments.last().unwrap().clone();

    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .open(&last_path)?;

    let schema = Self::build_schema(codecs);
    let active_bytes = file.metadata()?.len();

    let mut wal = Self {
        dir: dir.to_path_buf(),
        active_file: file,
        active_start_seq: last_start_seq,
        active_bytes,
        next_seq: 0,
        config,
        schema,
    };

    // Scan the active segment for the last valid seq + crash recovery
    let (last_seq, has_mutations) = wal.scan_last_seq()?;
    // Update active_bytes after potential truncation
    wal.active_bytes = wal.active_file.metadata()?.len();

    // Also scan all prior sealed segments if the active one is empty
    if !has_mutations && segments.len() > 1 {
        // Check earlier segments for the highest seq
        for (_, seg_path) in segments.iter().rev().skip(1) {
            let seg_file = File::open(seg_path)?;
            let mut pos: u64 = 0;
            let mut seg_last = 0u64;
            let mut seg_has = false;
            while let Some((entry, next_pos)) = read_next_frame(&seg_file, pos)? {
                if let WalEntry::Mutations(record) = entry {
                    seg_last = record.seq;
                    seg_has = true;
                }
                pos = next_pos;
            }
            if seg_has {
                wal.next_seq = seg_last + 1;
                break;
            }
        }
        if has_mutations {
            wal.next_seq = last_seq + 1;
        }
    } else if has_mutations {
        wal.next_seq = last_seq + 1;
    }

    Ok(wal)
}
```

**Step 4: Un-ignore the existing tests that use `open` and update them**

Update `open_existing_wal` test: change path to directory, add config parameter. Update all other tests that use `Wal::open` similarly.

**Step 5: Run all WAL tests**

Run: `cargo test -p minkowski-persist -- wal::tests`
Expected: all PASS

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "refactor(persist): Wal::open() for segmented directory"
```

---

### Task 4: Segment rollover in append()

Modify `append()` to check byte threshold and roll to a new segment when exceeded.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn append_rolls_to_new_segment() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    // Very small segment size to force rollover
    let config = WalConfig { max_segment_bytes: 128 };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    // Append enough records to trigger at least one rollover
    for i in 0..20 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    assert_eq!(wal.next_seq(), 20);
    let segments = list_segments(&wal_dir).unwrap();
    assert!(segments.len() > 1, "should have rolled to multiple segments");

    // Every segment file should start with a schema preamble
    for (_, seg_path) in &segments {
        let file = File::open(seg_path).unwrap();
        let (entry, _) = read_next_frame(&file, 0).unwrap().unwrap();
        assert!(matches!(entry, WalEntry::Schema(_)));
    }
}

#[test]
fn open_after_rollover_recovers_next_seq() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig { max_segment_bytes: 128 };
    {
        let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();
        for i in 0..10 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }
    }

    let wal2 = Wal::open(&wal_dir, &codecs, config).unwrap();
    assert_eq!(wal2.next_seq(), 10);
}

#[test]
fn replay_across_segments() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig { max_segment_bytes: 128 };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let mut world2 = World::new();
    codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);
    let last = wal.replay(&mut world2, &codecs).unwrap();
    assert_eq!(last, 9);
    assert_eq!(world2.query::<(&Pos,)>().count(), 10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- append_rolls open_after_rollover replay_across`
Expected: test failures (no rollover happening yet)

**Step 3: Implement segment rollover**

Update `append()`:

```rust
pub fn append(
    &mut self,
    changeset: &EnumChangeSet,
    codecs: &CodecRegistry,
) -> Result<u64, WalError> {
    let seq = self.next_seq;
    let record = Self::changeset_to_record(seq, changeset, codecs)?;
    let entry = WalEntry::Mutations(record);
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
        .map_err(|e| WalError::Format(e.to_string()))?;

    let mut writer = BufWriter::new(&self.active_file);
    let len: u32 = payload.len().try_into().map_err(|_| {
        WalError::Format(format!(
            "WAL record too large: {} bytes exceeds u32 max",
            payload.len()
        ))
    })?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;

    self.active_bytes += 4 + payload.len() as u64;
    self.next_seq += 1;

    // Roll to new segment if threshold exceeded
    if self.active_bytes >= self.config.max_segment_bytes as u64 {
        self.roll_segment()?;
    }

    Ok(seq)
}

fn roll_segment(&mut self) -> Result<(), WalError> {
    let seg_path = self.dir.join(segment_filename(self.next_seq));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(&seg_path)?;
    self.active_file = file;
    self.active_start_seq = self.next_seq;
    self.active_bytes = self.write_schema_preamble()?;
    Ok(())
}
```

Update `replay()` and `replay_from()` to iterate across all segments:

```rust
pub fn replay(&mut self, world: &mut World, codecs: &CodecRegistry) -> Result<u64, WalError> {
    self.replay_from(0, world, codecs)
}

pub fn replay_from(
    &mut self,
    from_seq: u64,
    world: &mut World,
    codecs: &CodecRegistry,
) -> Result<u64, WalError> {
    let segments = list_segments(&self.dir)?;
    let mut last_seq = if from_seq > 0 { from_seq - 1 } else { 0 };

    for (_, seg_path) in &segments {
        let seg_file = File::open(seg_path)?;
        let mut pos: u64 = 0;
        let mut remap: Option<HashMap<ComponentId, ComponentId>> = None;

        while let Some((entry, next_pos)) = read_next_frame(&seg_file, pos)? {
            match entry {
                WalEntry::Schema(schema) => {
                    remap = Some(codecs.build_remap(&schema.components)?);
                }
                WalEntry::Mutations(record) => {
                    if record.seq >= from_seq {
                        apply_record(&record, world, codecs, remap.as_ref())?;
                        last_seq = record.seq;
                    }
                }
            }
            pos = next_pos;
        }
    }

    Ok(last_seq)
}
```

Note: `replay_from` no longer uses `self.read_next_entry` (which truncates) — it reads sealed segments with `read_next_frame` (no truncation). Only the active segment should do crash recovery, and that's handled by `open()`.

**Step 4: Run tests**

Run: `cargo test -p minkowski-persist -- wal::tests`
Expected: all PASS including new rollover tests

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "feat(persist): segment rollover in append()"
```

---

### Task 5: delete_segments_before + oldest_seq

Add the truncation primitive and introspection methods.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn delete_segments_before() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig { max_segment_bytes: 128 };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    for i in 0..20 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let before = wal.segment_count();
    assert!(before > 2, "need multiple segments for this test");

    // Delete segments whose records are all before seq 10
    let deleted = wal.delete_segments_before(10).unwrap();
    assert!(deleted > 0);
    assert_eq!(wal.segment_count(), before - deleted);

    // Verify the oldest remaining segment contains seq >= some value
    let oldest = wal.oldest_seq();
    assert!(oldest.is_some());
}

#[test]
fn delete_segments_before_preserves_active() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig { max_segment_bytes: 128 };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    // Try to delete everything — should never delete active segment
    let deleted = wal.delete_segments_before(u64::MAX).unwrap();
    assert!(wal.segment_count() >= 1, "active segment must survive");

    // WAL should still be appendable
    let mut cs = EnumChangeSet::new();
    let e = world.alloc_entity();
    cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },));
    wal.append(&cs, &codecs).unwrap();
}

#[test]
fn oldest_seq_after_creation() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig::default();
    let wal = Wal::create(&wal_dir, &codecs, config).unwrap();
    assert_eq!(wal.oldest_seq(), Some(0));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- delete_segments oldest_seq_after`
Expected: compilation error (methods don't exist)

**Step 3: Implement**

```rust
/// Delete all segment files whose entire seq range is before `seq`.
/// A segment is safe to delete if the next segment's start_seq <= `seq`.
/// The active (last) segment is never deleted.
/// Returns the number of segments deleted.
pub fn delete_segments_before(&mut self, seq: u64) -> Result<usize, WalError> {
    let segments = list_segments(&self.dir)?;
    if segments.len() <= 1 {
        return Ok(0);
    }

    let mut deleted = 0;
    for i in 0..segments.len() - 1 {
        // Segment i is safe to delete if the next segment starts at or before `seq`
        let next_start = segments[i + 1].0;
        if next_start <= seq {
            std::fs::remove_file(&segments[i].1)?;
            deleted += 1;
        } else {
            break; // segments are sorted, no point continuing
        }
    }

    Ok(deleted)
}

/// Start-seq of the oldest remaining segment, or `None` if no segments exist.
pub fn oldest_seq(&self) -> Option<u64> {
    list_segments(&self.dir)
        .ok()
        .and_then(|s| s.first().map(|(seq, _)| *seq))
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski-persist -- delete_segments oldest_seq`
Expected: all PASS

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "feat(persist): delete_segments_before + oldest_seq"
```

---

### Task 6: Refactor WalCursor for multi-segment

Update `WalCursor` to traverse segment directories with lazy file advancement.

**Files:**
- Modify: `crates/minkowski-persist/src/replication.rs`

**Step 1: Write the failing tests**

Add to `replication.rs` test module:

```rust
#[test]
fn cursor_spans_segments() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = crate::wal::WalConfig { max_segment_bytes: 128 };
    let mut wal = crate::wal::Wal::create(&wal_dir, &codecs, config).unwrap();

    for i in 0..20 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    // Verify multiple segments exist
    let seg_count = wal.segment_count();
    assert!(seg_count > 1);

    let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    assert_eq!(batch.records.len(), 20);
    assert_eq!(batch.records[0].seq, 0);
    assert_eq!(batch.records[19].seq, 19);
}

#[test]
fn cursor_behind_after_delete() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = crate::wal::WalConfig { max_segment_bytes: 128 };
    let mut wal = crate::wal::Wal::create(&wal_dir, &codecs, config).unwrap();

    for i in 0..20 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    // Delete old segments
    wal.delete_segments_before(15).unwrap();

    // Trying to open a cursor at seq 0 should fail with CursorBehind
    let result = WalCursor::open(&wal_dir, 0);
    assert!(matches!(result, Err(WalError::CursorBehind { .. })));
}

#[test]
fn cursor_picks_up_new_segments() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = crate::wal::WalConfig { max_segment_bytes: 128 };
    let mut wal = crate::wal::Wal::create(&wal_dir, &codecs, config).unwrap();

    // Write 5 records
    for i in 0..5 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
    let batch1 = cursor.next_batch(100).unwrap();
    assert_eq!(batch1.records.len(), 5);

    // Write 5 more (may create new segments)
    for i in 5..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    // Cursor should pick up new records (possibly in new segments)
    let batch2 = cursor.next_batch(100).unwrap();
    assert_eq!(batch2.records.len(), 5);
    assert_eq!(batch2.records[0].seq, 5);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- cursor_spans cursor_behind_after cursor_picks_up`
Expected: compilation errors (WalCursor::open signature changed)

**Step 3: Rewrite WalCursor**

```rust
use crate::wal::{list_segments, read_next_frame, WalError};

pub struct WalCursor {
    dir: PathBuf,
    file: File,
    pos: u64,
    next_seq: u64,
    schema: Option<WalSchema>,
    current_segment_start_seq: u64,
}

impl WalCursor {
    /// Open a WAL directory for reading, starting from `from_seq`.
    pub fn open(dir: &Path, from_seq: u64) -> Result<Self, WalError> {
        let segments = list_segments(dir)?;
        if segments.is_empty() {
            return Err(WalError::Format("no WAL segments found".into()));
        }

        // Find the segment containing from_seq: largest start_seq <= from_seq
        let seg_idx = match segments.iter().rposition(|(start, _)| *start <= from_seq) {
            Some(idx) => idx,
            None => {
                // All segments start after from_seq
                return Err(WalError::CursorBehind {
                    requested: from_seq,
                    oldest: segments[0].0,
                });
            }
        };

        let (start_seq, seg_path) = &segments[seg_idx];
        let file = File::open(seg_path)?;
        let mut pos: u64 = 0;
        let mut schema = None;

        // Scan forward to from_seq
        loop {
            match read_next_frame(&file, pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    schema = Some(s);
                    pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    if record.seq >= from_seq {
                        break; // Don't advance past this record
                    }
                    pos = next_pos;
                }
                None => break,
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            file,
            pos,
            next_seq: from_seq,
            schema,
            current_segment_start_seq: *start_seq,
        })
    }

    pub fn next_batch(&mut self, limit: usize) -> Result<ReplicationBatch, WalError> {
        let mut records = Vec::new();

        while records.len() < limit {
            match read_next_frame(&self.file, self.pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    self.next_seq = record.seq + 1;
                    records.push(record);
                    self.pos = next_pos;
                }
                None => {
                    // Try to advance to next segment
                    if !self.try_advance_segment()? {
                        break; // No more segments — caught up
                    }
                }
            }
        }

        let schema = self
            .schema
            .clone()
            .unwrap_or_else(|| WalSchema { components: vec![] });
        Ok(ReplicationBatch { schema, records })
    }

    /// Try to open the next segment file. Returns true if advanced.
    fn try_advance_segment(&mut self) -> Result<bool, WalError> {
        let segments = list_segments(&self.dir)?;
        // Find next segment after current
        let next = segments
            .iter()
            .find(|(start, _)| *start > self.current_segment_start_seq);

        match next {
            Some((start_seq, path)) => {
                self.file = File::open(path)?;
                self.pos = 0;
                self.current_segment_start_seq = *start_seq;
                // Parse schema preamble of new segment
                if let Some((WalEntry::Schema(s), next_pos)) =
                    read_next_frame(&self.file, 0)?
                {
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn schema(&self) -> Option<&WalSchema> {
        self.schema.as_ref()
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}
```

**Step 4: Update existing replication tests**

All existing tests that call `WalCursor::open(&wal_path, seq)` where `wal_path` is a file need to change to use the WAL directory. The `create_test_wal` helper needs updating:

```rust
fn create_test_wal(dir: &std::path::Path, n: usize) -> (std::path::PathBuf, CodecRegistry) {
    let wal_dir = dir.join("test.wal");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = crate::wal::WalConfig::default();
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();
    for i in 0..n {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }
    (wal_dir, codecs)
}
```

Update all test paths similarly: `dir.path().join("test.wal")` becomes a directory, `Wal::create` gets config, `WalCursor::open` gets dir.

**Step 5: Run all replication tests**

Run: `cargo test -p minkowski-persist -- replication::tests`
Expected: all PASS

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/replication.rs
git commit -m "refactor(persist): WalCursor multi-segment support"
```

---

### Task 7: Update Durable + lib.rs exports

Update `Durable` tests and re-exports for the new API.

**Files:**
- Modify: `crates/minkowski-persist/src/durable.rs` (tests only — Durable holds `Wal` which changed internally but its `append` signature didn't)
- Modify: `crates/minkowski-persist/src/lib.rs` (ensure `WalConfig` is re-exported)

**Step 1: Update durable.rs tests**

Every test in `durable.rs` does `Wal::create(&wal_path, &codecs)` — update to `Wal::create(&wal_dir, &codecs, WalConfig::default())`. Change file paths to directory paths:

```rust
// Before:
let wal_path = dir.path().join("test.wal");
let wal = Wal::create(&wal_path, &codecs).unwrap();

// After:
let wal_dir = dir.path().join("test.wal");
let wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
```

Add `use crate::wal::WalConfig;` to the test module imports.

**Step 2: Verify lib.rs exports**

Ensure `lib.rs` has:
```rust
pub use wal::{Wal, WalConfig, WalError};
```

**Step 3: Run durable tests**

Run: `cargo test -p minkowski-persist -- durable::tests`
Expected: all 3 tests PASS

**Step 4: Run full persist crate tests**

Run: `cargo test -p minkowski-persist`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/durable.rs crates/minkowski-persist/src/lib.rs
git commit -m "chore(persist): update Durable tests + re-exports for segmented WAL"
```

---

### Task 8: Update examples

Update `persist.rs` and `replicate.rs` examples for the new segmented WAL API.

**Files:**
- Modify: `examples/examples/persist.rs`
- Modify: `examples/examples/replicate.rs`

**Step 1: Update persist.rs**

Key changes:
- `wal_path` becomes `wal_dir` (a directory)
- `Wal::create(&wal_path, &codecs)` → `Wal::create(&wal_dir, &codecs, WalConfig::default())`
- `Wal::open(&wal_path, &load_codecs)` → `Wal::open(&wal_dir, &load_codecs, WalConfig::default())`
- Cleanup: `remove_file` → `remove_dir_all` for the WAL directory
- Add `use minkowski_persist::WalConfig;` to imports

```rust
// Phase 2 changes:
let wal_dir = dir.join("example.wal");
let _ = std::fs::remove_dir_all(&wal_dir);  // cleanup
let wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

// Phase 4 changes:
let mut replay_wal = Wal::open(&wal_dir, &load_codecs, WalConfig::default()).unwrap();
```

**Step 2: Update replicate.rs**

Key changes:
- Same file→directory path changes
- `WalCursor::open(&wal_path, snap_seq)` → `WalCursor::open(&wal_dir, snap_seq)`
- Add `WalConfig` import

```rust
let wal_dir = dir.join("source.wal");
let _ = std::fs::remove_dir_all(&wal_dir);
let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
// ...
let mut cursor = WalCursor::open(&wal_dir, snap_seq).unwrap();
```

**Step 3: Run examples**

Run: `cargo run -p minkowski-examples --example persist --release`
Expected: runs successfully, prints phases 1-4

Run: `cargo run -p minkowski-examples --example replicate --release`
Expected: runs successfully, prints convergence

**Step 4: Run clippy on workspace**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean

**Step 5: Run full test suite**

Run: `cargo test -p minkowski-persist && cargo test -p minkowski`
Expected: all tests PASS

**Step 6: Commit**

```bash
git add examples/examples/persist.rs examples/examples/replicate.rs
git commit -m "chore(examples): update persist + replicate for segmented WAL"
```

---

### Task 9: Full integration test + example enhancement

Add a test that exercises the complete lifecycle: write → rollover → snapshot → delete old segments → cursor pull → convergence. Optionally enhance the `replicate` example to demonstrate truncation.

**Files:**
- Modify: `crates/minkowski-persist/src/replication.rs` (add integration test)

**Step 1: Write the integration test**

Add to the replication test module:

```rust
#[test]
fn full_segmented_replication_with_truncation() {
    use crate::snapshot::Snapshot;

    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("source.wal");
    let snap_path = dir.path().join("source.snap");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    // Small segments to force rollover
    let config = crate::wal::WalConfig { max_segment_bytes: 128 };
    let mut wal = crate::wal::Wal::create(&wal_dir, &codecs, config).unwrap();

    // Phase 1: Spawn 10 entities (pre-snapshot)
    for i in 0..10 {
        world.spawn((Pos { x: i as f32, y: 0.0 },));
    }

    // Phase 2: Snapshot
    let snap = Snapshot::new();
    let header = snap
        .save(&snap_path, &world, &codecs, wal.next_seq())
        .unwrap();
    assert_eq!(header.entity_count, 10);

    // Phase 3: WAL mutations (will create multiple segments)
    for i in 10..30 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let seg_count = wal.segment_count();
    assert!(seg_count > 1, "need multiple segments");

    // Phase 4: Delete old segments (simulate retention policy)
    let deleted = wal.delete_segments_before(10).unwrap();
    assert!(deleted > 0);

    // Phase 5: Replica from snapshot + cursor
    let mut replica_codecs = CodecRegistry::new();
    let mut tmp = World::new();
    replica_codecs.register_as::<Pos>("pos", &mut tmp);

    let (mut replica, snap_seq) = snap.load(&snap_path, &replica_codecs).unwrap();
    assert_eq!(replica.query::<(&Pos,)>().count(), 10);

    let mut cursor = WalCursor::open(&wal_dir, snap_seq).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    assert_eq!(batch.records.len(), 20);

    let last = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();
    assert_eq!(last, Some(19));
    assert_eq!(replica.query::<(&Pos,)>().count(), 30);
}
```

**Step 2: Run the test**

Run: `cargo test -p minkowski-persist -- full_segmented_replication`
Expected: PASS

**Step 3: Run everything**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test -p minkowski-persist && cargo test -p minkowski`
Expected: all clean and passing

**Step 4: Commit**

```bash
git add crates/minkowski-persist/src/replication.rs
git commit -m "test(persist): full segmented replication + truncation integration test"
```
