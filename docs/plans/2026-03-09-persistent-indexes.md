# Persistent Indexes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BTreeIndex and HashIndex crash-recoverable by persisting index state to disk, so recovery loads from file + WAL tail instead of full rebuild.

**Architecture:** Add `as_raw_parts`/`from_raw_parts` to index types in core crate for controlled serialization access. Add `ChangeTick::to_raw`/`from_raw` for tick serialization. Add `PersistentIndex` trait, `IndexPersistError`, rkyv-based save/load, and `AutoCheckpoint` integration in persist crate. Atomic rename on write prevents crash corruption.

**Tech Stack:** rkyv (serialization), crc32fast (checksums), parking_lot (Mutex for AutoCheckpoint), tempfile (tests)

**Design doc:** `docs/plans/2026-03-09-persistent-indexes-design.md`

---

### Task 1: ChangeTick serialization surface

Add `to_raw`/`from_raw` methods so the persist crate can serialize the tick value.

**Files:**
- Modify: `crates/minkowski/src/tick.rs`

**Step 1: Write the failing test**

Add to `tick.rs` `mod tests`:

```rust
#[test]
fn change_tick_round_trip_via_u64() {
    let tick = ChangeTick(Tick::new(42));
    let raw = tick.to_raw();
    let restored = ChangeTick::from_raw(raw);
    assert_eq!(tick, restored);
}

#[test]
fn change_tick_default_round_trips() {
    let tick = ChangeTick::default();
    assert_eq!(ChangeTick::from_raw(tick.to_raw()), tick);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- change_tick_round_trip`

**Step 3: Implement**

Add to `tick.rs`:

```rust
impl ChangeTick {
    /// Convert to raw u64 for serialization.
    pub fn to_raw(self) -> u64 {
        self.0.raw()
    }

    /// Reconstruct from raw u64. The caller must ensure the value
    /// represents a valid tick from the same world.
    pub fn from_raw(raw: u64) -> Self {
        Self(Tick::new(raw))
    }
}
```

Remove `#[allow(dead_code)]` from `Tick::new` and `Tick::raw` (now used).

**Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- change_tick_round_trip`

**Step 5: Commit**

```
feat: add ChangeTick::to_raw/from_raw for index serialization
```

---

### Task 2: BTreeIndex raw parts API

Add borrowing `as_raw_parts` and constructing `from_raw_parts` for serialization access.

**Files:**
- Modify: `crates/minkowski/src/index.rs`

**Step 1: Write the failing test**

Add to `index.rs` `mod tests`:

```rust
#[test]
fn btree_raw_parts_round_trip() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let (tree, reverse, last_sync) = idx.as_raw_parts();
    let restored = BTreeIndex::<Score>::from_raw_parts(
        tree.clone(), reverse.clone(), last_sync,
    );

    assert_eq!(restored.get(&Score(10)).len(), 1);
    assert!(restored.get(&Score(10)).contains(&e1));
    assert_eq!(restored.get(&Score(20)).len(), 1);
    assert!(restored.get(&Score(20)).contains(&e2));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- btree_raw_parts`

**Step 3: Implement**

Add to `BTreeIndex<T>` impl block:

```rust
/// Borrow the internal state for serialization.
pub fn as_raw_parts(&self) -> (&BTreeMap<T, Vec<Entity>>, &HashMap<Entity, T>, ChangeTick) {
    (&self.tree, &self.reverse, self.last_sync)
}

/// Reconstruct from deserialized parts.
pub fn from_raw_parts(
    tree: BTreeMap<T, Vec<Entity>>,
    reverse: HashMap<Entity, T>,
    last_sync: ChangeTick,
) -> Self {
    Self { tree, reverse, last_sync }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- btree_raw_parts`

**Step 5: Commit**

```
feat: add BTreeIndex::as_raw_parts/from_raw_parts
```

---

### Task 3: HashIndex raw parts API

Same pattern as Task 2.

**Files:**
- Modify: `crates/minkowski/src/index.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn hash_raw_parts_round_trip() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));

    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    let (map, reverse, last_sync) = idx.as_raw_parts();
    let restored = HashIndex::<Score>::from_raw_parts(
        map.clone(), reverse.clone(), last_sync,
    );

    assert_eq!(restored.get(&Score(10)).len(), 1);
    assert!(restored.get(&Score(10)).contains(&e1));
    assert_eq!(restored.get(&Score(20)).len(), 1);
    assert!(restored.get(&Score(20)).contains(&e2));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- hash_raw_parts`

**Step 3: Implement**

Add to `HashIndex<T>` impl block:

```rust
pub fn as_raw_parts(&self) -> (&HashMap<T, Vec<Entity>>, &HashMap<Entity, T>, ChangeTick) {
    (&self.map, &self.reverse, self.last_sync)
}

pub fn from_raw_parts(
    map: HashMap<T, Vec<Entity>>,
    reverse: HashMap<Entity, T>,
    last_sync: ChangeTick,
) -> Self {
    Self { map, reverse, last_sync }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- hash_raw_parts`

**Step 5: Commit**

```
feat: add HashIndex::as_raw_parts/from_raw_parts
```

---

### Task 4: IndexPersistError, PersistentIndex trait, and file I/O helpers

Create the persist-side module with the trait, error type, file format constants, and envelope read/write helpers.

**Files:**
- Create: `crates/minkowski-persist/src/index.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Step 1: Create the module**

```rust
// crates/minkowski-persist/src/index.rs
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use minkowski::SpatialIndex;

/// Index file magic identifying the format.
pub(crate) const INDEX_MAGIC: [u8; 8] = *b"MK2INDXK";

/// Header size: magic (8) + CRC32 (4) + reserved (4) + length (8) = 24.
pub(crate) const INDEX_HEADER_SIZE: usize = 24;

#[derive(Debug, thiserror::Error)]
pub enum IndexPersistError {
    #[error("index I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("index format error: {0}")]
    Format(String),
}

/// A secondary index that can be saved to disk and loaded on recovery.
///
/// `save` is object-safe — `AutoCheckpoint` can hold registered indexes
/// and call `save` on each. `load` is on the concrete type (returns
/// `Self`, not object-safe).
///
/// Writes use atomic rename: data goes to `path.tmp`, then is renamed
/// to `path`. A crash during write cannot corrupt the previous file.
///
/// After loading, call [`SpatialIndex::update`] to catch up with
/// mutations that occurred after the index was last saved.
pub trait PersistentIndex: SpatialIndex {
    /// Serialize the index state to a file.
    fn save(&self, path: &Path) -> Result<(), IndexPersistError>;
}

/// Write an index envelope: `[magic][crc32][reserved][len][payload]`.
/// Uses atomic rename — writes to `path.idx.tmp`, then renames.
pub(crate) fn write_index_file(path: &Path, payload: &[u8]) -> Result<(), IndexPersistError> {
    let tmp_path = path.with_extension("idx.tmp");
    let crc = crc32fast::hash(payload);
    let len = payload.len() as u64;

    let file = File::create(&tmp_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&INDEX_MAGIC)?;
    writer.write_all(&crc.to_le_bytes())?;
    writer.write_all(&[0u8; 4])?; // reserved for 8-byte alignment
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    drop(writer);

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Read and validate an index file. Returns the verified payload bytes.
pub(crate) fn read_index_file(path: &Path) -> Result<Vec<u8>, IndexPersistError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < INDEX_HEADER_SIZE {
        return Err(IndexPersistError::Format("index file too small".into()));
    }
    if bytes[..8] != INDEX_MAGIC {
        return Err(IndexPersistError::Format("invalid index file magic".into()));
    }
    let stored_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let end = INDEX_HEADER_SIZE
        .checked_add(len)
        .ok_or_else(|| IndexPersistError::Format("invalid payload length".into()))?;
    if bytes.len() < end {
        return Err(IndexPersistError::Format(format!(
            "index truncated: expected {len} payload bytes, got {}",
            bytes.len() - INDEX_HEADER_SIZE
        )));
    }
    let payload = &bytes[INDEX_HEADER_SIZE..end];
    let actual_crc = crc32fast::hash(payload);
    if actual_crc != stored_crc {
        return Err(IndexPersistError::Format(format!(
            "index checksum mismatch: expected {stored_crc:#010x}, got {actual_crc:#010x}"
        )));
    }
    Ok(payload.to_vec())
}
```

**Step 2: Register module in lib.rs**

Add to `crates/minkowski-persist/src/lib.rs`:
```rust
pub mod index;
pub use index::{IndexPersistError, PersistentIndex};
```

**Step 3: Verify it compiles**

Run: `cargo check -p minkowski-persist`

**Step 4: Commit**

```
feat: add PersistentIndex trait, IndexPersistError, file I/O helpers
```

---

### Task 5: BTreeIndex PersistentIndex impl (save + load)

Implement rkyv-based save/load for BTreeIndex.

**Files:**
- Modify: `crates/minkowski-persist/src/index.rs`

**Step 1: Write the failing tests**

Add to `index.rs` at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::{BTreeIndex, SpatialIndex, World};
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
             Archive, Serialize, Deserialize)]
    #[repr(C)]
    struct Score(u32);

    #[test]
    fn btree_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("score.idx");

        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));
        let e3 = world.spawn((Score(10),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        idx.save(&path).unwrap();
        let loaded = load_btree_index::<Score>(&path).unwrap();

        assert_eq!(loaded.get(&Score(10)).len(), 2);
        assert!(loaded.get(&Score(10)).contains(&e1));
        assert!(loaded.get(&Score(10)).contains(&e3));
        assert_eq!(loaded.get(&Score(20)).len(), 1);
        assert!(loaded.get(&Score(20)).contains(&e2));
    }

    #[test]
    fn btree_empty_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.idx");

        let mut world = World::new();
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_btree_index::<Score>(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn btree_load_missing_file() {
        let result = load_btree_index::<Score>(Path::new("/nonexistent/score.idx"));
        assert!(matches!(result, Err(IndexPersistError::Io(_))));
    }

    #[test]
    fn btree_crc_mismatch_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.idx");

        let mut world = World::new();
        world.spawn((Score(1),));
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let mut data = std::fs::read(&path).unwrap();
        data[INDEX_HEADER_SIZE] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = load_btree_index::<Score>(&path);
        let err = result.err().unwrap();
        let msg = format!("{err}");
        assert!(msg.contains("checksum"), "expected checksum error: {msg}");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- btree_save_load`

**Step 3: Implement**

The rkyv payload structure and save/load functions. The key insight: serialize the index as a flat list of `(key_bytes, entity_bits_vec)` entries plus the reverse map and tick. Use a concrete non-generic rkyv struct with `Vec<u8>` for keys to avoid complex generic bounds:

```rust
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use minkowski::tick::ChangeTick;
use minkowski::{BTreeIndex, Component, Entity};

/// On-disk representation of an index. Keys are rkyv-serialized as byte blobs
/// to avoid propagating generic rkyv bounds through the trait.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct IndexPayload {
    /// Forward map entries: (rkyv key bytes, entity bits).
    entries: Vec<(Vec<u8>, Vec<u64>)>,
    /// Reverse map entries: (entity bits, rkyv key bytes).
    reverse: Vec<(u64, Vec<u8>)>,
    /// ChangeTick at last synchronization.
    last_sync_tick: u64,
}

impl<T> PersistentIndex for BTreeIndex<T>
where
    T: Component + Ord + Clone + Hash
        + rkyv::Archive
        + for<'a> rkyv::Serialize<
            rkyv::ser::allocator::ArenaHandle<'a>,
            rkyv::util::AlignedVec,
        >,
{
    fn save(&self, path: &Path) -> Result<(), IndexPersistError> {
        let (tree, reverse, last_sync) = self.as_raw_parts();

        let entries: Vec<(Vec<u8>, Vec<u64>)> = tree
            .iter()
            .map(|(k, entities)| {
                let key_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(k)
                    .map_err(|e| IndexPersistError::Format(e.to_string()))
                    .unwrap(); // key serialization should not fail for well-typed data
                let bits: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
                (key_bytes.to_vec(), bits)
            })
            .collect();

        let rev: Vec<(u64, Vec<u8>)> = reverse
            .iter()
            .map(|(entity, k)| {
                let key_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(k)
                    .map_err(|e| IndexPersistError::Format(e.to_string()))
                    .unwrap();
                (entity.to_bits(), key_bytes.to_vec())
            })
            .collect();

        let payload_data = IndexPayload {
            entries,
            reverse: rev,
            last_sync_tick: last_sync.to_raw(),
        };

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&payload_data)
            .map_err(|e| IndexPersistError::Format(e.to_string()))?;

        write_index_file(path, &payload)
    }
}

/// Load a BTreeIndex from a persisted file.
pub fn load_btree_index<T>(path: &Path) -> Result<BTreeIndex<T>, IndexPersistError>
where
    T: Component + Ord + Clone + Hash
        + rkyv::Archive
        + rkyv::Deserialize<T, rkyv::de::Pool>,
    T::Archived: rkyv::Deserialize<T, rkyv::de::Pool>,
{
    let payload_bytes = read_index_file(path)?;
    let archived = rkyv::access::<rkyv::Archived<IndexPayload>, rkyv::rancor::Error>(&payload_bytes)
        .map_err(|e| IndexPersistError::Format(e.to_string()))?;

    let data: IndexPayload = rkyv::deserialize::<IndexPayload, rkyv::rancor::Error>(archived)
        .map_err(|e| IndexPersistError::Format(e.to_string()))?;

    let mut tree: BTreeMap<T, Vec<Entity>> = BTreeMap::new();
    for (key_bytes, entity_bits) in &data.entries {
        let key: T = rkyv::from_bytes::<T, rkyv::rancor::Error>(key_bytes)
            .map_err(|e| IndexPersistError::Format(e.to_string()))?;
        let entities: Vec<Entity> = entity_bits.iter().map(|&b| Entity::from_bits(b)).collect();
        tree.insert(key, entities);
    }

    let mut reverse: HashMap<Entity, T> = HashMap::new();
    for (entity_bits, key_bytes) in &data.reverse {
        let entity = Entity::from_bits(*entity_bits);
        let key: T = rkyv::from_bytes::<T, rkyv::rancor::Error>(key_bytes)
            .map_err(|e| IndexPersistError::Format(e.to_string()))?;
        reverse.insert(entity, key);
    }

    let last_sync = ChangeTick::from_raw(data.last_sync_tick);
    Ok(BTreeIndex::from_raw_parts(tree, reverse, last_sync))
}
```

Note: The exact rkyv trait bounds may need adjustment during implementation — rkyv 0.8's serializer traits are complex. The general pattern is correct; the exact `Serialize<S>` bound depends on the allocator strategy. Use `cargo check` to iterate on the bounds.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist -- btree_`

**Step 5: Commit**

```
feat: BTreeIndex save/load with CRC32 and atomic rename
```

---

### Task 6: HashIndex PersistentIndex impl (save + load)

Same pattern as Task 5, loading into `HashMap` instead of `BTreeMap`.

**Files:**
- Modify: `crates/minkowski-persist/src/index.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn hash_save_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("score_hash.idx");

    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));

    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);
    idx.save(&path).unwrap();

    let loaded = load_hash_index::<Score>(&path).unwrap();
    assert_eq!(loaded.get(&Score(10)).len(), 1);
    assert!(loaded.get(&Score(10)).contains(&e1));
    assert_eq!(loaded.get(&Score(20)).len(), 1);
    assert!(loaded.get(&Score(20)).contains(&e2));
}

#[test]
fn hash_crc_mismatch_detected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt_hash.idx");

    let mut world = World::new();
    world.spawn((Score(1),));
    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);
    idx.save(&path).unwrap();

    let mut data = std::fs::read(&path).unwrap();
    data[INDEX_HEADER_SIZE] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let result = load_hash_index::<Score>(&path);
    assert!(result.is_err());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- hash_save_load`

**Step 3: Implement**

`PersistentIndex` impl for `HashIndex<T>` (identical structure to BTreeIndex, but `as_raw_parts` returns `&HashMap<T, Vec<Entity>>`), and `load_hash_index<T>` function that reconstructs into `HashMap`. The serialized `IndexPayload` is identical — only the load target differs.

**Step 4: Run tests**

Run: `cargo test -p minkowski-persist -- hash_`

**Step 5: Commit**

```
feat: HashIndex save/load with CRC32 and atomic rename
```

---

### Task 7: Stale index catch-up test

Verify that loading a stale index and calling `update()` produces the correct state.

**Files:**
- Modify: `crates/minkowski-persist/src/index.rs` (tests)

**Step 1: Write the test**

```rust
#[test]
fn btree_stale_catch_up_after_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stale.idx");

    let mut world = World::new();
    world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    idx.save(&path).unwrap();

    // Mutate after save
    *world.get_mut::<Score>(e2).unwrap() = Score(30);
    world.spawn((Score(40),));

    // Load stale index, catch up
    let mut loaded = load_btree_index::<Score>(&path).unwrap();
    loaded.update(&mut world);

    // Verify: Score(20) gone, Score(30) and Score(40) present
    assert!(loaded.get(&Score(20)).is_empty());
    assert_eq!(loaded.get(&Score(30)).len(), 1);
    assert_eq!(loaded.get(&Score(40)).len(), 1);
    assert_eq!(loaded.get(&Score(10)).len(), 1);
}
```

**Step 2: Run test — should pass immediately**

Run: `cargo test -p minkowski-persist -- btree_stale_catch_up`

This should pass without new code — it exercises the existing `update()` path with the loaded `last_sync` tick.

**Step 3: Commit**

```
test: stale index catch-up after load
```

---

### Task 8: AutoCheckpoint index integration

Extend `AutoCheckpoint` to optionally save persistent indexes on checkpoint.

**Files:**
- Modify: `crates/minkowski-persist/src/checkpoint.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn auto_checkpoint_saves_registered_index() {
    use crate::index::{load_btree_index, PersistentIndex};
    use minkowski::{BTreeIndex, SpatialIndex};
    use parking_lot::Mutex;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");
    let snap_dir = dir.path().join("snaps");
    let idx_path = dir.path().join("score.idx");
    std::fs::create_dir_all(&snap_dir).unwrap();

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    // Also register Score for the index
    // (Score needs rkyv derives — use the test type)

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(128),
    };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    // Build and register a persistent index
    let mut idx = BTreeIndex::<Score>::new();
    // ... spawn entities, rebuild index ...
    let idx = Arc::new(Mutex::new(idx));

    let mut handler = AutoCheckpoint::new(&snap_dir);
    handler.register_index(idx_path.clone(), idx.clone());

    // Trigger checkpoint
    // ... append enough WAL data ...
    handler.on_checkpoint_needed(&mut world, &mut wal, &codecs).unwrap();

    // Verify index file was created
    assert!(idx_path.exists());
}
```

Note: The exact test setup depends on getting the right types registered. Adjust during implementation.

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- auto_checkpoint_saves`

**Step 3: Implement**

Modify `AutoCheckpoint`:

```rust
use crate::index::{IndexPersistError, PersistentIndex};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct AutoCheckpoint {
    snap_dir: PathBuf,
    indexes: Vec<(PathBuf, Arc<Mutex<dyn PersistentIndex>>)>,
}

impl AutoCheckpoint {
    pub fn new(snap_dir: &Path) -> Self {
        Self {
            snap_dir: snap_dir.to_path_buf(),
            indexes: Vec::new(),
        }
    }

    /// Register a persistent index to be saved on each checkpoint.
    /// Index save failures are non-fatal — logged but do not fail the checkpoint.
    pub fn register_index(
        &mut self,
        path: PathBuf,
        index: Arc<Mutex<dyn PersistentIndex>>,
    ) {
        self.indexes.push((path, index));
    }
}

impl CheckpointHandler for AutoCheckpoint {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let seq = wal.next_seq();
        let path = self.snap_dir.join(format!("checkpoint-{seq:06}.snap"));
        let snap = Snapshot::new();
        snap.save(&path, world, codecs, seq)?;

        // Save registered indexes (non-fatal on failure)
        for (idx_path, index) in &self.indexes {
            if let Err(e) = index.lock().save(idx_path) {
                eprintln!("warning: index save failed for {}: {e}", idx_path.display());
            }
        }

        wal.acknowledge_snapshot(seq)?;
        Ok(())
    }
}
```

**Step 4: Run test**

Run: `cargo test -p minkowski-persist -- auto_checkpoint_saves`

**Step 5: Commit**

```
feat: AutoCheckpoint saves registered persistent indexes
```

---

### Task 9: Full recovery integration test

End-to-end test: spawn, index, checkpoint, mutate via WAL, crash, recover, load index, catch up, verify.

**Files:**
- Modify: `crates/minkowski-persist/src/index.rs` (tests)

**Step 1: Write the test**

```rust
#[test]
fn full_recovery_with_persistent_index() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("recovery.wal");
    let snap_dir = dir.path().join("snaps");
    let idx_path = dir.path().join("score.idx");
    std::fs::create_dir_all(&snap_dir).unwrap();

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Score>("score", &mut world);

    let config = WalConfig { ... };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    // Phase 1: spawn entities, build index, save everything
    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = minkowski::EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Score(i),));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    idx.save(&idx_path).unwrap();

    let snap = Snapshot::new();
    let seq = wal.next_seq();
    snap.save(&snap_dir.join("checkpoint.snap"), &world, &codecs, seq).unwrap();
    wal.acknowledge_snapshot(seq).unwrap();

    // Phase 2: more mutations after checkpoint (WAL tail)
    let e = world.alloc_entity();
    let mut cs = minkowski::EnumChangeSet::new();
    cs.spawn_bundle(&mut world, e, (Score(99),));
    wal.append(&cs, &codecs).unwrap();
    cs.apply(&mut world);

    let expected_count = 11;

    // Phase 3: simulate crash — drop everything
    drop(wal);
    drop(world);

    // Phase 4: recover
    let snap = Snapshot::new();
    let (mut world2, recovered_seq) = snap.load(
        &snap_dir.join("checkpoint.snap"), &codecs
    ).unwrap();

    let mut wal2 = Wal::open(&wal_dir, &codecs, config).unwrap();
    wal2.replay(&mut world2, &codecs).unwrap();

    // Phase 5: load index and catch up
    let mut idx2 = load_btree_index::<Score>(&idx_path).unwrap();
    idx2.update(&mut world2);

    // Verify: all entities present including post-checkpoint Score(99)
    assert_eq!(idx2.get(&Score(99)).len(), 1);
    let total: usize = (0..100).map(|i| idx2.get(&Score(i)).len()).sum();
    assert_eq!(total, expected_count);
}
```

**Step 2: Run test**

Run: `cargo test -p minkowski-persist -- full_recovery`

**Step 3: Fix any issues, commit**

```
test: full recovery integration with persistent index
```

---

### Task 10: Export, docs, clippy, final cleanup

**Files:**
- Modify: `crates/minkowski-persist/src/lib.rs` (exports)
- Modify: `crates/minkowski/src/lib.rs` (re-export ChangeTick methods if needed)

**Step 1: Verify public API**

Ensure `lib.rs` exports:
- `PersistentIndex`
- `IndexPersistError`
- `load_btree_index`
- `load_hash_index`

**Step 2: Run full test suite**

```bash
cargo test -p minkowski --lib
cargo test -p minkowski-persist
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

**Step 3: Commit**

```
chore: exports, docs, clippy cleanup for persistent indexes
```

---

### Task 11: Example (optional)

Extend the `persist` example to demonstrate persistent index recovery. This validates the API from an external consumer.

**Files:**
- Modify: `examples/examples/persist.rs`

**Step 1: Add index persistence to the example**

After the existing WAL/snapshot demo, add:
- Build a `BTreeIndex` on a component
- Save it alongside the checkpoint
- On recovery, load the index and `update()`
- Print timings showing the speedup vs `rebuild()`

**Step 2: Run the example**

```bash
cargo run -p minkowski-examples --example persist --release
```

**Step 3: Commit**

```
example: demonstrate persistent index recovery in persist example
```
