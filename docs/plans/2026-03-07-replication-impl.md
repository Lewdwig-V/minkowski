# Replication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pull-based, transport-agnostic replication primitives: WalCursor, ReplicationBatch, apply_batch.

**Architecture:** Extract shared WAL reading/applying helpers from wal.rs, build WalCursor (read-only file cursor), ReplicationBatch (self-describing wire type), and apply_batch (standalone apply function) in a new replication.rs module. Wal::replay_from is refactored to use the shared helpers. No changes to codec.rs, snapshot.rs, durable.rs, or core minkowski crate.

**Tech Stack:** Rust, rkyv (serialization), minkowski-persist crate.

---

### Task 1: Extract read_next_frame helper + add CursorBehind error

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Context:** `Wal::read_next_entry` currently reads a WAL frame AND truncates on EOF/corruption (crash recovery). We need a shared version that reads without truncating, so `WalCursor` (read-only) can use it too. We also add the `CursorBehind` error variant now as a future hook.

**Step 1: Add the CursorBehind error variant**

In `wal.rs`, add a new variant to `WalError`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("WAL I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("WAL codec error: {0}")]
    Codec(#[from] CodecError),
    #[error("WAL format error: {0}")]
    Format(String),
    #[error("cursor behind: requested seq {requested} but oldest available is {oldest}")]
    CursorBehind { requested: u64, oldest: u64 },
}
```

**Step 2: Extract `read_next_frame` as a free function**

Add this function above the `Wal` impl block, after the existing `read_exact_at` function:

```rust
/// Try to read the next WAL entry at byte offset `pos`.
/// Returns `Ok(Some((entry, next_pos)))` on success, `Ok(None)` if EOF
/// or the frame is incomplete/corrupt. Does NOT truncate the file —
/// callers decide how to handle partial frames.
pub(crate) fn read_next_frame(file: &File, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
    let mut len_buf = [0u8; 4];
    match read_exact_at(file, pos, &mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    match read_exact_at(file, pos + 4, &mut payload) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    match rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(&payload) {
        Ok(entry) => Ok(Some((entry, pos + 4 + len as u64))),
        Err(_) => Ok(None),
    }
}
```

**Step 3: Refactor `Wal::read_next_entry` to use `read_next_frame`**

Replace the body of the existing `read_next_entry` method:

```rust
fn read_next_entry(&mut self, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
    match read_next_frame(&self.file, pos)? {
        Some(result) => Ok(Some(result)),
        None => {
            self.file.set_len(pos)?;
            Ok(None)
        }
    }
}
```

**Step 4: Run all existing WAL tests to verify no regression**

Run: `cargo test -p minkowski-persist --lib -- wal`
Expected: all 12 WAL tests pass (no behavior change)

**Step 5: Commit**

```
git add crates/minkowski-persist/src/wal.rs
git commit -m "refactor(persist): extract read_next_frame, add CursorBehind error"
```

---

### Task 2: Extract apply_record as a pub(crate) free function

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Context:** `Wal::apply_record` is currently a private static method on `Wal`. Both `Wal::replay_from` and the future `apply_batch` need it. Move it to a module-level `pub(crate)` function so `replication.rs` can call it.

**Step 1: Move `apply_record` from `impl Wal` to module level**

Cut the `apply_record` method from the `impl Wal` block and paste it as a free function, changing visibility to `pub(crate)`:

```rust
/// Apply a single WAL record to a World, optionally remapping component IDs.
pub(crate) fn apply_record(
    record: &crate::record::WalRecord,
    world: &mut World,
    codecs: &CodecRegistry,
    remap: Option<&HashMap<ComponentId, ComponentId>>,
) -> Result<(), WalError> {
    // ... body unchanged ...
}
```

**Step 2: Update the call site in `Wal::replay_from`**

Change `Self::apply_record(...)` to `apply_record(...)` (now a free function, not a method):

```rust
WalEntry::Mutations(record) => {
    if record.seq >= from_seq {
        apply_record(&record, world, codecs, remap.as_ref())?;
        last_seq = record.seq;
    }
}
```

**Step 3: Run all existing WAL tests**

Run: `cargo test -p minkowski-persist --lib -- wal`
Expected: all 12 WAL tests pass

**Step 4: Commit**

```
git add crates/minkowski-persist/src/wal.rs
git commit -m "refactor(persist): extract apply_record as pub(crate) free function"
```

---

### Task 3: Add ReplicationBatch type

**Files:**
- Create: `crates/minkowski-persist/src/replication.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Context:** `ReplicationBatch` is a self-describing, rkyv-serializable wire type. It lives in its own module alongside `WalCursor` and `apply_batch` (added in later tasks). Start with just the type + serialization + tests.

**Step 1: Write the failing test**

Create `crates/minkowski-persist/src/replication.rs` with:

```rust
use rkyv::{Archive, Deserialize, Serialize};

use crate::record::{WalRecord, WalSchema};
use crate::wal::WalError;

/// Self-describing replication payload. Every batch carries its own schema
/// so receivers can decode without prior handshake.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ReplicationBatch {
    pub schema: WalSchema,
    pub records: Vec<WalRecord>,
}

impl ReplicationBatch {
    /// Serialize to bytes via rkyv.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WalError> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map(|v| v.to_vec())
            .map_err(|e| WalError::Format(e.to_string()))
    }

    /// Deserialize from bytes via rkyv.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WalError> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Format(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{ComponentSchema, SerializedMutation};

    fn test_schema() -> WalSchema {
        WalSchema {
            components: vec![ComponentSchema {
                id: 0,
                name: "pos".into(),
                size: 8,
                align: 4,
            }],
        }
    }

    #[test]
    fn batch_round_trip() {
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![
                WalRecord {
                    seq: 0,
                    mutations: vec![SerializedMutation::Despawn { entity: 1 }],
                },
                WalRecord {
                    seq: 1,
                    mutations: vec![],
                },
            ],
        };

        let bytes = batch.to_bytes().unwrap();
        let restored = ReplicationBatch::from_bytes(&bytes).unwrap();

        assert_eq!(restored.records.len(), 2);
        assert_eq!(restored.records[0].seq, 0);
        assert_eq!(restored.records[1].seq, 1);
        assert_eq!(restored.schema.components.len(), 1);
        assert_eq!(restored.schema.components[0].name, "pos");
    }

    #[test]
    fn empty_batch_round_trip() {
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![],
        };

        let bytes = batch.to_bytes().unwrap();
        let restored = ReplicationBatch::from_bytes(&bytes).unwrap();
        assert!(restored.records.is_empty());
    }
}
```

**Step 2: Register the module in lib.rs**

Add to `crates/minkowski-persist/src/lib.rs`:

```rust
pub mod replication;

pub use replication::ReplicationBatch;
```

**Step 3: Run tests**

Run: `cargo test -p minkowski-persist --lib -- replication`
Expected: 2 tests pass (`batch_round_trip`, `empty_batch_round_trip`)

**Step 4: Commit**

```
git add crates/minkowski-persist/src/replication.rs crates/minkowski-persist/src/lib.rs
git commit -m "feat(persist): add ReplicationBatch wire type"
```

---

### Task 4: Implement WalCursor

**Files:**
- Modify: `crates/minkowski-persist/src/replication.rs`

**Context:** `WalCursor` opens a WAL file read-only, parses the schema preamble, and yields `ReplicationBatch`es on demand. Uses `read_next_frame` from Task 1. The schema is stored on the cursor and included in every batch produced by `next_batch`.

**Step 1: Write the failing tests**

Add these tests to the `#[cfg(test)] mod tests` block in `replication.rs`:

```rust
use crate::codec::CodecRegistry;
use minkowski::{EnumChangeSet, World};
use crate::wal::Wal;

#[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Debug)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Debug)]
struct Health(u32);

/// Helper: create a WAL with N spawn mutations and return the path.
fn create_test_wal(dir: &std::path::Path, n: usize) -> (std::path::PathBuf, CodecRegistry) {
    let wal_path = dir.join("test.wal");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let mut wal = Wal::create(&wal_path, &codecs).unwrap();
    for i in 0..n {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }
    (wal_path, codecs)
}

#[test]
fn cursor_reads_from_seq_zero() {
    let dir = tempfile::tempdir().unwrap();
    let (wal_path, _codecs) = create_test_wal(dir.path(), 3);

    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();

    assert_eq!(batch.records.len(), 3);
    assert_eq!(batch.records[0].seq, 0);
    assert_eq!(batch.records[1].seq, 1);
    assert_eq!(batch.records[2].seq, 2);
    assert_eq!(cursor.next_seq(), 3);

    // Schema should be present
    assert!(cursor.schema().is_some());
    assert_eq!(batch.schema.components.len(), 1);
    assert_eq!(batch.schema.components[0].name, "pos");
}

#[test]
fn cursor_reads_from_mid_seq() {
    let dir = tempfile::tempdir().unwrap();
    let (wal_path, _codecs) = create_test_wal(dir.path(), 5);

    let mut cursor = WalCursor::open(&wal_path, 3).unwrap();
    let batch = cursor.next_batch(100).unwrap();

    assert_eq!(batch.records.len(), 2);
    assert_eq!(batch.records[0].seq, 3);
    assert_eq!(batch.records[1].seq, 4);
    assert_eq!(cursor.next_seq(), 5);
}

#[test]
fn cursor_at_end_returns_empty_batch() {
    let dir = tempfile::tempdir().unwrap();
    let (wal_path, _codecs) = create_test_wal(dir.path(), 2);

    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
    let batch1 = cursor.next_batch(100).unwrap();
    assert_eq!(batch1.records.len(), 2);

    // Second call: cursor is at end
    let batch2 = cursor.next_batch(100).unwrap();
    assert!(batch2.records.is_empty());
}

#[test]
fn cursor_respects_batch_limit() {
    let dir = tempfile::tempdir().unwrap();
    let (wal_path, _codecs) = create_test_wal(dir.path(), 5);

    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();

    let batch1 = cursor.next_batch(2).unwrap();
    assert_eq!(batch1.records.len(), 2);
    assert_eq!(batch1.records[0].seq, 0);
    assert_eq!(batch1.records[1].seq, 1);

    let batch2 = cursor.next_batch(2).unwrap();
    assert_eq!(batch2.records.len(), 2);
    assert_eq!(batch2.records[0].seq, 2);
    assert_eq!(batch2.records[1].seq, 3);

    let batch3 = cursor.next_batch(2).unwrap();
    assert_eq!(batch3.records.len(), 1);
    assert_eq!(batch3.records[0].seq, 4);
}

#[test]
fn cursor_behind_error() {
    // Placeholder: CursorBehind is a future hook for WAL rotation.
    // Today nothing triggers it, but the error type exists.
    let err = WalError::CursorBehind {
        requested: 0,
        oldest: 5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("cursor behind"));
    assert!(msg.contains('0'));
    assert!(msg.contains('5'));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist --lib -- replication`
Expected: FAIL — `WalCursor` not defined

**Step 3: Implement WalCursor**

Add to `replication.rs`, above the `ReplicationBatch` definition:

```rust
use std::fs::File;
use std::path::Path;

use crate::record::{WalEntry, WalSchema};
use crate::wal::{read_next_frame, WalError};

/// Read-only cursor over a WAL file. Opens its own file handle so it
/// can read concurrently with an active writer.
pub struct WalCursor {
    file: File,
    pos: u64,
    next_seq: u64,
    schema: Option<WalSchema>,
}

impl WalCursor {
    /// Open a WAL file for reading, starting from `from_seq`.
    /// Parses the schema preamble (if present) and scans forward to the
    /// first record with `seq >= from_seq`.
    pub fn open(path: &Path, from_seq: u64) -> Result<Self, WalError> {
        let file = File::open(path)?;
        let mut pos: u64 = 0;
        let mut schema = None;

        // Read entries, collecting schema and skipping records before from_seq
        loop {
            match read_next_frame(&file, pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    schema = Some(s);
                    pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    if record.seq >= from_seq {
                        // Don't advance past this record — next_batch will read it
                        break;
                    }
                    pos = next_pos;
                }
                None => break, // end of file
            }
        }

        Ok(Self {
            file,
            pos,
            next_seq: from_seq,
            schema,
        })
    }

    /// Read up to `limit` records from the current position.
    /// Returns a `ReplicationBatch` with the schema and records.
    /// An empty `records` vec means the cursor has caught up.
    pub fn next_batch(&mut self, limit: usize) -> Result<ReplicationBatch, WalError> {
        let schema = self
            .schema
            .clone()
            .unwrap_or_else(|| WalSchema { components: vec![] });

        let mut records = Vec::new();

        while records.len() < limit {
            match read_next_frame(&self.file, self.pos)? {
                Some((WalEntry::Schema(_), next_pos)) => {
                    // Skip schema entries mid-file (shouldn't happen, but be safe)
                    self.pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    self.next_seq = record.seq + 1;
                    records.push(record);
                    self.pos = next_pos;
                }
                None => break, // end of file
            }
        }

        Ok(ReplicationBatch { schema, records })
    }

    /// The schema parsed from the WAL preamble, if present.
    pub fn schema(&self) -> Option<&WalSchema> {
        self.schema.as_ref()
    }

    /// Next expected sequence number. Useful for persisting cursor position.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski-persist --lib -- replication`
Expected: all 7 tests pass (2 batch + 5 cursor)

**Step 5: Commit**

```
git add crates/minkowski-persist/src/replication.rs
git commit -m "feat(persist): implement WalCursor"
```

---

### Task 5: Implement apply_batch

**Files:**
- Modify: `crates/minkowski-persist/src/replication.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Context:** `apply_batch` deserializes a `ReplicationBatch` and applies each record to a target World, using the shared `apply_record` from Task 2.

**Step 1: Write the failing tests**

Add these tests to `replication.rs`:

```rust
#[test]
fn apply_batch_spawns_entities() {
    let dir = tempfile::tempdir().unwrap();
    let (wal_path, codecs) = create_test_wal(dir.path(), 3);

    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();

    // Apply to a fresh world with the same codecs
    let mut replica = World::new();
    let mut replica_codecs = CodecRegistry::new();
    replica_codecs.register_as::<Pos>("pos", &mut replica);

    let last_seq = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

    assert_eq!(last_seq, 2);
    assert_eq!(replica.query::<(&Pos,)>().count(), 3);
}

#[test]
fn apply_batch_insert_remove() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Health>("health", &mut world);

    let mut wal = Wal::create(&wal_path, &codecs).unwrap();

    // Record 0: spawn entity with Pos
    let e = world.alloc_entity();
    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
    wal.append(&cs, &codecs).unwrap();
    cs.apply(&mut world);

    // Record 1: insert Health, remove Pos
    let mut cs2 = EnumChangeSet::new();
    cs2.insert::<Health>(&mut world, e, Health(100));
    cs2.remove::<Pos>(&mut world, e);
    wal.append(&cs2, &codecs).unwrap();
    cs2.apply(&mut world);

    drop(wal);

    // Replicate
    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();

    let mut replica = World::new();
    let mut replica_codecs = CodecRegistry::new();
    replica_codecs.register_as::<Pos>("pos", &mut replica);
    replica_codecs.register_as::<Health>("health", &mut replica);

    apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

    // Entity should have Health but not Pos
    assert_eq!(replica.query::<(&Health,)>().count(), 1);
    assert_eq!(replica.query::<(&Pos,)>().count(), 0);
    let h = replica.query::<(&Health,)>().next().unwrap().0;
    assert_eq!(h.0, 100);
}

#[test]
fn apply_batch_cross_process_remap() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("cross.wal");

    // Source: Pos=0, Health=1
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Health>("health", &mut world);

    let mut wal = Wal::create(&wal_path, &codecs).unwrap();
    let e = world.alloc_entity();
    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 }, Health(50)));
    wal.append(&cs, &codecs).unwrap();
    drop(wal);

    // Replica: Health=0, Pos=1 (opposite order)
    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();

    let mut replica = World::new();
    let mut replica_codecs = CodecRegistry::new();
    replica_codecs.register_as::<Health>("health", &mut replica);
    replica_codecs.register_as::<Pos>("pos", &mut replica);

    apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

    let positions: Vec<(f32, f32)> = replica
        .query::<(&Pos,)>()
        .map(|p| (p.0.x, p.0.y))
        .collect();
    assert_eq!(positions, vec![(1.0, 2.0)]);

    let health: Vec<u32> = replica.query::<(&Health,)>().map(|h| h.0 .0).collect();
    assert_eq!(health, vec![50]);
}

#[test]
fn apply_batch_preserves_transaction_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("boundaries.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let mut wal = Wal::create(&wal_path, &codecs).unwrap();

    // Record 0: spawn entity
    let e = world.alloc_entity();
    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
    wal.append(&cs, &codecs).unwrap();
    cs.apply(&mut world);

    // Record 1: despawn the same entity
    let mut cs2 = EnumChangeSet::new();
    cs2.record_despawn(e);
    wal.append(&cs2, &codecs).unwrap();
    cs2.apply(&mut world);

    drop(wal);

    // Replicate both records in one batch
    let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    assert_eq!(batch.records.len(), 2);

    let mut replica = World::new();
    let mut replica_codecs = CodecRegistry::new();
    replica_codecs.register_as::<Pos>("pos", &mut replica);

    apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

    // Entity was spawned then despawned — should not be alive
    assert_eq!(replica.query::<(&Pos,)>().count(), 0);
}

#[test]
fn apply_empty_batch() {
    let batch = ReplicationBatch {
        schema: WalSchema { components: vec![] },
        records: vec![],
    };

    let mut world = World::new();
    let codecs = CodecRegistry::new();

    let last_seq = apply_batch(&batch, &mut world, &codecs).unwrap();
    assert_eq!(last_seq, 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist --lib -- replication`
Expected: FAIL — `apply_batch` not defined

**Step 3: Implement apply_batch**

Add to `replication.rs`:

```rust
use std::collections::HashMap;

use minkowski::{ComponentId, World};

use crate::codec::CodecRegistry;
use crate::wal::apply_record;

/// Apply a `ReplicationBatch` to a target World.
///
/// Builds a component ID remap from the batch schema, then applies each
/// record atomically (one `EnumChangeSet` per record). Returns the last
/// applied seq, or 0 if the batch is empty.
pub fn apply_batch(
    batch: &ReplicationBatch,
    world: &mut World,
    codecs: &CodecRegistry,
) -> Result<u64, WalError> {
    let remap: Option<HashMap<ComponentId, ComponentId>> = if batch.schema.components.is_empty() {
        None
    } else {
        Some(codecs.build_remap(&batch.schema.components)?)
    };

    let mut last_seq = 0u64;
    for record in &batch.records {
        apply_record(record, world, codecs, remap.as_ref())?;
        last_seq = record.seq;
    }

    Ok(last_seq)
}
```

**Step 4: Update lib.rs exports**

Update the replication exports in `crates/minkowski-persist/src/lib.rs`:

```rust
pub use replication::{apply_batch, ReplicationBatch, WalCursor};
```

**Step 5: Run tests**

Run: `cargo test -p minkowski-persist --lib -- replication`
Expected: all 12 tests pass

**Step 6: Commit**

```
git add crates/minkowski-persist/src/replication.rs crates/minkowski-persist/src/lib.rs
git commit -m "feat(persist): implement apply_batch"
```

---

### Task 6: Full replication integration test

**Files:**
- Modify: `crates/minkowski-persist/src/replication.rs`

**Context:** End-to-end test: Durable source writes mutations, snapshot taken, cursor reads from snapshot seq, apply_batch to replica, verify convergence.

**Step 1: Write the integration test**

Add to `replication.rs` tests:

```rust
#[test]
fn full_replication_flow() {
    use crate::durable::Durable;
    use crate::snapshot::Snapshot;
    use minkowski::Optimistic;

    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("source.wal");
    let snap_path = dir.path().join("source.snap");

    // -- Source setup --
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Health>("health", &mut world);

    // Spawn some entities
    for i in 0..5 {
        world.spawn((Pos { x: i as f32, y: 0.0 }, Health(100)));
    }

    // Take snapshot
    let wal = Wal::create(&wal_path, &codecs).unwrap();
    let snap = Snapshot::new();
    let header = snap
        .save(&snap_path, &world, &codecs, wal.next_seq())
        .unwrap();
    assert_eq!(header.entity_count, 5);

    // Write durable mutations after snapshot
    let strategy = Optimistic::new(&world);
    let durable = Durable::new(strategy, wal, codecs.clone());

    // Mutate: insert Health on new entities
    for _ in 0..3 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },));
        durable.append_and_apply(&cs, &mut world);
    }

    let wal_seq = durable.wal_seq();

    // -- Replica setup --
    let mut replica_codecs = CodecRegistry::new();
    let mut tmp = World::new();
    replica_codecs.register_as::<Pos>("pos", &mut tmp);
    replica_codecs.register_as::<Health>("health", &mut tmp);

    // Load snapshot
    let (mut replica, snap_seq) = snap.load(&snap_path, &replica_codecs).unwrap();
    assert_eq!(replica.query::<(&Pos,)>().count(), 5);

    // Pull WAL records from snapshot seq
    let mut cursor = WalCursor::open(&wal_path, snap_seq).unwrap();
    let batch = cursor.next_batch(100).unwrap();

    // Apply batch
    let last = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

    // Verify convergence
    assert_eq!(replica.query::<(&Pos,)>().count(), 8); // 5 from snapshot + 3 from WAL
    assert!(last >= snap_seq);
}
```

**Step 2: Run the test**

Run: `cargo test -p minkowski-persist --lib -- replication::tests::full_replication_flow`

Note: This test may need adjustment depending on whether `Durable` has an `append_and_apply` method or if we need to use `transact`. Check the `Durable` API first. If `Durable` doesn't have `append_and_apply`, we can write directly to the WAL and apply the changeset manually (same pattern as other WAL tests). The implementer should read `durable.rs` and adjust accordingly.

**Step 3: Fix any issues and verify all tests pass**

Run: `cargo test -p minkowski-persist --lib -- replication`
Expected: all 13 tests pass

**Step 4: Commit**

```
git add crates/minkowski-persist/src/replication.rs
git commit -m "test(persist): full replication flow integration test"
```

---

### Task 7: Add replicate example

**Files:**
- Create: `examples/examples/replicate.rs`
- Modify: `examples/Cargo.toml` (add `[[example]]` entry if needed)

**Context:** Demonstrates the full replication flow in a user-facing example. In-process (no transport), proving the primitives work.

**Step 1: Check examples/Cargo.toml structure**

Read `examples/Cargo.toml` to see how existing examples are declared. If examples are auto-discovered (no `[[example]]` entries), skip the Cargo.toml edit. If they have explicit entries, add one for `replicate`.

**Step 2: Write the example**

Create `examples/examples/replicate.rs`:

```rust
//! Replication — pull-based WAL cursor for read replicas.
//!
//! Demonstrates the full replication lifecycle:
//! 1. Create a source world and spawn entities
//! 2. Write mutations via Durable transactions (WAL-backed)
//! 3. Take a snapshot
//! 4. Open a WalCursor at the snapshot seq
//! 5. Pull a ReplicationBatch and apply to a replica world
//! 6. Verify source and replica converge
//!
//! Run: cargo run -p minkowski-examples --example replicate --release

use minkowski::{EnumChangeSet, Optimistic, World};
use minkowski_persist::{
    apply_batch, CodecRegistry, Durable, ReplicationBatch, Snapshot, Wal, WalCursor,
};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Vel {
    dx: f32,
    dy: f32,
}

fn main() {
    let dir = std::env::temp_dir().join("minkowski-replicate-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("source.wal");
    let snap_path = dir.join("source.snap");

    // Clean up from previous runs
    let _ = std::fs::remove_file(&wal_path);
    let _ = std::fs::remove_file(&snap_path);

    // -- Phase 1: Source world --
    println!("Phase 1: Creating source world...");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Vel>("vel", &mut world);

    for i in 0..20 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }
    println!("  Spawned {} entities", world.entity_count());

    // -- Phase 2: Snapshot + WAL mutations --
    println!("Phase 2: Snapshot + durable mutations...");
    let wal = Wal::create(&wal_path, &codecs).unwrap();
    let snap = Snapshot::new();
    let header = snap
        .save(&snap_path, &world, &codecs, wal.next_seq())
        .unwrap();
    println!(
        "  Snapshot: {} entities, WAL seq {}",
        header.entity_count, header.wal_seq
    );

    // Write mutations after snapshot
    let strategy = Optimistic::new(&world);
    let durable = Durable::new(strategy, wal, codecs.clone());

    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (
                Pos {
                    x: 100.0 + i as f32,
                    y: 100.0,
                },
                Vel {
                    dx: -1.0,
                    dy: -0.5,
                },
            ),
        );
        // Write to WAL then apply
        let wal_lock = durable.wal();
        wal_lock.lock().append(&cs, durable.codecs()).unwrap();
        cs.apply(&mut world);
    }
    println!(
        "  After mutations: {} entities, WAL seq {}",
        world.entity_count(),
        durable.wal_seq()
    );

    // -- Phase 3: Replica via snapshot + cursor --
    println!("Phase 3: Replicating...");
    let mut replica_codecs = CodecRegistry::new();
    let mut tmp = World::new();
    replica_codecs.register_as::<Pos>("pos", &mut tmp);
    replica_codecs.register_as::<Vel>("vel", &mut tmp);

    let (mut replica, snap_seq) = snap.load(&snap_path, &replica_codecs).unwrap();
    println!(
        "  Loaded snapshot: {} entities at seq {}",
        replica.query::<(&Pos,)>().count(),
        snap_seq
    );

    let mut cursor = WalCursor::open(&wal_path, snap_seq).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    println!(
        "  Pulled batch: {} records, schema has {} components",
        batch.records.len(),
        batch.schema.components.len()
    );

    // Demonstrate wire format
    let wire_bytes = batch.to_bytes().unwrap();
    println!("  Wire format: {} bytes", wire_bytes.len());
    let batch = ReplicationBatch::from_bytes(&wire_bytes).unwrap();

    let last_seq = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();
    println!("  Applied up to seq {}", last_seq);

    // -- Phase 4: Verify convergence --
    println!("Phase 4: Verifying convergence...");
    let source_count = world.entity_count();
    let replica_count = replica.query::<(&Pos,)>().count();
    println!(
        "  Source: {} entities, Replica: {} entities",
        source_count, replica_count
    );
    assert_eq!(source_count, replica_count, "replica should match source");

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone. Source and replica converged.");
}
```

Note: The implementer should check whether `Durable` exposes `wal()` and `codecs()` methods. If not, use `Wal` directly (same pattern as the `persist` example). Adjust the mutation write loop accordingly — the key point is writing mutations to the WAL and applying them to the source world.

**Step 3: Run the example**

Run: `cargo run -p minkowski-examples --example replicate --release`
Expected: prints phases 1-4, asserts convergence, exits cleanly.

**Step 4: Run clippy and fmt**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean

**Step 5: Commit**

```
git add examples/examples/replicate.rs
git commit -m "feat(examples): add replicate example"
```

---

### Task 8: Final validation + CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md` (update example list and test count)
- Modify: `README.md` (add replicate example to examples table if one exists)

**Step 1: Run full test suite**

Run: `cargo test -p minkowski-persist`
Expected: all persist tests pass (existing WAL/snapshot + new replication tests)

Run: `cargo test -p minkowski`
Expected: all 368 core tests pass (no changes to core)

**Step 2: Run clippy + fmt**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean

**Step 3: Update CLAUDE.md**

Add the replicate example to the example commands list:

```
cargo run -p minkowski-examples --example replicate --release   # Pull-based WAL replication: cursor + batch + apply to replica (20 source + 10 WAL, convergence check)
```

Update the persist test count if it changed.

**Step 4: Update README.md**

If README has an examples table, add the replicate example. Match existing format.

**Step 5: Commit**

```
git add CLAUDE.md README.md
git commit -m "docs: add replicate example, update test counts"
```
