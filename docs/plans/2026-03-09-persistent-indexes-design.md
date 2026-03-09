# Persistent Indexes

**Date:** 2026-03-09
**Status:** Approved

## Problem

After crash recovery (snapshot restore + WAL replay), secondary indexes (`BTreeIndex`, `HashIndex`) must be rebuilt from scratch via `rebuild()`, which scans every entity with the indexed component. For large worlds this is O(n) and can take seconds — a startup latency spike and potential DoS vector.

## Decision

Make indexes persistable so recovery loads the index from disk and applies only the WAL tail. Recovery time becomes proportional to the WAL tail, not world size.

## Design

### `PersistentIndex` trait

```rust
// In minkowski-persist
pub trait PersistentIndex: SpatialIndex {
    fn save(&self, path: &Path) -> Result<(), IndexPersistError>;
}
```

Object-safe — `AutoCheckpoint` can hold `Vec<(&Path, &dyn PersistentIndex)>` and call `save` on each one.

`load` is on the concrete types (returns `Self`, not object-safe):

```rust
impl<T: ...> BTreeIndex<T> {
    pub fn load(path: &Path) -> Result<Self, IndexPersistError>;
}
```

### Serialization boundary

`BTreeIndex` and `HashIndex` are defined in `minkowski` (core). `PersistentIndex` impls live in `minkowski-persist`. To bridge this without exposing private fields:

```rust
// In minkowski, on BTreeIndex<T> and HashIndex<T>
pub fn as_raw_parts(&self) -> (&BTreeMap<T, Vec<Entity>>, &HashMap<Entity, T>, ChangeTick);
pub fn from_raw_parts(tree: BTreeMap<T, Vec<Entity>>, reverse: HashMap<Entity, T>, last_sync: ChangeTick) -> Self;
```

Core provides the mechanism (controlled access to internals). Persist crate provides the policy (rkyv serialization).

### File format

```
[magic: 8B "MK2INDXK"]
[crc32: 4B LE]
[reserved: 4B]
[len: u64 LE]
[rkyv payload: len bytes]
```

Same envelope as snapshots and WAL segments. CRC32 (IEEE via `crc32fast`) over the rkyv payload.

Payload structure:

```rust
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct IndexData<T> {
    entries: Vec<(T, Vec<u64>)>,   // forward map (key -> entity bits)
    reverse: Vec<(u64, T)>,        // reverse map (entity bits -> key)
    last_sync_tick: u64,           // ChangeTick at last save
}
```

BTreeIndex and HashIndex share the same serialized representation — the difference is which collection they load into.

### Conditional implementation

Not every `BTreeIndex<T>` is persistable — only those whose key type supports rkyv:

```rust
impl<T> PersistentIndex for BTreeIndex<T>
where
    T: Component + Ord + Clone + rkyv::Archive + rkyv::Serialize<...> + rkyv::Deserialize<...>
{ ... }
```

### Error type

```rust
#[derive(Debug, thiserror::Error)]
pub enum IndexPersistError {
    #[error("index I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("index format error: {0}")]
    Format(String),
}
```

### AutoCheckpoint integration

```rust
impl AutoCheckpoint {
    pub fn register_index(&mut self, path: PathBuf, index: Arc<Mutex<dyn PersistentIndex>>);
}
```

On checkpoint fire:
1. Save snapshot (existing)
2. For each registered index: `index.lock().save(&path)?`
3. Acknowledge snapshot (existing)

Index save failures are non-fatal — same as checkpoint handler errors. The snapshot and WAL are the source of truth; indexes are a performance optimization.

`Arc<Mutex<>>` is necessary because `AutoCheckpoint` holds index references across checkpoint calls while the user also needs mutable access for `update()` between checkpoints.

### Recovery flow

```rust
// Try to load persistent index, fall back to rebuild
let mut score_index = match BTreeIndex::<Score>::load("index-score.idx") {
    Ok(idx) => idx,
    Err(_) => {
        let mut idx = BTreeIndex::new();
        idx.rebuild(&mut world);
        idx
    }
};
score_index.update(&mut world); // catch up with WAL tail
```

### WAL catch-up

The loaded `last_sync` tick enables the existing `update()` path — `query_changed_since(last_sync)` returns exactly the entities that changed after the index was saved. No new replay infrastructure needed. Tick values survive the persistence round-trip because WAL replay marks columns with the original commit ticks.

### Crate placement

| Component | Crate |
|---|---|
| `as_raw_parts` / `from_raw_parts` | `minkowski` (index types) |
| `PersistentIndex` trait | `minkowski-persist` |
| `save` / `load` impls | `minkowski-persist` |
| `IndexPersistError` | `minkowski-persist` |
| `AutoCheckpoint::register_index` | `minkowski-persist` |

Core crate gains no new dependencies. No feature flags.

## Testing strategy

**Unit tests (minkowski-persist):**
- Round-trip: build index, save, load, verify entries match
- CRC mismatch: corrupt payload byte, verify `IndexPersistError::Format`
- Missing file: `load` on nonexistent path returns `Io` error
- Empty index: save/load round-trip with zero entries
- Stale index catch-up: save, mutate world, load, `update`, verify

**Integration test:**
- Full recovery: spawn, build index, save, checkpoint, append WAL, drop everything, restore snapshot, replay WAL, load index, `update`, verify index matches world

## Key properties

- Core crate unchanged (no rkyv dependency)
- Persistence is opt-in per index
- Recovery time proportional to WAL tail, not world size
- `rebuild()` remains as fallback for corrupt/missing index files
- `SpatialIndex` trait untouched
- Consistent file format envelope across WAL, snapshots, and indexes
