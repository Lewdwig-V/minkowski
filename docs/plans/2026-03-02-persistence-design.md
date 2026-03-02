# Persistence — WAL + Bincode Snapshots

## Problem

Minkowski has no durable state. All world data lives in memory and is lost on process exit. The transaction system already captures mutations as `EnumChangeSet` — a natural WAL entry — but there's no serialization layer or I/O plumbing to persist them.

## Scope

WAL + bincode snapshots. rkyv zero-copy snapshots are a future format addition behind `WireFormat`, not a migration. Bincode snapshots serialize through serde, decoupling the on-disk format from in-memory layout. This is the right property while internals are still evolving.

## Architecture

### Crate: `minkowski-persist`

New workspace member. Depends on `minkowski` + `serde` + `bincode`. Core crate stays serde-free. The persist crate composes from existing primitives — same external composition pattern as `SpatialIndex`, `Access`, and `TransactionStrategy`.

### CodecRegistry

Maps `ComponentId` → serialize/deserialize function pairs. Separate from core's `ComponentRegistry` — different concerns, different crates, different lifetimes. Core maps types to IDs and stores layout metadata. The persist crate maps IDs to serde codecs.

```rust
pub struct CodecRegistry {
    codecs: HashMap<ComponentId, ComponentCodec>,
}

struct ComponentCodec {
    name: &'static str,
    layout: Layout,
    serialize_fn: unsafe fn(*const u8, &mut dyn erased_serde::Serializer) -> Result<()>,
    deserialize_fn: fn(&mut dyn erased_serde::Deserializer) -> Result<OwnedBytes>,
}

impl CodecRegistry {
    /// Register a component type for persistence. Requires Serialize + DeserializeOwned.
    /// Uses world to obtain the ComponentId (idempotent registration).
    pub fn register<T: Component + Serialize + DeserializeOwned>(&mut self, world: &mut World);

    /// Serialize a component's raw bytes through its registered codec.
    pub fn serialize(&self, id: ComponentId, ptr: *const u8, out: &mut Vec<u8>) -> Result<()>;

    /// Deserialize component bytes, returning owned allocation.
    pub fn deserialize(&self, id: ComponentId, bytes: &[u8]) -> Result<Vec<u8>>;
}
```

Components without a registered codec cause errors, not silent skips.

### WireFormat trait

Abstracts the serialization format. Bincode now, rkyv later as a second impl.

```rust
pub trait WireFormat {
    fn serialize_record(&self, record: &WalRecord, out: &mut Vec<u8>) -> Result<()>;
    fn deserialize_record(&self, bytes: &[u8]) -> Result<WalRecord>;
    fn serialize_snapshot(&self, snapshot: &SnapshotData, out: &mut Vec<u8>) -> Result<()>;
    fn deserialize_snapshot(&self, bytes: &[u8]) -> Result<SnapshotData>;
}

pub struct Bincode;
impl WireFormat for Bincode { ... }
```

### WAL

Append-only log of serialized `EnumChangeSet`s with monotonic sequence numbers. Fully external to the transaction system — the user calls `wal.append()` after `tx.commit()`. The persist crate never touches `transaction.rs`.

```rust
pub struct Wal<W: WireFormat> {
    file: File,
    format: W,
    next_seq: u64,
}

impl<W: WireFormat> Wal<W> {
    pub fn create(path: &Path, format: W) -> Result<Self>;
    pub fn open(path: &Path, format: W) -> Result<Self>;
    pub fn append(&mut self, changeset: &EnumChangeSet, codecs: &CodecRegistry) -> Result<u64>;
    pub fn replay(&self, world: &mut World, codecs: &CodecRegistry) -> Result<u64>;
    pub fn replay_from(&self, seq: u64, world: &mut World, codecs: &CodecRegistry) -> Result<u64>;
}
```

**Record format**: `[len: u32][seq: u64][payload: Vec<u8>]`. Length-prefixed for streaming reads. Each payload is a `WalRecord` serialized through `WireFormat`, containing the mutation list with component bytes encoded through codecs.

**WalRecord**:

```rust
struct WalRecord {
    seq: u64,
    mutations: Vec<SerializedMutation>,
}

enum SerializedMutation {
    Spawn { entity: u64, components: Vec<(ComponentId, Vec<u8>)> },
    Despawn { entity: u64 },
    Insert { entity: u64, component_id: ComponentId, data: Vec<u8> },
    Remove { entity: u64, component_id: ComponentId },
}
```

This is the serde-friendly mirror of core's `Mutation` enum. Entity is serialized as raw u64 (preserving generation). Component data goes through CodecRegistry.

### Snapshots

Hybrid format: archetype metadata header + column blobs serialized through codecs. `query_raw(&self)` is the read path — no side effects, no tick advancement.

```rust
pub struct Snapshot<W: WireFormat> {
    format: W,
}

impl<W: WireFormat> Snapshot<W> {
    pub fn save(&self, path: &Path, world: &World, codecs: &CodecRegistry) -> Result<SnapshotHeader>;
    pub fn load(&self, path: &Path, codecs: &CodecRegistry) -> Result<World>;
}
```

**SnapshotData layout**:

```rust
struct SnapshotData {
    wal_seq: u64,
    schema: Vec<ComponentSchema>,
    allocator: AllocatorState,
    archetypes: Vec<ArchetypeData>,
    sparse: Vec<SparseComponentData>,
}

struct ComponentSchema {
    id: ComponentId,
    name: String,
    size: usize,
    align: usize,
}

struct AllocatorState {
    generations: Vec<u32>,
    free_list: Vec<u32>,
}

struct ArchetypeData {
    component_ids: Vec<ComponentId>,
    entities: Vec<u64>,
    columns: Vec<ColumnData>,
}

struct ColumnData {
    component_id: ComponentId,
    values: Vec<Vec<u8>>,  // one encoded blob per row
}

/// Sparse components live in HashMap<Entity, T>, not archetype columns.
/// Without this section, they silently vanish on save and restore.
struct SparseComponentData {
    component_id: ComponentId,
    entries: Vec<(u64, Vec<u8>)>,  // (entity_raw, serialized_value)
}
```

Recovery: `Snapshot::load(path, codecs)` → `Wal::replay_from(header.wal_seq, &mut world, codecs)`.

### Core changes

Minimal. `query_raw` already provides the component data read path for archetype columns. Small additions:

1. **`World::entity_allocator_state(&self)`** — returns a read-only view of generations vec and free list for snapshot serialization.
2. **`World::archetype_layouts(&self)`** — returns iterator over archetype metadata (component IDs, entity list, column count) for snapshot structure. Or expose through existing `query_raw` machinery if sufficient.
3. **`World::sparse_component_ids(&self)`** — returns which ComponentIds are sparse-registered. The persist crate needs this to know which codecs to use for the sparse section.
4. **`World::read_sparse<T: Component>(&self, comp_id) -> impl Iterator<Item = (Entity, &T)>`** — typed read-only iteration over a sparse component's HashMap. The CodecRegistry knows the concrete type (it was registered with `register::<T>()`), so it can call this accessor with the right type parameter.

All are `&self` methods — pure reads, no mutation.

### What the persist crate does NOT do

- Modify core's `ComponentRegistry` or `ComponentInfo`
- Touch `transaction.rs` or the commit path
- Add serde dependencies to the core crate
- Handle concurrent writers (single-writer WAL for now)
- Schema migration or versioning (component names enable future validation)
- Log rotation or compaction (append-only, truncated by new snapshot)

### Integration with transactions

Fully external. The user controls when WAL entries are written:

```rust
let mut tx = strategy.begin(&mut world, &access);
// ... execute ...
let reverse = tx.commit(&mut world)?;
wal.append(&reverse, &codecs)?;  // user decides to persist
```

The reverse changeset from `commit()` is the natural WAL entry — it captures exactly what changed. If the user wants to persist the forward changeset instead (for replay rather than undo), they construct it themselves.

### Testing

- Round-trip: serialize changeset → deserialize → apply → compare world state
- WAL replay: create world, apply N changesets via WAL, verify final state
- Snapshot round-trip: save world → load into new world → compare
- Recovery: snapshot at seq N → append more WAL entries → recover → verify
- Sparse round-trip: world with sparse components → snapshot → restore → verify sparse data present
- Mixed archetype + sparse: world with both dense and sparse components → full round-trip
- Missing codec: attempt to serialize unregistered component → error
- Missing sparse codec: sparse component without codec → error (not silent skip)
- Empty world: snapshot/restore of empty world
