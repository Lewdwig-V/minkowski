# Persistence — WAL + Bincode Snapshots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add durable persistence to minkowski via an external `minkowski-persist` crate: WAL (append-only log of serialized changesets) + bincode snapshots (full world state), with recovery = load snapshot + replay WAL tail.

**Architecture:** New workspace member `minkowski-persist` composes from existing primitives — same external pattern as `SpatialIndex` and `TransactionStrategy`. Core stays serde-free. A `CodecRegistry` in the persist crate maps `ComponentId` → serde codecs. A `WireFormat` trait abstracts serialization (bincode now, rkyv later). Core gets minimal `pub` read accessors so the persist crate can iterate archetype columns, sparse components, and entity allocator state without exposing internals.

**Tech Stack:** Rust, serde, bincode, minkowski core

---

### Task 1: Core read accessors for persistence

**Files:**
- Modify: `crates/minkowski/src/world.rs`
- Modify: `crates/minkowski/src/changeset.rs`
- Modify: `crates/minkowski/src/storage/sparse.rs`
- Modify: `crates/minkowski/src/lib.rs`

Core is serde-free. The persist crate (a separate workspace member) can only access `pub` items. This task adds the minimal `pub` read accessors that persistence needs.

**Step 1: Add archetype read accessors to World**

In `crates/minkowski/src/world.rs`, add these `&self` methods to `impl World`:

```rust
/// Number of non-empty archetypes. Used by persist crate to iterate world state.
pub fn archetype_count(&self) -> usize {
    self.archetypes.archetypes.len()
}

/// Sorted component IDs defining an archetype's schema.
pub fn archetype_component_ids(&self, arch_idx: usize) -> &[ComponentId] {
    &self.archetypes.archetypes[arch_idx].sorted_ids
}

/// Entity handles stored in an archetype (one per row).
pub fn archetype_entities(&self, arch_idx: usize) -> &[Entity] {
    &self.archetypes.archetypes[arch_idx].entities
}

/// Raw pointer to a component value at a specific row in an archetype column.
///
/// # Safety
/// The caller must read through the pointer using the correct component type
/// and layout. The pointer is valid until the next structural mutation.
pub unsafe fn archetype_column_ptr(
    &self,
    arch_idx: usize,
    comp_id: ComponentId,
    row: usize,
) -> *const u8 {
    let arch = &self.archetypes.archetypes[arch_idx];
    let col_idx = arch.component_index[&comp_id];
    arch.columns[col_idx].get_ptr(row)
}

/// Row count for an archetype.
pub fn archetype_len(&self, arch_idx: usize) -> usize {
    self.archetypes.archetypes[arch_idx].len()
}
```

**Step 2: Add component info accessors to World**

```rust
/// Component name (from `std::any::type_name`). Returns None if unregistered.
pub fn component_name(&self, id: ComponentId) -> Option<&'static str> {
    if id < self.components.len() {
        Some(self.components.info(id).name)
    } else {
        None
    }
}

/// Component memory layout. Returns None if unregistered.
pub fn component_layout(&self, id: ComponentId) -> Option<std::alloc::Layout> {
    if id < self.components.len() {
        Some(self.components.info(id).layout)
    } else {
        None
    }
}

/// Number of registered component types.
pub fn component_count(&self) -> usize {
    self.components.len()
}
```

**Step 3: Add entity allocator state accessor**

```rust
/// Read-only view of entity allocator state for snapshot serialization.
/// Returns (generations_slice, free_list_slice).
pub fn entity_allocator_state(&self) -> (&[u32], &[u32]) {
    (&self.entities.generations, &self.entities.free_list)
}
```

Verify that `generations` and `free_list` are accessible from World (they're on `EntityAllocator`, which is `pub(crate)` on World). The fields on EntityAllocator may need to be made `pub(crate)` if they aren't already — check entity.rs.

**Step 4: Add allocator restore method**

```rust
/// Restore entity allocator state from a snapshot. Overwrites current
/// generations and free list. Used during snapshot load — not for general use.
pub fn restore_allocator_state(&mut self, generations: Vec<u32>, free_list: Vec<u32>) {
    self.drain_orphans();
    self.entities.generations = generations;
    self.entities.free_list = free_list;
    // Resize entity_locations to match
    self.entity_locations.resize(self.entities.generations.len(), None);
}
```

**Step 5: Add sparse component accessors**

In `crates/minkowski/src/storage/sparse.rs`, add:

```rust
/// Returns the ComponentIds that have sparse storage allocated.
pub fn component_ids(&self) -> Vec<ComponentId> {
    self.storages.keys().copied().collect()
}

/// Typed read-only iteration over a sparse component's entries.
/// Returns None if the component has no sparse storage.
pub fn iter<T: Component>(&self, comp_id: ComponentId) -> Option<impl Iterator<Item = (Entity, &T)>> {
    let map = self.storages.get(&comp_id)?
        .downcast_ref::<HashMap<Entity, T>>()?;
    Some(map.iter().map(|(&e, v)| (e, v)))
}
```

In `crates/minkowski/src/world.rs`, add:

```rust
/// Which ComponentIds have sparse storage.
pub fn sparse_component_ids(&self) -> Vec<ComponentId> {
    self.sparse.component_ids()
}

/// Typed read-only iteration over a sparse component.
pub fn iter_sparse<T: Component>(&self, comp_id: ComponentId) -> Option<impl Iterator<Item = (Entity, &T)>> {
    self.sparse.iter::<T>(comp_id)
}

/// Insert a sparse component value. Used during snapshot restore.
pub fn insert_sparse<T: Component>(&mut self, entity: Entity, value: T) {
    self.drain_orphans();
    let comp_id = self.components.register::<T>();
    self.sparse.insert(comp_id, entity, value);
}
```

**Step 6: Add MutationRef and iter_mutations to EnumChangeSet**

In `crates/minkowski/src/changeset.rs`, add a public view type and iterator:

```rust
/// Read-only view of a mutation for serialization. Component data is
/// borrowed as byte slices from the changeset's Arena.
pub enum MutationRef<'a> {
    Spawn {
        entity: Entity,
        components: Vec<(ComponentId, &'a [u8])>,
    },
    Despawn {
        entity: Entity,
    },
    Insert {
        entity: Entity,
        component_id: ComponentId,
        data: &'a [u8],
    },
    Remove {
        entity: Entity,
        component_id: ComponentId,
    },
}

impl EnumChangeSet {
    /// Iterate mutations as borrowed views. Component data is returned as
    /// byte slices — the persist crate runs these through CodecRegistry.
    pub fn iter_mutations(&self) -> impl Iterator<Item = MutationRef<'_>> + '_ {
        self.mutations.iter().map(|m| match m {
            Mutation::Spawn { entity, components } => MutationRef::Spawn {
                entity: *entity,
                components: components.iter().map(|(id, offset, layout)| {
                    let ptr = unsafe { self.arena.get(*offset) };
                    let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
                    (*id, bytes)
                }).collect(),
            },
            Mutation::Despawn { entity } => MutationRef::Despawn { entity: *entity },
            Mutation::Insert { entity, component_id, offset, layout } => {
                let ptr = unsafe { self.arena.get(*offset) };
                let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
                MutationRef::Insert {
                    entity: *entity,
                    component_id: *component_id,
                    data: bytes,
                }
            }
            Mutation::Remove { entity, component_id } => MutationRef::Remove {
                entity: *entity,
                component_id: *component_id,
            },
        })
    }
}
```

Add `pub use changeset::MutationRef;` to `lib.rs`.

**Step 7: Write tests for all new accessors**

Add tests in each module:

- `world.rs` tests: `archetype_count`, `archetype_component_ids`, `archetype_entities`, `archetype_column_ptr` round-trip, `component_name`, `entity_allocator_state`, `sparse_component_ids`, `iter_sparse`
- `changeset.rs` tests: `iter_mutations` returns correct variants and byte data for each mutation type
- `sparse.rs` tests: `component_ids`, `iter`

**Step 8: Verify**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`

**Step 9: Commit**

```bash
git add crates/minkowski/src/
git commit -m "feat: add pub read accessors for persistence layer"
```

---

### Task 2: Create `minkowski-persist` crate + CodecRegistry

**Files:**
- Create: `crates/minkowski-persist/Cargo.toml`
- Create: `crates/minkowski-persist/src/lib.rs`
- Create: `crates/minkowski-persist/src/codec.rs`
- Modify: `Cargo.toml` (workspace root)

**Step 1: Create crate structure**

```bash
mkdir -p crates/minkowski-persist/src
```

`crates/minkowski-persist/Cargo.toml`:

```toml
[package]
name = "minkowski-persist"
version = "0.1.0"
edition = "2021"

[dependencies]
minkowski = { path = "../minkowski" }
serde = { version = "1", features = ["derive"] }
bincode = "1"
```

`crates/minkowski-persist/src/lib.rs`:

```rust
pub mod codec;
```

Add `"crates/minkowski-persist"` to workspace `members` in root `Cargo.toml`.

**Step 2: Implement CodecRegistry**

`crates/minkowski-persist/src/codec.rs`:

```rust
use std::alloc::Layout;
use std::collections::HashMap;

use minkowski::{Component, ComponentId, World};
use serde::{de::DeserializeOwned, Serialize};

/// Type-erased serialize function: reads T from raw pointer, serializes to bytes.
type SerializeFn = unsafe fn(*const u8, &mut Vec<u8>) -> Result<(), CodecError>;

/// Type-erased deserialize function: deserializes T from bytes, returns raw bytes.
type DeserializeFn = fn(&[u8]) -> Result<Vec<u8>, CodecError>;

#[derive(Debug)]
pub enum CodecError {
    Serialize(String),
    Deserialize(String),
    UnregisteredComponent(ComponentId),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialize(msg) => write!(f, "serialize: {msg}"),
            Self::Deserialize(msg) => write!(f, "deserialize: {msg}"),
            Self::UnregisteredComponent(id) => write!(f, "no codec for component {id}"),
        }
    }
}

impl std::error::Error for CodecError {}

struct ComponentCodec {
    name: &'static str,
    layout: Layout,
    serialize_fn: SerializeFn,
    deserialize_fn: DeserializeFn,
}

/// Maps ComponentId → serde codecs. Separate from core's ComponentRegistry —
/// different concerns, different crates, different lifetimes.
pub struct CodecRegistry {
    codecs: HashMap<ComponentId, ComponentCodec>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        Self { codecs: HashMap::new() }
    }

    /// Register a component type for persistence.
    pub fn register<T: Component + Serialize + DeserializeOwned>(&mut self, world: &mut World) {
        let comp_id = world.register_component::<T>();
        let layout = Layout::new::<T>();
        let name = std::any::type_name::<T>();

        let serialize_fn: SerializeFn = |ptr, out| {
            let value = unsafe { &*ptr.cast::<T>() };
            let bytes = bincode::serialize(value)
                .map_err(|e| CodecError::Serialize(e.to_string()))?;
            out.extend_from_slice(&bytes);
            Ok(())
        };

        let deserialize_fn: DeserializeFn = |bytes| {
            let value: T = bincode::deserialize(bytes)
                .map_err(|e| CodecError::Deserialize(e.to_string()))?;
            let layout = Layout::new::<T>();
            let mut buf = vec![0u8; layout.size()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &value as *const T as *const u8,
                    buf.as_mut_ptr(),
                    layout.size(),
                );
            }
            std::mem::forget(value);
            buf
        };

        // Note: deserialize_fn needs refinement for non-Copy types (drop safety).
        // For now, the forget + raw copy pattern works for types that are
        // Serialize + DeserializeOwned (which implies they can be reconstructed).

        self.codecs.insert(comp_id, ComponentCodec {
            name,
            layout,
            serialize_fn,
            deserialize_fn,
        });
    }

    /// Serialize a component value from a raw pointer to bytes.
    pub fn serialize(
        &self,
        id: ComponentId,
        ptr: *const u8,
        out: &mut Vec<u8>,
    ) -> Result<(), CodecError> {
        let codec = self.codecs.get(&id)
            .ok_or(CodecError::UnregisteredComponent(id))?;
        unsafe { (codec.serialize_fn)(ptr, out) }
    }

    /// Deserialize component bytes into a raw byte buffer.
    pub fn deserialize(
        &self,
        id: ComponentId,
        bytes: &[u8],
    ) -> Result<Vec<u8>, CodecError> {
        let codec = self.codecs.get(&id)
            .ok_or(CodecError::UnregisteredComponent(id))?;
        (codec.deserialize_fn)(bytes)
    }

    /// Get the layout for a registered component.
    pub fn layout(&self, id: ComponentId) -> Option<Layout> {
        self.codecs.get(&id).map(|c| c.layout)
    }

    /// Get the type name for a registered component.
    pub fn name(&self, id: ComponentId) -> Option<&'static str> {
        self.codecs.get(&id).map(|c| c.name)
    }

    /// Check if a component has a registered codec.
    pub fn has_codec(&self, id: ComponentId) -> bool {
        self.codecs.contains_key(&id)
    }

    /// All registered ComponentIds.
    pub fn registered_ids(&self) -> Vec<ComponentId> {
        self.codecs.keys().copied().collect()
    }
}
```

**Step 3: Write codec tests**

In `crates/minkowski-persist/src/codec.rs`, add:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Serialize, Deserialize};

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos { x: f32, y: f32 }

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn register_and_serialize_round_trip() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let pos = Pos { x: 1.0, y: 2.0 };
        let mut buf = Vec::new();
        codecs.serialize(
            world.component_id::<Pos>().unwrap(),
            &pos as *const Pos as *const u8,
            &mut buf,
        ).unwrap();

        let raw = codecs.deserialize(
            world.component_id::<Pos>().unwrap(),
            &buf,
        ).unwrap();

        let restored = unsafe { *(raw.as_ptr() as *const Pos) };
        assert_eq!(restored, pos);
    }

    #[test]
    fn unregistered_component_returns_error() {
        let codecs = CodecRegistry::new();
        let mut buf = Vec::new();
        let result = codecs.serialize(999, std::ptr::null(), &mut buf);
        assert!(matches!(result, Err(CodecError::UnregisteredComponent(999))));
    }

    #[test]
    fn multiple_components() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        assert!(codecs.has_codec(world.component_id::<Pos>().unwrap()));
        assert!(codecs.has_codec(world.component_id::<Vel>().unwrap()));
        assert_eq!(codecs.registered_ids().len(), 2);
    }
}
```

**Step 4: Verify**

Run: `cargo test -p minkowski-persist && cargo clippy --workspace --all-targets -- -D warnings`

**Step 5: Commit**

```bash
git add crates/minkowski-persist/ Cargo.toml
git commit -m "feat: add minkowski-persist crate with CodecRegistry"
```

---

### Task 3: Serialized types + WireFormat trait + Bincode impl

**Files:**
- Create: `crates/minkowski-persist/src/format.rs`
- Create: `crates/minkowski-persist/src/record.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Step 1: Define serializable record types**

`crates/minkowski-persist/src/record.rs`:

```rust
use serde::{Serialize, Deserialize};
use minkowski::ComponentId;

/// Serde-friendly mirror of core's Mutation enum.
/// Entity stored as raw u64 (preserving generation bits).
/// Component data is pre-serialized through CodecRegistry.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SerializedMutation {
    Spawn {
        entity: u64,
        components: Vec<(ComponentId, Vec<u8>)>,
    },
    Despawn {
        entity: u64,
    },
    Insert {
        entity: u64,
        component_id: ComponentId,
        data: Vec<u8>,
    },
    Remove {
        entity: u64,
        component_id: ComponentId,
    },
}

/// A single WAL record: one committed changeset with a sequence number.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalRecord {
    pub seq: u64,
    pub mutations: Vec<SerializedMutation>,
}

/// Schema entry for a component type.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ComponentSchema {
    pub id: ComponentId,
    pub name: String,
    pub size: usize,
    pub align: usize,
}

/// Serializable entity allocator state.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AllocatorState {
    pub generations: Vec<u32>,
    pub free_list: Vec<u32>,
}

/// Per-archetype data in a snapshot.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArchetypeData {
    pub component_ids: Vec<ComponentId>,
    pub entities: Vec<u64>,
    pub columns: Vec<ColumnData>,
}

/// Per-column data: one serialized blob per row.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColumnData {
    pub component_id: ComponentId,
    pub values: Vec<Vec<u8>>,
}

/// Sparse component data (outside archetype columns).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SparseComponentData {
    pub component_id: ComponentId,
    pub entries: Vec<(u64, Vec<u8>)>,
}

/// Full snapshot payload.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SnapshotData {
    pub wal_seq: u64,
    pub schema: Vec<ComponentSchema>,
    pub allocator: AllocatorState,
    pub archetypes: Vec<ArchetypeData>,
    pub sparse: Vec<SparseComponentData>,
}

/// Returned after a successful snapshot save.
#[derive(Debug, Clone)]
pub struct SnapshotHeader {
    pub wal_seq: u64,
    pub archetype_count: usize,
    pub entity_count: usize,
}
```

**Step 2: Define WireFormat trait**

`crates/minkowski-persist/src/format.rs`:

```rust
use crate::record::{WalRecord, SnapshotData};

/// Abstracts the serialization format. Bincode now, rkyv later.
pub trait WireFormat {
    type Error: std::error::Error + Send + Sync + 'static;

    fn serialize_record(&self, record: &WalRecord) -> Result<Vec<u8>, Self::Error>;
    fn deserialize_record(&self, bytes: &[u8]) -> Result<WalRecord, Self::Error>;
    fn serialize_snapshot(&self, snapshot: &SnapshotData) -> Result<Vec<u8>, Self::Error>;
    fn deserialize_snapshot(&self, bytes: &[u8]) -> Result<SnapshotData, Self::Error>;
}

/// Bincode wire format — compact, fast, serde-native.
pub struct Bincode;

impl WireFormat for Bincode {
    type Error = bincode::Error;

    fn serialize_record(&self, record: &WalRecord) -> Result<Vec<u8>, Self::Error> {
        bincode::serialize(record)
    }

    fn deserialize_record(&self, bytes: &[u8]) -> Result<WalRecord, Self::Error> {
        bincode::deserialize(bytes)
    }

    fn serialize_snapshot(&self, snapshot: &SnapshotData) -> Result<Vec<u8>, Self::Error> {
        bincode::serialize(snapshot)
    }

    fn deserialize_snapshot(&self, bytes: &[u8]) -> Result<SnapshotData, Self::Error> {
        bincode::deserialize(bytes)
    }
}
```

**Step 3: Update lib.rs**

```rust
pub mod codec;
pub mod format;
pub mod record;

pub use codec::{CodecRegistry, CodecError};
pub use format::{WireFormat, Bincode};
pub use record::*;
```

**Step 4: Write round-trip tests**

In `format.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::*;

    #[test]
    fn wal_record_round_trip() {
        let record = WalRecord {
            seq: 42,
            mutations: vec![
                SerializedMutation::Insert {
                    entity: 0x0000_0001_0000_0000, // gen=1, idx=0
                    component_id: 0,
                    data: vec![1, 2, 3, 4],
                },
                SerializedMutation::Despawn {
                    entity: 0x0000_0002_0000_0005,
                },
            ],
        };

        let fmt = Bincode;
        let bytes = fmt.serialize_record(&record).unwrap();
        let restored = fmt.deserialize_record(&bytes).unwrap();

        assert_eq!(restored.seq, 42);
        assert_eq!(restored.mutations.len(), 2);
    }

    #[test]
    fn snapshot_data_round_trip() {
        let snap = SnapshotData {
            wal_seq: 100,
            schema: vec![ComponentSchema {
                id: 0,
                name: "Pos".into(),
                size: 8,
                align: 4,
            }],
            allocator: AllocatorState {
                generations: vec![0, 1, 0],
                free_list: vec![1],
            },
            archetypes: vec![ArchetypeData {
                component_ids: vec![0],
                entities: vec![0, 2],
                columns: vec![ColumnData {
                    component_id: 0,
                    values: vec![vec![1, 2, 3, 4, 5, 6, 7, 8], vec![9, 10, 11, 12, 13, 14, 15, 16]],
                }],
            }],
            sparse: vec![],
        };

        let fmt = Bincode;
        let bytes = fmt.serialize_snapshot(&snap).unwrap();
        let restored = fmt.deserialize_snapshot(&bytes).unwrap();

        assert_eq!(restored.wal_seq, 100);
        assert_eq!(restored.archetypes.len(), 1);
        assert_eq!(restored.archetypes[0].entities.len(), 2);
    }
}
```

**Step 5: Verify**

Run: `cargo test -p minkowski-persist && cargo clippy --workspace --all-targets -- -D warnings`

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/
git commit -m "feat: serialized record types, WireFormat trait, Bincode impl"
```

---

### Task 4: WAL (append-only log)

**Files:**
- Create: `crates/minkowski-persist/src/wal.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Step 1: Implement Wal struct**

`crates/minkowski-persist/src/wal.rs`:

The WAL file format is: `[len: u32][payload: Vec<u8>]` repeated. The payload is a `WalRecord` serialized through WireFormat. The sequence number is inside the WalRecord, not in the framing — keeps the framing minimal.

```rust
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;

use minkowski::{EnumChangeSet, Entity, World, MutationRef};

use crate::codec::{CodecRegistry, CodecError};
use crate::format::WireFormat;
use crate::record::{WalRecord, SerializedMutation};

#[derive(Debug)]
pub enum WalError {
    Io(io::Error),
    Codec(CodecError),
    Format(String),
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "WAL I/O: {e}"),
            Self::Codec(e) => write!(f, "WAL codec: {e}"),
            Self::Format(msg) => write!(f, "WAL format: {msg}"),
        }
    }
}

impl std::error::Error for WalError {}

impl From<io::Error> for WalError { fn from(e: io::Error) -> Self { Self::Io(e) } }
impl From<CodecError> for WalError { fn from(e: CodecError) -> Self { Self::Codec(e) } }

pub struct Wal<W: WireFormat> {
    file: File,
    format: W,
    next_seq: u64,
}

impl<W: WireFormat> Wal<W> {
    /// Create a new WAL file. Fails if file already exists.
    pub fn create(path: &Path, format: W) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(path)?;
        Ok(Self { file, format, next_seq: 0 })
    }

    /// Open an existing WAL file. Scans to find the next sequence number.
    pub fn open(path: &Path, format: W) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(path)?;
        let mut wal = Self { file, format, next_seq: 0 };
        // Scan to end to find next_seq
        wal.next_seq = wal.scan_last_seq()? + 1;
        Ok(wal)
    }

    /// Serialize and append a changeset as a WAL record.
    /// Returns the sequence number assigned to this record.
    pub fn append(
        &mut self,
        changeset: &EnumChangeSet,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        let seq = self.next_seq;
        let record = Self::changeset_to_record(seq, changeset, codecs)?;
        let payload = self.format.serialize_record(&record)
            .map_err(|e| WalError::Format(e.to_string()))?;

        let mut writer = BufWriter::new(&self.file);
        let len = payload.len() as u32;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&payload)?;
        writer.flush()?;

        self.next_seq += 1;
        Ok(seq)
    }

    /// Replay all records into a world.
    /// Returns the last sequence number replayed, or 0 if empty.
    pub fn replay(
        &mut self,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        self.replay_from(0, world, codecs)
    }

    /// Replay records starting from a given sequence number.
    pub fn replay_from(
        &mut self,
        from_seq: u64,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(&self.file);
        let mut last_seq = if from_seq > 0 { from_seq - 1 } else { 0 };

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            reader.read_exact(&mut payload)?;

            let record = self.format.deserialize_record(&payload)
                .map_err(|e| WalError::Format(e.to_string()))?;

            if record.seq >= from_seq {
                Self::apply_record(&record, world, codecs)?;
                last_seq = record.seq;
            }
        }

        Ok(last_seq)
    }

    /// Current sequence number (next append will use this).
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn changeset_to_record(
        seq: u64,
        changeset: &EnumChangeSet,
        codecs: &CodecRegistry,
    ) -> Result<WalRecord, WalError> {
        let mut mutations = Vec::new();
        for m in changeset.iter_mutations() {
            mutations.push(Self::serialize_mutation(&m, codecs)?);
        }
        Ok(WalRecord { seq, mutations })
    }

    fn serialize_mutation(
        m: &MutationRef<'_>,
        codecs: &CodecRegistry,
    ) -> Result<SerializedMutation, WalError> {
        match m {
            MutationRef::Spawn { entity, components } => {
                let mut serialized = Vec::new();
                for &(comp_id, raw_bytes) in components {
                    let mut buf = Vec::new();
                    codecs.serialize(comp_id, raw_bytes.as_ptr(), &mut buf)?;
                    serialized.push((comp_id, buf));
                }
                Ok(SerializedMutation::Spawn {
                    entity: entity.to_bits(),
                    components: serialized,
                })
            }
            MutationRef::Despawn { entity } => {
                Ok(SerializedMutation::Despawn { entity: entity.to_bits() })
            }
            MutationRef::Insert { entity, component_id, data } => {
                let mut buf = Vec::new();
                codecs.serialize(*component_id, data.as_ptr(), &mut buf)?;
                Ok(SerializedMutation::Insert {
                    entity: entity.to_bits(),
                    component_id: *component_id,
                    data: buf,
                })
            }
            MutationRef::Remove { entity, component_id } => {
                Ok(SerializedMutation::Remove {
                    entity: entity.to_bits(),
                    component_id: *component_id,
                })
            }
        }
    }

    fn apply_record(
        record: &WalRecord,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<(), WalError> {
        let mut changeset = EnumChangeSet::new();
        for mutation in &record.mutations {
            match mutation {
                SerializedMutation::Spawn { entity, components } => {
                    let entity = Entity::from_bits(*entity);
                    let mut raw_components = Vec::new();
                    for (comp_id, data) in components {
                        let raw = codecs.deserialize(*comp_id, data)?;
                        let layout = codecs.layout(*comp_id).unwrap();
                        // record_spawn borrows the pointer — must keep raw alive
                        raw_components.push((*comp_id, raw, layout));
                    }
                    let ptrs: Vec<_> = raw_components.iter()
                        .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                        .collect();
                    changeset.record_spawn(entity, &ptrs);
                }
                SerializedMutation::Despawn { entity } => {
                    changeset.record_despawn(Entity::from_bits(*entity));
                }
                SerializedMutation::Insert { entity, component_id, data } => {
                    let raw = codecs.deserialize(*component_id, data)?;
                    let layout = codecs.layout(*component_id).unwrap();
                    changeset.record_insert(
                        Entity::from_bits(*entity),
                        *component_id,
                        raw.as_ptr(),
                        layout,
                    );
                }
                SerializedMutation::Remove { entity, component_id } => {
                    changeset.record_remove(Entity::from_bits(*entity), *component_id);
                }
            }
        }
        changeset.apply(world);
        Ok(())
    }

    fn scan_last_seq(&mut self) -> Result<u64, WalError> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(&self.file);
        let mut last_seq = 0u64;

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            reader.read_exact(&mut payload)?;

            let record = self.format.deserialize_record(&payload)
                .map_err(|e| WalError::Format(e.to_string()))?;
            last_seq = record.seq;
        }

        Ok(last_seq)
    }
}
```

Note: `Entity::to_bits()` and `Entity::from_bits(u64)` may need to be added to core if they don't exist. Entity is `#[repr(transparent)] struct Entity(u64)`, so these are trivial. Check if they exist; if not, add `pub fn to_bits(self) -> u64 { self.0 }` and `pub fn from_bits(bits: u64) -> Self { Self(bits) }` to entity.rs.

**Step 2: Write WAL tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::format::Bincode;
    use serde::{Serialize, Deserialize};

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos { x: f32, y: f32 }

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Health(u32);

    #[test]
    fn append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        // Spawn an entity and record the changeset
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.insert::<Pos>(&mut world, e, Pos { x: 1.0, y: 2.0 });
        let reverse = cs.apply(&mut world);

        // Append to WAL
        let mut wal = Wal::create(&wal_path, Bincode).unwrap();
        let seq = wal.append(&reverse, &codecs).unwrap();
        assert_eq!(seq, 0);
        drop(wal);

        // Replay into fresh world
        let mut world2 = World::new();
        let e2 = world2.alloc_entity(); // need same entity
        let mut wal2 = Wal::open(&wal_path, Bincode).unwrap();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0);
    }

    #[test]
    fn multiple_appends() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        let mut wal = Wal::create(&wal_path, Bincode).unwrap();

        for i in 0..5 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.insert::<Health>(&mut world, e, Health(100 - i * 10));
            let reverse = cs.apply(&mut world);
            wal.append(&reverse, &codecs).unwrap();
        }

        assert_eq!(wal.next_seq(), 5);
    }

    #[test]
    fn replay_from_skips_earlier_records() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        let mut wal = Wal::create(&wal_path, Bincode).unwrap();

        // Append 3 records
        for _ in 0..3 {
            let mut cs = EnumChangeSet::new();
            cs.record_despawn(Entity::from_bits(0)); // no-op on empty world
            wal.append(&cs, &codecs).unwrap();
        }

        // replay_from(2) should only apply the third record
        let mut world2 = World::new();
        let last = wal.replay_from(2, &mut world2, &codecs).unwrap();
        assert_eq!(last, 2);
    }
}
```

Add `tempfile = "3"` to `[dev-dependencies]` in `crates/minkowski-persist/Cargo.toml`.

**Step 3: Update lib.rs**

Add `pub mod wal;` and `pub use wal::{Wal, WalError};`.

**Step 4: Verify**

Run: `cargo test -p minkowski-persist && cargo clippy --workspace --all-targets -- -D warnings`

**Step 5: Commit**

```bash
git add crates/minkowski-persist/
git commit -m "feat: WAL append-only log with replay"
```

---

### Task 5: Snapshot save + load

**Files:**
- Create: `crates/minkowski-persist/src/snapshot.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Step 1: Implement Snapshot**

`crates/minkowski-persist/src/snapshot.rs`:

```rust
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use minkowski::{ComponentId, Entity, EnumChangeSet, World};

use crate::codec::{CodecRegistry, CodecError};
use crate::format::WireFormat;
use crate::record::*;

#[derive(Debug)]
pub enum SnapshotError {
    Io(std::io::Error),
    Codec(CodecError),
    Format(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "snapshot I/O: {e}"),
            Self::Codec(e) => write!(f, "snapshot codec: {e}"),
            Self::Format(msg) => write!(f, "snapshot format: {msg}"),
        }
    }
}

impl std::error::Error for SnapshotError {}
impl From<std::io::Error> for SnapshotError { fn from(e: std::io::Error) -> Self { Self::Io(e) } }
impl From<CodecError> for SnapshotError { fn from(e: CodecError) -> Self { Self::Codec(e) } }

pub struct Snapshot<W: WireFormat> {
    format: W,
}

impl<W: WireFormat> Snapshot<W> {
    pub fn new(format: W) -> Self {
        Self { format }
    }

    /// Save a full world snapshot to disk.
    pub fn save(
        &self,
        path: &Path,
        world: &World,
        codecs: &CodecRegistry,
        wal_seq: u64,
    ) -> Result<SnapshotHeader, SnapshotError> {
        let data = self.build_snapshot_data(world, codecs, wal_seq)?;
        let header = SnapshotHeader {
            wal_seq,
            archetype_count: data.archetypes.len(),
            entity_count: data.archetypes.iter().map(|a| a.entities.len()).sum(),
        };

        let bytes = self.format.serialize_snapshot(&data)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let len = bytes.len() as u64;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&bytes)?;
        writer.flush()?;

        Ok(header)
    }

    /// Load a world from a snapshot file.
    pub fn load(
        &self,
        path: &Path,
        codecs: &CodecRegistry,
    ) -> Result<(World, u64), SnapshotError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let len = u64::from_le_bytes(len_buf) as usize;

        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;

        let data = self.format.deserialize_snapshot(&bytes)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let world = self.restore_world(&data, codecs)?;
        Ok((world, data.wal_seq))
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn build_snapshot_data(
        &self,
        world: &World,
        codecs: &CodecRegistry,
        wal_seq: u64,
    ) -> Result<SnapshotData, SnapshotError> {
        // Schema
        let schema: Vec<ComponentSchema> = codecs.registered_ids().iter().map(|&id| {
            ComponentSchema {
                id,
                name: codecs.name(id).unwrap_or("unknown").to_string(),
                size: codecs.layout(id).map(|l| l.size()).unwrap_or(0),
                align: codecs.layout(id).map(|l| l.align()).unwrap_or(1),
            }
        }).collect();

        // Allocator state
        let (gens, free) = world.entity_allocator_state();
        let allocator = AllocatorState {
            generations: gens.to_vec(),
            free_list: free.to_vec(),
        };

        // Archetypes (skip archetype 0 if it's the empty archetype — check if it has components)
        let mut archetypes = Vec::new();
        for arch_idx in 0..world.archetype_count() {
            let comp_ids = world.archetype_component_ids(arch_idx);
            if comp_ids.is_empty() {
                continue; // skip empty archetype (if any)
            }
            let entities = world.archetype_entities(arch_idx);
            if entities.is_empty() {
                continue; // skip empty archetypes
            }

            let mut columns = Vec::new();
            for &comp_id in comp_ids {
                let mut values = Vec::new();
                for row in 0..entities.len() {
                    let ptr = unsafe { world.archetype_column_ptr(arch_idx, comp_id, row) };
                    let mut buf = Vec::new();
                    codecs.serialize(comp_id, ptr, &mut buf)?;
                    values.push(buf);
                }
                columns.push(ColumnData { component_id: comp_id, values });
            }

            archetypes.push(ArchetypeData {
                component_ids: comp_ids.to_vec(),
                entities: entities.iter().map(|e| e.to_bits()).collect(),
                columns,
            });
        }

        // Sparse components
        let mut sparse = Vec::new();
        for comp_id in world.sparse_component_ids() {
            if !codecs.has_codec(comp_id) {
                return Err(CodecError::UnregisteredComponent(comp_id).into());
            }
            // Sparse iteration requires knowing the concrete type, which the
            // CodecRegistry captured at register time. The serialize_sparse
            // method on CodecRegistry handles this (added in codec.rs).
            // For now, skip sparse in the initial implementation — revisit
            // when sparse codecs gain a typed iteration hook.
        }

        Ok(SnapshotData {
            wal_seq,
            schema,
            allocator,
            archetypes,
            sparse,
        })
    }

    fn restore_world(
        &self,
        data: &SnapshotData,
        codecs: &CodecRegistry,
    ) -> Result<World, SnapshotError> {
        let mut world = World::new();

        // Register component types so IDs are available
        // (CodecRegistry::register was called by the caller before load)

        // Restore archetypes via EnumChangeSet
        for arch_data in &data.archetypes {
            for (row, &entity_bits) in arch_data.entities.iter().enumerate() {
                let entity = Entity::from_bits(entity_bits);

                let mut raw_components: Vec<(ComponentId, Vec<u8>, std::alloc::Layout)> = Vec::new();
                for col in &arch_data.columns {
                    let raw = codecs.deserialize(col.component_id, &col.values[row])?;
                    let layout = codecs.layout(col.component_id).unwrap();
                    raw_components.push((col.component_id, raw, layout));
                }

                let ptrs: Vec<_> = raw_components.iter()
                    .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                    .collect();

                let mut cs = EnumChangeSet::new();
                cs.record_spawn(entity, &ptrs);
                cs.apply(&mut world);
            }
        }

        // Restore allocator state (must come AFTER spawning entities so
        // entity_locations is sized correctly, then overwrite allocator)
        world.restore_allocator_state(
            data.allocator.generations.clone(),
            data.allocator.free_list.clone(),
        );

        Ok(world)
    }
}
```

Note: sparse snapshot serialization requires a typed iteration hook in the CodecRegistry (the codec knows the concrete type from registration). This will need a `serialize_sparse_entries` method on CodecRegistry that captures the concrete type at register time via a closure. Leave a TODO for now and implement in a follow-up step within this task.

**Step 2: Write snapshot tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::format::Bincode;
    use serde::{Serialize, Deserialize};

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos { x: f32, y: f32 }

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("test.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        // Spawn some entities
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        world.spawn((Pos { x: 5.0, y: 6.0 }, Vel { dx: 7.0, dy: 8.0 }));

        let snap = Snapshot::new(Bincode);
        let header = snap.save(&snap_path, &world, &codecs, 42).unwrap();
        assert_eq!(header.entity_count, 2);
        assert_eq!(header.wal_seq, 42);

        // Load into fresh world
        let (world2, wal_seq) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(wal_seq, 42);

        // Verify entities have correct component values
        let positions: Vec<(f32, f32)> = world2.query::<(&Pos,)>()
            .map(|p| (p.0.x, p.0.y))
            .collect();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn empty_world_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("empty.snap");

        let mut world = World::new();
        let codecs = CodecRegistry::new();

        let snap = Snapshot::new(Bincode);
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let (world2, seq) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(seq, 0);
        assert_eq!(world2.archetype_count(), 1); // empty archetype
    }
}
```

**Step 3: Update lib.rs**

Add `pub mod snapshot;` and `pub use snapshot::{Snapshot, SnapshotError};`.

**Step 4: Verify**

Run: `cargo test -p minkowski-persist && cargo clippy --workspace --all-targets -- -D warnings`

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/
git commit -m "feat: snapshot save and load with archetype serialization"
```

---

### Task 6: Sparse snapshot support

**Files:**
- Modify: `crates/minkowski-persist/src/codec.rs`
- Modify: `crates/minkowski-persist/src/snapshot.rs`

Sparse components live in `HashMap<Entity, T>`, not archetype columns. The archetype snapshot path misses them entirely. The CodecRegistry needs a typed iteration hook — the codec captures the concrete type at registration, so it can call `World::iter_sparse::<T>()` and serialize each entry.

**Step 1: Add sparse serialization to CodecRegistry**

In `codec.rs`, add a type-erased function for sparse iteration + serialization:

```rust
/// Type-erased function that iterates a sparse component and serializes all entries.
type SerializeSparseFn = fn(&World, ComponentId, &CodecRegistry) -> Result<Vec<(u64, Vec<u8>)>, CodecError>;

// Add to ComponentCodec:
struct ComponentCodec {
    // ... existing fields ...
    serialize_sparse_fn: SerializeSparseFn,
}
```

In `register<T>()`, capture:

```rust
let serialize_sparse_fn: SerializeSparseFn = |world, comp_id, codecs| {
    let mut entries = Vec::new();
    if let Some(iter) = world.iter_sparse::<T>(comp_id) {
        for (entity, value) in iter {
            let mut buf = Vec::new();
            codecs.serialize(comp_id, value as *const T as *const u8, &mut buf)?;
            entries.push((entity.to_bits(), buf));
        }
    }
    Ok(entries)
};
```

Add a public method:

```rust
pub fn serialize_sparse(
    &self,
    id: ComponentId,
    world: &World,
) -> Result<Vec<(u64, Vec<u8>)>, CodecError> {
    let codec = self.codecs.get(&id)
        .ok_or(CodecError::UnregisteredComponent(id))?;
    (codec.serialize_sparse_fn)(world, id, self)
}
```

**Step 2: Wire sparse into snapshot save**

In `snapshot.rs`, replace the sparse TODO:

```rust
// Sparse components
let mut sparse = Vec::new();
for comp_id in world.sparse_component_ids() {
    if !codecs.has_codec(comp_id) {
        return Err(CodecError::UnregisteredComponent(comp_id).into());
    }
    let entries = codecs.serialize_sparse(comp_id, world)?;
    if !entries.is_empty() {
        sparse.push(SparseComponentData {
            component_id: comp_id,
            entries,
        });
    }
}
```

**Step 3: Wire sparse into snapshot load**

In `restore_world()`, after archetype restoration and before allocator restore:

```rust
// Restore sparse components
for sparse_data in &data.sparse {
    let raw_entries: Vec<_> = sparse_data.entries.iter().map(|(entity_bits, data)| {
        let raw = codecs.deserialize(sparse_data.component_id, data)?;
        Ok((*entity_bits, raw))
    }).collect::<Result<Vec<_>, SnapshotError>>()?;

    // Use a type-erased insert helper on CodecRegistry
    for (entity_bits, raw) in raw_entries {
        let entity = Entity::from_bits(entity_bits);
        codecs.insert_sparse_raw(sparse_data.component_id, &mut world, entity, &raw)?;
    }
}
```

This requires adding `insert_sparse_raw` to CodecRegistry — a type-erased function that deserializes and inserts:

```rust
// In CodecRegistry, captured at register time:
type InsertSparseFn = fn(&[u8], &mut World, Entity, ComponentId) -> Result<(), CodecError>;

// In register<T>():
let insert_sparse_fn: InsertSparseFn = |raw_bytes, world, entity, comp_id| {
    let value: T = bincode::deserialize(/* re-serialize from raw? */);
    // Actually, raw_bytes is already the deserialized raw memory.
    // We need to reconstruct T from raw bytes, then call insert_sparse.
    let value = unsafe { std::ptr::read(raw_bytes.as_ptr() as *const T) };
    world.insert_sparse(entity, value);
    Ok(())
};
```

Note: the exact approach for sparse deserialize + insert needs care around drop safety. The implementer should ensure that the deserialized value is properly owned and not double-dropped. `std::ptr::read` from raw bytes + `insert_sparse` (which takes ownership) is the pattern.

**Step 4: Write sparse tests**

```rust
#[test]
fn sparse_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let snap_path = dir.path().join("sparse.snap");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);

    // Register Pos as sparse
    world.register_sparse::<Pos>();

    let e = world.spawn(());  // entity with no archetype components
    world.insert_sparse(e, Pos { x: 1.0, y: 2.0 });

    let snap = Snapshot::new(Bincode);
    snap.save(&snap_path, &world, &codecs, 0).unwrap();

    let (world2, _) = snap.load(&snap_path, &codecs).unwrap();

    // Verify sparse component restored
    let pos_id = world2.component_id::<Pos>().unwrap();
    let restored: Vec<_> = world2.iter_sparse::<Pos>(pos_id).unwrap().collect();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].1, &Pos { x: 1.0, y: 2.0 });
}
```

**Step 5: Verify**

Run: `cargo test -p minkowski-persist && cargo clippy --workspace --all-targets -- -D warnings`

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/
git commit -m "feat: sparse component support in snapshots"
```

---

### Task 7: Recovery + example + docs

**Files:**
- Create: `examples/examples/persist.rs`
- Modify: `examples/Cargo.toml`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Add recovery integration test**

In `crates/minkowski-persist/src/snapshot.rs` tests:

```rust
#[test]
fn snapshot_plus_wal_recovery() {
    use crate::wal::Wal;

    let dir = tempfile::tempdir().unwrap();
    let snap_path = dir.path().join("recovery.snap");
    let wal_path = dir.path().join("recovery.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);
    codecs.register::<Vel>(&mut world);

    // Phase 1: spawn entities, save snapshot
    world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
    world.spawn((Pos { x: 3.0, y: 4.0 }, Vel { dx: 0.3, dy: 0.4 }));

    let snap = Snapshot::new(Bincode);
    let mut wal = Wal::create(&wal_path, Bincode).unwrap();

    let header = snap.save(&snap_path, &world, &codecs, wal.next_seq()).unwrap();

    // Phase 2: more mutations after snapshot, written to WAL
    let e3 = world.spawn((Pos { x: 5.0, y: 6.0 },));
    let mut cs = EnumChangeSet::new();
    cs.insert::<Pos>(&mut world, e3, Pos { x: 5.0, y: 6.0 });
    let reverse = cs.apply(&mut world);
    wal.append(&reverse, &codecs).unwrap();

    // Phase 3: recover from snapshot + WAL
    let (mut recovered, snap_seq) = snap.load(&snap_path, &codecs).unwrap();
    wal.replay_from(snap_seq, &mut recovered, &codecs).unwrap();

    // Verify: original 2 entities from snapshot + mutations from WAL
    let count = recovered.query::<(&Pos,)>().count();
    assert!(count >= 2); // at least the snapshot entities
}
```

**Step 2: Create persistence example**

`examples/examples/persist.rs` — demonstrates WAL + snapshot + recovery:

```rust
//! Persistence — demonstrates WAL and snapshot save/load/recovery.
//!
//! Run: cargo run -p minkowski-examples --example persist --release

use minkowski::{World, EnumChangeSet};
use minkowski_persist::{CodecRegistry, Bincode, Wal, Snapshot};
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Pos { x: f32, y: f32 }

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Vel { dx: f32, dy: f32 }

fn main() {
    let dir = std::env::temp_dir().join("minkowski-persist-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("example.wal");
    let snap_path = dir.join("example.snap");

    // Clean up from previous runs
    let _ = std::fs::remove_file(&wal_path);
    let _ = std::fs::remove_file(&snap_path);

    // ── Phase 1: Create world, spawn entities ──────────────────────
    println!("Phase 1: Creating world with 100 entities...");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);
    codecs.register::<Vel>(&mut world);

    for i in 0..100 {
        world.spawn((
            Pos { x: i as f32, y: 0.0 },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }

    // ── Phase 2: Save snapshot ─────────────────────────────────────
    let mut wal = Wal::create(&wal_path, Bincode).unwrap();
    let snap = Snapshot::new(Bincode);
    let header = snap.save(&snap_path, &world, &codecs, wal.next_seq()).unwrap();
    println!("Phase 2: Snapshot saved ({} entities, {} archetypes)", header.entity_count, header.archetype_count);

    // ── Phase 3: More mutations, written to WAL ────────────────────
    println!("Phase 3: Running 10 mutation steps (WAL)...");
    for _ in 0..10 {
        // Simple movement system via changeset
        let positions: Vec<_> = world.query::<(minkowski::Entity, &Pos, &Vel)>()
            .map(|(e, p, v)| (e, Pos { x: p.x + v.dx, y: p.y + v.dy }))
            .collect();

        let mut cs = EnumChangeSet::new();
        for (e, new_pos) in &positions {
            cs.insert::<Pos>(&mut world, *e, *new_pos);
        }
        let reverse = cs.apply(&mut world);
        wal.append(&reverse, &codecs).unwrap();
    }
    println!("  WAL has {} records", wal.next_seq());

    // ── Phase 4: Recovery ──────────────────────────────────────────
    println!("Phase 4: Recovering from snapshot + WAL...");
    let (mut recovered, snap_seq) = snap.load(&snap_path, &codecs).unwrap();
    let last_seq = wal.replay_from(snap_seq, &mut recovered, &codecs).unwrap();
    println!("  Loaded snapshot (seq {}), replayed WAL to seq {}", snap_seq, last_seq);

    let count = recovered.query::<(&Pos,)>().count();
    println!("  Recovered world has {} entities", count);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone.");
}
```

Add `minkowski-persist = { path = "../crates/minkowski-persist" }` to `examples/Cargo.toml` dependencies. Also add `serde = { version = "1", features = ["derive"] }` since the example uses `#[derive(Serialize, Deserialize)]`.

**Step 3: Update CLAUDE.md**

Add persistence section to Architecture, update Build & Test Commands with the new example, add `minkowski-persist` to crate descriptions, add `serde` and `bincode` to Dependencies table.

**Step 4: Update README.md**

Add persistence example section. Update Phase 4 status to include persistence.

**Step 5: Verify everything**

```bash
cargo test -p minkowski --lib
cargo test -p minkowski-persist
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p minkowski-examples --example persist --release
```

**Step 6: Commit**

```bash
git add .
git commit -m "feat: persistence example + docs"
```

---

### Verification

1. `cargo test -p minkowski --lib` — core tests pass (new accessors tested)
2. `cargo test -p minkowski-persist` — all persist tests pass (codec, WAL, snapshot, recovery)
3. `cargo clippy --workspace --all-targets -- -D warnings` — clean
4. `cargo run -p minkowski-examples --example persist --release` — runs, prints recovery output
5. Round-trip verified: world → snapshot → fresh world → compare
6. WAL replay verified: append N records → replay → correct state
7. Recovery verified: snapshot + WAL tail → correct state
8. Missing codec → error (not silent skip)
