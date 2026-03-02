use serde::{Deserialize, Serialize};

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

/// Schema entry for a component type in a snapshot.
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
