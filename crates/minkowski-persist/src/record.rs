use rkyv::{Archive, Deserialize, Serialize};

use minkowski::ComponentId;

/// rkyv-friendly mirror of core's Mutation enum.
/// Entity stored as raw u64 (preserving generation bits).
/// Component data is pre-serialized through CodecRegistry.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
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
    /// Insert a component into sparse storage (not archetypes).
    SparseInsert {
        entity: u64,
        component_id: ComponentId,
        data: Vec<u8>,
    },
    /// Remove a component from sparse storage.
    SparseRemove {
        entity: u64,
        component_id: ComponentId,
    },
}

/// A single WAL record: one committed changeset with a sequence number.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct WalRecord {
    pub seq: u64,
    pub mutations: Vec<SerializedMutation>,
}

/// Schema entry describing a component type. Used in both snapshot schemas
/// and WAL preambles. Fields are sender-local: `id` is meaningful only in
/// the originating World's ID space.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ComponentSchema {
    pub id: ComponentId,
    pub name: String,
    pub size: usize,
    pub align: usize,
}

/// Serializable entity allocator state.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct AllocatorState {
    pub generations: Vec<u32>,
    pub free_list: Vec<u32>,
}

/// Per-archetype data in a snapshot.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ArchetypeData {
    pub component_ids: Vec<ComponentId>,
    pub entities: Vec<u64>,
    pub columns: Vec<ColumnData>,
}

/// Per-column data: one serialized blob per row.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ColumnData {
    pub component_id: ComponentId,
    pub values: Vec<Vec<u8>>,
}

/// Sparse component data (outside archetype columns).
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct SparseComponentData {
    pub component_id: ComponentId,
    pub entries: Vec<(u64, Vec<u8>)>,
}

/// Full snapshot payload.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct SnapshotData {
    pub wal_seq: u64,
    pub schema: Vec<ComponentSchema>,
    pub allocator: AllocatorState,
    pub archetypes: Vec<ArchetypeData>,
    pub sparse: Vec<SparseComponentData>,
}

/// Schema preamble: maps sender-local IDs to stable names.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct WalSchema {
    pub components: Vec<ComponentSchema>,
}

/// A WAL file entry: either a schema preamble (first record) or mutation data.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub enum WalEntry {
    Schema(WalSchema),
    Mutations(WalRecord),
    Checkpoint { snapshot_seq: u64 },
}

/// Self-describing replication payload. Every batch carries its own schema
/// so receivers can decode without prior handshake.
///
/// Serialize with [`to_bytes`](ReplicationBatch::to_bytes) for transport
/// over any medium (network, channels, shared memory). Deserialize with
/// [`from_bytes`](ReplicationBatch::from_bytes) on the receiving end.
/// Apply to a target [`World`](minkowski::World) via
/// [`apply_batch`](crate::replication::apply_batch).
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ReplicationBatch {
    pub schema: WalSchema,
    pub records: Vec<WalRecord>,
}

/// Returned after a successful snapshot save.
#[derive(Debug, Clone)]
pub struct SnapshotHeader {
    pub wal_seq: u64,
    pub archetype_count: usize,
    pub entity_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_schema_clone() {
        let schema = ComponentSchema {
            id: 0,
            name: "pos".into(),
            size: 8,
            align: 4,
        };
        let cloned = schema.clone();
        assert_eq!(cloned.id, 0);
        assert_eq!(cloned.name, "pos");
    }

    #[test]
    fn wal_entry_checkpoint_variant() {
        let checkpoint = WalEntry::Checkpoint { snapshot_seq: 42 };
        assert!(matches!(
            checkpoint,
            WalEntry::Checkpoint { snapshot_seq: 42 }
        ));
    }

    #[test]
    fn wal_entry_variants() {
        let schema = WalEntry::Schema(WalSchema { components: vec![] });
        assert!(matches!(schema, WalEntry::Schema(_)));

        let mutations = WalEntry::Mutations(WalRecord {
            seq: 0,
            mutations: vec![],
        });
        assert!(matches!(mutations, WalEntry::Mutations(_)));
    }
}
