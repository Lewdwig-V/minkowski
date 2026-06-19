use rkyv::{Archive, Deserialize, Serialize};

use minkowski::ComponentId;

pub use minkowski_lsm::codec::ComponentSchema;

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
    Checkpoint { flush_seq: u64 },
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
        let checkpoint = WalEntry::Checkpoint { flush_seq: 42 };
        assert!(matches!(checkpoint, WalEntry::Checkpoint { flush_seq: 42 }));
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
