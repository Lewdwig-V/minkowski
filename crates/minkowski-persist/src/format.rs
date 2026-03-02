use crate::record::{SnapshotData, WalRecord};

/// Abstracts the serialization format. Bincode now, rkyv later as a second impl.
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
                    entity: 0x0000_0001_0000_0000,
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
                    values: vec![
                        vec![1, 2, 3, 4, 5, 6, 7, 8],
                        vec![9, 10, 11, 12, 13, 14, 15, 16],
                    ],
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

    #[test]
    fn empty_wal_record() {
        let record = WalRecord {
            seq: 0,
            mutations: vec![],
        };
        let fmt = Bincode;
        let bytes = fmt.serialize_record(&record).unwrap();
        let restored = fmt.deserialize_record(&bytes).unwrap();
        assert_eq!(restored.seq, 0);
        assert!(restored.mutations.is_empty());
    }
}
