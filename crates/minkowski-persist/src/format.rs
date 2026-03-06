use crate::record::{SnapshotData, WalRecord};

#[derive(Debug)]
pub struct FormatError(pub String);

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "format: {}", self.0)
    }
}

impl std::error::Error for FormatError {}

pub fn serialize_record(record: &WalRecord) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(record)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn deserialize_record(bytes: &[u8]) -> Result<WalRecord, FormatError> {
    rkyv::from_bytes::<WalRecord, rkyv::rancor::Error>(bytes)
        .map_err(|e| FormatError(e.to_string()))
}

pub fn serialize_snapshot(snapshot: &SnapshotData) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(snapshot)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn deserialize_snapshot(bytes: &[u8]) -> Result<SnapshotData, FormatError> {
    rkyv::from_bytes::<SnapshotData, rkyv::rancor::Error>(bytes)
        .map_err(|e| FormatError(e.to_string()))
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

        let bytes = serialize_record(&record).unwrap();
        let restored = deserialize_record(&bytes).unwrap();

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

        let bytes = serialize_snapshot(&snap).unwrap();
        let restored = deserialize_snapshot(&bytes).unwrap();

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
        let bytes = serialize_record(&record).unwrap();
        let restored = deserialize_record(&bytes).unwrap();
        assert_eq!(restored.seq, 0);
        assert!(restored.mutations.is_empty());
    }
}
