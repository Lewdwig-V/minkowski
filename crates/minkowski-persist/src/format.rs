use crate::record::{SnapshotData, WalEntry};

#[derive(Debug, thiserror::Error)]
#[error("rkyv format error: {0}")]
pub struct FormatError(pub String);

pub fn serialize_snapshot(snapshot: &SnapshotData) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(snapshot)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn serialize_wal_entry(entry: &WalEntry) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(entry)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn deserialize_wal_entry(bytes: &[u8]) -> Result<WalEntry, FormatError> {
    rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(bytes).map_err(|e| FormatError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::*;

    #[test]
    fn wal_schema_round_trip() {
        let schema = WalSchema {
            components: vec![
                ComponentSchema {
                    id: 0,
                    name: "pos".into(),
                    size: 8,
                    align: 4,
                },
                ComponentSchema {
                    id: 1,
                    name: "vel".into(),
                    size: 8,
                    align: 4,
                },
            ],
        };
        let entry = WalEntry::Schema(schema);
        let bytes = serialize_wal_entry(&entry).unwrap();
        let restored = deserialize_wal_entry(&bytes).unwrap();
        match restored {
            WalEntry::Schema(s) => {
                assert_eq!(s.components.len(), 2);
                assert_eq!(s.components[0].name, "pos");
                assert_eq!(s.components[1].id, 1);
            }
            _ => panic!("expected Schema"),
        }
    }

    #[test]
    fn wal_entry_mutations_round_trip() {
        let record = WalRecord {
            seq: 7,
            mutations: vec![SerializedMutation::Despawn { entity: 42 }],
        };
        let entry = WalEntry::Mutations(record);
        let bytes = serialize_wal_entry(&entry).unwrap();
        let restored = deserialize_wal_entry(&bytes).unwrap();
        match restored {
            WalEntry::Mutations(r) => {
                assert_eq!(r.seq, 7);
                assert_eq!(r.mutations.len(), 1);
            }
            _ => panic!("expected Mutations"),
        }
    }
}
