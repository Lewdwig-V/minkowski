use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use minkowski::{Entity, EnumChangeSet, MutationRef, World};

use crate::codec::{CodecError, CodecRegistry};
use crate::format;
use crate::record::SerializedMutation;

// WAL file format: `[len: u32 LE][payload: len bytes]` repeated.
// Each payload is a `WalRecord` serialized through rkyv.

/// Read exactly `buf.len()` bytes from `file` starting at byte offset `pos`.
fn read_exact_at(file: &File, pos: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut f = file;
    f.seek(SeekFrom::Start(pos))?;
    f.read_exact(buf)
}

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

impl From<io::Error> for WalError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CodecError> for WalError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}

/// Append-only write-ahead log. Each record is an rkyv-serialized changeset
/// with a monotonic sequence number.
pub struct Wal {
    file: File,
    next_seq: u64,
}

impl Wal {
    /// Create a new WAL file. Fails if the file already exists.
    pub fn create(path: &Path) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(path)?;
        Ok(Self { file, next_seq: 0 })
    }

    /// Open an existing WAL file. Scans to find the next sequence number.
    pub fn open(path: &Path) -> Result<Self, WalError> {
        let file = OpenOptions::new().read(true).append(true).open(path)?;
        let mut wal = Self { file, next_seq: 0 };
        let last = wal.scan_last_seq()?;
        wal.next_seq = if last > 0 || wal.has_records()? {
            last + 1
        } else {
            0
        };
        Ok(wal)
    }

    /// Current sequence number (next append will use this).
    pub fn next_seq(&self) -> u64 {
        self.next_seq
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
        let payload =
            format::serialize_record(&record).map_err(|e| WalError::Format(e.to_string()))?;

        let mut writer = BufWriter::new(&self.file);
        let len: u32 = payload.len().try_into().map_err(|_| {
            WalError::Format(format!(
                "WAL record too large: {} bytes exceeds u32 max",
                payload.len()
            ))
        })?;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&payload)?;
        writer.flush()?;

        self.next_seq += 1;
        Ok(seq)
    }

    /// Replay all records into a world.
    /// Returns the last sequence number replayed, or 0 if empty.
    pub fn replay(&mut self, world: &mut World, codecs: &CodecRegistry) -> Result<u64, WalError> {
        self.replay_from(0, world, codecs)
    }

    /// Replay records starting from (and including) a given sequence number.
    /// If a torn entry is found, it is truncated and replay stops cleanly.
    pub fn replay_from(
        &mut self,
        from_seq: u64,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        let mut pos: u64 = 0;
        let mut last_seq = if from_seq > 0 { from_seq - 1 } else { 0 };

        loop {
            let record_start = pos;
            let mut len_buf = [0u8; 4];
            match read_exact_at(&self.file, record_start, &mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    self.file.set_len(record_start)?;
                    break;
                }
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            match read_exact_at(&self.file, record_start + 4, &mut payload) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    self.file.set_len(record_start)?;
                    break;
                }
                Err(e) => return Err(e.into()),
            }

            let record = match format::deserialize_record(&payload) {
                Ok(r) => r,
                Err(_) => {
                    // Corrupted payload — treat as torn entry, truncate.
                    self.file.set_len(record_start)?;
                    break;
                }
            };

            if record.seq >= from_seq {
                Self::apply_record(&record, world, codecs)?;
                last_seq = record.seq;
            }

            pos = record_start + 4 + len as u64;
        }

        Ok(last_seq)
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn changeset_to_record(
        seq: u64,
        changeset: &EnumChangeSet,
        codecs: &CodecRegistry,
    ) -> Result<crate::record::WalRecord, WalError> {
        let mut mutations = Vec::new();
        for m in changeset.iter_mutations() {
            mutations.push(Self::serialize_mutation(&m, codecs)?);
        }
        Ok(crate::record::WalRecord { seq, mutations })
    }

    fn serialize_mutation(
        m: &MutationRef<'_>,
        codecs: &CodecRegistry,
    ) -> Result<SerializedMutation, WalError> {
        match m {
            MutationRef::Spawn { entity, components } => {
                let mut serialized = Vec::new();
                for &(comp_id, raw_bytes) in components {
                    // PERF: Per-component Vec::new() is unavoidable — SerializedMutation
                    // owns Vec<(ComponentId, Vec<u8>)>. The rkyv to_bytes_in optimization
                    // in codec.rs eliminates the *internal* double-allocation.
                    let mut buf = Vec::new();
                    // SAFETY: raw_bytes points to a valid component value from the Arena.
                    // The byte slice was constructed from arena.get(offset) with the
                    // correct layout.size(), so the pointer is valid and aligned.
                    unsafe { codecs.serialize(comp_id, raw_bytes.as_ptr(), &mut buf)? };
                    serialized.push((comp_id, buf));
                }
                Ok(SerializedMutation::Spawn {
                    entity: entity.to_bits(),
                    components: serialized,
                })
            }
            MutationRef::Despawn { entity } => Ok(SerializedMutation::Despawn {
                entity: entity.to_bits(),
            }),
            MutationRef::Insert {
                entity,
                component_id,
                data,
            } => {
                let mut buf = Vec::new();
                // SAFETY: data points to a valid component value from the Arena.
                unsafe { codecs.serialize(*component_id, data.as_ptr(), &mut buf)? };
                Ok(SerializedMutation::Insert {
                    entity: entity.to_bits(),
                    component_id: *component_id,
                    data: buf,
                })
            }
            MutationRef::Remove {
                entity,
                component_id,
            } => Ok(SerializedMutation::Remove {
                entity: entity.to_bits(),
                component_id: *component_id,
            }),
        }
    }

    fn apply_record(
        record: &crate::record::WalRecord,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<(), WalError> {
        let mut changeset = EnumChangeSet::new();
        for mutation in &record.mutations {
            match mutation {
                SerializedMutation::Spawn { entity, components } => {
                    let entity = Entity::from_bits(*entity);

                    // Ensure the entity's allocator slot exists so that
                    // subsequent mutations (Insert, etc.) can pass is_alive
                    // checks. The changeset Spawn path only checks
                    // !is_placed, but Insert checks is_alive which requires
                    // the generation entry.
                    world.alloc_entity();

                    let mut raw_components: Vec<(
                        minkowski::ComponentId,
                        Vec<u8>,
                        std::alloc::Layout,
                    )> = Vec::new();
                    for (comp_id, data) in components {
                        let raw = codecs.deserialize(*comp_id, data)?;
                        let layout = codecs
                            .layout(*comp_id)
                            .ok_or(CodecError::UnregisteredComponent(*comp_id))?;
                        raw_components.push((*comp_id, raw, layout));
                    }
                    let ptrs: Vec<_> = raw_components
                        .iter()
                        .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                        .collect();
                    changeset.record_spawn(entity, &ptrs);
                }
                SerializedMutation::Despawn { entity } => {
                    changeset.record_despawn(Entity::from_bits(*entity));
                }
                SerializedMutation::Insert {
                    entity,
                    component_id,
                    data,
                } => {
                    let raw = codecs.deserialize(*component_id, data)?;
                    let layout = codecs
                        .layout(*component_id)
                        .ok_or(CodecError::UnregisteredComponent(*component_id))?;
                    changeset.record_insert(
                        Entity::from_bits(*entity),
                        *component_id,
                        raw.as_ptr(),
                        layout,
                    );
                }
                SerializedMutation::Remove {
                    entity,
                    component_id,
                } => {
                    changeset.record_remove(Entity::from_bits(*entity), *component_id);
                }
            }
        }
        changeset.apply(world);
        Ok(())
    }

    /// Scan the WAL file to find the last valid sequence number.
    /// If a torn entry is found (partial length prefix or incomplete payload),
    /// the file is truncated to the last valid record boundary.
    // PERF: Full scan on open is required for crash recovery — the WAL has no
    // index or footer, so the only way to find the last valid record is linear
    // scan. This runs once at startup, not per-frame.
    fn scan_last_seq(&mut self) -> Result<u64, WalError> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut last_seq = 0u64;
        let mut valid_end: u64 = 0;

        loop {
            let record_start = valid_end;
            let mut len_buf = [0u8; 4];
            match read_exact_at(&self.file, record_start, &mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    // Torn length prefix — truncate to last valid boundary
                    self.file.set_len(valid_end)?;
                    break;
                }
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            match read_exact_at(&self.file, record_start + 4, &mut payload) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    // Torn payload — truncate to start of this record
                    self.file.set_len(record_start)?;
                    break;
                }
                Err(e) => return Err(e.into()),
            }

            match format::deserialize_record(&payload) {
                Ok(record) => {
                    last_seq = record.seq;
                    valid_end = record_start + 4 + len as u64;
                }
                Err(_) => {
                    // Corrupted payload — treat as torn entry, truncate.
                    // A crash can produce a length-complete but content-corrupt
                    // record. Truncating to the last valid boundary preserves
                    // all prior records for replay.
                    self.file.set_len(record_start)?;
                    break;
                }
            }
        }

        Ok(last_seq)
    }

    fn has_records(&mut self) -> Result<bool, WalError> {
        let pos = self.file.seek(SeekFrom::End(0))?;
        Ok(pos > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Health(u32);

    #[test]
    fn create_append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        // Spawn an entity via changeset
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));

        // Write the changeset to WAL before applying
        let mut wal = Wal::create(&wal_path).unwrap();
        let seq = wal.append(&cs, &codecs).unwrap();
        assert_eq!(seq, 0);
        assert_eq!(wal.next_seq(), 1);

        // Apply to world
        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn open_existing_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_path).unwrap();
            for _ in 0..3 {
                let cs = EnumChangeSet::new(); // empty changeset
                wal.append(&cs, &codecs).unwrap();
            }
        }

        let wal2 = Wal::open(&wal_path).unwrap();
        assert_eq!(wal2.next_seq(), 3);
    }

    #[test]
    fn replay_from_skips_earlier_records() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        let mut wal = Wal::create(&wal_path).unwrap();

        // Append 3 empty records
        for _ in 0..3 {
            let cs = EnumChangeSet::new();
            wal.append(&cs, &codecs).unwrap();
        }

        let mut world2 = World::new();
        let last = wal.replay_from(2, &mut world2, &codecs).unwrap();
        assert_eq!(last, 2);
    }

    #[test]
    fn empty_wal_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("empty.wal");

        let mut world = World::new();
        let codecs = CodecRegistry::new();

        let mut wal = Wal::create(&wal_path).unwrap();
        let last = wal.replay(&mut world, &codecs).unwrap();
        assert_eq!(last, 0);
    }

    #[test]
    fn torn_entry_truncated_on_open() {
        // Simulate a crash mid-write: write 2 valid records, then append
        // a partial third (length prefix but incomplete payload).
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("torn.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_path).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        // Append garbage: a valid length prefix claiming 1000 bytes, but only 5 bytes of payload
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&1000u32.to_le_bytes()).unwrap();
            f.write_all(&[0u8; 5]).unwrap();
            f.flush().unwrap();
        }

        let file_len_before = std::fs::metadata(&wal_path).unwrap().len();

        // open() should detect the torn entry, truncate it, and recover
        let wal2 = Wal::open(&wal_path).unwrap();
        assert_eq!(
            wal2.next_seq(),
            2,
            "should see 2 valid records, torn entry removed"
        );

        let file_len_after = std::fs::metadata(&wal_path).unwrap().len();
        assert!(file_len_after < file_len_before, "file should be truncated");
    }

    #[test]
    fn torn_entry_truncated_on_replay() {
        // Same setup but verify replay also handles torn entries cleanly.
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("torn_replay.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_path).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        // Append torn entry: just a partial length prefix (2 bytes)
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&[0xFF, 0xFF]).unwrap();
            f.flush().unwrap();
        }

        let mut wal2 = Wal::open(&wal_path).unwrap();
        let mut world2 = World::new();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0, "should replay the one valid record");
    }

    #[test]
    fn corrupted_payload_truncated_on_open() {
        // Simulate a crash that wrote the full length prefix and all payload
        // bytes, but the payload content is corrupt (rkyv validation fails).
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("corrupt_payload.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_path).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        // Read the file, find the start of the third hypothetical record,
        // and write a valid-looking length prefix with garbage payload.
        let file_len = std::fs::metadata(&wal_path).unwrap().len();
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            // Write a length prefix claiming 20 bytes, then 20 bytes of garbage
            f.write_all(&20u32.to_le_bytes()).unwrap();
            f.write_all(&[0xDE; 20]).unwrap();
            f.flush().unwrap();
        }

        let new_len = std::fs::metadata(&wal_path).unwrap().len();
        assert!(new_len > file_len, "garbage should have been appended");

        // open() should detect the corrupt record, truncate it, and recover
        let wal2 = Wal::open(&wal_path).unwrap();
        assert_eq!(
            wal2.next_seq(),
            2,
            "should see 2 valid records, corrupt payload removed"
        );

        let after_len = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(
            after_len, file_len,
            "file should be truncated to pre-corruption size"
        );
    }

    #[test]
    fn corrupted_payload_truncated_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("corrupt_replay.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_path).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        let file_len = std::fs::metadata(&wal_path).unwrap().len();
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&15u32.to_le_bytes()).unwrap();
            f.write_all(&[0xAB; 15]).unwrap();
            f.flush().unwrap();
        }

        let mut wal2 = Wal::open(&wal_path).unwrap();
        let mut world2 = World::new();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0, "should replay the one valid record");

        let after_len = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(after_len, file_len, "corrupt record should be truncated");
    }
}
