use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use minkowski::{ComponentId, Entity, EnumChangeSet, MutationRef, World};

use crate::codec::{CodecError, CodecRegistry};
use crate::record::{ComponentSchema, SerializedMutation, WalEntry, WalSchema};

// WAL file format: `[len: u32 LE][payload: len bytes]` repeated.
// Each payload is a `WalRecord` serialized through rkyv.

/// Read exactly `buf.len()` bytes from `file` starting at byte offset `pos`.
fn read_exact_at(file: &File, pos: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut f = file;
    f.seek(SeekFrom::Start(pos))?;
    f.read_exact(buf)
}

#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("WAL I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("WAL codec error: {0}")]
    Codec(#[from] CodecError),
    #[error("WAL format error: {0}")]
    Format(String),
    #[error("cursor behind: requested seq {requested} but oldest available is {oldest}")]
    CursorBehind { requested: u64, oldest: u64 },
}

/// Maximum WAL frame size (256 MB). Rejects corrupt length prefixes
/// that would cause multi-gigabyte allocations.
const MAX_FRAME_SIZE: usize = 256 * 1024 * 1024;

/// Try to read the next WAL entry at byte offset `pos`.
/// Returns `Ok(Some((entry, next_pos)))` on success, `Ok(None)` if the
/// file ends cleanly at a frame boundary or a partial frame is found
/// (torn write). Returns `Err` on corrupt payload or oversized frame.
/// Does NOT truncate the file — callers decide how to handle errors.
pub(crate) fn read_next_frame(file: &File, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
    let mut len_buf = [0u8; 4];
    match read_exact_at(file, pos, &mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_SIZE {
        return Err(WalError::Format(format!(
            "WAL frame at offset {pos} claims {len} bytes, exceeding maximum {MAX_FRAME_SIZE}"
        )));
    }
    let mut payload = vec![0u8; len];
    match read_exact_at(file, pos + 4, &mut payload) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let entry = rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(&payload)
        .map_err(|e| WalError::Format(format!("corrupt WAL entry at byte offset {pos}: {e}")))?;
    Ok(Some((entry, pos + 4 + len as u64)))
}

/// Apply a single WAL record to a World, optionally remapping component IDs.
pub(crate) fn apply_record(
    record: &crate::record::WalRecord,
    world: &mut World,
    codecs: &CodecRegistry,
    remap: Option<&HashMap<ComponentId, ComponentId>>,
) -> Result<(), WalError> {
    // When no schema preamble exists (legacy WAL), use identity mapping.
    // When a schema exists, unmapped IDs are an error — the sender wrote
    // a mutation for a component not in its own preamble, which is corrupt.
    let remap_id = |id: ComponentId| -> Result<ComponentId, WalError> {
        match remap {
            None => Ok(id),
            Some(r) => r
                .get(&id)
                .copied()
                .ok_or(WalError::Codec(CodecError::UnregisteredComponent(id))),
        }
    };

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

                let mut raw_components: Vec<(minkowski::ComponentId, Vec<u8>, std::alloc::Layout)> =
                    Vec::new();
                for (comp_id, data) in components {
                    let local_id = remap_id(*comp_id)?;
                    let raw = codecs.deserialize(local_id, data)?;
                    let layout = codecs
                        .layout(local_id)
                        .ok_or(CodecError::UnregisteredComponent(local_id))?;
                    raw_components.push((local_id, raw, layout));
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
                let local_id = remap_id(*component_id)?;
                let raw = codecs.deserialize(local_id, data)?;
                let layout = codecs
                    .layout(local_id)
                    .ok_or(CodecError::UnregisteredComponent(local_id))?;
                changeset.record_insert(Entity::from_bits(*entity), local_id, raw.as_ptr(), layout);
            }
            SerializedMutation::Remove {
                entity,
                component_id,
            } => {
                changeset.record_remove(Entity::from_bits(*entity), remap_id(*component_id)?);
            }
        }
    }
    changeset.apply(world);
    Ok(())
}

/// Append-only write-ahead log. Each record is an rkyv-serialized changeset
/// with a monotonic sequence number.
pub struct Wal {
    file: File,
    next_seq: u64,
}

impl Wal {
    /// Create a new WAL file with a schema preamble. Fails if the file already exists.
    pub fn create(path: &Path, codecs: &CodecRegistry) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(path)?;
        let mut wal = Self { file, next_seq: 0 };
        wal.write_schema_preamble(codecs)?;
        Ok(wal)
    }

    /// Open an existing WAL file. Scans to find the next sequence number.
    pub fn open(path: &Path, codecs: &CodecRegistry) -> Result<Self, WalError> {
        let _ = codecs; // accepted for API symmetry; schema parsed during replay
        let file = OpenOptions::new().read(true).append(true).open(path)?;
        let mut wal = Self { file, next_seq: 0 };
        let (last, has_mutations) = wal.scan_last_seq()?;
        wal.next_seq = if has_mutations { last + 1 } else { 0 };
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
        let entry = WalEntry::Mutations(record);
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;

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
    /// If the WAL contains a schema preamble, component IDs are remapped
    /// from the sender's ID space to the receiver's via stable names.
    pub fn replay_from(
        &mut self,
        from_seq: u64,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        let mut pos: u64 = 0;
        let mut last_seq = if from_seq > 0 { from_seq - 1 } else { 0 };
        let mut remap: Option<HashMap<ComponentId, ComponentId>> = None;

        while let Some((entry, next_pos)) = self.read_next_entry(pos)? {
            match entry {
                WalEntry::Schema(schema) => {
                    remap = Some(codecs.build_remap(&schema.components)?);
                }
                WalEntry::Mutations(record) => {
                    if record.seq >= from_seq {
                        apply_record(&record, world, codecs, remap.as_ref())?;
                        last_seq = record.seq;
                    }
                }
            }
            pos = next_pos;
        }

        Ok(last_seq)
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Write a schema preamble as the first entry in a new WAL file.
    fn write_schema_preamble(&mut self, codecs: &CodecRegistry) -> Result<(), WalError> {
        let mut components = Vec::new();
        for &id in &codecs.registered_ids() {
            let name = codecs.stable_name(id).unwrap().to_string();
            let layout = codecs.layout(id).unwrap();
            components.push(ComponentSchema {
                id,
                name,
                size: layout.size(),
                align: layout.align(),
            });
        }
        let entry = WalEntry::Schema(WalSchema { components });
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;
        let mut writer = BufWriter::new(&self.file);
        let len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| WalError::Format("schema preamble too large".into()))?;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&payload)?;
        writer.flush()?;
        Ok(())
    }

    /// Try to read and deserialize the next WAL entry starting at `pos`.
    /// On EOF, partial frame, or corrupt data, truncates the file to `pos`
    /// (crash recovery) and returns `Ok(None)`.
    fn read_next_entry(&mut self, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
        match read_next_frame(&self.file, pos) {
            Ok(Some(result)) => Ok(Some(result)),
            Ok(None) | Err(WalError::Format(_)) => {
                self.file.set_len(pos)?;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

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

    /// Scan the WAL file to find the last valid sequence number.
    /// If a torn entry is found (partial length prefix or incomplete payload),
    /// the file is truncated to the last valid record boundary.
    // PERF: Full scan on open is required for crash recovery — the WAL has no
    // index or footer, so the only way to find the last valid record is linear
    // scan. This runs once at startup, not per-frame.
    /// Scan the WAL file to find the last valid mutation sequence number.
    /// Returns `(last_seq, has_mutations)`. Schema entries are skipped.
    fn scan_last_seq(&mut self) -> Result<(u64, bool), WalError> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut last_seq = 0u64;
        let mut has_mutations = false;
        let mut pos: u64 = 0;

        while let Some((entry, next_pos)) = self.read_next_entry(pos)? {
            if let WalEntry::Mutations(record) = entry {
                last_seq = record.seq;
                has_mutations = true;
            }
            pos = next_pos;
        }

        Ok((last_seq, has_mutations))
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
        let mut wal = Wal::create(&wal_path, &codecs).unwrap();
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
            let mut wal = Wal::create(&wal_path, &codecs).unwrap();
            for _ in 0..3 {
                let cs = EnumChangeSet::new(); // empty changeset
                wal.append(&cs, &codecs).unwrap();
            }
        }

        let wal2 = Wal::open(&wal_path, &codecs).unwrap();
        assert_eq!(wal2.next_seq(), 3);
    }

    #[test]
    fn replay_from_skips_earlier_records() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        let mut wal = Wal::create(&wal_path, &codecs).unwrap();

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

        let mut wal = Wal::create(&wal_path, &codecs).unwrap();
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
            let mut wal = Wal::create(&wal_path, &codecs).unwrap();
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
        let wal2 = Wal::open(&wal_path, &codecs).unwrap();
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
            let mut wal = Wal::create(&wal_path, &codecs).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        // Append torn entry: just a partial length prefix (2 bytes)
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&[0xFF, 0xFF]).unwrap();
            f.flush().unwrap();
        }

        let mut wal2 = Wal::open(&wal_path, &codecs).unwrap();
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
            let mut wal = Wal::create(&wal_path, &codecs).unwrap();
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
        let wal2 = Wal::open(&wal_path, &codecs).unwrap();
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
            let mut wal = Wal::create(&wal_path, &codecs).unwrap();
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

        let mut wal2 = Wal::open(&wal_path, &codecs).unwrap();
        let mut world2 = World::new();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0, "should replay the one valid record");

        let after_len = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(after_len, file_len, "corrupt record should be truncated");
    }

    #[test]
    fn create_writes_schema_preamble() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("schema.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);
        codecs.register_as::<Health>("health", &mut world);

        let _wal = Wal::create(&wal_path, &codecs).unwrap();

        // Re-open and verify schema is readable and seq starts at 0
        let wal2 = Wal::open(&wal_path, &codecs).unwrap();
        assert_eq!(wal2.next_seq(), 0);
    }

    #[test]
    fn wal_without_schema_preamble_replays_with_identity_mapping() {
        // WAL with Mutations entries but no Schema preamble — replay uses
        // identity mapping (no remap). Note: this is NOT backwards-compatible
        // with pre-stable-identity WAL files (which used a different rkyv root
        // type). This tests the "no schema" branch of the new format.
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("legacy.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        // Manually write a WAL with only Mutations (no Schema preamble)
        {
            let file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .read(true)
                .open(&wal_path)
                .unwrap();

            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Pos { x: 42.0, y: 99.0 },));

            // Build mutation record manually
            let mut mutations = Vec::new();
            for m in cs.iter_mutations() {
                mutations.push(Wal::serialize_mutation(&m, &codecs).unwrap());
            }
            let record = crate::record::WalRecord { seq: 0, mutations };
            let entry = WalEntry::Mutations(record);
            let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();

            let mut writer = std::io::BufWriter::new(&file);
            writer
                .write_all(&(payload.len() as u32).to_le_bytes())
                .unwrap();
            writer.write_all(&payload).unwrap();
            writer.flush().unwrap();
        }

        // Open and replay — should work without schema (no remapping)
        let mut wal = Wal::open(&wal_path, &codecs).unwrap();
        let mut world2 = World::new();
        codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);

        let last = wal.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0);
        assert_eq!(world2.query::<(&Pos,)>().count(), 1);
        let p = world2.query::<(&Pos,)>().next().unwrap().0;
        assert_eq!(p.x, 42.0);
        assert_eq!(p.y, 99.0);
    }

    #[test]
    fn wal_cross_process_different_registration_order() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("cross.wal");

        // "Process A": Pos=0, Health=1
        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Health>("health", &mut world_a);

        let mut wal = Wal::create(&wal_path, &codecs_a).unwrap();

        let e = world_a.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 }, Health(100)));
        wal.append(&cs, &codecs_a).unwrap();
        cs.apply(&mut world_a);

        drop(wal);

        // "Process B": Health=0, Pos=1 (opposite order)
        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Health>("health", &mut world_b);
        codecs_b.register_as::<Pos>("pos", &mut world_b);

        let mut wal_b = Wal::open(&wal_path, &codecs_b).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        // Verify: data is correct despite different registration order
        let positions: Vec<(f32, f32)> =
            world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let health: Vec<u32> = world_b.query::<(&Health,)>().map(|h| h.0 .0).collect();
        assert_eq!(health, vec![100]);
    }

    #[test]
    fn wal_cross_process_insert_and_remove_remapped() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("cross_insert.wal");

        // "Process A": Pos=0, Health=1
        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Health>("health", &mut world_a);

        let mut wal = Wal::create(&wal_path, &codecs_a).unwrap();

        // Spawn with Pos only
        let e = world_a.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 },));
        wal.append(&cs, &codecs_a).unwrap();
        cs.apply(&mut world_a);

        // Insert Health, then Remove Pos
        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world_a, e, Health(50));
        cs2.remove::<Pos>(&mut world_a, e);
        wal.append(&cs2, &codecs_a).unwrap();
        cs2.apply(&mut world_a);

        drop(wal);

        // "Process B": opposite order
        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Health>("health", &mut world_b);
        codecs_b.register_as::<Pos>("pos", &mut world_b);

        let mut wal_b = Wal::open(&wal_path, &codecs_b).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        // Entity should have Health(50) but no Pos
        let health: Vec<u32> = world_b.query::<(&Health,)>().map(|h| h.0 .0).collect();
        assert_eq!(health, vec![50]);
        assert_eq!(world_b.query::<(&Pos,)>().count(), 0);
    }
}
