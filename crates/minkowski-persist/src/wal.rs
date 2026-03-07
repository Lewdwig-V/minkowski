use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

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

/// Configuration for segmented WAL.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum bytes per segment file before rolling to a new segment.
    /// Default: 64 MB.
    pub max_segment_bytes: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Generate the filename for a segment starting at `start_seq`.
fn segment_filename(start_seq: u64) -> String {
    format!("wal-seq{start_seq:06}.seg")
}

/// Parse the start-seq from a segment filename. Returns `None` if the
/// filename doesn't match the expected pattern.
fn parse_segment_start_seq(filename: &str) -> Option<u64> {
    let rest = filename.strip_prefix("wal-seq")?.strip_suffix(".seg")?;
    rest.parse().ok()
}

/// List all segment files in a directory, sorted by start-seq ascending.
/// Returns `(start_seq, full_path)` pairs.
pub(crate) fn list_segments(dir: &Path) -> Result<Vec<(u64, PathBuf)>, WalError> {
    let mut segments = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(seq) = parse_segment_start_seq(&name_str) {
            segments.push((seq, entry.path()));
        }
    }
    segments.sort_by_key(|(seq, _)| *seq);
    Ok(segments)
}

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

/// Segmented append-only write-ahead log. Each segment is an rkyv-serialized
/// stream of `WalEntry` frames with a schema preamble. Segments roll over
/// when they exceed `WalConfig::max_segment_bytes`.
pub struct Wal {
    dir: PathBuf,
    active_file: File,
    active_start_seq: u64,
    active_bytes: u64,
    next_seq: u64,
    config: WalConfig,
    schema: WalSchema,
}

impl Wal {
    /// Create a new segmented WAL directory with the first segment.
    pub fn create(dir: &Path, codecs: &CodecRegistry, config: WalConfig) -> Result<Self, WalError> {
        std::fs::create_dir_all(dir)?;
        let schema = Self::build_schema(codecs);
        let seg_path = dir.join(segment_filename(0));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&seg_path)?;
        let mut wal = Self {
            dir: dir.to_path_buf(),
            active_file: file,
            active_start_seq: 0,
            active_bytes: 0,
            next_seq: 0,
            config,
            schema,
        };
        wal.active_bytes = wal.write_schema_preamble()?;
        Ok(wal)
    }

    /// Open an existing segmented WAL directory.
    /// Scans for segments, opens the last one for appending, recovers `next_seq`.
    /// Config governs future segment rollover.
    pub fn open(dir: &Path, codecs: &CodecRegistry, config: WalConfig) -> Result<Self, WalError> {
        let segments = list_segments(dir)?;
        if segments.is_empty() {
            return Err(WalError::Format(
                "no WAL segments found in directory".into(),
            ));
        }

        let (_last_start_seq, last_path) = segments.last().unwrap().clone();

        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&last_path)?;

        let schema = Self::build_schema(codecs);

        let mut wal = Self {
            dir: dir.to_path_buf(),
            active_file: file,
            active_start_seq: _last_start_seq,
            active_bytes: 0,
            next_seq: 0,
            config,
            schema,
        };

        // Crash recovery: scan the active segment, truncating torn/corrupt tail
        let (active_last_seq, active_has) = wal.scan_active_segment()?;
        wal.active_bytes = wal.active_file.metadata()?.len();

        if active_has {
            wal.next_seq = active_last_seq + 1;
        } else {
            // Active segment has no mutations — check earlier segments
            for (_, seg_path) in segments.iter().rev().skip(1) {
                let seg_file = File::open(seg_path)?;
                let mut pos: u64 = 0;
                let mut seg_last = 0u64;
                let mut seg_has = false;
                while let Some((entry, next_pos)) = read_next_frame(&seg_file, pos)? {
                    if let WalEntry::Mutations(record) = entry {
                        seg_last = record.seq;
                        seg_has = true;
                    }
                    pos = next_pos;
                }
                if seg_has {
                    wal.next_seq = seg_last + 1;
                    break;
                }
            }
        }

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

        {
            let mut writer = BufWriter::new(&self.active_file);
            let len: u32 = payload.len().try_into().map_err(|_| {
                WalError::Format(format!(
                    "WAL record too large: {} bytes exceeds u32 max",
                    payload.len()
                ))
            })?;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&payload)?;
            writer.flush()?;
        }

        self.active_bytes += 4 + payload.len() as u64;
        self.next_seq += 1;

        // Roll to new segment if threshold exceeded
        if self.active_bytes >= self.config.max_segment_bytes as u64 {
            self.roll_segment()?;
        }

        Ok(seq)
    }

    /// Replay all records across all segments into a world.
    /// Returns the last sequence number replayed, or 0 if empty.
    pub fn replay(&mut self, world: &mut World, codecs: &CodecRegistry) -> Result<u64, WalError> {
        self.replay_from(0, world, codecs)
    }

    /// Replay records starting from (and including) a given sequence number.
    /// Iterates across all segments. Schema preambles are used for component
    /// ID remapping from the sender's ID space to the receiver's.
    pub fn replay_from(
        &mut self,
        from_seq: u64,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        let segments = list_segments(&self.dir)?;
        let mut last_seq = if from_seq > 0 { from_seq - 1 } else { 0 };

        for (_, seg_path) in &segments {
            let seg_file = File::open(seg_path)?;
            let mut pos: u64 = 0;
            let mut remap: Option<HashMap<ComponentId, ComponentId>> = None;

            while let Some((entry, next_pos)) = read_next_frame(&seg_file, pos)? {
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
        }

        Ok(last_seq)
    }

    /// Delete all segment files whose entire seq range is before `seq`.
    /// A segment is safe to delete if the next segment's start_seq <= `seq`.
    /// The active (last) segment is never deleted.
    /// Returns the number of segments deleted.
    pub fn delete_segments_before(&mut self, seq: u64) -> Result<usize, WalError> {
        let segments = list_segments(&self.dir)?;
        if segments.len() <= 1 {
            return Ok(0);
        }

        let mut deleted = 0;
        for i in 0..segments.len() - 1 {
            let next_start = segments[i + 1].0;
            if next_start <= seq {
                std::fs::remove_file(&segments[i].1)?;
                deleted += 1;
            } else {
                break; // segments are sorted, no point continuing
            }
        }

        Ok(deleted)
    }

    /// Number of segment files in the WAL directory.
    pub fn segment_count(&self) -> usize {
        list_segments(&self.dir).map(|s| s.len()).unwrap_or(0)
    }

    /// Start-seq of the oldest remaining segment, or `None` if no segments exist.
    pub fn oldest_seq(&self) -> Option<u64> {
        list_segments(&self.dir)
            .ok()
            .and_then(|s| s.first().map(|(seq, _)| *seq))
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn build_schema(codecs: &CodecRegistry) -> WalSchema {
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
        WalSchema { components }
    }

    /// Write a schema preamble to the active segment. Returns bytes written.
    fn write_schema_preamble(&mut self) -> Result<u64, WalError> {
        let entry = WalEntry::Schema(self.schema.clone());
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;
        let mut writer = BufWriter::new(&self.active_file);
        let len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| WalError::Format("schema preamble too large".into()))?;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&payload)?;
        writer.flush()?;
        Ok(4 + payload.len() as u64)
    }

    /// Roll to a new segment file.
    fn roll_segment(&mut self) -> Result<(), WalError> {
        let seg_path = self.dir.join(segment_filename(self.next_seq));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&seg_path)?;
        self.active_file = file;
        self.active_start_seq = self.next_seq;
        self.active_bytes = self.write_schema_preamble()?;
        Ok(())
    }

    /// Try to read the next entry from the active segment.
    /// On EOF, partial frame, or corrupt data, truncates the file to `pos`
    /// (crash recovery) and returns `Ok(None)`.
    fn read_next_entry(&mut self, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
        match read_next_frame(&self.active_file, pos) {
            Ok(Some(result)) => Ok(Some(result)),
            Ok(None) | Err(WalError::Format(_)) => {
                self.active_file.set_len(pos)?;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Scan the active segment for crash recovery. Truncates torn/corrupt tail.
    /// Returns `(last_seq, has_mutations)`.
    // PERF: Full scan on open is required for crash recovery — the WAL has no
    // index or footer, so the only way to find the last valid record is linear
    // scan. This runs once at startup, not per-frame.
    fn scan_active_segment(&mut self) -> Result<(u64, bool), WalError> {
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

    fn default_config() -> WalConfig {
        WalConfig::default()
    }

    fn small_config() -> WalConfig {
        WalConfig {
            max_segment_bytes: 128,
        }
    }

    #[test]
    fn create_append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let seq = wal.append(&cs, &codecs).unwrap();
        assert_eq!(seq, 0);
        assert_eq!(wal.next_seq(), 1);

        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn open_existing_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            for _ in 0..3 {
                let cs = EnumChangeSet::new();
                wal.append(&cs, &codecs).unwrap();
            }
        }

        let wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal2.next_seq(), 3);
    }

    #[test]
    fn replay_from_skips_earlier_records() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

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
        let wal_dir = dir.path().join("empty.wal");

        let mut world = World::new();
        let codecs = CodecRegistry::new();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let last = wal.replay(&mut world, &codecs).unwrap();
        assert_eq!(last, 0);
    }

    #[test]
    fn torn_entry_truncated_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("torn.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        // Append garbage to the active segment
        let seg_path = wal_dir.join(segment_filename(0));
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            f.write_all(&1000u32.to_le_bytes()).unwrap();
            f.write_all(&[0u8; 5]).unwrap();
            f.flush().unwrap();
        }

        let file_len_before = std::fs::metadata(&seg_path).unwrap().len();

        let wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal2.next_seq(), 2);

        let file_len_after = std::fs::metadata(&seg_path).unwrap().len();
        assert!(file_len_after < file_len_before, "file should be truncated");
    }

    #[test]
    fn torn_entry_truncated_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("torn_replay.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        // Append torn entry to the active segment
        let seg_path = wal_dir.join(segment_filename(0));
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            f.write_all(&[0xFF, 0xFF]).unwrap();
            f.flush().unwrap();
        }

        let mut wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        let mut world2 = World::new();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0, "should replay the one valid record");
    }

    #[test]
    fn corrupted_payload_truncated_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("corrupt_payload.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        let seg_path = wal_dir.join(segment_filename(0));
        let file_len = std::fs::metadata(&seg_path).unwrap().len();
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            f.write_all(&20u32.to_le_bytes()).unwrap();
            f.write_all(&[0xDE; 20]).unwrap();
            f.flush().unwrap();
        }

        let new_len = std::fs::metadata(&seg_path).unwrap().len();
        assert!(new_len > file_len);

        let wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal2.next_seq(), 2);

        let after_len = std::fs::metadata(&seg_path).unwrap().len();
        assert_eq!(after_len, file_len);
    }

    #[test]
    fn corrupted_payload_truncated_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("corrupt_replay.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        let seg_path = wal_dir.join(segment_filename(0));
        let file_len = std::fs::metadata(&seg_path).unwrap().len();
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            f.write_all(&15u32.to_le_bytes()).unwrap();
            f.write_all(&[0xAB; 15]).unwrap();
            f.flush().unwrap();
        }

        let mut wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        let mut world2 = World::new();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0);

        let after_len = std::fs::metadata(&seg_path).unwrap().len();
        assert_eq!(after_len, file_len);
    }

    #[test]
    fn create_writes_schema_preamble() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("schema.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);
        codecs.register_as::<Health>("health", &mut world);

        let _wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal2.next_seq(), 0);
    }

    #[test]
    fn wal_cross_process_different_registration_order() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("cross.wal");

        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Health>("health", &mut world_a);

        let mut wal = Wal::create(&wal_dir, &codecs_a, default_config()).unwrap();

        let e = world_a.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 }, Health(100)));
        wal.append(&cs, &codecs_a).unwrap();
        cs.apply(&mut world_a);

        drop(wal);

        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Health>("health", &mut world_b);
        codecs_b.register_as::<Pos>("pos", &mut world_b);

        let mut wal_b = Wal::open(&wal_dir, &codecs_b, default_config()).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        let positions: Vec<(f32, f32)> =
            world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let health: Vec<u32> = world_b.query::<(&Health,)>().map(|h| h.0 .0).collect();
        assert_eq!(health, vec![100]);
    }

    #[test]
    fn segment_filename_format() {
        assert_eq!(segment_filename(0), "wal-seq000000.seg");
        assert_eq!(segment_filename(47), "wal-seq000047.seg");
        assert_eq!(segment_filename(123456), "wal-seq123456.seg");
    }

    #[test]
    fn parse_segment_start_seq_valid() {
        assert_eq!(parse_segment_start_seq("wal-seq000000.seg"), Some(0));
        assert_eq!(parse_segment_start_seq("wal-seq000047.seg"), Some(47));
        assert_eq!(parse_segment_start_seq("wal-seq123456.seg"), Some(123456));
    }

    #[test]
    fn parse_segment_start_seq_invalid() {
        assert_eq!(parse_segment_start_seq("not-a-segment.txt"), None);
        assert_eq!(parse_segment_start_seq("wal-seq.seg"), None);
        assert_eq!(parse_segment_start_seq("wal-seqABCDEF.seg"), None);
    }

    #[test]
    fn list_segments_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("wal-seq000100.seg"), b"").unwrap();
        std::fs::write(dir.path().join("wal-seq000000.seg"), b"").unwrap();
        std::fs::write(dir.path().join("wal-seq000050.seg"), b"").unwrap();
        std::fs::write(dir.path().join("not-a-segment.txt"), b"").unwrap();

        let segments = list_segments(dir.path()).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].0, 0);
        assert_eq!(segments[1].0, 50);
        assert_eq!(segments[2].0, 100);
    }

    #[test]
    fn list_segments_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let segments = list_segments(dir.path()).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn wal_config_default() {
        let config = WalConfig::default();
        assert_eq!(config.max_segment_bytes, 64 * 1024 * 1024);
    }

    #[test]
    fn wal_cross_process_insert_and_remove_remapped() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("cross_insert.wal");

        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Health>("health", &mut world_a);

        let mut wal = Wal::create(&wal_dir, &codecs_a, default_config()).unwrap();

        let e = world_a.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 },));
        wal.append(&cs, &codecs_a).unwrap();
        cs.apply(&mut world_a);

        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world_a, e, Health(50));
        cs2.remove::<Pos>(&mut world_a, e);
        wal.append(&cs2, &codecs_a).unwrap();
        cs2.apply(&mut world_a);

        drop(wal);

        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Health>("health", &mut world_b);
        codecs_b.register_as::<Pos>("pos", &mut world_b);

        let mut wal_b = Wal::open(&wal_dir, &codecs_b, default_config()).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        let health: Vec<u32> = world_b.query::<(&Health,)>().map(|h| h.0 .0).collect();
        assert_eq!(health, vec![50]);
        assert_eq!(world_b.query::<(&Pos,)>().count(), 0);
    }

    // ── Segmented WAL tests ──────────────────────────────────────────

    #[test]
    fn create_segmented_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();
        assert_eq!(wal.next_seq(), 0);
        assert_eq!(wal.segment_count(), 1);
        assert!(wal_dir.is_dir());
        assert_eq!(wal.oldest_seq(), Some(0));
    }

    #[test]
    fn open_empty_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("empty.wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        let codecs = CodecRegistry::new();
        let result = Wal::open(&wal_dir, &codecs, default_config());
        assert!(result.is_err());
    }

    #[test]
    fn append_rolls_to_new_segment() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();

        for i in 0..20 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }

        assert_eq!(wal.next_seq(), 20);
        assert!(
            wal.segment_count() > 1,
            "should have rolled to multiple segments"
        );

        // Every segment should start with a schema preamble
        let segments = list_segments(&wal_dir).unwrap();
        for (_, seg_path) in &segments {
            let file = File::open(seg_path).unwrap();
            let (entry, _) = read_next_frame(&file, 0).unwrap().unwrap();
            assert!(matches!(entry, WalEntry::Schema(_)));
        }
    }

    #[test]
    fn open_after_rollover_recovers_next_seq() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();
            for i in 0..10 {
                let e = world.alloc_entity();
                let mut cs = EnumChangeSet::new();
                cs.spawn_bundle(
                    &mut world,
                    e,
                    (Pos {
                        x: i as f32,
                        y: 0.0,
                    },),
                );
                wal.append(&cs, &codecs).unwrap();
                cs.apply(&mut world);
            }
        }

        let wal2 = Wal::open(&wal_dir, &codecs, small_config()).unwrap();
        assert_eq!(wal2.next_seq(), 10);
    }

    #[test]
    fn replay_across_segments() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();

        for i in 0..10 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }

        let mut world2 = World::new();
        codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);
        let last = wal.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 9);
        assert_eq!(world2.query::<(&Pos,)>().count(), 10);
    }

    #[test]
    fn delete_segments_before_removes_old() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();

        for i in 0..20 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }

        let before = wal.segment_count();
        assert!(before > 2);

        let deleted = wal.delete_segments_before(10).unwrap();
        assert!(deleted > 0);
        assert_eq!(wal.segment_count(), before - deleted);
        assert!(wal.oldest_seq().is_some());
    }

    #[test]
    fn delete_segments_preserves_active() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();

        for i in 0..10 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }

        wal.delete_segments_before(u64::MAX).unwrap();
        assert!(wal.segment_count() >= 1);

        // WAL should still be appendable
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },));
        wal.append(&cs, &codecs).unwrap();
    }
}
