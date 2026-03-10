use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use minkowski::{ComponentId, Entity, EnumChangeSet, MutationRef, World};

use crate::codec::{CodecError, CodecRegistry};
use crate::record::{ComponentSchema, SerializedMutation, WalEntry, WalSchema};

// WAL segment format (v2):
//   [segment_magic: 4 bytes "MKW2"]
//   [frame0: len+crc+payload]          — schema preamble
//   [frame1: len+crc+payload]          — data / checkpoint
//   ...
//
// Frame format: `[len: u32 LE][crc32: u32 LE][payload: len bytes]`.
// Each payload is a `WalEntry` (Schema | Mutations | Checkpoint) serialized
// through rkyv. The CRC32 (IEEE via crc32fast) covers the payload bytes
// and catches silent data corruption that rkyv validation alone might miss.
//
// Legacy v1 segments (no magic header, no CRC32) are detected at open time
// and produce a hard `WalError::Format` error — they are never silently
// truncated or reinterpreted.

/// Frame header size: 4 bytes length + 4 bytes CRC32.
const FRAME_HEADER_SIZE: u64 = 8;

/// Segment file magic identifying v2 format with CRC32 checksums.
const SEGMENT_MAGIC: [u8; 4] = *b"MKW2";

/// Size of the segment magic header in bytes.
const SEGMENT_MAGIC_SIZE: u64 = 4;

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
    #[error("WAL checksum mismatch at byte offset {offset}: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch {
        offset: u64,
        expected: u32,
        actual: u32,
    },
    #[error("cursor behind: requested seq {requested} but oldest available is {oldest}")]
    CursorBehind { requested: u64, oldest: u64 },
    #[error("WAL apply error: {0}")]
    Apply(#[from] minkowski::ApplyError),
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
    /// Maximum bytes of mutation data between checkpoint markers.
    /// `None` disables checkpoint enforcement (default).
    pub max_bytes_between_checkpoints: Option<usize>,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: None,
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

/// Validate the segment magic at the start of a file. Returns `Ok(())` if
/// the magic matches v2 format. Returns `Err(WalError::Format)` with a
/// descriptive message if the file uses a legacy v1 format (no magic header).
/// Returns `Ok(())` on UnexpectedEof (empty/torn file — caller handles recovery).
fn validate_segment_magic(file: &File, path: &Path) -> Result<(), WalError> {
    let mut buf = [0u8; SEGMENT_MAGIC_SIZE as usize];
    match read_exact_at(file, 0, &mut buf) {
        Ok(()) => {}
        // Empty or torn file — not a legacy format issue, just incomplete.
        // Caller handles this via truncation / rewrite.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
        Err(e) => return Err(e.into()),
    }
    if buf != SEGMENT_MAGIC {
        return Err(WalError::Format(format!(
            "segment {} uses legacy v1 format (no CRC32 checksums); \
             migrate by replaying into a new WAL or rebuild from snapshot",
            path.display()
        )));
    }
    Ok(())
}

/// Write the segment magic header. Returns bytes written (always SEGMENT_MAGIC_SIZE).
fn write_segment_magic(writer: &mut BufWriter<&File>) -> Result<u64, WalError> {
    writer.write_all(&SEGMENT_MAGIC)?;
    Ok(SEGMENT_MAGIC_SIZE)
}

/// Try to read the next WAL entry at byte offset `pos`.
/// Returns `Ok(Some((entry, next_pos)))` on success, `Ok(None)` if the
/// file ends cleanly at a frame boundary or a partial frame is found
/// (torn write). Returns `Err` on corrupt payload, checksum mismatch,
/// or oversized frame. Does NOT truncate the file — callers decide how
/// to handle errors.
pub(crate) fn read_next_frame(file: &File, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
    // Read 8-byte header: [len: u32 LE][crc32: u32 LE]
    let mut header_buf = [0u8; FRAME_HEADER_SIZE as usize];
    match read_exact_at(file, pos, &mut header_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len =
        u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]) as usize;
    let stored_crc =
        u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);
    if len > MAX_FRAME_SIZE {
        return Err(WalError::Format(format!(
            "WAL frame at offset {pos} claims {len} bytes, exceeding maximum {MAX_FRAME_SIZE}"
        )));
    }
    let mut payload = vec![0u8; len];
    match read_exact_at(file, pos + FRAME_HEADER_SIZE, &mut payload) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let actual_crc = crc32fast::hash(&payload);
    if actual_crc != stored_crc {
        return Err(WalError::ChecksumMismatch {
            offset: pos,
            expected: stored_crc,
            actual: actual_crc,
        });
    }
    let entry = rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(&payload)
        .map_err(|e| WalError::Format(format!("corrupt WAL entry at byte offset {pos}: {e}")))?;
    Ok(Some((entry, pos + FRAME_HEADER_SIZE + len as u64)))
}

/// Write a single WAL frame: `[len: u32 LE][crc32: u32 LE][payload]`.
/// Returns the total bytes written (header + payload).
fn write_frame(writer: &mut BufWriter<&File>, payload: &[u8]) -> Result<u64, WalError> {
    let len: u32 = payload.len().try_into().map_err(|_| {
        WalError::Format(format!(
            "WAL frame too large: {} bytes exceeds u32 max",
            payload.len()
        ))
    })?;
    let crc = crc32fast::hash(payload);
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&crc.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(FRAME_HEADER_SIZE + payload.len() as u64)
}

/// Apply a single WAL record to a World, optionally remapping component IDs.
pub(crate) fn apply_record(
    record: &crate::record::WalRecord,
    world: &mut World,
    codecs: &CodecRegistry,
    remap: Option<&HashMap<ComponentId, ComponentId>>,
) -> Result<(), WalError> {
    // When no remap is provided, use identity mapping (same-process replay).
    // When a schema-derived remap exists, unmapped IDs are an error — the
    // sender wrote a mutation for a component not in its own preamble.
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
            SerializedMutation::SparseInsert {
                entity,
                component_id,
                data,
            } => {
                let local_id = remap_id(*component_id)?;
                let raw = codecs.deserialize(local_id, data)?;
                let layout = codecs
                    .layout(local_id)
                    .ok_or(CodecError::UnregisteredComponent(local_id))?;
                changeset.record_sparse_insert(
                    Entity::from_bits(*entity),
                    local_id,
                    raw.as_ptr(),
                    layout,
                );
            }
            SerializedMutation::SparseRemove {
                entity,
                component_id,
            } => {
                changeset
                    .record_sparse_remove(Entity::from_bits(*entity), remap_id(*component_id)?);
            }
        }
    }
    changeset.apply(world).map_err(WalError::Apply)?;
    Ok(())
}

/// Read-only snapshot of WAL statistics. Plain data struct — no references
/// to internal state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalStats {
    pub next_seq: u64,
    pub segment_count: usize,
    pub oldest_seq: Option<u64>,
    pub bytes_since_checkpoint: u64,
    pub last_checkpoint_seq: Option<u64>,
    pub checkpoint_needed: bool,
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
    last_checkpoint_seq: Option<u64>,
    bytes_since_checkpoint: u64,
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
            last_checkpoint_seq: None,
            bytes_since_checkpoint: 0,
        };
        wal.active_bytes = wal.write_segment_header()?;
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

        let (last_start_seq, last_path) = segments.last().unwrap().clone();

        // Validate magic on all sealed segments before touching the active one.
        // A legacy v1 segment must produce a hard error, not silent truncation.
        for (_, seg_path) in segments.iter().rev().skip(1) {
            let seg_file = File::open(seg_path)?;
            validate_segment_magic(&seg_file, seg_path)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&last_path)?;

        // Validate magic on the active segment. Empty/torn files pass through
        // (validate_segment_magic returns Ok on UnexpectedEof).
        validate_segment_magic(&file, &last_path)?;

        let schema = Self::build_schema(codecs);

        let mut wal = Self {
            dir: dir.to_path_buf(),
            active_file: file,
            active_start_seq: last_start_seq,
            active_bytes: 0,
            next_seq: 0,
            config,
            schema,
            last_checkpoint_seq: None,
            bytes_since_checkpoint: 0,
        };

        // Crash recovery: scan the active segment, truncating torn/corrupt tail.
        // Frame scanning starts after the segment magic header.
        let (active_last_seq, active_has) = wal.scan_active_segment()?;
        wal.active_bytes = wal.active_file.metadata()?.len();

        // If crash recovery truncated the segment to empty (or below magic
        // size), rewrite the full segment header (magic + schema preamble).
        if wal.active_bytes <= SEGMENT_MAGIC_SIZE {
            wal.active_file.set_len(0)?;
            wal.active_bytes = wal.write_segment_header()?;
        }

        if active_has {
            wal.next_seq = active_last_seq + 1;
        } else {
            // Active segment has no mutations — check earlier segments
            for (_, seg_path) in segments.iter().rev().skip(1) {
                let seg_file = File::open(seg_path)?;
                let mut pos: u64 = SEGMENT_MAGIC_SIZE;
                let mut seg_last = 0u64;
                let mut seg_has = false;
                while let Some((entry, next_pos)) = read_next_frame(&seg_file, pos)? {
                    match entry {
                        WalEntry::Mutations(record) => {
                            seg_last = record.seq;
                            seg_has = true;
                        }
                        WalEntry::Schema(_) | WalEntry::Checkpoint { .. } => {}
                    }
                    pos = next_pos;
                }
                if seg_has {
                    wal.next_seq = seg_last + 1;
                    break;
                }
            }
            // If no mutations found anywhere (e.g. all earlier segments
            // were truncated), the active segment's start_seq is the
            // minimum safe next_seq — it was assigned from next_seq at
            // rollover time, so reusing anything below it would collide
            // with already-issued sequence numbers.
            if wal.next_seq < wal.active_start_seq {
                wal.next_seq = wal.active_start_seq;
            }
        }

        // If scan_active_segment did not find a checkpoint, scan sealed
        // segments in reverse to recover the most recent one. Accumulate
        // mutation bytes between that checkpoint and the active segment
        // so bytes_since_checkpoint is accurate across segment boundaries.
        if wal.last_checkpoint_seq.is_none() {
            for (_, seg_path) in segments.iter().rev().skip(1) {
                let seg_file = File::open(seg_path)?;
                let mut pos: u64 = SEGMENT_MAGIC_SIZE;
                let mut seg_mutation_bytes: u64 = 0;
                let mut found = false;
                while let Some((entry, next_pos)) = read_next_frame(&seg_file, pos)? {
                    let frame_bytes = next_pos - pos;
                    match entry {
                        WalEntry::Checkpoint { snapshot_seq } => {
                            wal.last_checkpoint_seq = Some(snapshot_seq);
                            seg_mutation_bytes = 0;
                            found = true;
                        }
                        WalEntry::Mutations(_) => {
                            seg_mutation_bytes += frame_bytes;
                        }
                        WalEntry::Schema(_) => {}
                    }
                    pos = next_pos;
                }
                // bytes_since_checkpoint already holds the active segment's
                // mutation bytes (from scan_active_segment). Add mutation
                // bytes from this sealed segment that came after the last
                // checkpoint (or all of them if no checkpoint in this segment).
                wal.bytes_since_checkpoint += seg_mutation_bytes;
                if found {
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

    /// Returns `true` when the WAL has accumulated more bytes since the
    /// last checkpoint than `max_bytes_between_checkpoints` allows.
    pub fn checkpoint_needed(&self) -> bool {
        match self.config.max_bytes_between_checkpoints {
            Some(max) => self.bytes_since_checkpoint >= max as u64,
            None => false,
        }
    }

    /// The sequence number of the last acknowledged snapshot, if any.
    pub fn last_checkpoint_seq(&self) -> Option<u64> {
        self.last_checkpoint_seq
    }

    /// Snapshot of WAL statistics for observability.
    pub fn stats(&self) -> WalStats {
        WalStats {
            next_seq: self.next_seq(),
            segment_count: self.segment_count(),
            oldest_seq: self.oldest_seq(),
            bytes_since_checkpoint: self.bytes_since_checkpoint,
            last_checkpoint_seq: self.last_checkpoint_seq(),
            checkpoint_needed: self.checkpoint_needed(),
        }
    }

    /// Record that a snapshot was taken at the given seq.
    /// Writes a `Checkpoint` entry to the WAL stream and resets the byte counter.
    pub fn acknowledge_snapshot(&mut self, seq: u64) -> Result<(), WalError> {
        assert!(
            seq <= self.next_seq,
            "cannot checkpoint future sequence {seq}, WAL is at {}",
            self.next_seq
        );
        let entry = WalEntry::Checkpoint { snapshot_seq: seq };
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;

        let frame_bytes = {
            let mut writer = BufWriter::new(&self.active_file);
            write_frame(&mut writer, &payload)?
        };

        self.active_bytes += frame_bytes;
        self.last_checkpoint_seq = Some(seq);
        self.bytes_since_checkpoint = 0;
        Ok(())
    }

    /// Serialize and append a changeset as a WAL record.
    /// Returns the sequence number assigned to this record.
    ///
    /// If the active segment exceeds `max_segment_bytes` after the write,
    /// rollover to a new segment is attempted. Rollover failure is *not*
    /// propagated — the mutation is already durable in the current segment
    /// and the next `append` will retry the roll.
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

        let frame_bytes = {
            let mut writer = BufWriter::new(&self.active_file);
            write_frame(&mut writer, &payload)?
        };
        self.active_bytes += frame_bytes;
        self.bytes_since_checkpoint += frame_bytes;
        self.next_seq += 1;

        // Roll to new segment if threshold exceeded. Failure is non-fatal:
        // the mutation is already persisted and the oversized segment is
        // still valid. The next append will retry.
        if self.active_bytes >= self.config.max_segment_bytes as u64 {
            let _ = self.roll_segment();
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
            validate_segment_magic(&seg_file, seg_path)?;
            let mut pos: u64 = SEGMENT_MAGIC_SIZE;
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
                    WalEntry::Checkpoint { .. } => {}
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

    /// Write the segment header (magic + schema preamble) to the active
    /// segment. Returns total bytes written (magic + frame).
    fn write_segment_header(&mut self) -> Result<u64, WalError> {
        let entry = WalEntry::Schema(self.schema.clone());
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;
        let mut writer = BufWriter::new(&self.active_file);
        let magic_bytes = write_segment_magic(&mut writer)?;
        let frame_bytes = write_frame(&mut writer, &payload)?;
        Ok(magic_bytes + frame_bytes)
    }

    /// Roll to a new segment file. All I/O completes before internal state
    /// is updated so a failure leaves `self` unchanged.
    fn roll_segment(&mut self) -> Result<(), WalError> {
        let seg_path = self.dir.join(segment_filename(self.next_seq));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&seg_path)?;

        // Write segment header (magic + schema preamble) to the NEW file
        // before touching self.
        let entry = WalEntry::Schema(self.schema.clone());
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;
        let preamble_bytes = {
            let mut writer = BufWriter::new(&file);
            let magic_bytes = write_segment_magic(&mut writer)?;
            let frame_bytes = write_frame(&mut writer, &payload)?;
            magic_bytes + frame_bytes
        };

        // All I/O succeeded — atomically update state.
        self.active_file = file;
        self.active_start_seq = self.next_seq;
        self.active_bytes = preamble_bytes;
        Ok(())
    }

    /// Try to read the next entry from the active segment.
    /// On EOF, partial frame, or corrupt data, truncates the file to `pos`
    /// (crash recovery) and returns `Ok(None)`.
    fn read_next_entry(&mut self, pos: u64) -> Result<Option<(WalEntry, u64)>, WalError> {
        match read_next_frame(&self.active_file, pos) {
            Ok(Some(result)) => Ok(Some(result)),
            Ok(None) | Err(WalError::Format(_) | WalError::ChecksumMismatch { .. }) => {
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
        let mut pos: u64 = SEGMENT_MAGIC_SIZE;
        let mut bytes_after_checkpoint: u64 = 0;

        while let Some((entry, next_pos)) = self.read_next_entry(pos)? {
            let frame_bytes = next_pos - pos;
            match entry {
                WalEntry::Mutations(record) => {
                    last_seq = record.seq;
                    has_mutations = true;
                    bytes_after_checkpoint += frame_bytes;
                }
                WalEntry::Checkpoint { snapshot_seq } => {
                    self.last_checkpoint_seq = Some(snapshot_seq);
                    bytes_after_checkpoint = 0;
                }
                WalEntry::Schema(_) => {}
            }
            pos = next_pos;
        }

        self.bytes_since_checkpoint = bytes_after_checkpoint;
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
            MutationRef::SparseInsert {
                entity,
                component_id,
                data,
            } => {
                let mut buf = Vec::new();
                // SAFETY: data points to a valid component value from the Arena.
                unsafe { codecs.serialize(*component_id, data.as_ptr(), &mut buf)? };
                Ok(SerializedMutation::SparseInsert {
                    entity: entity.to_bits(),
                    component_id: *component_id,
                    data: buf,
                })
            }
            MutationRef::SparseRemove {
                entity,
                component_id,
            } => Ok(SerializedMutation::SparseRemove {
                entity: entity.to_bits(),
                component_id: *component_id,
            }),
        }
    }
}

// ── WalCursor ─────────────────────────────────────────────────────────

use crate::record::ReplicationBatch;

/// Read-only cursor over a segmented WAL directory. Opens its own file
/// handles so it can read concurrently with an active writer. Lazily
/// advances across segment files.
///
/// This is a **filesystem-specific** utility for reading WAL records from
/// local segment files. For network replication, serialize
/// [`ReplicationBatch`] on the source and transport it yourself —
/// `WalCursor` is one way to produce batches, not the only way.
pub struct WalCursor {
    dir: PathBuf,
    file: File,
    pos: u64,
    next_seq: u64,
    schema: Option<WalSchema>,
    current_segment_start_seq: u64,
}

impl WalCursor {
    /// Open a WAL directory for reading, starting from `from_seq`.
    /// Finds the segment containing `from_seq`, parses its schema preamble,
    /// and scans forward to the first record with `seq >= from_seq`.
    /// Returns `Err(CursorBehind)` if all segments start after `from_seq`.
    pub fn open(dir: &Path, from_seq: u64) -> Result<Self, WalError> {
        let segments = list_segments(dir)?;
        if segments.is_empty() {
            return Err(WalError::Format("no WAL segments found".into()));
        }

        // Find segment containing from_seq: largest start_seq <= from_seq
        let Some(seg_idx) = segments.iter().rposition(|(start, _)| *start <= from_seq) else {
            return Err(WalError::CursorBehind {
                requested: from_seq,
                oldest: segments[0].0,
            });
        };

        let (start_seq, seg_path) = &segments[seg_idx];
        let file = File::open(seg_path)?;
        validate_segment_magic(&file, seg_path)?;
        let mut pos: u64 = SEGMENT_MAGIC_SIZE;
        let mut schema = None;

        // Scan forward to from_seq
        loop {
            match read_next_frame(&file, pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    schema = Some(s);
                    pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    if record.seq >= from_seq {
                        break; // Don't advance past this record
                    }
                    pos = next_pos;
                }
                Some((WalEntry::Checkpoint { .. }, next_pos)) => {
                    pos = next_pos;
                }
                None => break,
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            file,
            pos,
            next_seq: from_seq,
            schema,
            current_segment_start_seq: *start_seq,
        })
    }

    /// Read up to `limit` records from the current position.
    /// Returns a `ReplicationBatch` with the schema and records.
    /// An empty `records` vec means the cursor has caught up.
    /// Lazily advances across segment boundaries.
    pub fn next_batch(&mut self, limit: usize) -> Result<ReplicationBatch, WalError> {
        let mut records = Vec::new();

        while records.len() < limit {
            match read_next_frame(&self.file, self.pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    self.next_seq = record.seq + 1;
                    records.push(record);
                    self.pos = next_pos;
                }
                Some((WalEntry::Checkpoint { .. }, next_pos)) => {
                    self.pos = next_pos;
                }
                None => {
                    // Try to advance to next segment
                    if !self.try_advance_segment()? {
                        break; // No more segments — caught up
                    }
                }
            }
        }

        let schema = self
            .schema
            .clone()
            .unwrap_or_else(|| WalSchema { components: vec![] });
        Ok(ReplicationBatch { schema, records })
    }

    /// Try to open the next segment file. Returns true if advanced.
    fn try_advance_segment(&mut self) -> Result<bool, WalError> {
        let segments = list_segments(&self.dir)?;
        let next = segments
            .iter()
            .find(|(start, _)| *start > self.current_segment_start_seq);

        match next {
            Some((start_seq, path)) => {
                self.file = File::open(path)?;
                validate_segment_magic(&self.file, path)?;
                self.pos = SEGMENT_MAGIC_SIZE;
                self.current_segment_start_seq = *start_seq;
                // Parse schema preamble of new segment
                if let Some((WalEntry::Schema(s), next_pos)) =
                    read_next_frame(&self.file, SEGMENT_MAGIC_SIZE)?
                {
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// The schema parsed from the WAL preamble, if present.
    pub fn schema(&self) -> Option<&WalSchema> {
        self.schema.as_ref()
    }

    /// Next expected sequence number. Useful for persisting cursor position.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
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
            max_bytes_between_checkpoints: None,
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

        cs.apply(&mut world).unwrap();
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
        cs.apply(&mut world_a).unwrap();

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
        cs.apply(&mut world_a).unwrap();

        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world_a, e, Health(50));
        cs2.remove::<Pos>(&mut world_a, e);
        wal.append(&cs2, &codecs_a).unwrap();
        cs2.apply(&mut world_a).unwrap();

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
            cs.apply(&mut world).unwrap();
        }

        assert_eq!(wal.next_seq(), 20);
        assert!(
            wal.segment_count() > 1,
            "should have rolled to multiple segments"
        );

        // Every segment should start with magic + schema preamble
        let segments = list_segments(&wal_dir).unwrap();
        for (_, seg_path) in &segments {
            let file = File::open(seg_path).unwrap();
            // First frame starts after the 4-byte segment magic.
            let (entry, _) = read_next_frame(&file, SEGMENT_MAGIC_SIZE).unwrap().unwrap();
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
                cs.apply(&mut world).unwrap();
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
            cs.apply(&mut world).unwrap();
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
            cs.apply(&mut world).unwrap();
        }

        let before = wal.segment_count();
        assert!(before > 2);

        let deleted = wal.delete_segments_before(10).unwrap();
        assert!(deleted > 0);
        assert_eq!(wal.segment_count(), before - deleted);
        assert!(wal.oldest_seq().is_some());
    }

    // ── Checkpoint tests ──────────────────────────────────────────

    #[test]
    fn wal_config_checkpoint_default_disabled() {
        let config = WalConfig::default();
        assert!(config.max_bytes_between_checkpoints.is_none());
    }

    #[test]
    fn wal_stats_reflects_state() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(1024),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        let s0 = wal.stats();
        assert_eq!(s0.next_seq, 0);
        assert_eq!(s0.segment_count, 1);
        assert_eq!(s0.oldest_seq, Some(0));
        assert_eq!(s0.bytes_since_checkpoint, 0);
        assert_eq!(s0.last_checkpoint_seq, None);
        assert!(!s0.checkpoint_needed);

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();

        let s1 = wal.stats();
        assert_eq!(s1.next_seq, 1);
        assert!(s1.bytes_since_checkpoint > 0);
    }

    #[test]
    fn checkpoint_needed_when_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        assert!(!wal.checkpoint_needed());
        assert_eq!(wal.last_checkpoint_seq(), None);
    }

    #[test]
    fn checkpoint_needed_after_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        assert!(!wal.checkpoint_needed());

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
            cs.apply(&mut world).unwrap();
        }

        assert!(wal.checkpoint_needed());
    }

    #[test]
    fn acknowledge_snapshot_writes_checkpoint_and_resets() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

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
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.checkpoint_needed());

        let seq = wal.next_seq();
        wal.acknowledge_snapshot(seq).unwrap();

        assert_eq!(wal.last_checkpoint_seq(), Some(seq));
        assert!(!wal.checkpoint_needed());
    }

    #[test]
    fn acknowledge_snapshot_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(1024),
        };

        {
            let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world).unwrap();

            wal.acknowledge_snapshot(wal.next_seq()).unwrap();
        }

        let wal2 = Wal::open(&wal_dir, &codecs, config).unwrap();
        assert_eq!(wal2.last_checkpoint_seq(), Some(1));
        assert!(!wal2.checkpoint_needed());
    }

    #[test]
    fn checkpoint_recovered_from_sealed_segment() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        // Use small segments so rollover happens quickly
        let config = WalConfig {
            max_segment_bytes: 128,
            max_bytes_between_checkpoints: Some(4096),
        };

        let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();

        // Write some records, then checkpoint
        for i in 0..3 {
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
            cs.apply(&mut world).unwrap();
        }
        let ckpt_seq = wal.next_seq();
        wal.acknowledge_snapshot(ckpt_seq).unwrap();
        assert_eq!(wal.last_checkpoint_seq(), Some(ckpt_seq));

        // Write more records to force rollover past the checkpoint's segment
        for i in 3..20 {
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
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.segment_count() > 1, "must have rolled over");
        drop(wal);

        // Reopen — checkpoint was in an earlier sealed segment
        let wal2 = Wal::open(&wal_dir, &codecs, config).unwrap();
        assert_eq!(
            wal2.last_checkpoint_seq(),
            Some(ckpt_seq),
            "checkpoint must be recovered from sealed segment"
        );
        // bytes_since_checkpoint may differ slightly due to scan granularity
        // but must be non-zero (mutations were written after checkpoint)
        assert!(
            wal2.bytes_since_checkpoint > 0,
            "bytes_since_checkpoint should count mutations after checkpoint"
        );
    }

    #[test]
    fn replay_skips_checkpoint_entries() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

        for i in 0..3 {
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
            cs.apply(&mut world).unwrap();
        }
        wal.acknowledge_snapshot(wal.next_seq()).unwrap();
        for i in 3..5 {
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
            cs.apply(&mut world).unwrap();
        }

        let mut world2 = World::new();
        codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);
        let last = wal.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 4);
        assert_eq!(world2.query::<(&Pos,)>().count(), 5);
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
            cs.apply(&mut world).unwrap();
        }

        wal.delete_segments_before(u64::MAX).unwrap();
        assert!(wal.segment_count() >= 1);

        // WAL should still be appendable
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },));
        wal.append(&cs, &codecs).unwrap();
    }

    #[test]
    fn open_after_truncate_all_does_not_reuse_seq() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let last_seq;
        {
            let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();

            // Write enough to cause rollover into multiple segments
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
                cs.apply(&mut world).unwrap();
            }
            assert!(wal.segment_count() > 1);

            // Delete all old segments, leaving only the active one
            wal.delete_segments_before(u64::MAX).unwrap();
            last_seq = wal.next_seq();
        }

        // Reopen — next_seq must not regress below active_start_seq
        let wal2 = Wal::open(&wal_dir, &codecs, small_config()).unwrap();
        assert!(
            wal2.next_seq() >= last_seq,
            "next_seq {} regressed below {} after reopen with truncated segments",
            wal2.next_seq(),
            last_seq,
        );
    }

    #[test]
    fn open_rewrites_schema_after_torn_preamble() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);
        codecs.register_as::<Health>("health", &mut world);

        // Create WAL with enough appends to roll over
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
                cs.apply(&mut world).unwrap();
            }
            assert!(wal.segment_count() > 1);
        }

        // Simulate a crash that tore the active segment's schema preamble:
        // truncate the last segment file to 0 bytes.
        let segments = list_segments(&wal_dir).unwrap();
        let (_, last_seg_path) = segments.last().unwrap();
        std::fs::write(last_seg_path, b"").unwrap();

        // Reopen — should recover and rewrite the schema preamble
        let mut wal2 = Wal::open(&wal_dir, &codecs, small_config()).unwrap();

        // Append a new record and verify the segment is self-describing
        // by replaying from a fresh process with different registration order.
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },));
        wal2.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();
        drop(wal2);

        // Open with reversed registration order to exercise remap
        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Health>("health", &mut world_b);
        codecs_b.register_as::<Pos>("pos", &mut world_b);

        let mut wal_b = Wal::open(&wal_dir, &codecs_b, small_config()).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        // The post-recovery record should have been remapped correctly
        let positions: Vec<(f32, f32)> =
            world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert!(
            positions.contains(&(99.0, 99.0)),
            "post-recovery record should be replayable with remap"
        );
    }

    // ── Sparse durability tests ──────────────────────────────────────

    #[test]
    fn sparse_insert_wal_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("sparse.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Health>(&mut world);

        // Record spawn + sparse insert in one changeset.
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
        cs.insert_sparse::<Health>(&mut world, e, Health(100));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let seq = wal.append(&cs, &codecs).unwrap();
        assert_eq!(seq, 0);
        cs.apply(&mut world).unwrap();

        // Verify sparse component is present.
        assert_eq!(world.get::<Health>(e), Some(&Health(100)));

        // Replay into a fresh world.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2);
        codecs2.register::<Health>(&mut world2);

        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        let e2 = Entity::from_bits(e.to_bits());
        assert_eq!(
            world2.get::<Health>(e2),
            Some(&Health(100)),
            "sparse component should survive WAL replay"
        );
    }

    #[test]
    fn sparse_remove_wal_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("sparse_rm.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Health>(&mut world);

        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Health>(e, Health(50));
        assert_eq!(world.get::<Health>(e), Some(&Health(50)));

        // Record sparse removal.
        let mut cs = EnumChangeSet::new();
        cs.remove_sparse::<Health>(&mut world, e);

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Health>(e), None);

        // Replay into fresh world that has the entity with sparse component.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2);
        codecs2.register::<Health>(&mut world2);

        let e2 = world2.spawn((Pos { x: 1.0, y: 2.0 },));
        world2.insert_sparse::<Health>(e2, Health(50));

        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        assert_eq!(
            world2.get::<Health>(e2),
            None,
            "sparse removal should survive WAL replay"
        );
    }

    #[test]
    fn sparse_insert_overwrite_wal_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("sparse_ow.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Health>(&mut world);

        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Health>(e, Health(10));

        // Overwrite sparse component.
        let mut cs = EnumChangeSet::new();
        cs.insert_sparse::<Health>(&mut world, e, Health(999));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Health>(e), Some(&Health(999)));

        // Replay into world with old sparse value.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2);
        codecs2.register::<Health>(&mut world2);

        let e2 = world2.spawn((Pos { x: 1.0, y: 2.0 },));
        world2.insert_sparse::<Health>(e2, Health(10));

        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        assert_eq!(
            world2.get::<Health>(e2),
            Some(&Health(999)),
            "sparse overwrite should survive WAL replay"
        );
    }

    #[test]
    fn sparse_wal_replay_sets_sparse_routing_flag() {
        // Verifies that mark_sparse is called during WAL replay so that
        // world.has() and world.get() route to sparse storage correctly,
        // even when the replay world never called insert_sparse directly.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("sparse_routing.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Health>(&mut world);

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
        cs.insert_sparse::<Health>(&mut world, e, Health(42));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();

        // Replay into fresh world — Health registered via codecs.register
        // (not register_sparse), so sparse flag only comes from mark_sparse
        // inside changeset apply.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2);
        codecs2.register::<Health>(&mut world2);

        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        let e2 = Entity::from_bits(e.to_bits());
        assert!(
            world2.has::<Health>(e2),
            "has() must route to sparse storage after WAL replay"
        );
        assert_eq!(world2.get::<Health>(e2), Some(&Health(42)));

        // Also verify dense component survived.
        assert_eq!(
            world2.get::<Pos>(e2),
            Some(&Pos { x: 1.0, y: 2.0 }),
            "dense component from same changeset should also survive"
        );
    }

    // ── CRC32 checksum tests ──────────────────────────────────────────

    #[test]
    fn checksum_mismatch_detected_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("crc.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        let seg_path = wal_dir.join(segment_filename(0));
        let file_len_before = std::fs::metadata(&seg_path).unwrap().len();

        // Append a frame with valid length and valid-sized payload, but wrong CRC.
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            let payload = [0xDE; 32];
            let wrong_crc: u32 = 0xDEADBEEF;
            f.write_all(&32u32.to_le_bytes()).unwrap(); // len
            f.write_all(&wrong_crc.to_le_bytes()).unwrap(); // wrong CRC
            f.write_all(&payload).unwrap(); // payload
            f.flush().unwrap();
        }

        let new_len = std::fs::metadata(&seg_path).unwrap().len();
        assert!(new_len > file_len_before);

        // Open should detect checksum mismatch and truncate the corrupt frame.
        let wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal2.next_seq(), 2);

        let after_len = std::fs::metadata(&seg_path).unwrap().len();
        assert_eq!(
            after_len, file_len_before,
            "corrupt frame should be truncated"
        );
    }

    #[test]
    fn checksum_mismatch_detected_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("crc_replay.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world);

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs).unwrap();
        }

        let seg_path = wal_dir.join(segment_filename(0));
        let file_len = std::fs::metadata(&seg_path).unwrap().len();

        // Append frame with wrong CRC.
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            let payload = [0xAB; 24];
            let wrong_crc: u32 = 0xCAFEBABE;
            f.write_all(&24u32.to_le_bytes()).unwrap();
            f.write_all(&wrong_crc.to_le_bytes()).unwrap();
            f.write_all(&payload).unwrap();
            f.flush().unwrap();
        }

        let mut wal2 = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        let mut world2 = World::new();
        let last = wal2.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 0, "should replay the one valid record");

        let after_len = std::fs::metadata(&seg_path).unwrap().len();
        assert_eq!(after_len, file_len);
    }

    #[test]
    fn valid_frames_pass_checksum() {
        // End-to-end: write frames and read them back — CRC must match.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("valid_crc.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

        for i in 0..5 {
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
            cs.apply(&mut world).unwrap();
        }

        // Replay should succeed with no checksum errors.
        let mut world2 = World::new();
        codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);
        let last = wal.replay(&mut world2, &codecs).unwrap();
        assert_eq!(last, 4);
        assert_eq!(world2.query::<(&Pos,)>().count(), 5);
    }

    #[test]
    fn frame_header_size_is_eight() {
        assert_eq!(FRAME_HEADER_SIZE, 8);
    }

    // ── Legacy v1 format detection tests ─────────────────────────────

    #[test]
    fn legacy_v1_segment_detected_on_open() {
        // Simulate a legacy v1 segment: [len: u32 LE][payload] with no magic.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("legacy.wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Write a fake v1 segment: starts with a u32 length (no "MKW2" magic).
        let seg_path = wal_dir.join(segment_filename(0));
        {
            use std::io::Write;
            let mut f = File::create(&seg_path).unwrap();
            // Write a plausible v1 frame: [len=100][100 bytes of data]
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&[0u8; 100]).unwrap();
            f.flush().unwrap();
        }

        let codecs = CodecRegistry::new();
        let result = Wal::open(&wal_dir, &codecs, default_config());
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("legacy segment should produce an error"),
        };
        assert!(
            msg.contains("legacy v1 format"),
            "error should mention legacy format: {msg}"
        );
    }

    #[test]
    fn legacy_v1_segment_detected_on_replay() {
        // Create a valid v2 WAL, then corrupt one sealed segment to look like v1.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("legacy_replay.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();

        // Write enough to create multiple segments.
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
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.segment_count() > 1);

        // Overwrite the first segment's magic with garbage to simulate v1.
        let segments = list_segments(&wal_dir).unwrap();
        let (_, first_seg_path) = &segments[0];
        {
            use std::io::Write;
            let mut f = OpenOptions::new().write(true).open(first_seg_path).unwrap();
            // Overwrite the 4-byte magic with a v1-style length prefix.
            f.write_all(&50u32.to_le_bytes()).unwrap();
            f.flush().unwrap();
        }

        // Replay should detect the corrupted segment magic and error.
        let mut world2 = World::new();
        codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);
        let result = wal.replay(&mut world2, &codecs);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("legacy v1 format"),
            "replay should detect legacy format: {msg}"
        );
    }

    #[test]
    fn legacy_v1_segment_detected_on_cursor_open() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("legacy_cursor.wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Write a fake v1 segment.
        let seg_path = wal_dir.join(segment_filename(0));
        {
            use std::io::Write;
            let mut f = File::create(&seg_path).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&[0u8; 100]).unwrap();
            f.flush().unwrap();
        }

        let result = WalCursor::open(&wal_dir, 0);
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("cursor should produce an error for legacy segment"),
        };
        assert!(
            msg.contains("legacy v1 format"),
            "cursor should detect legacy format: {msg}"
        );
    }

    #[test]
    fn v2_segment_magic_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("v2_magic.wal");

        let codecs = CodecRegistry::new();

        let _wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

        let seg_path = wal_dir.join(segment_filename(0));
        let data = std::fs::read(&seg_path).unwrap();
        assert!(data.len() >= 4);
        assert_eq!(&data[0..4], b"MKW2", "segment must start with v2 magic");
    }
}
