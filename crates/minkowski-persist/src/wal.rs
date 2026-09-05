use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64 as StdAtomicU64;

use minkowski::{ComponentId, Entity, EnumChangeSet, MutationRef, World};

use crate::record::{ComponentSchema, SerializedMutation, WalEntry, WalSchema};
use minkowski_lsm::codec::{CodecError, CodecRegistry, CrcProof};

mod ingest;
mod plan;
mod range;
pub use ingest::{IngestError, JournaledFollower};
use plan::ValidatedRange;
pub use range::{WalFrameRange, WalRangeLimits, WalSegmentRun};

// WAL segment format (v2):
//   [segment_magic: 4 bytes "MKW3"]
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
const FRAME_HEADER_SIZE: u64 = 16; // [len: u32 LE][crc32: u32 LE][view: u64 LE]

/// Segment file magic identifying v2 format with CRC32 checksums.
const SEGMENT_MAGIC: [u8; 4] = *b"MKW3";

/// Size of the segment magic header in bytes.
const SEGMENT_MAGIC_SIZE: u64 = 4;

/// Read exactly `buf.len()` bytes from `file` starting at byte offset `pos`.
fn read_exact_at(file: &File, pos: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut f = file;
    f.seek(SeekFrom::Start(pos))?;
    f.read_exact(buf)
}

// create_dir_all may have created more than the immediate WAL directory.
fn sync_directory_ancestry(dir: &Path) -> io::Result<()> {
    for ancestor in dir.canonicalize()?.ancestors() {
        File::open(ancestor)?.sync_all()?;
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error(
        "tick regression at seq {seq}: record tick {record_tick} is below world tick {world_tick}"
    )]
    TickRegression {
        seq: u64,
        record_tick: u64,
        world_tick: u64,
    },
    #[error("WAL I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("WAL codec error: {0}")]
    Codec(#[from] CodecError),
    #[error("WAL format error: {0}")]
    Format(String),
    #[error(
        "WAL checksum mismatch at byte offset {offset}: expected {expected:#010x}, got {actual:#010x}"
    )]
    ChecksumMismatch {
        offset: u64,
        expected: u32,
        actual: u32,
    },
    #[error("cursor behind: requested seq {requested} but oldest available is {oldest}")]
    CursorBehind { requested: u64, oldest: u64 },
    #[error("range request {requested} is past durable tail {durable_tail}")]
    RangeAhead { requested: u64, durable_tail: u64 },
    #[error("range limits must all be positive")]
    InvalidRangeLimits,
    #[error("range limits cannot fit a mutation and its schema context")]
    RangeLimitTooSmall,
    #[error("retained WAL starts at {oldest}; prefix fence context requires rejoin")]
    MissingFenceContext { oldest: u64 },
    #[error("unresolved WAL history at seq {seq}: {reason}")]
    UnresolvedHistory { seq: u64, reason: &'static str },
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

/// Raise the shared storage fence and report whether this frame is stale.
fn observe_view(max_view: &mut u64, view: u64) -> bool {
    let stale = view < *max_view;
    *max_view = (*max_view).max(view);
    stale
}

/// Highest frame view stamped anywhere in the WAL directory. Reads only the
/// 16-byte frame headers (view lives in the header; payloads are skipped via
/// the length field). Sealed and active segments both count — the live view
/// counter must resume at or above every view the log has ever carried.
fn scan_max_view<'a, I>(segments: I) -> Result<u64, WalError>
where
    I: IntoIterator<Item = &'a (u64, PathBuf)>,
{
    let mut max_view: u64 = 0;
    for (_, seg_path) in segments {
        let file = File::open(seg_path)?;
        let mut pos: u64 = SEGMENT_MAGIC_SIZE;
        loop {
            let mut header_buf = [0u8; FRAME_HEADER_SIZE as usize];
            match read_exact_at(&file, pos, &mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len =
                u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]])
                    as u64;
            if len > MAX_FRAME_SIZE as u64 {
                break; // torn tail; scan_active_segment handles truncation
            }
            let view = u64::from_le_bytes([
                header_buf[8],
                header_buf[9],
                header_buf[10],
                header_buf[11],
                header_buf[12],
                header_buf[13],
                header_buf[14],
                header_buf[15],
            ]);
            observe_view(&mut max_view, view);
            pos += FRAME_HEADER_SIZE + len;
        }
    }
    Ok(max_view)
}

/// Original frame bytes, owned until the replay plan is consumed.
struct RawFrame {
    offset: u64,
    header: [u8; FRAME_HEADER_SIZE as usize],
    payload: rkyv::util::AlignedVec<16>,
}

impl RawFrame {
    fn read(file: &File, offset: u64) -> Result<Option<Self>, WalError> {
        let mut reader = file;
        reader.seek(SeekFrom::Start(offset))?;
        Self::read_from(&mut reader, offset)
    }

    fn read_from(reader: &mut impl Read, offset: u64) -> Result<Option<Self>, WalError> {
        let mut header = [0u8; FRAME_HEADER_SIZE as usize];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        if len > MAX_FRAME_SIZE {
            return Err(WalError::Format(format!(
                "WAL frame at offset {offset} claims {len} bytes, exceeding maximum {MAX_FRAME_SIZE}"
            )));
        }
        // Vec<u8> guarantees only byte alignment; rkyv validates in place.
        let mut payload = rkyv::util::AlignedVec::<16>::with_capacity(len);
        payload.resize(len, 0);
        match reader.read_exact(&mut payload) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        Ok(Some(Self {
            offset,
            header,
            payload,
        }))
    }

    fn verify(&self) -> Result<CrcProof, WalError> {
        let stored_crc = u32::from_le_bytes(self.header[4..8].try_into().unwrap());
        CrcProof::verify(&self.payload, stored_crc).ok_or(WalError::ChecksumMismatch {
            offset: self.offset,
            expected: stored_crc,
            actual: crc32fast::hash(&self.payload),
        })
    }

    fn decode(&self) -> Result<(WalEntry, CrcProof, u64), WalError> {
        let proof = self.verify()?;
        let view = u64::from_le_bytes(self.header[8..].try_into().unwrap());
        let entry =
            rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(&self.payload).map_err(|e| {
                WalError::Format(format!(
                    "corrupt WAL entry at byte offset {}: {e}",
                    self.offset
                ))
            })?;
        Ok((entry, proof, view))
    }

    fn next_offset(&self) -> u64 {
        self.offset + FRAME_HEADER_SIZE + self.payload.len() as u64
    }

    fn append_to(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.header);
        bytes.extend_from_slice(&self.payload);
    }

    /// Read a complete frame from a bounded buffer. Check the claimed size
    /// before allocating aligned storage; short network buffers are errors.
    fn read_bytes(bytes: &mut &[u8], offset: u64) -> Result<Self, WalError> {
        let header = bytes.get(..FRAME_HEADER_SIZE as usize).ok_or_else(|| {
            WalError::Format(format!("truncated frame header at offset {offset}"))
        })?;
        let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        if len > bytes.len() - FRAME_HEADER_SIZE as usize {
            return Err(WalError::Format(format!(
                "truncated frame payload at offset {offset}"
            )));
        }
        Self::read_from(bytes, offset)?
            .ok_or_else(|| WalError::Format(format!("truncated frame at offset {offset}")))
    }
}

/// Read one frame without truncating. A partial frame returns `None`; corrupt
/// payloads, checksums, and oversized frames return errors.
pub(crate) fn read_next_frame(
    file: &File,
    pos: u64,
) -> Result<Option<(WalEntry, u64, CrcProof, u64)>, WalError> {
    let Some(frame) = RawFrame::read(file, pos)? else {
        return Ok(None);
    };
    let (entry, proof, view) = frame.decode()?;
    Ok(Some((entry, frame.next_offset(), proof, view)))
}

/// Write a single WAL frame: `[len: u32 LE][crc32: u32 LE][view: u64 LE][payload]`.
/// Returns the total bytes written (header + payload).
fn write_frame(writer: &mut impl Write, payload: &[u8], view: u64) -> Result<u64, WalError> {
    let len: u32 = payload.len().try_into().map_err(|_| {
        WalError::Format(format!(
            "WAL frame too large: {} bytes exceeds u32 max",
            payload.len()
        ))
    })?;
    let crc = crc32fast::hash(payload);
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&crc.to_le_bytes())?;
    writer.write_all(&view.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(FRAME_HEADER_SIZE + payload.len() as u64)
}

/// A source component resolves separately to a decoder and a world column.
#[derive(Clone, Copy)]
pub(crate) struct ApplyComponent {
    codec_id: ComponentId,
    world_id: Option<ComponentId>,
}

pub(crate) type ApplyRemap = HashMap<ComponentId, ApplyComponent>;

/// Resolve a schema against both the codec registry and the destination world.
/// This does not register components or otherwise mutate the destination.
pub(crate) fn build_apply_remap(
    schema: Option<&[ComponentSchema]>,
    world: &World,
    codecs: &CodecRegistry,
) -> Result<ApplyRemap, WalError> {
    let source_to_codec = match schema {
        Some(schema) => {
            let remap = codecs.build_remap(schema)?;
            if remap.len() != schema.len()
                || remap.values().collect::<HashSet<_>>().len() != schema.len()
            {
                return Err(WalError::Format(
                    "duplicate component ID or name in WAL schema".into(),
                ));
            }
            remap
        }
        None => codecs
            .registered_ids()
            .into_iter()
            .map(|id| (id, id))
            .collect(),
    };
    let world_ids: HashMap<_, _> = (0..world.component_count())
        .filter_map(|id| {
            let type_id = world.component_type_id(id)?;
            Some((codecs.stable_name_by_type(type_id)?, id))
        })
        .collect();
    source_to_codec
        .into_iter()
        .map(|(source_id, codec_id)| {
            let name = codecs
                .stable_name(codec_id)
                .ok_or(CodecError::UnregisteredComponent(codec_id))?;
            // A segment schema can declare types that no retained record
            // uses. Require a destination column only when a record uses it.
            let world_id = world_ids.get(name).copied();
            Ok((source_id, ApplyComponent { codec_id, world_id }))
        })
        .collect()
}

/// Validate the commit boundary and return the tick after successful apply.
/// EnumChangeSet::apply_replay advances once, including for empty records.
fn validate_record_tick(
    record: &crate::record::WalRecord,
    world_tick: u64,
) -> Result<u64, WalError> {
    if record.tick_after < world_tick {
        return Err(WalError::TickRegression {
            seq: record.seq,
            record_tick: record.tick_after,
            world_tick,
        });
    }
    record
        .tick_after
        .checked_add(1)
        .ok_or_else(|| WalError::Format(format!("tick overflow at seq {}", record.seq)))
}

/// Apply a single WAL record using destination-specific component bindings.
pub(crate) fn apply_record(
    record: &crate::record::WalRecord,
    world: &mut World,
    codecs: &CodecRegistry,
    remap: Option<&ApplyRemap>,
    proof: Option<&CrcProof>,
) -> Result<(), WalError> {
    // INV-1: preflight and every application path share the same tick guard.
    validate_record_tick(record, world.current_tick())?;
    // Legacy schema-less batches use codec IDs as source IDs. They still
    // resolve the destination by type instead of assuming numeric identity.
    let fallback;
    let remap = match remap {
        Some(remap) => remap,
        None => {
            fallback = build_apply_remap(None, world, codecs)?;
            &fallback
        }
    };
    let remap_id = |id| resolve_apply_component(remap, codecs, id);

    let mut changeset = EnumChangeSet::new();
    for mutation in &record.mutations {
        match mutation {
            SerializedMutation::Spawn { entity, components } => {
                let entity = Entity::from_bits(*entity);
                let mut raw_components: Vec<(minkowski::ComponentId, Vec<u8>, std::alloc::Layout)> =
                    Vec::new();
                for (comp_id, data) in components {
                    let (codec_id, world_id) = remap_id(*comp_id)?;
                    let raw = codecs.decode(codec_id, data, proof)?;
                    let layout = codecs
                        .layout(codec_id)
                        .ok_or(CodecError::UnregisteredComponent(codec_id))?;
                    raw_components.push((world_id, raw, layout));
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
                let (codec_id, world_id) = remap_id(*component_id)?;
                let raw = codecs.decode(codec_id, data, proof)?;
                let layout = codecs
                    .layout(codec_id)
                    .ok_or(CodecError::UnregisteredComponent(codec_id))?;
                changeset.record_insert(Entity::from_bits(*entity), world_id, raw.as_ptr(), layout);
            }
            SerializedMutation::Remove {
                entity,
                component_id,
            } => {
                changeset.record_remove(Entity::from_bits(*entity), remap_id(*component_id)?.1);
            }
            SerializedMutation::SparseInsert {
                entity,
                component_id,
                data,
            } => {
                let (codec_id, world_id) = remap_id(*component_id)?;
                let raw = codecs.decode(codec_id, data, proof)?;
                let layout = codecs
                    .layout(codec_id)
                    .ok_or(CodecError::UnregisteredComponent(codec_id))?;
                changeset.record_sparse_insert(
                    Entity::from_bits(*entity),
                    world_id,
                    raw.as_ptr(),
                    layout,
                );
            }
            SerializedMutation::SparseRemove {
                entity,
                component_id,
            } => {
                changeset
                    .record_sparse_remove(Entity::from_bits(*entity), remap_id(*component_id)?.1);
            }
        }
    }
    world.set_current_tick(record.tick_after);
    changeset.apply_replay(world).map_err(WalError::Apply)?;
    Ok(())
}

fn resolve_apply_component(
    remap: &ApplyRemap,
    codecs: &CodecRegistry,
    id: ComponentId,
) -> Result<(ComponentId, ComponentId), WalError> {
    let binding = remap
        .get(&id)
        .ok_or(CodecError::UnregisteredComponent(id))?;
    let world_id = binding.world_id.ok_or_else(|| {
        WalError::Format(format!(
            "component '{}' is not registered in the destination world",
            codecs.stable_name(binding.codec_id).unwrap_or("unknown"),
        ))
    })?;
    Ok((binding.codec_id, world_id))
}

fn validate_record_components(
    record: &crate::record::WalRecord,
    remap: &ApplyRemap,
    codecs: &CodecRegistry,
) -> Result<(), WalError> {
    for mutation in &record.mutations {
        match mutation {
            SerializedMutation::Spawn { components, .. } => {
                for (id, _) in components {
                    resolve_apply_component(remap, codecs, *id)?;
                }
            }
            SerializedMutation::Insert { component_id, .. }
            | SerializedMutation::Remove { component_id, .. }
            | SerializedMutation::SparseInsert { component_id, .. }
            | SerializedMutation::SparseRemove { component_id, .. } => {
                resolve_apply_component(remap, codecs, *component_id)?;
            }
            SerializedMutation::Despawn { .. } => {}
        }
    }
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
    /// Failed segment-rollover attempts since WAL creation. A rising value
    /// means appends are going to an over-budget segment (data is safe — the
    /// frame is durable before the roll attempt — but the segment grows past
    /// `max_segment_bytes` until a roll succeeds).
    pub roll_failures: u64,
}

/// Monotonic view counter for frame stamping (stage 4.0 substrate, INV-2).
/// Starts at 0. Real view installation (quorum certificates) comes with 4.1;
/// today the counter only moves forward via [`Views::bump`].
pub struct Views {
    current: StdAtomicU64,
}

impl Views {
    fn new() -> Self {
        Self {
            current: StdAtomicU64::new(0),
        }
    }

    fn with_current(value: u64) -> Self {
        Self {
            current: StdAtomicU64::new(value),
        }
    }

    /// Current view. Frames are stamped with this value on write.
    pub fn current(&self) -> u64 {
        self.current.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Advance to the next view. Returns the new view.
    pub fn bump(&self) -> u64 {
        self.current
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            + 1
    }
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
    // Published together only after fsync. Control frames also need byte bounds.
    durable_next_seq: u64,
    durable_ends: BTreeMap<u64, u64>,
    config: WalConfig,
    schema: WalSchema,
    last_checkpoint_seq: Option<u64>,
    bytes_since_checkpoint: u64,
    roll_failures: u64,
    // An uncommitted rollover must be durably removed before later writes.
    pending_roll_cleanup: Option<PathBuf>,
    /// View counter for frame stamping (INV-2).
    pub views: Views,
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
            durable_next_seq: 0,
            durable_ends: BTreeMap::new(),
            config,
            schema,
            last_checkpoint_seq: None,
            bytes_since_checkpoint: 0,
            roll_failures: 0,
            pending_roll_cleanup: None,
            views: Views::new(),
        };
        wal.active_bytes = wal.write_segment_header()?;
        wal.active_file.sync_all()?;
        sync_directory_ancestry(dir)?;
        wal.publish_active_prefix();
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
            durable_next_seq: 0,
            durable_ends: BTreeMap::new(),
            config,
            schema,
            last_checkpoint_seq: None,
            bytes_since_checkpoint: 0,
            roll_failures: 0,
            pending_roll_cleanup: None,
            views: Views::new(),
        };

        // Crash recovery: scan the active segment, truncating torn/corrupt
        // tail. Frame scanning starts after the segment magic header.
        let sealed_max_view = scan_max_view(segments.iter().rev().skip(1))?;
        let (active_last_seq, active_has) = wal.scan_active_segment(sealed_max_view)?;

        // INV-2: the live view counter must resume at the highest view the
        // log has ever stamped — before any header rewrite, so a rewritten
        // preamble stamps the recovered view, not 0.
        wal.views = Views::with_current(scan_max_view(&segments)?);

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
                while let Some((entry, next_pos, _proof, _view)) = read_next_frame(&seg_file, pos)?
                {
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
                while let Some((entry, next_pos, _proof, _view)) = read_next_frame(&seg_file, pos)?
                {
                    let frame_bytes = next_pos - pos;
                    match entry {
                        WalEntry::Checkpoint { flush_seq } => {
                            wal.last_checkpoint_seq = Some(flush_seq);
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

        // Recovered bytes may have survived only in the page cache. Make both
        // recovery truncation and retained segments durable before serving them.
        for (start, path) in &segments {
            let file = File::open(path)?;
            file.sync_all()?;
            wal.durable_ends.insert(*start, file.metadata()?.len());
        }
        sync_directory_ancestry(dir)?;
        wal.durable_next_seq = wal.next_seq;
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
            next_seq: self.next_seq,
            segment_count: list_segments(&self.dir).map_or(0, |s| s.len()),
            oldest_seq: list_segments(&self.dir)
                .ok()
                .and_then(|s| s.first().map(|(seq, _)| *seq)),
            bytes_since_checkpoint: self.bytes_since_checkpoint,
            last_checkpoint_seq: self.last_checkpoint_seq(),
            checkpoint_needed: self.checkpoint_needed(),
            roll_failures: self.roll_failures,
        }
    }

    /// Record that a snapshot was taken at the given seq.
    /// Writes and synchronizes a `Checkpoint` entry, then publishes its byte
    /// endpoint for range reads. Resets the checkpoint byte counter.
    pub fn acknowledge_flush(&mut self, seq: u64) -> Result<(), WalError> {
        self.cleanup_pending_roll()?;
        assert!(
            seq <= self.next_seq,
            "cannot checkpoint future sequence {seq}, WAL is at {}",
            self.next_seq
        );
        let entry = WalEntry::Checkpoint { flush_seq: seq };
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;

        let frame_bytes = {
            let mut writer = BufWriter::new(&self.active_file);
            write_frame(&mut writer, &payload, self.views.current())?
        };

        self.active_bytes += frame_bytes;
        self.last_checkpoint_seq = Some(seq);
        self.bytes_since_checkpoint = 0;
        self.sync_active_prefix()
    }

    /// Serialize and append a changeset as a WAL record.
    /// Returns the sequence number assigned to this record.
    ///
    /// If the active segment exceeds `max_segment_bytes` after the write,
    /// rollover to a new segment is attempted. Rollover failure is *not*
    /// propagated — the mutation is already durable in the current segment
    /// and the next `append` will retry the roll. If removing a failed segment
    /// also fails, subsequent writes first retry that cleanup and return its
    /// error without writing until cleanup succeeds.
    pub fn append(
        &mut self,
        changeset: &EnumChangeSet,
        codecs: &CodecRegistry,
        tick_after: u64,
    ) -> Result<u64, WalError> {
        self.cleanup_pending_roll()?;
        let seq = self.next_seq;
        let record = Self::changeset_to_record(seq, changeset, codecs, tick_after)?;
        let entry = WalEntry::Mutations(record);
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|e| WalError::Format(e.to_string()))?;

        let frame_bytes = {
            let mut writer = BufWriter::new(&self.active_file);
            write_frame(&mut writer, &payload, self.views.current())?
        };

        // Advance in-memory state *before* fsync.  Once bytes have been
        // handed to the kernel via write(2), they may survive a crash even
        // if fsync reports failure (POSIX allows this).  If we kept the old
        // counters and the caller retried, the same sequence number would
        // appear twice in one segment — silent corruption that causes
        // WalCursor to skip records.  A gap (missing seq) is detectable on
        // replay and therefore the lesser evil.
        //
        // This ordering only matters for callers that continue after a sync
        // error: `Durable` treats a WAL error as fatal and never retries on
        // the same handle, and a restart reconstructs `next_seq` from the
        // durable segment bytes — so the in-memory gap heals naturally.
        self.active_bytes += frame_bytes;
        self.bytes_since_checkpoint += frame_bytes;
        self.next_seq += 1;

        // Durability: flush kernel buffers to stable storage so that a
        // process crash or power loss cannot lose the frame we just wrote.
        // Without this, BufWriter::flush only pushes to the page cache.
        self.sync_active_prefix()?;

        // Roll to new segment if threshold exceeded. Failure is non-fatal:
        // the mutation is already persisted and the oversized segment is
        // still valid. The next append will retry.
        if self.active_bytes >= self.config.max_segment_bytes as u64 && self.roll_segment().is_err()
        {
            // Non-fatal (frame already durable; retry on next append) but
            // observable: operators can watch this counter for a segment
            // stuck over budget.
            self.roll_failures += 1;
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
    ///
    /// Preflights the selected raw frames and all component references before
    /// changing the world. A private plan then applies one record at a time,
    /// preserving each record's tick and mutation order. Component payload
    /// decoding and state-dependent application can still fail during execution.
    ///
    /// # Error Recovery
    ///
    /// If replay fails (codec error, dead entity, corrupt frame), the World
    /// may be in a partially-applied state. Callers should discard the World
    /// and rebuild from the last known-good snapshot. This matches the WAL
    /// error classification: replay failure is fatal, not operational.
    ///
    /// On error, the returned `WalError` does not indicate how far replay
    /// progressed. The World should be discarded entirely.
    pub fn replay_from(
        &mut self,
        from_seq: u64,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, WalError> {
        ValidatedRange::read(&self.dir, from_seq, world, codecs)?
            .execute()
            .map(|(last_seq, _view)| last_seq)
    }

    /// Delete all segment files whose entire seq range is before `seq`.
    /// A segment is safe to delete if the next segment's start_seq <= `seq`.
    /// The active (last) segment is never deleted.
    /// Returns the number of segments deleted.
    pub fn delete_segments_before(&mut self, seq: u64) -> Result<usize, WalError> {
        self.cleanup_pending_roll()?;
        let segments = list_segments(&self.dir)?;
        if segments.len() <= 1 {
            return Ok(0);
        }

        let mut deleted = 0;
        for i in 0..segments.len() - 1 {
            let next_start = segments[i + 1].0;
            if next_start <= seq {
                std::fs::remove_file(&segments[i].1)?;
                self.durable_ends.remove(&segments[i].0);
                deleted += 1;
            } else {
                break; // segments are sorted, no point continuing
            }
        }

        Ok(deleted)
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn publish_active_prefix(&mut self) {
        self.durable_ends
            .insert(self.active_start_seq, self.active_bytes);
        self.durable_next_seq = self.next_seq;
    }

    fn sync_active_prefix(&mut self) -> Result<(), WalError> {
        self.active_file.sync_all()?;
        self.publish_active_prefix();
        Ok(())
    }

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
        let frame_bytes = write_frame(&mut writer, &payload, self.views.current())?;
        Ok(magic_bytes + frame_bytes)
    }

    /// Remove an uncommitted rollover before the old segment can grow past its
    /// start sequence. Keep the pending path until deletion is durable, so an
    /// unlink or directory-sync failure blocks subsequent writes for retry.
    fn cleanup_pending_roll(&mut self) -> Result<(), WalError> {
        if let Some(path) = &self.pending_roll_cleanup {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
            File::open(&self.dir)?.sync_all()?;
            self.pending_roll_cleanup = None;
        }
        Ok(())
    }

    /// Roll to a new segment file. Publish it only after synchronization;
    /// remove it on failure, retaining the current active segment.
    fn roll_segment(&mut self) -> Result<(), WalError> {
        self.roll_segment_with_sync(File::sync_all)
    }

    fn roll_segment_with_sync(
        &mut self,
        mut sync: impl FnMut(&File) -> io::Result<()>,
    ) -> Result<(), WalError> {
        self.cleanup_pending_roll()?;
        let seg_path = self.dir.join(segment_filename(self.next_seq));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&seg_path)?;
        self.pending_roll_cleanup = Some(seg_path);

        // Write segment header (magic + schema preamble) to the NEW file
        // before replacing the active segment. Every post-create error uses
        // the same cleanup path, including serialization and partial writes.
        let prepared = (|| -> Result<u64, WalError> {
            let entry = WalEntry::Schema(self.schema.clone());
            let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
                .map_err(|e| WalError::Format(e.to_string()))?;
            let preamble_bytes = {
                let mut writer = BufWriter::new(&file);
                let magic_bytes = write_segment_magic(&mut writer)?;
                let frame_bytes = write_frame(&mut writer, &payload, self.views.current())?;
                magic_bytes + frame_bytes
            };
            sync(&file)?;
            sync(&File::open(&self.dir)?)?;
            Ok(preamble_bytes)
        })();
        let preamble_bytes = match prepared {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(file);
                self.cleanup_pending_roll()?;
                return Err(error);
            }
        };

        // All I/O succeeded — atomically update state.
        self.pending_roll_cleanup = None;
        self.active_file = file;
        self.active_start_seq = self.next_seq;
        self.active_bytes = preamble_bytes;
        self.publish_active_prefix();
        Ok(())
    }

    /// Try to read the next entry from the active segment.
    /// On EOF, partial frame, or corrupt data, truncates the file to `pos`
    /// (crash recovery) and returns `Ok(None)`.
    fn read_next_entry(&mut self, pos: u64) -> Result<Option<(WalEntry, u64, u64)>, WalError> {
        match read_next_frame(&self.active_file, pos) {
            Ok(Some((entry, next_pos, _proof, view))) => Ok(Some((entry, next_pos, view))),
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
    /// Scan the active segment for crash recovery. Truncates torn/corrupt
    /// tail. A frame stamped with a view older than the newest view known
    /// for the directory (sealed views seeded via `sealed_max_view`, plus
    /// the segment's own newer frames) is a deposed leader's late write —
    /// the tail is truncated at it so the stale bytes cannot collide with
    /// new-leader writes at the same offsets. Returns `(last_seq, has)`.
    fn scan_active_segment(&mut self, sealed_max_view: u64) -> Result<(u64, bool), WalError> {
        let mut last_seq = 0u64;
        let mut has_mutations = false;
        let mut max_view: u64 = sealed_max_view;
        let mut pos: u64 = SEGMENT_MAGIC_SIZE;
        let mut bytes_after_checkpoint: u64 = 0;

        while let Some((entry, next_pos, view)) = self.read_next_entry(pos)? {
            let frame_bytes = next_pos - pos;
            if observe_view(&mut max_view, view) {
                // Stale-view tail: deposed leader's late write. Truncate it
                // and everything after; new-leader records reuse the space.
                self.active_file.set_len(pos)?;
                break;
            }
            match entry {
                WalEntry::Mutations(record) => {
                    last_seq = record.seq;
                    has_mutations = true;
                    bytes_after_checkpoint += frame_bytes;
                }
                WalEntry::Checkpoint { flush_seq } => {
                    self.last_checkpoint_seq = Some(flush_seq);
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
        tick_after: u64,
    ) -> Result<crate::record::WalRecord, WalError> {
        let mut mutations = Vec::new();
        for m in changeset.iter_mutations() {
            mutations.push(Self::serialize_mutation(&m, codecs)?);
        }
        Ok(crate::record::WalRecord {
            seq,
            mutations,
            tick_after,
        })
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
    /// Highest frame view seen so far. Frames stamped with an older view
    /// are from a deposed leader and are dropped, matching `replay_from`.
    max_view_seen: u64,
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

        // Scan forward to from_seq. Seeded view state matters: frames skipped
        // here still establish the segment's newest view, so a later stale
        // frame cannot slip through. Earlier segments seed the fence too —
        // a checkpoint or mutation in a view newer than the resume point's
        // segment fences stale frames from the first record shipped.
        let mut max_view_seen = scan_max_view(&segments[..seg_idx])?;
        loop {
            match read_next_frame(&file, pos)? {
                Some((WalEntry::Schema(s), next_pos, _proof, view)) => {
                    observe_view(&mut max_view_seen, view);
                    schema = Some(s);
                    pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos, _proof, view)) => {
                    observe_view(&mut max_view_seen, view);
                    if record.seq >= from_seq {
                        break; // Don't advance past this record
                    }
                    pos = next_pos;
                }
                Some((WalEntry::Checkpoint { .. }, next_pos, _proof, view)) => {
                    observe_view(&mut max_view_seen, view);
                    pos = next_pos;
                }
                None => break,
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            file,
            pos,
            max_view_seen,
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
                Some((WalEntry::Schema(s), next_pos, _proof, view)) => {
                    observe_view(&mut self.max_view_seen, view);
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos, _proof, view)) => {
                    if observe_view(&mut self.max_view_seen, view) {
                        // Stale-view frame: deposed leader's late write. The
                        // follower must never see it (P1: fence through
                        // replication, not just local replay).
                        self.pos = next_pos;
                        continue;
                    }
                    self.next_seq = record.seq + 1;
                    records.push(record);
                    self.pos = next_pos;
                }
                Some((WalEntry::Checkpoint { .. }, next_pos, _proof, view)) => {
                    observe_view(&mut self.max_view_seen, view);
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
                // Parse schema preamble of new segment; its view seeds the
                // fence for the segment's data frames.
                if let Some((WalEntry::Schema(s), next_pos, _proof, view)) =
                    read_next_frame(&self.file, SEGMENT_MAGIC_SIZE)?
                {
                    observe_view(&mut self.max_view_seen, view);
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski_lsm::codec::CodecRegistry;
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
    fn replay_preflight_validates_post_apply_ticks() {
        for (tick, next_tick) in [(5, 5), (5, 6), (u64::MAX - 1, u64::MAX)] {
            let dir = tempfile::tempdir().unwrap();
            let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
            let mut codecs = CodecRegistry::new();
            codecs.register_as::<Pos>("pos", &mut world).unwrap();
            let entity = world.spawn((Pos { x: 1.0, y: 2.0 },));
            world.set_current_tick(tick);
            let mut wal = Wal::create(dir.path(), &codecs, default_config()).unwrap();
            let file = OpenOptions::new()
                .append(true)
                .open(dir.path().join(segment_filename(0)))
                .unwrap();
            let mut writer = BufWriter::new(&file);
            for (seq, tick_after) in [(0, tick), (1, next_tick)] {
                let entry = WalEntry::Mutations(crate::record::WalRecord {
                    seq,
                    tick_after,
                    mutations: vec![SerializedMutation::Insert {
                        entity: entity.to_bits(),
                        component_id: 0,
                        data: rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 9.0, y: 10.0 })
                            .unwrap()
                            .to_vec(),
                    }],
                });
                let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
                write_frame(&mut writer, &bytes, 0).unwrap();
            }
            writer.flush().unwrap();

            let result = wal.replay_from(0, &mut world, &codecs);
            if next_tick == 6 {
                assert_eq!(result.unwrap(), 1);
                assert_eq!(world.current_tick(), 7);
                assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 9.0, y: 10.0 }));
            } else {
                if next_tick == tick {
                    assert!(matches!(
                        result,
                        Err(WalError::TickRegression {
                            seq: 1,
                            record_tick: 5,
                            world_tick: 6,
                        })
                    ));
                } else {
                    assert!(matches!(result, Err(WalError::Format(ref message))
                        if message.contains("tick overflow")));
                }
                assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
                assert_eq!(world.current_tick(), tick);
            }
        }
    }

    #[test]
    fn replay_preflight_rejects_late_component_reference() {
        let dir = tempfile::tempdir().unwrap();
        let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let entity = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let tick = world.current_tick();
        let mut wal = Wal::create(dir.path(), &codecs, default_config()).unwrap();
        let file = OpenOptions::new()
            .append(true)
            .open(dir.path().join(segment_filename(0)))
            .unwrap();
        let mut writer = BufWriter::new(&file);
        for (seq, component_id) in [(0, 0), (1, 99)] {
            let entry = WalEntry::Mutations(crate::record::WalRecord {
                seq,
                tick_after: tick + seq,
                mutations: vec![SerializedMutation::Insert {
                    entity: entity.to_bits(),
                    component_id,
                    data: rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 9.0, y: 10.0 })
                        .unwrap()
                        .to_vec(),
                }],
            });
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
            write_frame(&mut writer, &bytes, 0).unwrap();
        }
        writer.flush().unwrap();

        assert!(matches!(
            wal.replay_from(0, &mut world, &codecs),
            Err(WalError::Codec(CodecError::UnregisteredComponent(99)))
        ));
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.current_tick(), tick);
    }

    #[test]
    fn replay_plan_preserves_schema_and_fence_across_resume() {
        for corrupt_tail in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut codec_world = World::builder().memory_budget(1024 * 1024).build().unwrap();
            let mut codecs = CodecRegistry::new();
            codecs.register_as::<Pos>("pos", &mut codec_world).unwrap();
            let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
            world.register_component::<Health>();
            let entity = world.spawn((Pos { x: 1.0, y: 2.0 },));
            let tick = world.current_tick();
            assert_ne!(
                world.component_id::<Pos>(),
                codec_world.component_id::<Pos>()
            );
            let mut wal = Wal::create(dir.path(), &codecs, default_config()).unwrap();

            let mutation = |seq, component_id, x| {
                WalEntry::Mutations(crate::record::WalRecord {
                    seq,
                    tick_after: if seq < 4 { tick } else { tick + 1 },
                    mutations: vec![SerializedMutation::Insert {
                        entity: entity.to_bits(),
                        component_id,
                        data: rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x, y: 0.0 })
                            .unwrap()
                            .to_vec(),
                    }],
                })
            };
            let runs = [
                (
                    0,
                    vec![
                        (mutation(0, 0, 99.0), 0), // Before the replay floor.
                        (WalEntry::Checkpoint { flush_seq: 1 }, 7),
                        (mutation(1, 99, 99.0), 6), // Stale: no component resolution or effects.
                    ],
                ),
                (
                    2,
                    vec![
                        (
                            WalEntry::Schema(WalSchema {
                                components: vec![ComponentSchema {
                                    id: 77,
                                    name: "pos".into(),
                                    size: 8,
                                    align: 4,
                                }],
                            }),
                            0,
                        ), // Stale schema must still establish the new mapping.
                        (mutation(2, 77, 3.0), 7),
                        (WalEntry::Checkpoint { flush_seq: 3 }, 8),
                        (mutation(3, 99, 99.0), 7),
                        (mutation(4, 77, 5.0), 8),
                    ],
                ),
            ];
            for (start, entries) in runs {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.path().join(segment_filename(start)))
                    .unwrap();
                let mut writer = BufWriter::new(&file);
                if start != 0 {
                    write_segment_magic(&mut writer).unwrap();
                }
                for (entry, view) in entries {
                    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
                    write_frame(&mut writer, &bytes, view).unwrap();
                }
                if start == 2 && corrupt_tail {
                    let bytes =
                        rkyv::to_bytes::<rkyv::rancor::Error>(&mutation(5, 77, 7.0)).unwrap();
                    writer
                        .write_all(&(bytes.len() as u32).to_le_bytes())
                        .unwrap();
                    writer
                        .write_all(&(crc32fast::hash(&bytes) ^ 1).to_le_bytes())
                        .unwrap();
                    writer.write_all(&8u64.to_le_bytes()).unwrap();
                    writer.write_all(&bytes).unwrap();
                }
                writer.flush().unwrap();
            }

            let result = wal.replay_from(1, &mut world, &codecs);
            if corrupt_tail {
                assert!(matches!(result, Err(WalError::ChecksumMismatch { .. })));
                assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
                assert_eq!(world.current_tick(), tick);
            } else {
                assert_eq!(result.unwrap(), 4);
                assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 5.0, y: 0.0 }));
                assert_eq!(world.current_tick(), tick + 2);
            }
        }
    }

    #[test]
    fn create_append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let seq = wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            for _ in 0..3 {
                let cs = EnumChangeSet::new();
                wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

        for _ in 0..3 {
            let cs = EnumChangeSet::new();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Health>("health", &mut world).unwrap();

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
        codecs_a.register_as::<Pos>("pos", &mut world_a).unwrap();
        codecs_a
            .register_as::<Health>("health", &mut world_a)
            .unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs_a, default_config()).unwrap();

        let e = world_a.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 }, Health(100)))
            .unwrap();
        wal.append(&cs, &codecs_a, world_a.current_tick()).unwrap();
        cs.apply(&mut world_a).unwrap();

        drop(wal);

        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b
            .register_as::<Health>("health", &mut world_b)
            .unwrap();
        codecs_b.register_as::<Pos>("pos", &mut world_b).unwrap();

        let mut wal_b = Wal::open(&wal_dir, &codecs_b, default_config()).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        let positions: Vec<(f32, f32)> =
            world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let health: Vec<u32> = world_b.query::<(&Health,)>().map(|h| h.0.0).collect();
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
        codecs_a.register_as::<Pos>("pos", &mut world_a).unwrap();
        codecs_a
            .register_as::<Health>("health", &mut world_a)
            .unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs_a, default_config()).unwrap();

        let e = world_a.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs_a, world_a.current_tick()).unwrap();
        cs.apply(&mut world_a).unwrap();

        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world_a, e, Health(50));
        cs2.remove::<Pos>(&mut world_a, e);
        wal.append(&cs2, &codecs_a, world_a.current_tick()).unwrap();
        cs2.apply(&mut world_a).unwrap();

        drop(wal);

        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b
            .register_as::<Health>("health", &mut world_b)
            .unwrap();
        codecs_b.register_as::<Pos>("pos", &mut world_b).unwrap();

        let mut wal_b = Wal::open(&wal_dir, &codecs_b, default_config()).unwrap();
        wal_b.replay(&mut world_b, &codecs_b).unwrap();

        let health: Vec<u32> = world_b.query::<(&Health,)>().map(|h| h.0.0).collect();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let wal = Wal::create(&wal_dir, &codecs, small_config()).unwrap();
        assert_eq!(wal.next_seq(), 0);
        assert_eq!(wal.stats().segment_count, 1);
        assert!(wal_dir.is_dir());
        assert_eq!(wal.stats().oldest_seq, Some(0));
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }

        assert_eq!(wal.next_seq(), 20);
        assert!(
            wal.stats().segment_count > 1,
            "should have rolled to multiple segments"
        );

        // Every segment should start with magic + schema preamble
        let segments = list_segments(&wal_dir).unwrap();
        for (_, seg_path) in &segments {
            let file = File::open(seg_path).unwrap();
            // First frame starts after the 4-byte segment magic.
            let (entry, _, _proof, _view) =
                read_next_frame(&file, SEGMENT_MAGIC_SIZE).unwrap().unwrap();
            assert!(matches!(entry, WalEntry::Schema(_)));
        }
    }

    #[test]
    fn open_after_rollover_recovers_next_seq() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
                )
                .unwrap();
                wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }

        let before = wal.stats().segment_count;
        assert!(before > 2);

        let deleted = wal.delete_segments_before(10).unwrap();
        assert!(deleted > 0);
        assert_eq!(wal.stats().segment_count, before - deleted);
        assert!(wal.stats().oldest_seq.is_some());
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }

        assert!(wal.checkpoint_needed());
    }

    #[test]
    fn acknowledge_flush_writes_checkpoint_and_resets() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.checkpoint_needed());

        let seq = wal.next_seq();
        wal.acknowledge_flush(seq).unwrap();

        assert_eq!(wal.last_checkpoint_seq(), Some(seq));
        assert!(!wal.checkpoint_needed());
    }

    #[test]
    fn acknowledge_flush_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(1024),
        };

        {
            let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
                .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();

            wal.acknowledge_flush(wal.next_seq()).unwrap();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        let ckpt_seq = wal.next_seq();
        wal.acknowledge_flush(ckpt_seq).unwrap();
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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.stats().segment_count > 1, "must have rolled over");
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        wal.acknowledge_flush(wal.next_seq()).unwrap();
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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }

        wal.delete_segments_before(u64::MAX).unwrap();
        assert!(wal.stats().segment_count >= 1);

        // WAL should still be appendable
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
    }

    #[test]
    fn open_after_truncate_all_does_not_reuse_seq() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
                )
                .unwrap();
                wal.append(&cs, &codecs, world.current_tick()).unwrap();
                cs.apply(&mut world).unwrap();
            }
            assert!(wal.stats().segment_count > 1);

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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Health>("health", &mut world).unwrap();

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
                )
                .unwrap();
                wal.append(&cs, &codecs, world.current_tick()).unwrap();
                cs.apply(&mut world).unwrap();
            }
            assert!(wal.stats().segment_count > 1);
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
        cs.spawn_bundle(&mut world, e, (Pos { x: 99.0, y: 99.0 },))
            .unwrap();
        wal2.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        drop(wal2);

        // Open with reversed registration order to exercise remap
        let mut world_b = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b
            .register_as::<Health>("health", &mut world_b)
            .unwrap();
        codecs_b.register_as::<Pos>("pos", &mut world_b).unwrap();

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
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        // Record spawn + sparse insert in one changeset.
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        cs.insert_sparse::<Health>(&mut world, e, Health(100));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let seq = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        assert_eq!(seq, 0);
        cs.apply(&mut world).unwrap();

        // Verify sparse component is present.
        assert_eq!(world.get::<Health>(e), Some(&Health(100)));

        // Replay into a fresh world.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2).unwrap();
        codecs2.register::<Health>(&mut world2).unwrap();

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
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Health>(e, Health(50));
        assert_eq!(world.get::<Health>(e), Some(&Health(50)));

        // Record sparse removal.
        let mut cs = EnumChangeSet::new();
        cs.remove_sparse::<Health>(&mut world, e);

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Health>(e), None);

        // Replay into fresh world that has the entity with sparse component.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2).unwrap();
        codecs2.register::<Health>(&mut world2).unwrap();

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
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Health>(e, Health(10));

        // Overwrite sparse component.
        let mut cs = EnumChangeSet::new();
        cs.insert_sparse::<Health>(&mut world, e, Health(999));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Health>(e), Some(&Health(999)));

        // Replay into world with old sparse value.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2).unwrap();
        codecs2.register::<Health>(&mut world2).unwrap();

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

    /// Heap-dense (`String`) component survives a WAL append → replay round
    /// trip. WAL records are rkyv-encoded through the codec (resolved by
    /// `TypeId`), so a variable-length heap field is carried verbatim. This
    /// pins that guarantee with an exact-value assertion on the recovered
    /// `String`, not just a POD field.
    #[test]
    fn wal_replays_heap_component() {
        #[derive(Clone, PartialEq, Debug, Archive, Serialize, Deserialize)]
        struct Note {
            text: String,
        }

        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("heap.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Note>(&mut world).unwrap();

        // Record a spawn carrying a heap String.
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (Note {
                text: "wal survives".to_owned(),
            },),
        )
        .unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let seq = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        assert_eq!(seq, 0);
        cs.apply(&mut world).unwrap();
        assert_eq!(
            world.get::<Note>(e).map(|n| n.text.as_str()),
            Some("wal survives")
        );

        // Replay into a fresh world.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Note>(&mut world2).unwrap();

        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        let e2 = Entity::from_bits(e.to_bits());
        assert_eq!(
            world2.get::<Note>(e2),
            Some(&Note {
                text: "wal survives".to_owned()
            }),
            "heap String component should survive WAL replay value-exact"
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
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        cs.insert_sparse::<Health>(&mut world, e, Health(42));

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // Replay into fresh world — Health registered via codecs.register
        // (not register_sparse), so sparse flag only comes from mark_sparse
        // inside changeset apply.
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2).unwrap();
        codecs2.register::<Health>(&mut world2).unwrap();

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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
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
        codecs.register::<Health>(&mut world).unwrap();

        {
            let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
            wal.append(&EnumChangeSet::new(), &codecs, 0).unwrap();
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
    fn frame_round_trip_preserves_payload_and_view() {
        let file = tempfile::tempfile().unwrap();
        let mut writer = BufWriter::new(&file);
        let start = write_segment_magic(&mut writer).unwrap();
        let entry = WalEntry::Mutations(crate::record::WalRecord {
            seq: 42,
            tick_after: 7,
            mutations: vec![SerializedMutation::Despawn { entity: 99 }],
        });
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
        let size = write_frame(&mut writer, &payload, 3).unwrap();
        writer.flush().unwrap();

        let (restored, next, _proof, view) = read_next_frame(&file, start).unwrap().unwrap();
        assert_eq!(view, 3);
        assert_eq!(next, start + size);
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&restored)
                .unwrap()
                .as_slice(),
            payload.as_slice()
        );
        assert!(read_next_frame(&file, next).unwrap().is_none());
    }

    #[test]
    fn valid_frames_pass_checksum() {
        // End-to-end: write frames and read them back — CRC must match.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("valid_crc.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
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
    fn rollover_many_appends_does_not_collide() {
        // Reproduces the wal_append bench panic: rolls must always pick a
        // fresh segment filename.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("roll.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let config = WalConfig {
            max_segment_bytes: 512,
            ..default_config()
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();
        for i in 0..64 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
    }

    #[test]
    fn open_resumes_view_across_segments() {
        // The highest view may live in a sealed segment while the active
        // segment is older — open must resume from the true max.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("views.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let config = WalConfig {
            max_segment_bytes: 256, // roll on nearly every append
            ..default_config()
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();
        let ea = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ea, (Pos { x: 1.0, y: 1.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(wal.views.bump(), 1);
        let eb = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, eb, (Pos { x: 2.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        drop(wal);

        // Reopen: the live view must resume at 1, not 0.
        let mut wal = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal.views.current(), 1, "view must resume from the log max");

        // A post-restart append stamps view 1 and replays.
        let ec = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ec, (Pos { x: 3.0, y: 3.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        drop(wal);

        let mut recovered = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register_as::<Pos>("pos", &mut recovered).unwrap();
        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay_from(0, &mut recovered, &codecs2).unwrap();
        assert!(recovered.is_alive(ea));
        assert!(recovered.is_alive(eb));
        assert!(
            recovered.is_alive(ec),
            "post-restart append must not be fenced out"
        );
    }

    #[test]
    fn open_truncates_stale_view_tail_in_active_segment() {
        // A deposed leader's late write in the ACTIVE segment is truncated
        // at reopen: next_seq resumes past it and cursors never see it.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("tail.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let ea = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ea, (Pos { x: 1.0, y: 1.0 },))
            .unwrap();
        let seq_a = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(wal.views.bump(), 1);
        let eb = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, eb, (Pos { x: 2.0, y: 2.0 },))
            .unwrap();
        let seq_b = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // Late view-0 write lands at the end of the active segment.
        let mut cs_late = EnumChangeSet::new();
        let ec = world.alloc_entity();
        cs_late
            .spawn_bundle(&mut world, ec, (Pos { x: 3.0, y: 3.0 },))
            .unwrap();
        let record =
            Wal::changeset_to_record(seq_b + 1, &cs_late, &codecs, world.current_tick()).unwrap();
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&WalEntry::Mutations(record))
            .map_err(|e| WalError::Format(e.to_string()))
            .unwrap();
        {
            use std::io::{Seek, SeekFrom, Write as IoWrite};
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(wal_dir.join(segment_filename(0)))
                .unwrap();
            f.seek(SeekFrom::End(0)).unwrap();
            let mut writer = std::io::BufWriter::new(&f);
            write_frame(&mut writer, &payload, 0).unwrap();
            writer.flush().unwrap();
        }
        drop(wal);

        // Reopen truncates the stale tail.
        let mut wal = Wal::open(&wal_dir, &codecs, default_config()).unwrap();
        let ec = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ec, (Pos { x: 3.0, y: 3.0 },))
            .unwrap();
        let seq_d = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(
            seq_d,
            seq_b + 1,
            "post-truncation append must reuse the stale record's sequence"
        );
        drop(wal);

        let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
        let mut seen: Vec<u64> = Vec::new();
        loop {
            let batch = cursor.next_batch(10).unwrap();
            if batch.records.is_empty() {
                break;
            }
            seen.extend(batch.records.iter().map(|r| r.seq));
        }
        assert_eq!(
            seen,
            vec![seq_a, seq_b, seq_d],
            "stale record must never ship"
        );
    }

    #[test]
    fn cursor_seeks_seed_view_state() {
        // A cursor opened mid-log must seed max_view_seen from the frames it
        // skips, or a stale frame after the seek point would ship.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("seed.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let ea = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ea, (Pos { x: 1.0, y: 1.0 },))
            .unwrap();
        let seq_a = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(wal.views.bump(), 1);
        let eb = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, eb, (Pos { x: 2.0, y: 2.0 },))
            .unwrap();
        let seq_b = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // Stale view-0 write AFTER the newest view, in the same segment.
        let mut cs_late = EnumChangeSet::new();
        let ec = world.alloc_entity();
        cs_late
            .spawn_bundle(&mut world, ec, (Pos { x: 3.0, y: 3.0 },))
            .unwrap();
        let record =
            Wal::changeset_to_record(seq_b + 1, &cs_late, &codecs, world.current_tick()).unwrap();
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&WalEntry::Mutations(record))
            .map_err(|e| WalError::Format(e.to_string()))
            .unwrap();
        {
            use std::io::{Seek, SeekFrom, Write as IoWrite};
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(wal_dir.join(segment_filename(0)))
                .unwrap();
            f.seek(SeekFrom::End(0)).unwrap();
            let mut writer = std::io::BufWriter::new(&f);
            write_frame(&mut writer, &payload, 0).unwrap();
            writer.flush().unwrap();
        }
        drop(wal);

        // Cursor opened at seq_a + 1: the seek loop passes B (view 1) and
        // must fence the late view-0 frame it encounters.
        let mut cursor = WalCursor::open(&wal_dir, seq_a + 1).unwrap();
        let batch = cursor.next_batch(10).unwrap();
        let seqs: Vec<u64> = batch.records.iter().map(|r| r.seq).collect();
        assert!(
            !seqs.contains(&(seq_b + 1)),
            "stale frame after seek must not ship: {seqs:?}"
        );
    }

    #[test]
    fn cursor_checkpoint_frames_raise_view_fence() {
        // Checkpoints are stamped with the current view; a cursor must count
        // them toward max_view_seen or a later stale mutation ships.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("ckpt.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        let ea = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ea, (Pos { x: 1.0, y: 1.0 },))
            .unwrap();
        let seq_a = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        assert_eq!(wal.views.bump(), 1);
        wal.acknowledge_flush(seq_a + 1).unwrap(); // checkpoint stamped view 1
        drop(wal);

        // Deposed leader's late write at view 0.
        let mut cs_late = EnumChangeSet::new();
        let ec = world.alloc_entity();
        cs_late
            .spawn_bundle(&mut world, ec, (Pos { x: 3.0, y: 3.0 },))
            .unwrap();
        let record =
            Wal::changeset_to_record(seq_a + 2, &cs_late, &codecs, world.current_tick()).unwrap();
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&WalEntry::Mutations(record))
            .map_err(|e| WalError::Format(e.to_string()))
            .unwrap();
        {
            use std::io::{Seek, SeekFrom, Write as IoWrite};
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(wal_dir.join(segment_filename(0)))
                .unwrap();
            f.seek(SeekFrom::End(0)).unwrap();
            let mut writer = std::io::BufWriter::new(&f);
            write_frame(&mut writer, &payload, 0).unwrap();
            writer.flush().unwrap();
        }

        // Cursor from seq_a + 1: seek passes the view-1 checkpoint; the
        // stale view-0 mutation after it must not ship.
        let mut cursor = WalCursor::open(&wal_dir, seq_a + 1).unwrap();
        let batch = cursor.next_batch(10).unwrap();
        let seqs: Vec<u64> = batch.records.iter().map(|r| r.seq).collect();
        assert!(
            !seqs.contains(&(seq_a + 2)),
            "stale mutation after checkpoint must not ship: {seqs:?}"
        );
    }

    #[test]
    fn stale_view_frames_dropped_by_replay() {
        // INV-2 fence: a frame stamped with a view older than the newest
        // view already seen in the log is from a deposed leader and must
        // never replay.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("fence.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
        assert_eq!(wal.views.current(), 0);

        // Record A at view 0.
        let ea = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ea, (Pos { x: 1.0, y: 1.0 },))
            .unwrap();
        let _seq_a = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // Leader loses leadership: view bumps to 1. Record B at view 1.
        assert_eq!(wal.views.bump(), 1);
        let eb = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, eb, (Pos { x: 2.0, y: 2.0 },))
            .unwrap();
        let seq_b = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // The deposed leader's late write: a valid record, but stamped with
        // the old view. Hand-write it at the current end of the segment.
        let ec = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ec, (Pos { x: 3.0, y: 3.0 },))
            .unwrap();
        let record =
            Wal::changeset_to_record(seq_b + 1, &cs, &codecs, world.current_tick()).unwrap();
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&WalEntry::Mutations(record))
            .map_err(|e| WalError::Format(e.to_string()))
            .unwrap();
        {
            use std::io::{Seek, SeekFrom, Write as IoWrite};
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(wal_dir.join(segment_filename(0)))
                .unwrap();
            f.seek(SeekFrom::End(0)).unwrap();
            let mut writer = std::io::BufWriter::new(&f);
            write_frame(&mut writer, &payload, 0).unwrap();
            writer.flush().unwrap();
        }

        // Replay: A and B apply; the late view-0 frame is dropped.
        drop(wal);
        let mut recovered = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register_as::<Pos>("pos", &mut recovered).unwrap();
        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        let last = wal2.replay_from(0, &mut recovered, &codecs2).unwrap();
        assert_eq!(last, seq_b, "late stale-view record must not extend replay");
        assert!(recovered.is_alive(ea));
        assert!(recovered.is_alive(eb));
        assert!(
            !recovered.is_alive(ec),
            "deposed leader's spawn must be dropped"
        );
    }

    #[test]
    fn frame_header_size_is_sixteen() {
        // [len: u32][crc32: u32][view: u64] — stage 4.0 INV-2 stamping.
        assert_eq!(FRAME_HEADER_SIZE, 16);
    }

    // ── Legacy v1 format detection tests ─────────────────────────────

    #[test]
    fn legacy_v1_segment_detected_on_open() {
        // Simulate a legacy v1 segment: [len: u32 LE][payload] with no magic.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("legacy.wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Write a fake v1 segment: starts with a u32 length (no "MKW3" magic).
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
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

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
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.stats().segment_count > 1);

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
        assert_eq!(&data[0..4], b"MKW3", "segment must start with v2 magic");
    }

    #[test]
    fn replay_insert_then_despawn_preserves_order() {
        // Regression: batched replay must not reorder Insert before Despawn.
        // WAL: spawn(e) / insert(e, Health) / despawn(e)
        // After replay the entity must be dead.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("insert_despawn.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

        // Record 1: spawn with Pos
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // Record 2: insert Health then despawn in the same record
        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world, e, Health(99));
        cs2.record_despawn(e);
        wal.append(&cs2, &codecs, world.current_tick()).unwrap();
        cs2.apply(&mut world).unwrap();

        assert!(!world.is_alive(e));

        // Replay into fresh world (must re-register codecs on the new world)
        drop(wal);
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2).unwrap();
        codecs2.register::<Health>(&mut world2).unwrap();
        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        assert!(!world2.is_alive(e), "entity must be dead after replay");
        assert_eq!(
            world2.query::<(&Pos,)>().count(),
            0,
            "no Pos components should remain"
        );
        assert_eq!(
            world2.query::<(&Health,)>().count(),
            0,
            "no Health components should remain"
        );
    }

    #[test]
    fn replay_insert_then_remove_preserves_order() {
        // Regression: batched replay must not reorder Insert before Remove.
        // WAL: spawn(e, Pos+Health) / insert(e, Health(new)) / remove(e, Health)
        // After replay the entity must exist with Pos but without Health.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("insert_remove.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

        // Record 1: spawn with Pos + Health
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 }, Health(10)))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        // Record 2: overwrite Health, then remove it
        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world, e, Health(99));
        cs2.remove::<Health>(&mut world, e);
        wal.append(&cs2, &codecs, world.current_tick()).unwrap();
        cs2.apply(&mut world).unwrap();

        assert!(world.is_alive(e));
        assert_eq!(world.get::<Health>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));

        // Replay into fresh world (must re-register codecs on the new world)
        drop(wal);
        let mut world2 = World::new();
        let mut codecs2 = CodecRegistry::new();
        codecs2.register::<Pos>(&mut world2).unwrap();
        codecs2.register::<Health>(&mut world2).unwrap();
        let mut wal2 = Wal::open(&wal_dir, &codecs2, default_config()).unwrap();
        wal2.replay(&mut world2, &codecs2).unwrap();

        assert!(world2.is_alive(e), "entity must be alive after replay");
        assert_eq!(
            world2.get::<Health>(e),
            None,
            "Health must be removed after replay"
        );
        assert_eq!(
            world2.get::<Pos>(e),
            Some(&Pos { x: 1.0, y: 2.0 }),
            "Pos must survive replay"
        );
    }

    #[test]
    fn cursor_seeds_fence_from_earlier_segments() {
        // A cursor resumed into a later segment must still know about views
        // stamped in earlier segments: a checkpoint at view 1 in segment 0
        // fences a stale view-0 mutation in segment 1.
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("crossseg.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let config = WalConfig {
            max_segment_bytes: 256, // roll quickly
            ..default_config()
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();
        let ea = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, ea, (Pos { x: 1.0, y: 1.0 },))
            .unwrap();
        let seq_a = wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();
        // Fill segment 0 so the checkpoint below lands after a roll.
        for i in 1..6 {
            let e = world.alloc_entity();
            let mut cs2 = EnumChangeSet::new();
            cs2.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs2, &codecs, world.current_tick()).unwrap();
            cs2.apply(&mut world).unwrap();
        }
        assert_eq!(wal.views.bump(), 1);
        wal.acknowledge_flush(seq_a + 6).unwrap(); // view-1 checkpoint in segment 0
        drop(wal);

        // The deposed leader's stale write lands in a LATER segment
        // (hand-written at its head, after its preamble).
        let segments = crate::wal::list_segments(&wal_dir).unwrap();
        let (last_start, last_path) = segments.last().unwrap().clone();
        assert!(
            last_start > 0,
            "expected the checkpoint to have rolled a segment"
        );
        let mut cs_late = EnumChangeSet::new();
        let ec = world.alloc_entity();
        cs_late
            .spawn_bundle(&mut world, ec, (Pos { x: 4.0, y: 4.0 },))
            .unwrap();
        let record =
            Wal::changeset_to_record(last_start, &cs_late, &codecs, world.current_tick()).unwrap();
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&WalEntry::Mutations(record))
            .map_err(|e| WalError::Format(e.to_string()))
            .unwrap();
        {
            use std::io::{Seek, SeekFrom, Write as IoWrite};
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&last_path)
                .unwrap();
            f.seek(SeekFrom::End(0)).unwrap();
            let mut writer = std::io::BufWriter::new(&f);
            write_frame(&mut writer, &payload, 0).unwrap();
            writer.flush().unwrap();
        }

        // Cursor resumed at the later segment: the earlier segment's view-1
        // checkpoint must fence the stale view-0 mutation.
        let mut cursor = WalCursor::open(&wal_dir, last_start).unwrap();
        let batch = cursor.next_batch(10).unwrap();
        let seqs: Vec<u64> = batch.records.iter().map(|r| r.seq).collect();
        assert!(
            !seqs.contains(&last_start),
            "stale mutation in a later segment must not ship: {seqs:?}"
        );
    }
}
