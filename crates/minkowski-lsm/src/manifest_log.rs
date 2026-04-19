//! Persistent append-only log of manifest mutations.
//!
//! File layout:
//! - Bytes 0..8: file header `[magic: b"MKMF"; 4][version: u8; 1][max_level: u8; 1][reserved: 0u8; 2]`.
//! - Bytes 8..: zero or more frames, each `[len: u32 LE][crc32: u32 LE][payload]`.
//!
//! The frame format matches the WAL's (reimplemented here to avoid a
//! dependency on `minkowski-persist`). The file header is manifest-specific.
//!
//! ## Version history
//!
//! - v1: magic + version + 3 reserved bytes.
//! - v2 (current): magic + version + `max_level` byte + 2 reserved bytes.
//!   `max_level` is `N as u8` for an `LsmManifest<N>`; mismatches on recover
//!   surface as fatal `LsmError::Format`, preventing the silent cross-N tail
//!   truncation that v1 allowed.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::manifest::{LsmManifest, SortedRunMeta};
use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

// ── File header ─────────────────────────────────────────────────────────────

/// 4-byte magic: "M", "K", "M", "F" — Minkowski Manifest.
const MAGIC_BYTES: [u8; 4] = *b"MKMF";

/// Current on-disk format version. v1 logs are rejected with an
/// "unsupported manifest version" error — delete to rebuild from WAL.
const CURRENT_VERSION: u8 = 0x02;

/// Total header size in bytes: 4 magic + 1 version + 1 max_level + 2 reserved.
const HEADER_SIZE: u64 = 8;

/// Write the manifest log header at offset 0.
///
/// Layout: `[magic: 4][version: 1][max_level: 1][reserved: 2]`. `max_level`
/// carries `N as u8` so a later `recover::<N>()` can reject
/// manifests written for a different level count. Reserved bytes are
/// written as zero and ignored on read.
fn write_header<const N: usize>(file: &mut File) -> Result<(), LsmError> {
    let max_level = u8::try_from(N).map_err(|_| {
        LsmError::Format(format!(
            "N={N} does not fit in max_level byte (must be <= 255)"
        ))
    })?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&MAGIC_BYTES)?;
    file.write_all(&[CURRENT_VERSION])?;
    file.write_all(&[max_level])?;
    file.write_all(&[0u8; 2])?;
    Ok(())
}

/// Read and validate the manifest log header.
///
/// Returns `LsmError::Format` with a descriptive message on:
/// - File shorter than 8 bytes
/// - Magic bytes don't match `MKMF`
/// - Version byte doesn't match `CURRENT_VERSION`
/// - `max_level` byte doesn't match the caller's `N`
///
/// Reserved bytes are not validated (forward-compat).
fn validate_header<const N: usize>(file: &mut File) -> Result<(), LsmError> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; 8];
    match file.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(LsmError::Format(
                "not a manifest log: file too short for header".to_owned(),
            ));
        }
        Err(e) => return Err(LsmError::Io(e)),
    }
    if header[0..4] != MAGIC_BYTES {
        return Err(LsmError::Format(
            "not a manifest log: bad magic (delete manifest.log to rebuild from WAL)".to_owned(),
        ));
    }
    let version = header[4];
    if version != CURRENT_VERSION {
        return Err(LsmError::Format(format!(
            "unsupported manifest version {version} (delete manifest.log to rebuild from WAL)"
        )));
    }
    let stored_max_level = header[5];
    if (stored_max_level as usize) != N {
        return Err(LsmError::Format(format!(
            "manifest max_level mismatch: file recorded {stored_max_level}, requested N={N} \
             (delete manifest.log to rebuild from WAL, or construct LsmManifest<{stored_max_level}>)"
        )));
    }
    Ok(())
}

// ── Entry type ──────────────────────────────────────────────────────────────

/// A log entry that mutates manifest state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestEntry {
    AddRun {
        level: Level,
        meta: SortedRunMeta,
    },
    RemoveRun {
        level: Level,
        path: PathBuf,
    },
    PromoteRun {
        from_level: Level,
        to_level: Level,
        path: PathBuf,
    },
    SetSequence {
        next_sequence: SeqNo,
    },
    /// Atomic combination of `AddRun` + `SetSequence`.
    ///
    /// A single frame ensures that a crash can never leave the manifest with a
    /// new run recorded but the sequence pointer still at its old value.
    AddRunAndSequence {
        level: Level,
        meta: SortedRunMeta,
        next_sequence: SeqNo,
    },
    /// Atomic compaction commit: install the output run, remove all input runs.
    ///
    /// A single CRC-protected frame ensures recovery cannot see a partial
    /// compaction state (output + some remaining inputs at old levels). Either
    /// the frame applies entirely or it is tail-truncated, leaving the
    /// pre-compaction state intact.
    CompactionCommit {
        output_level: Level,
        output: SortedRunMeta,
        inputs: Vec<(Level, PathBuf)>,
    },
}

/// Sanity bound on the number of inputs per `CompactionCommit` frame.
/// Wire format carries `u32` so the limit is logical, not structural —
/// raise only if compaction granularity intentionally produces jobs
/// past this count.
pub(crate) const MAX_COMPACTION_INPUTS: usize = 1024;

// ── Frame codec ─────────────────────────────────────────────────────────────

/// Maximum payload size to accept when reading (guard against corrupt length).
const MAX_FRAME_PAYLOAD: usize = 1_048_576; // 1 MiB

fn write_frame(file: &mut File, pos: u64, payload: &[u8]) -> Result<u64, LsmError> {
    file.seek(SeekFrom::Start(pos))?;
    let len = payload.len() as u32;
    let crc = crc32fast::hash(payload);
    file.write_all(&len.to_le_bytes())?;
    file.write_all(&crc.to_le_bytes())?;
    file.write_all(payload)?;
    Ok(8 + payload.len() as u64)
}

fn read_frame(file: &File, pos: u64) -> Result<Option<(Vec<u8>, u64)>, LsmError> {
    let mut f = file;
    f.seek(SeekFrom::Start(pos))?;

    let mut header = [0u8; 8];
    let n = f.read(&mut header)?;
    if n == 0 {
        return Ok(None); // clean EOF
    }
    if n < 8 {
        return Err(LsmError::Format("truncated frame header".to_owned()));
    }

    let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let stored_crc = u32::from_le_bytes(header[4..8].try_into().unwrap());

    if len > MAX_FRAME_PAYLOAD {
        return Err(LsmError::Format(format!(
            "frame length {len} exceeds maximum"
        )));
    }

    let mut payload = vec![0u8; len];
    match f.read_exact(&mut payload) {
        Ok(()) => {}
        // A truncated payload (header fsynced, payload page never reached disk)
        // is a torn write, not an I/O error. Reclassify so the replay loop
        // treats it as tail corruption and truncates cleanly.
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(LsmError::Format("truncated frame payload".to_owned()));
        }
        Err(e) => return Err(LsmError::Io(e)),
    }

    let computed_crc = crc32fast::hash(&payload);
    if stored_crc != computed_crc {
        return Err(LsmError::Crc {
            offset: pos,
            expected: stored_crc,
            actual: computed_crc,
        });
    }

    Ok(Some((payload, pos + 8 + len as u64)))
}

// ── Entry codec ─────────────────────────────────────────────────────────────

/// Tag byte identifying a `ManifestEntry` variant on disk.
///
/// `#[repr(u8)]` pins the discriminant values so the enum is layout-compatible
/// with the existing wire format. Encode casts via `as u8`; decode parses via
/// `TryFrom<u8>`, which returns `LsmError::Format` on unknown bytes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ManifestTag {
    AddRun = 0x01,
    RemoveRun = 0x02,
    PromoteRun = 0x03,
    SetSequence = 0x04,
    AddRunAndSequence = 0x05,
    CompactionCommit = 0x06,
}

impl TryFrom<u8> for ManifestTag {
    type Error = LsmError;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        match byte {
            0x01 => Ok(Self::AddRun),
            0x02 => Ok(Self::RemoveRun),
            0x03 => Ok(Self::PromoteRun),
            0x04 => Ok(Self::SetSequence),
            0x05 => Ok(Self::AddRunAndSequence),
            0x06 => Ok(Self::CompactionCommit),
            other => Err(LsmError::Format(format!("unknown entry tag: {other:#04x}"))),
        }
    }
}

fn encode_path(buf: &mut Vec<u8>, path: &Path) -> Result<(), LsmError> {
    let s = path
        .to_str()
        .ok_or_else(|| LsmError::Format("non-UTF-8 path".to_owned()))?;
    let bytes = s.as_bytes();
    let len = u16::try_from(bytes.len())
        .map_err(|_| LsmError::Format(format!("path length {} exceeds u16", bytes.len())))?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

/// Encode an archetype coverage list with a checked u16 count prefix.
fn encode_coverage(buf: &mut Vec<u8>, coverage: &[u16]) -> Result<(), LsmError> {
    let count = u16::try_from(coverage.len()).map_err(|_| {
        LsmError::Format(format!(
            "archetype coverage count {} exceeds u16",
            coverage.len()
        ))
    })?;
    buf.extend_from_slice(&count.to_le_bytes());
    for &arch_id in coverage {
        buf.extend_from_slice(&arch_id.to_le_bytes());
    }
    Ok(())
}

fn decode_path(data: &[u8], offset: &mut usize) -> Result<PathBuf, LsmError> {
    if *offset + 2 > data.len() {
        return Err(LsmError::Format("truncated path length".to_owned()));
    }
    let path_len = u16::from_le_bytes(data[*offset..*offset + 2].try_into().unwrap()) as usize;
    *offset += 2;
    if *offset + path_len > data.len() {
        return Err(LsmError::Format("truncated path data".to_owned()));
    }
    let s = std::str::from_utf8(&data[*offset..*offset + path_len])
        .map_err(|e| LsmError::Format(format!("invalid UTF-8 in path: {e}")))?;
    *offset += path_len;
    Ok(PathBuf::from(s))
}

fn read_u64_le(data: &[u8], offset: &mut usize) -> Result<u64, LsmError> {
    if *offset + 8 > data.len() {
        return Err(LsmError::Format("truncated u64".to_owned()));
    }
    let val = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(val)
}

fn read_u16_le(data: &[u8], offset: &mut usize) -> Result<u16, LsmError> {
    if *offset + 2 > data.len() {
        return Err(LsmError::Format("truncated u16".to_owned()));
    }
    let val = u16::from_le_bytes(data[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    Ok(val)
}

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32, LsmError> {
    if *offset + 4 > data.len() {
        return Err(LsmError::Format("truncated u32".to_owned()));
    }
    let val = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

/// Encode a `SortedRunMeta` payload (path, seq range, coverage, page count,
/// size bytes). Counterpart of [`decode_meta`].
fn encode_meta(buf: &mut Vec<u8>, meta: &SortedRunMeta) -> Result<(), LsmError> {
    encode_path(buf, meta.path())?;
    buf.extend_from_slice(&meta.sequence_range().lo().get().to_le_bytes());
    buf.extend_from_slice(&meta.sequence_range().hi().get().to_le_bytes());
    encode_coverage(buf, meta.archetype_coverage())?;
    buf.extend_from_slice(&meta.page_count().get().to_le_bytes());
    buf.extend_from_slice(&meta.size_bytes().get().to_le_bytes());
    Ok(())
}

/// Decode a `SortedRunMeta` payload. Mirror of [`encode_meta`].
fn decode_meta(data: &[u8], offset: &mut usize) -> Result<SortedRunMeta, LsmError> {
    let path = decode_path(data, offset)?;
    let seq_lo = read_u64_le(data, offset)?;
    let seq_hi = read_u64_le(data, offset)?;
    let count = read_u16_le(data, offset)? as usize;
    if *offset + count * 2 > data.len() {
        return Err(LsmError::Format("truncated coverage data".to_owned()));
    }
    let mut coverage = Vec::with_capacity(count);
    for _ in 0..count {
        coverage.push(read_u16_le(data, offset)?);
    }
    let page_count = read_u64_le(data, offset)?;
    let size_bytes = read_u64_le(data, offset)?;
    let page_count = PageCount::new(page_count)
        .ok_or_else(|| LsmError::Format("page_count must be non-zero".to_owned()))?;
    SortedRunMeta::new(
        path,
        SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi))?,
        coverage,
        page_count,
        SizeBytes::new(size_bytes),
    )
}

fn encode_entry(entry: &ManifestEntry) -> Result<Vec<u8>, LsmError> {
    let mut buf = Vec::new();
    match entry {
        ManifestEntry::AddRun { level, meta } => {
            buf.push(ManifestTag::AddRun as u8);
            buf.push(level.as_u8());
            encode_meta(&mut buf, meta)?;
        }
        ManifestEntry::RemoveRun { level, path } => {
            buf.push(ManifestTag::RemoveRun as u8);
            buf.push(level.as_u8());
            encode_path(&mut buf, path)?;
        }
        ManifestEntry::PromoteRun {
            from_level,
            to_level,
            path,
        } => {
            buf.push(ManifestTag::PromoteRun as u8);
            buf.push(from_level.as_u8());
            buf.push(to_level.as_u8());
            encode_path(&mut buf, path)?;
        }
        ManifestEntry::SetSequence { next_sequence } => {
            buf.push(ManifestTag::SetSequence as u8);
            buf.extend_from_slice(&next_sequence.get().to_le_bytes());
        }
        ManifestEntry::AddRunAndSequence {
            level,
            meta,
            next_sequence,
        } => {
            buf.push(ManifestTag::AddRunAndSequence as u8);
            buf.push(level.as_u8());
            encode_meta(&mut buf, meta)?;
            buf.extend_from_slice(&next_sequence.get().to_le_bytes());
        }
        ManifestEntry::CompactionCommit {
            output_level,
            output,
            inputs,
        } => {
            // Reject at encode time what decode would reject at replay.
            // Without this, a caller could successfully append a frame
            // that the same codebase cannot replay — on restart, recovery
            // would classify it as tail corruption and truncate the log,
            // silently dropping this commit AND every subsequent entry.
            if inputs.len() > MAX_COMPACTION_INPUTS {
                return Err(LsmError::Format(format!(
                    "CompactionCommit input_count {} exceeds MAX_COMPACTION_INPUTS ({MAX_COMPACTION_INPUTS})",
                    inputs.len()
                )));
            }
            // Redundant after the check above — inputs.len() fits u32 given
            // MAX_COMPACTION_INPUTS = 1024 — but kept as defense-in-depth
            // in case MAX_COMPACTION_INPUTS ever grows past u32::MAX.
            let input_count = u32::try_from(inputs.len()).map_err(|_| {
                LsmError::Format(format!(
                    "CompactionCommit input_count {} exceeds u32",
                    inputs.len()
                ))
            })?;
            buf.push(ManifestTag::CompactionCommit as u8);
            buf.push(output_level.as_u8());
            encode_meta(&mut buf, output)?;
            buf.extend_from_slice(&input_count.to_le_bytes());
            for (level, path) in inputs {
                buf.push(level.as_u8());
                encode_path(&mut buf, path)?;
            }
        }
    }
    Ok(buf)
}

fn decode_entry(data: &[u8]) -> Result<ManifestEntry, LsmError> {
    if data.is_empty() {
        return Err(LsmError::Format("empty entry".to_owned()));
    }
    let tag = ManifestTag::try_from(data[0])?;
    let mut offset = 1;

    match tag {
        ManifestTag::AddRun => {
            if offset >= data.len() {
                return Err(LsmError::Format("truncated AddRun".to_owned()));
            }
            let level_byte = data[offset];
            offset += 1;
            let level = Level::new(level_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {level_byte}")))?;
            let meta = decode_meta(data, &mut offset)?;
            Ok(ManifestEntry::AddRun { level, meta })
        }
        ManifestTag::RemoveRun => {
            if offset >= data.len() {
                return Err(LsmError::Format("truncated RemoveRun".to_owned()));
            }
            let level_byte = data[offset];
            offset += 1;
            let level = Level::new(level_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {level_byte}")))?;
            let path = decode_path(data, &mut offset)?;
            Ok(ManifestEntry::RemoveRun { level, path })
        }
        ManifestTag::PromoteRun => {
            if offset + 2 > data.len() {
                return Err(LsmError::Format("truncated PromoteRun".to_owned()));
            }
            let from_byte = data[offset];
            offset += 1;
            let to_byte = data[offset];
            offset += 1;
            let from_level = Level::new(from_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {from_byte}")))?;
            let to_level = Level::new(to_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {to_byte}")))?;
            let path = decode_path(data, &mut offset)?;
            Ok(ManifestEntry::PromoteRun {
                from_level,
                to_level,
                path,
            })
        }
        ManifestTag::SetSequence => {
            let next_sequence = SeqNo::from(read_u64_le(data, &mut offset)?);
            Ok(ManifestEntry::SetSequence { next_sequence })
        }
        ManifestTag::AddRunAndSequence => {
            if offset >= data.len() {
                return Err(LsmError::Format("truncated AddRunAndSequence".to_owned()));
            }
            let level_byte = data[offset];
            offset += 1;
            let level = Level::new(level_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {level_byte}")))?;
            let meta = decode_meta(data, &mut offset)?;
            let next_sequence = SeqNo::from(read_u64_le(data, &mut offset)?);
            Ok(ManifestEntry::AddRunAndSequence {
                level,
                meta,
                next_sequence,
            })
        }
        ManifestTag::CompactionCommit => {
            if offset >= data.len() {
                return Err(LsmError::Format(
                    "truncated CompactionCommit output_level".to_owned(),
                ));
            }
            let output_level_byte = data[offset];
            offset += 1;
            let output_level = Level::new(output_level_byte).ok_or_else(|| {
                LsmError::Format(format!("invalid output_level {output_level_byte}"))
            })?;
            let output = decode_meta(data, &mut offset)?;
            let input_count = read_u32_le(data, &mut offset)? as usize;
            if input_count > MAX_COMPACTION_INPUTS {
                return Err(LsmError::Format(format!(
                    "CompactionCommit input_count {input_count} exceeds MAX_COMPACTION_INPUTS ({MAX_COMPACTION_INPUTS})"
                )));
            }
            let mut inputs = Vec::with_capacity(input_count);
            for _ in 0..input_count {
                if offset >= data.len() {
                    return Err(LsmError::Format(
                        "truncated CompactionCommit input level byte".to_owned(),
                    ));
                }
                let level_byte = data[offset];
                offset += 1;
                let level = Level::new(level_byte)
                    .ok_or_else(|| LsmError::Format(format!("invalid input level {level_byte}")))?;
                let path = decode_path(data, &mut offset)?;
                inputs.push((level, path));
            }
            Ok(ManifestEntry::CompactionCommit {
                output_level,
                output,
                inputs,
            })
        }
    }
}

// ── ManifestLog ─────────────────────────────────────────────────────────────

fn apply_entry<const N: usize>(
    manifest: &mut LsmManifest<N>,
    entry: &ManifestEntry,
) -> Result<(), LsmError> {
    match entry {
        ManifestEntry::AddRun { level, meta } => manifest.add_run(*level, meta.clone())?,
        ManifestEntry::RemoveRun { level, path } => {
            // A RemoveRun for a path the manifest doesn't know means log
            // corruption — the corresponding AddRun was lost, or entries
            // are out of order. Propagate so replay treats the rest as
            // tail garbage. Same policy as PromoteRun above.
            if manifest.remove_run(*level, path).is_none() {
                return Err(LsmError::Format(format!(
                    "RemoveRun: run {} not found at level {}",
                    path.display(),
                    level
                )));
            }
        }
        ManifestEntry::PromoteRun {
            from_level,
            to_level,
            path,
        } => {
            // A failed promote indicates log corruption — the source run is
            // missing. Propagate so the replay loop treats the rest of the
            // log as tail garbage rather than silently diverging.
            manifest.promote_run(*from_level, *to_level, path)?;
        }
        ManifestEntry::SetSequence { next_sequence } => {
            manifest.set_next_sequence(*next_sequence);
        }
        ManifestEntry::AddRunAndSequence {
            level,
            meta,
            next_sequence,
        } => {
            manifest.add_run(*level, meta.clone())?;
            manifest.set_next_sequence(*next_sequence);
        }
        ManifestEntry::CompactionCommit {
            output_level,
            output,
            inputs,
        } => {
            debug_assert!(
                inputs.len() <= MAX_COMPACTION_INPUTS,
                "CompactionCommit with {} inputs — check compaction granularity",
                inputs.len()
            );
            // Pre-validate: every input exists, no duplicates, output_level
            // is in-range for this manifest. Keeps apply_entry's
            // all-or-nothing contract independent of mutation ordering:
            // after this loop passes, every subsequent step must succeed.
            //
            // output_level in-range: a cross-N log is caught by the
            // header max_level gate, but a hand-forged frame could still
            // carry a level byte in [N, MAX_LEVELS). Explicit check here.
            if output_level.as_index() >= N {
                return Err(LsmError::Format(format!(
                    "CompactionCommit: output_level {output_level} out of range for {N}-level manifest"
                )));
            }
            let mut seen = std::collections::HashSet::with_capacity(inputs.len());
            for (level, path) in inputs {
                if !seen.insert((*level, path.as_path())) {
                    return Err(LsmError::Format(format!(
                        "CompactionCommit: duplicate input {} at level {}",
                        path.display(),
                        level
                    )));
                }
                let exists = manifest
                    .runs_at_level(*level)
                    .iter()
                    .any(|m| m.path() == path.as_path());
                if !exists {
                    return Err(LsmError::Format(format!(
                        "CompactionCommit: input run {} not found at level {}",
                        path.display(),
                        level
                    )));
                }
            }
            // All checks passed; mutations cannot fail. Use expect() rather
            // than debug_assert so release builds still panic loudly if an
            // invariant breaks.
            manifest
                .add_run(*output_level, output.clone())
                .expect("output_level pre-validated < N");
            for (level, path) in inputs {
                manifest.remove_run(*level, path).unwrap_or_else(|| {
                    panic!(
                        "pre-validated CompactionCommit input vanished: {} at level {}",
                        path.display(),
                        level
                    )
                });
            }
        }
    }
    Ok(())
}

/// Replay the frame sequence starting at `start` in the given file.
/// Truncates on torn-tail / decode / apply errors, as the existing
/// recovery contract requires. Returns the recovered manifest and the
/// post-truncation position (end of the valid frame region).
fn replay_frames<const N: usize>(
    file: &File,
    path: &Path,
    start: u64,
) -> Result<(LsmManifest<N>, u64), LsmError> {
    let mut manifest = LsmManifest::new();
    let mut pos: u64 = start;

    loop {
        let (payload, next_pos) = match read_frame(file, pos) {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(LsmError::Crc { .. } | LsmError::Format(_)) => {
                truncate_at(path, pos)?;
                break;
            }
            Err(e) => return Err(e),
        };

        let Ok(entry) = decode_entry(&payload) else {
            truncate_at(path, pos)?;
            break;
        };
        if apply_entry(&mut manifest, &entry).is_err() {
            truncate_at(path, pos)?;
            break;
        }
        pos = next_pos;
    }

    Ok((manifest, pos))
}

/// Persistent append-only log of manifest mutations.
///
/// Each entry is framed with a CRC32 checksum for integrity. On crash
/// recovery, [`recover`](Self::recover) reconstructs the manifest from the log,
/// tolerating a corrupt tail frame (torn write).
pub struct ManifestLog {
    file: File,
    write_pos: u64,
}

impl ManifestLog {
    /// Append an entry to the log, fsyncing for durability.
    pub fn append(&mut self, entry: &ManifestEntry) -> Result<(), LsmError> {
        let payload = encode_entry(entry)?;
        let written = write_frame(&mut self.file, self.write_pos, &payload)?;
        self.file.sync_all()?;
        self.write_pos += written;
        Ok(())
    }

    /// Load an existing manifest log or initialize a new empty one.
    ///
    /// If `path` does not exist: creates it, writes the header, fsyncs.
    /// Returns `(LsmManifest::new(), log_handle)` ready to append.
    ///
    /// If `path` exists: reads the 8-byte header and validates magic +
    /// version (rejecting unknown formats with `LsmError::Format`),
    /// replays frames from offset 8 onward (truncating torn tails), and
    /// returns `(recovered_manifest, log_handle)` with `write_pos` at
    /// end of valid data.
    pub fn recover<const N: usize>(path: &Path) -> Result<(LsmManifest<N>, Self), LsmError> {
        if !path.exists() {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(false)
                .open(path)?;
            write_header::<N>(&mut file)?;
            file.sync_all()?;
            return Ok((
                LsmManifest::new(),
                Self {
                    file,
                    write_pos: HEADER_SIZE,
                },
            ));
        }

        let mut file = OpenOptions::new().write(true).read(true).open(path)?;
        validate_header::<N>(&mut file)?;
        let (manifest, write_pos) = replay_frames::<N>(&file, path, HEADER_SIZE)?;
        Ok((manifest, Self { file, write_pos }))
    }

    /// Explicit fsync.
    pub fn sync(&mut self) -> Result<(), LsmError> {
        self.file.sync_all()?;
        Ok(())
    }
}

/// Truncate the file to `len` bytes for crash recovery.
fn truncate_at(path: &Path, len: u64) -> Result<(), LsmError> {
    let f = OpenOptions::new().write(true).open(path)?;
    f.set_len(len)?;
    f.sync_all()?;
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::types::{Level, PageCount, SizeBytes};

    fn test_meta(name: &str) -> SortedRunMeta {
        SortedRunMeta::new(
            PathBuf::from(name),
            SeqRange::new(SeqNo::from(10u64), SeqNo::from(20u64)).unwrap(),
            vec![0, 3, 7],
            PageCount::new(42).unwrap(),
            SizeBytes::new(8192),
        )
        .unwrap()
    }

    #[test]
    fn manifest_tag_try_from_u8_accepts_known_values() {
        assert_eq!(ManifestTag::try_from(0x01).unwrap(), ManifestTag::AddRun);
        assert_eq!(ManifestTag::try_from(0x02).unwrap(), ManifestTag::RemoveRun);
        assert_eq!(
            ManifestTag::try_from(0x03).unwrap(),
            ManifestTag::PromoteRun
        );
        assert_eq!(
            ManifestTag::try_from(0x04).unwrap(),
            ManifestTag::SetSequence
        );
        assert_eq!(
            ManifestTag::try_from(0x05).unwrap(),
            ManifestTag::AddRunAndSequence
        );
        assert_eq!(
            ManifestTag::try_from(0x06).unwrap(),
            ManifestTag::CompactionCommit
        );
    }

    #[test]
    fn manifest_tag_try_from_u8_rejects_unknown_values() {
        for byte in [0x00u8, 0x07, 0x7F, 0xFF] {
            let err = ManifestTag::try_from(byte).unwrap_err();
            assert!(matches!(err, LsmError::Format(_)));
            if let LsmError::Format(msg) = err {
                assert!(msg.contains("unknown entry tag"), "got: {msg}");
            }
        }
    }

    #[test]
    fn manifest_tag_as_u8_matches_discriminant() {
        assert_eq!(ManifestTag::AddRun as u8, 0x01);
        assert_eq!(ManifestTag::RemoveRun as u8, 0x02);
        assert_eq!(ManifestTag::PromoteRun as u8, 0x03);
        assert_eq!(ManifestTag::SetSequence as u8, 0x04);
        assert_eq!(ManifestTag::AddRunAndSequence as u8, 0x05);
        assert_eq!(ManifestTag::CompactionCommit as u8, 0x06);
    }

    #[test]
    fn encode_decode_add_run() {
        let meta = test_meta("10-20.run");
        let entry = ManifestEntry::AddRun {
            level: Level::L1,
            meta,
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_remove_run() {
        let entry = ManifestEntry::RemoveRun {
            level: Level::L2,
            path: PathBuf::from("old.run"),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_promote_run() {
        let entry = ManifestEntry::PromoteRun {
            from_level: Level::L0,
            to_level: Level::L1,
            path: PathBuf::from("promoted.run"),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_promote_run_at_l3_boundary() {
        let entry = ManifestEntry::PromoteRun {
            from_level: Level::L3,
            to_level: Level::L0,
            path: PathBuf::from("demoted.run"),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_set_sequence() {
        let entry = ManifestEntry::SetSequence {
            next_sequence: SeqNo::from(12345u64),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_add_run_and_sequence() {
        let meta = test_meta("atomic.run");
        let entry = ManifestEntry::AddRunAndSequence {
            level: Level::L0,
            meta,
            next_sequence: SeqNo::from(99u64),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_add_run_and_sequence_at_l3() {
        let meta = test_meta("l3.run");
        let entry = ManifestEntry::AddRunAndSequence {
            level: Level::L3,
            meta,
            next_sequence: SeqNo::from(42u64),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn replay_add_run_and_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        let (_, mut log) = ManifestLog::recover::<4>(&path).unwrap();
        log.append(&ManifestEntry::AddRunAndSequence {
            level: Level::L0,
            meta: test_meta("atomic.run"),
            next_sequence: SeqNo::from(42u64),
        })
        .unwrap();
        drop(log);

        let (manifest, _) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 1);
        assert_eq!(manifest.next_sequence(), SeqNo::from(42u64));
        assert_eq!(
            manifest.runs_at_level(Level::L0)[0].path(),
            Path::new("atomic.run")
        );
    }

    #[test]
    fn encode_decode_compaction_commit() {
        let entry = ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output: test_meta("compacted.run"),
            inputs: vec![
                (Level::L0, PathBuf::from("in1.run")),
                (Level::L0, PathBuf::from("in2.run")),
                (Level::L0, PathBuf::from("in3.run")),
                (Level::L0, PathBuf::from("in4.run")),
            ],
        };
        let payload = encode_entry(&entry).unwrap();
        assert_eq!(
            payload[0],
            ManifestTag::CompactionCommit as u8,
            "tag byte is 0x06"
        );
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn encode_decode_compaction_commit_zero_inputs() {
        // Not a useful commit (you'd never compact 0 runs) but the wire
        // format must round-trip cleanly.
        let entry = ManifestEntry::CompactionCommit {
            output_level: Level::L2,
            output: test_meta("empty_inputs.run"),
            inputs: vec![],
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn decode_compaction_commit_rejects_oversized_input_count() {
        // Hand-craft a header declaring input_count > MAX_COMPACTION_INPUTS.
        let meta = test_meta("victim.run");
        let mut buf = Vec::new();
        buf.push(ManifestTag::CompactionCommit as u8);
        buf.push(Level::L1.as_u8());
        encode_meta(&mut buf, &meta).unwrap();
        buf.extend_from_slice(&((MAX_COMPACTION_INPUTS as u32) + 1).to_le_bytes());
        // No actual inputs — we expect the count-check to fail before we
        // try to decode any.
        let err = decode_entry(&buf).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("exceeds MAX_COMPACTION_INPUTS"), "got: {msg}");
        }
    }

    #[test]
    fn encode_compaction_commit_rejects_oversized_input_list() {
        // Symmetry with the decode-side check: encode must also reject
        // oversized input lists so a caller can't successfully append a
        // frame the same codebase refuses to replay.
        let output = test_meta("out.run");
        let inputs: Vec<(Level, PathBuf)> = (0..=MAX_COMPACTION_INPUTS)
            .map(|i| (Level::L0, PathBuf::from(format!("in{i}.run"))))
            .collect();
        assert_eq!(inputs.len(), MAX_COMPACTION_INPUTS + 1);
        let entry = ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output,
            inputs,
        };
        let err = encode_entry(&entry).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("exceeds MAX_COMPACTION_INPUTS"), "got: {msg}");
        }
    }

    #[test]
    fn encode_compaction_commit_accepts_max_input_count() {
        // Boundary check: exactly MAX_COMPACTION_INPUTS must encode cleanly.
        let output = test_meta("out.run");
        let inputs: Vec<(Level, PathBuf)> = (0..MAX_COMPACTION_INPUTS)
            .map(|i| (Level::L0, PathBuf::from(format!("in{i}.run"))))
            .collect();
        let entry = ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output,
            inputs,
        };
        let payload = encode_entry(&entry).unwrap();
        // Round-trip through decode to confirm the accepted boundary is
        // also readable.
        let decoded = decode_entry(&payload).unwrap();
        match decoded {
            ManifestEntry::CompactionCommit { inputs, .. } => {
                assert_eq!(inputs.len(), MAX_COMPACTION_INPUTS);
            }
            _ => panic!("expected CompactionCommit"),
        }
    }

    #[test]
    fn apply_compaction_commit_atomic_roundtrip() {
        // Seed a manifest with 4 L0 runs, then apply a CompactionCommit
        // that consolidates them into one L1 run.
        let mut m: LsmManifest<4> = LsmManifest::new();
        for i in 1..=4 {
            m.add_run(Level::L0, test_meta(&format!("in{i}.run")))
                .unwrap();
        }
        assert_eq!(m.runs_at_level(Level::L0).len(), 4);

        let commit = ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output: test_meta("out.run"),
            inputs: (1..=4)
                .map(|i| (Level::L0, PathBuf::from(format!("in{i}.run"))))
                .collect(),
        };
        apply_entry(&mut m, &commit).unwrap();

        assert!(
            m.runs_at_level(Level::L0).is_empty(),
            "all inputs removed from L0"
        );
        assert_eq!(m.runs_at_level(Level::L1).len(), 1);
        assert_eq!(m.runs_at_level(Level::L1)[0].path(), Path::new("out.run"));
    }

    #[test]
    fn apply_compaction_commit_rejects_duplicate_inputs() {
        // Two identical (level, path) entries must error. Without this
        // check a buggy compactor could remove the real input + silently
        // succeed on the duplicate, masking the bug.
        let mut m: LsmManifest<4> = LsmManifest::new();
        m.add_run(Level::L0, test_meta("same.run")).unwrap();

        let commit = ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output: test_meta("out.run"),
            inputs: vec![
                (Level::L0, PathBuf::from("same.run")),
                (Level::L0, PathBuf::from("same.run")),
            ],
        };
        let err = apply_entry(&mut m, &commit).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("duplicate input"), "got: {msg}");
        }
        // Pre-apply state intact.
        assert_eq!(m.runs_at_level(Level::L0).len(), 1);
        assert!(m.runs_at_level(Level::L1).is_empty());
    }

    #[test]
    fn apply_compaction_commit_rejects_oor_output_level() {
        // Hand-forge a frame carrying output_level in [N, MAX_LEVELS) —
        // decode accepts this (Level::new enforces MAX_LEVELS), but
        // apply_entry must catch it as OOR for the target manifest before
        // mutating anything.
        let mut m: LsmManifest<4> = LsmManifest::new();
        m.add_run(Level::L0, test_meta("in.run")).unwrap();

        let oor_level = Level::new(5).unwrap(); // valid Level, OOR for N=4
        let commit = ManifestEntry::CompactionCommit {
            output_level: oor_level,
            output: test_meta("out.run"),
            inputs: vec![(Level::L0, PathBuf::from("in.run"))],
        };
        let err = apply_entry(&mut m, &commit).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("output_level"), "got: {msg}");
            assert!(msg.contains("out of range"), "got: {msg}");
        }
        // Pre-apply state intact.
        assert_eq!(m.runs_at_level(Level::L0).len(), 1);
    }

    #[test]
    fn apply_compaction_commit_missing_input_is_all_or_nothing() {
        // Seed L0 with 2 runs, but list 3 inputs in the commit — the third
        // doesn't exist. apply_entry must NOT mutate (no output installed,
        // no inputs removed) before it errors out.
        let mut m: LsmManifest<4> = LsmManifest::new();
        m.add_run(Level::L0, test_meta("real1.run")).unwrap();
        m.add_run(Level::L0, test_meta("real2.run")).unwrap();

        let commit = ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output: test_meta("out.run"),
            inputs: vec![
                (Level::L0, PathBuf::from("real1.run")),
                (Level::L0, PathBuf::from("real2.run")),
                (Level::L0, PathBuf::from("GHOST.run")),
            ],
        };
        let err = apply_entry(&mut m, &commit).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("GHOST.run"), "got: {msg}");
        }

        // Pre-apply state must be intact: no output installed, no inputs
        // removed.
        assert!(
            m.runs_at_level(Level::L1).is_empty(),
            "output must not be installed on failure"
        );
        assert_eq!(
            m.runs_at_level(Level::L0).len(),
            2,
            "real inputs must not be removed on failure"
        );
    }

    #[test]
    fn replay_compaction_commit_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");
        let (mut m, mut log) = ManifestLog::recover::<4>(&path).unwrap();

        // Seed with some L0 runs via the log (so they replay alongside
        // the commit).
        for i in 1..=3 {
            let name = format!("in{i}.run");
            log.append(&ManifestEntry::AddRun {
                level: Level::L0,
                meta: test_meta(&name),
            })
            .unwrap();
            m.add_run(Level::L0, test_meta(&name)).unwrap();
        }
        // Now record the compaction.
        log.append(&ManifestEntry::CompactionCommit {
            output_level: Level::L1,
            output: test_meta("out.run"),
            inputs: (1..=3)
                .map(|i| (Level::L0, PathBuf::from(format!("in{i}.run"))))
                .collect(),
        })
        .unwrap();
        drop(log);

        let (replayed, _) = ManifestLog::recover::<4>(&path).unwrap();
        assert!(replayed.runs_at_level(Level::L0).is_empty());
        assert_eq!(replayed.runs_at_level(Level::L1).len(), 1);
        assert_eq!(
            replayed.runs_at_level(Level::L1)[0].path(),
            Path::new("out.run")
        );
    }

    #[test]
    fn write_read_frame_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.log");
        let mut file = File::create(&path).unwrap();
        let payload = b"hello manifest";

        let written = write_frame(&mut file, 0, payload).unwrap();
        assert_eq!(written, 8 + payload.len() as u64);

        let file = File::open(&path).unwrap();
        let (read_payload, next_pos) = read_frame(&file, 0).unwrap().unwrap();
        assert_eq!(read_payload, payload);
        assert_eq!(next_pos, written);
    }

    #[test]
    fn read_frame_returns_none_at_eof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.log");
        File::create(&path).unwrap();

        let file = File::open(&path).unwrap();
        assert!(read_frame(&file, 0).unwrap().is_none());
    }

    #[test]
    fn read_frame_detects_corrupt_crc() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.log");

        // Write a valid frame.
        {
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .unwrap();
            write_frame(&mut file, 0, b"good data").unwrap();
        }

        // Corrupt a payload byte.
        {
            let mut data = fs::read(&path).unwrap();
            data[10] ^= 0xFF; // flip a byte in the payload
            fs::write(&path, &data).unwrap();
        }

        let file = File::open(&path).unwrap();
        let result = read_frame(&file, 0);
        assert!(matches!(result, Err(LsmError::Crc { .. })));
    }

    #[test]
    fn replay_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");
        // File doesn't exist → empty manifest.
        let (manifest, _) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(manifest.next_sequence(), SeqNo::from(0u64));
    }

    #[test]
    fn replay_three_add_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        let (_, mut log) = ManifestLog::recover::<4>(&path).unwrap();
        for i in 0..3 {
            let meta = test_meta(&format!("{i}.run"));
            log.append(&ManifestEntry::AddRun {
                level: Level::L0,
                meta,
            })
            .unwrap();
        }
        drop(log);

        let (manifest, _) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 3);
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 3);
    }

    #[test]
    fn replay_add_then_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        let (_, mut log) = ManifestLog::recover::<4>(&path).unwrap();
        let meta = test_meta("ephemeral.run");
        log.append(&ManifestEntry::AddRun {
            level: Level::L0,
            meta: meta.clone(),
        })
        .unwrap();
        log.append(&ManifestEntry::RemoveRun {
            level: Level::L0,
            path: meta.path().to_path_buf(),
        })
        .unwrap();
        drop(log);

        let (manifest, _) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
    }

    #[test]
    fn replay_tolerates_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        // Write 2 good entries.
        let (_, mut log) = ManifestLog::recover::<4>(&path).unwrap();
        log.append(&ManifestEntry::AddRun {
            level: Level::L0,
            meta: test_meta("a.run"),
        })
        .unwrap();
        log.append(&ManifestEntry::AddRun {
            level: Level::L0,
            meta: test_meta("b.run"),
        })
        .unwrap();
        drop(log);

        // Append garbage (simulates torn write).
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x42]).unwrap();
        }

        // Replay should recover the 2 good entries.
        let (manifest, mut log2) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 2);

        // File should be truncated to remove garbage; write_pos should be at end of valid data.
        let file_len = fs::metadata(&path).unwrap().len();
        assert_eq!(log2.write_pos, file_len);

        // Should be able to append after recovery.
        log2.append(&ManifestEntry::AddRun {
            level: Level::L1,
            meta: test_meta("c.run"),
        })
        .unwrap();
        drop(log2);

        let (manifest2, _) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest2.total_runs(), 3);
    }

    #[test]
    fn write_header_emits_expected_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        let mut file = File::create(&path).unwrap();
        write_header::<4>(&mut file).unwrap();
        drop(file);

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], b"MKMF");
        assert_eq!(bytes[4], 0x02, "version bumped to v2 for max_level");
        assert_eq!(bytes[5], 4, "max_level carries N as u8");
        assert_eq!(&bytes[6..8], &[0u8; 2]);
    }

    #[test]
    fn write_header_alternate_n_emits_matching_max_level() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr7.log");
        let mut file = File::create(&path).unwrap();
        write_header::<7>(&mut file).unwrap();
        drop(file);

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes[5], 7, "max_level reflects N=7");
    }

    #[test]
    fn write_header_returns_error_on_n_overflow_u8() {
        // N > 255 cannot fit in the max_level byte. write_header must
        // return LsmError::Format rather than panic, matching the rest of
        // the ManifestLog API's fallible contract.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overflow.log");
        let mut file = File::create(&path).unwrap();
        let err = write_header::<256>(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("max_level byte"), "got: {msg}");
            assert!(msg.contains("N=256"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_accepts_valid_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        write_header::<4>(&mut file).unwrap();
        validate_header::<4>(&mut file).unwrap();
    }

    #[test]
    fn validate_header_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"XXXX\x02\x04\x00\x00").unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header::<4>(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("bad magic"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_rejects_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        // v1 header (pre-PR-3) is now "unsupported"
        fs::write(&path, b"MKMF\x01\x00\x00\x00").unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header::<4>(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("unsupported manifest version"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_rejects_max_level_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        // Written with N=7; opened as N=4 must fail.
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        write_header::<7>(&mut file).unwrap();
        drop(file);

        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header::<4>(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("max_level mismatch"), "got: {msg}");
            assert!(msg.contains("file recorded 7"), "got: {msg}");
            assert!(msg.contains("N=4"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_rejects_file_too_short() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"MKMF").unwrap(); // only 4 bytes
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header::<4>(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("too short"), "got: {msg}");
        }
    }

    #[test]
    fn recover_creates_file_with_header_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.log");
        assert!(!path.exists());

        let (manifest, _log) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(manifest.next_sequence(), SeqNo::from(0u64));

        assert!(path.exists());
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], b"MKMF");
        assert_eq!(bytes[4], 0x02);
    }

    #[test]
    fn recover_accepts_valid_header_with_no_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.log");
        // Pre-create with just a header.
        {
            let mut file = File::create(&path).unwrap();
            write_header::<4>(&mut file).unwrap();
            file.sync_all().unwrap();
        }
        let (manifest, log) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(log.write_pos, 8);
    }

    #[test]
    fn recover_rejects_file_with_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.log");
        fs::write(&path, b"XXXXv1\x00\x00\x00").unwrap();
        let err = ManifestLog::recover::<4>(&path).err().unwrap();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn recover_rejects_file_with_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v99.log");
        fs::write(&path, b"MKMF\x63\x00\x00\x00").unwrap();
        let err = ManifestLog::recover::<4>(&path).err().unwrap();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn recover_replays_existing_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("populated.log");

        // Write a header using the helper directly.
        {
            let mut file = File::create(&path).unwrap();
            write_header::<4>(&mut file).unwrap();
            file.sync_all().unwrap();
        }

        // Reopen via recover, append an entry, reopen again.
        let (_, mut log) = ManifestLog::recover::<4>(&path).unwrap();
        log.append(&ManifestEntry::SetSequence {
            next_sequence: SeqNo::from(42u64),
        })
        .unwrap();
        drop(log);

        let (manifest, _log) = ManifestLog::recover::<4>(&path).unwrap();
        assert_eq!(manifest.next_sequence(), SeqNo::from(42u64));
    }
}
