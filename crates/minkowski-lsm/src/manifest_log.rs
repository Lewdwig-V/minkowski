//! Persistent append-only log of manifest mutations.
//!
//! File layout:
//! - Bytes 0..8: file header `[magic: b"MKMF"; 4][version: u8; 1][reserved: 0u8; 3]`.
//! - Bytes 8..: zero or more frames, each `[len: u32 LE][crc32: u32 LE][payload]`.
//!
//! The frame format matches the WAL's (reimplemented here to avoid a
//! dependency on `minkowski-persist`). The file header is manifest-specific.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::manifest::{LsmManifest, SortedRunMeta};
use crate::types::{Level, SeqNo, SeqRange};

// ── File header ─────────────────────────────────────────────────────────────

/// 4-byte magic: "M", "K", "M", "F" — Minkowski Manifest.
const MAGIC_BYTES: [u8; 4] = *b"MKMF";

const CURRENT_VERSION: u8 = 0x01;

/// Total header size in bytes: 4 magic + 1 version + 3 reserved.
const HEADER_SIZE: u64 = 8;

/// Write the manifest log header at offset 0.
///
/// Layout: `[magic: 4][version: 1][reserved: 3]`. Reserved bytes are
/// written as zero and ignored on read.
fn write_header(file: &mut File) -> Result<(), LsmError> {
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&MAGIC_BYTES)?;
    file.write_all(&[CURRENT_VERSION])?;
    file.write_all(&[0u8; 3])?;
    Ok(())
}

/// Read and validate the manifest log header.
///
/// Returns `LsmError::Format` with a descriptive message on:
/// - File shorter than 8 bytes
/// - Magic bytes don't match `MKMF`
/// - Version byte doesn't match `CURRENT_VERSION`
///
/// Reserved bytes are not validated (forward-compat).
fn validate_header(file: &mut File) -> Result<(), LsmError> {
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
            "unsupported manifest version {version}"
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
}

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

const TAG_ADD_RUN: u8 = 0x01;
const TAG_REMOVE_RUN: u8 = 0x02;
const TAG_PROMOTE_RUN: u8 = 0x03;
const TAG_SET_SEQUENCE: u8 = 0x04;
const TAG_ADD_RUN_AND_SEQUENCE: u8 = 0x05;

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

fn encode_entry(entry: &ManifestEntry) -> Result<Vec<u8>, LsmError> {
    let mut buf = Vec::new();
    match entry {
        ManifestEntry::AddRun { level, meta } => {
            buf.push(TAG_ADD_RUN);
            buf.push(level.as_u8());
            encode_path(&mut buf, meta.path())?;
            buf.extend_from_slice(&meta.sequence_range().lo().0.to_le_bytes());
            buf.extend_from_slice(&meta.sequence_range().hi().0.to_le_bytes());
            encode_coverage(&mut buf, meta.archetype_coverage())?;
            buf.extend_from_slice(&meta.page_count().to_le_bytes());
            buf.extend_from_slice(&meta.size_bytes().to_le_bytes());
        }
        ManifestEntry::RemoveRun { level, path } => {
            buf.push(TAG_REMOVE_RUN);
            buf.push(level.as_u8());
            encode_path(&mut buf, path)?;
        }
        ManifestEntry::PromoteRun {
            from_level,
            to_level,
            path,
        } => {
            buf.push(TAG_PROMOTE_RUN);
            buf.push(from_level.as_u8());
            buf.push(to_level.as_u8());
            encode_path(&mut buf, path)?;
        }
        ManifestEntry::SetSequence { next_sequence } => {
            buf.push(TAG_SET_SEQUENCE);
            buf.extend_from_slice(&next_sequence.0.to_le_bytes());
        }
        ManifestEntry::AddRunAndSequence {
            level,
            meta,
            next_sequence,
        } => {
            buf.push(TAG_ADD_RUN_AND_SEQUENCE);
            buf.push(level.as_u8());
            encode_path(&mut buf, meta.path())?;
            buf.extend_from_slice(&meta.sequence_range().lo().0.to_le_bytes());
            buf.extend_from_slice(&meta.sequence_range().hi().0.to_le_bytes());
            encode_coverage(&mut buf, meta.archetype_coverage())?;
            buf.extend_from_slice(&meta.page_count().to_le_bytes());
            buf.extend_from_slice(&meta.size_bytes().to_le_bytes());
            buf.extend_from_slice(&next_sequence.0.to_le_bytes());
        }
    }
    Ok(buf)
}

fn decode_entry(data: &[u8]) -> Result<ManifestEntry, LsmError> {
    if data.is_empty() {
        return Err(LsmError::Format("empty entry".to_owned()));
    }
    let tag = data[0];
    let mut offset = 1;

    match tag {
        TAG_ADD_RUN => {
            if offset >= data.len() {
                return Err(LsmError::Format("truncated AddRun".to_owned()));
            }
            let level_byte = data[offset];
            offset += 1;
            let level = Level::new(level_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {level_byte}")))?;
            let path = decode_path(data, &mut offset)?;
            let seq_lo = read_u64_le(data, &mut offset)?;
            let seq_hi = read_u64_le(data, &mut offset)?;
            let count = read_u16_le(data, &mut offset)? as usize;
            if offset + count * 2 > data.len() {
                return Err(LsmError::Format("truncated coverage data".to_owned()));
            }
            let mut coverage = Vec::with_capacity(count);
            for _ in 0..count {
                coverage.push(read_u16_le(data, &mut offset)?);
            }
            let page_count = read_u64_le(data, &mut offset)?;
            let size_bytes = read_u64_le(data, &mut offset)?;

            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,
                size_bytes,
            )?;
            Ok(ManifestEntry::AddRun { level, meta })
        }
        TAG_REMOVE_RUN => {
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
        TAG_PROMOTE_RUN => {
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
        TAG_SET_SEQUENCE => {
            let next_sequence = SeqNo(read_u64_le(data, &mut offset)?);
            Ok(ManifestEntry::SetSequence { next_sequence })
        }
        TAG_ADD_RUN_AND_SEQUENCE => {
            if offset >= data.len() {
                return Err(LsmError::Format("truncated AddRunAndSequence".to_owned()));
            }
            let level_byte = data[offset];
            offset += 1;
            let level = Level::new(level_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {level_byte}")))?;
            let path = decode_path(data, &mut offset)?;
            let seq_lo = read_u64_le(data, &mut offset)?;
            let seq_hi = read_u64_le(data, &mut offset)?;
            let count = read_u16_le(data, &mut offset)? as usize;
            if offset + count * 2 > data.len() {
                return Err(LsmError::Format("truncated coverage data".to_owned()));
            }
            let mut coverage = Vec::with_capacity(count);
            for _ in 0..count {
                coverage.push(read_u16_le(data, &mut offset)?);
            }
            let page_count = read_u64_le(data, &mut offset)?;
            let size_bytes = read_u64_le(data, &mut offset)?;
            let next_sequence = SeqNo(read_u64_le(data, &mut offset)?);

            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,
                size_bytes,
            )?;
            Ok(ManifestEntry::AddRunAndSequence {
                level,
                meta,
                next_sequence,
            })
        }
        _ => Err(LsmError::Format(format!("unknown entry tag: {tag:#04x}"))),
    }
}

// ── ManifestLog ─────────────────────────────────────────────────────────────

fn apply_entry(manifest: &mut LsmManifest, entry: &ManifestEntry) -> Result<(), LsmError> {
    match entry {
        ManifestEntry::AddRun { level, meta } => manifest.add_run(*level, meta.clone()),
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
            manifest.add_run(*level, meta.clone());
            manifest.set_next_sequence(*next_sequence);
        }
    }
    Ok(())
}

/// Replay the frame sequence starting at `start` in the given file.
/// Truncates on torn-tail / decode / apply errors, as the existing
/// recovery contract requires. Returns the recovered manifest and the
/// post-truncation position (end of the valid frame region).
fn replay_frames(file: &File, path: &Path, start: u64) -> Result<(LsmManifest, u64), LsmError> {
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
    pub fn recover(path: &Path) -> Result<(LsmManifest, Self), LsmError> {
        if !path.exists() {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(false)
                .open(path)?;
            write_header(&mut file)?;
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
        validate_header(&mut file)?;
        let (manifest, write_pos) = replay_frames(&file, path, HEADER_SIZE)?;
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
    use crate::types::Level;

    fn test_meta(name: &str) -> SortedRunMeta {
        SortedRunMeta::new(
            PathBuf::from(name),
            SeqRange::new(SeqNo(10), SeqNo(20)).unwrap(),
            vec![0, 3, 7],
            42,
            8192,
        )
        .unwrap()
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
    fn encode_decode_set_sequence() {
        let entry = ManifestEntry::SetSequence {
            next_sequence: SeqNo(12345),
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
            next_sequence: SeqNo(99),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn replay_add_run_and_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        let (_, mut log) = ManifestLog::recover(&path).unwrap();
        log.append(&ManifestEntry::AddRunAndSequence {
            level: Level::L0,
            meta: test_meta("atomic.run"),
            next_sequence: SeqNo(42),
        })
        .unwrap();
        drop(log);

        let (manifest, _) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 1);
        assert_eq!(manifest.next_sequence(), SeqNo(42));
        assert_eq!(
            manifest.runs_at_level(Level::L0)[0].path(),
            Path::new("atomic.run")
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
        let (manifest, _) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(manifest.next_sequence(), SeqNo(0));
    }

    #[test]
    fn replay_three_add_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        let (_, mut log) = ManifestLog::recover(&path).unwrap();
        for i in 0..3 {
            let meta = test_meta(&format!("{i}.run"));
            log.append(&ManifestEntry::AddRun {
                level: Level::L0,
                meta,
            })
            .unwrap();
        }
        drop(log);

        let (manifest, _) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 3);
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 3);
    }

    #[test]
    fn replay_add_then_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        let (_, mut log) = ManifestLog::recover(&path).unwrap();
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

        let (manifest, _) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
    }

    #[test]
    fn replay_tolerates_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.log");

        // Write 2 good entries.
        let (_, mut log) = ManifestLog::recover(&path).unwrap();
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
        let (manifest, mut log2) = ManifestLog::recover(&path).unwrap();
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

        let (manifest2, _) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest2.total_runs(), 3);
    }

    #[test]
    fn write_header_emits_expected_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        let mut file = File::create(&path).unwrap();
        write_header(&mut file).unwrap();
        drop(file);

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], b"MKMF");
        assert_eq!(bytes[4], 0x01);
        assert_eq!(&bytes[5..8], &[0u8; 3]);
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
        write_header(&mut file).unwrap();
        validate_header(&mut file).unwrap();
    }

    #[test]
    fn validate_header_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"XXXX\x01\x00\x00\x00").unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("bad magic"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_rejects_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"MKMF\xFF\x00\x00\x00").unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("unsupported manifest version"), "got: {msg}");
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
        let err = validate_header(&mut file).unwrap_err();
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

        let (manifest, _log) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(manifest.next_sequence(), SeqNo(0));

        assert!(path.exists());
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], b"MKMF");
        assert_eq!(bytes[4], 0x01);
    }

    #[test]
    fn recover_accepts_valid_header_with_no_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.log");
        // Pre-create with just a header.
        {
            let mut file = File::create(&path).unwrap();
            write_header(&mut file).unwrap();
            file.sync_all().unwrap();
        }
        let (manifest, log) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(log.write_pos, 8);
    }

    #[test]
    fn recover_rejects_file_with_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.log");
        fs::write(&path, b"XXXXv1\x00\x00\x00").unwrap();
        let err = ManifestLog::recover(&path).err().unwrap();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn recover_rejects_file_with_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v99.log");
        fs::write(&path, b"MKMF\x63\x00\x00\x00").unwrap();
        let err = ManifestLog::recover(&path).err().unwrap();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn recover_replays_existing_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("populated.log");

        // Write a header using the helper directly.
        {
            let mut file = File::create(&path).unwrap();
            write_header(&mut file).unwrap();
            file.sync_all().unwrap();
        }

        // Reopen via recover, append an entry, reopen again.
        let (_, mut log) = ManifestLog::recover(&path).unwrap();
        log.append(&ManifestEntry::SetSequence {
            next_sequence: SeqNo(42),
        })
        .unwrap();
        drop(log);

        let (manifest, _log) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.next_sequence(), SeqNo(42));
    }
}
