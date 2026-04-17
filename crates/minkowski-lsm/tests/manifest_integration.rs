//! Integration tests for the manifest system: flush_and_record, replay, cleanup.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use minkowski::World;
use minkowski_lsm::error::LsmError;
use minkowski_lsm::manifest_log::{ManifestEntry, ManifestLog};
use minkowski_lsm::manifest_ops::{cleanup_orphans, flush_and_record};
use minkowski_lsm::types::{Level, SeqNo};

#[derive(Clone, Copy)]
#[expect(dead_code)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
#[expect(dead_code)]
struct Vel {
    dx: f32,
    dy: f32,
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

#[test]
fn three_flushes_then_replay() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();

    // Flush 1: spawn some entities.
    for i in 0..5 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }
    let p1 = flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path())
        .unwrap()
        .unwrap();
    world.clear_all_dirty_pages();

    // Flush 2: spawn more.
    for i in 5..10 {
        world.spawn((Pos {
            x: i as f32,
            y: 1.0,
        },));
    }
    let p2 = flush_and_record(&world, (10, 20), &mut manifest, &mut log, dir.path())
        .unwrap()
        .unwrap();
    world.clear_all_dirty_pages();

    // Flush 3: spawn with a different archetype.
    world.spawn((Vel { dx: 1.0, dy: 2.0 },));
    let p3 = flush_and_record(&world, (20, 30), &mut manifest, &mut log, dir.path())
        .unwrap()
        .unwrap();

    assert_eq!(manifest.total_runs(), 3);
    assert_eq!(manifest.next_sequence(), SeqNo(30));
    assert!(p1.exists());
    assert!(p2.exists());
    assert!(p3.exists());

    // Replay the log from scratch — should reconstruct identical state.
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 3);
    assert_eq!(recovered.next_sequence(), SeqNo(30));
    assert_eq!(recovered.runs_at_level(Level::L0).len(), 3);

    // Verify run metadata matches.
    for (original, replayed) in manifest
        .runs_at_level(Level::L0)
        .iter()
        .zip(recovered.runs_at_level(Level::L0).iter())
    {
        assert_eq!(original.path(), replayed.path());
        assert_eq!(original.sequence_range(), replayed.sequence_range());
        assert_eq!(original.page_count(), replayed.page_count());
    }
}

// ── Crash recovery ──────────────────────────────────────────────────────────

#[test]
fn corrupt_tail_partial_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 },));

    // Two good flushes.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();
    world.clear_all_dirty_pages();

    world.spawn((Pos { x: 3.0, y: 4.0 },));
    flush_and_record(&world, (10, 20), &mut manifest, &mut log, dir.path()).unwrap();

    // Append garbage to simulate a torn write.
    {
        let mut f = fs::OpenOptions::new().append(true).open(&log_path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x42]).unwrap();
    }

    // Replay should recover the 2 good entries (each flush writes one atomic AddRunAndSequence entry).
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 2);
    assert_eq!(recovered.next_sequence(), SeqNo(20));
}

/// Truncate the log to every byte prefix 0..=file_len and replay each time.
///
/// For an append-only CRC-framed log, this is the canonical crash-coverage
/// test: any prefix must replay into a valid manifest, and extending the
/// prefix can only add information (monotonicity). Covers mid-header and
/// mid-payload truncation — the `read_exact` path that previously propagated
/// `UnexpectedEof` as fatal is exercised at every frame boundary.
#[test]
fn replay_converges_at_every_truncation_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    for i in 0..3u64 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
        let lo = i * 10;
        let hi = lo + 10;
        flush_and_record(&world, (lo, hi), &mut manifest, &mut log, dir.path()).unwrap();
        world.clear_all_dirty_pages();
    }

    let full_bytes = fs::read(&log_path).unwrap();
    assert!(!full_bytes.is_empty(), "log must have content");

    let mut prev_total_runs = 0usize;
    let mut prev_next_seq = SeqNo(0);

    for truncate_len in 0..=full_bytes.len() {
        let truncated_path = dir.path().join(format!("truncated_{truncate_len:05}.log"));
        fs::write(&truncated_path, &full_bytes[..truncate_len]).unwrap();

        if truncate_len < 8 {
            // Header missing or truncated: recover must return a
            // Format error; no manifest is produced.
            let err = ManifestLog::recover(&truncated_path).err().unwrap();
            assert!(
                matches!(err, LsmError::Format(_)),
                "truncate_len={truncate_len}: expected Format, got {err:?}"
            );
            continue;
        }

        // truncate_len == 8: valid header, no frames — first Ok case,
        // returns empty manifest. Subsequent iterations accumulate runs
        // as frame boundaries are crossed.
        let (replayed, _) = ManifestLog::recover(&truncated_path)
            .unwrap_or_else(|e| panic!("recover failed at truncate_len={truncate_len}: {e:?}"));

        assert!(
            replayed.total_runs() <= 3,
            "truncate_len={truncate_len}: total_runs={} > 3",
            replayed.total_runs()
        );

        // Monotonicity: extending the prefix never loses state.
        assert!(
            replayed.total_runs() >= prev_total_runs,
            "truncate_len={truncate_len}: total_runs rewound {prev_total_runs} -> {}",
            replayed.total_runs()
        );
        assert!(
            replayed.next_sequence() >= prev_next_seq,
            "truncate_len={truncate_len}: next_sequence rewound {prev_next_seq:?} -> {:?}",
            replayed.next_sequence()
        );
        prev_total_runs = replayed.total_runs();
        prev_next_seq = replayed.next_sequence();
    }

    assert_eq!(prev_total_runs, 3, "full replay must recover all 3 runs");
    assert_eq!(
        prev_next_seq,
        SeqNo(30),
        "full replay must recover final sequence"
    );
}

/// Replay must truncate the log when `apply_entry` fails, not silently skip
/// the bad entry.
#[test]
fn replay_truncates_log_on_promote_of_missing_run() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // Real flush: produces a genuine AddRunAndSequence entry.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    // Inject a PromoteRun that references a path the manifest doesn't know.
    // Models a corrupted log or an out-of-order mutation.
    log.append(&ManifestEntry::PromoteRun {
        from_level: Level::L0,
        to_level: Level::L1,
        path: PathBuf::from("ghost.run"),
    })
    .unwrap();
    // Anything after the bad entry must be discarded on replay.
    log.append(&ManifestEntry::SetSequence {
        next_sequence: SeqNo(999),
    })
    .unwrap();
    drop(log);

    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the first flush should survive replay"
    );
    assert!(
        recovered.next_sequence() < SeqNo(999),
        "SetSequence past the bad PromoteRun must not apply"
    );
    // The surviving AddRunAndSequence set next_sequence to 10.
    assert_eq!(recovered.next_sequence(), SeqNo(10));
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

#[test]
fn cleanup_removes_orphans_and_tmp() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 },));

    // One real flush.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    // Create orphan files.
    fs::write(dir.path().join("999-1000.run"), b"orphan").unwrap();
    fs::write(dir.path().join("crash.run.tmp"), b"incomplete").unwrap();

    let deleted = cleanup_orphans(dir.path(), &manifest).unwrap();
    assert_eq!(deleted, 2);

    // The real run file should still exist.
    assert_eq!(manifest.total_runs(), 1);
    let run_path = manifest.runs_at_level(Level::L0)[0].path();
    assert!(run_path.exists());
}

// ── Decode validation → tail truncation ─────────────────────────────────────

/// Regression: a frame whose decoded SortedRunMeta fails validation
/// (unsorted archetype_coverage) must be treated as tail garbage, not
/// propagated as a fatal error. Wires the SortedRunMeta::new invariant
/// check into the existing torn-tail recovery path.
#[test]
fn replay_truncates_log_on_unsorted_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // One real flush, produces a valid AddRunAndSequence frame.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    let len_after_first_frame = fs::metadata(&log_path).unwrap().len();

    // Manually craft an AddRun frame with unsorted archetype_coverage.
    // Bypass SortedRunMeta::new (can't call it — would error) by encoding
    // the bytes directly. Wire layout per manifest_log.rs::encode_entry.
    let mut payload = Vec::new();
    // TAG_ADD_RUN = 0x01 (see manifest_log.rs::encode_entry AddRun branch)
    payload.push(0x01);
    payload.push(0); // level
    // path: "x.run"
    let path_bytes = b"x.run";
    payload.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(path_bytes);
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_lo
    payload.extend_from_slice(&10u64.to_le_bytes()); // seq_hi
    // archetype_coverage: [3, 1] — intentionally unsorted (rejects invariant)
    payload.extend_from_slice(&2u16.to_le_bytes()); // count
    payload.extend_from_slice(&3u16.to_le_bytes());
    payload.extend_from_slice(&1u16.to_le_bytes());
    payload.extend_from_slice(&1u64.to_le_bytes()); // page_count
    payload.extend_from_slice(&1024u64.to_le_bytes()); // size_bytes

    // Frame format: [len: u32 LE][crc32: u32 LE][payload]
    let mut f = fs::OpenOptions::new().append(true).open(&log_path).unwrap();
    let len = payload.len() as u32;
    let crc = crc32fast::hash(&payload);
    f.write_all(&len.to_le_bytes()).unwrap();
    f.write_all(&crc.to_le_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.sync_all().unwrap();
    drop(f);

    // Replay must truncate back to end of first valid frame.
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the valid first flush survives"
    );

    let len_after_replay = fs::metadata(&log_path).unwrap().len();
    assert_eq!(
        len_after_replay, len_after_first_frame,
        "replay truncated the bad frame"
    );
}

/// Regression: a frame with a valid CRC but an invalid level byte
/// (>= NUM_LEVELS) must be treated as tail garbage. Level::new returns
/// None on invalid input, decode_entry surfaces LsmError::Format, and
/// the replay loop must truncate.
#[test]
fn replay_truncates_log_on_invalid_level_byte() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    let len_after_first_frame = fs::metadata(&log_path).unwrap().len();

    // Craft a REMOVE_RUN frame with level=255 (invalid; NUM_LEVELS is 4).
    // REMOVE_RUN is the simplest level-bearing entry to fabricate.
    let mut payload = Vec::new();
    // TAG_REMOVE_RUN = 0x02 (see manifest_log.rs::encode_entry RemoveRun branch)
    payload.push(0x02);
    payload.push(255); // invalid level byte
    let path_bytes = b"ghost.run";
    payload.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(path_bytes);

    let mut f = fs::OpenOptions::new().append(true).open(&log_path).unwrap();
    let len = payload.len() as u32;
    let crc = crc32fast::hash(&payload);
    f.write_all(&len.to_le_bytes()).unwrap();
    f.write_all(&crc.to_le_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.sync_all().unwrap();
    drop(f);

    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the valid first flush survives"
    );

    let len_after_replay = fs::metadata(&log_path).unwrap().len();
    assert_eq!(
        len_after_replay, len_after_first_frame,
        "replay truncated the invalid-level frame"
    );
}

/// Regression: a frame whose `SeqRange::new` call fails (seq_lo > seq_hi)
/// must be treated as tail garbage, not propagated as a fatal error.
/// Completes the three-error regression symmetry alongside
/// `replay_truncates_log_on_unsorted_coverage` and
/// `replay_truncates_log_on_invalid_level_byte`.
#[test]
fn replay_truncates_log_on_inverted_seq_range() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // One real flush, produces a valid AddRunAndSequence frame.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    let len_after_first_frame = fs::metadata(&log_path).unwrap().len();

    // Manually craft an AddRun frame with seq_lo=20 > seq_hi=10, which
    // makes SeqRange::new fail inside decode_entry → LsmError::Format.
    // Bypass SortedRunMeta::new by encoding the bytes directly.
    let mut payload = Vec::new();
    // TAG_ADD_RUN = 0x01
    payload.push(0x01);
    payload.push(0); // level (L0)
    // path: "bad.run"
    let path_bytes = b"bad.run";
    payload.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(path_bytes);
    payload.extend_from_slice(&20u64.to_le_bytes()); // seq_lo  (intentionally > seq_hi)
    payload.extend_from_slice(&10u64.to_le_bytes()); // seq_hi
    // archetype_coverage: [0] — valid (just one entry)
    payload.extend_from_slice(&1u16.to_le_bytes()); // count
    payload.extend_from_slice(&0u16.to_le_bytes()); // arch_id
    payload.extend_from_slice(&1u64.to_le_bytes()); // page_count
    payload.extend_from_slice(&1024u64.to_le_bytes()); // size_bytes

    // Frame format: [len: u32 LE][crc32: u32 LE][payload]
    let mut f = fs::OpenOptions::new().append(true).open(&log_path).unwrap();
    let len = payload.len() as u32;
    let crc = crc32fast::hash(&payload);
    f.write_all(&len.to_le_bytes()).unwrap();
    f.write_all(&crc.to_le_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.sync_all().unwrap();
    drop(f);

    // Replay must truncate back to end of first valid frame.
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the valid first flush survives"
    );

    let len_after_replay = fs::metadata(&log_path).unwrap().len();
    assert_eq!(
        len_after_replay, len_after_first_frame,
        "replay truncated the bad frame"
    );
}

/// Regression: a RemoveRun frame referencing a path the manifest doesn't
/// know is log corruption. apply_entry must propagate the error so replay
/// treats the rest of the log as tail garbage — same policy as PromoteRun.
#[test]
fn replay_truncates_log_on_remove_of_missing_run() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // One real flush — produces a valid AddRunAndSequence entry.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    // Inject a RemoveRun referencing a path the manifest doesn't know.
    log.append(&ManifestEntry::RemoveRun {
        level: Level::L0,
        path: PathBuf::from("ghost.run"),
    })
    .unwrap();
    // Anything after the bad entry must be discarded on replay.
    log.append(&ManifestEntry::SetSequence {
        next_sequence: SeqNo(999),
    })
    .unwrap();
    drop(log);

    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the valid first flush should survive"
    );
    // The trailing SetSequence must not have been applied.
    assert!(
        recovered.next_sequence() < SeqNo(999),
        "SetSequence past the bad RemoveRun must not apply"
    );
    assert_eq!(recovered.next_sequence(), SeqNo(10));
}

// ── recover() lifecycle and rejection regressions ───────────────────────────

/// A recover -> flush -> recover round trip reconstructs identical state.
/// Exercises the full lifecycle: open, write frames, close, reopen,
/// verify per-run metadata matches.
#[test]
fn recover_then_flush_then_recover_roundtrips_state() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");

    // Fresh recover creates the file.
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(manifest.total_runs(), 0);

    // Two flushes produce two AddRunAndSequence frames.
    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();
    world.clear_all_dirty_pages();
    world.spawn((Pos { x: 2.0, y: 0.0 },));
    flush_and_record(&world, (10, 20), &mut manifest, &mut log, dir.path()).unwrap();
    drop(log);

    // Second recover replays both entries.
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 2);
    assert_eq!(recovered.next_sequence(), SeqNo(20));

    // Metadata round-trips faithfully.
    for (orig, rec) in manifest
        .runs_at_level(Level::L0)
        .iter()
        .zip(recovered.runs_at_level(Level::L0).iter())
    {
        assert_eq!(orig.path(), rec.path());
        assert_eq!(orig.sequence_range(), rec.sequence_range());
    }
}

/// A file without the 8-byte magic+version header must be rejected
/// with a Format error. Documents the strict-reject compatibility
/// policy for legacy headerless logs.
#[test]
fn recover_rejects_file_without_header() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("legacy.log");

    // Write raw bytes that look like a frame length prefix (not a
    // header) — what a legacy headerless log would look like byte-for-byte.
    fs::write(&log_path, [0x20, 0x00, 0x00, 0x00, 0xAB, 0xCD, 0xEF, 0x12]).unwrap();

    let err = ManifestLog::recover(&log_path).err().unwrap();
    assert!(
        matches!(err, LsmError::Format(ref msg) if msg.contains("bad magic")),
        "expected bad-magic Format error, got {err:?}"
    );
}

/// A file with valid magic but an unrecognized version byte must be
/// rejected (forward-compat gate: an older binary reading a newer file
/// must fail loudly, not silently decode garbage).
#[test]
fn recover_rejects_file_with_unsupported_version() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("future.log");

    fs::write(&log_path, b"MKMF\x63\x00\x00\x00").unwrap(); // version 0x63

    let err = ManifestLog::recover(&log_path).err().unwrap();
    assert!(
        matches!(err, LsmError::Format(ref msg) if msg.contains("unsupported manifest version")),
        "expected version-mismatch Format error, got {err:?}"
    );
}

// ── Clean world ─────────────────────────────────────────────────────────────

#[test]
fn flush_and_record_clean_world_no_change() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 },));
    world.clear_all_dirty_pages();

    let result = flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();
    assert!(result.is_none());
    assert_eq!(manifest.total_runs(), 0);
    assert_eq!(manifest.next_sequence(), SeqNo(0));

    // Log should be empty — recover produces empty manifest.
    let (replayed, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(replayed.total_runs(), 0);
}
