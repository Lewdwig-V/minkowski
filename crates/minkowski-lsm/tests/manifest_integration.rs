//! Integration tests for the manifest system: flush_and_record, replay, cleanup.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use minkowski::World;
use minkowski_lsm::manifest::LsmManifest;
use minkowski_lsm::manifest_log::{ManifestEntry, ManifestLog};
use minkowski_lsm::manifest_ops::{cleanup_orphans, flush_and_record};

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
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

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
    assert_eq!(manifest.next_sequence(), 30);
    assert!(p1.exists());
    assert!(p2.exists());
    assert!(p3.exists());

    // Replay the log from scratch — should reconstruct identical state.
    let replayed = ManifestLog::replay(&log_path).unwrap();
    assert_eq!(replayed.total_runs(), 3);
    assert_eq!(replayed.next_sequence(), 30);
    assert_eq!(replayed.runs_at_level(0).len(), 3);

    // Verify run metadata matches.
    for (original, recovered) in manifest
        .runs_at_level(0)
        .iter()
        .zip(replayed.runs_at_level(0).iter())
    {
        assert_eq!(original.path(), recovered.path());
        assert_eq!(original.sequence_range(), recovered.sequence_range());
        assert_eq!(original.page_count(), recovered.page_count());
    }
}

// ── Crash recovery ──────────────────────────────────────────────────────────

#[test]
fn corrupt_tail_partial_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

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
    let recovered = ManifestLog::replay(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 2);
    assert_eq!(recovered.next_sequence(), 20);
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
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

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
    let mut prev_next_seq = 0u64;

    for truncate_len in 0..=full_bytes.len() {
        let truncated_path = dir.path().join(format!("truncated_{truncate_len:05}.log"));
        fs::write(&truncated_path, &full_bytes[..truncate_len]).unwrap();

        let replayed = ManifestLog::replay(&truncated_path)
            .unwrap_or_else(|e| panic!("replay failed at truncate_len={truncate_len}: {e:?}"));

        assert!(
            replayed.total_runs() <= 3,
            "truncate_len={truncate_len}: total_runs={} > 3",
            replayed.total_runs()
        );

        // Monotonicity: extending the prefix never loses state.
        assert!(
            replayed.total_runs() >= prev_total_runs,
            "truncate_len={truncate_len}: total_runs rewound {prev_total_runs} → {}",
            replayed.total_runs()
        );
        assert!(
            replayed.next_sequence() >= prev_next_seq,
            "truncate_len={truncate_len}: next_sequence rewound {prev_next_seq} → {}",
            replayed.next_sequence()
        );
        prev_total_runs = replayed.total_runs();
        prev_next_seq = replayed.next_sequence();
    }

    assert_eq!(prev_total_runs, 3, "full replay must recover all 3 runs");
    assert_eq!(prev_next_seq, 30, "full replay must recover final sequence");
}

/// Replay must truncate the log when `apply_entry` fails, not silently skip
/// the bad entry. Previously `let _ = promote_run(...)` ate the error; the
/// fix makes replay treat the rest of the log as tail garbage.
#[test]
fn replay_truncates_log_on_promote_of_missing_run() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // Real flush: produces a genuine AddRunAndSequence entry.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    // Inject a PromoteRun that references a path the manifest doesn't know.
    // Models a corrupted log or an out-of-order mutation.
    log.append(&ManifestEntry::PromoteRun {
        from_level: 0,
        to_level: 1,
        path: PathBuf::from("ghost.run"),
    })
    .unwrap();
    // Anything after the bad entry must be discarded on replay.
    log.append(&ManifestEntry::SetSequence { next_sequence: 999 })
        .unwrap();
    drop(log);

    let recovered = ManifestLog::replay(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the first flush should survive replay"
    );
    assert!(
        recovered.next_sequence() < 999,
        "SetSequence past the bad PromoteRun must not apply"
    );
    // The surviving AddRunAndSequence set next_sequence to 10.
    assert_eq!(recovered.next_sequence(), 10);
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

#[test]
fn cleanup_removes_orphans_and_tmp() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

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
    let run_path = manifest.runs_at_level(0)[0].path();
    assert!(run_path.exists());
}

// ── Clean world ─────────────────────────────────────────────────────────────

#[test]
fn flush_and_record_clean_world_no_change() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 },));
    world.clear_all_dirty_pages();

    let result = flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();
    assert!(result.is_none());
    assert_eq!(manifest.total_runs(), 0);
    assert_eq!(manifest.next_sequence(), 0);

    // Log should be empty — replay produces empty manifest.
    let replayed = ManifestLog::replay(&log_path).unwrap();
    assert_eq!(replayed.total_runs(), 0);
}
