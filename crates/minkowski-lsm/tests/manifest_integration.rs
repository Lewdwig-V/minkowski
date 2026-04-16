//! Integration tests for the manifest system: flush_and_record, replay, cleanup.

use std::fs;
use std::io::Write;

use minkowski::World;
use minkowski_lsm::manifest::LsmManifest;
use minkowski_lsm::manifest_log::ManifestLog;
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
