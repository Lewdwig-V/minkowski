//! End-to-end integration tests for the compactor public API.
//!
//! All tests exercise the full `flush → compact → verify` path through
//! `compact_one` / `compact_one_observed` / `LsmManifest::needs_compaction`.
//! Only the public API surface is used — no crate-internal helpers.

use std::collections::{HashMap, HashSet};
use std::fs;

use minkowski::World;
use minkowski_lsm::compactor::{compact_one, compact_one_observed};
use minkowski_lsm::format::ENTITY_SLOT;
use minkowski_lsm::manifest::LsmManifest;
use minkowski_lsm::manifest_log::{ManifestLog, ManifestTag};
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::reader::SortedRunReader;
use minkowski_lsm::types::{Level, SeqNo, SeqRange};
use minkowski_lsm::writer::EntityKey;

// ── Component types ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pos {
    x: f32,
    y: f32,
}

// ── Helper ───────────────────────────────────────────────────────────────────

/// Do one flush from `world` (which must have dirty pages) and return the run path.
fn do_flush<const N: usize>(
    world: &World,
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    run_dir: &std::path::Path,
    seq_lo: u64,
    seq_hi: u64,
) -> std::path::PathBuf {
    let seq_range = SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi)).unwrap();
    flush_and_record(world, seq_range, manifest, log, run_dir)
        .unwrap()
        .expect("world must be dirty for flush to produce Some")
}

/// Collect all entity IDs from the entity-slot pages of arch_id 0 in a reader.
fn collect_entity_ids(reader: &SortedRunReader) -> HashSet<u64> {
    let mut found = HashSet::new();
    let mut page_idx: u16 = 0;
    while let Ok(Some(page)) = reader.get_page(0, ENTITY_SLOT, page_idx) {
        let row_count = page.header().row_count as usize;
        let data = page.data();
        for r in 0..row_count {
            let off = r * 8;
            let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            found.insert(id);
        }
        page_idx += 1;
    }
    found
}

/// Collect (entity_id → (x, y)) from arch_id 0, slot 0 (the Pos column).
///
/// Entity IDs come from ENTITY_SLOT pages; Pos values come from slot-0 pages.
/// Both page sequences are in the same row order, so we zip them by page + row.
fn collect_pos_values(reader: &SortedRunReader) -> HashMap<u64, (f32, f32)> {
    let mut result = HashMap::new();
    let mut page_idx: u16 = 0;
    while let Ok(Some(entity_page)) = reader.get_page(0, ENTITY_SLOT, page_idx) {
        if let Ok(Some(pos_page)) = reader.get_page(0, 0, page_idx) {
            let row_count = entity_page.header().row_count as usize;
            let edata = entity_page.data();
            let pdata = pos_page.data();
            for r in 0..row_count {
                let eid = u64::from_le_bytes(edata[r * 8..r * 8 + 8].try_into().unwrap());
                // Pos is { x: f32, y: f32 } — 4 bytes each, total 8 bytes per row.
                let x = f32::from_le_bytes(pdata[r * 8..r * 8 + 4].try_into().unwrap());
                let y = f32::from_le_bytes(pdata[r * 8 + 4..r * 8 + 8].try_into().unwrap());
                result.insert(eid, (x, y));
            }
        }
        page_idx += 1;
    }
    result
}

// ── Test 1: flush × 4 → compact → L0 empty, L1 has one run ─────────────────

#[test]
fn flush_four_times_then_compact_consolidates_l0_to_l1() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // Flush 4 times into L0.
    for i in 0..4u64 {
        let mut world = World::new();
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
        do_flush(
            &world,
            &mut manifest,
            &mut log,
            dir.path(),
            i * 10,
            i * 10 + 9,
        );
    }
    assert_eq!(
        manifest.runs_at_level(Level::L0).len(),
        4,
        "pre: 4 runs at L0"
    );
    assert_eq!(
        manifest.runs_at_level(Level::L1).len(),
        0,
        "pre: 0 runs at L1"
    );

    // Compact.
    let report = compact_one(&mut manifest, &mut log, dir.path())
        .unwrap()
        .expect("K=4 runs must trigger compaction");

    assert_eq!(report.from_level, Level::L0);
    assert_eq!(report.to_level, Level::L1);
    assert_eq!(report.input_run_count, 4);

    assert_eq!(
        manifest.runs_at_level(Level::L0).len(),
        0,
        "post: L0 must be empty"
    );
    assert_eq!(
        manifest.runs_at_level(Level::L1).len(),
        1,
        "post: exactly one L1 run"
    );
}

// ── Test 2: compacted output contains all 40 entities from 4 flushes ────────

#[test]
fn compact_preserves_all_entities() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // 4 flushes × 10 entities each = 40 entities total. Use separate worlds
    // with offset entity allocation so IDs do not overlap.
    let entities_per_flush = 10usize;
    let n_flushes = 4usize;
    let mut all_bits: HashSet<u64> = HashSet::new();

    for flush_i in 0..n_flushes {
        let mut world = World::new();
        // Advance the allocator so each world produces distinct entity IDs.
        let skip = flush_i * entities_per_flush;
        let placeholders: Vec<_> = (0..skip)
            .map(|_| world.spawn((Pos { x: 0.0, y: 0.0 },)))
            .collect();
        for ph in placeholders {
            world.despawn(ph);
        }
        for j in 0..entities_per_flush {
            let e = world.spawn((Pos {
                x: (flush_i * 100 + j) as f32,
                y: (flush_i * 100 + j) as f32,
            },));
            all_bits.insert(e.to_bits());
        }
        let lo = (flush_i as u64) * 10;
        let hi = lo + 9;
        do_flush(&world, &mut manifest, &mut log, dir.path(), lo, hi);
    }

    assert_eq!(
        all_bits.len(),
        n_flushes * entities_per_flush,
        "entity IDs must all be unique across worlds"
    );

    // Compact.
    let report = compact_one(&mut manifest, &mut log, dir.path())
        .unwrap()
        .expect("K=4 runs must trigger compaction");

    // Open output and verify all 40 entities are present.
    let reader = SortedRunReader::open(&report.output_path).unwrap();
    let found = collect_entity_ids(&reader);

    assert_eq!(
        found.len(),
        n_flushes * entities_per_flush,
        "compacted run must contain all {} entities",
        n_flushes * entities_per_flush
    );
    for bits in &all_bits {
        assert!(
            found.contains(bits),
            "entity {bits:#x} is missing from the compacted output"
        );
    }
}

// ── Test 3: entity updated across flushes → newest version wins ─────────────

#[test]
fn compact_with_entity_updates_keeps_newest_version() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // We need 4 L0 runs to trigger compaction.  We'll use two "background"
    // entities with stable positions, plus one "hero" entity E that gets
    // updated across the first two flushes.  The last two flushes use a
    // different world (no dirty pages for E) so we flush the background pair
    // again to reach the K=4 threshold.
    //
    // Concretely:
    //   Flush 0: background entity B (Pos(0,0)), hero E first value (1.0, 2.0)
    //   Flush 1: hero E updated to (10.0, 20.0) [same entity bits, new dirty page]
    //   Flush 2 & 3: background-only worlds (need dirty pages)
    //
    // Post-compact: E must appear at (10.0, 20.0).

    // --- Flush 0 ---
    let mut world0 = World::new();
    let _b0 = world0.spawn((Pos { x: 0.0, y: 0.0 },));
    let hero_entity = world0.spawn((Pos { x: 1.0, y: 2.0 },));
    let hero_bits = hero_entity.to_bits();
    do_flush(&world0, &mut manifest, &mut log, dir.path(), 0, 9);

    // --- Flush 1: update hero's Pos ---
    // We create a new world where the hero entity is represented with the same
    // entity bits (by pre-allocating the same number of entities so the
    // allocator reaches the same index) and has the new component value as a
    // dirty page.
    //
    // The FlushWriter uses entity bits directly: the newest run wins on dedup.
    // Spawn 2 entities (same as world0: _b0 then hero), same positions in the
    // allocator.  Then directly record the hero's updated value.
    let mut world1 = World::new();
    let _b1 = world1.spawn((Pos { x: 0.0, y: 0.0 },)); // same index as _b0
    let hero1 = world1.spawn((Pos { x: 10.0, y: 20.0 },)); // same index as hero_entity
    // Confirm the entity bits match (same allocator advancement).
    assert_eq!(
        hero1.to_bits(),
        hero_bits,
        "hero entity must have the same bits in world1"
    );
    do_flush(&world1, &mut manifest, &mut log, dir.path(), 10, 19);

    // --- Flush 2: new entities (to get a dirty page) ---
    let mut world2 = World::new();
    world2.spawn((Pos { x: 100.0, y: 200.0 },));
    do_flush(&world2, &mut manifest, &mut log, dir.path(), 20, 29);

    // --- Flush 3: new entities ---
    let mut world3 = World::new();
    world3.spawn((Pos { x: 200.0, y: 300.0 },));
    do_flush(&world3, &mut manifest, &mut log, dir.path(), 30, 39);

    assert_eq!(manifest.runs_at_level(Level::L0).len(), 4);

    // Compact.
    let report = compact_one(&mut manifest, &mut log, dir.path())
        .unwrap()
        .expect("K=4 runs must trigger compaction");

    // Open output and locate the hero entity.
    let reader = SortedRunReader::open(&report.output_path).unwrap();
    let pos_values = collect_pos_values(&reader);

    assert!(
        pos_values.contains_key(&hero_bits),
        "hero entity must appear in compacted output"
    );
    let (x, y) = pos_values[&hero_bits];
    assert!(
        (x - 10.0_f32).abs() < f32::EPSILON && (y - 20.0_f32).abs() < f32::EPSILON,
        "hero entity must have newest value (10.0, 20.0), got ({x}, {y})"
    );
}

// ── Test 4: compact emits exactly one CompactionCommit entry ─────────────────

#[test]
fn compact_emits_exactly_one_compaction_commit_entry() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // 4 flushes.
    for i in 0..4u64 {
        let mut world = World::new();
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
        do_flush(
            &world,
            &mut manifest,
            &mut log,
            dir.path(),
            i * 10,
            i * 10 + 9,
        );
    }

    // Measure log length before compact.
    let len_before = fs::metadata(&log_path).unwrap().len();

    // Compact.
    compact_one(&mut manifest, &mut log, dir.path())
        .unwrap()
        .expect("K=4 runs must trigger compaction");

    // Drop the log handle to flush OS buffers, then read raw bytes.
    drop(log);
    let all_bytes = fs::read(&log_path).unwrap();

    // Count CompactionCommit tag bytes (0x06) in frames appended after the
    // pre-compact position. We scan payload bytes (skip frame headers) to
    // avoid false positives from length / CRC fields that happen to be 0x06.
    //
    // Frame layout: [len: u32 LE][crc32: u32 LE][payload...].
    // The tag byte is the first byte of each payload.
    let new_bytes = &all_bytes[len_before as usize..];
    let commit_tag = ManifestTag::CompactionCommit as u8;
    let mut commit_count = 0usize;

    let mut pos = 0usize;
    while pos + 8 <= new_bytes.len() {
        let frame_len = u32::from_le_bytes(new_bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start + frame_len;
        if payload_end > new_bytes.len() {
            break;
        }
        if frame_len > 0 && new_bytes[payload_start] == commit_tag {
            commit_count += 1;
        }
        pos = payload_end;
    }

    assert_eq!(
        commit_count, 1,
        "exactly one CompactionCommit frame must appear in the new log bytes"
    );
}

// ── Test 5: needs_compaction parity with compact_one ─────────────────────────

#[test]
fn needs_compaction_parity_with_compact_one() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // Empty — needs_compaction false, compact_one returns None.
    assert!(
        !manifest.needs_compaction(),
        "empty: needs_compaction must be false"
    );
    let result = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
    assert!(result.is_none(), "empty: compact_one must return None");

    // 3 runs — still below K=4.
    for i in 0..3u64 {
        let mut world = World::new();
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
        do_flush(
            &world,
            &mut manifest,
            &mut log,
            dir.path(),
            i * 10,
            i * 10 + 9,
        );
    }
    assert!(
        !manifest.needs_compaction(),
        "3 runs: needs_compaction must be false"
    );
    let result = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
    assert!(result.is_none(), "3 runs: compact_one must return None");

    // 4th run — at trigger.
    {
        let mut world = World::new();
        world.spawn((Pos { x: 99.0, y: 0.0 },));
        do_flush(&world, &mut manifest, &mut log, dir.path(), 30, 39);
    }
    assert!(
        manifest.needs_compaction(),
        "4 runs: needs_compaction must be true"
    );
    let result = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
    assert!(result.is_some(), "4 runs: compact_one must return Some");

    // Post-compact — L0 empty, L1 has 1 run; needs_compaction false again.
    assert!(
        !manifest.needs_compaction(),
        "post-compact: needs_compaction must be false"
    );
    let result = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
    assert!(
        result.is_none(),
        "post-compact: compact_one must return None"
    );
}

// ── Test 6: compact → drop → recover roundtrips post-compaction state ────────

#[test]
fn compact_then_recover_roundtrips_state() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // 4 flushes into L0, with distinct entity IDs across worlds (advance allocator).
    let entities_per_flush = 2usize;
    let n_flushes = 4usize;
    let mut all_bits: HashSet<u64> = HashSet::new();

    for flush_i in 0..n_flushes {
        let mut world = World::new();
        // Advance allocator so each world produces unique entity IDs.
        let skip = flush_i * entities_per_flush;
        let phs: Vec<_> = (0..skip)
            .map(|_| world.spawn((Pos { x: 0.0, y: 0.0 },)))
            .collect();
        for ph in phs {
            world.despawn(ph);
        }
        for j in 0..entities_per_flush {
            let e = world.spawn((Pos {
                x: (flush_i * 10 + j) as f32,
                y: 0.0,
            },));
            all_bits.insert(e.to_bits());
        }
        let lo = (flush_i as u64) * 10;
        do_flush(&world, &mut manifest, &mut log, dir.path(), lo, lo + 9);
    }

    assert_eq!(manifest.runs_at_level(Level::L0).len(), n_flushes);

    // Compact.
    let report = compact_one(&mut manifest, &mut log, dir.path())
        .unwrap()
        .expect("K=4 runs must trigger compaction");

    let output_path = report.output_path.clone();

    // Drop everything and recover from scratch.
    drop(log);
    drop(manifest);

    let (recovered, _) = ManifestLog::recover::<4>(&log_path).unwrap();

    assert_eq!(
        recovered.runs_at_level(Level::L0).len(),
        0,
        "recovered manifest must have 0 runs at L0"
    );
    assert_eq!(
        recovered.runs_at_level(Level::L1).len(),
        1,
        "recovered manifest must have 1 run at L1"
    );
    assert_eq!(
        recovered.runs_at_level(Level::L1)[0].path(),
        output_path.as_path(),
        "recovered L1 run path must match the compacted output"
    );
    assert!(
        !recovered.needs_compaction(),
        "recovered manifest must not need compaction"
    );

    // Output run file must still be readable and contain all entities.
    let reader = SortedRunReader::open(&output_path).unwrap();
    let found = collect_entity_ids(&reader);
    assert_eq!(
        found.len(),
        n_flushes * entities_per_flush,
        "compacted run must contain all {} entities",
        n_flushes * entities_per_flush
    );
    for bits in &all_bits {
        assert!(
            found.contains(bits),
            "entity {bits:#x} missing from recovered compacted run"
        );
    }
}

// ── Test 7: compact_one_observed fires once per output entity ────────────────
//
// This exercises the public compact_one_observed API end-to-end.

#[test]
fn compact_one_observed_fires_for_each_output_entity() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

    // 4 flushes × 3 entities each = 12 total.
    let entities_per_flush = 3usize;
    let n_flushes = 4usize;
    let mut all_bits: HashSet<u64> = HashSet::new();

    for flush_i in 0..n_flushes {
        let mut world = World::new();
        let skip = flush_i * entities_per_flush;
        let phs: Vec<_> = (0..skip)
            .map(|_| world.spawn((Pos { x: 0.0, y: 0.0 },)))
            .collect();
        for ph in phs {
            world.despawn(ph);
        }
        for j in 0..entities_per_flush {
            let e = world.spawn((Pos {
                x: (flush_i * 10 + j) as f32,
                y: 0.0,
            },));
            all_bits.insert(e.to_bits());
        }
        let lo = (flush_i as u64) * 10;
        do_flush(&world, &mut manifest, &mut log, dir.path(), lo, lo + 9);
    }

    let observed: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));
    let observed_clone = Rc::clone(&observed);

    let report = compact_one_observed(
        &mut manifest,
        &mut log,
        dir.path(),
        Some(&mut |key: EntityKey| {
            observed_clone.borrow_mut().push(key.0);
        }),
    )
    .unwrap()
    .expect("K=4 runs must produce Some(report)");

    assert_eq!(report.from_level, Level::L0);

    let seen = observed.borrow();
    assert_eq!(
        seen.len(),
        n_flushes * entities_per_flush,
        "observer must fire exactly once per output entity (expected {}, got {})",
        n_flushes * entities_per_flush,
        seen.len()
    );
    for bits in &all_bits {
        assert!(
            seen.contains(bits),
            "entity {bits:#x} was not observed by compact_one_observed"
        );
    }
}
