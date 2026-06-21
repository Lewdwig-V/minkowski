//! LSM level-count sweep harness.
//!
//! A plain data-collection `main()` (NOT a criterion benchmark). For each level
//! count `N` ∈ {3, 4, 5, 7} it drives a **growing** workload — each flush spawns
//! new entities (dirty-page flush, so each run carries only the new rows), so the
//! dataset accumulates and cascades through LSM levels. A constant dataset is
//! useless here: it just re-supersedes itself (uniform 0.25 write-amp) and never
//! reaches deep levels, so `N` is unobservable. With growth, every compaction
//! rewrites real (non-superseded) data, so cumulative write-amp reflects how many
//! levels the data passes through — which is what `N` controls.
//!
//! Per `N` it reports cumulative write-amp (exact, from each `CompactionReport`),
//! the total surviving run count (the recovery-cost signal — the bottom level
//! accumulates uncompacted runs), and a timed full `LsmRecovery::recover::<N>`:
//!
//! ```text
//! cargo bench -p minkowski-persist --bench lsm_level_sweep 2>&1 | grep SWEEP
//! ```

use minkowski_lsm::bench_support::{Layout, Shape, WorkloadParams, WriteAmp, build_world, grow};
use minkowski_lsm::compactor::compact_one;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::recovery::LsmRecovery;
use minkowski_lsm::types::{Level, SeqNo, SeqRange};
use std::time::{Duration, Instant};

/// Drive `flushes` growing flush rounds with level count `N`: each round (after
/// the first) spawns `per_flush` new entities, flushes, then compacts greedily to
/// convergence (`compact_one` returns `Ok(None)` once no `(level, archetype)`
/// group has `>= COMPACTION_TRIGGER` runs). Write-amp is accumulated exactly from
/// each `CompactionReport`. Returns `(write_amp, recovery_time, total_runs)`.
fn sweep_run<const N: usize>(
    params: &WorkloadParams,
    per_flush: usize,
    flushes: u64,
) -> (WriteAmp, Duration, usize) {
    let (mut world, codecs) = build_world(params);
    let mut count = params.entities;
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<N>(&log_path).unwrap();
    let mut wa = WriteAmp::default();

    for k in 0..flushes {
        if k > 0 {
            grow(&mut world, params, count, per_flush, 3000 + k);
            count += per_flush;
        }
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(k * 100), SeqNo::from((k + 1) * 100)).unwrap(),
            &mut manifest,
            &mut log,
            dir.path(),
            &codecs,
        )
        .unwrap()
        .expect("world is dirty, flush must produce a run");

        while let Some(r) = compact_one::<N>(&mut manifest, &mut log, dir.path()).unwrap() {
            wa.input_bytes += r.input_bytes;
            wa.output_bytes += r.output_bytes;
        }
    }

    // Surviving runs across all levels: the bottom level (never compacted upward)
    // accumulates, so a shallower N leaves more bottom-level runs → more to merge
    // on recovery.
    let total_runs: usize = (0..N as u8)
        .filter_map(Level::new)
        .map(|lvl| manifest.runs_at_level(lvl).len())
        .sum();

    let start = Instant::now();
    let (result, _, _) = LsmRecovery::recover::<N>(dir.path(), &log_path, &codecs).unwrap();
    let dur = start.elapsed();
    std::hint::black_box(&result);
    (wa, dur, total_runs)
}

/// Instantiate `sweep_run` for each level count `N` (a const generic, so explicit).
macro_rules! sweep_all {
    ($params:expr, $per_flush:expr, $flushes:expr, $($n:literal),+ $(,)?) => {
        $(
            let (wa, dur, runs) = sweep_run::<$n>($params, $per_flush, $flushes);
            eprintln!(
                "SWEEP N={} write_amp={:.4} recover_us={} total_runs={}",
                $n,
                wa.ratio(),
                dur.as_micros(),
                runs,
            );
        )+
    };
}

fn main() {
    // Growing, unique-entity workload (no supersession): every flush adds new
    // rows, so compaction rewrites real data and write-amp tracks cascade depth.
    let params = WorkloadParams {
        entities: 500,
        shape: Shape::Pod,
        layout: Layout::Fragmented,
        sparse: false,
        seed: 21,
    };
    // 64 flushes of +500 reaches the L2→L3 boundary (with K=4: 64 L0 → 16 L1 →
    // 4 L2 → 1 L3), the depth where N=3 (caps at L2) first diverges from N>=4.
    sweep_all!(&params, 500, 64, 3, 4, 5, 7);
}
