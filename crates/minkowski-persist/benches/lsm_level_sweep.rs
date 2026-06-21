//! LSM level / compaction-trigger sweep harness.
//!
//! This is a plain data-collection `main()` (NOT a criterion benchmark): for
//! each compaction trigger `N` ∈ {3, 4, 5, 7} and overwrite ratio
//! `ow` ∈ {0.0, 0.25, 1.0}, it drives a fixed number of flush rounds, compacts
//! greedily to convergence after each flush, accumulates exact write-amp from
//! every `CompactionReport`, and times a full `LsmRecovery::recover::<N>`.
//!
//! Output is a set of `SWEEP N=.. ow=.. write_amp=.. recover_us=..` lines on
//! stderr (one per `(N, ow)` pair) for the audit tooling to scrape:
//!
//! ```text
//! cargo bench -p minkowski-persist --bench lsm_level_sweep 2>&1 | grep SWEEP
//! ```
//!
//! `N` is a const generic, so each instantiation is explicit (via `sweep_all!`)
//! rather than a runtime loop.

use minkowski_lsm::bench_support::{
    Layout, Shape, WorkloadParams, WriteAmp, build_world, overwrite,
};
use minkowski_lsm::compactor::compact_one;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::recovery::LsmRecovery;
use minkowski_lsm::types::{SeqNo, SeqRange};
use std::time::{Duration, Instant};

/// Drive `flushes` flush rounds with compaction trigger `N`. After round 0,
/// each round mutates `ow` of the rows (`overwrite`) before flushing, then
/// compacts greedily to convergence — `compact_one` returns `Ok(None)` once no
/// `(level, archetype)` group has `>= N` runs, which is the convergence signal.
///
/// Write-amp is accumulated exactly from each `CompactionReport` (`input_bytes`
/// is the exact sum of consumed input run sizes; no estimation). After all
/// flushes, a full recovery is timed.
fn sweep_run<const N: usize>(
    params: &WorkloadParams,
    ow: f64,
    flushes: u64,
) -> (WriteAmp, Duration) {
    let (mut world, codecs) = build_world(params);
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<N>(&log_path).unwrap();
    let mut wa = WriteAmp::default();

    for k in 0..flushes {
        if k > 0 {
            overwrite(&mut world, ow, 2000 + k);
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

        // Compact greedily to convergence: `compact_one` returns `None` when no
        // group has `>= N` runs left to merge.
        while let Some(r) = compact_one::<N>(&mut manifest, &mut log, dir.path()).unwrap() {
            wa.input_bytes += r.input_bytes;
            wa.output_bytes += r.output_bytes;
        }
    }

    let start = Instant::now();
    let (result, _, _) = LsmRecovery::recover::<N>(dir.path(), &log_path, &codecs).unwrap();
    let dur = start.elapsed();
    std::hint::black_box(&result);
    (wa, dur)
}

/// Instantiate `sweep_run` for each compaction trigger `N` and print a line.
macro_rules! sweep_all {
    ($params:expr, $ow:expr, $flushes:expr, $($n:literal),+ $(,)?) => {
        $(
            let (wa, dur) = sweep_run::<$n>($params, $ow, $flushes);
            eprintln!(
                "SWEEP N={} ow={} write_amp={:.4} recover_us={}",
                $n,
                $ow,
                wa.ratio(),
                dur.as_micros(),
            );
        )+
    };
}

fn main() {
    let params = WorkloadParams {
        entities: 20_000,
        shape: Shape::Pod,
        layout: Layout::Fragmented,
        sparse: false,
        seed: 21,
    };
    let flushes = 12;
    for ow in [0.0_f64, 0.25, 1.0] {
        sweep_all!(&params, ow, flushes, 3, 4, 5, 7);
    }
}
