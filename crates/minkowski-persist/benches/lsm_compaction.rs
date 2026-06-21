//! criterion bench for the LSM compaction path, with write-amplification
//! capture. Each case builds `COMPACTION_TRIGGER` sorted runs into a tempdir
//! (one clean flush, then `overwrite` + reflush for the remaining runs), and
//! measures a single `compact_one::<4>` merge. The exact write-amplification
//! ratio (`output_bytes / input_bytes`) is read directly from the returned
//! `CompactionReport` — `input_bytes` is the exact sum of consumed input run
//! sizes — and emitted to stderr as a `WRITEAMP` log line for the audit/sweep
//! tooling to scrape (criterion graphs time only, not write-amp).
//!
//! Run with the `bench-support` feature (enabled via the dev-dep on
//! `minkowski-lsm`):
//!
//! ```text
//! cargo bench -p minkowski-persist --bench lsm_compaction
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use minkowski_lsm::bench_support::{
    Layout, Shape, WorkloadParams, WriteAmp, build_world, overwrite,
};
use minkowski_lsm::compactor::{COMPACTION_TRIGGER, compact_one};
use minkowski_lsm::manifest::LsmManifest;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::types::{SeqNo, SeqRange};
use std::path::Path;

/// Build `COMPACTION_TRIGGER` L0 runs into `dir`: one clean flush, then
/// `overwrite(ow, seed)` + reflush for each remaining run. Each flush gets a
/// distinct, non-overlapping `SeqRange` so the runs stay separate at L0 (the
/// compactor needs `>= COMPACTION_TRIGGER` distinct runs to fire).
fn build_k_runs(params: &WorkloadParams, ow: f64, dir: &Path) -> (LsmManifest<4>, ManifestLog) {
    let (mut world, codecs) = build_world(params);
    let log_path = dir.join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
    for k in 0..COMPACTION_TRIGGER as u64 {
        if k > 0 {
            overwrite(&mut world, ow, 1000 + k);
        }
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(k * 100), SeqNo::from((k + 1) * 100)).unwrap(),
            &mut manifest,
            &mut log,
            dir,
            &codecs,
        )
        .unwrap()
        .expect("world is dirty, flush must produce a run");
    }
    (manifest, log)
}

fn bench_compaction(c: &mut Criterion) {
    let mut g = c.benchmark_group("lsm_compaction");
    let cases = [
        (
            "pod_10k_ow25",
            WorkloadParams {
                entities: 10_000,
                shape: Shape::Pod,
                layout: Layout::Single,
                sparse: false,
                seed: 10,
            },
            0.25_f64,
        ),
        (
            "heap_10k_ow100",
            WorkloadParams {
                entities: 10_000,
                shape: Shape::Heap,
                layout: Layout::Single,
                sparse: false,
                seed: 11,
            },
            1.0_f64,
        ),
    ];
    for (name, params, ow) in cases {
        g.bench_function(name, |b| {
            // The K runs (clean flush + reflushes) are built per-iteration in the
            // setup closure so the measured region is only `compact_one`.
            b.iter_batched(
                || {
                    let dir = tempfile::tempdir().unwrap();
                    let (manifest, log) = build_k_runs(&params, ow, dir.path());
                    (dir, manifest, log)
                },
                |(dir, mut manifest, mut log)| {
                    let report = compact_one::<4>(&mut manifest, &mut log, dir.path())
                        .unwrap()
                        .expect("K >= COMPACTION_TRIGGER runs at L0, compaction must fire");
                    let wa = WriteAmp {
                        input_bytes: report.input_bytes,
                        output_bytes: report.output_bytes,
                    };
                    eprintln!(
                        "WRITEAMP {name} ow={ow} ratio={:.4} in={} out={}",
                        wa.ratio(),
                        report.input_bytes,
                        report.output_bytes,
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

criterion_group!(benches, bench_compaction);
criterion_main!(benches);
