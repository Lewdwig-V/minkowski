//! criterion throughput benches for the LSM flush and recovery paths over a
//! small representative workload matrix (POD/heap shapes × single/fragmented
//! layouts). Run with the `bench-support` feature (enabled via the dev-dep on
//! `minkowski-lsm`):
//!
//! ```text
//! cargo bench -p minkowski-persist --bench lsm_throughput
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use minkowski_lsm::bench_support::{Layout, Shape, WorkloadParams, build_world};
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::recovery::LsmRecovery;
use minkowski_lsm::types::{SeqNo, SeqRange};

/// Representative subset of the full workload matrix.
fn matrix() -> Vec<(&'static str, WorkloadParams)> {
    vec![
        (
            "pod_1k_single",
            WorkloadParams {
                entities: 1_000,
                shape: Shape::Pod,
                layout: Layout::Single,
                sparse: false,
                seed: 1,
            },
        ),
        (
            "pod_10k_frag",
            WorkloadParams {
                entities: 10_000,
                shape: Shape::Pod,
                layout: Layout::Fragmented,
                sparse: false,
                seed: 2,
            },
        ),
        (
            "heap_10k_single",
            WorkloadParams {
                entities: 10_000,
                shape: Shape::Heap,
                layout: Layout::Single,
                sparse: false,
                seed: 3,
            },
        ),
    ]
}

fn flush_seq_range() -> SeqRange {
    SeqRange::new(SeqNo::from(0u64), SeqNo::from(100u64)).unwrap()
}

fn bench_flush(c: &mut Criterion) {
    let mut g = c.benchmark_group("lsm_flush");
    for (name, params) in matrix() {
        let (world, codecs) = build_world(&params);
        g.bench_function(name, |b| {
            // Each flush needs a clean manifest + directory, so the per-iteration
            // setup (tempdir + fresh ManifestLog) is built outside the measured
            // region via iter_batched.
            b.iter_batched(
                || {
                    let dir = tempfile::tempdir().unwrap();
                    let log_path = dir.path().join("manifest.log");
                    (dir, log_path)
                },
                |(dir, log_path)| {
                    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
                    flush_and_record(
                        &world,
                        flush_seq_range(),
                        &mut manifest,
                        &mut log,
                        dir.path(),
                        &codecs,
                    )
                    .unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_recover(c: &mut Criterion) {
    let mut g = c.benchmark_group("lsm_recover");
    for (name, params) in matrix() {
        let (world, codecs) = build_world(&params);
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            flush_seq_range(),
            &mut manifest,
            &mut log,
            dir.path(),
            &codecs,
        )
        .unwrap()
        .expect("world is dirty, flush must produce a run");

        g.bench_function(name, |b| {
            b.iter(|| {
                let (result, _, _) =
                    LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
                std::hint::black_box(&result);
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_flush, bench_recover);
criterion_main!(benches);
