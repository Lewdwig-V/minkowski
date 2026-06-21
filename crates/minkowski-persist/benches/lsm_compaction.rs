//! criterion bench for the LSM compaction path, with write-amplification
//! capture. Each case builds `COMPACTION_TRIGGER` sorted runs into a tempdir —
//! each run flushes a fresh batch of `params.entities` NEW entities grown into
//! the world, clearing the dirty-page tracker after each flush so each run
//! carries only its own rows — and measures a single `compact_one::<4>` merge.
//!
//! K runs of unique data merge with no supersession, so the exact write-amp
//! ratio (`output_bytes / input_bytes`, read from the returned `CompactionReport`,
//! whose `input_bytes` is the exact sum of consumed input run sizes) is ~1.0 —
//! this bench measures the *merge throughput*, not dedup (dedup vs level count is
//! the `lsm_level_sweep` bench). The ratio is emitted to stderr as a `WRITEAMP`
//! line for the audit tooling to scrape (criterion graphs time only).
//!
//! NOTE: `grow` (spawning new entities) dirties pages; raw value mutation via
//! `world.query::<(&mut T,)>()` does NOT mark flush-dirty pages, so a flush after
//! such a mutation writes nothing.
//!
//! Run with the `bench-support` feature (enabled via the dev-dep on
//! `minkowski-lsm`):
//!
//! ```text
//! cargo bench -p minkowski-persist --bench lsm_compaction
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use minkowski_lsm::bench_support::{Layout, Shape, WorkloadParams, WriteAmp, build_world, grow};
use minkowski_lsm::compactor::{COMPACTION_TRIGGER, compact_one};
use minkowski_lsm::manifest::LsmManifest;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::types::{SeqNo, SeqRange};
use std::path::Path;

/// Build `COMPACTION_TRIGGER` L0 runs into `dir`: each iteration grows the world
/// by `params.entities` NEW entities (the first run is the initial population),
/// flushes, and clears the dirty-page tracker so the next flush writes only the
/// new rows. Each flush gets a distinct, non-overlapping `SeqRange` so the runs
/// stay separate at L0 (the compactor needs `>= COMPACTION_TRIGGER` to fire).
fn build_k_runs(params: &WorkloadParams, dir: &Path) -> (LsmManifest<4>, ManifestLog) {
    let (mut world, codecs) = build_world(params);
    let mut count = params.entities;
    let log_path = dir.join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
    for k in 0..COMPACTION_TRIGGER as u64 {
        if k > 0 {
            grow(&mut world, params, count, params.entities, 1000 + k);
            count += params.entities;
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
        world.clear_all_dirty_pages();
    }
    (manifest, log)
}

fn bench_compaction(c: &mut Criterion) {
    let mut g = c.benchmark_group("lsm_compaction");
    let cases = [
        (
            "pod_10k",
            WorkloadParams {
                entities: 10_000,
                shape: Shape::Pod,
                layout: Layout::Single,
                sparse: false,
                seed: 10,
            },
        ),
        (
            "heap_10k",
            WorkloadParams {
                entities: 10_000,
                shape: Shape::Heap,
                layout: Layout::Single,
                sparse: false,
                seed: 11,
            },
        ),
    ];
    for (name, params) in cases {
        g.bench_function(name, |b| {
            // The K runs are built per-iteration in the setup closure so the
            // measured region is only `compact_one`.
            b.iter_batched(
                || {
                    let dir = tempfile::tempdir().unwrap();
                    let (manifest, log) = build_k_runs(&params, dir.path());
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
                        "WRITEAMP {name} ratio={:.4} in={} out={}",
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
