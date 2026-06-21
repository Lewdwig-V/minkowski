//! Page-codec microbenchmarks for `minkowski-lsm` (iai-callgrind / callgrind).
//!
//! Instruction-count + cache-model measurement via valgrind callgrind —
//! deterministic and CI-stable for the sub-microsecond per-page codecs, where
//! wall-clock timing would be pure noise. Each bench's `setup` (building the run
//! / rows) is NOT measured; only the codec call in the bench body is.
//!
//! Requires valgrind + `cargo install iai-callgrind-runner` (matching the
//! `iai-callgrind` dev-dependency version). The repo's `.cargo/config.toml` sets
//! `target-cpu=native`, which emits instructions valgrind cannot execute (SIGILL),
//! so override the target to a valgrind-safe baseline when running this bench:
//!   RUSTFLAGS="-C target-cpu=x86-64-v2" \
//!     cargo bench -p minkowski-lsm --features bench-support --bench page_codec
//!
//! Benches:
//! - `get_page_pod`: decode one RawCopy dense page via `SortedRunReader::get_page`.
//! - `serialized_encode_256` / `serialized_decode_256`: the variable-length heap
//!   column codec on 256 rkyv `BenchName` rows.
//! - `rawcopy_column_256`: a plain memcpy of a 256×`size_of::<BenchPos>()` byte
//!   column — the POD comparison point against the rkyv path.

use std::any::TypeId;
use std::hint::black_box;
use std::path::PathBuf;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};

use minkowski_lsm::bench_support::{
    BenchName, BenchPos, Layout, Shape, WorkloadParams, build_world,
};
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::reader::SortedRunReader;
use minkowski_lsm::schema::StorageKind;
use minkowski_lsm::serialized_page;
use minkowski_lsm::types::{SeqNo, SeqRange};

/// Heap rows packed into one Serialized page for the codec benches.
const HEAP_ROWS: usize = 256;
/// POD rows for the RawCopy column-copy comparison.
const POD_ROWS: usize = 256;
/// Deterministic seed for the workload builder.
const SEED: u64 = 0xC0DE_C0DE;

/// Build a Pod sorted run on disk; return its path + the owning temp dir (kept
/// alive so the mmap-backed file is not deleted).
fn make_pod_run(entities: usize) -> (tempfile::TempDir, PathBuf) {
    let (world, codecs) = build_world(&WorkloadParams {
        entities,
        shape: Shape::Pod,
        layout: Layout::Single,
        sparse: false,
        seed: SEED,
    });
    let dir = tempfile::tempdir().expect("tempdir");
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).expect("recover manifest");
    let path = flush_and_record(
        &world,
        SeqRange::new(SeqNo::from(0u64), SeqNo::from(100u64)).expect("seq range"),
        &mut manifest,
        &mut log,
        dir.path(),
        &codecs,
    )
    .expect("flush")
    .expect("a Pod world is dirty, so a run is written");
    (dir, path)
}

/// Pick the first `(arch_id, slot)` resolving to a RawCopy dense page in the run.
/// (`component_slots_for_arch` is `pub(crate)`, so probe the public schema.)
fn first_pod_page(reader: &SortedRunReader) -> (u16, u16) {
    let arch_id = *reader
        .archetype_ids()
        .first()
        .expect("Pod run has at least one archetype");
    for entry in reader.schema().entries() {
        if entry.storage_kind() == StorageKind::RawCopy
            && reader
                .get_page(arch_id, entry.slot(), 0)
                .expect("get_page must not error")
                .is_some()
        {
            return (arch_id, entry.slot());
        }
    }
    panic!("no RawCopy dense page found in Pod run");
}

/// `HEAP_ROWS` rkyv-serialized `BenchName` rows, ready for `serialized_page::encode`.
fn heap_rows() -> Vec<Vec<u8>> {
    let (_world, codecs) = build_world(&WorkloadParams {
        entities: 1,
        shape: Shape::Heap,
        layout: Layout::Single,
        sparse: false,
        seed: SEED,
    });
    let ty = TypeId::of::<BenchName>();
    (0..HEAP_ROWS)
        .map(|i| {
            let value = BenchName {
                text: format!("entity-name-{i}"),
            };
            let mut bytes = Vec::new();
            // SAFETY: `ty` is `BenchName`'s TypeId and the pointer is a valid,
            // aligned `BenchName` for the duration of the call.
            unsafe {
                codecs
                    .serialize_by_type(ty, std::ptr::from_ref(&value).cast::<u8>(), &mut bytes)
                    .expect("codec registered for BenchName")
                    .expect("serialize ok");
            }
            bytes
        })
        .collect()
}

// ── setups (run before measurement; NOT counted) ────────────────────────────

fn setup_get_page() -> (tempfile::TempDir, SortedRunReader, u16, u16) {
    let (dir, path) = make_pod_run(1024);
    let reader = SortedRunReader::open(&path).expect("open run");
    let (arch_id, slot) = first_pod_page(&reader);
    (dir, reader, arch_id, slot)
}

fn setup_decode() -> Vec<u8> {
    serialized_page::encode(&heap_rows())
}

fn setup_column() -> Vec<u8> {
    vec![0xA5u8; POD_ROWS * size_of::<BenchPos>()]
}

// ── benches (only the body is measured) ─────────────────────────────────────

#[library_benchmark]
#[bench::pod(setup = setup_get_page)]
fn get_page_pod(input: (tempfile::TempDir, SortedRunReader, u16, u16)) {
    let (_dir, reader, arch_id, slot) = input;
    let page = reader
        .get_page(arch_id, slot, 0)
        .expect("get_page ok")
        .expect("page present");
    black_box(page.data().len());
}

#[library_benchmark]
#[bench::heap(setup = heap_rows)]
fn serialized_encode_256(rows: Vec<Vec<u8>>) {
    black_box(serialized_page::encode(black_box(&rows)));
}

#[library_benchmark]
#[bench::heap(setup = setup_decode)]
fn serialized_decode_256(body: Vec<u8>) {
    let decoded = serialized_page::decode(black_box(&body), HEAP_ROWS).expect("decode ok");
    black_box(decoded.len());
}

#[library_benchmark]
#[bench::pod(setup = setup_column)]
fn rawcopy_column_256(column: Vec<u8>) {
    black_box(black_box(&column[..]).to_vec());
}

library_benchmark_group!(
    name = page_codec;
    benchmarks = get_page_pod, serialized_encode_256, serialized_decode_256, rawcopy_column_256
);

main!(library_benchmark_groups = page_codec);
