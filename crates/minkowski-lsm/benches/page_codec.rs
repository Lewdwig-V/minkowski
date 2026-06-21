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
//! Two distinct layers are measured separately, because conflating them is
//! misleading:
//!
//! 1. **Page framing** (`serialized_page::encode`/`decode`) — the Arrow-style
//!    `[offsets:(n+1)×u32][values]` table *around* already-serialized row bytes.
//!    `decode` is zero-copy (returns borrowed `&[u8]` slices, no rkyv); `encode`
//!    builds the offset table and concatenates the row bytes. NO rkyv work here.
//! 2. **rkyv codec** (`serialize_by_type`/`deserialize_by_type`) — the actual
//!    per-row rkyv serialize / deserialize. `rkyv_decode_256` is the recovery-
//!    relevant cost: `from_bytes` + bytecheck + the `AlignedVec` realign + copy
//!    the native value out (and drop it, leak-free).
//!
//! Benches:
//! - `get_page_pod`: read one RawCopy dense page via `SortedRunReader::get_page`.
//! - `page_frame_encode_256` / `page_frame_decode_256`: offset-table framing of
//!   256 already-serialized rows (NOT rkyv — see layer 1 above).
//! - `rkyv_encode_256` / `rkyv_decode_256`: the real per-row rkyv serialize /
//!   deserialize of 256 heap `BenchName` rows (layer 2 — this is the rkyv cost).
//! - `rawcopy_column_256`: a plain memcpy of a 256×`size_of::<BenchPos>()` byte
//!   column — the POD comparison point against the rkyv path.

use std::any::TypeId;
use std::hint::black_box;
use std::path::PathBuf;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};

use minkowski_lsm::bench_support::{
    BenchName, BenchPos, Layout, Shape, WorkloadParams, build_world,
};
use minkowski_lsm::codec::{CodecRegistry, CrcProof};
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

/// A codec registry + the `BenchName` TypeId + `HEAP_ROWS` live values, for the
/// real per-row rkyv *serialize* bench.
fn setup_rkyv_encode() -> (CodecRegistry, TypeId, Vec<BenchName>) {
    let (_world, codecs) = build_world(&WorkloadParams {
        entities: 1,
        shape: Shape::Heap,
        layout: Layout::Single,
        sparse: false,
        seed: SEED,
    });
    let values = (0..HEAP_ROWS)
        .map(|i| BenchName {
            text: format!("entity-name-{i}"),
        })
        .collect();
    (codecs, TypeId::of::<BenchName>(), values)
}

/// A codec registry + the `BenchName` TypeId + `HEAP_ROWS` already-rkyv-serialized
/// row blobs, for the real per-row rkyv *decode* bench (the recovery-relevant cost).
fn setup_rkyv_decode() -> (CodecRegistry, TypeId, Vec<Vec<u8>>) {
    let (_world, codecs) = build_world(&WorkloadParams {
        entities: 1,
        shape: Shape::Heap,
        layout: Layout::Single,
        sparse: false,
        seed: SEED,
    });
    (codecs, TypeId::of::<BenchName>(), heap_rows())
}

/// Codec + `BenchName` TypeId + `HEAP_ROWS` serialized rows + a `CrcProof` per
/// row, for the unchecked decode bench. Proofs are minted in setup (not measured).
fn setup_rkyv_decode_unchecked() -> (CodecRegistry, TypeId, Vec<Vec<u8>>, Vec<CrcProof>) {
    let (_world, codecs) = build_world(&WorkloadParams {
        entities: 1,
        shape: Shape::Heap,
        layout: Layout::Single,
        sparse: false,
        seed: SEED,
    });
    let rows = heap_rows();
    let proofs = rows
        .iter()
        .map(|r| CrcProof::verify(r, crc32fast::hash(r)).expect("fresh crc matches"))
        .collect();
    (codecs, TypeId::of::<BenchName>(), rows, proofs)
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

// Layer 1: page framing (offset table) — NO rkyv.
#[library_benchmark]
#[bench::heap(setup = heap_rows)]
fn page_frame_encode_256(rows: Vec<Vec<u8>>) {
    black_box(serialized_page::encode(black_box(&rows)));
}

#[library_benchmark]
#[bench::heap(setup = setup_decode)]
fn page_frame_decode_256(body: Vec<u8>) {
    let decoded = serialized_page::decode(black_box(&body), HEAP_ROWS).expect("decode ok");
    black_box(decoded.len());
}

// Layer 2: the actual rkyv codec — this is the rkyv encode/decode overhead.
#[library_benchmark]
#[bench::heap(setup = setup_rkyv_encode)]
fn rkyv_encode_256(input: (CodecRegistry, TypeId, Vec<BenchName>)) {
    let (codecs, ty, values) = input;
    for v in &values {
        let mut buf = Vec::new();
        // SAFETY: `ty` is `BenchName`'s TypeId; the pointer is a valid, aligned
        // `BenchName` for the duration of the call.
        unsafe {
            codecs
                .serialize_by_type(ty, std::ptr::from_ref(v).cast::<u8>(), &mut buf)
                .expect("codec for BenchName")
                .expect("serialize ok");
        }
        black_box(buf.len());
    }
}

#[library_benchmark]
#[bench::heap(setup = setup_rkyv_decode)]
fn rkyv_decode_256(input: (CodecRegistry, TypeId, Vec<Vec<u8>>)) {
    let (codecs, ty, rows) = input;
    for row in &rows {
        let native = codecs
            .deserialize_by_type(ty, row)
            .expect("codec for BenchName")
            .expect("decode ok");
        // `native` byte-owns a reconstructed `BenchName` (with a heap `String`).
        // Read it out and drop it so the `String` is freed exactly once — the
        // `Vec<u8>` would otherwise leak it (a `Vec<u8>` runs no `BenchName::drop`).
        // This mirrors the ownership transfer recovery performs.
        // `native` is a `Vec<u8>` (byte alignment only), so use `read_unaligned`:
        // a plain `ptr::read` of a `BenchName` (pointer-aligned) would be UB if the
        // allocator doesn't over-align the buffer.
        // SAFETY: `native` holds a valid native `BenchName` image (deserialize_by_type
        // reconstructs the value and transfers ownership into the bytes).
        let name = unsafe { std::ptr::read_unaligned(native.as_ptr().cast::<BenchName>()) };
        black_box(name.text.len());
        drop(name);
    }
}

#[library_benchmark]
#[bench::heap(setup = setup_rkyv_decode_unchecked)]
fn rkyv_decode_unchecked_256(input: (CodecRegistry, TypeId, Vec<Vec<u8>>, Vec<CrcProof>)) {
    let (codecs, ty, rows, proofs) = input;
    for (row, proof) in rows.iter().zip(proofs.iter()) {
        let native = codecs
            .deserialize_unchecked_by_type(ty, row, proof)
            .expect("codec for BenchName")
            .expect("decode ok");
        // Same ownership dance as rkyv_decode_256: read out unaligned, drop once.
        let name = unsafe { std::ptr::read_unaligned(native.as_ptr().cast::<BenchName>()) };
        black_box(name.text.len());
        drop(name);
    }
}

#[library_benchmark]
#[bench::pod(setup = setup_column)]
fn rawcopy_column_256(column: Vec<u8>) {
    black_box(black_box(&column[..]).to_vec());
}

library_benchmark_group!(
    name = page_codec;
    benchmarks =
        get_page_pod,
        page_frame_encode_256,
        page_frame_decode_256,
        rkyv_encode_256,
        rkyv_decode_256,
        rkyv_decode_unchecked_256,
        rawcopy_column_256
);

main!(library_benchmark_groups = page_codec);
