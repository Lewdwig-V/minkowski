# Fuzz Testing Design

**Status:** Accepted
**Date:** 2026-03-08

## Goal

Add fuzz testing via `cargo-fuzz` (libFuzzer backend) to catch memory safety bugs in the storage layer and semantic correctness bugs under random operation sequences.

## Scope

Two crates: `minkowski` (core) and `minkowski-persist` (WAL + snapshots). Four fuzz targets.

## Component Types

All targets share 4 fixed component types with varied sizes and alignments:

```rust
#[derive(Clone, Copy, Debug, Arbitrary)]
struct A(u32);       // 4 bytes, 4-align
#[derive(Clone, Copy, Debug, Arbitrary)]
struct B(u64);       // 8 bytes, 8-align
#[derive(Clone, Copy, Debug, Arbitrary)]
struct C([u8; 3]);   // 3 bytes, 1-align (odd size)
#[derive(Clone, Copy, Debug, Arbitrary)]
struct D(f32);       // 4 bytes, 4-align
```

All are `Copy` (no drop glue). Different sizes exercise BlobVec's layout-based allocation and pointer arithmetic.

## Target 1: `fuzz_world_ops`

Random operation sequences against a `World`. The fuzzer derives `Arbitrary` on an operation enum:

```rust
enum Op {
    Spawn(BundleKind),
    Despawn(u8),
    Insert(u8, CompVal),
    Remove(u8, CompKind),
    Get(u8, CompKind),
    GetMut(u8, CompKind),
    Query(QueryShape),
    DespawnBatch(Vec<u8>),
}
```

`BundleKind` selects from concrete bundle tuples: `(A,)`, `(A, B)`, `(A, C)`, `(A, B, C, D)`, etc. `CompVal` wraps an `Arbitrary` value for each component type. `CompKind` selects which type. `QueryShape` selects from ~8 concrete query types.

The harness maintains a `Vec<Entity>` of live handles. `u8` indices are `% len` (skip if empty).

**Invariants checked after each op:**
- `world.entity_count()` matches `live.len()`
- Every entity in `live` satisfies `world.is_alive()` and `world.is_placed()`

**End-of-sequence cleanup:** despawn all, verify `entity_count() == 0`.

**Catches:** use-after-free in BlobVec, entity location corruption after archetype migration, generation mismatch bugs, swap-remove index errors, archetype creation edge cases.

## Target 2: `fuzz_reducers`

Exercises the query iteration unsafe paths (`init_fetch`, pointer arithmetic in `fetch`, typed slice construction in `for_each_chunk`) across varied archetype shapes.

```rust
enum ReducerOp {
    QueryMut(QueryShape),
    QueryRef(QueryShape),
    ForEachChunk(QueryShape),
    SpawnMore(BundleKind),
    DespawnSome(u8),
}
```

The harness spawns a random initial population, then applies a sequence of reducer operations interleaved with spawn/despawn to vary archetype shapes. Reducer closures do trivial work (read values, write incremented values). Only scheduled reducers (`QueryMut`, `QueryRef`) — no transactional handles.

**Invariants:** entity count consistency, query result count matches expected archetype membership.

**Catches:** pointer arithmetic errors in `fetch.rs` (40 unsafe blocks), slice construction bugs in `for_each_chunk`, archetype matching errors with varied component sets.

## Target 3: `fuzz_snapshot_load`

"Don't crash on bad input" target for snapshot deserialization.

Input byte layout: leading byte selects mode.
- `0x00`: remaining bytes fed directly to `Snapshot::load_from_bytes`. Must return `Ok` or `Err`, never panic.
- `0x01`: remaining bytes interpreted as `Arbitrary` operation sequence. Build a valid world, `save_to_bytes`, `load_from_bytes`, verify round-trip produces equivalent query results.

A `CodecRegistry` with all 4 component types is set up for both modes. Also registers with `rkyv` derives.

**Catches:** rkyv validation bypass on malformed archived data, component remap errors, allocator state corruption on restore, panics in error paths.

## Target 4: `fuzz_wal_replay`

"Don't crash on bad input" target for WAL record parsing and replay.

Same dual-mode structure:
- `0x00`: write fuzzed bytes to temp file, open as `Wal`, call `replay_from(0, ...)`. Must return `Ok` or `Err`.
- `0x01`: build valid changesets from `Arbitrary` ops, `append` to WAL, `replay_from` into a fresh world, verify convergence.

**Catches:** WAL record framing parser errors, schema preamble validation, rkyv deserialization of changeset records, sequence number handling edge cases.

## Project Layout

```
fuzz/
  Cargo.toml           # cargo-fuzz manifest
  fuzz_targets/
    fuzz_world_ops.rs
    fuzz_reducers.rs
    fuzz_snapshot_load.rs
    fuzz_wal_replay.rs
```

The `fuzz/Cargo.toml` depends on `minkowski` and `minkowski-persist` as path dependencies, plus `libfuzzer-sys` and `arbitrary` (with the `derive` feature).

## Running

```bash
cargo install cargo-fuzz    # one-time setup
cargo +nightly fuzz run fuzz_world_ops      # run indefinitely
cargo +nightly fuzz run fuzz_snapshot_load -- -max_len=65536  # cap input size
cargo +nightly fuzz run fuzz_world_ops -- -max_total_time=300  # 5 minute run
```

## CI Integration

Not in initial scope. Fuzz testing is a local development activity — run manually before releases or after changes to unsafe code. CI integration (OSS-Fuzz or scheduled nightly runs) can be added later.

## Alternatives Considered

- **`bolero`** — supports multiple backends (libFuzzer, AFL, proptest) but adds complexity. `cargo-fuzz` is the standard Rust fuzzing tool and sufficient for our needs.
- **`proptest`** — property-based testing, not true fuzzing. Good complement but doesn't get coverage-guided mutation. Could add later as separate prop tests.
- **Multi-threaded transaction fuzzing** — `cargo-fuzz` doesn't natively support multi-threaded targets. Transaction fuzzing would need a different framework (e.g. `loom` for concurrency testing). Out of scope.
