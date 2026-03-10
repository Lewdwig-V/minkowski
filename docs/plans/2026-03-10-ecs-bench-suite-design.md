# Standardized ECS Benchmark Suite — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a standardized ECS benchmark suite for Minkowski, modeled after the [ecs_bench_suite](https://github.com/rust-gamedev/ecs_bench_suite) conventions, so results are directly comparable with published numbers from Bevy, hecs, Legion, Specs, and Shipyard.

**Architecture:** New `minkowski-bench` crate with one criterion bench file per scenario, shared component types in `src/lib.rs`. Replaces the existing ad-hoc benches in `crates/minkowski/benches/`. Depends on both `minkowski` and `minkowski-persist` (for the serialize benchmark).

**Tech Stack:** Criterion, rkyv (via minkowski-persist), rayon (via minkowski).

---

## Component Types

Shared across all scenarios, defined in `crates/minkowski-bench/src/lib.rs`:

```rust
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Transform { pub matrix: [[f32; 4]; 4] }  // 64 bytes, cache-line sized

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Position { pub x: f32, pub y: f32, pub z: f32 }  // 12 bytes

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Rotation { pub x: f32, pub y: f32, pub z: f32 }  // 12 bytes

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Velocity { pub dx: f32, pub dy: f32, pub dz: f32 }  // 12 bytes
```

`#[repr(C)]` enables `raw_copy_size` for the rkyv zero-copy snapshot path. `align(16)` on Transform for SIMD friendliness. 3-component vectors match ecs_bench_suite's `vec3` convention.

## Scenarios

### 1. simple_insert (`benches/simple_insert.rs`)

**Entity count:** 10,000
**Setup:** Empty world each iteration
**Measured:** Spawn 10K entities with `(Transform, Position, Rotation, Velocity)` bundle

Sub-benchmarks:
- `simple_insert/batch` — `world.spawn()` loop
- `simple_insert/changeset` — `EnumChangeSet::spawn_bundle` + `apply()`

### 2. simple_iter (`benches/simple_iter.rs`)

**Entity count:** 10,000 with `(Transform, Position, Rotation, Velocity)`
**Setup:** Pre-populated world (not measured)
**Measured:** Iterate all entities, `vel.dx += pos.x * 0.1` (etc.)

Sub-benchmarks:
- `simple_iter/for_each` — per-element iteration
- `simple_iter/for_each_chunk` — typed slices, SIMD-friendly
- `simple_iter/par_for_each` — rayon parallel iteration

### 3. fragmented_iter (`benches/fragmented_iter.rs`)

**Fragment types:** 26 (A through Z), each a `(f32)` newtype
**Entities per fragment:** 20 (each gets the fragment type + `Position`)
**Total:** 520 entities across 26 archetypes
**Measured:** Iterate all entities with `Position`, increment x

Matches ecs_bench_suite exactly. Tests archetype scan overhead when most archetypes are tiny.

### 4. heavy_compute (`benches/heavy_compute.rs`)

**Entity count:** 1,000 with `(Transform,)`
**Measured:** Invert each 4x4 matrix in-place (~200 FLOPs per entity)

Sub-benchmarks:
- `heavy_compute/sequential` — `for_each_chunk`
- `heavy_compute/parallel` — `par_for_each`

### 5. add_remove (`benches/add_remove.rs`)

**Entity count:** 10,000 with `(Position,)`
**Measured:** Add `Velocity` to all, then remove `Velocity` from all

Tests archetype migration throughput.

### 6. schedule (`benches/schedule.rs`)

**Systems:** 5 independent `QueryMut` reducers via `ReducerRegistry`
**Entity count:** 10,000 with `(Transform, Position, Rotation, Velocity)`
**Measured:** Register 5 non-conflicting reducers, run all sequentially, measure Access conflict checking + dispatch overhead

Not outer-parallelism — demonstrates the cost of Minkowski's scheduling primitives.

### 7. serialize (`benches/serialize.rs`)

**Entity count:** 1,000 with `(Transform, Position, Rotation, Velocity)`

Sub-benchmarks:
- `serialize/snapshot_save` — `Snapshot::save_to_bytes`
- `serialize/snapshot_load` — `Snapshot::load_from_bytes` (rkyv zero-copy)
- `serialize/wal_append` — append 1K mutations via `Durable` commit
- `serialize/wal_replay` — replay WAL into fresh world

### 8. reducer (`benches/reducer.rs`)

Migrated from existing `crates/minkowski/benches/reducer.rs`. Minkowski-specific (no ecs_bench_suite equivalent).

**Entity count:** 10,000 with `(Position, Velocity)`
**Measured:** QueryMut, QueryMut chunk, QueryWriter, DynamicCtx for_each, DynamicCtx for_each_chunk

## Migration

The existing ad-hoc benchmarks in `crates/minkowski/benches/` are subsumed:

| Old bench | Replaced by |
|---|---|
| `spawn.rs` | `simple_insert` |
| `iterate.rs` | `simple_iter` |
| `fragmented.rs` | `fragmented_iter` |
| `parallel.rs` | `simple_iter/par_for_each` + `heavy_compute/parallel` |
| `add_remove.rs` | `add_remove` |
| `reducer.rs` | `reducer` (migrated as-is) |

After migration:
- Delete all 6 files from `crates/minkowski/benches/`
- Remove `hecs` dev-dependency from `crates/minkowski/Cargo.toml`
- Update `CLAUDE.md` bench commands to point at `minkowski-bench`

## Crate Structure

```
crates/minkowski-bench/
├── Cargo.toml          # [dev-dependencies] criterion, [dependencies] minkowski, minkowski-persist
├── src/
│   └── lib.rs          # Component types, shared setup helpers
└── benches/
    ├── simple_insert.rs
    ├── simple_iter.rs
    ├── fragmented_iter.rs
    ├── heavy_compute.rs
    ├── schedule.rs
    ├── add_remove.rs
    ├── serialize.rs
    └── reducer.rs
```

## Running

```bash
cargo bench -p minkowski-bench                    # All benchmarks
cargo bench -p minkowski-bench -- simple_iter     # Single scenario
cargo bench -p minkowski-bench -- simple_iter/par # Sub-benchmark filter
```
