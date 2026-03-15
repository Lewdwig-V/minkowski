# Join Batch Execution Design

**Version**: v1.3.1
**Date**: 2026-03-16
**Status**: Approved

## Problem

Join plans materialize matched entities into a flat `ScratchBuffer` (`Vec<Entity>`),
then hand them to the user callback one at a time. The callback typically calls
`world.get::<T>(entity)` per entity, which performs five redundant operations:

1. `entities.is_alive(entity)` -- generation check
2. `entity_locations[entity.index()]` -- location table lookup
3. `components.id::<T>()` -- component ID lookup (TypeId hash)
4. `archetype.column_index(id)` -- column search within archetype
5. `columns[col_idx].get_ptr(0)` + offset -- actual data access

Steps 1--4 are identical for consecutive entities in the same archetype. For a
10K-entity join across 5 archetypes, that's ~40,000 redundant lookups.

The join's `sorted_intersection` sorts by `Entity::to_bits()` (generation << 32 | index),
which gives entity-index order -- NOT archetype order. Entities spawned at
different times into different archetypes are interleaved arbitrarily, causing
random-access cache thrashing through the `entity_locations` table and component
columns.

## Solution: Archetype-Sorted Batch Execution

Two-phase post-join processing:

### Phase 1: Sort

After `run_join()` materializes the scratch buffer, sort entities in-place by
a packed `u64` key:

```
sort_key = (archetype_id as u64) << 32 | (row as u64)
```

Primary sort groups entities by archetype (contiguous runs). Secondary sort
orders rows within each archetype by physical memory position, enabling
hardware prefetcher linear stride detection.

Cost: O(N log N) comparisons on `u64`. For 10K entities, ~130K comparisons
at ~1ns each = ~130 us. This is the fixed tax that eliminates 40-50K redundant
lookups in the execution phase.

### Phase 2: Batch Execution

Walk the sorted scratch detecting archetype boundaries. For each run:

1. `Q::init_fetch(archetype, registry)` once -- resolves `ThinSlicePtr<T>`
   column pointers. O(1) per archetype.
2. For each entity in the run: `Q::fetch(&fetch, row)` -- single pointer add.
   No generation check, no TypeId hash, no column search.

The generation check is safe to skip because entities came from the join
collectors, which iterate live archetypes. Dead entities cannot appear in the
scratch buffer.

## API Surface

Three new execution methods on `QueryPlanResult`:

### `for_each_batched`

```rust
pub fn for_each_batched<Q, F>(
    &mut self,
    world: &mut World,
    callback: F,
) -> Result<(), PlanExecError>
where
    Q: WorldQuery,
    F: FnMut(Entity, Q::Item<'_>);
```

Per-entity callback with pre-resolved column pointers. Eliminates per-entity
`world.get()` overhead while preserving the familiar per-entity callback model.
Advances the read tick (same as `for_each`).

**Q validation**: `Q` is specified at the call site, not at plan-build time.
The implementation validates `Q` at runtime: for each archetype run, `init_fetch`
calls `registry.id::<T>()` and `archetype.column_index(id)`, both of which
return `Option`. If any required component in `Q` is missing from the archetype,
the method returns `Err(PlanExecError::ComponentMismatch)` -- a new error
variant. This is a runtime check (O(archetypes) cost, not O(entities)), matching
the existing `init_fetch` contract. `Option<&T>` terms are fine -- `init_fetch`
for `Option<&T>` returns `None` for missing columns, which `fetch` handles
by yielding `None`.

### `for_each_batched_raw`

```rust
pub fn for_each_batched_raw<Q, F>(
    &mut self,
    world: &World,
    callback: F,
) -> Result<(), PlanExecError>
where
    Q: ReadOnlyWorldQuery,
    F: FnMut(Entity, Q::Item<'_>);
```

Read-only variant. No tick advancement. Safe for use inside transactions.
Same `Q` validation as `for_each_batched`.

### `for_each_join_chunk`

```rust
pub fn for_each_join_chunk<Q, F>(
    &mut self,
    world: &mut World,
    callback: F,
) -> Result<(), PlanExecError>
where
    Q: WorldQuery,
    F: FnMut(&[Entity], &[usize], Q::Slice<'_>);
```

Chunk callback receives typed column slices per archetype run. Each call gets:
- `&[Entity]` -- matched entities in this archetype run (sorted by row)
- `&[usize]` -- row indices within the archetype (sorted, monotonically increasing)
- `Q::Slice<'_>` -- full column slice for the archetype

The row indices are extracted from `entity_locations` during the run-detection
phase and collected into a reusable `Vec<usize>` on `QueryPlanResult`. This
avoids exposing the `pub(crate) entity_locations` table to users. The callback
indexes into the slice using these row indices: `slice[rows[i]]`.

Advances the read tick. Same `Q` validation as `for_each_batched`.

**Why full-archetype slices (Option A) over gather-into-staging (Option B)?**
Copying 500 x 256B = 128KB per archetype run for fat structs defeats the purpose.
Option A gives the hardware prefetcher the same linear stride through the
BlobVec memory (rows are sorted), with zero intermediate copies. The sorted-row
property ensures monotonically increasing offsets -- the prefetcher sees a
forward stride even though the join only selected a subset of rows.

### Scan Plan Compatibility

All three methods work for scan-only plans (no joins). For scan plans, the
sort is a no-op (entities are already archetype-grouped), and each "run" is
one full archetype with row indices `0..archetype.len()`. This provides a
unified API for consumers who don't know whether their plan has joins.

For scan plans, `for_each_batched` is equivalent to `world.query::<Q>().for_each()`
but through the plan's filter/predicate pipeline. The value is that the caller
doesn't need to branch on "is this a join plan or a scan plan" -- the batched
methods handle both transparently.

## Implementation Details

### ScratchBuffer Extension

```rust
impl ScratchBuffer {
    fn sort_by_archetype(&mut self, entity_locations: &[Option<EntityLocation>]) {
        self.entities.sort_unstable_by_key(|e| {
            let loc = entity_locations[e.index() as usize]
                .expect("join produced dead entity");
            ((loc.archetype_id.0 as u64) << 32) | (loc.row as u64)
        });
    }
}
```

### Archetype Run Detection

Walk the sorted buffer linearly, detecting boundaries where `archetype_id`
changes. Each run is a contiguous `&[Entity]` slice.

### New Error Variant

```rust
PlanExecError::ComponentMismatch {
    component: &'static str,  // type_name::<T>()
    archetype_id: ArchetypeId,
}
```

Returned when `Q`'s required components are not present in an archetype
encountered during batch iteration. This is a programmer error (caller
specified wrong `Q`), not a runtime condition.

### Fetch Resolution per Run

For each archetype run:
1. Look up `ArchetypeId` -> `&Archetype` in `world.archetypes`
2. Call `Q::init_fetch(archetype, &world.components)` to get `ThinSlicePtr<T>`.
   If `init_fetch` panics (component not in archetype), catch this via a
   pre-check: verify `Q::required_ids(registry)` is a subset of the
   archetype's component set. Return `Err(ComponentMismatch)` if not.
3. For `for_each_batched`: iterate entities, call `Q::fetch(&fetch, row)` per entity
4. For `for_each_join_chunk`: call `Q::as_slice(&fetch, archetype.len())` once,
   collect row indices into a reusable `Vec<usize>` (stored on `QueryPlanResult`),
   yield `(&entities_in_run, &row_indices, slice)` to callback

### Execution Lifecycle

Each batched method follows the same lifecycle:

1. `run_join()` (or compiled scan) populates the scratch buffer
2. `sort_by_archetype()` sorts in-place
3. Archetype-run iteration with `init_fetch`/`fetch`/`as_slice`
4. Tick advancement (for `&mut World` variants)

The sort is performed **every call** -- it is not cached. This is correct
because `run_join()` repopulates the scratch from the current world state
on every call (entities may have been added/removed between calls). The
sort cost (~130 us for 10K entities) is negligible relative to the collection
and intersection phases.

Calling `execute()` followed by `for_each_batched()` is safe -- both re-run
the join from scratch. They do not share state beyond the reusable scratch
allocation.

### Safety Invariants

**Sort key lookup**: The `expect("join produced dead entity")` in
`sort_by_archetype` is a release-mode panic that should never fire because:
- Join collectors iterate live archetypes via `world.query()`
- `run_join()` operates on a point-in-time `&World` snapshot
- No mutations occur between collection and sort (the scratch is plan-local)

**Fetch safety**: `init_fetch` produces a `ThinSlicePtr` that is valid for
the archetype's current allocation. The `&mut World` / `&World` borrow on
the method prevents the callback from mutating entity liveness or triggering
archetype moves -- the world is borrowed by the batched method for the
duration of iteration. This is the same safety guarantee as `for_each`:
the callback receives entity handles and component data, not a world reference.

**Row bounds**: Row indices come from `entity_locations`, which is maintained
by the world's spawn/despawn/move operations. Since the world is borrowed
(not mutated) during iteration, row indices remain valid. `debug_assert!(row <
archetype.len())` provides defense-in-depth in debug builds.

## Benchmarks

### Core Join Benchmarks

Setup: 10K entities with `Score(u32)`, 8K also have `Team(u32)` (80% selectivity).

| Benchmark | Description |
|---|---|
| `join/for_each_get_10k` | Current: `run_join` -> `for_each` -> `world.get()` per entity |
| `join/for_each_batched_10k` | New: sort -> `init_fetch` per archetype -> `fetch` per entity |
| `join/for_each_chunk_10k` | New: sort -> `as_slice` per archetype -> callback per run |
| `join/manual_query_10k` | Baseline: `world.query::<(&Score, &Team)>()` direct iteration |

### Fat Struct Variant

Same four benchmarks with `FatData([u8; 256])` replacing `Score`. Isolates
cache-miss amplification from random-access patterns on large components.

| Benchmark | Description |
|---|---|
| `join_fat/for_each_get_10k` | Random-access 256B structs -- TLB + cache pressure |
| `join_fat/for_each_batched_10k` | Sorted linear stride through 256B columns |
| `join_fat/for_each_chunk_10k` | SIMD-ready slices over 256B columns |
| `join_fat/manual_query_10k` | Monomorphic baseline (no join overhead) |

### Selectivity Sweep

Measures sort cost vs batching savings across join selectivities.

| Benchmark | Join Selectivity |
|---|---|
| `join_selectivity/chunk_10pct` | 1K of 10K entities match |
| `join_selectivity/chunk_50pct` | 5K match |
| `join_selectivity/chunk_90pct` | 9K match |
| `join_selectivity/get_10pct` | Baseline comparison |
| `join_selectivity/get_50pct` | Baseline comparison |
| `join_selectivity/get_90pct` | Baseline comparison |

### Performance Predictions

| Benchmark | Predicted Time | Rationale |
|---|---|---|
| `join/for_each_get_10k` | 80-100 us | 8K x (gen check + loc lookup + TypeId hash + col search) |
| `join/for_each_batched_10k` | 10-15 us | 8K x (1 loc lookup + 1 ptr add), 2 archetype resolutions |
| `join/for_each_chunk_10k` | 8-12 us | 2 archetype slices, sorted row indices, prefetcher-friendly |
| `join/manual_query_10k` | 4-5 us | Monomorphic QueryIter baseline |
| `join_fat/for_each_get_10k` | 200-400 us | 256B random access = TLB + L2 misses |
| `join_fat/for_each_chunk_10k` | 10-20 us | Linear stride hides latency via prefetch |

Target: `for_each_chunk` within 1.5-2x of `manual_query`. If achieved, the
relational join overhead is effectively solved.

## Test Plan

Unit tests in `planner.rs::tests`:

1. `for_each_batched_yields_all_join_results` -- entity count + values match `execute()`
2. `for_each_batched_raw_no_tick_advance` -- read-only variant preserves tick
3. `for_each_join_chunk_yields_correct_slices` -- slice lengths sum to join result count
4. `for_each_join_chunk_works_for_scan_plans` -- non-join plans use existing archetype iteration
5. `for_each_batched_left_join` -- left join preserves all left entities in sorted order
6. `for_each_join_chunk_multi_archetype` -- 3+ archetypes, each chunk = one archetype
7. `for_each_batched_empty_join` -- zero results, callback never called
8. `for_each_batched_world_mismatch` -- returns `Err(WorldMismatch)`

### Miri Coverage

Add to `ci/miri-subset.txt` (exercises `init_fetch` -> `fetch`/`as_slice`
through type-erased `ThinSlicePtr` column pointers):

```
planner::tests::for_each_batched_yields_all_join_results
planner::tests::for_each_join_chunk_yields_correct_slices
planner::tests::for_each_join_chunk_multi_archetype
```

## Files Modified

| File | Change |
|---|---|
| `crates/minkowski/src/planner.rs` | `sort_by_archetype`, `for_each_batched`, `for_each_batched_raw`, `for_each_join_chunk` on `QueryPlanResult`; archetype-run iteration helper; `ComponentMismatch` error variant; `row_indices: Vec<usize>` field on `QueryPlanResult` |
| `crates/minkowski-bench/benches/planner.rs` | 14 new benchmarks (join, join_fat, join_selectivity) |
| `crates/minkowski-bench/src/lib.rs` | `FatData([u8; 256])` + `Team(u32)` component types |
| `ci/miri-subset.txt` | 3 new test entries |
| `ci/run-miri-subset.sh` | Add planner batch tests to EXACT_TESTS |
| `docs/perf-roadmap.md` | Update P1-2 status, add join batch results |

## Non-Goals

- Column-aware aggregate extraction (P1-2) -- same underlying problem but different
  shape requirements. Solve joins first, prove the pattern, then port to aggregates.
- `for_each_join_chunk_raw` -- can be added later if transactional join chunks are needed.
  Starting with the `&mut World` variant keeps scope tight.
- Automatic sort-skip threshold -- measure first. If the sort is always beneficial
  down to 100 entities, no threshold is needed.
