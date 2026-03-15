# Performance Roadmap

Known bottlenecks and optimization opportunities, ranked by estimated impact.
Updated: v1.2.0 (2026-03-14). All numbers from `cargo bench -p minkowski-bench`.

## Priority 1 — High Impact, Clear Path

### P1-1: Lock-free slab pool allocator + Thread-Local Cache --- COMPLETED

**Implementation**: Lock-free intrusive stack via 128-bit tagged pointer CAS
(`Atomic<u128>`). ABA prevention via 64-bit monotonic tag. Side table routes
deallocation to the correct class. Single-step overflow from exhausted class
to the next larger class. Atomic next-pointer read/write for Tree Borrows
compliance. Provenance restored via `self.base` offset arithmetic.

**Thread-Local Cache (TCache)**: Per-thread L1 cache with 6 per-class bins
(capacity 32, refill 16, spill 16). 15/16 allocations hit the L1 cache
(~3 instructions) instead of the global lock-free stack (~7 CAS ops).
Epoch-based lazy flush for Rayon hoarding prevention.
`World::flush_pool_caches()` exposes epoch bump to users.

**Results**: `simple_insert/pool` = 8.83 ms (unchanged, dominated by entity/archetype overhead),
`add_remove/pool` = 1.35 ms (was 8.03 ms, **6x improvement**). The add_remove benchmark
is allocation-heavy (alloc + dealloc per entity) and benefits directly from TCache.
simple_insert is dominated by entity creation and archetype management, not raw allocation.

**Benchmark**: `simple_insert/pool`, `add_remove/pool`

---

### P1-2: Aggregate extractor overhead (13x)

**Current**: `execute_aggregates` calls `world.get::<T>(entity)` per entity per
aggregate through a type-erased `Arc<dyn Fn>` extractor. Each call does an
`entity_locations` lookup + archetype column access.

**Evidence**: 76 µs for COUNT+SUM over 10K entities vs 5.8 µs for a manual
`world.query()` loop doing the same work — **13x overhead**.

**Target**: Column-aware aggregate path that reads directly from `BlobVec`
pointers during archetype iteration (like `for_each_chunk`). The extractor
would receive a typed slice instead of individual entity handles.

**Complexity**: Moderate — requires plumbing archetype column pointers through
the aggregate pipeline. The `AggregateExpr` API would need a `column_extractor`
variant alongside the current `entity_extractor`.

**Benchmark**: `planner/aggregate_count_sum_10k` vs `planner/manual_count_sum_10k`

---

### P1-4: Join batch execution --- COMPLETED

**Implementation**: Archetype-sorted batch execution for join plans.
After `run_join()` materialises entities into ScratchBuffer, sort by
packed `(archetype_id << 32 | row)` key. Walk archetype runs calling
`init_fetch` once per archetype, `fetch`/`as_slice` per entity.

**Results**: After join elimination (P1-5), inner joins become pure scans.
The batch extraction methods now operate on the scan path:

| Benchmark | v1.3.1 (join) | v1.3.2 (eliminated) |
|---|---|---|
| `join/for_each_get_10k` | 122 us | 47 us |
| `join/for_each_batched_10k` | 103 us | 27 us |
| `join/for_each_chunk_10k` | 109 us | 35 us |
| `join/manual_query_10k` | 4.8 us | 4.8 us |

The remaining gap vs manual_query is `execute()` scratch buffer overhead
(entity-by-entity push into Vec). `for_each()` on eliminated plans uses
the streaming scan path at ~10 us.

**API**: `for_each_batched`, `for_each_batched_raw`, `for_each_join_chunk`
on `QueryPlanResult`.

---

### P1-5: Join elimination (inner joins → scan rewrite) --- COMPLETED

**Implementation**: Build-time detection of inner joins that are pure
component-presence filters. Merges right-side required/changed bitsets
into the left-side scan, replaces compile_for_each factories with
merged-bitset closures, drops the join entirely. Phase 4b in
`ScanBuilder::build()`.

**Results**: Inner joins are eliminated entirely — no `run_join()`
materialization, sort, or intersection. The plan becomes a pure archetype
scan with the merged component requirements. `join/for_each_get_10k`
dropped from 122 us to 47 us (2.6x). `for_each_batched` dropped from
103 us to 27 us (3.8x). The `for_each()` streaming path (no scratch
buffer) runs at ~10 us — within 2x of the manual query baseline.

**API**: Automatic — no user code changes needed. `PlanWarning::JoinEliminated`
informs users when the optimization fires.

---

### P1-3: Spawn batching

**Current**: `world.spawn()` resolves the archetype per entity (hash lookup on
sorted component IDs). For homogeneous spawns (same bundle type), this is
redundant after the first call.

**Evidence**: `simple_insert/batch` = 1.74 ms (174 ns/entity). Archetype
lookup is ~20% of per-entity cost based on profiling.

**Target**: `world.spawn_batch(iter)` that resolves the archetype once, reserves
capacity with a single `BlobVec::reserve(n)`, then pushes N entities in a tight
loop. Expected ~30% improvement for homogeneous batch spawns.

**Complexity**: Low — the archetype resolution and BlobVec push are already
separate internal steps; batching is a matter of hoisting the resolution.

**Benchmark**: `simple_insert/batch`

---

## Priority 2 — Moderate Impact, Moderate Effort

### P2-1: Planner scan overhead (2.5x)

**Current**: `QueryPlanResult::for_each` dispatches through `Box<dyn FnMut>`,
while `world.query().for_each()` uses monomorphic `QueryIter`.

**Evidence**: 9.5 µs vs 3.9 µs for 10K entities — **2.5x overhead**.

**Analysis**: This is structural to type-erased plan composition. The plan is
compiled once and re-executed with different callbacks, preventing
monomorphization. Possible mitigations:
- `for_each_chunk`-style plan execution (yield typed slices, not individual
  entities) — would reduce callback count by archetype-entity-count factor
- Specialize the compiled closure for common query types at plan-build time

**Benchmark**: `planner/scan_for_each_10k` vs `planner/query_for_each_10k`

---

### P2-2: Custom filter per-entity overhead

**Current**: `Predicate::custom` dispatches through `Arc<dyn Fn>` per entity,
plus a `world.get::<T>(entity)` inside the user's closure.

**Evidence**: 63.4 µs for 50% selectivity over 10K entities (12.7 ns/entity).
The `world.get()` inside the custom closure is the dominant cost.

**Target**: Provide a column-aware custom filter variant that receives typed
slices, enabling batch evaluation without per-entity `world.get()`.

**Benchmark**: `planner/custom_filter_50pct`

---

### P2-3: QueryWriter apply phase

**Current**: `EnumChangeSet::apply` processes mutations one at a time with a
per-mutation enum match + entity_locations lookup + column memcpy.

**Evidence**: `reducer/query_writer_10k` = 93 µs (~11x `query_mut_10k`).
The apply phase is ~62% of the profile (per previous flamegraph analysis).

**Target**: Batch overwrite fast path — group mutations by archetype, sort by
row, and apply with a single pass per column. Estimated ~20-30% improvement
on the apply phase.

**Complexity**: Moderate — requires mutation grouping logic in `apply()` and
a fast path that bypasses the per-mutation enum match.

**Benchmark**: `reducer/query_writer_10k`, `changeset/apply_10k_overwrites`

---

### P2-4: WAL replay throughput

**Current**: Per-mutation deserialization + entity lookup + changeset apply.

**Evidence**: `serialize/wal_replay` = 1.72 ms for 1K mutations (1.72 µs/mutation),
while `wal_append` is 1.22 µs for a single mutation.

**Target**: Batch deserialization + archetype-grouped apply. Sort decoded
mutations by archetype before applying to improve cache locality.

**Benchmark**: `serialize/wal_replay`

---

## Priority 3 — Low Impact or Structural Limitations

### P3-1: `for_each` vs `for_each_chunk` gap (10x)

**Evidence**: 14.5 µs vs 1.5 µs — per-element callback prevents SIMD.

**Status**: Not an engine optimization. Users should prefer `for_each_chunk`
for numeric workloads. Document prominently.

### P3-2: DynamicCtx overhead (15x vs query_mut)

**Evidence**: 132 µs vs 8.6 µs. Remaining after identity hasher optimization.

**Status**: Structural — runtime type resolution + buffered writes. The
collect-then-write pattern can't be eliminated without losing dynamic dispatch.

### P3-3: Rayon `par_for_each` overhead at low entity counts

**Evidence**: 289 µs at 10K entities vs 1.5 µs for `for_each_chunk`.

**Status**: Rayon characteristic, not Minkowski. Amortizes above ~50K entities
or with expensive per-entity work.

### P3-4: Changeset spawn overhead (1.5x)

**Evidence**: `simple_insert/changeset` = 2.58 ms vs 1.74 ms for direct spawn.

**Status**: Inherent to data-driven mutation model (arena allocation + mutation
log). The overhead is the price of undo/redo and WAL compatibility.

### P3-5: MaterializedView cache copy

**Status**: `extend_from_slice` on every refresh. Borrowing the scratch directly
would tie `entities()` lifetime to `&mut self`. O(N) where N is typically small
for subscription queries. Not worth the API complexity.

---

## Baselines Reference (v1.2.0)

| Benchmark | Time | Per-entity |
|---|---|---|
| `simple_iter/for_each_chunk` | 1.5 µs | 0.15 ns |
| `simple_iter/for_each` | 14.5 µs | 1.45 ns |
| `reducer/query_mut_10k` | 8.6 µs | 0.86 ns |
| `reducer/query_mut_chunk_10k` | 1.5 µs | 0.15 ns |
| `reducer/query_writer_10k` | 93 µs | 9.3 ns |
| `reducer/dynamic_for_each_10k` | 132 µs | 13.2 ns |
| `simple_insert/batch` | 1.74 ms | 174 ns |
| `simple_insert/pool` | 8.83 ms | 883 ns |
| `add_remove/add_remove` | 1.30 ms | 130 ns |
| `add_remove/pool` | 1.35 ms | 135 ns |
| `planner/scan_for_each_10k` | 9.5 µs | 0.95 ns |
| `planner/query_for_each_10k` | 3.9 µs | 0.39 ns |
| `planner/btree_range_10pct` | 11.4 µs | 11.4 ns/match |
| `planner/hash_eq_1` | 39 ns | — |
| `planner/custom_filter_50pct` | 63.4 µs | 12.7 ns |
| `planner/changed_skip_10k` | 6.8 ns | — |
| `planner/aggregate_count_sum_10k` | 76.1 µs | 7.6 ns |
| `planner/manual_count_sum_10k` | 5.8 µs | 0.58 ns |
| `planner/execute_collect_10k` | 14.7 µs | 1.47 ns |
| `serialize/wal_replay` | 1.72 ms | 1.72 µs |
| `serialize/wal_append` | 1.22 µs | — |
