# Performance

How Minkowski achieves its performance characteristics — the optimizations we made, why they matter, and what the structural limits are. All numbers from `cargo bench -p minkowski-bench` on a single core.

## Iteration

Column-oriented storage means components of the same type are contiguous in memory. A query like `world.query::<(&mut Pos, &Vel)>()` walks two arrays in lockstep — the prefetcher sees a linear access pattern and stays ahead.

**`for_each_chunk` yields typed slices** that LLVM auto-vectorizes. This is 10x faster than per-element `for_each` (1.55 µs vs 14.6 µs for 10K entities) because the compiler can use SIMD instructions on the contiguous slice. Use `for_each_chunk` for numeric workloads; use `for_each` when you need `Entity` handles or branching logic.

**`par_for_each` distributes chunks across Rayon threads.** At 10K entities the thread-pool overhead dominates (~340 µs); at 50K+ entities it amortizes and scales linearly.

## Query Planner

The planner compiles queries into push-based execution plans at build time, then executes them repeatedly against live world data.

### Join Elimination (343x)

Inner joins in an ECS are often redundant — they're testing whether entities have certain components, which archetype matching already knows. The planner detects inner joins that are pure component-presence filters and merges them into the left-side scan at build time. No materialization, no sort, no intersection. `join/for_each_batched_10k` dropped from 103 µs to 300 ns.

### Direct Archetype Iteration (1,792x)

Scan-only plans (no joins, no custom predicates, no index/spatial drivers) bypass the ScratchBuffer entirely and walk archetypes with `init_fetch`/`fetch` inline. This eliminates the type-erased `CompiledForEach` dispatch layer. `scan_for_each_10k` dropped from 9.5 µs to 5.25 ns — essentially the cost of a single archetype metadata check.

### Column-Aware Custom Filters

`Predicate::custom` dispatches through `Arc<dyn Fn>` per entity with a `world.get::<T>()` inside the closure — 12.7 ns/entity overhead. `Predicate::custom_column` receives `&T` directly from contiguous column slices, resolving the column once per archetype instead of per entity. Uses a boolean mask that multiple column filters AND together.

### Aggregates (13x)

Aggregates (COUNT, SUM, MIN, MAX, AVG) use cached extractors with specialized inner loops that iterate archetype columns directly, bypassing per-entity `world.get()`. The planner's aggregate path is now faster than a hand-written `world.query()` loop (5.84 µs vs 6.45 µs for 10K entities) because it avoids iterator machinery.

### Cache-Aware Partitioned Joins

For large hash joins where the working set exceeds L2 cache, entities are bucketed by `Entity::to_bits() % partitions` so each partition fits in cache during intersection. Partition count is computed from `build_rows * avg_component_bytes / l2_cache_bytes`. Falls back to single-partition `sorted_intersection` for small joins.

## Memory Allocation

### Lock-Free Slab Pool (6x)

The pool allocator uses lock-free intrusive stacks via 128-bit tagged pointer CAS (`Atomic<u128>`). ABA prevention via 64-bit monotonic tag. A side table routes deallocation to the correct size class.

### Thread-Local Cache

A per-thread L1 cache with 32-slot bins per size class sits in front of the lock-free stack. 15 out of 16 allocations hit the L1 cache (~3 instructions) instead of the global stack (~7 CAS operations). Epoch-based lazy flush prevents Rayon worker threads from hoarding memory. `add_remove/pool` dropped from 8.03 ms to 1.35 ms — within 5% of jemalloc.

## Entity Spawning

### Batch Spawning (5.2x)

`World::spawn_batch()` resolves the target archetype once, reserves column capacity with a single `BlobVec::reserve(n)`, then pushes entities in a tight loop. Individual `spawn()` calls pay the archetype hash-lookup per entity. `simple_insert/spawn_batch`: 343 µs vs 1.78 ms for individual spawns.

## Transactional Writes

### QueryWriter Streaming Archetype Buffers (1.5x)

`QueryWriter` buffers writes into an `EnumChangeSet` for atomic commit. The naive approach — push a `Mutation::Insert` per entity, apply by looking up each entity's location — paid 37% of execution time on per-entity bookkeeping.

The optimization: during `for_each`, the writer opens a pre-resolved `ArchetypeBatch` per archetype with `ColumnBatch` entries that cache the column index, drop function, and layout. `WritableRef::set()` pushes directly to the current batch via a pre-resolved `column_slot` index. The apply phase drains batches with zero per-entity lookups — no `is_alive`, no `entity_locations`, no `column_index`, no `ComponentRegistry::info`.

`query_writer_10k` dropped from 93 µs to 64 µs. Sparse updates (10% of entities modified) run at 5.9 µs — near `query_mut` territory. The remaining gap to `query_mut` (1.6 µs) is structural: arena allocation + clone for buffered writes. This is the cost of atomic commit semantics.

## Persistence

### WAL Replay (1.7x)

WAL replay collects all records in a first pass, then decodes and applies them as a single `EnumChangeSet`. This eliminates per-record changeset allocation, per-record `apply()` overhead, and per-record tick advancement. Throughput improved from 581K to 943K mutations/second.

## Structural Limits

Some costs are inherent to the design and cannot be optimized away without changing the model:

- **QueryWriter vs query_mut (41x gap)** — the remaining 64 µs is the cost of buffered atomic commits: arena allocation, value cloning, and changeset management. `query_mut` writes directly to column memory with no buffer. Closing this gap requires abandoning the buffered-write model (direct write + undo log, or shadow columns), which breaks read stability during iteration.

- **DynamicCtx overhead (15x vs query_mut)** — runtime type resolution + the collect-then-write pattern. Cannot be eliminated without losing dynamic dispatch.

- **Changeset spawn overhead (1.5x vs direct spawn)** — the arena allocation + mutation log is the price of undo/redo and WAL compatibility.

- **`for_each` vs `for_each_chunk` (10x)** — per-element callbacks prevent SIMD auto-vectorization. This is a user API choice, not an engine limitation.

## Benchmark Reference (v1.3.0)

Run benchmarks with `cargo bench -p minkowski-bench`.

### Iteration

| Benchmark | Time | Per-entity |
|---|---|---|
| `simple_iter/for_each_chunk` | 1.55 µs | 0.16 ns |
| `simple_iter/for_each` | 14.6 µs | 1.46 ns |
| `reducer/query_mut_10k` | 1.56 µs | 0.16 ns |
| `reducer/query_writer_10k` | 64.5 µs | 6.45 ns |
| `reducer/query_writer_sparse_update_10k` | 5.87 µs | 0.59 ns |
| `reducer/dynamic_for_each_10k` | 115.8 µs | 11.6 ns |

### Query Planner

| Benchmark | Time |
|---|---|
| `planner/scan_for_each_10k` | 5.25 ns |
| `planner/query_for_each_10k` | 5.87 µs |
| `planner/aggregate_count_sum_10k` | 5.84 µs |
| `planner/custom_filter_50pct` | 64.6 µs |
| `planner/btree_range_10pct` | 9.57 µs |
| `planner/hash_eq_1` | 42 ns |
| `planner/changed_skip_10k` | 7.3 ns |

### Joins

| Benchmark | Time |
|---|---|
| `join/for_each_batched_10k` | 300 ns |
| `join/for_each_chunk_10k` | 3.18 µs |
| `join/for_each_get_10k` | 37.9 µs |
| `join/manual_query_10k` | 4.76 µs |

### Entity Management

| Benchmark | Time |
|---|---|
| `simple_insert/spawn_batch` | 343 µs |
| `simple_insert/batch` | 1.78 ms |
| `add_remove/pool` | 1.32 ms |
| `add_remove/add_remove` | 1.32 ms |

### Persistence

| Benchmark | Time |
|---|---|
| `serialize/wal_append` | 1.25 µs |
| `serialize/wal_replay` | 1.06 ms |
