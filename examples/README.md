# Examples

Each example is a standalone binary that exercises a different slice of the Minkowski API. Run any of them with:

```
cargo run -p minkowski-examples --example <name> --release
```

| Example | What it does |
|---|---|
| `boids` | Flocking simulation with 5 000 entities. Registers three `QueryMut` reducers (zero acceleration, compute forces, integrate), rebuilds a `SpatialGrid` each frame for O(k) neighbor lookup, and uses `CommandBuffer` for deferred despawn/respawn churn. Demonstrates SIMD auto-vectorization via `for_each_chunk`. |
| `life` | Conway's Game of Life on a 64x64 toroidal grid, running 500 generations with undo/redo. Uses `#[derive(Table)]` for typed row access, a `QueryMut` reducer to update neighbor counts, `Changed<CellState>` for incremental detection, and `EnumChangeSet` to record reversible mutations for 50-generation rewind followed by deterministic replay. |
| `nbody` | Barnes-Hut N-body gravity simulation with 2 000 entities. Builds a quadtree `SpatialIndex` each frame to approximate O(N log N) force computation, uses rayon snapshot-based parallel force accumulation, and dispatches the integration step through a `QueryMut` reducer. |
| `scheduler` | Greedy batch scheduler over 6 registered query reducers. Extracts `Access` bitsets, builds a conflict matrix, and assigns systems to non-conflicting batches via graph coloring — producing 3 batches where intra-batch systems could run in parallel. |
| `transaction` | Two-part comparison of raw `Tx` building blocks versus reducer-based dispatch. Part 1 shows `Sequential` begin/commit and `Optimistic` transact closure. Part 2 registers the same logic as query reducers, gaining strategy-agnostic dispatch and free conflict detection. |
| `battle` | Multi-threaded arena with 500 entities over 100 frames. Registers `EntityMut` reducers for attack and heal, dispatches through both `Optimistic` and `Pessimistic` strategies. Demonstrates low-conflict versus high-conflict modes and how each strategy handles retry. |
| `persist` | Full persistence lifecycle with 100 entities across 3 archetypes. Snapshot save/load, `Durable`-wrapped `QueryWriter` reducer (WAL-backed), crash recovery via snapshot + WAL replay. |
| `replicate` | Pull-based WAL replication via channel-simulated network. Source sends snapshot bytes + WAL batch bytes; replica reconstructs independently with its own `CodecRegistry` (different registration order). Verifies convergence. |
| `reducer` | Tour of all 7 reducer handle types: `EntityMut` (heal), `QueryMut` (gravity), `QueryRef` (logger), `Spawner` (spawn projectiles), `QueryWriter` (drag), `DynamicCtx` (runtime-flexible access). Plus name-based lookup, access conflict detection, and structural despawn. |
| `index` | Column index demo on a `Score` component across two archetypes (200 entities). `BTreeIndex` for O(log n) range queries, `HashIndex` for O(1) exact lookups, incremental updates via `ChangeTick`, stale-entry detection after despawn. Also demonstrates `QueryPlanner` with `IndexDriver` execution for planned range and eq queries. |
| `flatworm` | Planarian flatworm simulator with 200 worms over 1 000 frames. Chemotaxis, binary fission, starvation despawn, food respawning. Demonstrates `SpatialIndex` trait, `CommandBuffer` for deferred spawn/despawn, and `QueryMut`/`QueryRef` reducers. |
| `circuit` | Analog circuit simulator: 555 astable oscillator, LCR bandpass filter, 741 voltage follower. Circuit nodes are entities with `Voltage` components. Symplectic Euler integration, `QueryMut`/`QueryRef` reducers, ASCII waveform output. 200K steps at 100 ns timestep. |
| `tactical` | Multi-operator tactical map with server-authoritative replication. Exercises sparse components, `par_for_each`, `Optimistic` transactions, entity wire serialization, `HashIndex` stale filtering, and `EnumChangeSet` replication packets. Two operator threads communicate via `mpsc` channels. |
| `observe` | Observability companion demo. Captures `MetricsSnapshot` before and after entity churn, computes `MetricsDiff`, and renders Prometheus-format metrics via `PrometheusExporter`. |
| `blob` | Blob offloading demo. `BlobRef` components hold external keys (S3 paths); `MemoryBlobStore` lifecycle trait manages cleanup. Demonstrates entity spawn with blob refs, despawn + orphan collection, and store-level deletion. |
| `retention` | Tick-based entity retention with `Expiry` countdown component. `RetentionReducer` dispatches decrement counters; entities reaching zero are despawned. Shows dispatch-count TTL and progressive despawn over 8 frames. |
| `pool` | TigerBeetle-style pre-allocated memory pool. `WorldBuilder` with 16 MB budget, `try_spawn` until `PoolExhausted`, pool statistics. Demonstrates bounded-memory operation with no dynamic allocation on the data path. |
| `profile_changeset` | Profiling harness comparing `QueryWriter` vs `QueryMut` reducer throughput. 10K entities, 1K iterations, flamegraph-compatible output. Useful for benchmarking mutation strategies. |
| `planner` | Volcano query planner demo. Cost-based plan compilation, automatic BTree/Hash index selection, join execution, `for_each`/`for_each_raw` iteration, subscription queries with `Changed<T>`, aggregate computation, and `explain()` output. |
| `materialized_view` | Materialized views over subscription queries. `MaterializedView` wraps a `QueryPlanResult` with configurable `DebouncePolicy` for cached, change-detected entity snapshots. Demonstrates immediate and tick-based debouncing, invalidation, multi-index subscriptions, and dynamic policy switching. |
