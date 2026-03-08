# Changelog

## 1.0.0

Initial stable release of the Minkowski column-oriented ECS storage engine.

### Core ECS (`minkowski`)

**Storage**
- Column-oriented archetype storage with 64-byte aligned `BlobVec` columns
- Generational entity IDs (32-bit index + 32-bit generation packed into u64)
- `EntityAllocator` with free list recycling and lock-free `reserve()` via `AtomicU32`
- Archetype migration on component insert/remove with automatic `EntityLocation` fixup
- `PagedSparseSet` (4096-entry pages) for opt-in sparse component storage
- `CommandBuffer` for deferred structural mutations during iteration
- `WorldStats` for entity count, archetype count, and component registry introspection

**Queries**
- Tuple-typed queries with `WorldQuery` trait (1-12 tuples via macro)
- Transparent query cache with incremental archetype scan
- `Changed<T>` filter for tick-based change detection
- `par_for_each` parallel iteration via rayon
- `for_each_chunk` yields typed slices for SIMD auto-vectorization
- `Option<&T>` support (accesses without requiring)
- `query_raw(&self)` shared-ref read path for transactional reads

**Schema**
- `#[derive(Table)]` proc macro for named schemas with typed row accessors
- Table queries skip archetype matching entirely
- `TableDescriptor` caches archetype-to-column mapping

**Mutations**
- `EnumChangeSet` — data-driven mutation log with reversible `apply()` for undo/redo
- Typed safe helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`)
- Arena-backed component byte storage

**Change Detection**
- Per-column `changed_tick` marking on every mutable access path
- Monotonic tick counter auto-advancing on mutation and query
- `Changed<T>` skips entire archetypes at the column level

**Transactions**
- `Transact` trait with `transact()` retry loop and `begin()`/`try_commit()` building blocks
- `Sequential` — zero-cost passthrough for single-threaded use
- `Optimistic` — concurrent reads via `query_raw`, tick-based validation at commit
- `Pessimistic` — cooperative per-column locks with upgrade semantics, spin+yield backoff
- `Tx` split-phase design: reads through `&World`, writes buffered in `EnumChangeSet`
- `OrphanQueue` for leak-free entity ID cleanup on abort (automatic drain)
- `WorldId` cross-world corruption prevention

**Typed Reducers**
- `ReducerRegistry` with 6 typed registration methods + dynamic builder
- `EntityRef<C>` / `EntityMut<C>` — single-entity read/write handles
- `Spawner<B>` — entity creation via lock-free `reserve()`
- `QueryRef<Q>` / `QueryMut<Q>` — scheduled read/read-write iteration
- `QueryWriter<Q>` — buffered query iteration with `WritableRef<T>`, compatible with `Durable`
- `DynamicCtx` — runtime-validated access with `can_read`/`can_write`/`can_spawn`/`can_despawn`
- `ComponentSet` / `Contains<T, INDEX>` for compile-time component set declarations
- `Access` bitsets for scheduler conflict detection (`conflicts_with`, `is_compatible`)

**Indexing**
- `SpatialIndex` trait for external spatial data structures (grids, quadtrees, BVH)
- `BTreeIndex<T>` — O(log n) range queries on component values
- `HashIndex<T>` — O(1) exact match lookups
- Generational validation catches stale index entries automatically

### Persistence (`minkowski-persist`)

- Segmented WAL with append, replay, rotation, truncation, and crash recovery
- `Durable<S>` wraps any `Transact` strategy for WAL-backed durability
- rkyv zero-copy snapshots via mmap (`save_to_bytes` / `load_from_bytes`)
- `ReplicationBatch` — transport-agnostic wire format for incremental replication (`to_bytes`/`from_bytes`/`apply_batch`)
- `WalCursor` — filesystem-specific WAL reader for producing batches from local segment files
- `CodecRegistry` for stable cross-process component identity
- `CheckpointHandler` trait with `AutoCheckpoint` default
- WAL checkpoint markers for coordinated snapshot + truncation

### Observability (`minkowski-observe`)

- `MetricsSnapshot::capture()` — point-in-time world + WAL metrics
- `MetricsDiff::compute()` — diff between two snapshots with entity churn tracking
- `PrometheusExporter` — 13 OpenMetrics gauges for world state, WAL pressure, per-archetype breakdowns
- `Display` impls for human-readable snapshot and diff output
- Exact spawn/despawn counters via `EntityAllocator`

### Derive Macro (`minkowski-derive`)

- `#[derive(Table)]` generates `Ref<'w>` / `Mut<'w>` row accessors, `TableDescriptor`, and `query_table` / `query_table_mut` methods

### Python Bridge (`minkowski-py`)

- PyO3 bindings exposing `World` and `ReducerRegistry` to Python
- Arrow RecordBatch data transfer — loads directly into Polars DataFrames
- `world.query()` for zero-copy reads, `world.write_column()` for bulk writes
- `registry.run()` dispatches pre-compiled Rust reducers by name
- 9 registered component types and 5 built-in reducers (boids, gravity, life, movement)
- Jupyter notebook examples for boids, circuit simulation, and flatworm simulation

### Verification

- 398 unit tests across all modules
- Miri + Tree Borrows for undefined behavior detection
- ThreadSanitizer for data race detection
- Loom exhaustive concurrency verification (OrphanQueue, ColumnLockTable, EntityAllocator)
- Fuzz testing (World operations, query iteration, snapshot deserialization, WAL replay)
- CI: fmt, clippy, test, Miri, TSan, Loom on every PR

### Examples

14 examples covering the full API surface: boids, life, nbody, scheduler, transaction, battle, persist, replicate, reducer, index, flatworm, circuit, tactical, observe.

### Documentation

- Architecture Decision Records in `docs/adr/`
- Reducer correctness guide (`docs/reducer-correctness.md`)
- CLAUDE.md with build commands, architecture overview, key conventions
