# Changelog

## 1.0.4

### Performance

- **`EnumChangeSet::apply()` ~30-40% faster** — reverse changeset construction eliminated (was doubling per-mutation work with arena alloc + Vec push for undo data that no caller used). Single `next_tick()` atomic per `apply()` call instead of per mutation.

### Breaking Changes

- **`EnumChangeSet::apply()` returns `Result<(), ApplyError>`** instead of a reverse `EnumChangeSet`. The reverse changeset (for undo/redo) has been removed — it unconditionally doubled apply cost for a feature only used by one example.
- **`ApplyError` enum** — `DeadEntity(Entity)` and `AlreadyPlaced(Entity)` variants. Replaces `assert!` panics with fallible error returns, consistent with `ReducerError`.
- **`examples/life.rs` undo/redo removed** — the example now demonstrates Table queries and QueryMut without time-travel.

### Benchmarks

- **New `minkowski-bench` crate** with 8 standardized scenarios modeled after [ecs_bench_suite](https://github.com/rust-gamedev/ecs_bench_suite): `simple_insert`, `simple_iter`, `fragmented_iter`, `heavy_compute`, `add_remove`, `schedule`, `serialize`, `reducer`.
- Old ad-hoc benchmarks removed from `crates/minkowski/benches/`. `hecs` dev-dependency dropped.

## 1.0.3

### Persistent Indexes (`minkowski` + `minkowski-persist`)

- **`PersistentIndex` trait** — secondary indexes can now be saved to disk and loaded on recovery. Recovery time is proportional to the WAL tail, not world size. `save` is object-safe; `load` is on the concrete type.
- **`BTreeIndex::save/load`** and **`HashIndex::save/load`** — rkyv-based serialization with CRC32 integrity and atomic rename (write to `.tmp`, then rename). Conditional on key type supporting rkyv.
- **`load_btree_index<T>`** and **`load_hash_index<T>`** — free functions to load persisted indexes from disk.
- **`AutoCheckpoint::register_index`** — optionally saves persistent indexes alongside snapshots on checkpoint. Index save failures are non-fatal.
- **Index file format** — `[magic: 8B "MK2INDXK"][crc32: 4B LE][version: u32 LE][len: u64 LE][rkyv payload]`. Same envelope pattern as WAL segments and snapshots. Type discriminator in payload catches wrong-type loads.
- **`ChangeTick::to_raw/from_raw`** — public serialization surface for tick values, enabling the persist crate to save/restore index synchronization state.
- **`BTreeIndex::as_raw_parts/from_raw_parts`** and **`HashIndex::as_raw_parts/from_raw_parts`** — controlled access to internal state for serialization without exposing private fields.
- **`IndexPersistError`** — `Io` and `Format` variants, consistent with `SnapshotError` and `WalError`.

### Recovery flow

```rust
let post_restore_tick = world.change_tick(); // after snapshot restore, before WAL replay
wal.replay(&mut world, &codecs)?;

let mut index = match load_btree_index::<Score>("score.idx", post_restore_tick) {
    Ok(idx) => { idx.update(&mut world); idx }  // O(WAL tail)
    Err(_) => { let mut idx = BTreeIndex::new(); idx.rebuild(&mut world); idx }
};
```

`rebuild()` remains as fallback for missing or corrupt index files.

### Durability

- **`Snapshot::save` uses atomic rename + fsync** — writes to `.snap.tmp`, calls `sync_data()`, then renames. A crash during write cannot corrupt an existing snapshot.
- **`write_index_file` uses atomic rename + fsync** — same pattern. Best-effort cleanup of `.tmp` file on write failure.

## 1.0.2

### Integrity Checking (`minkowski-persist`)

- **CRC32 WAL frame checksums** — every WAL frame is now `[len: u32 LE][crc32: u32 LE][payload]`. CRC32 (IEEE via `crc32fast`, hardware-accelerated) covers the rkyv payload bytes. Checksum mismatches during replay are treated as torn writes and truncated during crash recovery.
- **CRC32 snapshot checksums** — v2 snapshot format `[magic: 8B "MK2SNAPK"][crc32: 4B LE][reserved: 4B][len: u64 LE][payload]`. Mismatches return `SnapshotError::Format`. Legacy v1 snapshots (no CRC) are still loadable.
- **WAL segment format versioning** — 4-byte magic `"MKW2"` identifies v2 segments. Legacy v1 segments produce a hard `WalError::Format` error with a migration message.
- **Entity generation consistency check on snapshot restore** — after restoring allocator state, validates that every archetype entity's generation matches the allocator. Returns `SnapshotError::Format` on mismatch (corrupt snapshot), consistent with all other corruption detection paths.

### Robustness (`minkowski`)

- **Archetype column-length consistency** — `debug_assert_consistent()` after `despawn`, `despawn_batch`, and all `EnumChangeSet::apply` structural mutations (Spawn, Insert migration, Remove migration). Catches column/entity count desync during development at zero release cost.
- **Arena bounds check** — `Arena::get` debug_assert validates offset bounds.
- **BlobVec pointer bounds** — `ptr_at` debug_assert validates index bounds.
- **EntityAllocator::reserve overflow** — CAS loop prevents atomic wraparound past `u32::MAX`.
- **ColumnLockTable release assertions** — validates lock state matches expectations on release.
- **EntityLocation row validity** — debug_assert in get_mut, insert, remove, despawn.

### Cleanup

- Removed unused slotted page infrastructure (append-only WAL has no use for page-level space management).

## 1.0.1

### Reducer Error Handling

- **`ReducerError` enum** replaces panics for API misuse in `ReducerRegistry`. Variants: `WrongKind`, `DuplicateName`, `TransactionConflict`, `InvalidId`. Args type mismatches remain panics (static programming errors, consistent with the assert boundary rule).
- All registration methods (`register_entity`, `register_spawner`, `register_query`, `register_query_writer`, `DynamicReducerBuilder::build`, etc.) now return `Result<_, ReducerError>`.
- Dispatch methods (`call`, `run`, `dynamic_call`) return `Result<(), ReducerError>` with bounds checking on reducer IDs.
- **`ReducerInfo`** struct for runtime introspection: `reducer_info()`, `query_reducer_info()`, `dynamic_reducer_info()` return name, kind, access pattern, change tracking, and despawn capability.
- **`DynamicCtx` introspection**: `is_declared<T>()`, `is_writable<T>()`, `is_removable<T>()`, `can_despawn()` for runtime capability checking.
- Registry introspection: `reducer_count()`, `dynamic_reducer_count()`, `registered_names()`.

### Lazy Tick Advancement for `Changed<T>`

- **`world.query()` now defers read-tick updates** until the `QueryIter` is actually iterated (via `next()`, `for_each_chunk()`, or `par_for_each()`). Dropping an iterator without consuming it preserves the `Changed<T>` window — subsequent queries still see those changes. This matches `QueryWriter`'s existing `queried` flag pattern.
- **`World::has_changed::<Q>()`** — peek at whether any archetype has changes for a query type without consuming the change window.
- **`World::advance_query_tick::<Q>()`** — explicitly consume the change window without iterating (e.g., during loading phases).
- **`World::query_tick_info::<Q>()`** — returns `QueryTickInfo` with debug visibility into tick state (last read tick, current world tick, pending status, matched archetype count).

### Sparse Data Durability (`minkowski` + `minkowski-persist`)

- **Sparse component mutations now survive WAL crash recovery.** `SparseInsert` and `SparseRemove` variants added to `Mutation`/`MutationRef` enums in `EnumChangeSet`, with full reverse changeset support for undo/redo.
- New typed helpers on `EnumChangeSet`: `insert_sparse<T>()`, `remove_sparse<T>()`, plus raw pre-resolved variants `insert_sparse_raw()`, `remove_sparse_raw()`.
- New transaction methods on `Tx`: `write_sparse<T>()`, `remove_sparse<T>()`, and pre-resolved variants.
- Raw type-erased methods on `SparseStorage`: `insert_raw()`, `get_raw()`, `remove_raw()` for changeset apply and WAL replay.
- `ComponentRegistry::mark_sparse()` ensures sparse routing flag is set after WAL replay.
- WAL `SerializedMutation` extended with `SparseInsert`/`SparseRemove` variants for rkyv serialization and cross-process replay.
- Entity liveness assertions on `SparseInsert`/`SparseRemove` apply paths, matching the existing dense insert safety checks.

### Bug Fixes

- Fixed `World::insert_sparse` calling `register()` instead of `register_sparse()`, which prevented `world.get()` from routing to sparse storage for components inserted via that path.

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
