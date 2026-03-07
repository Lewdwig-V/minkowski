---
description: >
  Comprehensive guide for the Minkowski ECS engine. Triggers on: Minkowski,
  ECS, World, Entity, Component, query, reducer, transaction, persistence,
  spatial index, data modeling, concurrency model, performance optimization,
  archetype, spawn, despawn, CommandBuffer, EnumChangeSet, Changed, Table,
  SpatialIndex, Access, ReducerRegistry, Optimistic, Pessimistic, Durable,
  QueryWriter, EntityMut, QueryMut, QueryRef, Spawner, DynamicCtx.
---

# Minkowski ECS Guide

> For architecture internals (storage model, data flow, archetype migration, key traits),
> see `CLAUDE.md`. This guide covers **what to use when** and **what to avoid**.

## Quick Reference

**World** is the sole mutable entry point. Core methods: `spawn`, `despawn`, `insert`,
`remove`, `get`, `get_mut`, `query`. Entity = u64 (low 32 = index, high 32 = generation).
Components are any `'static + Send + Sync` type. Direct World methods are for setup and
debugging; production mutations go through reducers.

**Queries** use tuple composition: `&T` read, `&mut T` write, `Entity` ID,
`Option<&T>` optional access (doesn't filter archetypes), `Changed<T>` filter (since
last time *this query* observed the column, not per-frame). `for_each_chunk` yields typed
slices for SIMD auto-vectorization. `par_for_each` uses rayon for parallel iteration.
Query cache is automatic — repeat queries skip archetype scans.

**Table** via `#[derive(Table)]` declares a fixed schema. `query_table`/`query_table_mut`
skip archetype matching entirely. Generated `Ref<'w>`/`Mut<'w>` types provide named field
access. Use when a set of components is always queried together.

**Transactions** come in three strategies: `Sequential` (zero overhead, single-threaded),
`Optimistic` (tick-based validation, cheap reads, retries on conflict), `Pessimistic`
(cooperative column locks, guaranteed commit, spin+yield backoff). `Tx` does NOT hold
`&mut World` — split-phase design enables parallel reads. `tx.query(&world)` requires
`ReadOnlyWorldQuery`.

**Reducers** type-erase closures with Access metadata in a `ReducerRegistry`.
Handle types: `EntityMut<C>` (single-entity mutation), `QueryMut<Q>` (bulk direct),
`QueryRef<Q>` (bulk read-only), `QueryWriter<Q>` (bulk buffered, durable-compatible),
`Spawner<B>` (entity creation), `DynamicCtx` (runtime-conditional access).
Dispatch: `call()` for transactional (entity, spawner, query writer), `run()` for
scheduled (query), `dynamic_call()` for dynamic.

**Persistence** via `minkowski-persist`: `Durable<S>` wraps any strategy with WAL
logging. `Snapshot` for point-in-time saves with rkyv serialization. `load_zero_copy()`
for mmap-based restore. `CodecRegistry` for component serialization (rkyv derives).
`#[repr(C)]` recommended on persistent components for zero-copy benefit.
Always call `sync_reserved()` after snapshot restore.

**Spatial Indexing** via `SpatialIndex` trait: `rebuild` (required) + `update` (optional,
defaults to rebuild). External to World — compose from queries and `is_alive()`. Grid for
uniform density, tree for clustered.

## Decision Flowcharts

### Data Modeling

- Fixed set of components always queried together? -> `#[derive(Table)]`
- Otherwise -> individual component structs, one concept per component
- Rare optional data (< 5% of entities)? -> `register_sparse`
- Common data present on most entities? -> archetype component (default)
- Component needs no heap allocation? -> derive `Copy` (enables zero-copy query)

### Query Patterns

- Need entity ID alongside data? -> include `Entity` in the query tuple
- Component may or may not be present? -> `Option<&T>` (access without filtering)
- Only care about recently changed data? -> `Changed<T>` filter
  - Note: first call always matches everything (no prior observation tick)
- Numeric tight loop (physics, particles)? -> `for_each_chunk` for SIMD slices
- CPU-heavy per-entity work? -> `par_for_each` with rayon
- Need raw typed slices? -> `for_each_chunk` yields `&[T]`/`&mut [T]` slices per archetype

### Mutation Strategy

- Setup / debugging / one-off? -> direct `world.spawn()`, `world.insert()`
- Inside a query loop needing structural changes (spawn/despawn)? -> `CommandBuffer`
- Need reversible mutations (undo/redo)? -> `EnumChangeSet` (apply returns reverse)
- Inside a transaction? -> reducer handles buffer automatically
- Need WAL-logged mutations? -> `QueryWriter` (buffers writes, compatible with Durable)

### Concurrency Model

- Single-threaded, no conflict detection needed? -> `Sequential`
- Multiple readers, occasional writes? -> `Optimistic` (tick validation, default 3 retries)
- Write-heavy, retries are expensive? -> `Pessimistic` (column locks, backoff)
- Need crash safety? -> `Durable<S>` wrapping any strategy
- Ask yourself: what is the read/write ratio? How expensive is a retry vs a lock?

### Reducer Type Selection

- Single entity read + write? -> `EntityMut<C>` via `register_entity`
- Single entity with despawn/remove? -> `register_entity_despawn`
- Bulk read-only iteration? -> `QueryRef<Q>` via `register_query_ref`
- Bulk read-write, direct mutation? -> `QueryMut<Q>` via `register_query`
  - Scheduled execution model, compile-time safety, requires `&mut World`
- Bulk read-write, buffered? -> `QueryWriter<Q>` via `register_query_writer`
  - Transactional execution, compatible with Durable/WAL
  - Use `WritableRef::modify()` for in-place updates
- Entity creation? -> `Spawner<B>` via `register_spawner`
- Access depends on runtime state? -> `DynamicCtx` via `registry.dynamic()`
  - Declare upper bounds: `can_read`, `can_write`, `can_spawn`, `can_remove`, `can_despawn`
  - Undeclared access panics in all builds (assert!, not debug_assert!)

### Persistence

- Need crash recovery? -> WAL via `Durable<S>`
- Need fast restore? -> periodic `Snapshot` saves (not every frame)
- Which components survive restart? -> register codecs for those types
- After snapshot restore -> `restore_allocator_state()` calls `sync_reserved()` internally; custom restoration paths must call it manually
- Mutation path -> `QueryWriter` reducer (buffers writes, changeset goes to WAL)

### Column Indexes

- Need all entities with component value in a range? -> `BTreeIndex<T>` (T: Ord)
- Need all entities with exact component value? -> `HashIndex<T>` (T: Hash + Eq)
- After index lookup, need component data for results? -> `world.get_batch::<T>(&entities)`
- Need mutable access to results? -> `world.get_batch_mut::<T>(&entities)` (panics on duplicates)
- Multiple component types? -> call `get_batch` once per type
- Incremental updates? -> each index owns a `ChangeTick`, uses `world.query_changed_since()`
- Stale entries from despawn/remove? -> `get_valid()`/`range_valid()` filter via `world.has::<T>()`

### Spatial Indexing

- Do you need spatial neighbor queries at all? -> only if yes, implement `SpatialIndex`
- Uniform entity density? -> grid (cell size ~ interaction radius)
- Clustered / variable density? -> quadtree, BVH, or k-d tree
- Entities move frequently? -> rebuild each frame, or override `update` for incremental
- Query results may include despawned entities -> always check `world.is_alive(entity)`

### Performance

- Numeric component for SIMD? -> `#[repr(align(16))]` or `[f32; 4]`
- Tight iteration loop? -> `for_each_chunk` (yields typed slices, LLVM vectorizes)
- CPU-bound per-entity? -> `par_for_each` (rayon parallel)
- Frequent `insert()`/`remove()` in loops? -> redesign to avoid archetype migration
- Many archetypes? -> query cache handles it, but consider consolidating

## Strong Defaults

Follow these unless you have a specific reason not to.

### Data Modeling
- One concept = one component. Do not nest domain structs as component fields.
- Derive `Copy` on components when possible (no heap allocation = zero-copy query).
- Use `#[derive(Table)]` when you have a fixed schema queried together (e.g., physics).
- Use `register_sparse` for rare optional data (e.g., DebugLabel, AIOverride).
- Bundle tuples for spawn: `world.spawn((Pos::new(0.0, 0.0), Vel::zero(), Health(100)))`.
- Keep components small. Large components bloat archetype columns and hurt cache locality.
- Marker components (zero-sized types) are free in storage — use them for tagging.

### Queries
- Start with tuple queries. Graduate to `Table` when the schema is stable and always
  queried as a unit.
- Use `for_each_chunk` for numeric tight loops — it yields contiguous typed slices that
  LLVM can auto-vectorize.
- Include `Entity` in the query tuple when you need to correlate results with entity IDs.
- Prefer `for_each` over collecting into a `Vec` — iteration is zero-allocation.
- When reading results of a `Changed<T>` query for the first time, expect all entities
  to match (there is no prior tick to compare against).
- Use `query_table`/`query_table_mut` for Table-derived schemas — they skip archetype
  matching entirely and provide named field access.

### Mutations
- Every production mutation should go through a reducer. Direct World methods for setup only.
- Register all systems in a `ReducerRegistry`, even single-threaded — it gives you free
  conflict detection via `Access` metadata.
- Use `CommandBuffer` for structural changes during query iteration.
- Use `EnumChangeSet` when you need reversible or serializable mutations.
- Prefer flag components over insert/remove cycling — archetype migration is expensive.
- When using `EnumChangeSet`, remember that `apply()` returns the reverse changeset for undo.

### Concurrency
- Start with `Sequential`. Add `Optimistic` when you actually need concurrency.
- Use `QueryWriter` over `QueryMut` when the reducer needs to be durable (WAL-compatible).
- For parallel dispatch with rayon, each thread calls `registry.call()` — the strategy
  handles retry internally.
- Construct `Optimistic`/`Pessimistic` with `::new(&world)` to capture the shared orphan
  queue handle. Do not construct multiple instances for the same World.
- Use `registry.reducer_access()` / `registry.query_reducer_access()` to extract Access
  metadata for scheduler conflict analysis.

### Persistence
- WAL for crash safety, snapshots for fast restore. Register codecs for every persisted
  component type.
- Always `sync_reserved()` after snapshot restore.
- Use `QueryWriter` reducers with `Durable` — they produce the changeset that gets logged
  to the WAL. `QueryMut` mutations bypass the changeset and cannot be logged.
- Snapshot periodically (not every frame) to bound WAL replay time on recovery.

## Pitfall Alerts

### `world.insert()` in a hot loop
Archetype migration is O(components) per entity — every component column gets copied.
Batch structural changes at setup time. For runtime state changes, prefer flag components
(e.g., `Stunned` marker) over adding/removing components per frame.

### `Changed<T>` semantics
`Changed<T>` means "since the last time *this specific query* observed this column."
It is NOT per-frame. The first call to a query with `Changed<T>` always matches everything
because there is no prior observation tick. Ticks are `pub(crate)` — not exposed to users.

### `query_raw()` vs `query()`
`query_raw(&self)` is the shared-ref read path used by transactions. It skips the query
cache and tick management. Do not use it for normal iteration — use `query(&mut self)`
which maintains the cache and supports `Changed<T>`.

### Missing `is_alive()` on spatial index results
Despawned entities leave stale entries in spatial indexes. The index cleans up on next
`rebuild`, but between rebuilds, query results may include dead entities. Always validate
with `world.is_alive(entity)` before accessing components.

### `&mut T` through `tx.query(&world)`
`tx.query(&world)` takes `&World` (shared ref) and requires `ReadOnlyWorldQuery`.
You cannot get `&mut T` through this path — it would be unsound (two transactions could
alias). Use `tx.write(entity, component)` to buffer mutations, or use `QueryWriter`
for bulk buffered writes.

### `QueryMut` for durable reducers
`QueryMut` mutates columns directly via `&mut World`. These mutations bypass the
`EnumChangeSet` and cannot be logged to a WAL. If your reducer needs durability, use
`QueryWriter` instead — it buffers writes through `WritableRef`, producing a changeset
that `Durable` can log before applying.

### Dynamic reducer undeclared access
Accessing a component not declared in the builder (`can_read`, `can_write`, etc.) panics
in ALL builds. This is `assert!`, not `debug_assert!`, because it protects the scheduler's
Access bitset invariant. Declare everything the reducer might touch, even conditionally.

### Forgetting `sync_reserved()` in custom restore paths
The standard `restore_allocator_state()` calls `sync_reserved()` automatically. But if you
add a new state restoration path that bypasses it, the atomic `next_reserved` counter stays
at 0 and `reserve()` hands out indices overlapping with restored entities. Any custom
restoration path must call `sync_reserved()` — the standard path handles it for you.

### Lock table per strategy instance
Each `Pessimistic` strategy has its own `ColumnLockTable`. Two separate `Pessimistic`
instances do not see each other's locks. If you need cooperative locking across contexts,
share a single strategy instance (it uses `Arc` internally).

### Entity ID lifecycle in transactions
Entity IDs from `tx.spawn()` / `Spawner::spawn()` are tracked automatically. On commit
they become placed entities. On abort (drop without commit, or conflict), the IDs go to
the shared `OrphanQueue` and World recycles them automatically. No manual cleanup needed,
but be aware that entity IDs are not valid until after commit.

### `Option<&T>` vs required components
`Option<&T>` accesses a component without requiring it for archetype matching. This means
the query matches archetypes that lack `T` — the option will be `None`. This is correct
for optional data, but `accessed_ids` includes `T` for conflict detection even though
`required_ids` does not. If you want to filter to only entities with `T`, use `&T` instead.

### Cross-world strategy usage
Each strategy captures a `WorldId` at construction. Using a strategy with the wrong World
panics — the `Arc<OrphanQueue>` connects to the wrong allocator, which would corrupt entity
lifecycles. Always construct strategies with the World they will be used with.

### Change detection through BlobVec::get_ptr
`BlobVec::get_ptr` is the read path. Writing through a pointer obtained from `get_ptr`
silently bypasses `Changed<T>` detection. All mutable access must go through
`get_ptr_mut(row, tick)` or a World method that marks the column. If you add a new
mutable accessor in custom code, ensure it uses the correct path.

### Archetype explosion from unique component combinations
Each unique set of component types creates a new archetype. Adding/removing components
dynamically (e.g., per-entity optional components via `insert`) can create many archetypes.
This doesn't break correctness, but increases query cache maintenance cost. Prefer marker
components spawned with the entity over dynamic insert/remove patterns.

### Duplicate components in a bundle
`Bundle::component_ids()` sorts and deduplicates. If you pass `(Pos, Pos)` in a spawn,
only one `Pos` column is created. The second value silently overwrites the first. This is
not an error, but it is almost certainly a bug in your code.

## Examples

Each example demonstrates specific patterns. Read the source for concrete API usage.

| Example | Key patterns | File |
|---|---|---|
| `reducer` | All handle types (EntityMut, QueryMut, QueryRef, Spawner, QueryWriter, DynamicCtx), structural mutations, conflict detection | `examples/examples/reducer.rs` |
| `boids` | Query reducers, SpatialGrid, SIMD via `for_each_chunk`, deferred commands | `examples/examples/boids.rs` |
| `life` | QueryMut, `#[derive(Table)]`, EnumChangeSet undo/redo via changeset reversal | `examples/examples/life.rs` |
| `nbody` | Query reducers, Barnes-Hut quadtree, rayon parallel force computation | `examples/examples/nbody.rs` |
| `scheduler` | ReducerRegistry for Access metadata, conflict detection, greedy batch scheduling | `examples/examples/scheduler.rs` |
| `transaction` | Three strategies (Sequential/Optimistic/Pessimistic), raw Tx + reducer comparison | `examples/examples/transaction.rs` |
| `battle` | EntityMut reducers, rayon parallel snapshot computation, sequential dispatch, tunable conflict rates | `examples/examples/battle.rs` |
| `persist` | QueryWriter + Durable, WAL, rkyv snapshots, zero-copy load, crash recovery | `examples/examples/persist.rs` |
| `index` | BTreeIndex range queries, HashIndex exact lookups, incremental update, batch fetch | `examples/examples/index.rs` |
| `flatworm` | SpatialIndex (FoodGrid), chemotaxis, CommandBuffer spawn/despawn, entity lifecycle (fission + starvation), query reducers | `examples/examples/flatworm.rs` |
| `circuit` | Entity-based circuit topology (node connectivity), query reducers (QueryMut, QueryRef), symplectic Euler integration, ReducerRegistry scheduling | `examples/examples/circuit.rs` |
| `tactical` | Sparse components (insert_sparse, iter_sparse), par_for_each, Optimistic Conflict inspection, Entity::to_bits/from_bits, world introspection, register_entity_despawn, HashIndex get_valid(), EnumChangeSet/MutationRef iteration, multi-threaded replication | `examples/examples/tactical.rs` |

### Pattern Quick-Find

- **Entity reducer (register + dispatch):** `reducer.rs` lines 37-44 (`register_entity`)
- **Query reducer (bulk mutable):** `reducer.rs` lines 59-67 (`register_query`)
- **Read-only query reducer:** `reducer.rs` lines 71-78 (`register_query_ref`)
- **Spawner reducer:** `reducer.rs` lines 82-89 (`register_spawner`)
- **QueryWriter with WritableRef:** `reducer.rs` lines 141-149 (`register_query_writer`)
- **Dynamic reducer with builder:** `reducer.rs` lines 185-215 (`registry.dynamic()`)
- **Dynamic for_each + despawn:** `reducer.rs` lines 253-270 (reaper pattern)
- **Name-based reducer lookup:** `reducer.rs` lines 166-171 (`reducer_id_by_name`)
- **Conflict detection:** `reducer.rs` lines 282-355 (`conflicts_with`)
- **SpatialIndex trait impl:** `boids.rs` (SpatialGrid), `nbody.rs` (BarnesHutTree)
- **Table derive + query_table / query_table_mut:** `life.rs` (named field access via CellRef/CellMut)
- **Durable transactions:** `persist.rs`
- **Parallel dispatch with rayon:** `battle.rs`
- **BTreeIndex range + HashIndex exact:** `index.rs` lines 40-63
- **Incremental index update (ChangeTick):** `index.rs` lines 71-89
- **Stale index validation (range_valid, get_valid):** `index.rs` lines 104-119
- **Index -> get_batch / get_batch_mut composition:** `index.rs` lines 128-161
- **SpatialIndex + CommandBuffer lifecycle:** `flatworm.rs` (FoodGrid, fission spawn, starvation despawn)
- **Entity-based circuit connectivity:** `circuit.rs` (nodes as entities, elements reference node Entity handles)
- **Sparse component lifecycle:** `tactical.rs` (insert_sparse for MoveOrder/IntelReport, iter_sparse for intel queries)
- **Entity bit packing for serialization:** `tactical.rs` (to_bits in commands, from_bits in replication)
- **EnumChangeSet as replication journal:** `tactical.rs` (iter_mutations + MutationRef pattern matching)
- **Optimistic Conflict inspection:** `tactical.rs` (catch Err(Conflict), display_with)
- **World introspection (archetype/component metadata):** `tactical.rs`
- **HashIndex stale filtering:** `tactical.rs` (get_valid after despawns)
- **par_for_each parallel iteration:** `tactical.rs` (position clamping pass)
- **register_entity_despawn:** `tactical.rs` (combat cleanup reducer)

## Common Patterns

### Typical simulation frame loop
```
1. Rebuild spatial indexes (if any)
2. Run read-only query reducers (sensors, logging)
3. Run mutation query reducers (physics, AI, rules)
4. Apply command buffers (structural changes from step 2-3)
5. Periodic: snapshot save, stats collection
```

### Registering and dispatching a query reducer
```rust
let id = registry.register_query::<(&mut Vel,), f32, _>(
    &mut world, "gravity",
    |mut query: QueryMut<'_, (&mut Vel,)>, dt: f32| {
        query.for_each(|(vel,)| { vel.0 -= 9.81 * dt; });
    },
);
registry.run(&mut world, id, 0.016f32);
```

### Entity reducer with transactional dispatch
```rust
let id = registry.register_entity::<(Health,), u32, _>(
    &mut world, "heal",
    |mut entity: EntityMut<'_, (Health,)>, amount: u32| {
        let hp = entity.get::<Health, 0>().0;
        entity.set::<Health, 0>(Health(hp + amount));
    },
);
let strategy = Optimistic::new(&world);
registry.call(&strategy, &mut world, id, (target_entity, 25u32)).unwrap();
```

### Dynamic reducer with conditional access
```rust
let id = registry.dynamic("shield", &mut world)
    .can_read::<Health>()
    .can_write::<Shield>()
    .build(|ctx: &mut DynamicCtx, entity: &Entity| {
        let hp = ctx.read::<Health>(*entity).0;
        if hp < 50 {
            ctx.write(*entity, Shield(100.0));
        }
    });
registry.dynamic_call(&strategy, &mut world, id, &entity).unwrap();
```

### Conflict detection for scheduling
```rust
let access_a = registry.query_reducer_access(system_a);
let access_b = registry.query_reducer_access(system_b);
if !access_a.conflicts_with(access_b) {
    // Safe to run in parallel
}
```

## References

- **Architecture details:** `CLAUDE.md` — storage model, data flow, archetype migration,
  trait specifications, transaction semantics, reducer system internals
- **API surface:** `crates/minkowski/src/lib.rs` — public re-exports
- **Proc macro:** `crates/minkowski-derive/` — `#[derive(Table)]` implementation
- **Persistence crate:** `crates/minkowski-persist/` — WAL, snapshots, Durable wrapper
- **Benchmarks:** `cargo bench -p minkowski` — criterion benchmarks for spawn, query, iteration
