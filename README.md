# Minkowski

A column-oriented database engine built from scratch by one human and one AI.

Minkowski is a storage engine for real-time interactive applications — games, simulations, collaborative tools — that need both the iteration speed of an [ECS][ecs] and the transactional guarantees of a database.

Most ECS engines give you fast iteration but no persistence, no rollback, no concurrency control. Most databases give you transactions but can't iterate 100,000 components per frame. Minkowski gives you both, and you only pay for the features you use.

## Quick start

> 💡 **Working with Claude Code?** custom commands provide on-demand ECS expertise in every session. Use `/design-doc` to plan a new feature, `/soundness-audit` to review concurrency invariants, or `/validate-api` and `/validate-macro` for compile-time correctness checks. Nine domain commands — `/minkowski:model`, `/minkowski:query`, `/minkowski:mutate`, `/minkowski:concurrency`, `/minkowski:reducer`, `/minkowski:index`, `/minkowski:persist`, `/minkowski:optimize`, `/minkowski:python` — explain the paradigm in depth. See [AI-Assisted Development](#ai-assisted-development) for the full list.

```rust
use minkowski::{World, ReducerRegistry, QueryMut};

#[derive(Clone, Copy)]
struct Pos { x: f32, y: f32 }
#[derive(Clone, Copy)]
struct Vel { dx: f32, dy: f32 }

let mut world = World::new();
let mut registry = ReducerRegistry::new();

// Spawn 1000 entities
for i in 0..1000 {
    world.spawn((Pos { x: i as f32, y: 0.0 }, Vel { dx: 1.0, dy: 0.5 }));
}

// Register a query reducer — the type signature declares what it reads and writes
let move_id = registry.register_query::<(&mut Pos, &Vel), (), _>(
    &mut world,
    "movement",
    |mut query: QueryMut<'_, (&mut Pos, &Vel)>, ()| {
        query.for_each(|(pos, vel)| {
            pos.x += vel.dx;
            pos.y += vel.dy;
        });
    },
);

// Dispatch — Access bitset extracted from the type signature at registration
registry.run(&mut world, move_id, ());
```

### Building & testing

```
cargo test -p minkowski                # 644+ tests
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p minkowski               # criterion benchmarks vs hecs
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib   # UB check
```

CI runs fmt, clippy, test, and Miri sequentially on every PR. A `ci-pass` aggregator job is the single required status check for branch protection.

## Design Principles

- **Three mutation tiers** — direct `world.get_mut()` (zero overhead), transactional (buffered changeset with conflict detection), durable (WAL-backed via `Durable<S>` wrapper). Each tier adds cost only for the guarantees it provides.
- **Typed reducers** — closure signatures declare their access pattern. The engine extracts read/write bitsets at registration time. Two reducers with disjoint access are proven non-conflicting at compile time — no runtime locks, no optimistic retry.
- **Split-phase transactions** — `Tx` does not hold `&mut World`. Transactions read concurrently via `&World`, buffer writes privately, and commit sequentially. The borrow checker enforces the phase boundary.
- **Composable persistence** — `Durable<S>` wraps any transaction strategy to add WAL logging. Zero-copy snapshot loading via mmap + rkyv. The core engine has no serialization dependency.
- **Mechanisms, not policy** — no built-in scheduler, no application lifecycle. Minkowski provides `Access` bitsets and `is_compatible()` for framework authors to build their own scheduling. Secondary indexes are external consumers of the change detection system.
- **Results, not panics** — user-triggerable error conditions (dead entities, pool exhaustion, invalid planner inputs, duplicate codec registrations) return `Result` so the caller decides whether to panic, retry, or recover. Internal invariants that users cannot violate through public APIs remain as panics.

## Deep Dive

For a comprehensive walkthrough of the internals — storage model, query engine, transaction system, persistence layer, memory management — see the [DeepWiki](https://deepwiki.com/Lewdwig-V/minkowski). It's auto-generated from the source and always up to date.

## Soundness

Minkowski uses `unsafe` for type-erased column storage and raw pointer iteration — the performance-critical paths that make an ECS fast. Six layers of verification ensure these paths are correct:

| Layer | What it catches | When it runs |
|---|---|---|
| **Type system + borrow checker** | Aliased `&mut T`, lifetime violations, `Send`/`Sync` misuse. `ReadOnlyWorldQuery` prevents `&mut T` through `&World`. | Every build |
| **644 unit tests** | Semantic bugs: entity lifecycle, archetype migration, change detection, transaction abort cleanup, reducer access boundaries | Every PR (CI) |
| **[Miri][miri] + [Tree Borrows][tree-borrows]** | Undefined behavior: use-after-free, uninitialized reads, aliasing violations in `unsafe` blocks. Full test suite passes under the strict Tree Borrows model. | Every PR (CI) |
| **[ThreadSanitizer][tsan]** | Data races: unsynchronized concurrent memory accesses. Full test suite including rayon `par_for_each` passes under TSan instrumentation. | Every PR (CI) |
| **[Loom][loom]** | Concurrency invariant violations: exhaustive thread interleaving enumeration over OrphanQueue push/drain, column lock acquire/upgrade/deadlock-freedom, and entity ID reservation contention. | Every PR (CI) |
| **[Fuzz testing][cargo-fuzz]** | Edge cases that structured tests miss: random operation sequences against World, query iteration across varied archetype shapes, malformed snapshot/WAL input. Coverage-guided mutation explores millions of paths. | Manual, pre-release |

## Storage

Any Rust type that is `'static + Send + Sync` is a component. Entities with the same set of component types share an [archetype][archetype] — components are stored in contiguous, cache-friendly [column arrays][soa]. Adding or removing a component moves the entity to the matching archetype automatically.

Entity IDs are [generational][generational-index] — recycled IDs get bumped generations, so stale handles are always detected. Sparse components are opt-in for tags and rarely-queried data, preventing archetype fragmentation.

```rust
let mut world = World::new();
let e = world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
world.insert(e, Health(100)).unwrap();   // migrates entity to new archetype (Err on dead entity)
world.remove::<Health>(e);      // migrates it back
```

## Queries

Queries are tuple-typed — `world.query::<(&mut Pos, &Vel)>()` iterates all entities with both components, giving mutable access to `Pos` and shared access to `Vel`. Results are cached per query type, so repeat calls are nearly free.

`Changed<T>` skips entities whose component hasn't been modified since your last read — useful for incremental processing. `par_for_each` distributes iteration across threads via [rayon][rayon], and `for_each_chunk` yields typed slices for [SIMD][simd] auto-vectorization. Table queries (`world.query_table::<Transform>()`) skip archetype matching entirely when you know the exact schema.

```rust
// Dynamic query -- cached, skips archetype scan on repeat calls
for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
    pos.x += vel.dx;
}

// Change detection -- skip archetypes untouched since last read
for (pos, _) in world.query::<(&mut Pos, Changed<Vel>)>() {
    // only entities whose Vel column was mutably accessed
}

// Chunk iteration -- yields typed slices for auto-vectorization
world.query::<(&mut Pos, &Vel)>().for_each_chunk(|(positions, velocities)| {
    for i in 0..positions.len() {
        positions[i].x += velocities[i].dx;
    }
});
```

### Query Planner

`QueryPlanner` compiles queries into cost-optimized execution plans using a [Volcano][volcano]-model optimizer with automatic index selection, then executes them against the live world. Plans are vectorized by default — each node processes data in morsel-sized batches mapped to `for_each_chunk`, enabling SIMD auto-vectorization. Three execution modes: `execute(&mut World) -> &[Entity]` (collects into plan-owned scratch buffer, supports joins), `for_each(&mut World, callback)` (zero-allocation scan iteration), and `for_each_raw(&World, callback)` (transactional read-only path).

```rust
use minkowski::{QueryPlanner, Predicate, JoinKind};

let mut planner = QueryPlanner::new(&world);
planner.add_btree_index::<Score>(&score_index, &world).unwrap();
planner.add_hash_index::<Team>(&team_index, &world).unwrap();

let mut plan = planner
    .scan::<(&Pos, &Score)>()
    .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
    .build();

// Inspect the plan
println!("{}", plan.explain());  // shows vectorized execution plan

// Execute: collect matching entities into plan-owned scratch buffer
let entities = plan.execute(&mut world);  // returns &[Entity]
for e in entities {
    let score = world.get::<Score>(*e).unwrap();
    // only entities with Score in [10..50) are returned
}

// Zero-alloc iteration for scan-only plans
plan.for_each(&mut world, |entity| {
    // process each matching entity directly — no intermediate buffer
});
```

`TablePlanner<T>` adds compile-time enforcement: if a table field is annotated with `#[index(btree)]` or `#[index(hash)]`, the planner requires the corresponding index at the type level. Missing indexes are type errors, not runtime warnings. See [Schema & Mutation](#schema--mutation) for the annotation syntax.

**Subscription queries** guarantee at compile time that every predicate is index-backed via `Indexed<T>` witnesses. Combined with `Changed<T>`, subscriptions skip archetypes whose indexed column has not been written since the last call — no delta tracking, caching, or event sourcing needed. (`Changed<T>` is archetype-granular: mutating one entity marks the entire column, so unchanged siblings in the same archetype may also pass.)

```rust
use minkowski::{Changed, HashDebounce, Indexed, Predicate, SubscriptionDebounce};

let witness = Indexed::btree(&score_index);
let mut sub = planner
    .subscribe::<(Changed<Score>, &Score)>()
    .where_eq(witness, Predicate::eq(Score(42)))
    .build()?;

// HashDebounce filters false positives from archetype-granular Changed<T>.
let mut debounce = HashDebounce::<Score>::new();

sub.for_each(&mut world, |entity| {
    let score = world.get::<Score>(entity).unwrap();
    if debounce.is_changed(entity, score) {
        // genuinely changed — react
    }
})?;
```

`HashDebounce<T>` is the default in-memory debounce filter. Implement `SubscriptionDebounce<T>` on your own type for external-backed deduplication.

### Materialized Views

`MaterializedView` wraps a `QueryPlanResult` and caches the matching entity list. On each `refresh()` call it re-executes the plan, but only if the debounce threshold has been met. Two layers of filtering: the plan's `Changed<T>` filter skips unchanged archetypes, and the configurable `DebouncePolicy` limits how often the plan runs at all.

```rust
use minkowski::{MaterializedView, DebouncePolicy};

let mut view = MaterializedView::new(plan)
    .with_debounce(DebouncePolicy::EveryNTicks(NonZeroU64::new(10).unwrap()));

// Per-frame: refresh and read
view.refresh(&mut world).unwrap();
for &entity in view.entities() {
    // cached, debounced result — stale by at most N frames
}
```

## Typed Reducers

Reducers are closures registered with the `ReducerRegistry`. The type signature declares exactly what the closure can access, and the registry extracts `Access` metadata at registration time for conflict detection.

| Handle | Use when you need to... | Execution model |
|---|---|---|
| `EntityRef<C>` | Inspect one entity's components without changing anything | Transactional |
| `EntityMut<C>` | Update or remove components on a single known entity | Transactional |
| `Spawner<B>` | Create new entities during a transaction | Transactional |
| `QueryWriter<Q>` | Iterate many entities and buffer writes for atomic commit | Transactional |
| `QueryRef<Q>` | Iterate many entities read-only (e.g. logging, census) | Scheduled |
| `QueryMut<Q>` | Iterate and mutate many entities directly | Scheduled |
| `DynamicCtx` | Decide which components to access at runtime | Dynamic |

Scheduled reducers (`QueryRef`, `QueryMut`) can run in parallel when their access patterns don't overlap — the registry can prove this at registration time. Transactional reducers (`EntityMut`, `Spawner`, `QueryWriter`) buffer writes and commit atomically through a [transaction strategy](#transactions). Dynamic reducers (`DynamicCtx`) let you choose which components to access at runtime when the set isn't known at compile time. See [`docs/reducer-correctness.md`](docs/reducer-correctness.md) for behavioral requirements (determinism, termination, no unwinding).

## Transactions

Three strategies over a unified `Transact` trait, so you can swap concurrency policies without changing your reducer logic:

- **Sequential** — zero overhead, direct world access. Use when single-threaded.
- **[Optimistic][occ]** — concurrent reads, buffered writes, validates at commit. Use when conflicts are rare.
- **[Pessimistic][pcc]** — acquires locks up front, commit always succeeds. Use when conflicts are frequent.

All strategies handle entity ID cleanup automatically — no IDs leak on abort, no manual drain step required.

## Persistence

The `minkowski-persist` crate adds crash-safe durability. `Durable<S>` wraps any transaction strategy — every committed mutation is [WAL][wal]-logged before the caller sees the result. Recovery loads the latest snapshot and replays subsequent WAL entries.

Snapshots serialize the full world state via [rkyv][rkyv] and can be saved to disk or transferred as bytes (`save_to_bytes` / `load_from_bytes`). `ReplicationBatch` is the transport-agnostic wire format for incremental replication — serialize it, send it over any medium (network, channels, shared memory), and `apply_batch` on the receiving end. `WalCursor` reads batches from local WAL files for same-server scenarios.

```rust
// Durable wraps any strategy — Optimistic, Pessimistic, or Sequential
let durable = Durable::new(strategy, wal, codecs);  // S = Optimistic here
durable.transact(&mut world, access, |tx, world| { /* ... */ });
// Changeset written to WAL on successful commit
```

**Persistence vs. the memory pool**: `Durable<S>` and `WorldBuilder`'s mmap pool solve different problems. The pool pre-allocates volatile RAM — anonymous `MAP_ANONYMOUS` pages that vanish when the process exits. It controls memory *layout and budget*, not durability. `Durable<S>` writes committed mutations to a WAL on *disk* before they're visible, giving crash safety. They compose naturally: a pooled World with `Durable<Optimistic>` gives crash-safe transactions within a fixed memory envelope. Use the pool alone for bounded in-memory workloads that don't need crash recovery. Use `Durable` alone for crash safety with the system allocator. Use both when you want both guarantees.

## Schema & Mutation

`#[derive(Table)]` declares a named schema with typed row accessors — queries against a table skip archetype matching entirely. Fields can be annotated with `#[index(btree)]` or `#[index(hash)]` to declare compile-time index requirements:

```rust
#[derive(Table)]
struct Scores {
    #[index(btree)]
    score: Score,       // generates HasBTreeIndex<Score> for Scores
    #[index(hash)]
    team: Team,         // generates HasHashIndex<Team> for Scores
    name: Name,         // no index — querying by name is a type error in TablePlanner
}

// Table query — skips archetype matching
for row in world.query_table::<Scores>() {
    println!("{:?}, {:?}", row.score, row.team);
}

// Compile-time index enforcement via TablePlanner
let idx = Scores::create_btree_index(&mut world);
let mut planner = TablePlanner::<Scores>::new(&world);
planner.add_btree_index::<Score>(&idx);  // compiles: Score has #[index(btree)]
// planner.add_btree_index::<Name>(&idx); // type error: Name has no #[index(btree)]
```

`EnumChangeSet` records mutations as data. `apply()` returns a reverse changeset — applying the reverse undoes the original changes, giving you automatic undo/redo. `CommandBuffer` provides deferred structural changes (spawn/despawn/insert/remove) during query iteration.

## Observability

The `minkowski-observe` crate captures engine metrics without instrumenting hot paths. `MetricsSnapshot::capture()` takes a point-in-time reading; `MetricsDiff::compute()` compares two snapshots to show what changed.

```rust
use minkowski_observe::{MetricsSnapshot, MetricsDiff};

let snap1 = MetricsSnapshot::capture(&world, Some(&wal));
// ... run simulation ...
let snap2 = MetricsSnapshot::capture(&world, Some(&wal));

let diff = MetricsDiff::compute(&snap1, &snap2);
println!("{diff}");
// --- Diff (1.2ms) ---
//   entity delta: +30  churn: 70
//   tick delta: 51  WAL seq delta: 50
//   archetype delta: +1
```

Entity churn tracking is exact — every spawn and despawn is counted. Per-archetype detail (entity count, component names, estimated byte footprint) is included in every snapshot.

`PrometheusExporter` converts snapshots into OpenMetrics text format — 15 gauges covering world state, memory pool, WAL pressure, and per-archetype breakdowns. No HTTP server included; call `exporter.render()` and mount the result on your own `/metrics` endpoint. Compatible with Grafana, Datadog, or any Prometheus-compatible tool.

## Indexing

Indexes are user-owned data structures that compose with the query system — World has no awareness of them. The `SpatialIndex` trait has two methods: `rebuild` (full reconstruction) and `update` (incremental via `Changed<T>`). Stale entity references are caught automatically by generational ID validation.

For column-value lookups, `BTreeIndex<T>` provides O(log n) range queries and `HashIndex<T>` provides O(1) exact match. Both support validated queries (`get_valid`, `range_valid`) that filter despawned entities and removed components at query time.

When using `#[derive(Table)]`, field-level `#[index(btree)]` / `#[index(hash)]` attributes generate marker traits (`HasBTreeIndex<T>`, `HasHashIndex<T>`) that let `TablePlanner` enforce index presence at compile time. See [Schema & Mutation](#schema--mutation) for usage.

Indexes whose key types support [rkyv][rkyv] can be persisted to disk via `PersistentIndex`. On crash recovery, `load` + `update()` catches up from the saved tick — recovery time is proportional to the WAL tail, not world size. `AutoCheckpoint` can save registered indexes alongside snapshots automatically.

```rust
// Persistent index recovery
let post_restore_tick = world.change_tick();
wal.replay(&mut world, &codecs)?;

let mut idx = match load_btree_index::<Score>(&idx_path, post_restore_tick) {
    Ok(idx) => { idx.update(&mut world); idx }  // O(WAL tail)
    Err(_) => { let mut idx = BTreeIndex::new(); idx.rebuild(&mut world); idx }
};
```

Examples include a [uniform grid][uniform-grid] for neighbor search, a [Barnes-Hut][barnes-hut] quadtree for force approximation, and a persistent `BTreeIndex` in the [`persist`](examples/examples/persist.rs) example.

## Memory Management

RAM is fast, expensive, and limited. Minkowski provides three complementary features to help you do more with less:

### Pre-allocated Memory Pool

`WorldBuilder` creates a World backed by a single [mmap][mmap] region — all memory is allocated upfront at startup, [TigerBeetle][tigerbeetle]-style. After initialization, `try_spawn` returns `Err(PoolExhausted)` and `try_insert` returns `Err(InsertError)` instead of crashing the process. No dynamic allocation on the data path.

```rust
use minkowski::{World, HugePages, PoolExhausted};

let mut world = World::builder()
    .memory_budget(64 * 1024 * 1024)   // 64 MB fixed budget
    .hugepages(HugePages::Try)          // 2 MiB pages if available
    .lock_all_memory(true)              // mlockall — prevent swapping
    .build()?;

// Spawn until the pool is full — no OOM kill, just a Result
match world.try_spawn((pos, vel)) {
    Ok(entity) => { /* success */ }
    Err(PoolExhausted { .. }) => { /* shed load, evict, or report */ }
}
```

The pool uses size-classed slab allocation (64 B to 1 MB) with a three-tier pre-fault chain: `MAP_POPULATE` → `mlock` → manual page touch. `World::new()` still works — it uses the system allocator with no budget limit.

### Blob Offloading

Large per-entity assets (images, meshes, audio) shouldn't live in column storage. `BlobRef` is a lightweight component that stores an external key (S3 path, URL, content hash). The `BlobStore` trait provides a cleanup hook — after despawning entities, collect orphaned refs and let your store delete the external data.

```rust
use minkowski_persist::{BlobRef, BlobStore};

// Entity holds a key, not the bytes
world.spawn((metadata, BlobRef::new("s3://bucket/mesh-00af.bin")));

// After despawn, clean up external storage
store.on_orphaned(&orphaned_refs);
```

`BlobRef` lives in `minkowski-persist` for [rkyv][rkyv] serialization support — blob references survive snapshots and WAL replay. The engine stores only the reference; blob lifecycle is the user's responsibility (same external composition pattern as [indexes](#indexing)).

### Retention

`Expiry` is a countdown component — it counts retention dispatches, not ticks or wall-clock time. Each call to `registry.run()` decrements all counters by one and despawns entities that reach zero. You control how often retention runs; the engine never runs it automatically.

```rust
use minkowski::{World, Expiry, ReducerRegistry};

let mut world = World::new();
let mut registry = ReducerRegistry::new();
let retention_id = registry.retention(&mut world);

// This entity survives 5 retention dispatches
world.spawn((data, Expiry::after(5)));

// Each call is one "retention cycle" — counters decrement, zeros despawn
registry.run(&mut world, retention_id, ());
```

Together, these three features let a Minkowski deployment run indefinitely within a fixed memory envelope: the pool caps total memory, blob offloading keeps large assets external, and retention prevents unbounded entity growth.

## Examples

20 examples cover the full API surface — from basic queries to multi-threaded replication. See [`examples/README.md`](examples/README.md) for the full catalogue.

```
cargo run -p minkowski-examples --example boids --release              # flocking simulation
cargo run -p minkowski-examples --example persist --release             # WAL + snapshot lifecycle
cargo run -p minkowski-examples --example replicate --release           # network replication via channels
cargo run -p minkowski-examples --example reducer --release             # tour of all 7 reducer handles
cargo run -p minkowski-examples --example pool --release                # pre-allocated memory pool
cargo run -p minkowski-examples --example blob --release                # blob offloading to external store
cargo run -p minkowski-examples --example retention --release           # tick-based entity retention
cargo run -p minkowski-examples --example planner --release             # query planner with index selection
cargo run -p minkowski-examples --example materialized_view --release   # cached subscription views
```

## Python / Jupyter Integration

The `minkowski-py` crate exposes the ECS to Python via PyO3. Rust owns storage and computation; Python owns orchestration and analysis. Data crosses the boundary as Arrow RecordBatches, which load directly into Polars DataFrames.

```python
import minkowski_py as mk

world = mk.World()
registry = mk.ReducerRegistry(world)

# Spawn entities with named components
world.spawn("Position,Velocity", pos_x=1.0, pos_y=2.0, vel_x=0.5, vel_y=0.0)

# Query → Polars DataFrame (one-copy + zero-copy)
df = world.query("Position", "Velocity")

# Call pre-compiled Rust reducers by name
registry.run("boids_forces", world, world_size=500.0, sep_r=25.0)

# Write modified data back into the ECS
world.write_column("Position", entity_ids, pos_x=new_x, pos_y=new_y)
```

**Setup:**
```bash
cd crates/minkowski-py
uv venv && uv pip install -e ".[dev]"
maturin develop --release
```

9 registered component types and 5 Rust reducers (boids, gravity, life, movement) ship out of the box. Adding new components and reducers requires touching Rust code — see `/minkowski:python` for the step-by-step guide.

## AI-Assisted Development

Minkowski was built with Claude Code from the first commit. The development workflow includes:

- **16 slash commands** — `/design-doc` for feature planning, `/soundness-audit` for concurrency review, `/self-audit` for mutation path and visibility checks, `/perf-shakedown` for automated performance analysis, `/validate-api` and `/validate-macro` for correctness checks, `/pr` for PR creation, plus 9 domain-specific commands (`/minkowski:model`, `/minkowski:query`, `/minkowski:mutate`, `/minkowski:concurrency`, `/minkowski:reducer`, `/minkowski:index`, `/minkowski:persist`, `/minkowski:optimize`, `/minkowski:python`)
- **Pre-commit hooks** — `cargo fmt` and `cargo clippy -D warnings` run automatically on every commit

The commands teach the paradigm, not just the API — they encode the design principles and invariants that emerged across iterative development.

## Glossary

| Term | Definition |
|---|---|
| [Archetype][archetype] | A unique combination of component types. Entities with the same components share an archetype, enabling contiguous column storage. |
| [Barnes-Hut][barnes-hut] | An O(N log N) approximation algorithm for N-body force computation using a quadtree to aggregate distant particles. |
| [Bitset][bitset] | A compact array of bits used here for fast archetype matching — query matching is a bitwise subset check. |
| [Column store][column-store] | A storage layout where each field is stored in its own contiguous array, enabling cache-friendly sequential access and SIMD vectorization. |
| [Component][component] | A data type attached to an entity. Any `'static + Send + Sync` Rust type qualifies. Stored in BlobVec columns within archetypes. |
| [ECS][ecs] | Entity-Component System — an architectural pattern where entities are IDs, components are data, and systems are logic. Decouples data layout from behavior. |
| [Generational index][generational-index] | An ID scheme pairing an array index with a generation counter. Reusing an index bumps the generation, so stale handles are detected without a free-list scan. |
| [Miri][miri] | An interpreter for Rust's Mid-level IR that detects undefined behavior (aliasing violations, use-after-free, data races) at runtime. |
| [OCC][occ] | Optimistic concurrency control — transactions execute without locks, then validate at commit that no conflicting writes occurred. |
| [PCC][pcc] | Pessimistic concurrency control — transactions acquire locks before accessing data, preventing conflicts at the cost of potential contention. |
| [Query planner](#query-planner) | A Volcano-model optimizer that compiles queries into vectorized execution plans with automatic index selection. |
| [Rayon][rayon] | A Rust library for data parallelism. Used here for `par_for_each` parallel query iteration. |
| [Reducer](#typed-reducers) | A registered closure whose type signature declares its data access. The registry extracts conflict metadata at registration time. |
| [SIMD][simd] | Single Instruction, Multiple Data — CPU instructions that process multiple values in parallel. Minkowski's 64-byte column alignment enables auto-vectorization. |
| [SoA][soa] | Struct of Arrays — storing each field in a separate array rather than interleaving fields per record. The storage layout archetypes use. |
| [Tree Borrows][tree-borrows] | An experimental Rust aliasing model (stricter than Stacked Borrows) that Miri can check. Minkowski passes under this model. |
| [Uniform grid][uniform-grid] | A spatial index dividing space into fixed-size cells. O(1) cell lookup, O(k) neighbor iteration where k is the number of occupied neighbor cells. |
| [WAL][wal] | Write-ahead log — an append-only file where every mutation is recorded before being applied. Enables crash recovery by replaying the log. |

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).

<!-- Link definitions -->
[archetype]: https://ajmmertens.medium.com/building-an-ecs-2-archetypes-and-vectorization-fe21690f6d51
[barnes-hut]: https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
[rkyv]: https://github.com/rkyv/rkyv
[bitset]: https://en.wikipedia.org/wiki/Bit_array
[column-store]: https://en.wikipedia.org/wiki/Column-oriented_DBMS
[component]: https://en.wikipedia.org/wiki/Entity_component_system#Components
[ecs]: https://en.wikipedia.org/wiki/Entity_component_system
[generational-index]: https://lucassardois.medium.com/generational-indices-guide-8e3c5f7fd594
[miri]: https://github.com/rust-lang/miri
[occ]: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
[pcc]: https://en.wikipedia.org/wiki/Lock_(database)
[rayon]: https://github.com/rayon-rs/rayon
[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[soa]: https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays
[tree-borrows]: https://perso.crans.org/vanille/treebor/
[uniform-grid]: https://en.wikipedia.org/wiki/Grid_(spatial_index)
[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
[tsan]: https://clang.llvm.org/docs/ThreadSanitizer.html
[loom]: https://github.com/tokio-rs/loom
[mmap]: https://en.wikipedia.org/wiki/Mmap
[tigerbeetle]: https://tigerbeetle.com/
[volcano]: https://en.wikipedia.org/wiki/Volcano_(query_processing)
[wal]: https://en.wikipedia.org/wiki/Write-ahead_logging
