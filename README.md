# Minkowski

A column-oriented database engine built from scratch by one human and one AI.

Minkowski is an [archetype][archetype] [ECS][ecs] that doubles as a transactional database engine. Its typed [reducer](#typed-reducers) system proves conflict freedom from closure signatures alone — no runtime checks needed for scheduled reducers. [Split-phase transactions](#transactions) enable safe concurrent reads by keeping `Tx` separate from `&mut World`, and [WAL][wal] persistence composes with any transaction strategy via a single wrapper type.

**Minkowski** is a storage engine for real-time interactive applications — games, simulations, collaborative tools — that need both the iteration speed of an ECS and the transactional guarantees of a database.
Most ECS engines give you fast iteration but no persistence, no rollback, no concurrency control. Most databases give you transactions but can't iterate 100,000 components per frame without blowing your cache budget. minkowski gives you both without making you pay for the one you're not using.
The core insight: archetype-based column storage is already a columnar database. Minkowski makes that explicit. Components are stored in flat, aligned, SIMD-friendly arrays. Queries resolve to bitset comparisons. Change detection uses a monotonic tick counter that provides total ordering without per-entity overhead. Everything you need for a database is already present in a well-designed ECS — you just have to expose the right primitives.

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

## What makes this interesting

- **[Column-oriented][column-store] ECS that's also a transactional database engine** — three mutation tiers (direct, transactional, durable) over the same columnar storage
- **Typed reducers** — closures whose type signatures prove conflict freedom, enabling compile-time scheduling without runtime validation. A reducer is a function with a declared access pattern encoded in its type signature. The engine extracts read/write bitsets from the types at registration time and uses them for conflict analysis. Two reducers with disjoint component access are proven non-conflicting at compile time — no runtime locks, no optimistic retry, no scheduler overhead. If you try to access a component you didn't declare, the compiler rejects it.
- **Split-phase transactions** — `Tx` does not hold `&mut World`, so concurrent reads via `&World` are sound by construction. Transactions read from shared &World references and buffer writes into a private changeset. Multiple transactions execute concurrently in a parallel phase, then commit sequentially. The Rust borrow checker enforces the phase boundary — you can't start committing until all shared references are released. No manual synchronization required.
- **AI-powered developer tooling** — 16 slash commands guide design decisions across the ECS paradigm, from data modeling to concurrency strategy
- **No undefined behaviour** — Miri verified under Tree Borrows
- **Composable persistence** — The `Durable<S>` wrapper takes any transaction strategy and guarantees that every committed transaction is WAL-logged before the caller sees the result. If the WAL write fails, the process panics — there's no silent data loss. Crash recovery replays the log from the last snapshot. Zero-copy snapshot loading via mmap + rkyv bypasses envelope deserialization and copies `#[repr(C)]` component bytes directly into columns without typed construction. The core engine has zero dependency on any serialization framework.
- **Zero-cost tiers** — Users who don't need transactions pay nothing — direct world.get_mut() has no overhead. Users who need transactions but not persistence pay only for the changeset buffer. Users who need durability add the Durable wrapper. Each tier adds cost only for the guarantees it provides.
- **Mechanisms, not policy** — There's no built-in scheduler, no application lifecycle, no domain-specific component types. Scheduling, system ordering, and parallelism strategy are the framework author's job — minkowski provides `Access` bitsets and `is_compatible()` so they can build their own. Secondary indexes (spatial grids, B-trees) are external consumers of the change detection system, not engine features. The litmus test: "does this require knowledge of what the user's program does?" If yes, it doesn't belong in minkowski.

## Column-Oriented Storage

Each unique combination of [component][component] types gets an [archetype][archetype] — a [struct of arrays][soa] where each column is a `BlobVec` (type-erased growable byte array with 64-byte alignment). Queries match archetypes via [bitset][bitset] subset checks, then iterate columns with raw pointer arithmetic. No virtual dispatch in the hot path.

Entities are [generational][generational-index] `u64` IDs (32-bit index + 32-bit generation). Recycled indices get bumped generations to prevent use-after-free. O(1) lookup from entity to archetype row via `Vec<Option<EntityLocation>>`. Sparse components (paged sparse sets with O(1) lookup) are opt-in for tags and rarely-queried data, preventing archetype fragmentation.

```rust
let mut world = World::new();
let e = world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
world.insert(e, Health(100));   // migrates entity to new archetype
world.remove::<Health>(e);      // migrates it back
```

## Query Engine

Two-tier query system with incremental caching. Dynamic queries (`world.query::<(&mut Pos, &Vel)>()`) cache matched archetype IDs per query type — repeat calls skip the archetype scan entirely, and only new archetypes are scanned incrementally. Static table queries (`world.query_table::<Transform>()`) bypass archetype matching altogether via pre-resolved column offsets.

`Changed<T>` filters skip entire archetypes whose column was not mutably accessed since the last read. Ticks auto-advance on every mutation and query — there is no manual `world.tick()` call. Parallel iteration via `par_for_each` ([rayon][rayon]) and chunk-based `for_each_chunk` yielding `&[T]`/`&mut [T]` slices for [SIMD][simd] auto-vectorization.

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

## Typed Reducers

Reducers are closures registered with the `ReducerRegistry`. The type signature declares exactly what the closure can access, and the registry extracts `Access` metadata at registration time for conflict detection.

| Handle | Access | Execution model |
|---|---|---|
| `EntityRef<C>` | Read components in set C for one entity | Transactional |
| `EntityMut<C>` | Read + buffered write + remove for one entity | Transactional |
| `Spawner<B>` | Create new entities with bundle B | Transactional |
| `QueryWriter<Q>` | Buffered bulk iteration via `WritableRef<T>` | Transactional |
| `QueryRef<Q>` | Read-only iteration | Scheduled |
| `QueryMut<Q>` | Read-write iteration with direct `&mut World` | Scheduled |
| `DynamicCtx` | Builder-declared upper bounds, buffered writes via `EnumChangeSet` | Dynamic |

Scheduled reducers (QueryRef, QueryMut) run with direct world access — conflict freedom is proven at registration from the Access bitsets. Transactional reducers (EntityMut, Spawner, QueryWriter) buffer writes into an `EnumChangeSet` and commit through a transaction strategy. Dynamic reducers (`DynamicCtx`) trade compile-time precision for runtime flexibility with builder-declared upper bounds.

## Transactions

Three strategies over a unified `Transact` trait. `Tx` does not hold `&mut World` — methods take world as a parameter, enabling split-phase execution where multiple transactions read concurrently via `&World` before committing sequentially.

- **Sequential** — zero-cost passthrough, all ops delegate directly to World
- **[Optimistic][occ]** — live reads via `query_raw(&self)`, buffered writes, tick-based validation at commit
- **[Pessimistic][pcc]** — cooperative per-column locks acquired at begin, buffered writes, commit always succeeds

Lock granularity is per-column `(ArchetypeId, ComponentId)`. The lock table is owned by the strategy, not World — concurrency policy is external to storage.

Entity IDs allocated during a transaction are tracked automatically. On abort, orphaned IDs are pushed to a shared `OrphanQueue` and drained by World at the next `&mut self` call — no entity ID ever leaks, regardless of how the transaction ends. `WorldId` checks prevent cross-world corruption when strategies are shared across threads.

## Persistence

The `minkowski-persist` crate provides [WAL][wal] (write-ahead log) and [rkyv][rkyv]-serialized snapshots. `Durable<S>` wraps any `Transact` strategy — on successful commit, the forward changeset is written to the WAL before being applied to World. Failed attempts (retries) are not logged. WAL write failure panics — the durability invariant is non-negotiable. Recovery loads the latest snapshot and replays subsequent WAL entries. `Snapshot::load_zero_copy` uses mmap + rkyv's archived types to skip envelope deserialization. For `#[repr(C)]` components whose archived layout matches native, archived bytes are copied directly into BlobVec columns without typed construction; other components fall back to `rkyv::from_bytes`.

```rust
// Durable wraps any strategy — Optimistic, Pessimistic, or Sequential
let durable = Durable::new(strategy, wal, codecs);  // S = Optimistic here
durable.transact(&mut world, access, |tx, world| { /* ... */ });
// Changeset written to WAL on successful commit
```

## Schema & Mutation

`#[derive(Table)]` generates compile-time schema declarations with typed row accessors (`FooRef<'w>` / `FooMut<'w>`). Table queries skip archetype matching entirely via pre-resolved column offsets.

```rust
#[derive(Table)]
struct Transform {
    pos: Pos,
    vel: Vel,
}

// Table query -- direct pointer arithmetic, zero archetype matching
for row in world.query_table::<Transform>() {
    println!("{}, {}", row.pos.x, row.vel.dx);
}
```

`EnumChangeSet` records mutations as data with component bytes in a contiguous arena. `apply()` returns a reverse changeset for automatic undo — applying the reverse restores the previous state. Typed helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) handle component registration and raw pointers internally. `CommandBuffer` provides deferred structural changes during iteration.

## Spatial Indexing

`SpatialIndex` is a lifecycle trait for user-owned spatial data structures. Indexes are fully external to World — they compose from existing query primitives. The trait has two methods: `rebuild` (full construction) and `update` (optional, for incremental updates via `Changed<T>`). Stale entity references are caught by generational validation at query time.

Two implementations ship as examples: a [uniform grid][uniform-grid] for O(N*k) neighbor search (boids) and a [Barnes-Hut][barnes-hut] quadtree for O(N log N) force approximation (nbody). Both demonstrate that the trait accommodates structurally different algorithms without friction. For column-value lookups, `BTreeIndex<T>` provides O(log n) range queries and `HashIndex<T>` provides O(1) exact match — both use `Changed<T>` for incremental updates.

## Examples

| Example | What it does | Run |
|---|---|---|
| `boids` | Flocking simulation with 5 000 entities. Registers three `QueryMut` reducers (zero acceleration, compute forces, integrate), rebuilds a `SpatialGrid` each frame for O(k) neighbor lookup, and uses `CommandBuffer` for deferred despawn/respawn churn. Demonstrates SIMD auto-vectorization via `for_each_chunk`. | `cargo run -p minkowski-examples --example boids --release` |
| `life` | Conway's Game of Life on a 64×64 toroidal grid, running 500 generations with undo/redo. Uses `#[derive(Table)]` for typed row access, a `QueryMut` reducer to update neighbor counts, `Changed<CellState>` for incremental detection, and `EnumChangeSet` to record reversible mutations for 50-generation rewind followed by deterministic replay. | `cargo run -p minkowski-examples --example life --release` |
| `nbody` | Barnes-Hut N-body gravity simulation with 2 000 entities. Builds a quadtree `SpatialIndex` each frame to approximate O(N log N) force computation, uses rayon snapshot-based parallel force accumulation, and dispatches the integration step through a `QueryMut` reducer. Demonstrates generational entity validation and archetype stability under spawn/despawn churn. | `cargo run -p minkowski-examples --example nbody --release` |
| `scheduler` | Greedy batch scheduler over 6 registered query reducers. Calls `registry.query_reducer_access(id)` to extract `Access` bitsets, builds a conflict matrix, and assigns systems to non-conflicting batches via graph coloring — producing 3 batches where intra-batch systems touch disjoint components and could run in parallel. | `cargo run -p minkowski-examples --example scheduler --release` |
| `transaction` | Two-part comparison of raw `Tx` building blocks versus reducer-based dispatch. Part 1 shows `Sequential` begin/commit and `Optimistic` transact closure at the low level. Part 2 registers the same logic as query reducers, gaining strategy-agnostic dispatch and free conflict detection without changing the closure body. | `cargo run -p minkowski-examples --example transaction --release` |
| `battle` | Multi-threaded arena with 500 entities over 100 frames. Registers `EntityMut` reducers for attack and heal, dispatches them through both `Optimistic` and `Pessimistic` strategies, and uses `rayon::join` for parallel snapshot reads before sequential reducer dispatch. Demonstrates low-conflict (disjoint component sets) versus high-conflict (shared `Health` column) modes and how each strategy handles retry. | `cargo run -p minkowski-examples --example battle --release` |
| `persist` | Full persistence lifecycle with 100 entities across 3 archetypes. Spawns entities with varied component sets (including sparse components), saves an rkyv `Snapshot`, applies mutations through a `Durable`-wrapped `QueryWriter` reducer (WAL-backed), recovers via snapshot + WAL replay, then demonstrates zero-copy snapshot loading via mmap. | `cargo run -p minkowski-examples --example persist --release` |
| `reducer` | Tour of all 7 reducer handle types in one program: `EntityMut` (heal), `QueryMut` (gravity), `QueryRef` (logger), `Spawner` (spawn projectiles), `QueryWriter` (drag), name-based lookup, access conflict detection between registered reducers, `DynamicCtx` (conditional runtime access with builder-declared bounds), and structural despawn via dynamic iteration. | `cargo run -p minkowski-examples --example reducer --release` |
| `index` | Column index demo on a `Score` component across two archetypes (200 entities). Builds a `BTreeIndex` for O(log n) range queries and a `HashIndex` for O(1) exact lookups, performs incremental updates via per-index `ChangeTick` after mutations, and validates stale-entry detection after despawn. | `cargo run -p minkowski-examples --example index --release` |
| `flatworm` | Planarian flatworm simulator with 200 worms over 1 000 frames. Implements chemotaxis (nutrition/distance² gradient via `FoodGrid` spatial index), binary fission when energy exceeds a threshold, starvation despawn below minimum energy, and food respawning. Demonstrates `SpatialIndex` trait, `CommandBuffer` for deferred spawn/despawn, and `QueryMut`/`QueryRef` reducers for movement, metabolism, and census. | `cargo run -p minkowski-examples --example flatworm --release` |
| `circuit` | Analog circuit simulator: 555 astable oscillator → LCR bandpass filter → 741 voltage follower. Circuit nodes are entities with `Voltage` components; elements (resistors, inductors, op-amp) reference node entities for connectivity. Uses symplectic Euler integration for L/C elements, `QueryMut`/`QueryRef` reducers via `ReducerRegistry`, and prints an ASCII waveform of the filter output. 200K steps at 100 ns timestep. | `cargo run -p minkowski-examples --example circuit --release` |
| `tactical` | Multi-operator tactical map with server-authoritative replication. Exercises 8 previously uncovered API gaps: sparse components (`insert_sparse`/`iter_sparse`), `par_for_each`, `Optimistic` transactions with `Conflict` inspection, `Entity::to_bits`/`from_bits` for wire serialization, world introspection (`archetype_count`, `component_name`), `register_entity_despawn`, `HashIndex::get_valid()` stale filtering, and `EnumChangeSet`/`MutationRef` iteration for replication packets. Two operator threads communicate with the server via `mpsc` channels. | `cargo run -p minkowski-examples --example tactical --release` |

## Python / Jupyter Integration

The `minkowski-py` crate exposes the ECS directly to Python via PyO3. Rust owns storage and computation; Python owns orchestration and analysis. Data crosses the boundary as Arrow RecordBatches — one `memcpy` from BlobVec into Arrow, then zero-copy into Polars DataFrames via `pyo3-arrow`.

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

## Architecture Decision Records

- Filtering is replaced by Bitset ANDing.
- Queries don't plan because the memory layout is the plan.
- Joins don't exist because archetypes are pre-joined.
- Defragmentation is constant via swap-remove.
- Vectorization is guaranteed via 64-byte aligned slices.
- Persistence is zero-copy via rkyv + mmap.
- Safety is compile-time via the split-phase transaction model.
- ACID-lite: Atomicity and isolation via ChangeSet and split-phase transactions.
- Indices: O(1) Entity Mapping, O(log N) Range-based and O(1) Archetype Bitsets.
- Storage: 64-byte aligned, dense SoA with Zero-copy Persistence.
- Execution: Hardware-native SSE via slice-based auto-vectorization.

Design decisions are documented as ADRs in [`docs/adr/`](docs/adr/). Each records what was decided, what alternatives were considered, and what trade-offs were accepted. Every feature was designed before implementation — the design conversation catches semantic bugs (concurrent state corruption, entity ID leaks, lock privilege errors) that compilation and tests miss.

## Building & Testing

```
cargo test -p minkowski                # 368 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p minkowski               # criterion benchmarks vs hecs
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib   # UB check
```

CI runs fmt, clippy, test, and Miri sequentially on every PR. A `ci-pass` aggregator job is the single required status check for branch protection — it explicitly verifies all four jobs succeeded, avoiding GitHub's "skipped = passed" loophole.

## Roadmap

| Feature | Rationale |
|---|---|
| Replication & sync | Filtered WAL replay for read replicas and client mirrors |
| Observability companion crate | `minkowski-observe`: pure consumer that calls `world.snapshot_metrics()`, diffs consecutive snapshots, computes rates (entity churn, changeset throughput, cache hit rates), and exports. Stretch within stretch: [ratatui][ratatui] TUI dashboard for live archetype sizes and sync diagnostics. Placed after replication because replication adds the most interesting metrics — sync latency, conflict rate across clients, changeset sizes over the wire |

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
[ratatui]: https://github.com/ratatui/ratatui
[wal]: https://en.wikipedia.org/wiki/Write-ahead_logging
