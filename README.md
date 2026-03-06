# Minkowski

A column-oriented database engine built from scratch by one human and one AI.

Minkowski is an archetype ECS that doubles as a transactional database engine. Its typed reducer system proves conflict freedom from closure signatures alone -- no runtime checks needed for scheduled reducers. Split-phase transactions enable safe concurrent reads by keeping `Tx` separate from `&mut World`, and WAL persistence composes with any transaction strategy via a single wrapper type. The engine ships with 295 tests across 4 crates, 8 runnable examples, and passes Miri under Tree Borrows -- all built in one week across 26 PRs.

## What makes this interesting

- **Column-oriented ECS that's also a transactional database engine** -- three mutation tiers (direct, transactional, durable) over the same columnar storage
- **Typed reducers** -- closures whose type signatures prove conflict freedom, enabling compile-time scheduling without runtime validation
- **Split-phase transactions** -- `Tx` does not hold `&mut World`, so concurrent reads via `&World` are sound by construction
- **AI-powered developer tooling** -- an auto-triggering skill provides passive expertise; 13 slash commands guide design decisions across the paradigm
- **Built from scratch in one week** -- 26 PRs, 295 tests, Miri verified under Tree Borrows

## Quick start

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

// Register a query reducer -- the type signature declares what it reads and writes
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

// Dispatch -- Access bitset extracted from the type signature at registration
registry.run(&mut world, move_id, ());
```

## Column-Oriented Storage

Each unique combination of component types gets an archetype -- a struct of arrays where each column is a `BlobVec` (type-erased growable byte array with 64-byte alignment). Queries match archetypes via `FixedBitSet` subset checks, then iterate columns with raw pointer arithmetic. No virtual dispatch in the hot path.

Entities are generational `u64` IDs (32-bit index + 32-bit generation). Recycled indices get bumped generations to prevent use-after-free. O(1) lookup from entity to archetype row via `Vec<Option<EntityLocation>>`. Sparse components (`HashMap<Entity, T>`) are opt-in for tags and rarely-queried data, preventing archetype fragmentation.

```rust
let mut world = World::new();
let e = world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
world.insert(e, Health(100));   // migrates entity to new archetype
world.remove::<Health>(e);      // migrates it back
```

## Query Engine

Two-tier query system with incremental caching. Dynamic queries (`world.query::<(&mut Pos, &Vel)>()`) cache matched archetype IDs per query type -- repeat calls skip the archetype scan entirely, and only new archetypes are scanned incrementally. Static table queries (`world.query_table::<Transform>()`) bypass archetype matching altogether via pre-resolved column offsets.

`Changed<T>` filters skip entire archetypes whose column was not mutably accessed since the last read. Ticks auto-advance on every mutation and query -- there is no manual `world.tick()` call. Parallel iteration via `par_for_each` (rayon) and chunk-based `for_each_chunk` yielding `&[T]`/`&mut [T]` slices for SIMD auto-vectorization.

```rust
// Dynamic query -- cached, skips archetype scan on repeat calls
for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
    pos.x += vel.dx;
}

// Change detection -- skip archetypes untouched since last read
for pos in world.query::<(&mut Pos, Changed<Vel>)>() {
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

Scheduled reducers (QueryRef, QueryMut) run with direct world access -- conflict freedom is proven at registration from the Access bitsets. Transactional reducers (EntityMut, Spawner, QueryWriter) buffer writes into an `EnumChangeSet` and commit through a transaction strategy. Dynamic reducers (`DynamicCtx`) trade compile-time precision for runtime flexibility with builder-declared upper bounds.

## Transactions

Three strategies over a unified `Transact` trait. `Tx` does not hold `&mut World` -- methods take world as a parameter, enabling split-phase execution where multiple transactions read concurrently via `&World` before committing sequentially.

- **Sequential** -- zero-cost passthrough, all ops delegate directly to World
- **Optimistic** -- live reads via `query_raw(&self)`, buffered writes, tick-based validation at commit
- **Pessimistic** -- cooperative per-column locks acquired at begin, buffered writes, commit always succeeds

Lock granularity is per-column `(ArchetypeId, ComponentId)`. The lock table is owned by the strategy, not World -- concurrency policy is external to storage.

Entity IDs allocated during a transaction are tracked automatically. On abort, orphaned IDs are pushed to a shared `OrphanQueue` and drained by World at the next `&mut self` call -- no entity ID ever leaks, regardless of how the transaction ends. `WorldId` checks prevent cross-world corruption when strategies are shared across threads.

## Persistence

The `minkowski-persist` crate provides WAL (write-ahead log) and bincode snapshots. `Durable<S, W>` wraps any `Transact` strategy -- on successful commit, the forward changeset is written to the WAL before being applied to World. Failed attempts (retries) are not logged. WAL write failure panics -- the durability invariant is non-negotiable. Recovery loads the latest snapshot and replays subsequent WAL entries.

```rust
// Durable wraps any strategy -- Optimistic, Pessimistic, or Sequential
let durable = Durable::new(strategy, wal, codecs);
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

`EnumChangeSet` records mutations as data with component bytes in a contiguous arena. `apply()` returns a reverse changeset for automatic undo -- applying the reverse restores the previous state. Typed helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) handle component registration and raw pointers internally. `CommandBuffer` provides deferred structural changes during iteration.

## Spatial Indexing

`SpatialIndex` is a lifecycle trait for user-owned spatial data structures. Indexes are fully external to World -- they compose from existing query primitives. The trait has two methods: `rebuild` (full reconstruction) and `update` (optional, for incremental updates via `Changed<T>`). Stale entity references are caught by generational validation at query time.

Two implementations ship as examples: a uniform grid for O(N*k) neighbor search (boids) and a Barnes-Hut quadtree for O(N log N) force approximation (nbody). Both demonstrate that the trait accommodates structurally different algorithms without friction.

## Examples

| Example | What it exercises | Run |
|---|---|---|
| `boids` | Query reducers, SpatialGrid, flocking simulation (5K entities) | `cargo run -p minkowski-examples --example boids --release` |
| `life` | QueryMut reducer, Table, EnumChangeSet undo/redo, Changed\<T> | `cargo run -p minkowski-examples --example life --release` |
| `nbody` | Query reducer, Barnes-Hut quadtree, SpatialIndex trait | `cargo run -p minkowski-examples --example nbody --release` |
| `scheduler` | ReducerRegistry conflict detection, greedy batch scheduling | `cargo run -p minkowski-examples --example scheduler --release` |
| `transaction` | Sequential/Optimistic/Pessimistic strategies, query reducers | `cargo run -p minkowski-examples --example transaction --release` |
| `battle` | EntityMut reducers, rayon parallel snapshots, tunable conflict | `cargo run -p minkowski-examples --example battle --release` |
| `persist` | QueryWriter reducer, Durable WAL, snapshot recovery | `cargo run -p minkowski-examples --example persist --release` |
| `reducer` | All 6 handle types, structural mutations, dynamic reducers | `cargo run -p minkowski-examples --example reducer --release` |

## AI-Assisted Development

Minkowski was built with Claude Code from the first commit. The development workflow includes:

- **Auto-triggering skill** (`minkowski-guide.md`) -- provides passive expertise on the ECS paradigm whenever Claude Code works in this repo
- **13 slash commands** -- `/design-doc` for feature planning, `/soundness-audit` for concurrency review, `/validate-api` and `/validate-macro` for correctness checks, plus 8 domain-specific commands (`/minkowski:model`, `/minkowski:query`, `/minkowski:mutate`, `/minkowski:concurrency`, `/minkowski:reducer`, `/minkowski:index`, `/minkowski:persist`, `/minkowski:optimize`)
- **Pre-commit hooks** -- `cargo fmt` and `cargo clippy -D warnings` run automatically on every commit

The skills teach the paradigm, not just the API -- they encode the design principles and invariants that emerged across 26 PRs of iterative development.

## Architecture Decision Records

Design decisions are documented as ADRs in [`docs/adr/`](docs/adr/). Each records what was decided, what alternatives were considered, and what trade-offs were accepted. Every feature was designed before implementation -- the design conversation catches semantic bugs (concurrent state corruption, entity ID leaks, lock privilege errors) that compilation and tests miss.

## Building & Testing

```
cargo test -p minkowski                # 295 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p minkowski               # criterion benchmarks vs hecs
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib   # UB check
```

CI runs fmt, clippy, test, and Miri sequentially on every PR. A `ci-pass` aggregator job is the single required status check for branch protection -- it explicitly verifies all four jobs succeeded, avoiding GitHub's "skipped = passed" loophole.

## Roadmap

| Feature | Rationale |
|---|---|
| Query planning (Volcano model) | Optimize complex queries across indexes |
| B-tree / hash indexes | O(log n) lookups by column value |
| rkyv zero-copy snapshots | Zero-copy deserialization matching BlobVec layout |
| Replication & sync | Filtered WAL replay for read replicas and client mirrors |

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
