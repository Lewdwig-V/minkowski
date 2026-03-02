# minkowski

A column-oriented database built atop a high-performance Archetype Entity-Component System.

## Overview

Minkowski is a column-oriented database engine built on an archetype ECS foundation. It combines the runtime flexibility of entity-component systems with the performance characteristics of analytical databases: cache-friendly columnar storage, SIMD-friendly iteration, and compile-time schema declarations.

**Core storage** — type-erased `BlobVec` columns packed into archetypes with 64-byte alignment, generational entity IDs, and sparse component opt-in via `HashMap<Entity, T>` to prevent archetype fragmentation.

**Query engine** — two-tier design with static table queries (direct pointer arithmetic, zero indirection) and dynamic component queries (bitset matching, incrementally cached). Parallel iteration via rayon, chunk-based `&[T]`/`&mut [T]` slices for auto-vectorization, and `Changed<T>` filters that skip entire archetypes untouched since the last read.

**Schema & mutation** — `#[derive(Table)]` proc macro for compile-time schema registration with typed row accessors. `EnumChangeSet` records mutations as data with automatic reverse generation for rollback. `CommandBuffer` for deferred structural changes during iteration.

**Concurrency & transactions** — `Access` extracts per-component read/write metadata for conflict detection. Three transaction strategies — Sequential (zero-cost passthrough), Optimistic (tick-based validation), and Pessimistic (cooperative per-column locks) — with split-phase design enabling concurrent reads via `World::query_raw(&self)`.

**Secondary indexes** — `SpatialIndex` lifecycle trait for user-owned spatial structures (grids, quadtrees, BVH). Indexes compose from query primitives and handle entity despawns via generational validation.

```rust
use minkowski::{World, Entity, CommandBuffer, EnumChangeSet, Table, Changed};

#[derive(Table)]
struct Transform {
    pos: Position,
    vel: Velocity,
}

struct Position { x: f32, y: f32 }
struct Velocity { dx: f32, dy: f32 }

let mut world = World::new();

// Spawn entities into archetypes
let e = world.spawn((Position { x: 0.0, y: 0.0 }, Velocity { dx: 1.0, dy: 0.0 }));

// Dynamic query (cached — skips archetype scan on repeat calls)
for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
    pos.x += vel.dx;
    pos.y += vel.dy;
}

// Parallel iteration
world.query::<(&mut Position, &Velocity)>().par_for_each(|(pos, vel)| {
    pos.x += vel.dx;
});

// Chunk iteration — yields &[T]/&mut [T] slices for SIMD auto-vectorization
world.query::<(&mut Position, &Velocity)>()
    .for_each_chunk(|(positions, velocities)| {
        for i in 0..positions.len() {
            positions[i].x += velocities[i].dx;
        }
    });

// Table query — typed access, bypasses archetype matching entirely
world.spawn(Transform {
    pos: Position { x: 0.0, y: 0.0 },
    vel: Velocity { dx: 1.0, dy: 0.0 },
});
for row in world.query_table::<Transform>() {
    println!("{}, {}", row.pos.x, row.vel.dx);
}

// Archetype migration
world.insert(e, Health(100));   // moves entity to new archetype
world.remove::<Health>(e);      // moves it back

// Change detection — skip unchanged archetypes
// Tick auto-advances on every mutation/query; no manual world.tick() call needed.
for pos in world.query::<(&mut Position, Changed<Velocity>)>() {
    // only runs for entities whose Velocity column was mutably accessed since last query
}

// Deferred mutation during iteration
let mut cmds = CommandBuffer::new();
for (entity, pos) in world.query::<(Entity, &Position)>() {
    if pos.x > 100.0 {
        cmds.despawn(entity);
    }
}
cmds.apply(&mut world);

// Data-driven mutations with automatic undo
let mut cs = EnumChangeSet::new();
cs.insert::<Velocity>(&mut world, e, Velocity { dx: 5.0, dy: 0.0 });
cs.remove::<Health>(&mut world, e);
let reverse = cs.apply(&mut world);  // apply and capture reverse
let _ = reverse.apply(&mut world);   // undo — restores previous state
```

### Storage design

Each unique combination of component types gets an **archetype** — a struct of arrays where each component type is a `BlobVec` (type-erased growable byte array). Queries match archetypes via `FixedBitSet` subset checks, then iterate columns with raw pointer arithmetic. No virtual dispatch in the hot loop.

**Entity** = u64 with 32-bit index + 32-bit generation. Recycled indices get bumped generations to prevent use-after-free. O(1) lookup from entity to archetype row via `Vec<Option<EntityLocation>>`.

**Sparse components** (opt-in `HashMap<Entity, T>`) for tags and rarely-present data. Dense archetype storage is the default.

### Boids example

A flocking simulation that exercises every ECS code path — spawn, despawn, multi-component queries, mutation, parallel iteration, chunk-based SIMD iteration, deferred commands, and archetype stability under entity churn.

```
$ cargo run -p minkowski-examples --example boids --release

frame 0000 | entities:  5000 | avg_vel: 2.00 | dt: 6.5ms
frame 0100 | entities:  5000 | avg_vel: 1.93 | dt: 3.7ms
frame 0200 | entities:  5000 | avg_vel: 1.87 | dt: 2.3ms
...
frame 0999 | entities:  5000 | avg_vel: 1.80 | dt: 3.0ms
Done.
```

5,000 boids with uniform spatial grid neighbor search (O(N·k) instead of O(N²)), parallel force computation, vectorized integration via `for_each_chunk`, and random spawn/despawn churn every 100 frames. Integration loops compile to branchless AVX-512 masked ops with `-C target-cpu=native`.

### Game of Life example

A 64×64 Conway's Game of Life that exercises the features boids doesn't cover — `Changed<T>`, `EnumChangeSet` typed API for reversible mutations, and time-travel via undo/replay.

```
$ cargo run -p minkowski-examples --example life --release

Game of Life: 64x64 grid, 4096 cells, 500 generations
Initial alive: 1843

gen    0 | alive: 1843 | changes:  892 | dt: 0.18ms
gen   50 | alive:  842 | changes:  198 | dt: 0.11ms
gen  100 | alive:  782 | changes:  172 | dt: 0.09ms
...
gen  499 | alive:  751 | changes:  148 | dt: 0.08ms

Rewinding 50 generations...
  rewind step  0 | alive:  751
  rewind step 10 | alive:  763
...
Verification passed: alive counts match.
```

`Changed<CellState>` skips the entire archetype when no cell state mutated — iteration cost drops to nearly zero for stable generations. Each generation builds an `EnumChangeSet` via `cs.insert::<CellState>()`, and `apply()` returns the reverse changeset automatically. Rewinding is just `reverse.apply(&mut world)` — no manual state tracking needed.

### N-body example

A Barnes-Hut gravity simulation that exercises the `SpatialIndex` trait with a quadtree — a fundamentally different spatial structure from the uniform grid used in boids.

```
$ cargo run -p minkowski-examples --example nbody --release

N-body: 2000 entities, 1000 frames, theta=0.50
frame 0000 | entities:  2000 | dt: 4.2ms
frame 0200 | entities:  2000 | dt: 2.8ms
frame 0400 | entities:  2000 | dt: 2.5ms
...
frame 0999 | entities:  2000 | dt: 2.3ms
Done.
```

2,000 bodies with O(N log N) Barnes-Hut force approximation via quadtree, parallel force computation via rayon, vectorized symplectic Euler integration via `for_each_chunk`, and random spawn/despawn churn to exercise generational validation of stale index entries.

### Scheduler example

A minimal conflict analysis demo showing how a framework author would use `Access` to detect data races between systems.

```
$ cargo run -p minkowski-examples --example scheduler --release

Conflict matrix:

       movement <-> gravity        CONFLICT
       movement <-> health_regen   independent
       movement <-> apply_damage   independent
       movement <-> log_positions  CONFLICT
       movement <-> log_health     independent
        gravity <-> health_regen   independent
        gravity <-> apply_damage   independent
        gravity <-> log_positions  independent
        gravity <-> log_health     independent
   health_regen <-> apply_damage   CONFLICT
   health_regen <-> log_positions  independent
   health_regen <-> log_health     CONFLICT
   apply_damage <-> log_positions  independent
   apply_damage <-> log_health     CONFLICT
  log_positions <-> log_health     independent

Batch assignment (3 batches):
  batch 0: [movement, health_regen]
  batch 1: [gravity, apply_damage, log_positions]
  batch 2: [log_health]
...
Done.
```

Six systems demonstrate every conflict case: write/write (`health_regen` vs `apply_damage`), read/write (`log_positions` vs `movement`), disjoint writes that parallelize (`movement` + `health_regen`), and read-only systems that batch with non-overlapping writers. The greedy batcher assigns systems to 3 batches — within each batch, systems touch disjoint components and could run in parallel.

### Transaction example

Demonstrates all three `TransactionStrategy` implementations on the same workload — Sequential (zero-cost), Optimistic (tick validation), and Pessimistic (cooperative locks).

```
$ cargo run -p minkowski-examples --example transaction --release

Transaction strategies — 100 entities with (Pos, Vel, Health)

1. Sequential (zero-cost passthrough)
   Movement system: writes Pos, reads Vel
  after 10 steps: avg pos = (59.5, 5.0)
  commit result:  Ok (always succeeds, zero overhead)

2. Optimistic (clean commit)
   Health decay system: reads Health, buffers writes, validates at commit
  after 10 steps: avg health = 70
  commit result:  Ok (no conflicts detected)

3. Optimistic (conflict demonstration)
   Declared access: reads Pos, writes Health
   But mutates Pos through live query — tick advances past snapshot
  commit result:  Err(Conflict) — read column was modified
  buffered writes discarded (transaction aborted)

4. Pessimistic (guaranteed commit)
   Same health decay, but with cooperative column locks
  after 10 steps: avg health = 40
  commit result:  Ok (locks guarantee success)

Done.
```

The caller decides the transaction boundary and the concurrency strategy. Sequential has zero overhead — the compiler inlines everything. Optimistic validates read-set ticks at commit. Pessimistic acquires per-column cooperative locks at begin. Tx types don't hold `&mut World` — methods take world as a parameter, enabling split-phase execution with concurrent reads via `rayon`.

### Battle example

A multi-threaded arena battle that stress-tests transaction strategies under tunable conflict pressure. 500 entities across 2 teams, with combat and healing systems running as parallel transactions via `rayon::join`.

```
$ cargo run -p minkowski-examples --example battle --release

Battle simulation: 500 entities (250 per team), 100 frames

=== Low conflict mode (disjoint targets) ===

Optimistic:
  commits: 200 | conflicts: 0 | avg frame: 0.19ms

Pessimistic:
  commits: 200 | conflicts: 0 | avg frame: 0.17ms

=== High conflict mode (overlapping targets) ===

Optimistic:
  commits: 100 | conflicts: 100 | avg frame: 0.16ms

Pessimistic:
  commits: 200 | conflicts: 0 | avg frame: 0.17ms

Done.
```

In low-conflict mode (disjoint entity targets), both strategies perform identically — zero conflicts. In high-conflict mode (overlapping targets), optimistic transactions see 100 conflicts (one per frame for the second-to-commit system), while pessimistic guarantees all 200 commits succeed via cooperative column locks.

### Benchmarks

Criterion benchmarks compare against [hecs](https://crates.io/crates/hecs):

```
$ cargo bench -p minkowski
```

Suites: `spawn` (10K entities), `iterate` (10K), `parallel` (100K vs sequential), `add_remove` (1K migration cycles), `fragmented` (20 archetypes).

## What's next

**Phase 4 — done:** `SpatialIndex` lifecycle trait, Barnes-Hut N-body example, boids grid refactored through the trait. `Access` struct for query conflict detection, scheduler example. `TransactionStrategy` trait with Sequential/Optimistic/Pessimistic implementations, transaction example.

| Phase | Feature | Why |
|---|---|---|
| 4 | Persistence — WAL + snapshots | Durable state via BlobVec memcpy to disk |
| 5 | Query planning (Volcano model) | Optimize complex queries across indexes |
| 5 | B-tree / hash indexes | Fast range and equality lookups on component fields |

The architecture is designed so each phase layers on without rewriting the previous one. BlobVec's type-erased byte storage is already memcpy-friendly for snapshots. `EnumChangeSet` already provides the mutation abstraction for persistence and rollback.

## Building

```
cargo build            # debug
cargo build --release  # optimized
cargo test             # all tests
cargo bench            # benchmarks
```

Requires Rust 2021 edition. Dependencies: `rayon`, `fixedbitset`, `minkowski-derive`.

## Architecture

### Storage

- **Column-oriented archetype storage** — each archetype is a collection of independently addressable `BlobVec` columns, not interleaved allocations. Enables per-column features (indexes, change ticks, compression) without touching archetype logic.
- **Dense vs. sparse split** — components that benefit from contiguous iteration live in archetypes. Marker tags and rarely-queried components live in sparse maps (`HashMap<Entity, T>`) and don't contribute to archetype identity. Prevents archetype fragmentation.
- **Entity IDs** — generational index `(u32 index, u32 generation)` packed into `u64`.

### Compile-Time Schema Registry

Proc macros define "tables" — known component bundles with fixed schemas:

```rust
#[derive(Table)]
struct Transform {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}
```

A table is a pre-registered archetype with statically-known column offsets. Queries against known tables compile to direct pointer arithmetic, bypassing the archetype graph entirely.

### Query Engine

Two-tier query system:

- **Static queries** — against known tables, skip archetype graph traversal, iterate as `&[T]` with zero indirection.
- **Dynamic queries** — arbitrary component combinations, matched archetypes cached per query type in `QueryCacheEntry` (incrementally updated only when new archetypes appear).

Matching uses **bitsets** — one per archetype, `(archetype_bits & query_bits) == query_bits`. Handles negation, optionals, and union queries with bitwise ops. Scales to hundreds of archetypes in microseconds.

The current hot path is pointer-chasing through pre-resolved column arrays cached per-query type. When indexes land (Phase 5), query planning will layer on top:

| Node | Purpose |
|------|---------|
| `Scan` | Iterate all rows in an archetype (current) |
| `IndexScan` | B-tree range lookup on an indexed column (planned) |
| `Filter` | Row-level predicate evaluation (planned) |
| `Merge` | Concatenate results from multiple archetypes (planned) |

### Indexes

Optional per-column B-tree or hash indexes:

```rust
#[derive(Table)]
struct Player {
    #[index(btree)]
    score: u64,
    #[index(hash)]
    name: String,
}
```

Enables O(log n) lookups by column value — essential for the database side, absent from pure-game ECS.

### Mutation

- **Deferred via command buffers** — structural changes (add/remove component, spawn/despawn) are batched and applied at sync points. Amortizes archetype migration cost and avoids iterator invalidation.
- **Data-driven changesets** — `EnumChangeSet` records mutations as an enum vec with component bytes in a contiguous arena. `apply()` returns a reverse changeset for rollback. Typed helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) handle raw pointers and component registration internally.
- **Lazy archetype migration** — if a newly added component doesn't affect any active query, store it in sparse storage and defer the archetype move until actually needed.
- **Change detection** — per-column tick tracking (`Changed<T>`). Entire archetypes skipped when unchanged since last query evaluation.

### Persistence

Log-structured persistence, modeled after SpacetimeDB / Redis AOF+RDB:

- **Commit log (WAL)** — source of truth on disk. Every mutation serialized and appended sequentially. Append-only, no random IO.
- **Snapshots** — periodic serialization of full world state. Recovery = load latest snapshot + replay subsequent log entries.
- **Not mmap** — direct memory-mapped archetype storage creates crash consistency, migration, and TLB pressure problems.

All mutations route through a `ChangeSet` abstraction, which is the natural serialization boundary for log entries.

### Serialization

| Layer | Format | Rationale |
|-------|--------|-----------|
| In-process systems | None | Typed references directly into archetype storage. Hot path never serializes. |
| Commit log & client sync | bincode (v1) | Compact, fast, serde-native. No schema in payload — both sides share Rust type definitions. |
| Snapshots (future) | rkyv | Zero-copy deserialization. Archived layout matches `BlobVec` layout, so snapshot load = mmap + pointer setup. |

Serialization is behind a `WireFormat` trait for swappability. bincode has no schema evolution story — acceptable during development (wipe on schema change), needs a migration strategy before shipping persistent data.

### Replication & Sync

Falls out naturally from the commit log:

- **Read replicas** — ship log entries to a second process, replay into its own world.
- **Client sync** — send filtered log entries (only entities the client can see). Client maintains a local ECS mirror.
- **Time-travel debugging** — replay log to tick N and stop.

## Build Roadmap

1. ~~`BlobVec` — type-erased aligned column storage~~
2. ~~`Archetype` — collection of `BlobVec` columns keyed by `ComponentId`~~
3. ~~`World` — entity allocator + archetype storage + archetype graph edges~~
4. ~~`Query<(A, B, ...)>` — tuple trait impls with bitset matching~~
5. ~~`#[derive(Table)]` — proc macro for compile-time schema registration~~
6. ~~`ChangeSet` — mutation abstraction (enables command buffers, persistence, replication)~~
7. ~~`Changed<T>` — per-column tick tracking, archetype-level change detection~~
8. ~~Secondary index hooks (`SpatialIndex` trait)~~
9. Commit log + snapshot persistence
10. Indexes (B-tree, hash)
11. Transaction isolation / MVCC
12. rkyv zero-copy snapshots

## Design Principles

- **Archetypes are an optimization, not a data model.** SpacetimeDB proves you can get excellent performance with fixed schemas and indexes. The archetype system is the tax for runtime flexibility — make it optional.
- **Two-tier everything.** Static (compile-time known) paths for database-like use, dynamic paths for game-like use. Same underlying storage, different access patterns.
- **Persistence is a log, not a flush.** Never serialize the world on every write. Append to a log, snapshot periodically.
- **Query planning is cheap because it barely runs.** Cache aggressively, re-evaluate incrementally.

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
