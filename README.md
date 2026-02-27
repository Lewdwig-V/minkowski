# minkowski

A column-oriented archetype ECS built from scratch in Rust. Game workloads first, database features later.

## What's here (Phase 1)

The foundational storage layer: type-erased BlobVec columns packed into archetypes, generational entity IDs, parallel query iteration via rayon, and deferred mutation through CommandBuffer.

```rust
use minkowski::{World, Entity, CommandBuffer};

struct Position { x: f32, y: f32 }
struct Velocity { dx: f32, dy: f32 }

let mut world = World::new();

// Spawn entities into archetypes
let e = world.spawn((Position { x: 0.0, y: 0.0 }, Velocity { dx: 1.0, dy: 0.0 }));

// Query and mutate
for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
    pos.x += vel.dx;
    pos.y += vel.dy;
}

// Parallel iteration
world.query::<(&mut Position, &Velocity)>().par_for_each(|(pos, vel)| {
    pos.x += vel.dx;
});

// Archetype migration
world.insert(e, Health(100));   // moves entity to new archetype
world.remove::<Health>(e);      // moves it back

// Deferred mutation during iteration
let mut cmds = CommandBuffer::new();
for (entity, pos) in world.query::<(Entity, &Position)>() {
    if pos.x > 100.0 {
        cmds.despawn(entity);
    }
}
cmds.apply(&mut world);
```

### Storage design

Each unique combination of component types gets an **archetype** — a struct of arrays where each component type is a `BlobVec` (type-erased growable byte array). Queries match archetypes via `FixedBitSet` subset checks, then iterate columns with raw pointer arithmetic. No virtual dispatch in the hot loop.

**Entity** = u64 with 32-bit index + 32-bit generation. Recycled indices get bumped generations to prevent use-after-free. O(1) lookup from entity to archetype row via `Vec<Option<EntityLocation>>`.

**Sparse components** (opt-in `HashMap<Entity, T>`) for tags and rarely-present data. Dense archetype storage is the default.

### Boids example

A flocking simulation that exercises every ECS code path — spawn, despawn, multi-component queries, mutation, parallel iteration, deferred commands, and archetype stability under entity churn.

```
$ cargo run -p minkowski --example boids --release

frame 0000 | entities:  5000 | avg_vel: 1.99 | dt: 9.9ms
frame 0100 | entities:  5000 | avg_vel: 1.94 | dt: 7.2ms
frame 0200 | entities:  5000 | avg_vel: 1.89 | dt: 7.8ms
...
frame 0999 | entities:  5000 | avg_vel: 1.89 | dt: 7.4ms
Done.
```

5,000 boids with brute-force N² neighbor search (separation, alignment, cohesion), parallel force computation, and random spawn/despawn churn every 100 frames.

### Benchmarks

Criterion benchmarks compare against [hecs](https://crates.io/crates/hecs):

```
$ cargo bench -p minkowski
```

Suites: `spawn` (10K entities), `iterate` (10K), `parallel` (100K vs sequential), `add_remove` (1K migration cycles), `fragmented` (20 archetypes).

## What's next (Phase 2+)

| Phase | Feature | Why |
|---|---|---|
| 2 | `#[derive(Component)]` proc macro | Compile-time schema validation, ergonomic derives |
| 2 | Query caching with generation tracking | Skip archetype re-scan when nothing changed |
| 3 | Change detection ticks | Systems only process entities that actually changed |
| 3 | Automatic system scheduling | Conflict detection, parallel system execution |
| 4 | Persistence — WAL + snapshots | Durable state via BlobVec memcpy to disk |
| 4 | Transaction semantics | Atomic multi-entity mutations with rollback |
| 5 | Query planning (Volcano model) | Optimize complex queries across indexes |
| 5 | B-tree / hash indexes | Fast range and equality lookups on component fields |

The architecture is designed so each phase layers on without rewriting the previous one. BlobVec's type-erased byte storage is already memcpy-friendly for snapshots. CommandBuffer's closure queue generalizes to ChangeSets for transactions.

## Building

```
cargo build            # debug
cargo build --release  # optimized
cargo test             # all tests
cargo bench            # benchmarks
```

Requires Rust 2021 edition. Dependencies: `rayon`, `fixedbitset`.

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
- **Dynamic queries** — arbitrary component combinations, walk the archetype graph with cached `QueryState` (re-evaluated only when new archetypes appear).

Matching uses **bitsets** — one per archetype, `(archetype_bits & query_bits) == query_bits`. Handles negation, optionals, and union queries with bitwise ops. Scales to hundreds of archetypes in microseconds.

Query planning uses a simple plan tree:

| Node | Purpose |
|------|---------|
| `Scan` | Iterate all rows in an archetype |
| `IndexScan` | B-tree range lookup on an indexed column |
| `Filter` | Row-level predicate evaluation |
| `Merge` | Concatenate results from multiple archetypes |

Plans are cached per-query. The planner runs rarely (query creation, new archetype events); the hot path is pointer chasing through pre-resolved column arrays.

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
- **Lazy archetype migration** — if a newly added component doesn't affect any active query, store it in sparse storage and defer the archetype move until actually needed.
- **Change detection** — per-column tick tracking (`Changed<T>`, `Added<T>`). Entire archetypes skipped when unchanged since last query evaluation.

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

1. `BlobVec` — type-erased aligned column storage
2. `Archetype` — collection of `BlobVec` columns keyed by `ComponentId`
3. `World` — entity allocator + archetype storage + archetype graph edges
4. `Query<(A, B, ...)>` — tuple trait impls with bitset matching
5. `#[derive(Table)]` — proc macro for compile-time schema registration
6. `ChangeSet` — mutation abstraction (enables command buffers, persistence, replication)
7. Commit log + snapshot persistence
8. Indexes (B-tree, hash)
9. Transaction isolation / MVCC
10. rkyv zero-copy snapshots

## Design Principles

- **Archetypes are an optimization, not a data model.** SpacetimeDB proves you can get excellent performance with fixed schemas and indexes. The archetype system is the tax for runtime flexibility — make it optional.
- **Two-tier everything.** Static (compile-time known) paths for database-like use, dynamic paths for game-like use. Same underlying storage, different access patterns.
- **Persistence is a log, not a flush.** Never serialize the world on every write. Append to a log, snapshot periodically.
- **Query planning is cheap because it barely runs.** Cache aggressively, re-evaluate incrementally.

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
