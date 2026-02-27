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

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
