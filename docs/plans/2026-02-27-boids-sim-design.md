# Boids Simulation — Design Document

## Goal

Replace the minimal `examples/boids.rs` with a proper flocking simulation that exercises every ECS code path: spawn, despawn, multi-component queries, mutation, parallel iteration, deferred commands, and archetype stability under entity churn. Serves as both a correctness validation tool and a performance baseline.

## Components

```rust
struct Position(Vec2);      // wraps [f32; 2]
struct Velocity(Vec2);
struct Acceleration(Vec2);  // zeroed each frame, accumulated by rules
```

All boids share one archetype: `(Position, Velocity, Acceleration)`. Vec2 is a local struct in the example — `[f32; 2]` with basic math ops (`Add`, `Sub`, `Mul`, `length()`, `normalized()`). No external dependency.

## Parameters

```rust
struct BoidParams {
    separation_radius: f32,    // 25.0
    alignment_radius: f32,     // 50.0
    cohesion_radius: f32,      // 50.0
    separation_weight: f32,    // 1.5
    alignment_weight: f32,     // 1.0
    cohesion_weight: f32,      // 1.0
    max_speed: f32,            // 4.0
    max_force: f32,            // 0.1
    world_size: f32,           // 500.0
}
```

Passed as a plain struct to system functions. Separation radius is smaller than alignment/cohesion — this creates the characteristic flocking behavior where entities repel very close neighbors but steer toward the broader group's heading and center.

## Frame Loop

Each frame executes 7 steps in order:

### 1. Zero accelerations
`for acc in query::<&mut Acceleration>()` → set to `Vec2::ZERO`. Prevents stale forces from accumulating across frames.

### 2. Snapshot
Collect `Vec<(Entity, Vec2, Vec2)>` of `(entity, position, velocity)`. This is the read-only neighbor data used by the force computation. O(N) allocation per frame — trivial compared to the O(N²) neighbor search.

### 3. Force accumulation (parallel)
`par_for_each` over the snapshot. For each boid, brute-force N² neighbor search:
- **Separation**: neighbors within `separation_radius` → accumulate repulsion vector (weighted)
- **Alignment**: neighbors within `alignment_radius` → steer toward average velocity (weighted)
- **Cohesion**: neighbors within `cohesion_radius` → steer toward average position (weighted)

Sum the three forces, clamp magnitude to `max_force`. Write results to a shared `Vec<(Entity, Vec2)>` (using `Mutex` or per-thread collection + merge).

### 4. Apply forces
Iterate force results, `world.get_mut::<Acceleration>(entity)` and add the clamped force.

### 5. Integration
- Query `(&mut Velocity, &Acceleration)` → `vel += acc * dt`, clamp speed to `max_speed`
- Query `(&mut Position, &Velocity)` → `pos += vel * dt`, wrap at world boundaries using `rem_euclid`

### 6. Spawn/despawn cycle (every 100 frames)
- Collect entity list from `query::<Entity>()`
- Pick 50 random indices using `fastrand::usize(0..count)`, despawn via `CommandBuffer`
- Apply commands, then spawn 50 fresh boids at random positions with random velocities

This tests: entity allocator recycling, archetype swap-remove with unpredictable removal patterns, deferred mutation, and that entity count stays stable.

### 7. Stats (every 100 frames)
Print: `frame NNNN | entities: N | avg_vel: N.NN | dt: N.Nms | archetypes: N`

Healthy indicators: entity count stable at 5000, avg_vel between 1.0-4.0, dt stable, archetypes = 1.

## What This Tests

| ECS path | How exercised |
|---|---|
| `world.spawn()` | Fresh boids every 100 frames |
| `world.despawn()` | Random despawns via CommandBuffer |
| `world.get_mut()` | Applying forces to individual entities |
| `query::<&T>()` | Snapshot collection, stats |
| `query::<&mut T>()` | Zero pass, integration |
| `query::<(A, B)>()` | Multi-component queries throughout |
| `par_for_each` | Force computation (hot loop) |
| `CommandBuffer` | Deferred despawns |
| Entity recycling | Spawn after despawn reuses indices |
| Archetype stability | Churn doesn't create new archetypes |
| Swap-remove correctness | Random removal pattern |

## Configuration

- **Entity count**: 5,000 (N²=25M distance checks — meaningful workload, <50ms/frame in release)
- **Frames**: 1,000
- **Churn**: every 100 frames, despawn/spawn 50 entities (1%)
- **DT**: 0.016 (60fps equivalent)

## Dependencies

- `fastrand` as dev-dependency (example-only, lightweight RNG)

## Stretch Goals (not in scope)

- `Flock(u8)` component with per-flock params → multi-archetype `Option<&Flock>` queries
- Obstacle entities → cross-archetype separation queries
- `--dump` CSV position output

## Explicit Non-Goals

- No graphical output
- No spatial index (N² is fine at 5K)
- No system scheduler or resource abstraction
