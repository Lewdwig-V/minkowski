# Boids Simulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `examples/boids.rs` with a proper flocking simulation that exercises every ECS code path and serves as both a correctness validation tool and performance baseline.

**Architecture:** Self-contained example with local Vec2 math type, BoidParams config struct, 7-step frame loop (zero, snapshot, parallel forces, apply, integrate, churn, stats). Brute-force N² neighbor search parallelized via `par_for_each`. Random entity churn every 100 frames tests spawn/despawn/recycling paths.

**Tech Stack:** Rust 2021, minkowski ECS (local crate), fastrand (dev-dependency), rayon (via minkowski's par_for_each), std::time::Instant for frame timing.

**Design doc:** `docs/plans/2026-02-27-boids-sim-design.md`

---

## Task 1: Add fastrand dev-dependency

**Files:**
- Modify: `crates/minkowski/Cargo.toml`

**Step 1: Add fastrand to dev-dependencies**

In `crates/minkowski/Cargo.toml`, add `fastrand = "2"` to the `[dev-dependencies]` section:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
hecs = "0.10"
fastrand = "2"
```

**Step 2: Verify it resolves**

Run: `cargo check -p minkowski`
Expected: compiles (fastrand downloaded and resolved)

**Step 3: Commit**

```bash
git add crates/minkowski/Cargo.toml
git commit -m "chore: add fastrand dev-dependency for boids example"
```

---

## Task 2: Implement Vec2 and BoidParams

**Files:**
- Modify: `crates/minkowski/examples/boids.rs` (full rewrite)

This task writes the foundational types. The existing `boids.rs` is completely replaced.

**Step 1: Write Vec2 type and BoidParams**

Replace the entire contents of `crates/minkowski/examples/boids.rs` with:

```rust
//! Boids flocking simulation — exercises every minkowski ECS code path.
//!
//! Run: cargo run -p minkowski --example boids --release
//!
//! Exercises: spawn, despawn, multi-component queries, mutation,
//! parallel iteration, deferred commands, archetype stability under churn.

use std::time::Instant;
use minkowski::{Entity, World, CommandBuffer};

// ── Vec2 ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-8 {
            Self::ZERO
        } else {
            Self { x: self.x / len, y: self.y / len }
        }
    }

    fn clamped(self, max_len: f32) -> Self {
        let len_sq = self.length_sq();
        if len_sq > max_len * max_len {
            self.normalized() * max_len
        } else {
            self
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y } }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) { self.x += rhs.x; self.y += rhs.y; }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y } }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self { Self { x: self.x * rhs, y: self.y * rhs } }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self { Self { x: self.x / rhs, y: self.y / rhs } }
}

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Position(Vec2);

#[derive(Clone, Copy)]
struct Velocity(Vec2);

#[derive(Clone, Copy)]
struct Acceleration(Vec2);

// ── Parameters ──────────────────────────────────────────────────────

struct BoidParams {
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
    max_speed: f32,
    max_force: f32,
    world_size: f32,
}

impl Default for BoidParams {
    fn default() -> Self {
        Self {
            separation_radius: 25.0,
            alignment_radius: 50.0,
            cohesion_radius: 50.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            max_speed: 4.0,
            max_force: 0.1,
            world_size: 500.0,
        }
    }
}

fn main() {
    println!("boids: types defined, simulation not yet implemented");
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p minkowski --example boids`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat(boids): add Vec2 math type and BoidParams configuration"
```

---

## Task 3: Implement spawn and frame loop skeleton

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Replace the `main()` function with the full frame loop**

Replace `fn main()` with:

```rust
const ENTITY_COUNT: usize = 5_000;
const FRAME_COUNT: usize = 1_000;
const CHURN_INTERVAL: usize = 100;
const CHURN_COUNT: usize = 50;
const DT: f32 = 0.016;

fn spawn_boid(world: &mut World, params: &BoidParams) -> Entity {
    let x = fastrand::f32() * params.world_size;
    let y = fastrand::f32() * params.world_size;
    let angle = fastrand::f32() * std::f32::consts::TAU;
    let speed = fastrand::f32() * params.max_speed;
    world.spawn((
        Position(Vec2::new(x, y)),
        Velocity(Vec2::new(angle.cos() * speed, angle.sin() * speed)),
        Acceleration(Vec2::ZERO),
    ))
}

fn main() {
    let params = BoidParams::default();
    let mut world = World::new();

    // Spawn initial boids
    for _ in 0..ENTITY_COUNT {
        spawn_boid(&mut world, &params);
    }

    for frame in 0..FRAME_COUNT {
        let frame_start = Instant::now();

        // Step 1: Zero accelerations
        for acc in world.query::<&mut Acceleration>() {
            acc.0 = Vec2::ZERO;
        }

        // Step 2: Snapshot for neighbor queries
        let snapshot: Vec<(Entity, Vec2, Vec2)> = world
            .query::<(Entity, &Position, &Velocity)>()
            .map(|(e, p, v)| (e, p.0, v.0))
            .collect();

        // Step 3: Force accumulation (parallel) — Task 4
        // Step 4: Apply forces — Task 4

        // Step 5: Integration
        for (vel, acc) in world.query::<(&mut Velocity, &Acceleration)>() {
            vel.0 = vel.0 + acc.0 * DT;
            vel.0 = vel.0.clamped(params.max_speed);
        }
        for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
            pos.0 = pos.0 + vel.0 * DT;
            pos.0.x = pos.0.x.rem_euclid(params.world_size);
            pos.0.y = pos.0.y.rem_euclid(params.world_size);
        }

        // Step 6: Spawn/despawn churn — Task 5
        // Step 7: Stats — Task 6

        let _ = (frame, frame_start, &snapshot);
    }

    println!("boids: {} frames complete", FRAME_COUNT);
}
```

**Step 2: Verify it compiles and runs**

Run: `cargo run -p minkowski --example boids --release`
Expected: prints "boids: 1000 frames complete" (fast, since force computation is stubbed)

**Step 3: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat(boids): implement spawn and frame loop skeleton with integration"
```

---

## Task 4: Implement parallel force computation

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

This is the core boids logic — N² neighbor search with separation, alignment, cohesion.

**Step 1: Add force computation and apply**

Replace the two placeholder comments (Steps 3 and 4) with:

```rust
        // Step 3: Force accumulation (parallel)
        let forces: Vec<(Entity, Vec2)> = {
            use std::sync::Mutex;
            let results = Mutex::new(Vec::with_capacity(snapshot.len()));

            snapshot.iter().for_each(|&(entity, pos, vel)| {
                let mut sep = Vec2::ZERO;
                let mut ali = Vec2::ZERO;
                let mut coh = Vec2::ZERO;
                let mut sep_count = 0u32;
                let mut ali_count = 0u32;
                let mut coh_count = 0u32;

                for &(_other_e, other_pos, other_vel) in &snapshot {
                    let diff = other_pos - pos;
                    let dist_sq = diff.length_sq();
                    if dist_sq < 1e-6 { continue; } // skip self

                    let dist = dist_sq.sqrt();

                    if dist < params.separation_radius {
                        // Repel: weighted inversely by distance
                        sep = sep - diff.normalized() * (1.0 / dist);
                        sep_count += 1;
                    }
                    if dist < params.alignment_radius {
                        ali = ali + other_vel;
                        ali_count += 1;
                    }
                    if dist < params.cohesion_radius {
                        coh = coh + other_pos;
                        coh_count += 1;
                    }
                }

                let mut force = Vec2::ZERO;

                if sep_count > 0 {
                    force += sep / sep_count as f32 * params.separation_weight;
                }
                if ali_count > 0 {
                    let desired = ali / ali_count as f32 - vel;
                    force += desired * params.alignment_weight;
                }
                if coh_count > 0 {
                    let center = coh / coh_count as f32;
                    let desired = center - pos;
                    force += desired * params.cohesion_weight;
                }

                let force = force.clamped(params.max_force);
                results.lock().unwrap().push((entity, force));
            });

            results.into_inner().unwrap()
        };

        // Step 4: Apply forces
        for (entity, force) in &forces {
            if let Some(acc) = world.get_mut::<Acceleration>(*entity) {
                acc.0 = acc.0 + *force;
            }
        }
```

Note: This uses sequential iteration initially (`.iter().for_each`). We switch to parallel in the next step after verifying correctness.

**Step 2: Verify it compiles and runs with reasonable physics**

Run: `cargo run -p minkowski --example boids --release`
Expected: completes 1000 frames without panic. Add a temporary print at the end:

Add before the final println:
```rust
    // Temporary: verify physics didn't explode
    let mut speed_sum = 0.0f32;
    let count = world.query::<&Velocity>().count();
    for vel in world.query::<&Velocity>() {
        speed_sum += vel.0.length();
    }
    println!("final avg_vel: {:.3}", speed_sum / count as f32);
```

Expected: avg_vel between 0.5 and 4.0 (nonzero, not exploding)

**Step 3: Switch to parallel iteration**

Replace `snapshot.iter().for_each(|&(entity, pos, vel)| {` with rayon parallel:

```rust
        let forces: Vec<(Entity, Vec2)> = {
            use rayon::prelude::*;
            snapshot.par_iter().map(|&(entity, pos, vel)| {
                // ... same force computation body ...
                (entity, force)
            }).collect()
        };
```

The full replacement: change the force block from using `Mutex<Vec<>>` with `iter().for_each` to using `par_iter().map().collect()` which returns `Vec<(Entity, Vec2)>` directly. This is cleaner and avoids the mutex.

**Step 4: Remove the temporary print, verify parallel version runs**

Run: `cargo run -p minkowski --example boids --release`
Expected: completes 1000 frames

**Step 5: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat(boids): implement parallel N² force computation with separation/alignment/cohesion"
```

---

## Task 5: Implement spawn/despawn churn

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Replace the churn placeholder comment**

Replace `// Step 6: Spawn/despawn churn — Task 5` with:

```rust
        // Step 6: Spawn/despawn churn
        if frame > 0 && frame % CHURN_INTERVAL == 0 {
            // Collect all entities
            let entities: Vec<Entity> = world.query::<Entity>().collect();
            let count = entities.len();

            // Despawn CHURN_COUNT random entities via CommandBuffer
            let mut cmds = CommandBuffer::new();
            for _ in 0..CHURN_COUNT.min(count) {
                let idx = fastrand::usize(..count);
                cmds.despawn(entities[idx]);
            }
            cmds.apply(&mut world);

            // Spawn fresh boids to maintain population
            let current = world.query::<&Position>().count();
            let deficit = ENTITY_COUNT.saturating_sub(current);
            for _ in 0..deficit {
                spawn_boid(&mut world, &params);
            }
        }
```

**Step 2: Verify entity count stays stable**

Run: `cargo run -p minkowski --example boids --release`
Add temporary print inside the churn block after spawning:
```rust
            println!("frame {frame}: churned — despawned then spawned back to {}", world.query::<&Position>().count());
```

Expected: entity count returns to 5000 (or close — some despawns may hit the same entity twice via random selection, which is fine).

**Step 3: Remove temporary print, commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat(boids): implement random spawn/despawn churn every 100 frames"
```

---

## Task 6: Implement stats output and final polish

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Replace the stats placeholder comment**

Replace `// Step 7: Stats — Task 6` with:

```rust
        // Step 7: Stats
        if frame % CHURN_INTERVAL == 0 || frame == FRAME_COUNT - 1 {
            let entity_count = world.query::<&Position>().count();
            let mut speed_sum = 0.0f32;
            for vel in world.query::<&Velocity>() {
                speed_sum += vel.0.length();
            }
            let avg_vel = if entity_count > 0 { speed_sum / entity_count as f32 } else { 0.0 };
            let dt_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "frame {:04} | entities: {:>5} | avg_vel: {:.2} | dt: {:.1}ms",
                frame, entity_count, avg_vel, dt_ms,
            );
        }
```

Also remove the old `println!("boids: {} frames complete", FRAME_COUNT);` at the end and replace with:

```rust
    println!("Done.");
```

And remove the `let _ = (frame, frame_start, &snapshot);` line (it was a placeholder to suppress warnings).

**Step 2: Run the full simulation**

Run: `cargo run -p minkowski --example boids --release`
Expected output (approximately):
```
frame 0000 | entities:  5000 | avg_vel: 2.10 | dt: 42.3ms
frame 0100 | entities:  5000 | avg_vel: 2.85 | dt: 38.1ms
frame 0200 | entities:  5000 | avg_vel: 3.12 | dt: 37.5ms
...
frame 0999 | entities:  5000 | avg_vel: 3.41 | dt: 36.8ms
Done.
```

Healthy indicators:
- Entity count stable at ~5000
- avg_vel between 1.0 and 4.0 (nonzero, not exploding, not collapsing to zero)
- dt stable (no growing memory leak or fragmentation)

**Step 3: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat(boids): add frame stats output — entity count, avg velocity, frame time"
```

---

## Final Verification

After all tasks complete:

1. **Full test suite**: `cargo test -p minkowski --lib` — all 60 tests still pass (example doesn't break lib)
2. **Boids runs**: `cargo run -p minkowski --example boids --release` — 1000 frames, stable entity count, nonzero avg_vel, stable frame time
3. **No warnings**: `cargo check -p minkowski --example boids 2>&1 | grep "^error"` — no errors
