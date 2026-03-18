//! Boids flocking simulation — exercises query reducers + spatial indexing.
//!
//! Run: cargo run -p minkowski-examples --example boids --release
//!
//! Exercises: spawn, despawn, query reducers (QueryMut), spatial grid (SpatialIndex),
//! deferred commands, archetype stability under churn.
//!
//! Each simulation step is a registered query reducer dispatched via `registry.run()`.
//! The spatial grid is rebuilt before the force-computation reducer runs; the reducer
//! captures the grid snapshot by shared reference for neighbor lookups.
//!
//! # Vectorization
//!
//! The integration loops (zero_accel and integrate) are designed to auto-vectorize.
//! LLVM generates branchless AVX-512 masked ops for the position/velocity
//! updates. Every link in this chain is required — remove any one and the
//! loop falls back to scalar:
//!
//! 1. **64-byte aligned columns** — BlobVec allocates with cache-line alignment.
//!    Without this, LLVM may not emit aligned loads or may add peel loops.
//! 2. **`for_each_chunk` yields `&[T]`/`&mut [T]` slices** — LLVM needs to see
//!    a contiguous slice with a known length to engage its loop vectorizer.
//!    Per-element `fetch(ptr, row)` across a trait boundary is opaque.
//! 3. **Index loop `for i in 0..len`** — iterating by index over the slice
//!    gives LLVM the simple induction variable it needs. Iterator adaptors
//!    can obscure the access pattern.
//! 4. **No opaque function calls in the loop body** — `rem_euclid` compiles
//!    to `fmodf` (scalar libm), which is an optimization barrier. The
//!    branchless conditional subtract wraps identically for bounded velocity
//!    and is fully inlineable.
//! 5. **`-C target-cpu=native`** (in `.cargo/config.toml`) — without this,
//!    rustc targets a baseline x86-64 that lacks AVX2/AVX-512.
//!
//! Verify with: `objdump -d -M intel target/release/examples/boids | grep vmulps`
//! You should see packed float ops (vaddps, vmulps, vblendmps), not scalar
//! (vaddss, vmulss).

use minkowski::{CommandBuffer, Entity, QueryMut, ReducerRegistry, SpatialIndex, World};
use std::time::Instant;

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
            Self {
                x: self.x / len,
                y: self.y / len,
            }
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
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

// ── Components ──────────────────────────────────────────────────────
//

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

// ── SpatialGrid ────────────────────────────────────────────────────

struct SpatialGrid {
    cell_size: f32,
    grid_w: usize,
    cells: Vec<Vec<usize>>,
    pub snapshot: Vec<(Entity, Vec2, Vec2)>,
}

impl SpatialGrid {
    fn new(cell_size: f32, world_size: f32) -> Self {
        let grid_w = (world_size / cell_size).ceil() as usize;
        Self {
            cell_size,
            grid_w,
            cells: Vec::new(),
            snapshot: Vec::new(),
        }
    }

    #[allow(clippy::cast_possible_wrap)]
    fn neighbors(&self, pos: Vec2) -> impl Iterator<Item = &(Entity, Vec2, Vec2)> {
        let cx = ((pos.x / self.cell_size) as usize).min(self.grid_w - 1);
        let cy = ((pos.y / self.cell_size) as usize).min(self.grid_w - 1);
        let grid_w = self.grid_w;
        (-1i32..=1).flat_map(move |dy| {
            (-1i32..=1).flat_map(move |dx| {
                let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                // SAFETY: nx < grid_w and ny < grid_w guaranteed by rem_euclid
                let cell = &self.cells[ny * grid_w + nx];
                cell.iter().map(|&j| &self.snapshot[j])
            })
        })
    }
}

impl SpatialIndex for SpatialGrid {
    fn rebuild(&mut self, world: &mut World) {
        self.snapshot = world
            .query::<(Entity, &Position, &Velocity)>()
            .map(|(e, p, v)| (e, p.0, v.0))
            .collect();

        self.cells.clear();
        self.cells.resize(self.grid_w * self.grid_w, Vec::new());
        for (i, &(_, pos, _)) in self.snapshot.iter().enumerate() {
            let cx = ((pos.x / self.cell_size) as usize).min(self.grid_w - 1);
            let cy = ((pos.y / self.cell_size) as usize).min(self.grid_w - 1);
            self.cells[cy * self.grid_w + cx].push(i);
        }
    }

    fn supports(&self, _expr: &minkowski::SpatialExpr) -> Option<minkowski::SpatialCost> {
        None
    }
    fn query(&self, _expr: &minkowski::SpatialExpr) -> Vec<Entity> {
        Vec::new()
    }
}

// ── Constants ───────────────────────────────────────────────────────

const ENTITY_COUNT: usize = 5_000;
const FRAME_COUNT: usize = 1_000;
const CHURN_INTERVAL: usize = 100;
const CHURN_COUNT: usize = 50;
const DT: f32 = 0.016;

// ── Helpers ─────────────────────────────────────────────────────────

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

/// Minimum-image distance on a toroidal world.
/// Returns the shortest signed difference between `a` and `b` wrapping at `world_size`.
fn wrapped_diff(a: f32, b: f32, world_size: f32) -> f32 {
    let d = a - b;
    if d > world_size * 0.5 {
        d - world_size
    } else if d < -world_size * 0.5 {
        d + world_size
    } else {
        d
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let params = BoidParams::default();
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    // Spawn initial boids
    for _ in 0..ENTITY_COUNT {
        spawn_boid(&mut world, &params);
    }

    let mut grid = SpatialGrid::new(params.cohesion_radius, params.world_size);

    // ── Register query reducers ─────────────────────────────────────

    // Step 1: Zero accelerations (chunk — enables vectorization)
    let zero_accel_id = registry
        .register_query::<(&mut Acceleration,), (), _>(
            &mut world,
            "zero_accel",
            |mut query: QueryMut<'_, (&mut Acceleration,)>, ()| {
                query.for_each(|(accs,)| {
                    for acc in accs.iter_mut() {
                        acc.0 = Vec2::ZERO;
                    }
                });
            },
        )
        .unwrap();

    // Step 2: Integrate — Euler step + velocity clamping + world wrapping
    let max_speed = params.max_speed;
    let ws = params.world_size;
    let integrate_id = registry
        .register_query::<(&mut Position, &mut Velocity, &Acceleration), f32, _>(
            &mut world,
            "integrate",
            move |mut query: QueryMut<'_, (&mut Position, &mut Velocity, &Acceleration)>,
                  dt: f32| {
                // Velocity integration + clamping
                query.for_each(|(poss, vels, accs)| {
                    for i in 0..vels.len() {
                        vels[i].0.x += accs[i].0.x * dt;
                        vels[i].0.y += accs[i].0.y * dt;
                        vels[i].0 = vels[i].0.clamped(max_speed);

                        // Position integration with branchless world wrapping.
                        let mut x = poss[i].0.x + vels[i].0.x * dt;
                        let mut y = poss[i].0.y + vels[i].0.y * dt;
                        if x >= ws {
                            x -= ws;
                        } else if x < 0.0 {
                            x += ws;
                        }
                        if y >= ws {
                            y -= ws;
                        } else if y < 0.0 {
                            y += ws;
                        }
                        poss[i].0.x = x;
                        poss[i].0.y = y;
                    }
                });
            },
        )
        .unwrap();

    for frame in 0..FRAME_COUNT {
        let frame_start = Instant::now();

        // Step 1: Zero accelerations via reducer
        registry.run(&mut world, zero_accel_id, ()).unwrap();

        // Step 2: Rebuild spatial grid
        grid.rebuild(&mut world);

        // Step 3: Compute boid forces from snapshot (sequential)
        // The grid snapshot contains (Entity, Position, Velocity) for all boids.
        // We iterate the snapshot, compute forces, and apply them directly.
        for idx in 0..grid.snapshot.len() {
            let (entity, pos, vel) = grid.snapshot[idx];

            let mut sep = Vec2::ZERO;
            let mut ali = Vec2::ZERO;
            let mut coh = Vec2::ZERO;
            let mut sep_count = 0u32;
            let mut ali_count = 0u32;
            let mut coh_count = 0u32;

            for &(_, other_pos, other_vel) in grid.neighbors(pos) {
                let diff = Vec2::new(
                    wrapped_diff(other_pos.x, pos.x, params.world_size),
                    wrapped_diff(other_pos.y, pos.y, params.world_size),
                );
                let dist_sq = diff.length_sq();
                if dist_sq < 1e-6 {
                    continue;
                }

                let dist = dist_sq.sqrt();

                if dist < params.separation_radius {
                    sep = sep - diff.normalized() * (1.0 / dist);
                    sep_count += 1;
                }
                if dist < params.alignment_radius {
                    ali += other_vel;
                    ali_count += 1;
                }
                if dist < params.cohesion_radius {
                    coh += diff;
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
                let desired = coh / coh_count as f32;
                force += desired * params.cohesion_weight;
            }

            let force = force.clamped(params.max_force);
            if let Some(acc) = world.get_mut::<Acceleration>(entity) {
                acc.0 += force;
            }
        }

        // Step 4: Integrate via reducer
        registry.run(&mut world, integrate_id, DT).unwrap();

        // Step 5: Spawn/despawn churn
        if frame > 0 && frame % CHURN_INTERVAL == 0 {
            let entities: Vec<Entity> = world.query::<Entity>().collect();
            let count = entities.len();

            let mut cmds = CommandBuffer::new();
            for _ in 0..CHURN_COUNT.min(count) {
                let idx = fastrand::usize(..count);
                cmds.despawn(entities[idx]);
            }
            cmds.apply(&mut world).unwrap();

            let current = world.query::<&Position>().count();
            let deficit = ENTITY_COUNT.saturating_sub(current);
            for _ in 0..deficit {
                spawn_boid(&mut world, &params);
            }
        }
        // Step 6: Stats
        if frame % CHURN_INTERVAL == 0 || frame == FRAME_COUNT - 1 {
            let entity_count = world.query::<&Position>().count();
            let mut speed_sum = 0.0f32;
            for vel in world.query::<&Velocity>() {
                speed_sum += vel.0.length();
            }
            let avg_vel = if entity_count > 0 {
                speed_sum / entity_count as f32
            } else {
                0.0
            };
            let dt_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "frame {:04} | entities: {:>5} | avg_vel: {:.2} | dt: {:.1}ms",
                frame, entity_count, avg_vel, dt_ms,
            );
        }
    }

    println!("Done.");
}
