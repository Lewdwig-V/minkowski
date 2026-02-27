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

// ── Main ────────────────────────────────────────────────────────────

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

        // Step 3: Force accumulation (parallel)
        let forces: Vec<(Entity, Vec2)> = {
            use rayon::prelude::*;
            snapshot.par_iter().map(|&(entity, pos, vel)| {
                let mut sep = Vec2::ZERO;
                let mut ali = Vec2::ZERO;
                let mut coh = Vec2::ZERO;
                let mut sep_count = 0u32;
                let mut ali_count = 0u32;
                let mut coh_count = 0u32;

                for &(_other_e, other_pos, other_vel) in &snapshot {
                    let diff = other_pos - pos;
                    let dist_sq = diff.length_sq();
                    if dist_sq < 1e-6 { continue; }

                    let dist = dist_sq.sqrt();

                    if dist < params.separation_radius {
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

                (entity, force.clamped(params.max_force))
            }).collect()
        };

        // Step 4: Apply forces
        for &(entity, force) in &forces {
            if let Some(acc) = world.get_mut::<Acceleration>(entity) {
                acc.0 = acc.0 + force;
            }
        }

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

        // Step 6: Spawn/despawn churn — TODO
        // Step 7: Stats — TODO

        let _ = (frame, frame_start, &snapshot);
    }

    println!("boids: {} frames complete", FRAME_COUNT);
}
