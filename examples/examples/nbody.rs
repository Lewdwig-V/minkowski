//! Barnes-Hut N-body gravity simulation — exercises the SpatialIndex trait
//! with a quadtree, a fundamentally different spatial structure from the
//! uniform grid used in the boids example.
//!
//! Run: cargo run -p minkowski-examples --example nbody --release
//!
//! Exercises: spawn, despawn, multi-component queries, mutation,
//! parallel iteration (rayon), deferred commands, SpatialIndex trait,
//! generational entity validation, archetype stability under churn,
//! query reducers (integration step).
//!
//! The Barnes-Hut algorithm approximates distant clusters of bodies as
//! single point masses, reducing the force computation from O(N^2) to
//! O(N log N). The `theta` parameter controls the accuracy/speed tradeoff:
//! lower theta = more accurate but slower.
//!
//! Force computation uses snapshot-based parallel rayon iteration (the
//! interesting part of the demo). Integration is dispatched through a
//! query reducer to demonstrate the pattern alongside spatial indexing.

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

// ── Toroidal helpers ─────────────────────────────────────────────────

/// Minimum-image signed difference on a toroidal domain of size `world_size`.
/// Returns the shortest path from `b` to `a` wrapping at boundaries.
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

/// Minimum-image vector from `from` to `to` on a toroidal world.
fn wrapped_offset(to: Vec2, from: Vec2, world_size: f32) -> Vec2 {
    Vec2::new(
        wrapped_diff(to.x, from.x, world_size),
        wrapped_diff(to.y, from.y, world_size),
    )
}

// ── Rect (axis-aligned bounding box) ────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Returns 0-3 for which child quadrant the point falls in.
    /// Layout: 0=NW, 1=NE, 2=SW, 3=SE (top-left origin).
    fn quadrant(&self, point: Vec2) -> usize {
        let mx = self.x + self.w * 0.5;
        let my = self.y + self.h * 0.5;
        let east = if point.x >= mx { 1 } else { 0 };
        let south = if point.y >= my { 2 } else { 0 };
        east | south
    }

    /// Returns the sub-rectangle for the given quadrant (0-3).
    fn child(&self, quadrant: usize) -> Rect {
        let hw = self.w * 0.5;
        let hh = self.h * 0.5;
        let dx = if quadrant & 1 != 0 { hw } else { 0.0 };
        let dy = if quadrant & 2 != 0 { hh } else { 0.0 };
        Rect::new(self.x + dx, self.y + dy, hw, hh)
    }
}

// ── QuadNode ────────────────────────────────────────────────────────

struct QuadNode {
    bounds: Rect,
    center_of_mass: Vec2,
    total_mass: f32,
    entity: Option<(Entity, Vec2, f32)>, // leaf: entity, position, mass
    children: Option<[usize; 4]>,        // internal: indices into nodes vec
}

impl QuadNode {
    fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            center_of_mass: Vec2::ZERO,
            total_mass: 0.0,
            entity: None,
            children: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.entity.is_none() && self.children.is_none()
    }
}

// ── BarnesHutTree ───────────────────────────────────────────────────

struct BarnesHutTree {
    nodes: Vec<QuadNode>,
    bounds: Rect,
    theta: f32,
    world_size: f32,
}

const MAX_DEPTH: usize = 20;

impl BarnesHutTree {
    fn new(bounds: Rect, theta: f32, world_size: f32) -> Self {
        Self {
            nodes: vec![QuadNode::new(bounds)],
            bounds,
            theta,
            world_size,
        }
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(QuadNode::new(self.bounds));
    }

    fn insert(&mut self, node_idx: usize, entity: Entity, pos: Vec2, mass: f32, depth: usize) {
        // Clamp position to bounds to handle floating-point edge cases
        let clamped = Vec2::new(
            pos.x
                .max(self.nodes[node_idx].bounds.x)
                .min(self.nodes[node_idx].bounds.x + self.nodes[node_idx].bounds.w - f32::EPSILON),
            pos.y
                .max(self.nodes[node_idx].bounds.y)
                .min(self.nodes[node_idx].bounds.y + self.nodes[node_idx].bounds.h - f32::EPSILON),
        );

        if depth >= MAX_DEPTH {
            // At max depth, accumulate mass and center of mass directly
            // instead of subdividing further. This handles coincident or
            // near-coincident positions without losing any bodies.
            let node = &mut self.nodes[node_idx];
            let ws = self.world_size;
            if node.total_mass > 0.0 {
                // Weighted merge using wrapped offset from existing center
                let offset = wrapped_offset(clamped, node.center_of_mass, ws);
                let new_mass = node.total_mass + mass;
                let shift = offset * (mass / new_mass);
                node.center_of_mass.x = (node.center_of_mass.x + shift.x).rem_euclid(ws);
                node.center_of_mass.y = (node.center_of_mass.y + shift.y).rem_euclid(ws);
                node.total_mass = new_mass;
            } else {
                node.center_of_mass = clamped;
                node.total_mass = mass;
            }
            // Clear entity — this node is now an aggregate bucket, not a
            // single-entity leaf. compute_force will use total_mass/center_of_mass.
            node.entity = None;
            return;
        }

        if self.nodes[node_idx].is_empty() {
            // Empty leaf — store the entity here
            self.nodes[node_idx].entity = Some((entity, clamped, mass));
            return;
        }

        if self.nodes[node_idx].children.is_some() {
            // Internal node — recurse into the appropriate child
            let q = self.nodes[node_idx].bounds.quadrant(clamped);
            let child_idx = self.nodes[node_idx].children.unwrap()[q];
            self.insert(child_idx, entity, clamped, mass, depth + 1);
            return;
        }

        // Leaf with an existing entity — split into 4 children and redistribute
        let existing = self.nodes[node_idx].entity.take().unwrap();
        let bounds = self.nodes[node_idx].bounds;

        // Create 4 child nodes
        let base = self.nodes.len();
        for q in 0..4 {
            self.nodes.push(QuadNode::new(bounds.child(q)));
        }
        self.nodes[node_idx].children = Some([base, base + 1, base + 2, base + 3]);

        // Re-insert the existing entity
        let eq = bounds.quadrant(existing.1);
        self.insert(base + eq, existing.0, existing.1, existing.2, depth + 1);

        // Insert the new entity
        let nq = bounds.quadrant(clamped);
        self.insert(base + nq, entity, clamped, mass, depth + 1);
    }

    fn aggregate(&mut self, node_idx: usize) {
        let ws = self.world_size;

        if let Some(children) = self.nodes[node_idx].children {
            // Internal node — recurse into children first
            for &child_idx in &children {
                self.aggregate(child_idx);
            }

            // Compute center of mass using a reference-point technique to
            // handle toroidal wrapping. Pick the first non-empty child's
            // center as the reference, compute wrapped offsets from it,
            // average, then shift back.
            let mut total_mass = 0.0f32;
            let mut ref_pos = Vec2::ZERO;
            let mut have_ref = false;
            let mut weighted_offset = Vec2::ZERO;

            for &child_idx in &children {
                let child = &self.nodes[child_idx];
                if child.total_mass > 0.0 {
                    if !have_ref {
                        ref_pos = child.center_of_mass;
                        have_ref = true;
                    }
                    let offset = wrapped_offset(child.center_of_mass, ref_pos, ws);
                    weighted_offset += offset * child.total_mass;
                    total_mass += child.total_mass;
                }
            }

            self.nodes[node_idx].total_mass = total_mass;
            if total_mass > 0.0 {
                let avg_offset = weighted_offset / total_mass;
                let mut com = ref_pos + avg_offset;
                // Re-wrap into [0, world_size)
                com.x = com.x.rem_euclid(ws);
                com.y = com.y.rem_euclid(ws);
                self.nodes[node_idx].center_of_mass = com;
            }
        } else if let Some((_, pos, mass)) = self.nodes[node_idx].entity {
            // Leaf with entity
            self.nodes[node_idx].center_of_mass = pos;
            self.nodes[node_idx].total_mass = mass;
        }
        // Empty leaf: total_mass stays 0
    }

    fn compute_force(&self, pos: Vec2, mass: f32, node_idx: usize) -> Vec2 {
        let node = &self.nodes[node_idx];
        let ws = self.world_size;

        if node.total_mass <= 0.0 {
            return Vec2::ZERO;
        }

        if node.children.is_none() {
            // Leaf or max-depth aggregate bucket — use center_of_mass/total_mass
            // directly. For single-entity leaves these equal the entity's
            // position and mass (set during aggregate). For max-depth buckets
            // they hold the accumulated values from all coincident bodies.
            let diff = wrapped_offset(node.center_of_mass, pos, ws);
            let dist_sq = diff.length_sq();
            if dist_sq < 1e-6 {
                // Same position (self-interaction or coincident) — skip
                return Vec2::ZERO;
            }
            let force_mag = G * mass * node.total_mass / (dist_sq + SOFTENING * SOFTENING);
            return diff.normalized() * force_mag;
        }

        if let Some(children) = node.children {
            // Internal node — check Barnes-Hut criterion using wrapped distance
            let s = node.bounds.w.max(node.bounds.h);
            let diff = wrapped_offset(node.center_of_mass, pos, ws);
            let d = diff.length();

            if d > 0.0 && s / d < self.theta {
                // Far enough — treat as point mass
                let dist_sq = diff.length_sq();
                let force_mag = G * mass * node.total_mass / (dist_sq + SOFTENING * SOFTENING);
                diff.normalized() * force_mag
            } else {
                // Too close — recurse into children
                let mut force = Vec2::ZERO;
                for &child_idx in &children {
                    force += self.compute_force(pos, mass, child_idx);
                }
                force
            }
        } else {
            Vec2::ZERO
        }
    }
}

impl SpatialIndex for BarnesHutTree {
    fn rebuild(&mut self, world: &mut World) {
        // Compute bounding box encompassing all positions
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut count = 0u32;

        for pos in world.query::<&Position>() {
            min_x = min_x.min(pos.0.x);
            min_y = min_y.min(pos.0.y);
            max_x = max_x.max(pos.0.x);
            max_y = max_y.max(pos.0.y);
            count += 1;
        }

        if count == 0 {
            self.clear();
            return;
        }

        // Add a small margin to avoid edge-case clamping issues
        let margin = 1.0;
        let size = (max_x - min_x).max(max_y - min_y) + margin * 2.0;
        self.bounds = Rect::new(min_x - margin, min_y - margin, size, size);

        self.clear();

        // Insert all entities
        let entities: Vec<(Entity, Vec2, f32)> = world
            .query::<(Entity, &Position, &Mass)>()
            .map(|(e, p, m)| (e, p.0, m.0))
            .collect();

        for (entity, pos, mass) in entities {
            self.insert(0, entity, pos, mass, 0);
        }

        // Bottom-up aggregation pass
        self.aggregate(0);
    }
}

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Position(Vec2);

#[derive(Clone, Copy)]
struct Velocity(Vec2);

#[derive(Clone, Copy)]
struct Mass(f32);

// ── Constants ───────────────────────────────────────────────────────

const ENTITY_COUNT: usize = 2_000;
const FRAME_COUNT: usize = 1_000;
const CHURN_INTERVAL: usize = 200;
const CHURN_COUNT: usize = 20;
const DT: f32 = 0.001;
const G: f32 = 6.674e-2;
const SOFTENING: f32 = 1.0;
const THETA: f32 = 0.5;
const WORLD_SIZE: f32 = 500.0;

// ── Helpers ─────────────────────────────────────────────────────────

fn spawn_body(world: &mut World) -> Entity {
    let x = fastrand::f32() * WORLD_SIZE;
    let y = fastrand::f32() * WORLD_SIZE;
    let vx = (fastrand::f32() - 0.5) * 10.0;
    let vy = (fastrand::f32() - 0.5) * 10.0;
    world.spawn((
        Position(Vec2::new(x, y)),
        Velocity(Vec2::new(vx, vy)),
        Mass(1.0),
    ))
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    // ── Register query reducers ─────────────────────────────────────

    // Integration reducer: applies velocity to position with toroidal wrapping.
    // Receives DT as argument so the timestep can vary per call.
    let integrate_id = registry
        .register_query::<(&mut Position, &Velocity), f32, _>(
            &mut world,
            "integrate",
            |mut query: QueryMut<'_, (&mut Position, &Velocity)>, dt: f32| {
                let ws = WORLD_SIZE;
                query.for_each_chunk(|(poss, vels)| {
                    for i in 0..poss.len() {
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

    // Spawn initial bodies
    for _ in 0..ENTITY_COUNT {
        spawn_body(&mut world);
    }

    let initial_bounds = Rect::new(0.0, 0.0, WORLD_SIZE, WORLD_SIZE);
    let mut tree = BarnesHutTree::new(initial_bounds, THETA, WORLD_SIZE);

    for frame in 0..FRAME_COUNT {
        let frame_start = Instant::now();

        // Step 1: Rebuild quadtree (needs &mut world for queries)
        tree.rebuild(&mut world);

        // Step 2: Snapshot for parallel force computation
        let snapshot: Vec<(Entity, Vec2, f32)> = world
            .query::<(Entity, &Position, &Mass)>()
            .map(|(e, p, m)| (e, p.0, m.0))
            .collect();

        // Step 3: Parallel force computation using rayon
        let forces: Vec<(Entity, Vec2)> = {
            use rayon::prelude::*;
            snapshot
                .par_iter()
                .map(|&(entity, pos, mass)| {
                    let accel = tree.compute_force(pos, mass, 0) / mass;
                    (entity, accel)
                })
                .collect()
        };

        // Step 4: Apply forces — update velocities
        for &(entity, accel) in &forces {
            if let Some(vel) = world.get_mut::<Velocity>(entity) {
                vel.0 += accel * DT;
            }
        }

        // Step 5: Integration — update positions via query reducer
        registry.run(&mut world, integrate_id, DT).unwrap();

        // Step 6: Spawn/despawn churn
        if frame > 0 && frame % CHURN_INTERVAL == 0 {
            // Take a snapshot of entities BEFORE despawning for staleness demo
            let pre_churn_snapshot: Vec<Entity> = snapshot.iter().map(|&(e, _, _)| e).collect();

            let entities: Vec<Entity> = world.query::<Entity>().collect();
            let count = entities.len();

            let mut cmds = CommandBuffer::new();
            for _ in 0..CHURN_COUNT.min(count) {
                let idx = fastrand::usize(..count);
                cmds.despawn(entities[idx]);
            }
            cmds.apply(&mut world);

            // Demonstrate generational validation: count stale entities
            // from the pre-churn snapshot that are no longer alive.
            let stale_count = pre_churn_snapshot
                .iter()
                .filter(|&&e| !world.is_alive(e))
                .count();
            if stale_count > 0 {
                println!(
                    "  churn: despawned {}, {} entities in old snapshot now stale",
                    CHURN_COUNT, stale_count,
                );
            }

            // Replenish to target count
            let current = world.query::<&Position>().count();
            let deficit = ENTITY_COUNT.saturating_sub(current);
            for _ in 0..deficit {
                spawn_body(&mut world);
            }
        }

        // Step 7: Stats
        if frame % CHURN_INTERVAL == 0 || frame == FRAME_COUNT - 1 {
            let entity_count = world.query::<&Position>().count();
            let dt_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "frame {:04} | entities: {:>5} | dt: {:.1}ms",
                frame, entity_count, dt_ms,
            );
        }
    }

    println!("Done.");
}
