//! Flatworm (planarian) simulator — exercises spatial indexing, chemotaxis,
//! entity lifecycle (fission + starvation), and query reducers.
//!
//! Run: cargo run -p minkowski-examples --example flatworm --release
//!
//! Exercises: spawn, despawn, query reducers (QueryMut, QueryRef), spatial grid
//! (SpatialIndex), deferred commands (CommandBuffer), entity churn via biological
//! fission and starvation.
//!
//! Planarian flatworms glide across a toroidal world, sensing nearby food via a
//! spatial grid. They turn toward the strongest chemical gradient (chemotaxis),
//! consume food on contact, and grow. When a worm accumulates enough energy it
//! undergoes binary fission — splitting into two smaller worms. Worms that starve
//! below a minimum energy threshold are despawned.
//!
//! Food pellets respawn at random positions to maintain a steady supply.

use minkowski::{CommandBuffer, Entity, QueryMut, QueryRef, ReducerRegistry, SpatialIndex, World};
use std::collections::HashSet;
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

    fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
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

// ── Components ──────────────────────────────────────────────────────

// Worm components
#[derive(Clone, Copy)]
struct Position(Vec2);

#[derive(Clone, Copy)]
struct Heading(f32); // radians

#[derive(Clone, Copy)]
struct Energy(f32);

#[derive(Clone, Copy)]
struct WormSize(f32); // body length, grows with energy

// Marker to distinguish worms from food in queries
#[derive(Clone, Copy)]
struct Worm;

// Food components
#[derive(Clone, Copy)]
struct Nutrition(f32);

#[derive(Clone, Copy)]
struct Food;

// ── Parameters ──────────────────────────────────────────────────────

struct SimParams {
    world_size: f32,
    worm_speed: f32,
    turn_rate: f32,
    sense_radius: f32,
    eat_radius: f32,
    energy_drain: f32,      // energy lost per tick
    fission_threshold: f32, // energy needed to split
    starvation_threshold: f32,
    min_worm_size: f32,
    max_worm_size: f32,
    food_energy: f32,
    food_count: usize,
    worm_count: usize,
    wander_strength: f32, // random heading perturbation
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            world_size: 400.0,
            worm_speed: 2.0,
            turn_rate: 0.15,
            sense_radius: 60.0,
            eat_radius: 8.0,
            energy_drain: 0.12,
            fission_threshold: 100.0,
            starvation_threshold: 5.0,
            min_worm_size: 3.0,
            max_worm_size: 12.0,
            food_energy: 20.0,
            food_count: 300,
            worm_count: 200,
            wander_strength: 0.3,
        }
    }
}

// ── Food spatial grid ───────────────────────────────────────────────

struct FoodGrid {
    cell_size: f32,
    grid_w: usize,
    cells: Vec<Vec<usize>>,
    snapshot: Vec<(Entity, Vec2, f32)>, // entity, position, nutrition
}

impl FoodGrid {
    fn new(cell_size: f32, world_size: f32) -> Self {
        let grid_w = (world_size / cell_size).ceil() as usize;
        Self {
            cell_size,
            grid_w,
            cells: Vec::new(),
            snapshot: Vec::new(),
        }
    }

    fn neighbors(&self, pos: Vec2) -> impl Iterator<Item = &(Entity, Vec2, f32)> {
        let cx = ((pos.x / self.cell_size) as usize).min(self.grid_w - 1);
        let cy = ((pos.y / self.cell_size) as usize).min(self.grid_w - 1);
        let grid_w = self.grid_w;
        (-1i32..=1).flat_map(move |dy| {
            (-1i32..=1).flat_map(move |dx| {
                let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                let cell = &self.cells[ny * grid_w + nx];
                cell.iter().map(|&j| &self.snapshot[j])
            })
        })
    }
}

impl SpatialIndex for FoodGrid {
    fn rebuild(&mut self, world: &mut World) {
        self.snapshot = world
            .query::<(Entity, &Position, &Nutrition, &Food)>()
            .map(|(e, p, n, _)| (e, p.0, n.0))
            .collect();

        self.cells.clear();
        self.cells.resize(self.grid_w * self.grid_w, Vec::new());
        for (i, &(_, pos, _)) in self.snapshot.iter().enumerate() {
            let cx = ((pos.x / self.cell_size) as usize).min(self.grid_w - 1);
            let cy = ((pos.y / self.cell_size) as usize).min(self.grid_w - 1);
            self.cells[cy * self.grid_w + cx].push(i);
        }
    }
}

// ── Constants ───────────────────────────────────────────────────────

const FRAME_COUNT: usize = 1_000;
const REPORT_INTERVAL: usize = 100;
const DT: f32 = 0.016;

// ── Helpers ─────────────────────────────────────────────────────────

fn wrap(mut v: f32, size: f32) -> f32 {
    if v >= size {
        v -= size;
    } else if v < 0.0 {
        v += size;
    }
    v
}

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

fn spawn_worm(world: &mut World, params: &SimParams) -> Entity {
    let x = fastrand::f32() * params.world_size;
    let y = fastrand::f32() * params.world_size;
    let heading = fastrand::f32() * std::f32::consts::TAU;
    let energy = 30.0 + fastrand::f32() * 30.0;
    world.spawn((
        Position(Vec2::new(x, y)),
        Heading(heading),
        Energy(energy),
        WormSize(params.min_worm_size + fastrand::f32() * 3.0),
        Worm,
    ))
}

fn spawn_food(world: &mut World, params: &SimParams) -> Entity {
    let x = fastrand::f32() * params.world_size;
    let y = fastrand::f32() * params.world_size;
    world.spawn((
        Position(Vec2::new(x, y)),
        Nutrition(params.food_energy),
        Food,
    ))
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let params = SimParams::default();
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    // Spawn initial worms
    for _ in 0..params.worm_count {
        spawn_worm(&mut world, &params);
    }

    // Spawn initial food
    for _ in 0..params.food_count {
        spawn_food(&mut world, &params);
    }

    let mut food_grid = FoodGrid::new(params.sense_radius, params.world_size);

    // ── Register query reducers ─────────────────────────────────────

    // Movement: advance worms along their heading
    let worm_speed = params.worm_speed;
    let ws = params.world_size;
    let move_id = registry.register_query::<(&mut Position, &Heading, &Worm), f32, _>(
        &mut world,
        "worm_move",
        move |mut query: QueryMut<'_, (&mut Position, &Heading, &Worm)>, dt: f32| {
            query.for_each_chunk(|(poss, headings, _worms)| {
                for i in 0..poss.len() {
                    let h = headings[i].0;
                    poss[i].0.x = wrap(poss[i].0.x + h.cos() * worm_speed * dt, ws);
                    poss[i].0.y = wrap(poss[i].0.y + h.sin() * worm_speed * dt, ws);
                }
            });
        },
    );

    // Metabolism: drain energy, update size
    let energy_drain = params.energy_drain;
    let min_size = params.min_worm_size;
    let max_size = params.max_worm_size;
    let fission_thr = params.fission_threshold;
    let metabolism_id = registry.register_query::<(&mut Energy, &mut WormSize, &Worm), f32, _>(
        &mut world,
        "metabolism",
        move |mut query: QueryMut<'_, (&mut Energy, &mut WormSize, &Worm)>, dt: f32| {
            query.for_each(|(energy, size, _)| {
                energy.0 -= energy_drain * dt;
                // Size scales linearly with energy, clamped
                let target = min_size + (energy.0 / fission_thr) * (max_size - min_size);
                size.0 = target.clamp(min_size, max_size);
            });
        },
    );

    // Census: count worms (read-only)
    let census_id = registry.register_query_ref::<(&Energy, &Worm), (), _>(
        &mut world,
        "census",
        |mut query: QueryRef<'_, (&Energy, &Worm)>, ()| {
            let mut total_energy = 0.0f32;
            let mut count = 0usize;
            query.for_each(|(energy, _)| {
                total_energy += energy.0;
                count += 1;
            });
            if count > 0 {
                let avg = total_energy / count as f32;
                println!("  census: {} worms, avg energy {:.1}", count, avg);
            }
        },
    );

    // ── Simulation loop ─────────────────────────────────────────────

    let mut total_fissions = 0usize;
    let mut total_deaths = 0usize;

    for frame in 0..FRAME_COUNT {
        let frame_start = Instant::now();

        // Step 1: Move worms
        registry.run(&mut world, move_id, DT);

        // Step 2: Metabolism
        registry.run(&mut world, metabolism_id, DT);

        // Step 3: Rebuild food grid
        food_grid.rebuild(&mut world);

        // Step 4: Chemotaxis — worms turn toward nearest food
        {
            let worms: Vec<(Entity, Vec2)> = world
                .query::<(Entity, &Position, &Worm)>()
                .map(|(e, p, _)| (e, p.0))
                .collect();

            for (entity, pos) in &worms {
                // Find best food within sense radius (nutrition / distance² gradient)
                let mut best_dir = Vec2::ZERO;
                let mut best_score = 0.0_f32;

                for &(_, food_pos, nutrition) in food_grid.neighbors(*pos) {
                    let diff = Vec2::new(
                        wrapped_diff(food_pos.x, pos.x, params.world_size),
                        wrapped_diff(food_pos.y, pos.y, params.world_size),
                    );
                    let dist_sq = diff.length_sq();
                    if dist_sq < 1e-6 {
                        continue;
                    }
                    let score = nutrition / dist_sq;
                    if score > best_score {
                        best_score = score;
                        best_dir = diff;
                    }
                }

                if let Some(h) = world.get_mut::<Heading>(*entity) {
                    if best_score > 0.0 {
                        // Turn toward food
                        let target_angle = best_dir.y.atan2(best_dir.x);
                        // Normalize delta to [-pi, pi] via modular arithmetic
                        let delta = (target_angle - h.0 + std::f32::consts::PI)
                            .rem_euclid(std::f32::consts::TAU)
                            - std::f32::consts::PI;
                        h.0 += delta * params.turn_rate;
                    } else {
                        // Random wander
                        h.0 += (fastrand::f32() - 0.5) * params.wander_strength;
                    }
                    // Keep heading in [0, TAU) to prevent unbounded drift
                    h.0 = h.0.rem_euclid(std::f32::consts::TAU);
                }
            }
        }

        // Step 5: Feeding — worms eat nearby food
        {
            let worms: Vec<(Entity, Vec2)> = world
                .query::<(Entity, &Position, &Worm)>()
                .map(|(e, p, _)| (e, p.0))
                .collect();

            let mut eaten = CommandBuffer::new();
            let mut eaten_set = HashSet::new();
            let eat_r_sq = params.eat_radius * params.eat_radius;

            for (worm_entity, worm_pos) in &worms {
                for &(food_entity, food_pos, nutrition) in food_grid.neighbors(*worm_pos) {
                    if eaten_set.contains(&food_entity) {
                        continue;
                    }
                    let diff = Vec2::new(
                        wrapped_diff(food_pos.x, worm_pos.x, params.world_size),
                        wrapped_diff(food_pos.y, worm_pos.y, params.world_size),
                    );
                    if diff.length_sq() < eat_r_sq {
                        if let Some(energy) = world.get_mut::<Energy>(*worm_entity) {
                            energy.0 += nutrition;
                        }
                        eaten_set.insert(food_entity);
                        eaten.despawn(food_entity);
                    }
                }
            }
            eaten.apply(&mut world);
        }

        // Step 6: Fission — worms that exceed the energy threshold split in two
        {
            let candidates: Vec<(Entity, Vec2, f32, f32)> = world
                .query::<(Entity, &Position, &Energy, &Heading, &Worm)>()
                .filter(|(_, _, e, _, _)| e.0 >= params.fission_threshold)
                .map(|(e, p, en, h, _)| (e, p.0, en.0, h.0))
                .collect();

            let mut cmds = CommandBuffer::new();
            for (parent, pos, energy, heading) in candidates {
                let half_energy = energy * 0.5;
                // Offspring spawns offset perpendicular to parent heading
                let offset = Vec2::new(
                    (heading + std::f32::consts::FRAC_PI_2).cos() * params.min_worm_size,
                    (heading + std::f32::consts::FRAC_PI_2).sin() * params.min_worm_size,
                );
                let child_pos = Vec2::new(
                    wrap(pos.x + offset.x, params.world_size),
                    wrap(pos.y + offset.y, params.world_size),
                );

                // Reduce parent energy
                if let Some(e) = world.get_mut::<Energy>(parent) {
                    e.0 = half_energy;
                }

                // Spawn child
                let child_heading = heading + std::f32::consts::PI; // opposite direction
                cmds.spawn((
                    Position(child_pos),
                    Heading(child_heading),
                    Energy(half_energy),
                    WormSize(params.min_worm_size),
                    Worm,
                ));
                total_fissions += 1;
            }
            cmds.apply(&mut world);
        }

        // Step 7: Starvation — despawn worms below minimum energy
        {
            let starved: Vec<Entity> = world
                .query::<(Entity, &Energy, &Worm)>()
                .filter(|(_, e, _)| e.0 <= params.starvation_threshold)
                .map(|(e, _, _)| e)
                .collect();

            let mut cmds = CommandBuffer::new();
            for entity in starved.iter().copied() {
                cmds.despawn(entity);
            }
            total_deaths += starved.len();
            cmds.apply(&mut world);
        }

        // Step 8: Replenish food
        {
            let current_food = world.query::<(&Food,)>().count();
            let deficit = params.food_count.saturating_sub(current_food);
            for _ in 0..deficit {
                spawn_food(&mut world, &params);
            }
        }

        // Step 9: Report
        if frame % REPORT_INTERVAL == 0 || frame == FRAME_COUNT - 1 {
            let worm_count = world.query::<(&Worm,)>().count();
            let food_count = world.query::<(&Food,)>().count();
            let dt_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "frame {:04} | worms: {:>4} | food: {:>4} | fissions: {} | deaths: {} | dt: {:.1}ms",
                frame, worm_count, food_count, total_fissions, total_deaths, dt_ms,
            );
            registry.run(&mut world, census_id, ());
        }
    }

    println!(
        "Done. Total fissions: {}, total deaths: {}",
        total_fissions, total_deaths
    );
}
