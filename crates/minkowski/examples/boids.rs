//! Terminal boids simulation — exercises the full minkowski ECS API.
//! Run: cargo run -p minkowski --example boids --release

use minkowski::{Entity, World, CommandBuffer};

#[derive(Clone, Copy)]
struct Position { x: f32, y: f32 }

#[derive(Clone, Copy)]
struct Velocity { dx: f32, dy: f32 }

#[derive(Clone, Copy)]
struct Acceleration { ax: f32, ay: f32 }

const BOID_COUNT: usize = 200;
const FRAMES: usize = 10_000;
const WORLD_SIZE: f32 = 100.0;
const MAX_SPEED: f32 = 2.0;
const SEPARATION_RADIUS: f32 = 5.0;
const ALIGNMENT_RADIUS: f32 = 10.0;
const COHESION_RADIUS: f32 = 15.0;
const DT: f32 = 0.016;

fn main() {
    let mut world = World::new();

    // Spawn initial boids
    for i in 0..BOID_COUNT {
        let angle = (i as f32 / BOID_COUNT as f32) * std::f32::consts::TAU;
        let r = WORLD_SIZE * 0.3;
        world.spawn((
            Position {
                x: WORLD_SIZE / 2.0 + r * angle.cos(),
                y: WORLD_SIZE / 2.0 + r * angle.sin(),
            },
            Velocity {
                dx: angle.sin() * MAX_SPEED * 0.5,
                dy: -angle.cos() * MAX_SPEED * 0.5,
            },
            Acceleration { ax: 0.0, ay: 0.0 },
        ));
    }

    let mut total_speed = 0.0f32;

    for frame in 0..FRAMES {
        // 1. Collect positions for neighbor queries
        let boids: Vec<(Entity, Position, Velocity)> = world
            .query::<(Entity, &Position, &Velocity)>()
            .map(|(e, p, v)| (e, *p, *v))
            .collect();

        // 2. Compute boid forces -> write accelerations via CommandBuffer
        let mut cmds = CommandBuffer::new();
        for &(entity, pos, vel) in &boids {
            let (mut sep_x, mut sep_y) = (0.0f32, 0.0f32);
            let (mut ali_dx, mut ali_dy, mut ali_count) = (0.0f32, 0.0f32, 0u32);
            let (mut coh_x, mut coh_y, mut coh_count) = (0.0f32, 0.0f32, 0u32);

            for &(_other_e, other_pos, other_vel) in &boids {
                let dx = other_pos.x - pos.x;
                let dy = other_pos.y - pos.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 0.001 { continue; }

                if dist < SEPARATION_RADIUS {
                    sep_x -= dx / dist;
                    sep_y -= dy / dist;
                }
                if dist < ALIGNMENT_RADIUS {
                    ali_dx += other_vel.dx;
                    ali_dy += other_vel.dy;
                    ali_count += 1;
                }
                if dist < COHESION_RADIUS {
                    coh_x += other_pos.x;
                    coh_y += other_pos.y;
                    coh_count += 1;
                }
            }

            let mut ax = sep_x * 1.5;
            let mut ay = sep_y * 1.5;

            if ali_count > 0 {
                ax += (ali_dx / ali_count as f32 - vel.dx) * 0.5;
                ay += (ali_dy / ali_count as f32 - vel.dy) * 0.5;
            }
            if coh_count > 0 {
                ax += (coh_x / coh_count as f32 - pos.x) * 0.01;
                ay += (coh_y / coh_count as f32 - pos.y) * 0.01;
            }

            cmds.insert(entity, Acceleration { ax, ay });
        }
        cmds.apply(&mut world);

        // 3. Integrate velocity from acceleration
        for (vel, acc) in world.query::<(&mut Velocity, &Acceleration)>() {
            vel.dx += acc.ax * DT;
            vel.dy += acc.ay * DT;
            let speed = (vel.dx * vel.dx + vel.dy * vel.dy).sqrt();
            if speed > MAX_SPEED {
                vel.dx = vel.dx / speed * MAX_SPEED;
                vel.dy = vel.dy / speed * MAX_SPEED;
            }
        }

        // 4. Integrate position from velocity + wrap around
        for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
            pos.x += vel.dx * DT;
            pos.y += vel.dy * DT;
            pos.x = pos.x.rem_euclid(WORLD_SIZE);
            pos.y = pos.y.rem_euclid(WORLD_SIZE);
        }

        // 5. Compute stats
        if frame % 1000 == 0 || frame == FRAMES - 1 {
            let entity_count = world.query::<&Position>().count();
            let mut speed_sum = 0.0f32;
            for vel in world.query::<&Velocity>() {
                speed_sum += (vel.dx * vel.dx + vel.dy * vel.dy).sqrt();
            }
            let avg_speed = speed_sum / entity_count as f32;
            total_speed += avg_speed;
            println!(
                "frame {frame:>5}: entities={entity_count}, avg_speed={avg_speed:.3}"
            );
        }
    }

    println!("Done. Overall avg speed: {:.3}", total_speed / ((FRAMES / 1000 + 1) as f32));
}
