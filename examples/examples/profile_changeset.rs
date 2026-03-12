//! Profiling harness for QueryWriter vs QueryMut overhead comparison.
//!
//! NOT a benchmark — this binary is designed for flamegraph capture with samply.
//! Criterion's measurement harness pollutes profiles with statistics overhead.
//!
//! Usage:
//!   cargo build -p minkowski-examples --example profile_changeset --release
//!   samply record target/release/examples/profile_changeset
//!
//! The profiler will show two distinct call subtrees under main():
//!   run_query_mut()   — baseline (direct mutation)
//!   run_query_writer() — subject (buffered via EnumChangeSet)
//!
//! Compare time spent in each to identify where the overhead lives.

use minkowski::{DynamicCtx, Optimistic, QueryMut, QueryWriter, ReducerRegistry, World};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

const ENTITY_COUNT: usize = 10_000;
const ITERATIONS: usize = 1_000;

fn setup_world() -> World {
    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
            Velocity {
                dx: 1.0,
                dy: 0.0,
                dz: 0.0,
            },
        ));
    }
    world
}

/// Baseline: direct mutation via QueryMut (scheduled reducer, no changeset).
#[inline(never)]
fn run_query_mut(world: &mut World, registry: &mut ReducerRegistry) {
    let id = registry
        .register_query::<(&mut Position, &Velocity), (), _>(
            world,
            "integrate_mut",
            |mut query: QueryMut<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(pos, vel)| {
                    pos.x += vel.dx;
                    pos.y += vel.dy;
                    pos.z += vel.dz;
                });
            },
        )
        .unwrap();

    for _ in 0..ITERATIONS {
        registry.run(world, id, ()).unwrap();
    }
}

/// Subject: buffered mutation via QueryWriter (transactional, uses EnumChangeSet).
#[inline(never)]
fn run_query_writer(world: &mut World, registry: &mut ReducerRegistry) {
    let strategy = Optimistic::new(world);
    let id = registry
        .register_query_writer::<(&mut Position, &Velocity), (), _>(
            world,
            "integrate_writer",
            |mut query: QueryWriter<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(mut pos, vel)| {
                    pos.modify(|p| {
                        p.x += vel.dx;
                        p.y += vel.dy;
                        p.z += vel.dz;
                    });
                });
            },
        )
        .unwrap();

    for _ in 0..ITERATIONS {
        registry.call(&strategy, world, id, ()).unwrap();
    }
}

/// Subject: dynamic reducer with collect-then-write pattern.
#[inline(never)]
fn run_dynamic(world: &mut World, registry: &mut ReducerRegistry) {
    let strategy = Optimistic::new(world);
    let id = registry
        .dynamic("integrate_dynamic", world)
        .can_read::<Position>()
        .can_read::<Velocity>()
        .can_write::<Position>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            let mut updates = Vec::new();
            ctx.for_each::<(minkowski::Entity, &Position, &Velocity)>(|(entity, pos, vel)| {
                updates.push((
                    entity,
                    Position {
                        x: pos.x + vel.dx,
                        y: pos.y + vel.dy,
                        z: pos.z + vel.dz,
                    },
                ));
            });
            for (entity, pos) in updates {
                ctx.write(entity, pos);
            }
        })
        .unwrap();

    for _ in 0..ITERATIONS {
        registry.dynamic_call(&strategy, world, id, &()).unwrap();
    }
}

fn main() {
    // Phase 1: QueryMut baseline
    let mut world = setup_world();
    let mut registry = ReducerRegistry::new();
    run_query_mut(&mut world, &mut registry);
    std::hint::black_box(&world);

    // Phase 2: QueryWriter subject
    let mut world = setup_world();
    let mut registry = ReducerRegistry::new();
    run_query_writer(&mut world, &mut registry);
    std::hint::black_box(&world);

    // Phase 3: DynamicCtx subject
    let mut world = setup_world();
    let mut registry = ReducerRegistry::new();
    run_dynamic(&mut world, &mut registry);
    std::hint::black_box(&world);
}
