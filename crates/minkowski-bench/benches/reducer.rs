use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::{DynamicCtx, Optimistic, QueryMut, QueryWriter, ReducerRegistry, World};
use minkowski_bench::{Position, Velocity};

fn setup_world() -> World {
    let mut world = World::new();
    for i in 0..10_000 {
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

fn bench_query_mut(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query::<(&mut Position, &Velocity), (), _>(
            &mut world,
            "integrate",
            |mut query: QueryMut<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(pos, vel)| {
                    pos.x += vel.dx;
                    pos.y += vel.dy;
                    pos.z += vel.dz;
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_mut_10k", |b| {
        b.iter(|| {
            registry.run(&mut world, id, ()).unwrap();
        });
    });
    drop(strategy);
}

fn bench_query_mut_chunk(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query::<(&mut Position, &Velocity), (), _>(
            &mut world,
            "integrate_chunk",
            |mut query: QueryMut<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each_chunk(|(positions, velocities)| {
                    for i in 0..positions.len() {
                        positions[i].x += velocities[i].dx;
                        positions[i].y += velocities[i].dy;
                        positions[i].z += velocities[i].dz;
                    }
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_mut_chunk_10k", |b| {
        b.iter(|| {
            registry.run(&mut world, id, ()).unwrap();
        });
    });
    drop(strategy);
}

fn bench_query_writer(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position, &Velocity), (), _>(
            &mut world,
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

    c.bench_function("reducer/query_writer_10k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}

fn bench_dynamic_for_each(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("integrate_dynamic", &mut world)
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

    c.bench_function("reducer/dynamic_for_each_10k", |b| {
        b.iter(|| {
            registry
                .dynamic_call(&strategy, &mut world, id, &())
                .unwrap();
        });
    });
}

fn bench_dynamic_for_each_chunk(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("integrate_dynamic_chunk", &mut world)
        .can_read::<Position>()
        .can_read::<Velocity>()
        .can_write::<Position>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            let mut updates = Vec::new();
            ctx.for_each_chunk::<(minkowski::Entity, &Position, &Velocity)>(
                |(entities, positions, velocities)| {
                    for i in 0..entities.len() {
                        updates.push((
                            entities[i],
                            Position {
                                x: positions[i].x + velocities[i].dx,
                                y: positions[i].y + velocities[i].dy,
                                z: positions[i].z + velocities[i].dz,
                            },
                        ));
                    }
                },
            );
            for (entity, pos) in updates {
                ctx.write(entity, pos);
            }
        })
        .unwrap();

    c.bench_function("reducer/dynamic_for_each_chunk_10k", |b| {
        b.iter(|| {
            registry
                .dynamic_call(&strategy, &mut world, id, &())
                .unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_query_mut,
    bench_query_mut_chunk,
    bench_query_writer,
    bench_dynamic_for_each,
    bench_dynamic_for_each_chunk,
);
criterion_main!(benches);
