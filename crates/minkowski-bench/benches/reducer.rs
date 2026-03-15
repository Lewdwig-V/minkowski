use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::{DynamicCtx, Optimistic, QueryMut, QueryWriter, ReducerRegistry, World};
use minkowski_bench::{Position, Team, Velocity};

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
                query.for_each(|(positions, velocities)| {
                    for i in 0..positions.len() {
                        positions[i].x += velocities[i].dx;
                        positions[i].y += velocities[i].dy;
                        positions[i].z += velocities[i].dz;
                    }
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
            ctx.for_each::<(minkowski::Entity, &Position, &Velocity)>(
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

    c.bench_function("reducer/dynamic_for_each_10k", |b| {
        b.iter(|| {
            registry
                .dynamic_call(&strategy, &mut world, id, &())
                .unwrap();
        });
    });
}

fn bench_query_writer_multi_comp(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position, &mut Velocity), (), _>(
            &mut world,
            "integrate_multi_comp",
            |mut query: QueryWriter<'_, (&mut Position, &mut Velocity)>, (): ()| {
                query.for_each(|(mut pos, mut vel)| {
                    pos.modify(|p| {
                        p.x += 1.0;
                        p.y += 1.0;
                        p.z += 1.0;
                    });
                    vel.modify(|v| {
                        v.dx *= 0.99;
                        v.dy *= 0.99;
                        v.dz *= 0.99;
                    });
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_writer_multi_comp_10k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}

fn bench_query_writer_sparse_update(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position, &Velocity), (), _>(
            &mut world,
            "sparse_update",
            |mut query: QueryWriter<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(mut pos, vel)| {
                    // Only ~10% of entities: those with velocity magnitude > 1.5
                    if vel.dx * vel.dx + vel.dy * vel.dy + vel.dz * vel.dz > 2.25 {
                        pos.modify(|p| {
                            p.x += vel.dx;
                            p.y += vel.dy;
                            p.z += vel.dz;
                        });
                    }
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_writer_sparse_update_10k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}

fn bench_query_writer_multi_arch(c: &mut Criterion) {
    let mut world = World::new();
    // 3K entities: Position only
    for i in 0..3_000 {
        world.spawn((Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        },));
    }
    // 3K entities: Position + Velocity
    for i in 0..3_000 {
        world.spawn((
            Position {
                x: i as f32,
                y: 1.0,
                z: 0.0,
            },
            Velocity {
                dx: 1.0,
                dy: 0.0,
                dz: 0.0,
            },
        ));
    }
    // 3K entities: Position + Velocity + Team
    for i in 0..3_000 {
        world.spawn((
            Position {
                x: i as f32,
                y: 2.0,
                z: 0.0,
            },
            Velocity {
                dx: 0.0,
                dy: 1.0,
                dz: 0.0,
            },
            Team(i as u32 % 4),
        ));
    }

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position,), (), _>(
            &mut world,
            "multi_arch_pos",
            |mut query: QueryWriter<'_, (&mut Position,)>, (): ()| {
                query.for_each(|(mut pos,)| {
                    pos.modify(|p| {
                        p.x += 0.1;
                    });
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_writer_multi_arch_9k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_query_mut,
    bench_query_writer,
    bench_dynamic_for_each,
    bench_query_writer_multi_comp,
    bench_query_writer_sparse_update,
    bench_query_writer_multi_arch,
);
criterion_main!(benches);
