use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use minkowski::{Entity, World};
use minkowski_bench::{Position, Velocity};

fn spawn_position_world(n: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let mut entities = Vec::with_capacity(n);
    for i in 0..n {
        let f = i as f32;
        let e = world.spawn((Position { x: f, y: f, z: f },));
        entities.push(e);
    }
    (world, entities)
}

fn spawn_position_world_pool(n: usize) -> (World, Vec<Entity>) {
    let mut world = World::builder()
        .memory_budget(64 * 1024 * 1024)
        .build()
        .unwrap();
    let mut entities = Vec::with_capacity(n);
    for i in 0..n {
        let f = i as f32;
        let e = world.spawn((Position { x: f, y: f, z: f },));
        entities.push(e);
    }
    (world, entities)
}

fn add_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_remove");

    group.bench_function("add_remove", |b| {
        b.iter_batched(
            || spawn_position_world(10_000),
            |(mut world, entities)| {
                for &e in &entities {
                    world
                        .insert(
                            e,
                            (Velocity {
                                dx: 1.0,
                                dy: 1.0,
                                dz: 1.0,
                            },),
                        )
                        .unwrap();
                }
                for &e in &entities {
                    world.remove::<Velocity>(e);
                }
            },
            BatchSize::SmallInput,
        );
    });

    // ── Pool allocator variant ────────────────────────────────────────

    group.bench_function("pool", |b| {
        b.iter_batched(
            || spawn_position_world_pool(10_000),
            |(mut world, entities)| {
                for &e in &entities {
                    world
                        .insert(
                            e,
                            (Velocity {
                                dx: 1.0,
                                dy: 1.0,
                                dz: 1.0,
                            },),
                        )
                        .unwrap();
                }
                for &e in &entities {
                    world.remove::<Velocity>(e);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, add_remove);
criterion_main!(benches);
