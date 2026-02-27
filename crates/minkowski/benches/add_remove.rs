use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Health(f32);

fn add_remove_minkowski(c: &mut Criterion) {
    c.bench_function("minkowski/add_remove_1k", |b| {
        b.iter_batched(
            || {
                let mut world = minkowski::World::new();
                let entities: Vec<_> = (0..1000)
                    .map(|i| world.spawn((Position { x: i as f32, y: 0.0 },)))
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for &e in &entities {
                    world.insert(e, Health(100.0));
                }
                for &e in &entities {
                    world.remove::<Health>(e);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn add_remove_hecs(c: &mut Criterion) {
    c.bench_function("hecs/add_remove_1k", |b| {
        b.iter_batched(
            || {
                let mut world = hecs::World::new();
                let entities: Vec<_> = (0..1000)
                    .map(|i| world.spawn((Position { x: i as f32, y: 0.0 },)))
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for &e in &entities {
                    world.insert_one(e, Health(100.0)).unwrap();
                }
                for &e in &entities {
                    world.remove_one::<Health>(e).unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, add_remove_minkowski, add_remove_hecs);
criterion_main!(benches);
