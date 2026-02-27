use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Velocity { dx: f32, dy: f32 }

fn spawn_minkowski(c: &mut Criterion) {
    c.bench_function("minkowski/spawn_10k", |b| {
        b.iter(|| {
            let mut world = minkowski::World::new();
            for i in 0..10_000 {
                world.spawn((
                    Position { x: i as f32, y: 0.0 },
                    Velocity { dx: 1.0, dy: 0.0 },
                ));
            }
        });
    });
}

fn spawn_hecs(c: &mut Criterion) {
    c.bench_function("hecs/spawn_10k", |b| {
        b.iter(|| {
            let mut world = hecs::World::new();
            for i in 0..10_000 {
                world.spawn((
                    Position { x: i as f32, y: 0.0 },
                    Velocity { dx: 1.0, dy: 0.0 },
                ));
            }
        });
    });
}

criterion_group!(benches, spawn_minkowski, spawn_hecs);
criterion_main!(benches);
