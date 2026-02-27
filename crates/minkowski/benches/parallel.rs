use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Velocity { dx: f32, dy: f32 }

fn parallel_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    for i in 0..100_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("minkowski/parallel_100k", |b| {
        b.iter(|| {
            world.query::<(&mut Position, &Velocity)>().par_for_each(|(pos, vel)| {
                pos.x += vel.dx;
                pos.y += vel.dy;
            });
        });
    });
}

fn sequential_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    for i in 0..100_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("minkowski/sequential_100k", |b| {
        b.iter(|| {
            for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        });
    });
}

criterion_group!(benches, parallel_minkowski, sequential_minkowski);
criterion_main!(benches);
