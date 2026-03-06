use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}
#[derive(Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
}

fn iterate_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    for i in 0..10_000 {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
            },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("minkowski/iterate_10k", |b| {
        b.iter(|| {
            for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        });
    });

    c.bench_function("minkowski/iterate_chunk_10k", |b| {
        b.iter(|| {
            world.query::<(&mut Position, &Velocity)>().for_each_chunk(
                |(positions, velocities)| {
                    for i in 0..positions.len() {
                        positions[i].x += velocities[i].dx;
                        positions[i].y += velocities[i].dy;
                    }
                },
            );
        });
    });
}

fn iterate_hecs(c: &mut Criterion) {
    let mut world = hecs::World::new();
    for i in 0..10_000 {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
            },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("hecs/iterate_10k", |b| {
        b.iter(|| {
            for (_id, (pos, vel)) in world.query_mut::<(&mut Position, &Velocity)>() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        });
    });
}

criterion_group!(benches, iterate_minkowski, iterate_hecs);
criterion_main!(benches);
