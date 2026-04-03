use criterion::{Criterion, criterion_group, criterion_main};
use minkowski_bench::{Position, Velocity, spawn_world};

fn iter_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_micro");

    // --- next() vs for_each_chunk vs par_for_each_chunk ---

    group.bench_function("next_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            let mut sum = 0.0f32;
            for pos in world.query::<&Position>() {
                sum += pos.x;
            }
            sum
        });
    });

    group.bench_function("for_each_chunk_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            let mut sum = 0.0f32;
            world.query::<&Position>().for_each_chunk(|positions| {
                for p in positions {
                    sum += p.x;
                }
            });
            sum
        });
    });

    group.bench_function("par_for_each_chunk_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            let sum = std::sync::atomic::AtomicU64::new(0);
            world.query::<&Position>().par_for_each_chunk(|positions| {
                let chunk: f64 = positions.iter().map(|p| p.x as f64).sum();
                sum.fetch_add(chunk.to_bits(), std::sync::atomic::Ordering::Relaxed);
            });
            sum.load(std::sync::atomic::Ordering::Relaxed)
        });
    });

    // --- ExactSizeIterator::len() vs old size_hint pattern ---

    group.bench_function("len_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| world.query::<&Position>().len());
    });

    // --- Fragmented archetype iteration (next vs chunk) ---

    group.bench_function("next_fragmented_10k", |b| {
        let mut world = minkowski::World::new();
        // 10 archetypes with 1000 entities each
        for i in 0..1000u32 {
            world.spawn((Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },));
        }
        for i in 0..1000u32 {
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
        b.iter(|| {
            let mut sum = 0.0f32;
            for pos in world.query::<&Position>() {
                sum += pos.x;
            }
            sum
        });
    });

    group.bench_function("for_each_chunk_fragmented_10k", |b| {
        let mut world = minkowski::World::new();
        for i in 0..1000u32 {
            world.spawn((Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },));
        }
        for i in 0..1000u32 {
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
        b.iter(|| {
            let mut sum = 0.0f32;
            world.query::<&Position>().for_each_chunk(|positions| {
                for p in positions {
                    sum += p.x;
                }
            });
            sum
        });
    });

    // --- Mutation: next vs for_each_chunk vs par_for_each_chunk ---

    group.bench_function("mutate_next_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            for (vel, pos) in world.query::<(&mut Velocity, &Position)>() {
                vel.dx += pos.x * 0.1;
            }
        });
    });

    group.bench_function("mutate_for_each_chunk_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            world
                .query::<(&mut Velocity, &Position)>()
                .for_each_chunk(|(vels, positions)| {
                    for (vel, pos) in vels.iter_mut().zip(positions.iter()) {
                        vel.dx += pos.x * 0.1;
                    }
                });
        });
    });

    group.bench_function("mutate_par_for_each_chunk_10k", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            world
                .query::<(&mut Velocity, &Position)>()
                .par_for_each_chunk(|(vels, positions)| {
                    for (vel, pos) in vels.iter_mut().zip(positions.iter()) {
                        vel.dx += pos.x * 0.1;
                    }
                });
        });
    });

    group.finish();
}

criterion_group!(benches, iter_micro);
criterion_main!(benches);
