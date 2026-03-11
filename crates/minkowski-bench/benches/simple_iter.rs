use criterion::{Criterion, criterion_group, criterion_main};
use minkowski_bench::{Position, Velocity, spawn_world};

fn simple_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_iter");

    group.bench_function("for_each", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            for (vel, pos) in world.query::<(&mut Velocity, &Position)>() {
                vel.dx += pos.x * 0.1;
                vel.dy += pos.y * 0.1;
                vel.dz += pos.z * 0.1;
            }
        });
    });

    group.bench_function("for_each_chunk", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            world
                .query::<(&mut Velocity, &Position)>()
                .for_each_chunk(|(vels, positions)| {
                    for (vel, pos) in vels.iter_mut().zip(positions.iter()) {
                        vel.dx += pos.x * 0.1;
                        vel.dy += pos.y * 0.1;
                        vel.dz += pos.z * 0.1;
                    }
                });
        });
    });

    group.bench_function("par_for_each", |b| {
        let mut world = spawn_world(10_000);
        b.iter(|| {
            world
                .query::<(&mut Velocity, &Position)>()
                .par_for_each(|(vel, pos)| {
                    vel.dx += pos.x * 0.1;
                    vel.dy += pos.y * 0.1;
                    vel.dz += pos.z * 0.1;
                });
        });
    });

    group.finish();
}

criterion_group!(benches, simple_iter);
criterion_main!(benches);
