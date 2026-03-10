use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::{invert_4x4, Transform};

fn spawn_heavy_world() -> minkowski::World {
    let mut world = minkowski::World::new();
    for i in 0..1_000 {
        let f = (i + 1) as f32;
        // Upper-triangular matrix (non-singular, det = f^4)
        world.spawn((Transform {
            matrix: [
                [f, f * 0.5, f * 0.3, f * 0.1],
                [0.0, f, f * 0.7, f * 0.2],
                [0.0, 0.0, f, f * 0.4],
                [0.0, 0.0, 0.0, f],
            ],
        },));
    }
    world
}

fn heavy_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("heavy_compute");

    group.bench_function("sequential", |b| {
        let mut world = spawn_heavy_world();
        b.iter(|| {
            world
                .query::<&mut Transform>()
                .for_each_chunk(|transforms| {
                    for t in transforms.iter_mut() {
                        t.matrix = invert_4x4(&t.matrix);
                    }
                });
        });
    });

    group.bench_function("parallel", |b| {
        let mut world = spawn_heavy_world();
        b.iter(|| {
            world.query::<&mut Transform>().par_for_each(|t| {
                t.matrix = invert_4x4(&t.matrix);
            });
        });
    });

    group.finish();
}

criterion_group!(benches, heavy_compute);
criterion_main!(benches);
