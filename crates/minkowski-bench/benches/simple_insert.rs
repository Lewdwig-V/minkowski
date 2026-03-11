use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::EnumChangeSet;
use minkowski_bench::{Position, Rotation, Transform, Velocity};

fn simple_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_insert");

    group.bench_function("batch", |b| {
        b.iter(|| {
            let mut world = minkowski::World::new();
            for i in 0..10_000 {
                let f = i as f32;
                world.spawn((
                    Transform {
                        matrix: [
                            [f, 0.0, 0.0, 0.0],
                            [0.0, f, 0.0, 0.0],
                            [0.0, 0.0, f, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    Position { x: f, y: f, z: f },
                    Rotation {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Velocity {
                        dx: 1.0,
                        dy: 1.0,
                        dz: 1.0,
                    },
                ));
            }
            world
        });
    });

    group.bench_function("changeset", |b| {
        b.iter(|| {
            let mut world = minkowski::World::new();
            let mut cs = EnumChangeSet::new();
            for i in 0..10_000 {
                let f = i as f32;
                let entity = world.alloc_entity();
                cs.spawn_bundle(
                    &mut world,
                    entity,
                    (
                        Transform {
                            matrix: [
                                [f, 0.0, 0.0, 0.0],
                                [0.0, f, 0.0, 0.0],
                                [0.0, 0.0, f, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                        Position { x: f, y: f, z: f },
                        Rotation {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        Velocity {
                            dx: 1.0,
                            dy: 1.0,
                            dz: 1.0,
                        },
                    ),
                );
            }
            cs.apply(&mut world).unwrap();
            world
        });
    });

    group.finish();
}

criterion_group!(benches, simple_insert);
criterion_main!(benches);
