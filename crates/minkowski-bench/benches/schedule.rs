use criterion::{criterion_group, criterion_main, Criterion};
use minkowski::{Optimistic, QueryMut, QueryReducerId, QueryRef, ReducerRegistry};
use minkowski_bench::{spawn_world, Position, Rotation, Transform, Velocity};

fn schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("schedule");

    let mut world = spawn_world(10_000);
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let sys_pos: QueryReducerId = registry
        .register_query::<(&mut Position,), (), _>(
            &mut world,
            "sys_pos",
            |mut q: QueryMut<'_, (&mut Position,)>, (): ()| {
                q.for_each(|(pos,)| {
                    pos.x += 1.0;
                });
            },
        )
        .unwrap();

    let sys_vel: QueryReducerId = registry
        .register_query::<(&mut Velocity,), (), _>(
            &mut world,
            "sys_vel",
            |mut q: QueryMut<'_, (&mut Velocity,)>, (): ()| {
                q.for_each(|(vel,)| {
                    vel.dx += 0.1;
                });
            },
        )
        .unwrap();

    let sys_rot: QueryReducerId = registry
        .register_query::<(&mut Rotation,), (), _>(
            &mut world,
            "sys_rot",
            |mut q: QueryMut<'_, (&mut Rotation,)>, (): ()| {
                q.for_each(|(rot,)| {
                    rot.x += 0.01;
                });
            },
        )
        .unwrap();

    let sys_transform: QueryReducerId = registry
        .register_query::<(&mut Transform,), (), _>(
            &mut world,
            "sys_transform",
            |mut q: QueryMut<'_, (&mut Transform,)>, (): ()| {
                q.for_each(|(t,)| {
                    t.matrix[0][0] += 0.001;
                });
            },
        )
        .unwrap();

    let sys_read_pos: QueryReducerId = registry
        .register_query_ref::<(&Position,), (), _>(
            &mut world,
            "sys_read_pos",
            |mut q: QueryRef<'_, (&Position,)>, (): ()| {
                q.for_each(|(_pos,)| {
                    // pure read — measures iteration overhead without mutation cost
                });
            },
        )
        .unwrap();

    group.bench_function("5_systems_10k", |b| {
        b.iter(|| {
            registry.run(&mut world, sys_pos, ()).unwrap();
            registry.run(&mut world, sys_vel, ()).unwrap();
            registry.run(&mut world, sys_rot, ()).unwrap();
            registry.run(&mut world, sys_transform, ()).unwrap();
            registry.run(&mut world, sys_read_pos, ()).unwrap();
        });
    });

    group.finish();
    drop(strategy);
}

criterion_group!(benches, schedule);
criterion_main!(benches);
