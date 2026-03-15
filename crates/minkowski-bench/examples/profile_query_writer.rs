//! Profiling harness for QueryWriter apply phase.
//! Run with: cargo flamegraph --example profile_query_writer -p minkowski-bench

use minkowski::{Optimistic, QueryWriter, ReducerRegistry, World};
use minkowski_bench::{Position, Velocity};

fn main() {
    let mut world = World::new();
    for i in 0..10_000 {
        let f = i as f32;
        world.spawn((
            Position { x: f, y: f, z: f },
            Velocity {
                dx: 1.0,
                dy: 1.0,
                dz: 1.0,
            },
        ));
    }

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position, &Velocity), (), _>(
            &mut world,
            "integrate_writer",
            |mut query: QueryWriter<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(mut pos, vel)| {
                    pos.modify(|p| {
                        p.x += vel.dx;
                        p.y += vel.dy;
                        p.z += vel.dz;
                    });
                });
            },
        )
        .unwrap();

    // Warm up
    for _ in 0..10 {
        registry.call(&strategy, &mut world, id, ()).unwrap();
    }

    // Hot loop — this is what perf will profile.
    for _ in 0..10_000 {
        registry.call(&strategy, &mut world, id, ()).unwrap();
    }
}
