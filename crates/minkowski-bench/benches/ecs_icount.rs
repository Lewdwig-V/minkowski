//! Instruction-count benchmarks for the three headline ECS hot paths.
//!
//! These mirror the criterion benches in `planner.rs` and `reducer.rs`, but use
//! iai-callgrind so the numbers are deterministic and machine-independent — the
//! right gate for regressions (criterion wall-clock can't be compared across
//! machines or time).
//!
//! Each bench has a `setup` fn (NOT measured) that builds the world + plan, and
//! a body that runs ONLY the measured query/execute call.
//!
//! IMPORTANT: the repo's `.cargo/config.toml` sets `target-cpu=native`, which
//! makes valgrind SIGILL. Run these with:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=x86-64-v2" cargo bench -p minkowski-bench --bench ecs_icount
//! ```

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use minkowski::{
    JoinKind, Optimistic, QueryMut, QueryPlanResult, QueryPlanner, QueryReducerId, ReducerRegistry,
    World,
};
use minkowski_bench::{Position, Score, Team, Velocity};

// ── Helpers copied from planner.rs / reducer.rs ──────────────────────────────
// (benches can't share modules easily; component types come from the lib crate.)

/// Spawn `n` entities with `Score(0..n)` across a single archetype.
fn score_world(n: u32) -> World {
    let mut world = World::new();
    for i in 0..n {
        world.spawn((Score(i),));
    }
    world
}

/// Spawn `n` entities with `Score`, with `join_pct` fraction also getting `Team`.
fn join_world(n: u32, join_pct: f64) -> (World, u32) {
    let mut world = World::new();
    let threshold = (n as f64 * join_pct) as u32;
    for i in 0..n {
        if i < threshold {
            world.spawn((Score(i), Team(i % 5)));
        } else {
            world.spawn((Score(i),));
        }
    }
    (world, threshold)
}

/// Spawn `n` entities with (Position, Velocity) in a single archetype.
fn setup_world(n: u32) -> World {
    let mut world = World::new();
    for i in 0..n {
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
    world
}

// ── scan_for_each_10k ────────────────────────────────────────────────────────
// Mirrors planner.rs:59 (`planner` group, `scan_for_each_10k`).
// Measured: plan.execute_stream(&mut world, |_| count += 1).

/// NOT measured: build the world + compile the scan plan.
fn setup_scan() -> (World, QueryPlanResult) {
    let world = score_world(10_000);
    let planner = QueryPlanner::new(&world);
    let plan = planner.scan::<(&Score,)>().build();
    (world, plan)
}

#[library_benchmark]
#[bench::scan(setup = setup_scan)]
fn scan_for_each_10k((mut world, mut plan): (World, QueryPlanResult)) -> u32 {
    let mut count = 0u32;
    plan.execute_stream(black_box(&mut world), |_| count += 1)
        .unwrap();
    black_box(count)
}

// ── join_for_each_batched_10k ────────────────────────────────────────────────
// Mirrors planner.rs:284 (`join` group, `for_each_batched_10k`).
// Measured: plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, (score,)| ...).

/// NOT measured: build the join world + compile the inner-join plan.
fn setup_join() -> (World, QueryPlanResult) {
    let (world, _) = join_world(10_000, 0.8);
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    (world, plan)
}

#[library_benchmark]
#[bench::join(setup = setup_join)]
fn join_for_each_batched_10k((mut world, mut plan): (World, QueryPlanResult)) -> u64 {
    let mut sum = 0u64;
    plan.execute_stream_batched::<(&Score,), _>(black_box(&mut world), |_, (score,)| {
        sum += score.0 as u64;
    })
    .unwrap();
    black_box(sum)
}

// ── query_mut_10k ────────────────────────────────────────────────────────────
// Mirrors reducer.rs:24 (`bench_query_mut`, "reducer/query_mut_10k").
// Measured: registry.run(&mut world, id, ()).

/// NOT measured: build the world, strategy and register the integrate reducer.
fn setup_query_mut() -> (World, ReducerRegistry, QueryReducerId) {
    let mut world = setup_world(10_000);
    // `Optimistic::new` captures the world's WorldId; the strategy is not needed
    // by `registry.run`, but we construct it to mirror the criterion setup.
    let _strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query::<(&mut Position, &Velocity), (), _>(
            &mut world,
            "integrate",
            |mut query: QueryMut<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(positions, velocities)| {
                    for i in 0..positions.len() {
                        positions[i].x += velocities[i].dx;
                        positions[i].y += velocities[i].dy;
                        positions[i].z += velocities[i].dz;
                    }
                });
            },
        )
        .unwrap();
    (world, registry, id)
}

#[library_benchmark]
#[bench::query_mut(setup = setup_query_mut)]
fn query_mut_10k((mut world, registry, id): (World, ReducerRegistry, QueryReducerId)) {
    registry
        .run(black_box(&mut world), black_box(id), ())
        .unwrap();
}

library_benchmark_group!(
    name = ecs_hot_paths;
    benchmarks = scan_for_each_10k, join_for_each_batched_10k, query_mut_10k
);

main!(library_benchmark_groups = ecs_hot_paths);
