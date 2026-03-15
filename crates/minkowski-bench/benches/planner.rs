use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::{
    AggregateExpr, BTreeIndex, Changed, HashIndex, Predicate, QueryPlanner, SpatialIndex, World,
};
use minkowski_bench::Score;

/// Spawn `n` entities with `Score(0..n)` across a single archetype.
fn score_world(n: u32) -> World {
    let mut world = World::new();
    for i in 0..n {
        world.spawn((Score(i),));
    }
    world
}

fn planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner");

    // ── Scan: planner for_each vs world.query() for_each ────────────
    //
    // Measures the overhead of plan compilation + type-erased dispatch
    // (CompiledForEach Box<dyn FnMut>) vs monomorphic QueryIter.

    group.bench_function("scan_for_each_10k", |b| {
        let mut world = score_world(10_000);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();
        drop(planner);

        b.iter(|| {
            let mut count = 0u32;
            plan.for_each(&mut world, |_| count += 1).unwrap();
            count
        });
    });

    group.bench_function("query_for_each_10k", |b| {
        let mut world = score_world(10_000);

        b.iter(|| {
            let mut count = 0u32;
            for _ in world.query::<(&Score,)>() {
                count += 1;
            }
            count
        });
    });

    // ── Index-driven: BTree range scan ──────────────────────────────
    //
    // IndexGather via pre-bound BTree lookup closure.
    // 10% selectivity: Score(0..1000) out of 10K entities.

    group.bench_function("btree_range_10pct", |b| {
        let mut world = score_world(10_000);
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index::<Score>(&idx, &world).unwrap();
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(0)..Score(1000)))
            .build();
        drop(planner);

        b.iter(|| {
            let mut count = 0u32;
            plan.for_each(&mut world, |_| count += 1).unwrap();
            count
        });
    });

    // ── Index-driven: Hash exact lookup ─────────────────────────────

    group.bench_function("hash_eq_1", |b| {
        let mut world = score_world(10_000);
        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index::<Score>(&idx, &world).unwrap();
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(5000)))
            .build();
        drop(planner);

        b.iter(|| {
            let mut count = 0u32;
            plan.for_each(&mut world, |_| count += 1).unwrap();
            count
        });
    });

    // ── Scan with custom filter ─────────────────────────────────────
    //
    // Measures per-entity Arc<dyn Fn> dispatch overhead for custom predicates.

    group.bench_function("custom_filter_50pct", |b| {
        let mut world = score_world(10_000);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "score < 5000",
                0.5,
                |world: &World, entity| world.get::<Score>(entity).is_some_and(|s| s.0 < 5000),
            ))
            .build();
        drop(planner);

        b.iter(|| {
            let mut count = 0u32;
            plan.for_each(&mut world, |_| count += 1).unwrap();
            count
        });
    });

    // ── Changed<T> filtering ────────────────────────────────────────
    //
    // First call sees all entities (new). Second call sees 0 (no mutations).
    // Measures the tick comparison cost per archetype.

    group.bench_function("changed_skip_10k", |b| {
        let mut world = score_world(10_000);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();
        drop(planner);

        // First call: populate last_read_tick.
        plan.for_each(&mut world, |_| {}).unwrap();

        // Subsequent calls: all archetypes are skipped (no mutations).
        b.iter(|| {
            let mut count = 0u32;
            plan.for_each(&mut world, |_| count += 1).unwrap();
            count
        });
    });

    // ── Aggregates ──────────────────────────────────────────────────
    //
    // Single-pass COUNT + SUM over 10K entities via execute_aggregates.
    // Measures type-erased extractor overhead (per-entity world.get()).

    group.bench_function("aggregate_count_sum_10k", |b| {
        let mut world = score_world(10_000);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();
        drop(planner);

        b.iter(|| plan.execute_aggregates(&mut world).unwrap());
    });

    // ── Manual aggregate baseline ───────────────────────────────────
    //
    // Same COUNT + SUM via world.query() for_each — no type erasure.
    // Shows the cost of the extractor indirection.

    group.bench_function("manual_count_sum_10k", |b| {
        let mut world = score_world(10_000);

        b.iter(|| {
            let mut count = 0u64;
            let mut sum = 0.0f64;
            for (score,) in world.query::<(&Score,)>() {
                count += 1;
                sum += score.0 as f64;
            }
            (count, sum)
        });
    });

    // ── Execute (scratch buffer collection) ─────────────────────────
    //
    // Collects all matching entities into the plan-owned scratch buffer.
    // Measures entity push + scratch reuse across calls.

    group.bench_function("execute_collect_10k", |b| {
        let mut world = score_world(10_000);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();
        drop(planner);

        b.iter(|| plan.execute(&mut world).unwrap().len());
    });

    group.finish();
}

criterion_group!(benches, planner);
criterion_main!(benches);
