use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::{
    AggregateExpr, BTreeIndex, Changed, HashIndex, JoinKind, Predicate, QueryPlanner, SpatialIndex,
    World,
};
use minkowski_bench::{FatData, Score, Team};

/// Spawn `n` entities with `Score(0..n)` across a single archetype.
fn score_world(n: u32) -> World {
    let mut world = World::new();
    for i in 0..n {
        world.spawn((Score(i),));
    }
    world
}

/// Spawn `n` entities with `Score`, with `join_pct` fraction also getting `Team`.
/// Returns (world, number_of_joined_entities).
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

/// Spawn `n` entities with `FatData`, with `join_pct` fraction also getting `Team`.
fn fat_join_world(n: u32, join_pct: f64) -> (World, u32) {
    let mut world = World::new();
    let threshold = (n as f64 * join_pct) as u32;
    for i in 0..n {
        let fat = FatData {
            data: [i as u8; 256],
        };
        if i < threshold {
            world.spawn((fat, Team(i % 5)));
        } else {
            world.spawn((fat,));
        }
    }
    (world, threshold)
}

fn planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner");

    // ── Scan: planner execute_stream vs world.query() iteration ────────────
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
            plan.execute_stream(&mut world, |_| count += 1).unwrap();
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
            plan.execute_stream(&mut world, |_| count += 1).unwrap();
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
            plan.execute_stream(&mut world, |_| count += 1).unwrap();
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
            plan.execute_stream(&mut world, |_| count += 1).unwrap();
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
        plan.execute_stream(&mut world, |_| {}).unwrap();

        // Subsequent calls: all archetypes are skipped (no mutations).
        b.iter(|| {
            let mut count = 0u32;
            plan.execute_stream(&mut world, |_| count += 1).unwrap();
            count
        });
    });

    // ── Aggregates ──────────────────────────────────────────────────
    //
    // Single-pass COUNT + SUM over 10K entities via aggregate.
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

        b.iter(|| plan.aggregate(&mut world).unwrap());
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

        b.iter(|| plan.execute_collect(&mut world).unwrap().len());
    });

    group.finish();
}

fn join_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("join");

    // ── Baseline: execute_collect() + world.get() ────────────────────────────
    group.bench_function("for_each_get_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let entities = plan.execute_collect(&mut world).unwrap();
            let mut sum = 0u64;
            for &entity in entities {
                if let Some(score) = world.get::<Score>(entity) {
                    sum += score.0 as u64;
                }
            }
            sum
        });
    });

    // ── New: execute_stream_batched ───────────────────────────────────────
    group.bench_function("for_each_batched_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, (score,)| {
                sum += score.0 as u64;
            })
            .unwrap();
            sum
        });
    });

    // ── New: for_each_join_chunk ────────────────────────────────────
    group.bench_function("for_each_chunk_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |_, rows, (scores,)| {
                for &row in rows {
                    sum += scores[row].0 as u64;
                }
            })
            .unwrap();
            sum
        });
    });

    // ── Manual baseline: world.query() (no join) ────────────────────
    group.bench_function("manual_query_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);

        b.iter(|| {
            let mut sum = 0u64;
            for (score, _team) in world.query::<(&Score, &Team)>() {
                sum += score.0 as u64;
            }
            sum
        });
    });

    group.finish();
}

fn join_fat_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_fat");

    group.bench_function("for_each_get_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&FatData,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let entities = plan.execute_collect(&mut world).unwrap();
            let mut sum = 0u64;
            for &entity in entities {
                if let Some(fat) = world.get::<FatData>(entity) {
                    sum += fat.data[0] as u64;
                }
            }
            sum
        });
    });

    group.bench_function("for_each_batched_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&FatData,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.execute_stream_batched::<(&FatData,), _>(&mut world, |_, (fat,)| {
                sum += fat.data[0] as u64;
            })
            .unwrap();
            sum
        });
    });

    group.bench_function("for_each_chunk_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&FatData,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.execute_stream_join_chunk::<(&FatData,), _>(&mut world, |_, rows, (fats,)| {
                for &row in rows {
                    sum += fats[row].data[0] as u64;
                }
            })
            .unwrap();
            sum
        });
    });

    group.bench_function("manual_query_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);

        b.iter(|| {
            let mut sum = 0u64;
            for (fat, _team) in world.query::<(&FatData, &Team)>() {
                sum += fat.data[0] as u64;
            }
            sum
        });
    });

    group.finish();
}

fn join_selectivity_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_selectivity");

    for (label, pct) in [("10pct", 0.1), ("50pct", 0.5), ("90pct", 0.9)] {
        group.bench_function(format!("get_{label}"), |b| {
            let (mut world, _) = join_world(10_000, pct);
            let planner = QueryPlanner::new(&world);
            let mut plan = planner
                .scan::<(&Score,)>()
                .join::<(&Team,)>(JoinKind::Inner)
                .build();
            drop(planner);

            b.iter(|| {
                let entities = plan.execute_collect(&mut world).unwrap();
                let mut sum = 0u64;
                for &entity in entities {
                    if let Some(score) = world.get::<Score>(entity) {
                        sum += score.0 as u64;
                    }
                }
                sum
            });
        });

        group.bench_function(format!("chunk_{label}"), |b| {
            let (mut world, _) = join_world(10_000, pct);
            let planner = QueryPlanner::new(&world);
            let mut plan = planner
                .scan::<(&Score,)>()
                .join::<(&Team,)>(JoinKind::Inner)
                .build();
            drop(planner);

            b.iter(|| {
                let mut sum = 0u64;
                plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |_, rows, (scores,)| {
                    for &row in rows {
                        sum += scores[row].0 as u64;
                    }
                })
                .unwrap();
                sum
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    planner,
    join_benches,
    join_fat_benches,
    join_selectivity_benches
);
criterion_main!(benches);
