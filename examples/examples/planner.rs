//! Compiled push-based query planner — cost-based plan compilation with index selection,
//! join optimization, subscription queries, constraint validation, and spatial
//! index execution.
//!
//! Run: cargo run -p minkowski-examples --example planner --release
//!
//! Demonstrates:
//! - Automatic index selection (BTree for ranges, Hash for equality)
//! - Missing index warnings
//! - Join order optimization (hash join vs nested-loop)
//! - Subscription queries with compiler-enforced indexes
//! - Constraint-based validation
//! - EXPLAIN output for plan inspection
//! - Plan execution against a live World (execute returns &[Entity])
//! - Zero-allocation for_each iteration for scan-only plans
//! - Transactional for_each_raw with &World (no tick advancement)
//! - Spatial index execution via add_spatial_index_with_lookup

use std::sync::Arc;

use minkowski::{
    BTreeIndex, CardinalityConstraint, Changed, Entity, HashIndex, Indexed, JoinKind, Predicate,
    QueryPlanner, SpatialCost, SpatialExpr, SpatialIndex, World,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Team(u32);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
#[expect(dead_code)]
struct Vel {
    dx: f32,
    dy: f32,
}

fn main() {
    let mut world = World::new();

    // Spawn 1000 entities across two archetypes
    for i in 0u32..500 {
        world.spawn((
            Score(i),
            Team(i % 5),
            Pos {
                x: i as f32,
                y: 0.0,
            },
        ));
    }
    for i in 500u32..1000 {
        world.spawn((
            Score(i),
            Team(i % 5),
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.0 },
        ));
    }
    println!(
        "Spawned {} entities across 2 archetypes\n",
        world.entity_count()
    );

    // Build indexes
    let mut score_btree = BTreeIndex::<Score>::new();
    score_btree.rebuild(&mut world);
    let score_btree = Arc::new(score_btree);
    let mut team_hash = HashIndex::<Team>::new();
    team_hash.rebuild(&mut world);
    let team_hash = Arc::new(team_hash);

    println!("=== 1. Index Selection ===\n");

    // Create planner and register indexes
    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&score_btree, &world).unwrap();
    planner.add_hash_index(&team_hash, &world).unwrap();

    // Range query on Score — should use BTree index
    let plan = planner
        .scan::<(&Score, &Pos)>()
        .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
        .build();
    println!("{}", plan.explain());

    // Equality on Team — should use Hash index
    let plan = planner
        .scan::<(&Team, &Pos)>()
        .filter(Predicate::eq(Team(2)))
        .build();
    println!("{}", plan.explain());

    println!("=== 2. Missing Index Warnings ===\n");

    // Query on Pos without an index — should warn
    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::eq(Pos { x: 42.0, y: 0.0 }))
        .build();
    println!("{}", plan.explain());

    // Range on Team (only has Hash) — should warn about mismatch
    let plan = planner
        .scan::<(&Team,)>()
        .filter(Predicate::range::<Team, _>(Team(1)..Team(3)))
        .build();
    println!("{}", plan.explain());

    println!("=== 3. Join Optimization ===\n");

    // Large join — should pick hash join
    let plan = planner
        .scan::<(&Score, &Pos)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    println!("Large join (1000x1000):");
    println!("{}", plan.explain());

    // Small join — should pick nested-loop
    let plan = planner
        .scan_with_estimate::<(&Score,)>(10)
        .join::<(&Team,)>(JoinKind::Inner)
        .with_right_estimate(5)
        .unwrap()
        .build();
    println!("Small join (10x5):");
    println!("{}", plan.explain());

    println!("=== 4. Subscription Queries (Compiler-Enforced Indexes) ===\n");

    // Every predicate must provide an Indexed<T> witness — no full scans.
    // Predicates carry the actual value and selectivity estimate.
    let score_witness = Indexed::btree(&score_btree);
    let team_witness = Indexed::hash(&team_hash);

    let sub = planner
        .subscribe::<(&Score, &Team)>()
        .where_eq(
            score_witness,
            Predicate::eq(Score(42)).with_selectivity(0.001), // very selective
        )
        .where_eq(
            team_witness,
            Predicate::eq(Team(2)).with_selectivity(0.2), // less selective
        )
        .build()
        .unwrap();
    println!("{}", sub.explain());

    println!("=== 5. Constraint-Based Optimization ===\n");

    // Build two candidate plans and pick the cheaper one
    let full_scan = planner.scan::<(&Score,)>().build();
    let indexed = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();
    let chosen = full_scan.cheaper(&indexed);
    println!("Full scan cost:   {:.1}", full_scan.cost().total());
    println!("Indexed cost:     {:.1}", indexed.cost().total());
    println!("Chosen plan cost: {:.1}\n", chosen.cost().total());

    // Validate cardinality constraints
    let violations = full_scan.validate_constraints(&[
        ("under_500", CardinalityConstraint::AtMost(500)),
        ("at_least_100", CardinalityConstraint::AtLeast(100)),
    ]);
    if violations.is_empty() {
        println!("All constraints satisfied for full scan plan");
    } else {
        for v in &violations {
            println!("Violation: {v}");
        }
    }

    println!("\n=== 6. Multi-Predicate Optimization ===\n");

    // Multiple predicates — most selective drives the index lookup
    let plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(2)).with_selectivity(0.2))
        .filter(Predicate::range::<Score, _>(Score(100)..Score(200)).with_selectivity(0.1))
        .build();
    println!("{}", plan.explain());

    println!("=== 7. Plan Details ===\n");

    // build() produces a single execution plan — chunked scans, branchless
    // filters, and partitioned joins. No separate "scalar" or "vectorized"
    // paths: LLVM auto-vectorizes the contiguous 64-byte-aligned slices.
    let plan = planner
        .scan::<(&Score, &Pos)>()
        .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    println!("{}", plan.explain());
    println!("Plan cost: {:.1}", plan.cost().total());

    // Drop planner to release the &world borrow before execute() needs &mut world.
    drop(planner);

    println!("=== 8. Plan Execution (execute → &[Entity]) ===\n");

    // Plans aren't just advisory — execute() runs them against the live world.
    // Each block builds a fresh planner because execute() takes &mut World.

    // Simple scan: find all entities with Score and Pos
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score, &Pos)>().build();
        let entities = plan.execute(&mut world).unwrap();
        println!(
            "Scan(&Score, &Pos): {} entities (expected {})",
            entities.len(),
            world.entity_count()
        );
    }

    // Filtered: only Score in [100..200)
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
            .build();
        let entities = plan.execute(&mut world).unwrap();
        println!(
            "Score in [100..200): {} entities (expected 100)",
            entities.len()
        );
        // Verify correctness
        for e in entities {
            let s = world.get::<Score>(*e).unwrap().0;
            assert!((100..200).contains(&s));
        }
    }

    // Index-driven: Team == 2 via hash index
    {
        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&team_hash, &world).unwrap();
        let mut plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(2)))
            .build();
        let entities = plan.execute(&mut world).unwrap();
        println!("Team == 2: {} entities (expected 200)", entities.len());
        for e in entities {
            assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
        }
    }

    // Join execution: intersect Score+Pos entities with Team entities
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score, &Pos)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let entities = plan.execute(&mut world).unwrap();
        println!("(&Score, &Pos) JOIN (&Team,): {} entities", entities.len());
        // All entities have both Score+Pos and Team, so all 1000 match
        assert_eq!(entities.len(), 1000);
    }

    // Custom filter: even scores only
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>("even", 0.5, |w, e| {
                w.get::<Score>(e).is_some_and(|s| s.0 % 2 == 0)
            }))
            .build();
        let entities = plan.execute(&mut world).unwrap();
        println!("Even scores: {} entities (expected 500)", entities.len());
        assert_eq!(entities.len(), 500);
    }

    println!("\n=== 9. Zero-Alloc Scan (for_each) ===\n");

    // for_each avoids the scratch buffer entirely — no intermediate Vec.
    // Ideal for scan-only plans where you process entities one at a time.
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score, &Pos)>().build();
        let mut count = 0;
        plan.for_each(&mut world, |_entity| {
            count += 1;
        })
        .unwrap();
        println!(
            "for_each scan: {count} entities (expected {})",
            world.entity_count()
        );
        assert_eq!(count, world.entity_count());
    }

    // for_each with filter: only high scores
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(900)..Score(1000)))
            .build();
        let mut count = 0;
        plan.for_each(&mut world, |_entity| {
            count += 1;
        })
        .unwrap();
        println!("for_each filtered [900..1000): {count} entities (expected 100)");
        assert_eq!(count, 100);
    }

    println!("\n=== 10. Transactional Read (for_each_raw) ===\n");

    // for_each_raw takes &World instead of &mut World — no tick advancement,
    // no query cache mutation. Designed for use inside transactions where
    // only a shared reference is available.
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score, &Pos)>().build();
        let mut count = 0;
        plan.for_each_raw(&world, |_entity| {
            count += 1;
        })
        .unwrap();
        println!("Read-only scan found {count} entities (no &mut World needed)");
        assert_eq!(count, world.entity_count());
    }

    // for_each_raw with filter
    {
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let mut found = Vec::new();
        plan.for_each_raw(&world, |entity| {
            found.push(entity);
        })
        .unwrap();
        println!(
            "Read-only filtered: {} entity with Score(42) (expected 1)",
            found.len()
        );
        assert_eq!(found.len(), 1);
        assert_eq!(*world.get::<Score>(found[0]).unwrap(), Score(42));
    }

    println!("\n=== 11. Changed<T> Filtering ===\n");

    // Changed<T> skips archetypes whose column tick hasn't advanced since
    // the last for_each call. Entities in different archetypes can be
    // selectively matched — only archetypes with a mutated Score column
    // pass the filter.
    //
    // We need a fresh world for this demo so the tick baseline is clean.
    // Two archetypes: Score-only (5 entities) and Score+Team (5 entities).
    {
        let mut cworld = World::new();
        let score_only: Vec<_> = (0u32..5).map(|i| cworld.spawn((Score(i),))).collect();
        let _score_team: Vec<_> = (0u32..5)
            .map(|i| cworld.spawn((Score(i + 100), Team(i))))
            .collect();

        let planner = QueryPlanner::new(&cworld);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // First call: both archetypes were written at spawn time, so
        // Changed<Score> matches both. Every entity is visible.
        let mut first_count = 0;
        plan.for_each(&mut cworld, |_| first_count += 1).unwrap();
        println!("First for_each (all new): {first_count} entities (expected 10, both archetypes)");
        assert_eq!(first_count, 10);

        // Mutate one entity in the Score-only archetype via get_mut,
        // which marks that column's changed_tick.
        let _ = cworld.get_mut::<Score>(score_only[0]);

        // Second call: only the Score-only archetype column was touched.
        // Score+Team archetype is stale — skipped entirely.
        let mut second_count = 0;
        plan.for_each(&mut cworld, |_| second_count += 1).unwrap();
        println!(
            "Second for_each (one archetype mutated): {second_count} entities (expected 5, Score-only archetype)"
        );
        assert_eq!(second_count, 5);

        // Third call: nothing changed since the last read tick.
        let mut third_count = 0;
        plan.for_each(&mut cworld, |_| third_count += 1).unwrap();
        println!(
            "Third for_each (no new changes): {third_count} entities (expected 0, nothing changed)"
        );
        assert_eq!(third_count, 0);
    }

    println!("\n=== 12. Spatial Index Execution ===\n");

    // A simple spatial index that stores (Entity, x, y) and does a linear
    // distance check. The planner bridges to it via a lookup closure —
    // the planner has no knowledge of the index's internal structure.
    struct LinearSpatialIndex {
        entries: Vec<(Entity, f32, f32)>,
    }

    impl LinearSpatialIndex {
        fn new() -> Self {
            Self {
                entries: Vec::new(),
            }
        }

        /// Return entities within `radius` of `(cx, cy)`.
        fn query_within(&self, cx: f32, cy: f32, radius: f32) -> Vec<Entity> {
            self.entries
                .iter()
                .filter_map(|&(e, x, y)| {
                    let dx = x - cx;
                    let dy = y - cy;
                    if dx * dx + dy * dy <= radius * radius {
                        Some(e)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    impl SpatialIndex for LinearSpatialIndex {
        fn rebuild(&mut self, world: &mut World) {
            self.entries = world
                .query::<(Entity, &Pos)>()
                .map(|(e, p)| (e, p.x, p.y))
                .collect();
        }

        fn supports(&self, expr: &SpatialExpr) -> Option<SpatialCost> {
            match expr {
                SpatialExpr::Within { .. } => Some(SpatialCost {
                    estimated_rows: (self.entries.len() as f64 * 0.05).max(1.0),
                    cpu: 10.0,
                }),
                _ => None,
            }
        }
    }

    // Build and populate the index.
    let mut spatial_idx = LinearSpatialIndex::new();
    spatial_idx.rebuild(&mut world);

    // Entities with Pos.x in [0, 999], y = 0.
    // Center (50, 0) with radius 10 → Score values 40..=60 (21 entities).
    let cx = 50.0f32;
    let radius = 10.0f32;

    // Wrap in Arc so the planner and the lookup closure share ownership.
    let spatial_arc: std::sync::Arc<LinearSpatialIndex> = std::sync::Arc::new(spatial_idx);

    // Distance filter: always applied as post-scan or post-lookup refinement.
    // Required both as the scan-path filter (when no spatial index is available)
    // and as refinement after index-gather (to handle lossy indexes).
    let cx_f64 = cx as f64;
    let radius_f64 = radius as f64;
    let scan_filter = move |world: &World, e: Entity| {
        world.get::<Pos>(e).is_some_and(|p| {
            let dx = p.x as f64 - cx_f64;
            let dy = p.y as f64;
            dx * dx + dy * dy <= radius_f64 * radius_f64
        })
    };

    // The lookup closure: maps SpatialExpr → candidate Entity list.
    // This is the bridge between the planner's expression protocol and the
    // index's concrete query API.
    let spatial_arc_lookup = std::sync::Arc::clone(&spatial_arc);
    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index_with_lookup::<Pos>(
            std::sync::Arc::clone(&spatial_arc) as std::sync::Arc<dyn SpatialIndex + Send + Sync>,
            &world,
            move |expr| match expr {
                SpatialExpr::Within { center, radius } => {
                    assert!(center.len() >= 2, "expected at least 2D coordinates");
                    let qx = center[0] as f32;
                    let qy = center[1] as f32;
                    spatial_arc_lookup.query_within(qx, qy, *radius as f32)
                }
                _ => Vec::new(),
            },
        )
        .unwrap();

    let plan = planner
        .scan::<(&Pos, &Score)>()
        .filter(Predicate::within::<Pos>(vec![cx as f64, 0.0], radius as f64, scan_filter).unwrap())
        .build();

    // EXPLAIN shows SpatialGather as the driving access.
    println!("EXPLAIN spatial within plan:");
    println!("{}", plan.explain());

    // Execute: collect entities near (50, 0) within radius 10.
    let mut plan = plan;
    let spatial_results = plan.execute(&mut world).unwrap();
    println!(
        "Spatial within ({cx}, 0) r={radius}: {} entities",
        spatial_results.len()
    );
    // Verify: all results have Pos.x within [cx-radius, cx+radius] and y=0.
    for &e in spatial_results {
        let p = world.get::<Pos>(e).unwrap();
        let dist = ((p.x - cx) * (p.x - cx) + p.y * p.y).sqrt();
        assert!(
            dist <= radius + f32::EPSILON,
            "entity at ({}, {}) is outside radius {radius}",
            p.x,
            p.y
        );
    }
    println!("All results verified within radius.\n");

    // for_each variant — zero-allocation iteration.
    // Build a fresh spatial index (needs &mut world) before creating the planner.
    let mut spatial_idx2 = LinearSpatialIndex::new();
    spatial_idx2.rebuild(&mut world);
    let arc2 = std::sync::Arc::new(spatial_idx2);
    let arc2c = std::sync::Arc::clone(&arc2);

    let mut planner2 = QueryPlanner::new(&world);
    planner2
        .add_spatial_index_with_lookup::<Pos>(
            arc2 as std::sync::Arc<dyn SpatialIndex + Send + Sync>,
            &world,
            move |expr| match expr {
                SpatialExpr::Within { center, radius } => {
                    let qx = center.first().copied().unwrap_or(0.0) as f32;
                    let qy = center.get(1).copied().unwrap_or(0.0) as f32;
                    arc2c.query_within(qx, qy, *radius as f32)
                }
                _ => Vec::new(),
            },
        )
        .unwrap();

    let scan_filter2 = move |world: &World, e: Entity| {
        world.get::<Pos>(e).is_some_and(|p| {
            let dx = p.x as f64 - cx as f64;
            let dy = p.y as f64;
            dx * dx + dy * dy <= (radius as f64) * (radius as f64)
        })
    };
    let mut plan2 = planner2
        .scan::<(&Pos, &Score)>()
        .filter(
            Predicate::within::<Pos>(vec![cx as f64, 0.0], radius as f64, scan_filter2).unwrap(),
        )
        .build();
    let mut spatial_count = 0;
    plan2.for_each(&mut world, |_e| spatial_count += 1).unwrap();
    println!("for_each spatial: {spatial_count} entities (matches execute result)");
    assert_eq!(spatial_count, spatial_results.len());

    println!("\n=== 13. Index-Driven Execution ===\n");

    // BTree and Hash indexes drive execution directly — the predicate's lookup
    // value is pre-bound at plan-build time (IndexDriver), so the execution
    // path never handles dyn Any. Same validation as spatial: is_alive →
    // archetype location → required components → Changed<T> → filter refinement.

    // BTree eq lookup: find entities with Score(42).
    {
        let mut btree_exec = BTreeIndex::<Score>::new();
        btree_exec.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_btree_index(&Arc::new(btree_exec), &world)
            .unwrap();

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        println!("EXPLAIN index eq plan:");
        println!("{}", plan.explain());

        let mut plan = plan;
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        println!("Index eq lookup Score(42): {count} entities");
        // Score(42) was spawned once in the range 0..500.
        assert_eq!(count, 1);
    }

    // Hash eq lookup: find entities with Team(2).
    {
        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&team_hash, &world).unwrap();

        let plan = planner
            .scan::<(&Team,)>()
            .filter(Predicate::eq::<Team>(Team(2)))
            .build();

        println!("\nEXPLAIN hash eq plan:");
        println!("{}", plan.explain());

        let mut plan = plan;
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        println!("Hash eq lookup Team(2): {count} entities (expected 200)");
        assert_eq!(count, 200);
    }

    // BTree range lookup: find entities with Score in [100..200).
    {
        let mut btree_range = BTreeIndex::<Score>::new();
        btree_range.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_btree_index(&Arc::new(btree_range), &world)
            .unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
            .build();

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        println!("BTree range lookup Score(100..200): {count} entities (expected 100)");
        assert_eq!(count, 100);
    }

    // ── 14. Subscription with Changed<T> ──────────────────────────────
    //
    // Separate scope: QueryPlanner borrows &world, but for_each needs &mut world.
    // Build the plan, drop the planner, then execute.
    println!("\n=== 14. Subscription with Changed<T> ===\n");
    {
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let score_idx = Arc::new(score_idx);
        let witness = Indexed::btree(&*score_idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&score_idx, &world).unwrap();

        // Changed<Score> in the query type means: only yield entities whose
        // Score column was mutated since the last call to this plan.
        let mut sub = planner
            .subscribe::<(Changed<Score>, &Score)>()
            .where_eq(witness, Predicate::eq(Score(42)).with_selectivity(0.001))
            .build()
            .unwrap();

        // First call: everything is "new" (never read before).
        let mut first_count = 0;
        sub.for_each(&mut world, |_| first_count += 1).unwrap();
        println!("First call (all new):     {first_count} entities");

        // Second call: nothing changed → zero results.
        let mut second_count = 0;
        sub.for_each(&mut world, |_| second_count += 1).unwrap();
        println!("Second call (no changes): {second_count} entities");
        assert_eq!(second_count, 0);
    }

    println!("\nDone.");
}
