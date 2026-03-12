//! Volcano query planner — cost-based plan compilation with index selection,
//! join optimization, subscription queries, and constraint validation.
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
//! - Plan execution against a live World

use std::sync::Arc;

use minkowski::{
    BTreeIndex, CardinalityConstraint, HashIndex, Indexed, JoinKind, Predicate, QueryPlanner,
    SpatialIndex, VectorizeOpts, World,
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
    planner.add_btree_index(&score_btree, &world);
    planner.add_hash_index(&team_hash, &world);

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
        .build();
    println!("Small join (10x5):");
    println!("{}", plan.explain());

    println!("=== 4. Subscription Queries (Compiler-Enforced Indexes) ===\n");

    // Every predicate must provide an Indexed<T> witness — no full scans
    let score_witness = Indexed::btree(&score_btree);
    let team_witness = Indexed::hash(&team_hash);

    let sub = planner
        .subscribe::<(&Score, &Team)>()
        .where_eq(score_witness, 0.001) // very selective
        .where_eq(team_witness, 0.2) // less selective
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

    println!("=== 7. Vectorized Execution (Default) ===\n");

    // build() compiles to vectorized execution by default.
    // Scans become chunked, filters become SIMD-friendly, joins are partitioned.
    let plan = planner
        .scan::<(&Score, &Pos)>()
        .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    println!("{}", plan.explain());

    // Compare vectorized (default) vs logical cost
    println!("Logical plan cost:    {:.1}", plan.logical_cost().total());
    println!("Vectorized plan cost: {:.1}", plan.cost().total());

    // Re-lower with custom opts for a different cache hierarchy
    let small_cache_opts = VectorizeOpts {
        l2_cache_bytes: 128 * 1024,
        avg_component_bytes: 32,
        target_chunk_rows: 1024,
    };
    let vec_plan_small = plan.vectorize(small_cache_opts);
    println!("\nWith 128 KiB L2 cache, 32-byte components:");
    println!("{}", vec_plan_small.explain());

    println!("=== 8. Plan Execution ===\n");

    // Plans aren't just advisory — execute() runs them against the live world.

    // Simple scan: find all entities with Score and Pos
    let plan = planner.scan::<(&Score, &Pos)>().build();
    let entities = plan.execute(&world);
    println!(
        "Scan(&Score, &Pos): {} entities (expected {})",
        entities.len(),
        world.entity_count()
    );

    // Filtered: only Score in [100..200)
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
        .build();
    let entities = plan.execute(&world);
    println!(
        "Score in [100..200): {} entities (expected 100)",
        entities.len()
    );
    // Verify correctness
    for e in &entities {
        let s = world.get::<Score>(*e).unwrap().0;
        assert!((100..200).contains(&s));
    }

    // Index-driven: Team == 2 via hash index
    let plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(2)))
        .build();
    let entities = plan.execute(&world);
    println!("Team == 2: {} entities (expected 200)", entities.len());
    for e in &entities {
        assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
    }

    // Join execution: intersect Score+Pos entities with Team entities
    let plan = planner
        .scan::<(&Score, &Pos)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let entities = plan.execute(&world);
    println!("(&Score, &Pos) JOIN (&Team,): {} entities", entities.len());
    // All entities have both Score+Pos and Team, so all 1000 match
    assert_eq!(entities.len(), 1000);

    // Custom filter: even scores only
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>("even", 0.5, |w, e| {
            w.get::<Score>(e).is_some_and(|s| s.0 % 2 == 0)
        }))
        .build();
    let entities = plan.execute(&world);
    println!("Even scores: {} entities (expected 500)", entities.len());
    assert_eq!(entities.len(), 500);

    println!("\nDone.");
}
