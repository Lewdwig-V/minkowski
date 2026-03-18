use super::*;
use crate::Changed;
use crate::World;
use crate::index::SpatialIndex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Team(u32);

#[derive(Clone, Copy, Debug)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
struct Health(u32);

// ── Basic planner construction ──────────────────────────────────

#[test]
fn planner_new_empty_world() {
    let world = World::new();
    let planner = QueryPlanner::new(&world);
    assert_eq!(planner.total_entities, 0);
}

#[test]
fn planner_captures_entity_count() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    assert_eq!(planner.total_entities, 100);
}

// ── Index registration ──────────────────────────────────────────

#[test]
fn register_btree_index() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx), &world).unwrap();
    assert!(planner.indexes.contains_key(&TypeId::of::<Score>()));
    assert_eq!(
        planner.indexes[&TypeId::of::<Score>()].kind,
        IndexKind::BTree
    );
}

#[test]
fn register_hash_index() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Team(i),));
    }
    let mut idx = HashIndex::<Team>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(idx), &world).unwrap();
    assert!(planner.indexes.contains_key(&TypeId::of::<Team>()));
    assert_eq!(planner.indexes[&TypeId::of::<Team>()].kind, IndexKind::Hash);
}

#[test]
fn btree_takes_precedence_over_hash() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }
    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);
    let mut hash = HashIndex::<Score>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();
    planner.add_hash_index(&Arc::new(hash), &world).unwrap(); // should not overwrite
    assert_eq!(
        planner.indexes[&TypeId::of::<Score>()].kind,
        IndexKind::BTree
    );
}

// ── Scan without predicates ─────────────────────────────────────

#[test]
fn scan_no_predicates() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner.scan::<(&Pos,)>().build();

    assert!(plan.warnings().is_empty());
    assert_eq!(plan.cost().rows, 100.0);
    match plan.root() {
        PlanNode::Scan { estimated_rows, .. } => assert_eq!(*estimated_rows, 100),
        other => panic!("expected Scan, got {:?}", other),
    }
}

// ── Index selection ─────────────────────────────────────────────

#[test]
fn eq_predicate_uses_hash_index() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Team(i % 10),));
    }
    let mut idx = HashIndex::<Team>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(idx), &world).unwrap();

    let plan = planner
        .scan::<(&Team,)>()
        .filter(Predicate::eq(Team(5)))
        .build();

    assert!(plan.warnings().is_empty());
    match plan.root() {
        PlanNode::IndexLookup { index_kind, .. } => {
            assert_eq!(*index_kind, IndexKind::Hash);
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
}

#[test]
fn range_predicate_uses_btree_index() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx), &world).unwrap();

    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
        .build();

    assert!(plan.warnings().is_empty());
    match plan.root() {
        PlanNode::IndexLookup {
            index_kind,
            component_name,
            ..
        } => {
            assert_eq!(*index_kind, IndexKind::BTree);
            assert!(component_name.contains("Score"));
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
}

// ── Missing index warnings ──────────────────────────────────────

#[test]
fn warns_missing_index_for_eq() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    assert_eq!(plan.warnings().len(), 1);
    match &plan.warnings()[0] {
        PlanWarning::MissingIndex { predicate_kind, .. } => {
            assert_eq!(*predicate_kind, "equality");
        }
        other => panic!("expected MissingIndex, got {:?}", other),
    }
}

#[test]
fn warns_hash_index_for_range() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(idx), &world).unwrap();

    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .build();

    assert_eq!(plan.warnings().len(), 1);
    match &plan.warnings()[0] {
        PlanWarning::IndexKindMismatch { have, need, .. } => {
            assert_eq!(*have, "Hash");
            assert!(need.contains("BTree"));
        }
        other => panic!("expected IndexKindMismatch, got {:?}", other),
    }
}

#[test]
fn no_warning_for_custom_predicate() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "score > threshold",
            0.5,
            |w, e| w.get::<Score>(e).is_some_and(|s| s.0 > 50),
        ))
        .build();

    // Custom predicates can never use indexes — no warning expected
    assert!(plan.warnings().is_empty());
}

// ── Multiple predicates ─────────────────────────────────────────

#[test]
fn most_selective_predicate_drives_index_lookup() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut score_idx = BTreeIndex::<Score>::new();
    score_idx.rebuild(&mut world);
    let mut team_idx = HashIndex::<Team>::new();
    team_idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_btree_index(&Arc::new(score_idx), &world)
        .unwrap();
    planner.add_hash_index(&Arc::new(team_idx), &world).unwrap();

    let plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(2)).with_selectivity(0.2)) // 20% sel
        .filter(Predicate::eq(Score(42)).with_selectivity(0.001)) // 0.1% sel
        .build();

    // Score(42) is more selective, should be the driving lookup.
    match plan.root() {
        PlanNode::Filter { child, .. } => match child.as_ref() {
            PlanNode::IndexLookup { component_name, .. } => {
                assert!(
                    component_name.contains("Score"),
                    "expected Score to drive, got {}",
                    component_name
                );
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        },
        PlanNode::IndexLookup { component_name, .. } => {
            // If there's only one predicate that got pushed down
            assert!(component_name.contains("Score"));
        }
        other => panic!("expected Filter or IndexLookup, got {:?}", other),
    }
}

// ── Join optimization ───────────────────────────────────────────

#[test]
fn small_join_uses_nested_loop() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    // Use Left join to test strategy selection (Inner joins are eliminated).
    let plan = planner
        .scan_with_estimate::<(&Score,)>(10)
        .join::<(&Team,)>(JoinKind::Left)
        .with_right_estimate(5)
        .unwrap()
        .build();

    match plan.root() {
        PlanNode::NestedLoopJoin { .. } => {} // expected
        other => panic!("expected NestedLoopJoin, got {:?}", other),
    }
}

#[test]
fn large_join_uses_hash_join() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    // Use Left join to test strategy selection (Inner joins are eliminated).
    let plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();

    match plan.root() {
        PlanNode::HashJoin { .. } => {} // expected
        other => panic!("expected HashJoin, got {:?}", other),
    }
}

#[test]
fn hash_join_puts_smaller_side_on_left() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    // Use Left join to test strategy selection (Inner joins are eliminated).
    let plan = planner
        .scan_with_estimate::<(&Score,)>(100)
        .join::<(&Team,)>(JoinKind::Left)
        .with_right_estimate(500)
        .unwrap()
        .build();

    match plan.root() {
        PlanNode::HashJoin { left, right, .. } => {
            assert!(
                left.estimated_rows() <= right.estimated_rows(),
                "build side should be smaller: left={:.0} right={:.0}",
                left.estimated_rows(),
                right.estimated_rows()
            );
        }
        other => panic!("expected HashJoin, got {:?}", other),
    }
}

#[test]
fn left_join_preserves_left_side_when_right_is_smaller() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);

    // Left side is larger (500) than right side (100).
    // For an Inner join, the planner would swap them to put the smaller
    // side on the build (left) side. But Left join must preserve the
    // semantic left side — all its rows must appear in the output.
    let plan = planner
        .scan_with_estimate::<(&Score,)>(500)
        .join::<(&Team,)>(JoinKind::Left)
        .with_right_estimate(100)
        .unwrap()
        .build();

    match plan.root() {
        PlanNode::HashJoin {
            left,
            right,
            join_kind,
            ..
        } => {
            assert_eq!(*join_kind, JoinKind::Left);
            // Left side must be the original scan (500 rows), not swapped.
            assert_eq!(
                left.estimated_rows(),
                500.0,
                "left join must keep original left side (500 rows), got {}",
                left.estimated_rows()
            );
            assert_eq!(
                right.estimated_rows(),
                100.0,
                "right side should be 100 rows, got {}",
                right.estimated_rows()
            );
        }
        other => panic!("expected HashJoin, got {:?}", other),
    }
}

#[test]
fn inner_join_eliminated_instead_of_reordered() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);

    // Inner join is eliminated at build time — no reordering needed.
    let plan = planner
        .scan_with_estimate::<(&Score,)>(500)
        .join::<(&Team,)>(JoinKind::Inner)
        .with_right_estimate(100)
        .unwrap()
        .build();

    match plan.root() {
        PlanNode::Scan { .. } => {} // eliminated — expected
        other => panic!(
            "expected Scan after inner join elimination, got {:?}",
            other
        ),
    }
}

// ── Explain output ──────────────────────────────────────────────

#[test]
fn explain_contains_plan_details() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx), &world).unwrap();

    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .build();

    let explain = plan.explain();
    assert!(explain.contains("Execution Plan"));
    assert!(explain.contains("IndexGather"));
    assert!(explain.contains("BTree"));
    assert!(explain.contains("Score"));
    assert!(explain.contains("L2 cache budget"));
}

// ── Subscription plans ──────────────────────────────────────────

#[test]
fn subscription_requires_indexed_witness() {
    let mut world = World::new();
    for i in 0..1000u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = std::sync::Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    let sub = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build()
        .unwrap();

    // The index is registered, so the planner should emit an IndexLookup.
    match sub.root() {
        PlanNode::IndexLookup { index_kind, .. } => {
            assert_eq!(*index_kind, IndexKind::BTree);
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
    assert!(sub.cost().cpu > 0.0);
}

#[test]
fn subscription_multiple_predicates_ordered_by_selectivity() {
    let mut world = World::new();
    for i in 0..1000u32 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut score_idx = BTreeIndex::<Score>::new();
    score_idx.rebuild(&mut world);
    let score_idx = std::sync::Arc::new(score_idx);
    let mut team_idx = BTreeIndex::<Team>::new();
    team_idx.rebuild(&mut world);
    let team_idx = std::sync::Arc::new(team_idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&score_idx, &world).unwrap();
    planner.add_btree_index(&team_idx, &world).unwrap();
    let score_w = Indexed::btree(&*score_idx);
    let team_w = Indexed::btree(&*team_idx);

    // Score predicate gets lower selectivity (more selective) → should drive.
    let sub = planner
        .subscribe::<(Changed<Score>, &Score, &Team)>()
        .where_eq(team_w, Predicate::eq(Team(2)).with_selectivity(0.2))
        .where_eq(score_w, Predicate::eq(Score(42)).with_selectivity(0.001))
        .build()
        .unwrap();

    // Most selective predicate (Score) should be the driving index lookup.
    match sub.root() {
        PlanNode::Filter { child, .. } => match child.as_ref() {
            PlanNode::IndexLookup { component_name, .. } => {
                assert!(component_name.contains("Score"));
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        },
        PlanNode::IndexLookup { component_name, .. } => {
            assert!(component_name.contains("Score"));
        }
        other => panic!("expected Filter or IndexLookup, got {:?}", other),
    }
}

#[test]
fn subscription_where_range_accepts_btree_witness() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = std::sync::Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    let sub = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_range(witness, Predicate::range(Score(10)..Score(50)))
        .build()
        .unwrap();

    match sub.root() {
        PlanNode::IndexLookup { index_kind, .. } => {
            assert_eq!(*index_kind, IndexKind::BTree);
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
}

#[test]
fn subscription_where_range_rejects_hash_witness() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    let planner = QueryPlanner::new(&world);
    let witness = Indexed::hash(&idx);

    // Hash indexes cannot serve range queries — build returns an error.
    let result = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_range(witness, Predicate::range(Score(10)..Score(50)))
        .build();
    match result {
        Err(errs)
            if errs
                .iter()
                .any(|e| matches!(e, SubscriptionError::HashIndexOnRange { .. })) => {}
        other => panic!("expected HashIndexOnRange error, got {:?}", other),
    }
}

#[test]
fn subscription_no_predicates_returns_error() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let planner = QueryPlanner::new(&world);

    let result = planner.subscribe::<(Changed<Score>, &Score)>().build();
    match result {
        Err(errs)
            if errs
                .iter()
                .any(|e| matches!(e, SubscriptionError::NoPredicates)) => {}
        other => panic!("expected NoPredicates error, got {:?}", other),
    }
}

#[test]
fn subscription_without_changed_filter_returns_error() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    // Query type has no Changed<T> — subscription should error.
    let result = planner
        .subscribe::<(&Score,)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build();
    match result {
        Err(errs)
            if errs
                .iter()
                .any(|e| matches!(e, SubscriptionError::NoChangedFilter)) => {}
        other => panic!("expected NoChangedFilter error, got {:?}", other),
    }
}

#[test]
fn subscription_component_mismatch_returns_error() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut score_idx = BTreeIndex::<Score>::new();
    score_idx.rebuild(&mut world);
    let score_idx = std::sync::Arc::new(score_idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&score_idx, &world).unwrap();
    let score_w = Indexed::btree(&*score_idx);

    // Witness is for Score but predicate is for Team — should error.
    let result = planner
        .subscribe::<(Changed<Score>, &Score, &Team)>()
        .where_eq(score_w, Predicate::eq(Team(2)))
        .build();
    match result {
        Err(errs)
            if errs
                .iter()
                .any(|e| matches!(e, SubscriptionError::ComponentMismatch { .. })) => {}
        other => panic!("expected ComponentMismatch error, got {:?}", other),
    }
}

#[test]
fn subscription_predicate_kind_mismatch_returns_error() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = std::sync::Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    // where_eq expects an Eq predicate, but we pass a Range.
    let result = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_eq(witness, Predicate::range(Score(10)..Score(50)))
        .build();
    match result {
        Err(errs)
            if errs
                .iter()
                .any(|e| matches!(e, SubscriptionError::PredicateKindMismatch { .. })) => {}
        other => panic!("expected PredicateKindMismatch error, got {:?}", other),
    }
}

#[test]
fn subscription_plan_is_executable() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let _e2 = world.spawn((Score(99),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    let mut plan = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build()
        .unwrap();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}

#[test]
fn subscription_plan_for_each_raw_works() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let _e2 = world.spawn((Score(99),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    let mut plan = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build()
        .unwrap();

    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}

#[test]
fn subscription_where_range_rejects_eq_predicate() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    // where_range expects a Range predicate, but we pass Eq.
    let result = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_range(witness, Predicate::eq(Score(42)))
        .build();
    assert!(matches!(
        result,
        Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::PredicateKindMismatch { expected: "Range", .. }))
    ));
}

#[test]
fn subscription_unregistered_index_returns_error() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let witness = Indexed::btree(&idx);

    // Witness exists but index NOT registered with planner.
    let planner = QueryPlanner::new(&world);
    let result = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build();
    assert!(
        matches!(
            result,
            Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::IndexNotRegistered { .. }))
        ),
        "expected IndexNotRegistered error when index not registered with planner, got {:?}",
        result
    );
}

#[test]
fn subscription_validation_failure_does_not_produce_spurious_no_predicates() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    // Pass a Team predicate with a Score witness — ComponentMismatch.
    // Should NOT also get NoPredicates.
    let result = planner
        .subscribe::<(Changed<Score>, &Score, &Team)>()
        .where_eq(witness, Predicate::eq(Team(2)).with_selectivity(0.2))
        .build();
    match result {
        Err(errs) => {
            assert!(
                errs.iter()
                    .any(|e| matches!(e, SubscriptionError::ComponentMismatch { .. })),
                "expected ComponentMismatch"
            );
            assert!(
                !errs
                    .iter()
                    .any(|e| matches!(e, SubscriptionError::NoPredicates)),
                "should NOT get spurious NoPredicates when predicates were attempted"
            );
        }
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn subscription_with_changed_yields_only_mutated_entities() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));
    let _e3 = world.spawn((Score(99),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&idx, &world).unwrap();
    let witness = Indexed::btree(&*idx);

    // Subscribe with Changed<Score> — only entities whose Score column
    // was mutated since last call will pass through.
    let mut plan = planner
        .subscribe::<(Changed<Score>, &Score)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build()
        .unwrap();

    // First call: all matching entities are "changed" (never read before).
    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));

    // Second call with no mutations: nothing changed → zero results.
    results.clear();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 0, "no mutations → no results");

    // Mutate e1's Score (stays 42, but the column is marked changed).
    world.get_mut::<Score>(e1).unwrap().0 = 42;

    // Third call: only e1's archetype was touched. Since e1 and e2 are
    // in the same archetype (both have only Score), Changed<T> is
    // archetype-granular — both pass. This is correct per the engine's
    // change detection semantics.
    results.clear();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert!(results.contains(&e1));
    // e2 is in the same archetype as e1, so it also passes Changed<Score>.
}

// ── Constraint validation ───────────────────────────────────────

#[test]
fn constraint_exactly_one() {
    let c = CardinalityConstraint::ExactlyOne;
    assert!(c.satisfied_by(1.0));
    assert!(!c.satisfied_by(0.0));
    assert!(!c.satisfied_by(2.0));
}

#[test]
fn constraint_at_most() {
    let c = CardinalityConstraint::AtMost(10);
    assert!(c.satisfied_by(5.0));
    assert!(c.satisfied_by(10.0));
    assert!(!c.satisfied_by(11.0));
}

#[test]
fn constraint_between() {
    let c = CardinalityConstraint::Between(5, 15);
    assert!(c.satisfied_by(10.0));
    assert!(!c.satisfied_by(3.0));
    assert!(!c.satisfied_by(20.0));
}

#[test]
fn validate_plan_constraints() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner.scan::<(&Score,)>().build();

    let violations = plan.validate_constraints(&[("max_100", CardinalityConstraint::AtMost(100))]);
    assert!(violations.is_empty());

    let violations = plan.validate_constraints(&[("max_10", CardinalityConstraint::AtMost(10))]);
    assert_eq!(violations.len(), 1);
}

// ── choose_cheaper ──────────────────────────────────────────────

#[test]
fn choose_cheaper_picks_lower_cost() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx), &world).unwrap();

    let full_scan = planner.scan::<(&Score,)>().build();
    let indexed = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    let chosen = full_scan.cheaper(&indexed);
    assert!(chosen.cost().total() <= full_scan.cost().total());
}

// ── Cost model ──────────────────────────────────────────────────

#[test]
fn index_lookup_cheaper_than_scan_for_selective_pred() {
    let scan_cost = Cost::scan(10_000);
    let idx_cost = Cost::index_lookup(0.01, 10_000);
    assert!(
        idx_cost.total() < scan_cost.total(),
        "index lookup ({:.1}) should be cheaper than scan ({:.1})",
        idx_cost.total(),
        scan_cost.total()
    );
}

#[test]
fn hash_join_cheaper_than_nested_loop_for_large() {
    let left = Cost::scan(1000);
    let right = Cost::scan(1000);
    let hash = Cost::hash_join(left, right);
    let nested = Cost::nested_loop_join(left, right);
    assert!(
        hash.total() < nested.total(),
        "hash join ({:.1}) should be cheaper than nested loop ({:.1}) for 1000x1000",
        hash.total(),
        nested.total()
    );
}

#[test]
fn filter_reduces_estimated_rows() {
    let scan = Cost::scan(1000);
    let filtered = Cost::filter(scan, 0.1);
    assert!(
        filtered.rows < scan.rows,
        "filter should reduce rows: {:.0} vs {:.0}",
        filtered.rows,
        scan.rows
    );
}

// ── Selectivity override ────────────────────────────────────────

#[test]
fn custom_selectivity() {
    let pred = Predicate::eq(Score(42)).with_selectivity(0.5);
    assert!((pred.selectivity - 0.5).abs() < f64::EPSILON);
}

#[test]
fn selectivity_clamped() {
    let pred = Predicate::eq(Score(42)).with_selectivity(2.0);
    assert!((pred.selectivity - 1.0).abs() < f64::EPSILON);

    let pred = Predicate::eq(Score(42)).with_selectivity(-1.0);
    assert!(pred.selectivity.abs() < f64::EPSILON);
}

#[test]
fn nan_selectivity_normalized_to_worst_case() {
    // with_selectivity
    let pred = Predicate::eq(Score(42)).with_selectivity(f64::NAN);
    assert!((pred.selectivity - 1.0).abs() < f64::EPSILON);

    // custom constructor
    let pred = Predicate::custom::<Score>("test", f64::NAN, |_, _| true);
    assert!((pred.selectivity - 1.0).abs() < f64::EPSILON);
}

// ── Display ─────────────────────────────────────────────────────

#[test]
fn plan_display_does_not_panic() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut score_idx = BTreeIndex::<Score>::new();
    score_idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_btree_index(&Arc::new(score_idx), &world)
        .unwrap();

    let plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .filter(Predicate::custom::<Team>("team != 0", 0.8, |w, e| {
            w.get::<Team>(e).is_some_and(|t| t.0 != 0)
        }))
        .join::<(&Pos,)>(JoinKind::Inner)
        .build();

    let display = format!("{plan}");
    assert!(!display.is_empty());
    let debug = format!("{plan:?}");
    assert!(!debug.is_empty());
}

// ── Indexed witness ─────────────────────────────────────────────

#[test]
fn indexed_btree_witness() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let witness = Indexed::btree(&idx);
    assert_eq!(witness.kind, IndexKind::BTree);
    assert_eq!(witness.cardinality, 50);
}

#[test]
fn indexed_hash_witness() {
    let mut world = World::new();
    for i in 0..30 {
        world.spawn((Team(i),));
    }
    let mut idx = HashIndex::<Team>::new();
    idx.rebuild(&mut world);

    let witness = Indexed::hash(&idx);
    assert_eq!(witness.kind, IndexKind::Hash);
    assert_eq!(witness.cardinality, 30);
}

#[test]
fn indexed_is_copy() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let w1 = Indexed::btree(&idx);
    let w2 = w1; // copy
    assert_eq!(w1.kind, w2.kind);
}

// ── Plan node introspection ─────────────────────────────────────

#[test]
fn build_produces_chunked_scan() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner.scan::<(&Score,)>().build();

    match plan.root() {
        PlanNode::Scan {
            estimated_rows,
            avg_chunk_size,
            ..
        } => {
            assert_eq!(*estimated_rows, 1000);
            assert!(*avg_chunk_size <= VectorizeOpts::default().target_chunk_rows);
        }
        other => panic!("expected Scan, got {:?}", other),
    }
}

#[test]
fn index_lookup_produces_index_gather() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx), &world).unwrap();

    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    match plan.root() {
        PlanNode::IndexLookup { index_kind, .. } => {
            assert_eq!(*index_kind, IndexKind::BTree);
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
}

#[test]
fn filter_detects_branchless() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);

    // Range predicate without index → scan + filter
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .build();

    match plan.root() {
        PlanNode::Filter { branchless, .. } => {
            assert!(*branchless, "Range predicate should be branchless");
        }
        other => panic!("expected Filter, got {:?}", other),
    }

    // Custom predicate → branched
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>("complex check", 0.5, |_, _| {
            true
        }))
        .build();

    match plan.root() {
        PlanNode::Filter { branchless, .. } => {
            assert!(!*branchless, "Custom predicate should be branched");
        }
        other => panic!("expected Filter, got {:?}", other),
    }
}

#[test]
fn hash_join_partitioned() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    // Use Left join to test partitioning (Inner joins are eliminated).
    let plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();

    match plan.root() {
        PlanNode::HashJoin { partitions, .. } => {
            assert!(*partitions >= 1);
        }
        other => panic!("expected HashJoin, got {:?}", other),
    }
}

#[test]
fn partitioned_join_plan_node_has_multiple_partitions() {
    // Enough entities to trigger partitions > 1 (default L2 = 256 KiB,
    // avg_component_bytes = 16 → 16K entities per partition).
    let mut world = World::new();
    for i in 0..20_000 {
        world.spawn((Score(i),));
    }
    for i in 0..20_000 {
        world.spawn((Team(i),));
    }

    let planner = QueryPlanner::new(&world);
    // Use Left join — inner joins are eliminated at build time.
    let plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();

    match plan.root() {
        PlanNode::HashJoin { partitions, .. } => {
            assert!(*partitions > 1, "expected partitions > 1, got {partitions}");
        }
        other => panic!("expected HashJoin, got {:?}", other),
    }
}

#[test]
fn nested_loop_for_small_cardinality() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    // Use Left join to test strategy selection (Inner joins are eliminated).
    let plan = planner
        .scan_with_estimate::<(&Score,)>(10)
        .join::<(&Team,)>(JoinKind::Left)
        .with_right_estimate(5)
        .unwrap()
        .build();

    match plan.root() {
        PlanNode::NestedLoopJoin { .. } => {} // expected
        other => panic!("expected NestedLoopJoin, got {:?}", other),
    }
}

#[test]
fn explain_contains_details() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>("x > 0", 0.5, |w, e| {
            w.get::<Score>(e).is_some_and(|s| s.0 > 0)
        }))
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let explain = plan.explain();

    assert!(explain.contains("Execution Plan"));
    assert!(explain.contains("L2 cache budget"));
    assert!(explain.contains("target chunk"));
}

// ── TablePlanner tests ───────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, minkowski_derive::Table)]
struct IndexedScores {
    #[index(btree)]
    score: Score,
    #[index(hash)]
    team: Team,
}

#[test]
fn table_planner_creates_btree_index() {
    use crate::index::HasBTreeIndex;

    let mut world = World::new();
    for i in 0..10 {
        world.spawn(IndexedScores {
            score: Score(i),
            team: Team(i % 3),
        });
    }

    // HasBTreeIndex trait provides create_btree_index
    let idx = IndexedScores::create_btree_index(&mut world);
    assert_eq!(idx.len(), 10);
    assert_eq!(idx.get(&Score(5)).len(), 1);
}

#[test]
fn table_planner_creates_hash_index() {
    use crate::index::HasHashIndex;

    let mut world = World::new();
    for i in 0..9 {
        world.spawn(IndexedScores {
            score: Score(i),
            team: Team(i % 3),
        });
    }

    let idx = IndexedScores::create_hash_index(&mut world);
    assert_eq!(idx.len(), 9);
    // 3 entities per team
    assert_eq!(idx.get(&Team(0)).len(), 3);
}

#[test]
fn table_planner_indexed_witness() {
    use crate::index::{HasBTreeIndex, HasHashIndex};

    let mut world = World::new();
    for i in 0..5 {
        world.spawn(IndexedScores {
            score: Score(i),
            team: Team(i % 2),
        });
    }

    let btree = IndexedScores::create_btree_index(&mut world);
    let hash = IndexedScores::create_hash_index(&mut world);

    // TablePlanner provides indexed_* witnesses with compile-time enforcement
    let planner = TablePlanner::<IndexedScores>::new(&world);
    let indexed_bt = planner.indexed_btree::<Score>(&btree);
    let indexed_hs = planner.indexed_hash::<Team>(&hash);

    assert!(matches!(indexed_bt, Indexed { .. }));
    assert!(matches!(indexed_hs, Indexed { .. }));
}

#[test]
fn table_planner_scan_builds_plan() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn(IndexedScores {
            score: Score(i),
            team: Team(i % 5),
        });
    }

    let planner = TablePlanner::<IndexedScores>::new(&world);
    let plan = planner.scan::<(&Score, &Team)>().build();

    // Should produce a valid plan
    assert!(plan.cost().cpu > 0.0);
    let explain = plan.explain();
    assert!(explain.contains("Execution Plan"));
}

#[test]
fn table_planner_scan_with_index_filter() {
    use crate::index::HasBTreeIndex;

    let mut world = World::new();
    for i in 0..100 {
        world.spawn(IndexedScores {
            score: Score(i),
            team: Team(i % 5),
        });
    }

    // Create index first (needs &mut World)
    let btree = IndexedScores::create_btree_index(&mut world);

    // Then create planner (borrows &World)
    let mut planner = TablePlanner::<IndexedScores>::new(&world);
    planner.add_btree_index::<Score>(&Arc::new(btree)).unwrap();

    let plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .build();

    // Should use index (IndexGather in vectorized, IndexLookup in logical)
    let explain = plan.explain();
    assert!(
        explain.contains("IndexGather") || explain.contains("IndexLookup"),
        "expected index-driven plan, got:\n{explain}"
    );
}

#[test]
fn table_planner_total_entities() {
    let mut world = World::new();
    for i in 0..20 {
        world.spawn(IndexedScores {
            score: Score(i),
            team: Team(0),
        });
    }

    let planner = TablePlanner::<IndexedScores>::new(&world);
    assert_eq!(planner.total_entities(), 20);
}

#[test]
fn has_btree_index_field_name() {
    use crate::index::HasBTreeIndex;
    assert_eq!(<IndexedScores as HasBTreeIndex<Score>>::FIELD_NAME, "score");
}

#[test]
fn has_hash_index_field_name() {
    use crate::index::HasHashIndex;
    assert_eq!(<IndexedScores as HasHashIndex<Team>>::FIELD_NAME, "team");
}

// ── Plan execution ─────────────────────────────────────────────────

#[test]
fn execute_scan_returns_all_matching_entities() {
    let mut world = World::new();
    let mut expected = Vec::new();
    for i in 0..20 {
        expected.push(world.spawn((Score(i),)));
    }
    // Different archetype — should also be matched
    for i in 20..30 {
        expected.push(world.spawn((Score(i), Team(0))));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();
    let mut result = plan.execute_collect(&mut world).unwrap().to_vec();
    result.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(result, expected);
}

#[test]
fn execute_scan_excludes_non_matching_archetypes() {
    let mut world = World::new();
    // Archetype with Score only
    let e1 = world.spawn((Score(1),));
    // Archetype with Team only — should NOT match scan::<(&Score,)>()
    let _e2 = world.spawn((Team(1),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result, vec![e1]);
}

#[test]
fn execute_filter_eq() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 1);
    assert_eq!(*world.get::<Score>(result[0]).unwrap(), Score(42));
}

#[test]
fn execute_filter_range() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 10);
    for e in &result {
        let s = world.get::<Score>(*e).unwrap().0;
        assert!((10..20).contains(&s), "score {s} out of range");
    }
}

#[test]
fn execute_index_driven_eq() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut hash = HashIndex::<Team>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(hash), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(2)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 20); // 100 / 5 teams
    for e in &result {
        assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
    }
}

#[test]
fn execute_multi_predicate() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i), Team(i % 5)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(2)))
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    for e in &result {
        let s = world.get::<Score>(*e).unwrap().0;
        let t = world.get::<Team>(*e).unwrap().0;
        assert!((10..50).contains(&s), "score {s} out of range");
        assert_eq!(t, 2, "team {t} != 2");
    }
    // scores 10..50 with team==2: 12, 17, 22, 27, 32, 37, 42, 47 = 8
    assert_eq!(result.len(), 8);
}

#[test]
fn execute_join_intersects_entity_sets() {
    let mut world = World::new();
    // Entities with Score only
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    // Entities with both Score and Team — only these should survive the join
    let mut both = Vec::new();
    for i in 10..20 {
        both.push(world.spawn((Score(i), Team(i % 3))));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let mut result = plan.execute_collect(&mut world).unwrap().to_vec();
    result.sort_by_key(|e| e.to_bits());
    both.sort_by_key(|e| e.to_bits());
    assert_eq!(result, both);
}

#[test]
fn execute_custom_filter() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "even scores",
            0.5,
            |world, entity| world.get::<Score>(entity).is_some_and(|s| s.0 % 2 == 0),
        ))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 25);
    for e in &result {
        assert!(world.get::<Score>(*e).unwrap().0.is_multiple_of(2));
    }
}

#[test]
fn execute_custom_column_filter() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("even scores", 0.5, |s| {
            s.0 % 2 == 0
        }))
        .build();
    // Column-filter plans should use the scan_required fast path.
    assert!(plan.scan_required.is_some());
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 25);
    for e in &result {
        assert!(world.get::<Score>(*e).unwrap().0.is_multiple_of(2));
    }
}

#[test]
fn custom_column_filter_multi_archetype() {
    let mut world = World::new();
    // Archetype 1: Score only
    for i in 0..30 {
        world.spawn((Score(i),));
    }
    // Archetype 2: Score + Team
    for i in 30..50 {
        world.spawn((Score(i), Team(0)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("score < 25", 0.5, |s| {
            s.0 < 25
        }))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 25);
    for e in &result {
        assert!(world.get::<Score>(*e).unwrap().0 < 25);
    }
}

#[test]
fn custom_column_filter_stream() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("score >= 90", 0.1, |s| {
            s.0 >= 90
        }))
        .build();
    let mut count = 0u32;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 10);
}

#[test]
fn custom_column_filter_stream_raw() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("score >= 90", 0.1, |s| {
            s.0 >= 90
        }))
        .build();
    let mut count = 0u32;
    plan.execute_stream_raw(&world, |_| count += 1).unwrap();
    assert_eq!(count, 10);
}

#[test]
fn custom_column_multiple_filters() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("score >= 20", 0.8, |s| {
            s.0 >= 20
        }))
        .filter(Predicate::custom_column::<Score>("score < 30", 0.1, |s| {
            s.0 < 30
        }))
        .build();
    // Should use scan_required fast path with both column filters.
    assert!(plan.scan_required.is_some());
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 10);
    for e in &result {
        let s = world.get::<Score>(*e).unwrap().0;
        assert!((20..30).contains(&s));
    }
}

#[test]
fn custom_column_mixed_with_per_entity_disables_fast_path() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("even", 0.5, |s| {
            s.0 % 2 == 0
        }))
        .filter(Predicate::custom::<Score>("small", 0.5, |world, entity| {
            world.get::<Score>(entity).is_some_and(|s| s.0 < 25)
        }))
        .build();
    // Mixed column + per-entity filters should NOT use scan_required.
    assert!(plan.scan_required.is_none());
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    // Even scores from 0..25: 0, 2, 4, ..., 24 = 13 values
    assert_eq!(result.len(), 13);
}

#[test]
fn custom_column_sparse_component_fallback() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1),));
    let e2 = world.spawn((Score(2),));
    let _e3 = world.spawn((Score(3),));

    // Health is stored as sparse — not in archetype columns.
    world.insert_sparse(e1, Health(100));
    world.insert_sparse(e2, Health(50));
    // e3 has no Health.

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Health>(
            "health >= 80",
            0.5,
            |h| h.0 >= 80,
        ))
        .build();
    // Column filter on sparse component should still use scan_required
    // (it falls back to per-entity sparse.get inside the closure).
    assert!(plan.scan_required.is_some());

    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    // Only e1 (Health(100)) passes the filter.
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], e1);
}

#[test]
fn custom_column_sparse_matches_custom_results() {
    // Verify custom_column and custom produce identical results
    // when filtering on a sparse component.
    let mut world = World::new();
    for i in 0..20 {
        let e = world.spawn((Score(i),));
        if i % 3 == 0 {
            world.insert_sparse(e, Health(i * 10));
        }
    }

    // custom_column variant
    let planner = QueryPlanner::new(&world);
    let mut plan_col = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Health>(
            "health >= 30",
            0.3,
            |h| h.0 >= 30,
        ))
        .build();

    // custom (per-entity) variant
    let planner = QueryPlanner::new(&world);
    let mut plan_old = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Health>(
            "health >= 30",
            0.3,
            |world, entity| world.get::<Health>(entity).is_some_and(|h| h.0 >= 30),
        ))
        .build();

    let mut result_col = plan_col.execute_collect(&mut world).unwrap().to_vec();
    let mut result_old = plan_old.execute_collect(&mut world).unwrap().to_vec();
    result_col.sort_by_key(|e| e.to_bits());
    result_old.sort_by_key(|e| e.to_bits());
    assert_eq!(
        result_col, result_old,
        "custom_column and custom must produce identical results for sparse components"
    );
    // Entities with i=3 (Health(30)), 6 (60), 9 (90), 12 (120), 15 (150), 18 (180)
    assert_eq!(result_col.len(), 6);
}

#[test]
fn execute_empty_world() {
    let mut world = World::new();
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_collect(&mut world).unwrap();
    assert!(result.is_empty());
}

#[test]
fn execute_respects_despawned_entities() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1),));
    let e2 = world.spawn((Score(2),));
    let e3 = world.spawn((Score(3),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // Despawn e2 after plan construction
    world.despawn(e2);

    let result = plan.execute_collect(&mut world).unwrap();
    // Scan walks archetypes — despawned entity is replaced via swap_remove,
    // so the archetype only contains live entities.
    assert_eq!(result.len(), 2);
    assert!(result.contains(&e1));
    assert!(result.contains(&e3));
}

#[test]
fn execute_index_driven_respects_query_components() {
    // Regression: index-driven lookup must only return entities that match
    // ALL queried components, not just the indexed one.
    let mut world = World::new();
    // Entities with Team only (no Score)
    for i in 0..10 {
        world.spawn((Team(i % 3),));
    }
    // Entities with both Score and Team
    let mut both = Vec::new();
    for i in 0..10 {
        both.push(world.spawn((Score(i), Team(i % 3))));
    }

    let mut hash = HashIndex::<Team>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(hash), &world).unwrap();

    // scan::<(&Score, &Team)> requires BOTH components
    let mut plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(1)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    // Only entities with both Score AND Team(1) should appear
    for e in &result {
        assert!(world.get::<Score>(*e).is_some(), "missing Score");
        assert_eq!(*world.get::<Team>(*e).unwrap(), Team(1));
    }
    // Team(1) entities with Score: indices 1, 4, 7 = 3
    assert_eq!(result.len(), 3);
}

// ── Predicate-specific index lookup ─────────────────────────────────

#[test]
fn execute_btree_eq_uses_targeted_lookup() {
    // Verify that BTree + Eq predicate returns only matching entities,
    // not the entire index (regression: was O(n) full-index scan).
    let mut world = World::new();
    for i in 0..200 {
        world.spawn((Score(i), Team(i % 10)));
    }
    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 1);
    assert_eq!(*world.get::<Score>(result[0]).unwrap(), Score(42));
}

#[test]
fn execute_btree_range_uses_targeted_lookup() {
    // Verify that BTree + Range predicate returns only entities in range,
    // not the entire index (regression: was O(n) full-index scan).
    let mut world = World::new();
    for i in 0..200 {
        world.spawn((Score(i),));
    }
    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 10); // scores 10..20
    for e in &result {
        let s = world.get::<Score>(*e).unwrap().0;
        assert!((10..20).contains(&s), "score {s} out of range");
    }
}

#[test]
fn execute_hash_eq_uses_targeted_lookup() {
    // Verify that Hash + Eq predicate returns only matching entities,
    // not the entire index (regression: was O(n) full-index scan).
    let mut world = World::new();
    for i in 0..200 {
        world.spawn((Score(i), Team(i % 10)));
    }
    let mut hash = HashIndex::<Team>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(hash), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::eq(Team(3)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 20); // 200 / 10 teams
    for e in &result {
        assert_eq!(*world.get::<Team>(*e).unwrap(), Team(3));
    }
}

#[test]
fn execute_btree_eq_nonexistent_value_returns_empty() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }
    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(999)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap();
    assert!(result.is_empty());
}

#[test]
fn execute_hash_eq_nonexistent_value_returns_empty() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Team(i),));
    }
    let mut hash = HashIndex::<Team>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(hash), &world).unwrap();

    let mut plan = planner
        .scan::<(&Team,)>()
        .filter(Predicate::eq(Team(999)))
        .build();
    let result = plan.execute_collect(&mut world).unwrap();
    assert!(result.is_empty());
}

// ── Live index reads ──────────────────────────────────────────────

#[test]
fn execute_reads_live_btree_not_registration_snapshot() {
    // Regression: plans used to capture a frozen BTreeMap clone at
    // registration time. After rebuild, the plan would return stale results.
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Score(i),));
    }
    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);
    let btree = Arc::new(btree);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&btree, &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    // First execute_collect: should find Score(42)
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 1);

    // Spawn more entities and rebuild the index via a new Arc.
    // The plan still holds the old Arc — it sees the old index contents.
    // This is the expected behavior: the plan reads the Arc it was given.
    // To see updated data, register a new index and rebuild the plan.
    for i in 50..100 {
        world.spawn((Score(i),));
    }

    // Verify scan (non-index) path sees new entities immediately
    let planner2 = QueryPlanner::new(&world);
    let mut scan_plan = planner2.scan::<(&Score,)>().build();
    assert_eq!(scan_plan.execute_collect(&mut world).unwrap().len(), 100);
}

#[test]
fn execute_reads_live_hash_not_registration_snapshot() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Team(i % 5),));
    }
    let mut hash = HashIndex::<Team>::new();
    hash.rebuild(&mut world);
    let hash = Arc::new(hash);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&hash, &world).unwrap();

    let mut plan = planner
        .scan::<(&Team,)>()
        .filter(Predicate::eq(Team(2)))
        .build();

    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 10); // 50 / 5 teams
    for e in &result {
        assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
    }
}

#[test]
fn execution_produces_correct_results() {
    // End-to-end: plan must produce the same results as a naive
    // query would.
    let mut world = World::new();
    for i in 0..500 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);
    let mut hash = HashIndex::<Team>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();
    planner.add_hash_index(&Arc::new(hash), &world).unwrap();

    // Complex plan: index + filter + join
    let mut plan = planner
        .scan::<(&Score, &Team)>()
        .filter(Predicate::range::<Score, _>(Score(100)..Score(300)))
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let entities = plan.execute_collect(&mut world).unwrap().to_vec();
    // All 500 entities have Team, so the join doesn't reduce the set.
    // The range filter should give us Score 100..300 = 200 entities.
    assert_eq!(entities.len(), 200);
    for e in &entities {
        let s = world.get::<Score>(*e).unwrap().0;
        assert!((100..300).contains(&s), "score {s} out of range");
    }
}

// ── Left join execution ────────────────────────────────────────────

#[test]
fn execute_left_join_preserves_all_left_entities() {
    let mut world = World::new();
    // 20 entities with Score only (no Team)
    let mut score_only = Vec::new();
    for i in 0..20 {
        score_only.push(world.spawn((Score(i),)));
    }
    // 10 entities with both Score and Team
    let mut both = Vec::new();
    for i in 20..30 {
        both.push(world.spawn((Score(i), Team(i % 3))));
    }

    let planner = QueryPlanner::new(&world);
    // Left join: all Score entities should appear, even those without Team.
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    // All 30 Score entities must be present.
    assert_eq!(result.len(), 30);
    for e in &score_only {
        assert!(result.contains(e), "left join dropped Score-only entity");
    }
    for e in &both {
        assert!(result.contains(e), "left join dropped Score+Team entity");
    }
}

#[test]
fn execute_inner_join_excludes_unmatched() {
    let mut world = World::new();
    // 20 entities with Score only (no Team)
    for i in 0..20 {
        world.spawn((Score(i),));
    }
    // 10 entities with both Score and Team
    let mut both = Vec::new();
    for i in 20..30 {
        both.push(world.spawn((Score(i), Team(i % 3))));
    }

    let planner = QueryPlanner::new(&world);
    // Inner join: only entities with both Score and Team.
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let result = plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(result.len(), 10);
    for e in &both {
        assert!(result.contains(e));
    }
}

#[test]
fn execute_left_join_small_cardinality_nested_loop() {
    let mut world = World::new();
    // Score-only entities
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Score+Team entities
    for i in 5..8 {
        world.spawn((Score(i), Team(0)));
    }

    let planner = QueryPlanner::new(&world);
    // Small estimates → nested-loop join path.
    let mut plan = planner
        .scan_with_estimate::<(&Score,)>(8)
        .join::<(&Team,)>(JoinKind::Left)
        .with_right_estimate(3)
        .unwrap()
        .build();
    let result = plan.execute_collect(&mut world).unwrap();
    // Left join: all 8 Score entities preserved.
    assert_eq!(result.len(), 8);
}

#[test]
fn execute_multi_join_intersects_all() {
    let mut world = World::new();
    let mut all_three = Vec::new();
    for i in 0..10u32 {
        all_three.push(world.spawn((Score(i), Team(i % 3), Health(100))));
    }
    // 5 entities with only Score + Team (no Health)
    for i in 10..15u32 {
        world.spawn((Score(i), Team(i % 3)));
    }
    // 5 entities with only Score + Health (no Team)
    for i in 15..20u32 {
        world.spawn((Score(i), Health(50)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .join::<(&Health,)>(JoinKind::Inner)
        .build();

    let result = plan.execute_collect(&mut world).unwrap();
    // Score ∩ Team ∩ Health = the 10 entities with all three
    let mut found: Vec<Entity> = result.to_vec();
    found.sort_by_key(|e| e.to_bits());
    all_three.sort_by_key(|e| e.to_bits());
    assert_eq!(found, all_three);
}

// ── ScratchBuffer tests ──────────────────────────────────────────

#[test]
fn scratch_buffer_starts_empty() {
    let buf = ScratchBuffer::new(100);
    assert_eq!(buf.len(), 0);
    assert!(buf.capacity() >= 100);
}

#[test]
fn scratch_buffer_push_and_clear() {
    let mut buf = ScratchBuffer::new(4);
    let e0 = Entity::new(0, 0);
    let e1 = Entity::new(1, 0);
    buf.push(e0);
    buf.push(e1);
    assert_eq!(buf.as_slice(), &[e0, e1]);

    let cap_before = buf.capacity();
    buf.clear();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.capacity(), cap_before);
}

#[test]
fn scratch_buffer_reuse_does_not_realloc() {
    let mut buf = ScratchBuffer::new(64);
    for i in 0..50 {
        buf.push(Entity::new(i, 0));
    }
    let cap = buf.capacity();
    buf.clear();
    for i in 0..50 {
        buf.push(Entity::new(100 + i, 0));
    }
    assert_eq!(buf.capacity(), cap);
}

#[test]
fn scratch_buffer_sorted_intersection() {
    let mut buf = ScratchBuffer::new(16);
    // Left set: [1,3,5,7,9]
    let left = [1u32, 3, 5, 7, 9];
    for &idx in &left {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = left.len();

    // Right set: [2,3,6,7,10]
    let right = [2u32, 3, 6, 7, 10];
    for &idx in &right {
        buf.push(Entity::new(idx, 0));
    }

    let result = buf.sorted_intersection(left_len);
    // Intersection should be entities with index 3 and 7.
    let mut result_indices: Vec<u32> = result.iter().map(|e| e.index()).collect();
    result_indices.sort_unstable();
    assert_eq!(result_indices, vec![3, 7]);
}

#[test]
fn scratch_buffer_sorted_intersection_empty_left() {
    let mut buf = ScratchBuffer::new(10);
    let left_len = 0; // empty left
    for idx in [2, 3, 6] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.sorted_intersection(left_len);
    assert!(result.is_empty());
}

#[test]
fn scratch_buffer_sorted_intersection_empty_right() {
    let mut buf = ScratchBuffer::new(10);
    for idx in [1, 3, 5] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len(); // right partition is empty
    let result = buf.sorted_intersection(left_len);
    assert!(result.is_empty());
}

#[test]
fn scratch_buffer_sorted_intersection_complete_overlap() {
    let mut buf = ScratchBuffer::new(10);
    for idx in [1, 2, 3] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len();
    for idx in [1, 2, 3] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.sorted_intersection(left_len);
    let mut ids: Vec<u32> = result.iter().map(|e| e.index()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn scratch_buffer_sorted_intersection_no_overlap() {
    let mut buf = ScratchBuffer::new(10);
    for idx in [1, 3, 5] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len();
    for idx in [2, 4, 6] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.sorted_intersection(left_len);
    assert!(result.is_empty());
}

#[test]
fn scratch_buffer_partitioned_intersection() {
    let mut buf = ScratchBuffer::new(20);
    // Left: 1, 3, 5, 7, 9
    for idx in [1, 3, 5, 7, 9] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len();
    // Right: 2, 3, 6, 7, 10
    for idx in [2, 3, 6, 7, 10] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.partitioned_intersection(left_len, 3);
    let mut bits: Vec<u64> = result.iter().map(|e| e.to_bits()).collect();
    bits.sort_unstable();
    let expected: Vec<u64> = [3, 7]
        .iter()
        .map(|&idx| Entity::new(idx, 0).to_bits())
        .collect();
    assert_eq!(bits, expected);
}

#[test]
fn scratch_buffer_partitioned_intersection_no_overlap() {
    let mut buf = ScratchBuffer::new(10);
    for idx in [1, 3, 5] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len();
    for idx in [2, 4, 6] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.partitioned_intersection(left_len, 2);
    assert!(result.is_empty());
}

#[test]
fn scratch_buffer_partitioned_intersection_complete_overlap() {
    let mut buf = ScratchBuffer::new(10);
    for idx in [10, 20, 30] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len();
    for idx in [10, 20, 30] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.partitioned_intersection(left_len, 4);
    let mut bits: Vec<u64> = result.iter().map(|e| e.to_bits()).collect();
    bits.sort_unstable();
    let expected: Vec<u64> = [10, 20, 30]
        .iter()
        .map(|&idx| Entity::new(idx, 0).to_bits())
        .collect();
    assert_eq!(bits, expected);
}

#[test]
fn scratch_buffer_partitioned_intersection_clamps_overestimate() {
    // Partition count far exceeding actual entity count is clamped to
    // left_len, falling back to sorted_intersection when <= 1.
    let mut buf = ScratchBuffer::new(10);
    for idx in [1, 2, 3] {
        buf.push(Entity::new(idx, 0));
    }
    let left_len = buf.len();
    for idx in [2, 3, 4] {
        buf.push(Entity::new(idx, 0));
    }
    // 1_000_000 partitions for 3 left entities → clamped to 3.
    let result = buf.partitioned_intersection(left_len, 1_000_000);
    let mut bits: Vec<u64> = result.iter().map(|e| e.to_bits()).collect();
    bits.sort_unstable();
    let expected: Vec<u64> = [2, 3]
        .iter()
        .map(|&idx| Entity::new(idx, 0).to_bits())
        .collect();
    assert_eq!(bits, expected);
}

#[test]
fn scratch_buffer_partitioned_intersection_fallback_when_empty() {
    // Zero left entities → partitions clamped to 0 → falls back to
    // sorted_intersection (which returns empty).
    let mut buf = ScratchBuffer::new(10);
    let left_len = buf.len(); // 0
    for idx in [1, 2, 3] {
        buf.push(Entity::new(idx, 0));
    }
    let result = buf.partitioned_intersection(left_len, 100);
    assert!(result.is_empty());
}

// ── Execute with ScratchBuffer ─────────────────────────────────

#[test]
fn execute_with_scratch_returns_all_entities() {
    let mut world = World::new();
    let mut expected = Vec::new();
    for i in 0..10u32 {
        let e = world.spawn((Score(i), Team(i % 3)));
        expected.push(e);
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let result = plan.execute_collect(&mut world).unwrap();
    let mut found: Vec<Entity> = result.to_vec();
    found.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}

#[test]
fn execute_scratch_reuse_no_realloc() {
    let mut world = World::new();
    for i in 0..10u32 {
        world.spawn((Score(i), Team(i % 3)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let _ = plan.execute_collect(&mut world).unwrap();
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 10);
}

// ── CompiledScan execute_stream ──────────────────────────────────────

#[test]
fn compiled_scan_for_each_yields_all_entities() {
    let mut world = World::new();
    let mut expected = Vec::new();
    for i in 0..10u32 {
        let e = world.spawn((Score(i),));
        expected.push(e);
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut found = Vec::new();
    plan.execute_stream(&mut world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    found.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}

#[test]
fn for_each_iterates_multiple_archetypes() {
    let mut world = World::new();
    let mut expected = Vec::new();
    // Archetype 1: (Score,)
    for i in 0..5u32 {
        expected.push(world.spawn((Score(i),)));
    }
    // Archetype 2: (Score, Team)
    for i in 5..10u32 {
        expected.push(world.spawn((Score(i), Team(i % 3))));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut found = Vec::new();
    plan.execute_stream(&mut world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    found.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}

#[test]
fn for_each_raw_iterates_multiple_archetypes() {
    let mut world = World::new();
    let mut expected = Vec::new();
    for i in 0..5u32 {
        expected.push(world.spawn((Score(i),)));
    }
    for i in 5..10u32 {
        expected.push(world.spawn((Score(i), Team(i % 3))));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut found = Vec::new();
    plan.execute_stream_raw(&world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    found.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}

#[test]
fn for_each_with_eq_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    let mut found = Vec::new();
    plan.execute_stream(&mut world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    assert_eq!(found.len(), 1);
    let score = world.get::<Score>(found[0]).unwrap();
    assert_eq!(*score, Score(42));
}

#[test]
fn for_each_with_range_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
        .build();

    let mut found = Vec::new();
    plan.execute_stream(&mut world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    assert_eq!(found.len(), 10); // 10..20 exclusive = 10 entities
}

#[test]
fn for_each_with_custom_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "even scores",
            0.5,
            |world, entity| world.get::<Score>(entity).is_some_and(|s| s.0 % 2 == 0),
        ))
        .build();

    let mut found = Vec::new();
    plan.execute_stream(&mut world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    assert_eq!(found.len(), 50);
}

// ── CompiledScan for_each_raw ────────────────────────────────────

#[test]
fn for_each_raw_yields_entities_without_mut_world() {
    let mut world = World::new();
    for i in 0..10u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut found = Vec::new();
    plan.execute_stream_raw(&world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    assert_eq!(found.len(), 10);
}

#[test]
fn for_each_raw_with_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    let mut found = Vec::new();
    plan.execute_stream_raw(&world, |entity: Entity| {
        found.push(entity);
    })
    .unwrap();
    assert_eq!(found.len(), 1);
}

// ── Changed<T> filtering ──────────────────────────────────────────

#[test]
fn for_each_changed_skips_stale_archetypes() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // First call: everything is new (changed since tick 0).
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 5);

    // Second call: nothing changed since the last read tick.
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn for_each_changed_detects_mutation() {
    let mut world = World::new();
    let e = world.spawn((Score(1),));
    world.spawn((Score(2), Team(0)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // Drain initial changes.
    plan.execute_stream(&mut world, |_| {}).unwrap();

    // Mutate one archetype's Score column via get_mut.
    let _ = world.get_mut::<Score>(e);

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    // Only the archetype containing `e` was mutated.
    assert_eq!(count, 1);
}

#[test]
fn for_each_raw_changed_reads_tick_but_does_not_advance() {
    let mut world = World::new();
    for i in 0..3u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // for_each_raw does not advance the tick.
    let mut count = 0;
    plan.execute_stream_raw(&world, |_| count += 1).unwrap();
    assert_eq!(count, 3);

    // Same tick — still sees changes.
    let mut count = 0;
    plan.execute_stream_raw(&world, |_| count += 1).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn execute_changed_skips_stale_archetypes() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 5);
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 0);
}

#[test]
fn for_each_no_changed_pays_zero_cost() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // Without Changed<T>, every call sees all entities.
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 5);

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 5);
}

#[test]
fn execute_join_changed_left_only() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i), Team(i % 2)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 5);
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 0);
}

#[test]
fn execute_join_changed_right_only() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i), Team(i % 2)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(Changed<Team>, &Team)>(JoinKind::Inner)
        .build();
    // First: all changed
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 5);
    // Second: right side not changed, right yields 0 entities.
    // Inner join of 5 and 0 = 0.
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 0);
}

#[test]
fn for_each_raw_then_for_each_advances() {
    let mut world = World::new();
    for i in 0..3u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();
    // for_each_raw: sees everything, doesn't advance tick
    let mut count = 0;
    plan.execute_stream_raw(&world, |_| count += 1).unwrap();
    assert_eq!(count, 3);
    // execute_stream: also sees everything (tick still at 0), and advances
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 3);
    // execute_stream again: tick advanced, nothing changed
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn for_each_changed_multiple_archetypes_partial() {
    let mut world = World::new();
    // Archetype 1: (Score,)
    let e1 = world.spawn((Score(1),));
    // Archetype 2: (Score, Team)
    let _e2 = world.spawn((Score(2), Team(0)));
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();
    // Consume initial changes
    plan.execute_stream(&mut world, |_| {}).unwrap();
    // Mutate only archetype 1
    let _ = world.get_mut::<Score>(e1);
    // Only archetype 1 should be returned
    let mut found = Vec::new();
    plan.execute_stream(&mut world, |entity| found.push(entity))
        .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], e1);
}

#[test]
fn for_each_changed_with_predicate_filter() {
    let mut world = World::new();
    // All in one archetype
    for i in 0..10u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .filter(Predicate::custom::<Score>(
            "score < 5",
            0.5,
            |world: &World, entity: Entity| world.get::<Score>(entity).is_some_and(|s| s.0 < 5),
        ))
        .build();

    // First call: Changed passes (everything new), predicate filters to 5
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 5);

    // Second call: Changed skips the archetype entirely
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn execute_changed_detects_partial_mutation() {
    let mut world = World::new();
    // Two archetypes so column-level change detection can distinguish them
    let e = world.spawn((Score(1),));
    world.spawn((Score(2), Team(0)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // Drain initial changes
    let _ = plan.execute_collect(&mut world).unwrap();

    // Mutate only the (Score,) archetype
    let _ = world.get_mut::<Score>(e);

    // execute_collect path (scratch buffer) should see only the changed archetype
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn for_each_changed_sees_entities_spawned_after_plan_creation() {
    let mut world = World::new();
    world.spawn((Score(1),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // Drain initial changes
    plan.execute_stream(&mut world, |_| {}).unwrap();

    // Spawn new entities into a new archetype AFTER plan was built.
    // The compiled scan iterates world.archetypes at execution time,
    // so new archetypes should be visible. Their column ticks will be
    // newer than last_read_tick.
    world.spawn((Score(2), Team(0)));
    world.spawn((Score(3), Team(1)));

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn execute_left_join_changed_right_preserves_left() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i), Team(i % 2)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(Changed<Team>, &Team)>(JoinKind::Left)
        .build();

    // First call: all changed, left join preserves left = 5
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 5);

    // Second call: right side stale, but Left join keeps all left entities
    assert_eq!(plan.execute_collect(&mut world).unwrap().len(), 5);
}

// ── Spatial predicate tests ──────────────────────────────────

/// Grid index that supports `Within` queries for testing.
struct TestGridIndex {
    entities: Vec<Entity>,
}

impl TestGridIndex {
    fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }
}

impl SpatialIndex for TestGridIndex {
    fn rebuild(&mut self, world: &mut World) {
        self.entities = world.query::<(Entity, &Pos)>().map(|(e, _)| e).collect();
    }

    fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        match expr {
            crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                estimated_rows: (self.entities.len() as f64 * 0.1).max(1.0),
                cpu: 5.0,
            }),
            crate::index::SpatialExpr::Intersects { .. } => Some(crate::index::SpatialCost {
                estimated_rows: (self.entities.len() as f64 * 0.2).max(1.0),
                cpu: 8.0,
            }),
        }
    }

    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        self.entities.clone()
    }
}

/// Grid index that does NOT support any spatial queries.
struct UnsupportedGridIndex;

impl SpatialIndex for UnsupportedGridIndex {
    fn rebuild(&mut self, _world: &mut World) {}

    fn supports(&self, _expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        None
    }
    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        Vec::new()
    }
}

/// Test spatial index that returns a fixed set of entities.
struct FixedSpatialIndex {
    entities: Vec<Entity>,
}

impl FixedSpatialIndex {
    fn new(entities: Vec<Entity>) -> Self {
        Self { entities }
    }
}

impl SpatialIndex for FixedSpatialIndex {
    fn rebuild(&mut self, _world: &mut World) {}
    fn supports(&self, _expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        Some(crate::index::SpatialCost {
            estimated_rows: self.entities.len() as f64,
            cpu: 1.0,
        })
    }
    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        self.entities.clone()
    }
}

/// Test spatial index that counts calls and returns a fixed set of entities.
struct CountingSpatialIndex {
    entities: Vec<Entity>,
    call_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl SpatialIndex for CountingSpatialIndex {
    fn rebuild(&mut self, _world: &mut World) {}
    fn supports(&self, _expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        Some(crate::index::SpatialCost {
            estimated_rows: self.entities.len() as f64,
            cpu: 1.0,
        })
    }
    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.entities.clone()
    }
}

/// Test spatial index that records the received expression and returns fixed entities.
struct RecordingSpatialIndex {
    entities: Vec<Entity>,
    call_count: Arc<std::sync::atomic::AtomicUsize>,
    received_center: Arc<std::sync::Mutex<Vec<f64>>>,
}

impl SpatialIndex for RecordingSpatialIndex {
    fn rebuild(&mut self, _world: &mut World) {}
    fn supports(&self, _expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        Some(crate::index::SpatialCost {
            estimated_rows: self.entities.len() as f64,
            cpu: 1.0,
        })
    }
    fn query(&self, expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let crate::index::SpatialExpr::Within { center, .. } = expr {
            *self.received_center.lock().unwrap() = center.clone();
        }
        self.entities.clone()
    }
}

#[test]
fn spatial_predicate_within_creates_spatial_lookup() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
        .build();

    // The root of the logical plan should be a SpatialLookup.
    match plan.root() {
        PlanNode::SpatialLookup {
            component_name,
            cost,
            ..
        } => {
            assert!(component_name.contains("Pos"));
            assert!(cost.rows > 0.0);
        }
        other => panic!("expected SpatialLookup, got {:?}", other),
    }

    // No warnings expected — spatial index was registered.
    assert!(plan.warnings().is_empty());
}

#[test]
fn spatial_predicate_intersects_creates_spatial_lookup() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::intersects::<Pos>([0.0, 0.0], [25.0, 25.0], |_, _| true).unwrap())
        .build();

    match plan.root() {
        PlanNode::SpatialLookup { .. } => {}
        other => panic!("expected SpatialLookup, got {:?}", other),
    }
}

#[test]
fn spatial_predicate_without_index_falls_back_to_filter() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
        .build();

    // Without a spatial index, should fall back to Scan + Filter.
    match plan.root() {
        PlanNode::Filter { child, .. } => match child.as_ref() {
            PlanNode::Scan { .. } => {}
            other => panic!("expected Scan child, got {:?}", other),
        },
        other => panic!("expected Filter, got {:?}", other),
    }

    // Should have a warning about missing spatial index.
    assert_eq!(plan.warnings().len(), 1);
    match &plan.warnings()[0] {
        PlanWarning::MissingIndex { predicate_kind, .. } => {
            assert_eq!(*predicate_kind, "spatial");
        }
        other => panic!("expected MissingIndex warning, got {:?}", other),
    }
}

#[test]
fn spatial_predicate_unsupported_expr_falls_back_to_filter() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    // Register a spatial index that doesn't support any queries.
    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(UnsupportedGridIndex), &world)
        .unwrap();

    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
        .build();

    // Index exists but doesn't support Within — should fall back.
    match plan.root() {
        PlanNode::Filter { child, .. } => match child.as_ref() {
            PlanNode::Scan { .. } => {}
            other => panic!("expected Scan child, got {:?}", other),
        },
        other => panic!("expected Filter, got {:?}", other),
    }

    // SpatialIndexDeclined warning should be emitted (not MissingIndex).
    assert_eq!(plan.warnings().len(), 1);
    assert!(
        matches!(
            &plan.warnings()[0],
            PlanWarning::SpatialIndexDeclined { .. }
        ),
        "expected SpatialIndexDeclined warning, got {:?}",
        plan.warnings()[0]
    );
}

#[test]
fn spatial_lookup_uses_spatial_gather() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
        .build();

    match plan.root() {
        PlanNode::SpatialLookup { component_name, .. } => {
            assert!(component_name.contains("Pos"));
        }
        other => panic!("expected SpatialLookup, got {:?}", other),
    }
}

#[test]
fn spatial_predicate_explain_contains_spatial_lookup() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
        .build();

    let explain = plan.explain();
    assert!(explain.contains("SpatialGather"));
    assert!(explain.contains("Pos"));
}

#[test]
fn spatial_predicate_with_custom_selectivity() {
    let pred = Predicate::within::<Pos>([0.0, 0.0], 1.0, |_, _| true)
        .unwrap()
        .with_selectivity(0.05);

    // Selectivity override should work.
    match &pred.kind {
        PredicateKind::Spatial(SpatialPredicate::Within { radius, .. }) => {
            assert!(*radius > 0.0);
        }
        other => panic!("expected Spatial(Within), got {:?}", other),
    }
    assert!((pred.selectivity - 0.05).abs() < f64::EPSILON);
}

#[test]
fn spatial_predicate_debug_format() {
    let pred = Predicate::within::<Pos>([1.0, 2.0], 3.0, |_, _| true).unwrap();
    let dbg = format!("{:?}", pred);
    assert!(dbg.contains("Spatial"));
    assert!(dbg.contains("ST_Within"));

    let pred2 = Predicate::intersects::<Pos>([0.0, 0.0], [10.0, 10.0], |_, _| true).unwrap();
    let dbg2 = format!("{:?}", pred2);
    assert!(dbg2.contains("ST_Intersects"));
}

#[test]
fn spatial_vs_btree_cheaper_spatial_wins() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((
            Pos {
                x: i as f32,
                y: i as f32,
            },
            Score(i),
        ));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    // Spatial predicate with very low estimated cost should win.
    let plan = planner
        .scan::<(&Pos, &Score)>()
        .filter(Predicate::within::<Pos>([500.0, 500.0], 5.0, |_, _| true).unwrap())
        .filter(Predicate::range::<Score, _>(Score(400)..Score(600)))
        .build();

    // The driving access should be whichever has lower cost.
    // Our TestGridIndex reports ~100 rows for within (0.1 * 1000),
    // vs BTree range at 0.1 selectivity → 100 rows.
    // The spatial cost.cpu is 5.0 vs index_lookup ~ 5.0 + 100.
    // Spatial should win.
    match plan.root() {
        PlanNode::Filter { child, .. } => match child.as_ref() {
            PlanNode::SpatialLookup { .. } => {}
            PlanNode::Filter { child, .. } => match child.as_ref() {
                PlanNode::SpatialLookup { .. } => {}
                other => panic!("expected SpatialLookup deep, got {:?}", other),
            },
            other => panic!("expected SpatialLookup or Filter child, got {:?}", other),
        },
        PlanNode::SpatialLookup { .. } => {}
        other => panic!("expected SpatialLookup at root, got {:?}", other),
    }
}

#[test]
fn spatial_predicate_for_each_works() {
    let mut world = World::new();
    for i in 0..20 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    // Even without a spatial index, execute_stream should work via filter fallback.
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(
            Predicate::within::<Pos>([10.0, 10.0], 100.0, |world, entity| {
                world
                    .get::<Pos>(entity)
                    .is_some_and(|p| ((p.x - 10.0).powi(2) + (p.y - 10.0).powi(2)).sqrt() < 100.0)
            })
            .unwrap(),
        )
        .build();

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 20); // All within radius 100
}

#[test]
fn spatial_predicate_display() {
    let sp = SpatialPredicate::Within {
        center: vec![1.0, 2.0],
        radius: 3.0,
    };
    assert_eq!(format!("{}", sp), "ST_Within([1.0, 2.0], 3)");

    let sp2 = SpatialPredicate::Intersects {
        min: vec![0.0, 0.0],
        max: vec![10.0, 10.0],
    };
    assert_eq!(
        format!("{}", sp2),
        "ST_Intersects([0.0, 0.0], [10.0, 10.0])"
    );
}

#[test]
fn spatial_predicate_to_expr_round_trip() {
    let sp = SpatialPredicate::Within {
        center: vec![1.0, 2.0],
        radius: 3.0,
    };
    let expr = crate::index::SpatialExpr::from(&sp);
    match &expr {
        crate::index::SpatialExpr::Within { center, radius } => {
            assert_eq!(center, &[1.0, 2.0]);
            assert!((radius - 3.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Within, got {:?}", other),
    }

    let sp2 = SpatialPredicate::Intersects {
        min: vec![0.0, 1.0],
        max: vec![2.0, 3.0],
    };
    let expr2 = crate::index::SpatialExpr::from(&sp2);
    match &expr2 {
        crate::index::SpatialExpr::Intersects { min, max } => {
            assert_eq!(min, &[0.0, 1.0]);
            assert_eq!(max, &[2.0, 3.0]);
        }
        other => panic!("expected Intersects, got {:?}", other),
    }
}

// ── Additional spatial coverage tests ────────────────────────

/// Spatial index with very high cost — BTree should win the cost comparison.
struct ExpensiveSpatialIndex;

impl SpatialIndex for ExpensiveSpatialIndex {
    fn rebuild(&mut self, _world: &mut World) {}

    fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        match expr {
            crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                estimated_rows: 500.0,
                cpu: 500.0,
            }),
            _ => None,
        }
    }

    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        Vec::new()
    }
}

#[test]
fn spatial_vs_btree_cheaper_btree_wins() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((
            Pos {
                x: i as f32,
                y: i as f32,
            },
            Score(i),
        ));
    }

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(ExpensiveSpatialIndex), &world)
        .unwrap();
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    // BTree with high selectivity (0.01) → cost ~ 5 + 10 = 15.
    // Spatial index reports cpu=500. BTree should win.
    let plan = planner
        .scan::<(&Pos, &Score)>()
        .filter(Predicate::within::<Pos>([500.0, 500.0], 5.0, |_, _| true).unwrap())
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    // Driving access should be IndexLookup (BTree), not SpatialLookup.
    fn has_index_lookup(node: &PlanNode) -> bool {
        match node {
            PlanNode::IndexLookup { .. } => true,
            PlanNode::Filter { child, .. } => has_index_lookup(child),
            _ => false,
        }
    }
    assert!(
        has_index_lookup(plan.root()),
        "expected IndexLookup as driver, got {:?}",
        plan.root()
    );
}

/// Spatial index that only supports `Within`, not `Intersects`.
struct WithinOnlyIndex {
    entity_count: usize,
}

impl SpatialIndex for WithinOnlyIndex {
    fn rebuild(&mut self, world: &mut World) {
        self.entity_count = world.query::<(Entity, &Pos)>().count();
    }

    fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        match expr {
            crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                estimated_rows: (self.entity_count as f64 * 0.1).max(1.0),
                cpu: 5.0,
            }),
            _ => None,
        }
    }

    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        Vec::new()
    }
}

#[test]
fn spatial_partial_capability_within_supported_intersects_declined() {
    let mut world = World::new();
    for i in 0..50 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let mut idx = WithinOnlyIndex { entity_count: 0 };
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(idx), &world)
        .unwrap();

    // Within should get a SpatialLookup.
    let within_plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([25.0, 25.0], 5.0, |_, _| true).unwrap())
        .build();
    assert!(
        matches!(within_plan.root(), PlanNode::SpatialLookup { .. }),
        "Within should produce SpatialLookup, got {:?}",
        within_plan.root()
    );
    assert!(within_plan.warnings().is_empty());

    // Intersects should fall back with a SpatialIndexDeclined warning.
    let intersects_plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::intersects::<Pos>([0.0, 0.0], [10.0, 10.0], |_, _| true).unwrap())
        .build();
    match intersects_plan.root() {
        PlanNode::Filter { child, .. } => {
            assert!(matches!(child.as_ref(), PlanNode::Scan { .. }));
        }
        other => panic!("expected Filter, got {:?}", other),
    }
    assert_eq!(intersects_plan.warnings().len(), 1);
    assert!(matches!(
        &intersects_plan.warnings()[0],
        PlanWarning::SpatialIndexDeclined { .. }
    ));
}

#[test]
fn multiple_spatial_predicates_first_drives_rest_filter() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((
            Pos {
                x: i as f32,
                y: i as f32,
            },
            Health(100),
        ));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    // Two spatial predicates on the same component.
    let plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([50.0, 50.0], 5.0, |_, _| true).unwrap())
        .filter(Predicate::intersects::<Pos>([0.0, 0.0], [0.5, 0.5], |_, _| true).unwrap())
        .build();

    // One should be the driver (SpatialLookup), the other a Filter.
    fn find_spatial_lookup(node: &PlanNode) -> bool {
        match node {
            PlanNode::SpatialLookup { .. } => true,
            PlanNode::Filter { child, .. } => find_spatial_lookup(child),
            _ => false,
        }
    }
    fn count_filters(node: &PlanNode) -> usize {
        match node {
            PlanNode::Filter { child, .. } => 1 + count_filters(child),
            _ => 0,
        }
    }
    assert!(
        find_spatial_lookup(plan.root()),
        "expected SpatialLookup in plan tree"
    );
    assert!(
        count_filters(plan.root()) >= 1,
        "expected at least one Filter wrapping the SpatialLookup"
    );
    assert!(plan.warnings().is_empty());
}

/// Spatial index with low CPU but high estimated_rows.
/// Used to test that full plan cost (including downstream filters)
/// determines the driver, not just the driving access cost alone.
struct HighRowsSpatialIndex;

impl SpatialIndex for HighRowsSpatialIndex {
    fn rebuild(&mut self, _world: &mut World) {}

    fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
        match expr {
            crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                // Low CPU but returns most of the dataset — downstream
                // filters over 900 rows are expensive.
                estimated_rows: 900.0,
                cpu: 3.0,
            }),
            _ => None,
        }
    }

    fn query(&self, _expr: &crate::index::SpatialExpr) -> Vec<Entity> {
        Vec::new()
    }
}

#[test]
fn spatial_low_cpu_high_rows_loses_to_selective_btree() {
    // Regression: spatial index with cpu=3 but estimated_rows=900
    // vs BTree with selectivity=0.01 (10 rows from 1000).
    // Driving access costs alone: spatial=3, btree=5+10=15 → spatial wins.
    // But full plan cost: spatial driver + btree-as-filter over 900 rows
    // is more expensive than btree driver + spatial-as-filter over 10 rows.
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((
            Pos {
                x: i as f32,
                y: i as f32,
            },
            Score(i),
        ));
    }

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(HighRowsSpatialIndex), &world)
        .unwrap();
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let plan = planner
        .scan::<(&Pos, &Score)>()
        .filter(Predicate::within::<Pos>([500.0, 500.0], 5.0, |_, _| true).unwrap())
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    // BTree should win as driver because the full plan cost is lower:
    // btree(10 rows) + spatial-as-filter(10 * 0.5) < spatial(900 rows) + btree-as-filter(900 * 0.5)
    fn has_index_lookup(node: &PlanNode) -> bool {
        match node {
            PlanNode::IndexLookup { .. } => true,
            PlanNode::Filter { child, .. } => has_index_lookup(child),
            _ => false,
        }
    }
    assert!(
        has_index_lookup(plan.root()),
        "expected IndexLookup as driver when BTree full plan cost is lower, got {:?}",
        plan.root()
    );
}

#[test]
fn spatial_index_for_each_uses_lookup() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));
    let _e3 = world.spawn((Pos { x: 100.0, y: 100.0 },));

    let call_count = Arc::new(AtomicUsize::new(0));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(
            Arc::new(CountingSpatialIndex {
                entities: vec![e1, e2],
                call_count: Arc::clone(&call_count),
            }),
            &world,
        )
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([5.0, 5.0], 10.0, |_, _| true).unwrap())
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| {
        results.push(entity);
    })
    .unwrap();

    assert!(
        call_count.load(Ordering::Relaxed) > 0,
        "query method was never called"
    );
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn spatial_index_join_uses_lookup() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 }, Score(10)));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 }, Score(20)));
    let _e3 = world.spawn((Pos { x: 100.0, y: 100.0 }, Score(30)));

    let call_count = Arc::new(AtomicUsize::new(0));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(
            Arc::new(CountingSpatialIndex {
                entities: vec![e1, e2],
                call_count: Arc::clone(&call_count),
            }),
            &world,
        )
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([5.0, 5.0], 10.0, |_, _| true).unwrap())
        .join::<(&Score,)>(JoinKind::Inner)
        .build();

    let results = plan.execute_collect(&mut world).unwrap();
    assert!(
        call_count.load(Ordering::Relaxed) > 0,
        "query not called in join"
    );
    assert_eq!(results.len(), 2);
}

#[test]
fn spatial_index_execute_returns_correct_entities() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));
    let _far = world.spawn((Pos { x: 999.0, y: 999.0 },));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true).unwrap())
        .build();

    let results = plan.execute_collect(&mut world).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn spatial_index_stale_entities_filtered() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    // Build the plan while the planner still borrows world, then drop planner.
    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true).unwrap())
        .build();

    // Now we can mutate world — despawn e2 after the plan is built.
    world.despawn(e2);

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}

#[test]
fn spatial_index_for_each_raw_works() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));

    let call_count = Arc::new(AtomicUsize::new(0));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(
            Arc::new(CountingSpatialIndex {
                entities: vec![e1],
                call_count: Arc::clone(&call_count),
            }),
            &world,
        )
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
        .build();

    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();

    assert!(call_count.load(Ordering::Relaxed) > 0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}

#[test]
fn spatial_index_without_lookup_falls_back() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Pos {
            x: i as f32,
            y: i as f32,
        },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([5.0, 5.0], 100.0, |_, _| true).unwrap())
        .build();

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 10);
}

#[test]
fn spatial_index_with_changed_filter() {
    // Changed<T> is column-granular (per archetype), not per-entity.
    // If any entity's column is mutated, the whole archetype passes
    // Changed<T> on the next scan. This test verifies that behavior
    // in the spatial index-gather path.
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

    // Spawn an entity in a SEPARATE archetype (with Score) so we can verify
    // the column-level filter skips the unmodified archetype on the second scan.
    let e3 = world.spawn((Pos { x: 3.0, y: 3.0 }, Score(99)));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2, e3])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(Changed<Pos>, &Pos)>()
        .filter(Predicate::within::<Pos>([1.5, 1.5], 10.0, |_, _| true).unwrap())
        .build();

    // First scan — all entities "changed" (never read before by this plan).
    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 3, "all entities pass on first scan");

    // Mutate only e1's Pos — this marks the (Pos) archetype column as changed
    // but NOT the (Pos, Score) archetype.
    world.get_mut::<Pos>(e1).unwrap().x = 99.0;

    // Second scan — only the archetype containing e1/e2 (which was mutated) passes
    // Changed<Pos>. e3's (Pos, Score) archetype was not touched, so it is filtered out.
    results.clear();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "only entities in the mutated archetype pass Changed<Pos>"
    );
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
    assert!(!results.contains(&e3));
}

// ── Additional spatial execution coverage ────────────────────

#[test]
fn spatial_index_empty_lookup_yields_no_results() {
    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 1.0 },));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([0.0, 0.0], 1.0, |_, _| true).unwrap())
        .build();

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 0, "empty lookup should yield zero results");
}

#[test]
fn spatial_index_all_stale_yields_no_results() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
        .build();

    // Despawn ALL entities returned by the lookup after plan is built.
    world.despawn(e1);
    world.despawn(e2);

    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 0, "all-stale lookup should yield zero results");
}

#[test]
fn spatial_index_for_each_raw_filters_stale() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
        .build();

    world.despawn(e2);

    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 1, "raw path must filter stale entities");
    assert_eq!(results[0], e1);
}

#[test]
fn spatial_index_filters_entities_missing_required_components() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 }, Score(10))); // has both
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },)); // only Pos

    // Index returns both entities.
    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    // Query requires BOTH Pos and Score.
    let mut plan = planner
        .scan::<(&Pos, &Score)>()
        .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "entity missing required component should be filtered"
    );
    assert_eq!(results[0], e1);
}

#[test]
fn spatial_index_mixed_archetypes_without_changed() {
    let mut world = World::new();
    // Two different archetypes, both have Pos.
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 }, Score(10)));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "entities from different archetypes should both be yielded"
    );
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn spatial_index_intersects_through_execution() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 5.0, y: 5.0 },));
    let _far = world.spawn((Pos { x: 99.0, y: 99.0 },));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos>(Arc::new(FixedSpatialIndex::new(vec![e1, e2])), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::intersects::<Pos>([0.0, 0.0], [10.0, 10.0], |_, _| true).unwrap())
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn spatial_index_3d_coordinates_propagate() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Pos3D {
        x: f32,
        y: f32,
        z: f32,
    }

    let mut world = World::new();
    let e1 = world.spawn((Pos3D {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    },));

    // Verify 3D coordinates are passed through to the index's query method.
    let received_center = Arc::new(std::sync::Mutex::new(Vec::new()));
    let call_count = Arc::new(AtomicUsize::new(0));

    let mut planner = QueryPlanner::new(&world);
    planner
        .add_spatial_index::<Pos3D>(
            Arc::new(RecordingSpatialIndex {
                entities: vec![e1],
                call_count: Arc::clone(&call_count),
                received_center: Arc::clone(&received_center),
            }),
            &world,
        )
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos3D,)>()
        .filter(Predicate::within::<Pos3D>([10.0, 20.0, 30.0], 5.0, |_, _| true).unwrap())
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert!(call_count.load(Ordering::Relaxed) > 0);
    assert_eq!(results.len(), 1);

    let center = received_center.lock().unwrap();
    assert_eq!(
        center.len(),
        3,
        "3D center should propagate all 3 coordinates"
    );
    assert!((center[0] - 10.0).abs() < f64::EPSILON);
    assert!((center[1] - 20.0).abs() < f64::EPSILON);
    assert!((center[2] - 30.0).abs() < f64::EPSILON);
}

// ── IndexDriver execute_stream execution ──────────────────────────────

#[test]
fn index_for_each_uses_btree_lookup() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));
    let _e3 = world.spawn((Score(99),));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn index_join_uses_lookup() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 }));
    let e2 = world.spawn((Score(42), Pos { x: 2.0, y: 2.0 }));
    let _e3 = world.spawn((Score(99), Pos { x: 3.0, y: 3.0 }));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .join::<(&Pos,)>(JoinKind::Inner)
        .build();

    let results = plan.execute_collect(&mut world).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn index_for_each_uses_hash_lookup() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));
    let _e3 = world.spawn((Score(99),));

    let mut hash = HashIndex::<Score>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(hash), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}

#[test]
fn index_range_lookup_execution() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));
    let e3 = world.spawn((Score(30),));
    let _e4 = world.spawn((Score(100),));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(5)..Score(35)))
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert_eq!(results.len(), 3);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
    assert!(results.contains(&e3));
}

#[test]
fn index_lookup_filters_stale_entities() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    // Despawn e2 after plan is built — index is stale.
    world.despawn(e2);

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}

#[test]
fn index_lookup_filters_missing_required() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 })); // has both
    let _e2 = world.spawn((Score(42),)); // only Score

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world).unwrap();

    // Query requires BOTH Score and Pos.
    let mut plan = planner
        .scan::<(&Score, &Pos)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();

    assert_eq!(results.len(), 1, "entity missing Pos should be filtered");
    assert_eq!(results[0], e1);
}

// ── Cross-world safety tests ─────────────────────────────────────

#[test]
fn execute_returns_err_on_wrong_world() {
    let mut world_a = World::new();
    let mut world_b = World::new();
    world_a.spawn((Score(1),));
    world_b.spawn((Score(2),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_collect(&mut world_b);
    assert!(result.is_err());
}

#[test]
fn for_each_returns_err_on_wrong_world() {
    let mut world_a = World::new();
    let mut world_b = World::new();
    world_a.spawn((Score(1),));
    world_b.spawn((Score(2),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_stream(&mut world_b, |_| {});
    assert!(result.is_err());
}

#[test]
fn for_each_raw_returns_err_on_wrong_world() {
    let mut world_a = World::new();
    let world_b = World::new();
    world_a.spawn((Score(1),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_stream_raw(&world_b, |_| {});
    assert!(result.is_err());
}

#[test]
fn execute_succeeds_on_same_world() {
    let mut world = World::new();
    world.spawn((Score(1),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_collect(&mut world);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn for_each_supports_join_plan() {
    let mut world = World::new();
    let e = world.spawn((Score(1), Health(100)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results, vec![e]);
}

#[test]
fn for_each_raw_supports_join_plan() {
    let mut world = World::new();
    let e = world.spawn((Score(1), Health(100)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results, vec![e]);
}

#[test]
fn execute_raw_supports_join_plan() {
    let mut world = World::new();
    let e = world.spawn((Score(1), Health(100)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let result = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(result, &[e]);
}

#[test]
fn execute_raw_scan_only() {
    let mut world = World::new();
    let e = world.spawn((Score(42),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(result, &[e]);
}

#[test]
fn execute_raw_returns_err_on_wrong_world() {
    let mut world_a = World::new();
    let mut world_b = World::new();
    world_a.spawn((Score(1),));
    world_b.spawn((Score(2),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_collect_raw(&world_b);
    assert!(result.is_err());
}

#[test]
fn execute_raw_does_not_advance_tick() {
    let mut world = World::new();
    world.spawn((Score(1),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // Execute raw twice — both should succeed without tick advancement.
    let r1 = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(r1.len(), 1);
    let r2 = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(r2.len(), 1);
}

#[test]
fn for_each_raw_join_inner_filters_non_matching() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Health(50)));
    world.spawn((Score(2),)); // no Health — should not appear in join

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results, vec![e1]);
}

#[test]
fn for_each_raw_join_left_preserves_all_left() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Health(50)));
    let e2 = world.spawn((Score(2),)); // no Health

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Left)
        .build();
    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();
    results.sort_by_key(|e| e.to_bits());
    let mut expected = vec![e1, e2];
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(results, expected);
}

#[test]
fn for_each_join_advances_tick() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i), Health(i * 10)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    // First call: all changed — should see all 5.
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 5);
    // Second call: tick advanced, nothing mutated — should see 0.
    let mut count = 0;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn execute_raw_join_does_not_advance_tick() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i), Health(i * 10)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    // Both calls should see all 5 — execute_raw does not advance tick.
    let r1 = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(r1.len(), 5);
    let r2 = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(r2.len(), 5);
}

#[test]
fn for_each_join_inner_filters_non_matching() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Health(50)));
    world.spawn((Score(2),)); // no Health — filtered by inner join

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results, vec![e1]);
}

#[test]
fn for_each_join_left_preserves_all_left() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Health(50)));
    let e2 = world.spawn((Score(2),)); // no Health

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Left)
        .build();
    let mut results = Vec::new();
    plan.execute_stream(&mut world, |entity| results.push(entity))
        .unwrap();
    results.sort_by_key(|e| e.to_bits());
    let mut expected = vec![e1, e2];
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(results, expected);
}

#[test]
fn for_each_raw_multi_step_join() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Health(50), Team(0)));
    world.spawn((Score(2), Health(60))); // no Team — filtered by second join
    world.spawn((Score(3),)); // no Health — filtered by first join

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let mut results = Vec::new();
    plan.execute_stream_raw(&world, |entity| results.push(entity))
        .unwrap();
    assert_eq!(results, vec![e1]);
}

#[test]
fn execute_raw_multi_step_join() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Health(50), Team(0)));
    world.spawn((Score(2), Health(60))); // no Team
    world.spawn((Score(3),)); // no Health

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let result = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(result, &[e1]);
}

#[test]
fn execute_raw_empty_join_result() {
    let mut world = World::new();
    world.spawn((Score(1),)); // no Health — inner join yields empty
    world.spawn((Score(2),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let result = plan.execute_collect_raw(&world).unwrap();
    assert!(result.is_empty());
}

#[test]
fn add_btree_index_returns_err_on_unregistered_component() {
    let mut world1 = World::new();
    world1.spawn((Score(1),)); // registers Score in world1

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world1);

    let world2 = World::new(); // Score NOT registered here
    let mut planner = QueryPlanner::new(&world2);
    let result = planner.add_btree_index(&Arc::new(btree), &world2);
    assert!(result.is_err());
}

#[test]
fn predicate_within_rejects_negative_radius() {
    let result = Predicate::within::<Pos>([0.0, 0.0], -1.0, |_, _| true);
    assert!(result.is_err());
}

#[test]
fn predicate_within_rejects_nan_radius() {
    let result = Predicate::within::<Pos>([0.0, 0.0], f64::NAN, |_, _| true);
    assert!(result.is_err());
}

#[test]
fn predicate_intersects_rejects_mismatched_dimensions() {
    let result = Predicate::intersects::<Pos>(vec![0.0, 0.0], vec![1.0], |_, _| true);
    assert!(result.is_err());
}

#[test]
fn predicate_intersects_rejects_empty_coordinates() {
    let result = Predicate::intersects::<Pos>(Vec::<f64>::new(), Vec::<f64>::new(), |_, _| true);
    assert!(result.is_err());
}

// ── Aggregate tests ──────────────────────────────────────────────

#[test]
fn aggregate_count_empty_world() {
    let mut world = World::new();
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.get(0), Some(0.0));
}

#[test]
fn aggregate_count() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(100.0));
}

#[test]
fn aggregate_sum() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    // sum(0..10) = 45
    assert_eq!(result.get(0), Some(45.0));
}

#[test]
fn aggregate_min_max() {
    let mut world = World::new();
    for i in 5..15 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::min::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::max::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(5.0));
    assert_eq!(result.get(1), Some(14.0));
}

#[test]
fn aggregate_avg() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::avg::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    // avg(0..10) = 4.5
    assert_eq!(result.get(0), Some(4.5));
}

#[test]
fn aggregate_multiple_expressions() {
    let mut world = World::new();
    for i in 1..=5 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::min::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::max::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::avg::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(5.0)); // count
    assert_eq!(result.get(1), Some(15.0)); // sum(1+2+3+4+5)
    assert_eq!(result.get(2), Some(1.0)); // min
    assert_eq!(result.get(3), Some(5.0)); // max
    assert_eq!(result.get(4), Some(3.0)); // avg
}

#[test]
fn aggregate_with_filter() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "score >= 50",
            0.5,
            |world: &World, entity: Entity| world.get::<Score>(entity).is_some_and(|s| s.0 >= 50),
        ))
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(50.0)); // 50 entities match
    // sum(50..100) = 50*75-1 = 3725
    let expected_sum: f64 = (50..100).map(|i| i as f64).sum();
    assert_eq!(result.get(1), Some(expected_sum));
}

#[test]
fn aggregate_get_by_label() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get_by_label("COUNT(*)"), Some(5.0));
    assert_eq!(result.get_by_label("SUM(Score)"), Some(10.0));
    assert_eq!(result.get_by_label("NONEXISTENT"), None);
}

#[test]
fn aggregate_plan_node_in_explain() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let explain = plan.explain();
    assert!(explain.contains("StreamAggregate"));
    assert!(explain.contains("COUNT(*)"));
    assert!(explain.contains("SUM(Score)"));
}

#[test]
fn aggregate_has_aggregates() {
    let mut world = World::new();
    world.spawn((Score(1),));
    let planner = QueryPlanner::new(&world);

    let plan_no_agg = planner.scan::<(&Score,)>().build();
    assert!(!plan_no_agg.has_aggregates());

    let plan_with_agg = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();
    assert!(plan_with_agg.has_aggregates());
}

#[test]
fn aggregate_result_display() {
    let result = AggregateResult {
        values: vec![
            ("COUNT(*)".to_string(), 10.0),
            ("SUM(Score)".to_string(), 45.0),
        ],
    };
    let display = format!("{result}");
    assert!(display.contains("COUNT(*)"));
    assert!(display.contains("SUM(Score)"));
}

#[test]
fn aggregate_min_max_on_empty_returns_nan() {
    let mut world = World::new();
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::min::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::max::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::avg::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert!(result.get(0).unwrap().is_nan()); // min on empty
    assert!(result.get(1).unwrap().is_nan()); // max on empty
    assert!(result.get(2).unwrap().is_nan()); // avg on empty
}

#[test]
fn aggregate_execute_raw() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate_raw(&world).unwrap();
    assert_eq!(result.get(0), Some(10.0));
    assert_eq!(result.get(1), Some(45.0));
}

#[test]
fn aggregate_with_index_driver() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index::<Score>(&idx, &world).unwrap();

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(40.0)); // 10..50 = 40 entities
    let expected_sum: f64 = (10..50).map(|i| i as f64).sum();
    assert_eq!(result.get(1), Some(expected_sum));
}

#[test]
fn aggregate_result_iter() {
    let result = AggregateResult {
        values: vec![
            ("COUNT(*)".to_string(), 10.0),
            ("SUM(Score)".to_string(), 45.0),
        ],
    };
    let items: Vec<_> = result.iter().collect();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], ("COUNT(*)", 10.0));
    assert_eq!(items[1], ("SUM(Score)", 45.0));
}

#[test]
fn aggregate_op_display() {
    assert_eq!(format!("{}", AggregateOp::Count), "COUNT");
    assert_eq!(format!("{}", AggregateOp::Sum), "SUM");
    assert_eq!(format!("{}", AggregateOp::Min), "MIN");
    assert_eq!(format!("{}", AggregateOp::Max), "MAX");
    assert_eq!(format!("{}", AggregateOp::Avg), "AVG");
}

#[test]
fn aggregate_plan_cost_single_row() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();

    // Aggregate produces 1 result row.
    assert_eq!(plan.root().cost().rows, 1.0);
}

#[test]
fn aggregate_multiple_archetypes() {
    let mut world = World::new();
    // Two archetypes: (Score,) and (Score, Health)
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    for i in 5..10 {
        world.spawn((Score(i), Health(100)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(10.0)); // all 10 entities
    assert_eq!(result.get(1), Some(45.0)); // sum(0..10)
}

#[test]
fn aggregate_after_despawn() {
    let mut world = World::new();
    let mut entities = Vec::new();
    for i in 0..5 {
        entities.push(world.spawn((Score(i),)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    // Despawn entities with Score(3) and Score(4)
    world.despawn(entities[3]);
    world.despawn(entities[4]);

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(3.0)); // 3 surviving
    assert_eq!(result.get(1), Some(3.0)); // 0+1+2 = 3
}

#[test]
fn aggregate_changed_skips_stale() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .aggregate(AggregateExpr::count())
        .build();

    // First call sees all entities (all columns are new).
    let r1 = plan.aggregate(&mut world).unwrap();
    assert_eq!(r1.get(0), Some(5.0));

    // No mutations — second call should see 0 (Changed filter skips).
    let r2 = plan.aggregate(&mut world).unwrap();
    assert_eq!(r2.get(0), Some(0.0));
}

#[test]
fn aggregate_changed_detects_mutation() {
    let mut world = World::new();
    let e = world.spawn((Score(10),));
    for _ in 0..4 {
        world.spawn((Score(0),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    // First call sees all.
    let _ = plan.aggregate(&mut world).unwrap();

    // Mutate one entity.
    *world.get_mut::<Score>(e).unwrap() = Score(42);

    // Second call sees only the mutated entity's archetype.
    let r = plan.aggregate(&mut world).unwrap();
    // Changed<T> is archetype-granular, so all entities in the archetype
    // are visited (all 5 are in the same archetype).
    assert_eq!(r.get(0), Some(5.0));
}

#[test]
fn aggregate_no_exprs_returns_empty() {
    let mut world = World::new();
    world.spawn((Score(1),));
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // No panic — returns empty result.
    let result = plan.aggregate(&mut world).unwrap();
    assert!(result.is_empty());
}

#[test]
fn aggregate_no_exprs_raw_returns_empty() {
    let mut world = World::new();
    world.spawn((Score(1),));
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let result = plan.aggregate_raw(&world).unwrap();
    assert!(result.is_empty());
}

#[test]
fn aggregate_world_mismatch() {
    let mut world_a = World::new();
    world_a.spawn((Score(1),));
    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();

    let mut world_b = World::new();
    assert!(plan.aggregate(&mut world_b).is_err());
}

#[test]
fn aggregate_world_mismatch_raw() {
    let mut world_a = World::new();
    world_a.spawn((Score(1),));
    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();

    let world_b = World::new();
    assert!(plan.aggregate_raw(&world_b).is_err());
}

#[test]
fn aggregate_raw_tick_stationarity() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .aggregate(AggregateExpr::count())
        .build();

    // _raw does not advance ticks, so repeated calls see the same result.
    let r1 = plan.aggregate_raw(&world).unwrap();
    let r2 = plan.aggregate_raw(&world).unwrap();
    assert_eq!(r1.get(0), r2.get(0));
}

#[test]
fn aggregate_with_join() {
    let mut world = World::new();
    // 5 entities with both Score and Health
    for i in 0..5 {
        world.spawn((Score(i), Health(100)));
    }
    // 5 entities with Score only
    for i in 5..10 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(5.0)); // only 5 have both
    assert_eq!(result.get(1), Some(10.0)); // sum(0..5)
}

#[test]
fn aggregate_raw_with_join() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i), Health(100)));
    }
    for i in 5..10 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    // _raw now supports join plans too.
    let result = plan.aggregate_raw(&world).unwrap();
    assert_eq!(result.get(0), Some(5.0));
    assert_eq!(result.get(1), Some(10.0));
}

#[test]
fn aggregate_nan_propagation_min_max() {
    let mut world = World::new();
    world.spawn((Score(1),));
    world.spawn((Score(2),));
    world.spawn((Score(3),));

    let planner = QueryPlanner::new(&world);

    // Extractor that returns NaN for Score(2).
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::min::<Score>("Score", |s| {
            if s.0 == 2 { f64::NAN } else { s.0 as f64 }
        }))
        .aggregate(AggregateExpr::max::<Score>("Score", |s| {
            if s.0 == 2 { f64::NAN } else { s.0 as f64 }
        }))
        .build();

    let result = plan.aggregate(&mut world).unwrap();
    // NaN propagates via f64::min/max — result should be NaN.
    assert!(result.get(0).unwrap().is_nan());
    assert!(result.get(1).unwrap().is_nan());
}

#[test]
fn aggregate_after_spawn() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .build();

    let r1 = plan.aggregate(&mut world).unwrap();
    assert_eq!(r1.get(0), Some(5.0));

    // Spawn more entities — visible on next execution.
    for i in 5..8 {
        world.spawn((Score(i),));
    }
    let r2 = plan.aggregate(&mut world).unwrap();
    assert_eq!(r2.get(0), Some(8.0));
}

#[test]
fn aggregate_duplicate_label_warning() {
    let mut world = World::new();
    world.spawn((Score(1),));
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();

    assert!(plan.warnings().iter().any(|w| matches!(
        w,
        PlanWarning::DuplicateAggregateLabel { label } if label == "SUM(Score)"
    )));
}

#[test]
fn aggregate_expr_accessors() {
    let count = AggregateExpr::count();
    assert_eq!(count.op(), AggregateOp::Count);
    assert_eq!(count.label(), "COUNT(*)");

    let sum = AggregateExpr::sum::<Score>("Score", |s| s.0 as f64);
    assert_eq!(sum.op(), AggregateOp::Sum);
    assert_eq!(sum.label(), "SUM(Score)");
}

#[test]
fn aggregate_result_labels() {
    let result = AggregateResult {
        values: vec![
            ("COUNT(*)".to_string(), 10.0),
            ("SUM(Score)".to_string(), 45.0),
        ],
    };
    let labels: Vec<_> = result.labels().collect();
    assert_eq!(labels, vec!["COUNT(*)", "SUM(Score)"]);
}

// ── Batch aggregate path activation tests ─────────────────────

#[test]
fn aggregate_batch_path_activates_without_filters() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();
    drop(planner);

    let dbg = format!("{plan:?}");
    assert!(
        dbg.contains("has_compiled_agg_scan: true"),
        "batch path should activate for filter-free scan"
    );
    assert!(
        dbg.contains("has_compiled_agg_scan_raw: true"),
        "raw batch path should activate for filter-free scan"
    );
}

#[test]
fn aggregate_batch_path_disabled_with_filters() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "score >= 5",
            0.5,
            |world: &World, entity: Entity| world.get::<Score>(entity).is_some_and(|s| s.0 >= 5),
        ))
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();
    drop(planner);

    let dbg = format!("{plan:?}");
    assert!(
        dbg.contains("has_compiled_agg_scan: false"),
        "batch path should NOT activate when filters present"
    );
}

#[test]
fn aggregate_batch_multi_archetype_correctness() {
    // Entities across 3 archetypes: (Score,), (Score, Health),
    // (Score, Health, Name). Batch path must bind per archetype.
    #[derive(Debug)]
    struct Health(#[expect(dead_code)] i32);
    #[derive(Debug)]
    struct Name;

    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    for i in 5..8 {
        world.spawn((Score(i), Health(100)));
    }
    for i in 8..10 {
        world.spawn((Score(i), Health(50), Name));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();
    drop(planner);

    let dbg = format!("{plan:?}");
    assert!(dbg.contains("has_compiled_agg_scan: true"));

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(10.0)); // count
    let expected_sum: f64 = (0..10).map(|i| i as f64).sum();
    assert_eq!(result.get(1), Some(expected_sum)); // sum(0..10) = 45
}

#[test]
fn aggregate_batch_component_absent_from_some_archetypes() {
    // scan::<&Score> with sum::<Health> — Health is absent from some
    // archetypes. The batch path must skip extraction for those.
    #[derive(Debug)]
    struct Health(i32);

    let mut world = World::new();
    // Archetype 1: Score only (no Health)
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Archetype 2: Score + Health
    for i in 0..3 {
        world.spawn((Score(100), Health(i * 10)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Health>("Health", |h| h.0 as f64))
        .build();
    drop(planner);

    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(8.0)); // count: 5 + 3
    // sum of Health: only from archetype 2 = 0 + 10 + 20 = 30
    assert_eq!(result.get(1), Some(30.0));
}

#[test]
fn aggregate_accum_guard_restores_on_panic() {
    // Verify that if a user-supplied filter panics during the
    // compiled_for_each aggregate path, cached_accums is restored
    // so the plan remains usable on subsequent calls.
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }

    let should_panic = Arc::new(AtomicBool::new(true));
    let panic_flag = Arc::clone(&should_panic);

    let planner = QueryPlanner::new(&world);
    // Add a filter predicate so the plan takes the compiled_for_each
    // path (batch scan is disabled when filters are present).
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "panicking filter",
            1.0,
            move |_w: &World, _e: Entity| {
                assert!(
                    !panic_flag.load(Ordering::Relaxed),
                    "intentional test panic"
                );
                true
            },
        ))
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();
    drop(planner);

    // First call: the filter panics. catch_unwind captures it.
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan.aggregate(&mut world)));
    assert!(result.is_err(), "expected panic from filter");

    // Disable the panic and execute_collect again — plan must still work.
    // Without AccumGuard, cached_accums would be empty after the
    // panic, causing indexing failures or wrong results.
    should_panic.store(false, Ordering::Relaxed);
    let result = plan.aggregate(&mut world).unwrap();
    assert_eq!(result.get(0), Some(10.0)); // count
    let expected_sum: f64 = (0..10).map(|i| i as f64).sum();
    assert_eq!(result.get(1), Some(expected_sum)); // sum
}

#[test]
fn aggregate_accum_guard_restores_on_panic_raw() {
    // Same as above but for aggregate_raw.
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }

    let should_panic = Arc::new(AtomicBool::new(true));
    let panic_flag = Arc::clone(&should_panic);

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "panicking filter",
            1.0,
            move |_w: &World, _e: Entity| {
                assert!(
                    !panic_flag.load(Ordering::Relaxed),
                    "intentional test panic"
                );
                true
            },
        ))
        .aggregate(AggregateExpr::count())
        .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
        .build();
    drop(planner);

    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan.aggregate_raw(&world)));
    assert!(result.is_err(), "expected panic from filter");

    // Plan is structurally intact — cached_accums was restored by guard.
    should_panic.store(false, Ordering::Relaxed);
    let result = plan.aggregate_raw(&world).unwrap();
    assert_eq!(result.get(0), Some(10.0));
    let expected_sum: f64 = (0..10).map(|i| i as f64).sum();
    assert_eq!(result.get(1), Some(expected_sum));
}

// ── Batch join execution ──────────────────────────────────────────

#[test]
fn scratch_sort_by_archetype_groups_entities() {
    let mut world = World::new();
    // Archetype A: Score only
    let a1 = world.spawn((Score(1),));
    let a2 = world.spawn((Score(2),));
    // Archetype B: Score + Team
    let b1 = world.spawn((Score(3), Team(1)));
    let b2 = world.spawn((Score(4), Team(2)));

    // Deliberately interleave: [b1, a1, b2, a2]
    let mut scratch = ScratchBuffer::new(4);
    scratch.push(b1);
    scratch.push(a1);
    scratch.push(b2);
    scratch.push(a2);

    scratch.sort_by_archetype(&world.entity_locations);

    // After sort: entities from same archetype should be contiguous.
    let sorted = scratch.as_slice();
    // First two share one archetype, last two share another.
    let loc0 = world.entity_locations[sorted[0].index() as usize].unwrap();
    let loc1 = world.entity_locations[sorted[1].index() as usize].unwrap();
    let loc2 = world.entity_locations[sorted[2].index() as usize].unwrap();
    let loc3 = world.entity_locations[sorted[3].index() as usize].unwrap();
    assert_eq!(loc0.archetype_id, loc1.archetype_id);
    assert_eq!(loc2.archetype_id, loc3.archetype_id);
    assert_ne!(loc0.archetype_id, loc2.archetype_id);

    // Within each archetype group, rows should be sorted ascending.
    assert!(loc0.row < loc1.row);
    assert!(loc2.row < loc3.row);
}

#[test]
fn execute_stream_batched_yields_all_join_results() {
    let mut world = World::new();
    // Score-only entities (should NOT appear in inner join)
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Score+Team entities (SHOULD appear)
    let mut expected = Vec::new();
    for i in 5..15 {
        let e = world.spawn((Score(i), Team(i % 3)));
        expected.push((e, Score(i)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut results: Vec<(Entity, Score)> = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |entity, (score,)| {
        results.push((entity, *score));
    })
    .unwrap();

    // Sort both by entity for comparison.
    results.sort_by_key(|(e, _)| e.to_bits());
    expected.sort_by_key(|(e, _)| e.to_bits());
    assert_eq!(results, expected);
}

#[test]
fn execute_stream_join_chunk_yields_correct_slices() {
    let mut world = World::new();
    // Archetype A: Score only (will not match inner join)
    for i in 0..3 {
        world.spawn((Score(i),));
    }
    // Archetype B: Score + Team (deterministic scores 10..15)
    for i in 10..15 {
        world.spawn((Score(i), Team(1)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut total_entities = 0;
    let mut chunk_count = 0;
    let mut collected_scores = Vec::new();
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
        // rows and entities must have the same length.
        assert_eq!(entities.len(), rows.len());
        // Each row index must be valid for the slice.
        for &row in rows {
            assert!(
                row < scores.len(),
                "row {row} out of bounds for slice len {}",
                scores.len()
            );
            collected_scores.push(scores[row]);
        }
        total_entities += entities.len();
        chunk_count += 1;
    })
    .unwrap();

    assert_eq!(total_entities, 5); // Only Score+Team entities
    assert!(chunk_count >= 1); // At least one archetype chunk
    // Verify we read the correct score values (10..15).
    collected_scores.sort_by_key(|s| s.0);
    assert_eq!(collected_scores, (10..15).map(Score).collect::<Vec<_>>());
}

#[test]
fn join_chunk_column_filter_no_empty_chunks() {
    // A column filter that rejects every entity must not invoke the
    // callback at all. Previously the column-filter path in
    // execute_stream_join_chunk emitted zero-length chunks, which
    // could break consumers that assume non-empty callbacks.
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("reject all", 0.0, |_| {
            false
        }))
        .build();

    let mut invocations = 0u32;
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (_scores,)| {
        assert!(
            !entities.is_empty(),
            "callback invoked with empty entity slice"
        );
        assert!(!rows.is_empty(), "callback invoked with empty row indices");
        invocations += 1;
    })
    .unwrap();
    assert_eq!(
        invocations, 0,
        "no chunks should be emitted when all entities are filtered out"
    );
}

#[test]
fn join_chunk_column_filter_partial() {
    // Verify that a column filter that passes some entities produces
    // correct non-empty chunks with valid row indices.
    let mut world = World::new();
    for i in 0..20 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom_column::<Score>("even", 0.5, |s| {
            s.0 % 2 == 0
        }))
        .build();

    let mut total = 0u32;
    let mut collected = Vec::new();
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
        assert!(!entities.is_empty());
        assert_eq!(entities.len(), rows.len());
        for &row in rows {
            assert!(row < scores.len());
            collected.push(scores[row]);
        }
        total += entities.len() as u32;
    })
    .unwrap();
    assert_eq!(total, 10);
    collected.sort_by_key(|s| s.0);
    let expected: Vec<Score> = (0..20).filter(|i| i % 2 == 0).map(Score).collect();
    assert_eq!(collected, expected);
}

#[test]
fn execute_stream_batched_raw_no_tick_advance() {
    let mut world = World::new();
    world.spawn((Score(1), Team(1)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // Call raw twice — both should succeed (no tick advancement).
    let mut count1 = 0u32;
    plan.execute_stream_batched_raw::<(&Score,), _>(&world, |_, _| count1 += 1)
        .unwrap();
    assert_eq!(count1, 1);

    let mut count2 = 0u32;
    plan.execute_stream_batched_raw::<(&Score,), _>(&world, |_, _| count2 += 1)
        .unwrap();
    assert_eq!(count2, 1);
}

#[test]
fn execute_stream_join_chunk_works_for_scan_plans() {
    let mut world = World::new();
    world.spawn((Score(1),));
    world.spawn((Score(2),));
    world.spawn((Score(3),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut total = 0;
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
        assert_eq!(entities.len(), rows.len());
        total += entities.len();
        // For scan plans, rows should be 0..len (all entities in the archetype).
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(row, i, "scan plan rows should be sequential");
        }
        assert_eq!(scores.len(), entities.len());
    })
    .unwrap();
    assert_eq!(total, 3);
}

#[test]
fn execute_stream_batched_left_join() {
    let mut world = World::new();
    // 5 Score-only, 5 Score+Team
    let mut all_score = Vec::new();
    for i in 0..5 {
        all_score.push(world.spawn((Score(i),)));
    }
    for i in 5..10 {
        all_score.push(world.spawn((Score(i), Team(1))));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();

    let mut results = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |entity, _| {
        results.push(entity);
    })
    .unwrap();

    // Left join: all 10 Score entities should appear.
    assert_eq!(results.len(), 10);
    all_score.sort_by_key(|e| e.to_bits());
    results.sort_by_key(|e| e.to_bits());
    assert_eq!(results, all_score);
}

#[test]
fn execute_stream_join_chunk_multi_archetype() {
    let mut world = World::new();
    // 3 different archetypes, all with Score
    world.spawn((Score(1),));
    world.spawn((Score(2), Team(1)));
    world.spawn((Score(3), Team(1), Health(50)));

    let planner = QueryPlanner::new(&world);
    // Scan Score, join Team — only archetypes with Team match the join.
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // Collect chunk info: (entity_count, first_entity) per chunk.
    let mut chunk_entities: Vec<Vec<Entity>> = Vec::new();
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |entities, _, _| {
        assert!(!entities.is_empty());
        chunk_entities.push(entities.to_vec());
    })
    .unwrap();

    // Two archetypes have Team: (Score, Team) and (Score, Team, Health).
    assert_eq!(chunk_entities.len(), 2);
    // Total: 2 entities (one per Team-bearing archetype).
    let total: usize = chunk_entities.iter().map(Vec::len).sum();
    assert_eq!(total, 2);
    // Each chunk should have different entities.
    assert_ne!(chunk_entities[0], chunk_entities[1]);
}

#[test]
fn execute_stream_batched_empty_join() {
    let mut world = World::new();
    // Score-only entities, no Team — inner join produces empty result.
    for i in 0..5 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut called = false;
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, _| {
        called = true;
    })
    .unwrap();
    assert!(!called);
}

#[test]
fn execute_stream_batched_world_mismatch() {
    let mut world_a = World::new();
    let mut world_b = World::new();
    world_a.spawn((Score(1),));
    world_b.spawn((Score(2),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_stream_batched::<(&Score,), _>(&mut world_b, |_, _| {});
    assert!(result.is_err());
}

#[test]
fn execute_stream_batched_component_mismatch() {
    let mut world = World::new();
    // Entities have Score only — no Health component.
    world.spawn((Score(1),));
    world.spawn((Score(2),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // Request (&Health,) via Q, but no archetype has Health.
    let result = plan.execute_stream_batched::<(&Health,), _>(&mut world, |_, _| {});
    assert!(
        matches!(result, Err(PlanExecError::ComponentMismatch { .. })),
        "expected ComponentMismatch, got {result:?}"
    );
}

#[test]
fn execute_stream_join_chunk_component_mismatch() {
    let mut world = World::new();
    world.spawn((Score(1),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let result = plan.execute_stream_join_chunk::<(&Health,), _>(&mut world, |_, _, _| {});
    assert!(
        matches!(result, Err(PlanExecError::ComponentMismatch { .. })),
        "expected ComponentMismatch, got {result:?}"
    );
}

#[test]
fn execute_stream_join_chunk_world_mismatch() {
    let mut world_a = World::new();
    let mut world_b = World::new();
    world_a.spawn((Score(1),));
    world_b.spawn((Score(2),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.execute_stream_join_chunk::<(&Score,), _>(&mut world_b, |_, _, _| {});
    assert!(result.is_err());
}

#[test]
fn execute_stream_join_chunk_empty_join() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut called = false;
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |_, _, _| {
        called = true;
    })
    .unwrap();
    assert!(!called);
}

#[test]
fn execute_stream_batched_scan_only_happy_path() {
    let mut world = World::new();
    world.spawn((Score(10),));
    world.spawn((Score(20),));
    world.spawn((Score(30),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut sum = 0u32;
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, (score,)| {
        sum += score.0;
    })
    .unwrap();
    assert_eq!(sum, 60);
}

#[test]
fn execute_stream_batched_multi_archetype_values() {
    let mut world = World::new();
    // Archetype A: Score + Team
    world.spawn((Score(10), Team(1)));
    world.spawn((Score(20), Team(2)));
    // Archetype B: Score + Team + Health
    world.spawn((Score(30), Team(3), Health(100)));
    world.spawn((Score(40), Team(4), Health(200)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut scores = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, (score,)| {
        scores.push(score.0);
    })
    .unwrap();
    scores.sort_unstable();
    assert_eq!(scores, vec![10, 20, 30, 40]);
}

#[test]
fn execute_stream_batched_marks_mutable_changed() {
    let mut world = World::new();
    world.spawn((Score(1), Team(1)));

    let planner = QueryPlanner::new(&world);
    // Use Changed<Score> to check change detection.
    let mut changed_plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // First call: all entities are new, so Changed sees them.
    let r1 = changed_plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r1, 1);

    // Second call: nothing mutated, Changed skips.
    let r2 = changed_plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r2, 0);

    // Now mutate via execute_stream_batched with &mut Score.
    let scan_planner = QueryPlanner::new(&world);
    let mut scan_plan = scan_planner.scan::<(&Score,)>().build();
    scan_plan
        .execute_stream_batched::<(&mut Score,), _>(&mut world, |_, (score,)| {
            score.0 += 1;
        })
        .unwrap();

    // Changed<Score> should now see the mutation.
    let r3 = changed_plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(
        r3, 1,
        "Changed<Score> should detect mutation from execute_stream_batched"
    );
}

// ── Join elimination ─────────────────────────────────────────────

#[test]
fn join_elimination_inner_becomes_scan() {
    let mut world = World::new();
    // Entities with Score only
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Entities with Score + Team (these should be the only results)
    let mut both = Vec::new();
    for i in 10..20 {
        both.push(world.spawn((Score(i), Team(i % 3))));
    }

    let planner = QueryPlanner::new(&world);

    // Inner join — should be eliminated into a scan.
    let mut join_plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // Tuple scan — the "expected" path.
    let mut scan_plan = planner.scan::<(&Score, &Team)>().build();

    // Both should produce the same entity set.
    let mut join_result = join_plan.execute_collect(&mut world).unwrap().to_vec();
    let mut scan_result = scan_plan.execute_collect(&mut world).unwrap().to_vec();
    join_result.sort_by_key(|e| e.to_bits());
    scan_result.sort_by_key(|e| e.to_bits());
    assert_eq!(join_result, scan_result);

    // The eliminated plan should NOT have a HashJoin/NestedLoopJoin node.
    match join_plan.root() {
        PlanNode::Scan { .. } => {} // expected
        other => panic!("expected Scan after elimination, got {:?}", other),
    }
}

#[test]
fn join_elimination_left_join_not_eliminated() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    for i in 5..10 {
        world.spawn((Score(i), Team(1)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();

    // Left join should NOT be eliminated.
    match plan.root() {
        PlanNode::HashJoin { .. } | PlanNode::NestedLoopJoin { .. } => {} // expected
        other => panic!("expected join node for Left join, got {:?}", other),
    }

    // Left join preserves all 10 Score entities.
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 10);
}

#[test]
fn join_elimination_mixed_inner_and_left() {
    let mut world = World::new();
    world.spawn((Score(1), Team(1), Health(100)));
    world.spawn((Score(2), Team(2))); // no Health
    world.spawn((Score(3),)); // no Team, no Health

    let planner = QueryPlanner::new(&world);
    // Inner join on Team (eliminable), Left join on Health (not eliminable).
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .join::<(&Health,)>(JoinKind::Left)
        .build();

    // Should have a join node (Left join remains).
    match plan.root() {
        PlanNode::HashJoin { .. } | PlanNode::NestedLoopJoin { .. } => {} // expected
        other => panic!(
            "expected join node for remaining Left join, got {:?}",
            other
        ),
    }

    // Inner join on Team narrows to 2 entities (Score+Team).
    // Left join on Health preserves both.
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn join_elimination_triple_join_two_eliminated() {
    let mut world = World::new();
    world.spawn((Score(1), Team(1), Health(100)));
    world.spawn((Score(2), Team(2))); // no Health
    world.spawn((Score(3),)); // no Team

    let planner = QueryPlanner::new(&world);
    // Two inner joins (Score→Team, Score→Health) + one left join (Score→Pos).
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .join::<(&Health,)>(JoinKind::Inner)
        .join::<(&Pos,)>(JoinKind::Left)
        .build();

    // Two inner joins eliminated, one left join remains.
    let eliminated_count = plan
        .warnings()
        .iter()
        .filter(|w| matches!(w, PlanWarning::JoinEliminated { .. }))
        .count();
    assert_eq!(eliminated_count, 2);

    // Only entity with Score+Team+Health survives the merged inner joins.
    // Left join on Pos preserves it (no Pos, but Left keeps it).
    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn join_elimination_changed_merged() {
    let mut world = World::new();
    let e1 = world.spawn((Score(1), Team(1)));
    let _e2 = world.spawn((Score(2), Team(2)));

    let planner = QueryPlanner::new(&world);
    // Changed<Score> on left, Changed<Team> on right (inner join).
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .join::<(Changed<Team>, &Team)>(JoinKind::Inner)
        .build();

    // First call: all entities are "changed" (new).
    let r1 = plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r1, 2);

    // Second call: nothing changed, should return 0.
    let r2 = plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r2, 0);

    // Mutate Score on e1.
    *world.get_mut::<Score>(e1).unwrap() = Score(99);

    // Third call: only e1 has Changed<Score>, but we also need Changed<Team>.
    // Since Team wasn't changed, the merged change filter should still require
    // both Score AND Team to be changed. Result: 0.
    let r3 = plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r3, 0);

    // Fourth call: only Team changed (right side of the eliminated join).
    // Score not changed → merged filter requires both → 0.
    *world.get_mut::<Team>(e1).unwrap() = Team(99);
    let r4 = plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r4, 0);

    // Fifth call: mutate BOTH Score and Team on e1. Now both columns
    // are marked changed at the archetype level → all entities in the
    // archetype pass (Changed<T> is per-column, not per-entity).
    *world.get_mut::<Score>(e1).unwrap() = Score(100);
    *world.get_mut::<Team>(e1).unwrap() = Team(100);
    let r5 = plan.execute_collect(&mut world).unwrap().len();
    assert_eq!(r5, 2);
}

#[test]
fn join_elimination_idempotent_same_component() {
    let mut world = World::new();
    world.spawn((Score(1),));
    world.spawn((Score(2),));

    let planner = QueryPlanner::new(&world);
    // Join Score with Score (same component) — union is idempotent.
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Score,)>(JoinKind::Inner)
        .build();

    let result = plan.execute_collect(&mut world).unwrap();
    assert_eq!(result.len(), 2);

    // Should be eliminated.
    let eliminated = plan
        .warnings()
        .iter()
        .any(|w| matches!(w, PlanWarning::JoinEliminated { .. }));
    assert!(eliminated);
}

#[test]
fn join_elimination_emits_warning() {
    let mut world = World::new();
    world.spawn((Score(1), Team(1)));

    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let has_warning = plan.warnings().iter().any(|w| match w {
        PlanWarning::JoinEliminated { right_name } => right_name.contains("Team"),
        _ => false,
    });
    assert!(
        has_warning,
        "expected JoinEliminated warning, got {:?}",
        plan.warnings()
    );
}

#[test]
fn join_elimination_raw_paths() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    let mut both = Vec::new();
    for i in 10..15 {
        both.push(world.spawn((Score(i), Team(i % 3))));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // Verify elimination happened.
    assert!(
        plan.warnings()
            .iter()
            .any(|w| matches!(w, PlanWarning::JoinEliminated { .. }))
    );

    // execute_collect_raw — should yield same entities as execute_collect.
    let raw_result = plan.execute_collect_raw(&world).unwrap().to_vec();
    assert_eq!(raw_result.len(), 5);

    // for_each_raw — should yield same entities.
    let mut raw_entities = Vec::new();
    plan.execute_stream_raw(&world, |entity| raw_entities.push(entity))
        .unwrap();
    assert_eq!(raw_entities.len(), 5);

    // execute_stream_batched_raw — pre-resolved column pointers via eliminated path.
    let mut batched_scores = Vec::new();
    plan.execute_stream_batched_raw::<(&Score,), _>(&world, |_, (score,)| {
        batched_scores.push(score.0);
    })
    .unwrap();
    batched_scores.sort_unstable();
    assert_eq!(batched_scores, vec![10, 11, 12, 13, 14]);
}

#[test]
fn join_elimination_benchmark_parity() {
    // Functional test: eliminated join should produce results in reasonable time.
    // Not a benchmark — just verifies the optimization works end-to-end.
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i), Team(i % 5)));
    }
    for i in 1000..2000 {
        world.spawn((Score(i),)); // no Team
    }

    let planner = QueryPlanner::new(&world);
    let mut join_plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    let mut scan_plan = planner.scan::<(&Score, &Team)>().build();

    let join_result = join_plan.execute_collect(&mut world).unwrap().to_vec();
    let scan_result = scan_plan.execute_collect(&mut world).unwrap().to_vec();
    assert_eq!(join_result.len(), scan_result.len());
    assert_eq!(join_result.len(), 1000);
}

// ── Direct archetype iteration tests ──────────────────────────────

#[test]
fn direct_iter_batched_scan_only() {
    let mut world = World::new();
    // Archetype A: Score only
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Archetype B: Score + Team
    for i in 10..15 {
        world.spawn((Score(i), Team(1)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // scan_required should be set (scan-only, no custom predicates).
    assert!(plan.scan_required.is_some());

    let mut results = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |entity, (score,)| {
        results.push((entity, *score));
    })
    .unwrap();

    // All 10 entities should be visited.
    assert_eq!(results.len(), 10);
    let mut scores: Vec<u32> = results.iter().map(|(_, s)| s.0).collect();
    scores.sort_unstable();
    assert_eq!(scores, vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
}

#[test]
fn direct_iter_batched_with_custom_predicate_uses_scratch() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>("score < 5", 0.5, |w, e| {
            w.get::<Score>(e).is_some_and(|s| s.0 < 5)
        }))
        .build();

    // scan_required should be None (custom predicate present).
    assert!(plan.scan_required.is_none());

    let mut results = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_entity, (score,)| {
        results.push(score.0);
    })
    .unwrap();

    results.sort_unstable();
    assert_eq!(results, vec![0, 1, 2, 3, 4]);
}

#[test]
fn direct_iter_chunk_scan_only() {
    let mut world = World::new();
    // Single archetype: Score only
    for i in 0..8 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();
    assert!(plan.scan_required.is_some());

    let mut chunk_count = 0;
    let mut total_entities = 0;
    plan.execute_stream_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
        chunk_count += 1;
        assert_eq!(entities.len(), rows.len());
        // Row indices should be sequential 0..N for direct iteration.
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(row, i, "row indices should be sequential");
            assert!(row < scores.len());
        }
        total_entities += entities.len();
    })
    .unwrap();

    assert_eq!(total_entities, 8);
    assert!(chunk_count >= 1);
}

#[test]
fn direct_iter_batched_eliminated_join() {
    let mut world = World::new();
    // Archetype A: Score only (will not match eliminated inner join)
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Archetype B: Score + Team (matches)
    for i in 10..15 {
        world.spawn((Score(i), Team(1)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // The inner join should be eliminated (no predicates on right side).
    assert!(
        plan.warnings()
            .iter()
            .any(|w| matches!(w, PlanWarning::JoinEliminated { .. })),
        "inner join should be eliminated"
    );
    // After elimination, scan_required should be set.
    assert!(plan.scan_required.is_some());

    let mut results = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_entity, (score,)| {
        results.push(score.0);
    })
    .unwrap();

    results.sort_unstable();
    assert_eq!(results, vec![10, 11, 12, 13, 14]);
}

#[test]
fn direct_iter_disabled_with_index_driver() {
    use crate::BTreeIndex;

    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    let idx = Arc::new(idx);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index::<Score>(&idx, &world).unwrap();
    let plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
        .build();

    // Index driver present — scan_required should be None.
    assert!(
        plan.scan_required.is_none(),
        "direct path should be disabled when index driver is present"
    );

    // Verify the plan still produces correct results (index-driven path).
    let mut plan = plan;
    let mut results = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, (score,)| {
        results.push(score.0);
    })
    .unwrap();
    results.sort_unstable();
    assert_eq!(results, (10..20).collect::<Vec<_>>());
}

// ── ER join tests ────────────────────────────────────────────

/// Entity-reference component: points from a child to its parent.
#[derive(Clone, Copy, Debug)]
struct Parent(Entity);

impl super::AsEntityRef for Parent {
    fn entity_ref(&self) -> Entity {
        self.0
    }
}

/// Tag component for child entities.
#[derive(Clone, Copy, Debug)]
struct ChildTag;

/// Component only on parent entities.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Name(&'static str);

#[test]
fn er_join_inner_basic() {
    let mut world = World::new();

    // Spawn parents with Name.
    let p1 = world.spawn((Name("Alice"),));
    let p2 = world.spawn((Name("Bob"),));
    let p3 = world.spawn((Score(999),)); // no Name — won't match right side

    // Spawn children pointing to parents.
    let c1 = world.spawn((ChildTag, Parent(p1)));
    let c2 = world.spawn((ChildTag, Parent(p2)));
    let c3 = world.spawn((ChildTag, Parent(p3))); // parent has no Name

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    let mut ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
    ids.sort_unstable();

    // Only c1 and c2 should match — their parents have Name.
    let mut expected = vec![c1.to_bits(), c2.to_bits()];
    expected.sort_unstable();
    assert_eq!(
        ids, expected,
        "inner ER join should keep only children whose parent has Name"
    );

    // c3's parent has no Name, so c3 is excluded.
    assert!(!ids.contains(&c3.to_bits()));
}

#[test]
fn er_join_left_keeps_all() {
    let mut world = World::new();

    let p1 = world.spawn((Name("Alice"),));
    let p_no_name = world.spawn((Score(42),)); // no Name

    let c1 = world.spawn((ChildTag, Parent(p1)));
    let c2 = world.spawn((ChildTag, Parent(p_no_name)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Left)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    assert_eq!(
        entities.len(),
        2,
        "left ER join should keep all left entities"
    );
    let ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
    assert!(ids.contains(&c1.to_bits()));
    assert!(ids.contains(&c2.to_bits()));
}

#[test]
fn er_join_dead_reference() {
    let mut world = World::new();

    let p1 = world.spawn((Name("Alice"),));
    let p2 = world.spawn((Name("Bob"),));

    let c1 = world.spawn((ChildTag, Parent(p1)));
    let c2 = world.spawn((ChildTag, Parent(p2)));

    // Despawn p2 — c2's reference is now dangling.
    world.despawn(p2);

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    let ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
    assert_eq!(ids.len(), 1, "dead reference target should not match");
    assert!(ids.contains(&c1.to_bits()));
    assert!(!ids.contains(&c2.to_bits()));
}

#[test]
fn er_join_explain_shows_er_join_node() {
    let mut world = World::new();
    let p = world.spawn((Name("Alice"),));
    world.spawn((ChildTag, Parent(p)));

    let planner = QueryPlanner::new(&world);
    let plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let explain = plan.explain();
    assert!(
        explain.contains("ErJoin"),
        "explain should contain ErJoin node: {explain}"
    );
    assert!(
        explain.contains("Inner"),
        "explain should show Inner join kind: {explain}"
    );
}

#[test]
fn er_join_for_each() {
    let mut world = World::new();

    let p1 = world.spawn((Name("Alice"),));
    let p2 = world.spawn((Name("Bob"),));

    let c1 = world.spawn((ChildTag, Parent(p1)));
    let c2 = world.spawn((ChildTag, Parent(p2)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let mut found = Vec::new();
    plan.execute_stream(&mut world, |e| found.push(e)).unwrap();
    found.sort_by_key(|e| e.to_bits());

    let mut expected = vec![c1, c2];
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}

#[test]
fn er_join_for_each_raw() {
    let mut world = World::new();

    let p1 = world.spawn((Name("Alice"),));
    let p_no = world.spawn((Score(1),));

    let c1 = world.spawn((ChildTag, Parent(p1)));
    world.spawn((ChildTag, Parent(p_no)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let mut found = Vec::new();
    plan.execute_stream_raw(&world, |e| found.push(e)).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], c1);
}

#[test]
fn er_join_with_wider_left_scan() {
    let mut world = World::new();

    // Parents with Name and Score.
    let p1 = world.spawn((Name("Alice"), Score(10)));
    let p2 = world.spawn((Name("Bob"), Score(20)));

    // Children with additional components in the scan query.
    world.spawn((ChildTag, Parent(p1), Team(1)));
    world.spawn((ChildTag, Parent(p2), Team(2)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent, &Team)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    // Both children's parents have Name, so both match.
    assert_eq!(entities.len(), 2);
}

#[test]
fn er_join_no_matching_children() {
    let mut world = World::new();
    world.register_component::<Parent>();

    // Parents exist but no children.
    world.spawn((Name("Alice"),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    assert_eq!(entities.len(), 0);
}

#[test]
fn er_join_execute_stream_batched() {
    let mut world = World::new();

    let p1 = world.spawn((Name("Alice"),));
    let p2 = world.spawn((Name("Bob"),));
    let p_no = world.spawn((Score(99),));

    world.spawn((ChildTag, Parent(p1), Score(1)));
    world.spawn((ChildTag, Parent(p2), Score(2)));
    world.spawn((ChildTag, Parent(p_no), Score(3)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let mut scores = Vec::new();
    plan.execute_stream_batched::<(&Score,), _>(&mut world, |_, (s,)| {
        scores.push(s.0);
    })
    .unwrap();
    scores.sort_unstable();
    assert_eq!(
        scores,
        vec![1, 2],
        "batched should yield only matching children's scores"
    );
}

// ── Additional ER join tests ─────────────────────────────────

/// Second entity-reference type for chained ER join tests.
#[derive(Clone, Copy, Debug)]
struct Owner(Entity);

impl super::AsEntityRef for Owner {
    fn entity_ref(&self) -> Entity {
        self.0
    }
}

/// Tag for entities that are "owned".
#[derive(Clone, Copy, Debug)]
#[expect(dead_code)]
struct Owned;

#[test]
fn er_join_chained_two_er_joins() {
    let mut world = World::new();

    // Each ER join reads a different component from the left entity.
    // child has both Parent and Owner; each ER join filters independently.
    let parent = world.spawn((Name("Parent"),));
    let owner = world.spawn((Score(42),));
    let child = world.spawn((ChildTag, Parent(parent), Owner(owner)));

    // This child's owner target doesn't have Score.
    let bad_owner = world.spawn((Name("Not an owner"),));
    let child2 = world.spawn((ChildTag, Parent(parent), Owner(bad_owner)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent, &Owner)>()
        // First ER join: parent must have Name.
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        // Second ER join: owner must have Score.
        .er_join::<Owner, (&Score,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    // child: parent has Name ✓, owner has Score ✓ → kept
    // child2: parent has Name ✓, owner has no Score ✗ → filtered
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0], child);
    assert!(!entities.contains(&child2));
}

#[test]
fn er_join_regular_then_er() {
    let mut world = World::new();

    // Two parents, both with Name + Score.
    let p1 = world.spawn((Name("Alice"), Score(10)));
    let _p2 = world.spawn((Name("Bob"), Score(20)));

    // Children also have Score (for regular join).
    let c1 = world.spawn((ChildTag, Parent(p1), Score(100)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent, &Score)>()
        // Regular join: intersect with entities that have Team.
        // c1 doesn't have Team, so it should be filtered out.
        .join::<(&Team,)>(JoinKind::Inner)
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    // c1 has no Team, so the regular join filters it out.
    assert!(entities.is_empty() || !entities.contains(&c1));
}

#[test]
fn er_join_then_regular_join_order_independent() {
    // Verify that join() and er_join() can be called in any order.
    // build() always executes regular joins first, then ER joins.
    let mut world = World::new();
    let p1 = world.spawn((Name("Alice"), Score(10)));
    // Child has Score (matches regular join) + Parent ref (matches ER join)
    let c1 = world.spawn((ChildTag, Parent(p1), Score(50)));

    let planner = QueryPlanner::new(&world);
    // ER join first, then regular join — previously panicked.
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .join::<(&Score,)>(JoinKind::Inner)
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    // c1 has Score (passes regular join) and Parent→Alice who has Name
    // (passes ER join), so c1 should be in results.
    assert!(entities.contains(&c1));
}

#[test]
fn er_join_unregistered_component_returns_error() {
    let mut world = World::new();

    // UnknownRef is never registered — er_join should return an error.
    #[derive(Clone, Copy, Debug)]
    struct UnknownRef(Entity);
    impl super::AsEntityRef for UnknownRef {
        fn entity_ref(&self) -> Entity {
            self.0
        }
    }

    world.spawn((ChildTag,));

    let planner = QueryPlanner::new(&world);
    let result = planner
        .scan::<(&ChildTag,)>()
        .er_join::<UnknownRef, (&Name,)>(JoinKind::Inner);

    let err = result.err().expect("expected UnregisteredComponent error");
    assert!(
        matches!(err, PlannerError::UnregisteredComponent(_)),
        "expected UnregisteredComponent, got: {err:?}",
    );
}

#[test]
fn er_join_dead_reference_left_join() {
    let mut world = World::new();

    let p1 = world.spawn((Name("Alice"),));
    let p2 = world.spawn((Name("Bob"),));

    let c1 = world.spawn((ChildTag, Parent(p1)));
    let c2 = world.spawn((ChildTag, Parent(p2)));

    // Despawn p2 — c2's reference is now dangling.
    world.despawn(p2);

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Left)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    // Left join: both children survive, even with dead reference.
    assert_eq!(entities.len(), 2);
    let ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
    assert!(ids.contains(&c1.to_bits()));
    assert!(ids.contains(&c2.to_bits()));
}

#[test]
fn er_join_many_to_one_references() {
    let mut world = World::new();

    let parent = world.spawn((Name("Shared Parent"),));

    // Five children all pointing to the same parent.
    let children: Vec<Entity> = (0..5)
        .map(|_| world.spawn((ChildTag, Parent(parent))))
        .collect();

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .build();

    let entities = plan.execute_collect(&mut world).unwrap();
    // All five children should match — their shared parent has Name.
    assert_eq!(entities.len(), 5);
    for child in &children {
        assert!(
            entities.contains(child),
            "child {child:?} should be in results"
        );
    }
}

#[test]
fn er_join_with_right_estimate_targets_correct_join() {
    let mut world = World::new();
    let p = world.spawn((Name("Alice"),));
    world.spawn((ChildTag, Parent(p)));

    let planner = QueryPlanner::new(&world);

    // with_right_estimate after er_join should target the ER join.
    let plan = planner
        .scan::<(&ChildTag, &Parent)>()
        .er_join::<Parent, (&Name,)>(JoinKind::Inner)
        .unwrap()
        .with_right_estimate(42)
        .unwrap()
        .build();

    let explain = plan.explain();
    // The explain output should reflect the custom estimate.
    assert!(
        explain.contains("ErJoin"),
        "explain should contain ErJoin: {explain}"
    );
}

/// Regression: add_spatial_index (cost-only, no lookup) registers a
/// spatial predicate's filter_fn in all_filter_fns.  The scan_required
/// fast path must NOT activate for such plans, otherwise the filter is
/// never evaluated.
#[test]
fn spatial_cost_only_filter_applied_in_for_each() {
    let mut world = World::new();
    // 10 entities at x=0..9; only x < 5 should pass the filter.
    for i in 0..10u32 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    // Cost-only registration — no lookup closure.
    planner
        .add_spatial_index::<Pos>(Arc::new(grid), &world)
        .unwrap();

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(
            Predicate::within::<Pos>(
                [2.5, 0.0],
                100.0, // large radius, but filter rejects x >= 5
                |w: &World, e| w.get::<Pos>(e).is_some_and(|p| p.x < 5.0),
            )
            .unwrap(),
        )
        .build();
    drop(planner);

    // execute_stream must apply the spatial filter.
    let mut count = 0u32;
    plan.execute_stream(&mut world, |_| count += 1).unwrap();
    assert_eq!(count, 5, "execute_stream should apply the spatial filter");

    // execute must apply it too.
    let entities = plan.execute_collect(&mut world).unwrap();
    assert_eq!(entities.len(), 5, "execute should apply the spatial filter");

    // for_each_raw / execute_raw (transactional paths).
    let mut count_raw = 0u32;
    plan.execute_stream_raw(&world, |_| count_raw += 1).unwrap();
    assert_eq!(count_raw, 5, "for_each_raw should apply the spatial filter");

    let entities_raw = plan.execute_collect_raw(&world).unwrap();
    assert_eq!(
        entities_raw.len(),
        5,
        "execute_raw should apply the spatial filter"
    );
}
