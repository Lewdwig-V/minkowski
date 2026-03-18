//! Materialized query views — cached, debounced snapshots of subscription
//! query results for real-time computed data.
//!
//! Run: cargo run -p minkowski-examples --example materialized_view --release
//!
//! Demonstrates:
//! - Building an index-backed subscription plan
//! - Wrapping it in a MaterializedView for automatic caching
//! - Debounce policies (Immediate vs EveryNTicks)
//! - Change detection integration (Changed<T> filters stale reads)
//! - Invalidation for forced refresh
//! - Using the view as a real-time computed data source

use std::num::NonZeroU64;
use std::sync::Arc;

use minkowski::{
    BTreeIndex, Changed, DebouncePolicy, HashIndex, Indexed, MaterializedView, Predicate,
    QueryPlanner, RefreshOutcome, SpatialIndex, World,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Team(u32);

fn main() {
    let mut world = World::new();

    // Spawn 200 entities across two teams.
    for i in 0u32..200 {
        world.spawn((Score(i), Team(i % 4)));
    }
    println!("Spawned {} entities\n", world.entity_count());

    // ── 1. Simple scan view (no index needed) ───────────────────────────

    println!("=== 1. Simple Scan View ===\n");
    {
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let mut view = MaterializedView::new(plan);

        view.refresh(&mut world).unwrap();
        println!(
            "Full scan view: {} entities, refreshed {} time(s)",
            view.len(),
            view.refresh_count()
        );
        assert_eq!(view.len(), 200);
    }

    // ── 2. Index-backed subscription view ───────────────────────────────

    println!("\n=== 2. Index-Backed Subscription View ===\n");
    {
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let score_idx = Arc::new(score_idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&score_idx, &world).unwrap();
        let witness = Indexed::btree(&*score_idx);

        let plan = planner
            .subscribe::<(Changed<Score>, &Score)>()
            .where_range(witness, Predicate::range::<Score, _>(Score(50)..Score(100)))
            .build()
            .unwrap();

        println!("Plan:\n{}", plan.explain());

        let mut view = MaterializedView::new(plan);
        view.refresh(&mut world).unwrap();
        println!("Subscription view [50..100): {} entities", view.len());
        assert_eq!(view.len(), 50);

        // Verify all entities have Score in [50, 100).
        for &e in view.entities() {
            let s = world.get::<Score>(e).unwrap().0;
            assert!((50..100).contains(&s), "unexpected Score({s})");
        }
    }

    // ── 3. Change detection ─────────────────────────────────────────────

    println!("\n=== 3. Change Detection (Changed<T>) ===\n");
    {
        let mut world2 = World::new();
        let entities: Vec<_> = (0u32..10).map(|i| world2.spawn((Score(i),))).collect();

        let planner = QueryPlanner::new(&world2);
        let plan = planner.scan::<(Changed<Score>, &Score)>().build();
        let mut view = MaterializedView::new(plan);

        // First refresh: all entities are new.
        view.refresh(&mut world2).unwrap();
        println!("First refresh:  {} entities (all new)", view.len());
        assert_eq!(view.len(), 10);

        // No mutations → Changed<Score> filters everything.
        view.refresh(&mut world2).unwrap();
        println!("No mutations:   {} entities", view.len());
        assert_eq!(view.len(), 0);

        // Mutate one entity.
        let _ = world2.get_mut::<Score>(entities[3]);
        view.refresh(&mut world2).unwrap();
        println!(
            "After mutation: {} entities (archetype-granular)",
            view.len()
        );
        // Archetype-granular: all 10 in the same archetype are returned.
        assert_eq!(view.len(), 10);
    }

    // ── 4. Debounce policy ──────────────────────────────────────────────

    println!("\n=== 4. Debounce Policy (EveryNTicks) ===\n");
    {
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let mut view = MaterializedView::new(plan)
            .with_debounce(DebouncePolicy::EveryNTicks(NonZeroU64::new(5).unwrap()));

        // First call always refreshes.
        let outcome = view.refresh(&mut world).unwrap();
        println!("Call 1: outcome={outcome:?}, len={}", view.len());
        assert_eq!(outcome, RefreshOutcome::Refreshed);

        // Calls 2..5 are suppressed (ticks_since_refresh: 1, 2, 3, 4).
        for call in 2..=5 {
            let outcome = view.refresh(&mut world).unwrap();
            println!(
                "Call {call}: outcome={outcome:?}, len={} (cached)",
                view.len()
            );
            assert_eq!(outcome, RefreshOutcome::Suppressed);
        }

        // Call 6 triggers refresh (ticks_since_refresh reaches 5).
        let outcome = view.refresh(&mut world).unwrap();
        println!("Call 6: outcome={outcome:?}, len={}", view.len());
        assert_eq!(outcome, RefreshOutcome::Refreshed);
        assert_eq!(view.refresh_count(), 2);
    }

    // ── 5. Invalidation ─────────────────────────────────────────────────

    println!("\n=== 5. Invalidation ===\n");
    {
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let mut view = MaterializedView::new(plan)
            .with_debounce(DebouncePolicy::EveryNTicks(NonZeroU64::new(100).unwrap()));

        view.refresh(&mut world).unwrap();
        println!(
            "Initial: {} entities, refresh_count={}",
            view.len(),
            view.refresh_count()
        );

        // Would normally be suppressed for 99 more calls.
        let outcome = view.refresh(&mut world).unwrap();
        println!("After 1 call: outcome={outcome:?}");
        assert_eq!(outcome, RefreshOutcome::Suppressed);

        // Force refresh via invalidate.
        view.invalidate();
        let outcome = view.refresh(&mut world).unwrap();
        println!(
            "After invalidate: outcome={outcome:?}, refresh_count={}",
            view.refresh_count()
        );
        assert_eq!(outcome, RefreshOutcome::Refreshed);
        assert_eq!(view.refresh_count(), 2);
    }

    // ── 6. Multi-index subscription view ────────────────────────────────

    println!("\n=== 6. Multi-Index Subscription View ===\n");
    {
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let score_idx = Arc::new(score_idx);

        let mut team_idx = HashIndex::<Team>::new();
        team_idx.rebuild(&mut world);
        let team_idx = Arc::new(team_idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&score_idx, &world).unwrap();
        planner.add_hash_index(&team_idx, &world).unwrap();

        let score_w = Indexed::btree(&*score_idx);
        let team_w = Indexed::hash(&*team_idx);

        let plan = planner
            .subscribe::<(Changed<Score>, &Score, &Team)>()
            .where_range(
                score_w,
                Predicate::range::<Score, _>(Score(0)..Score(50)).with_selectivity(0.25),
            )
            .where_eq(team_w, Predicate::eq(Team(0)).with_selectivity(0.25))
            .build()
            .unwrap();

        let mut view = MaterializedView::new(plan)
            .with_debounce(DebouncePolicy::EveryNTicks(NonZeroU64::new(3).unwrap()));

        view.refresh(&mut world).unwrap();
        println!(
            "Multi-index view (Score<50, Team=0): {} entities",
            view.len()
        );

        // Score 0..50 with Team==0 means scores 0, 4, 8, 12, ..., 48 → 13 entities.
        for &e in view.entities() {
            let s = world.get::<Score>(e).unwrap().0;
            let t = world.get::<Team>(e).unwrap().0;
            assert!(s < 50, "Score {s} out of range");
            assert_eq!(t, 0, "Team {t} != 0");
        }
        println!("All entities verified: Score < 50 and Team == 0");
    }

    // ── 7. Dynamic policy switching ─────────────────────────────────────

    println!("\n=== 7. Dynamic Policy Switching ===\n");
    {
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let mut view = MaterializedView::new(plan)
            .with_debounce(DebouncePolicy::EveryNTicks(NonZeroU64::new(10).unwrap()));

        view.refresh(&mut world).unwrap();
        println!("EveryNTicks(10): refresh_count={}", view.refresh_count());

        // Suppressed.
        view.refresh(&mut world).unwrap();
        println!("Suppressed call: refresh_count={}", view.refresh_count());

        // Switch to immediate.
        view.set_policy(DebouncePolicy::Immediate);
        view.refresh(&mut world).unwrap();
        println!(
            "Switched to Immediate: refresh_count={}",
            view.refresh_count()
        );
        assert_eq!(view.refresh_count(), 2);
    }

    println!("\nDone.");
}
