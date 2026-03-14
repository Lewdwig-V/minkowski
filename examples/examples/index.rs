//! Column indexes — B-tree range queries, hash exact lookups, and planned queries.
//!
//! Run: cargo run -p minkowski-examples --example index --release
//!
//! Demonstrates BTreeIndex and HashIndex on a Score component:
//! - Build indexes from world state
//! - Range query (scores in a range)
//! - Exact lookup
//! - Incremental update after mutations via per-index ChangeTick
//! - Stale entry detection after despawn
//! - Planned queries via QueryPlanner with IndexDriver execution

use std::sync::Arc;

use minkowski::{BTreeIndex, HashIndex, Predicate, QueryPlanner, SpatialIndex, World};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy)]
#[expect(dead_code)]
struct Name(&'static str);

fn main() {
    let mut world = World::new();

    // Spawn entities with scores across different archetypes
    let mut entities = Vec::new();
    for i in 0..200 {
        let e = if i % 2 == 0 {
            world.spawn((Score(i),))
        } else {
            // Different archetype — indexes must span both
            world.spawn((Score(i), Name("player")))
        };
        entities.push(e);
    }

    println!("Spawned {} entities with Score 0..200", entities.len());
    println!();

    // -- Build indexes --
    let mut btree = BTreeIndex::<Score>::new();
    let mut hash = HashIndex::<Score>::new();
    btree.rebuild(&mut world);
    hash.rebuild(&mut world);

    // -- Range query (B-tree) --
    let range_results: Vec<_> = btree
        .range(Score(50)..Score(55))
        .flat_map(|(_, entities)| entities.iter().copied())
        .collect();
    println!("BTree range [50..55): {} entities", range_results.len());
    for e in &range_results {
        let score = world.get::<Score>(*e).unwrap();
        println!("  entity {:?} -> score {}", e, score.0);
    }
    println!();

    // -- Exact lookup (both) --
    let btree_42 = btree.get(&Score(42));
    let hash_42 = hash.get(&Score(42));
    println!("BTree exact Score(42): {} entities", btree_42.len());
    println!("Hash  exact Score(42): {} entities", hash_42.len());
    assert_eq!(btree_42.len(), hash_42.len());
    println!();

    // -- Mutate some scores --
    println!("Mutating scores 0..10 to 1000..1010...");
    for (i, &e) in entities.iter().enumerate().take(10) {
        *world.get_mut::<Score>(e).unwrap() = Score(1000 + i as u32);
    }

    // Incremental update — each index tracks its own ChangeTick, so both
    // see the same changes independently.
    btree.update(&mut world);
    hash.update(&mut world);

    let high_scores: Vec<_> = btree
        .range(Score(1000)..)
        .flat_map(|(_, entities)| entities.iter().copied())
        .collect();
    println!(
        "BTree range [1000..): {} entities (expected 10)",
        high_scores.len()
    );
    assert_eq!(high_scores.len(), 10);

    // Old values should be gone
    assert!(btree.get(&Score(0)).is_empty());
    assert!(hash.get(&Score(0)).is_empty());
    println!("Old value Score(0) cleared from both indexes.");
    println!();

    // -- Despawn and stale detection --
    let victim = entities[50];
    let victim_score = world.get::<Score>(victim).unwrap().0;
    println!("Despawning entity with Score({})...", victim_score);
    world.despawn(victim);

    // Stale entry still in index
    let stale = btree.get(&Score(victim_score));
    assert_eq!(stale.len(), 1);
    assert!(!world.is_alive(stale[0]));
    println!("  Stale entry present, is_alive = false");

    // range_valid / get_valid: filter stale entries without a full rebuild
    let valid_range: Vec<_> = btree
        .range_valid(Score(victim_score)..=Score(victim_score), &world)
        .collect();
    assert!(valid_range.is_empty());
    println!(
        "  range_valid filters stale entry (found {})",
        valid_range.len()
    );

    let valid_hash: Vec<_> = hash.get_valid(&Score(victim_score), &world).collect();
    assert!(valid_hash.is_empty());
    println!(
        "  hash get_valid filters stale entry (found {})",
        valid_hash.len()
    );

    // Rebuild cleans up permanently
    btree.rebuild(&mut world);
    hash.rebuild(&mut world);
    assert!(btree.get(&Score(victim_score)).is_empty());
    println!("  After rebuild: stale entry cleaned up");
    println!();

    // -- Batch fetch: index -> get_batch composition --
    // Rebuild btree (it was rebuilt after despawn above)
    let low_scores: Vec<_> = btree
        .range(Score(10)..Score(20))
        .flat_map(|(_, entities)| entities.iter().copied())
        .collect();

    println!(
        "Batch fetch: {} entities with Score in [10..20)",
        low_scores.len()
    );

    // Batch read — one call instead of N individual get() calls.
    // Groups by archetype internally for cache locality.
    let names = world.get_batch::<Name>(&low_scores);
    let mut named = 0;
    let mut unnamed = 0;
    for name in &names {
        match name {
            Some(_) => named += 1,
            None => unnamed += 1,
        }
    }
    println!(
        "  {} with Name component, {} without (different archetypes)",
        named, unnamed
    );

    // Batch mutable access — boost scores by 100 in one call
    let mut scores_mut = world.get_batch_mut::<Score>(&low_scores);
    for score in scores_mut.iter_mut().flatten() {
        score.0 += 100;
    }
    drop(scores_mut);
    let boosted = world.get::<Score>(low_scores[0]).unwrap().0;
    println!(
        "  After get_batch_mut boost: first entity score = {}",
        boosted
    );
    println!();

    // -- Planned queries via QueryPlanner (the recommended path) --
    //
    // Sync indexes after the batch mutation above so the planner sees
    // current state. Without this, the btree/hash still map old keys.
    btree.update(&mut world);
    hash.update(&mut world);

    // Wrap indexes in Arc so the planner can hold live references to them.
    // The planner calls the lookup closure at execution time — it reads the
    // live index, not a snapshot captured at registration.
    let btree = Arc::new(btree);
    let hash = Arc::new(hash);

    // Range query via BTree — build plan, drop planner, then execute.
    // The planner borrows &world, so it must be released before for_each
    // takes &mut world.
    let mut range_plan = {
        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index::<Score>(&btree, &world).unwrap();
        planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(150)..Score(160)))
            .build()
    };

    println!(
        "Planned range [150..160) explain:\n{}",
        range_plan.explain()
    );
    let mut planned_range_count = 0;
    range_plan
        .for_each(&mut world, |_| planned_range_count += 1)
        .unwrap();
    println!("Planned range [150..160): {planned_range_count} entities (expected 10)");
    assert_eq!(planned_range_count, 10);
    println!();

    // Exact lookup via Hash — separate planner so the Hash index is
    // actually selected (BTree would shadow it if both were registered
    // for the same component, since add_hash_index uses or_insert).
    let mut eq_plan = {
        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index::<Score>(&hash, &world).unwrap();
        planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build()
    };

    println!("Planned eq explain:\n{}", eq_plan.explain());
    let mut planned_eq_count = 0;
    eq_plan
        .for_each(&mut world, |_| planned_eq_count += 1)
        .unwrap();
    println!("Planned eq Score(42): {planned_eq_count} entities (expected 1)");
    assert_eq!(planned_eq_count, 1);

    println!();
    println!("Done.");
}
