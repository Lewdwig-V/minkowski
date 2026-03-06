//! Column indexes — B-tree range queries and hash exact lookups.
//!
//! Run: cargo run -p minkowski-examples --example index --release
//!
//! Demonstrates BTreeIndex and HashIndex on a Score component:
//! - Build indexes from world state
//! - Range query (scores in a range)
//! - Exact lookup
//! - Incremental update after mutations via Changed<T>
//! - Stale entry detection after despawn

use minkowski::{BTreeIndex, HashIndex, SpatialIndex, World};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy)]
#[allow(dead_code)]
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

    // Incremental update — only changed archetypes rescanned.
    //
    // Note: Changed<T> is per-query-type, so BTreeIndex and HashIndex share
    // the same query signature internally. Calling update on both would cause
    // the second to miss the changes (the first already advanced the tick).
    // In practice you'd pick one index type per component, or use rebuild for
    // the second. Here we demonstrate update on btree, then rebuild on hash.
    btree.update(&mut world);
    hash.rebuild(&mut world);

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

    // Rebuild cleans up
    btree.rebuild(&mut world);
    assert!(btree.get(&Score(victim_score)).is_empty());
    println!("  After rebuild: stale entry cleaned up");
    println!();

    println!("Done.");
}
