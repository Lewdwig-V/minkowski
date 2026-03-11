//! Memory pool: TigerBeetle-style pre-allocated World with bounded memory.
//!
//! Demonstrates `WorldBuilder` with a fixed memory budget. Spawns entities
//! until the pool is exhausted, showing graceful error handling via
//! `try_spawn`. Pool stats show capacity and usage throughout.
//!
//! Run: cargo run -p minkowski-examples --example pool --release

use minkowski::{Entity, HugePages, PoolExhausted, World};

fn main() {
    println!("--- Allocating World with 16 MB pool ---");
    let mut world = World::builder()
        .memory_budget(16 * 1024 * 1024)
        .hugepages(HugePages::Off) // Off for portability in examples
        .build()
        .expect("failed to allocate memory pool");

    let stats = world.stats();
    println!(
        "Pool: {:.1} MB capacity, {:.1} KB used",
        stats.pool_capacity.unwrap() as f64 / 1_048_576.0,
        stats.pool_used.unwrap() as f64 / 1024.0,
    );

    // Spawn entities until pool is exhausted
    println!("\n--- Spawning entities ---");
    let mut count = 0u32;
    loop {
        match world.try_spawn((count, count as f64)) {
            Ok(_) => count += 1,
            Err(PoolExhausted { .. }) => {
                println!("Pool exhausted after {count} entities");
                break;
            }
        }
        if count.is_multiple_of(10_000) {
            let s = world.stats();
            println!(
                "  {count} entities, {:.1} KB used / {:.1} MB total",
                s.pool_used.unwrap() as f64 / 1024.0,
                s.pool_capacity.unwrap() as f64 / 1_048_576.0,
            );
        }
    }

    let stats = world.stats();
    println!(
        "\nFinal: {} entities, {:.1} MB used / {:.1} MB total",
        stats.entity_count,
        stats.pool_used.unwrap() as f64 / 1_048_576.0,
        stats.pool_capacity.unwrap() as f64 / 1_048_576.0,
    );

    // Despawn half the entities to show pool memory recovery
    println!("\n--- Despawning half the entities ---");
    let half = count / 2;
    let to_despawn: Vec<Entity> = world
        .query::<(Entity, &u32)>()
        .filter_map(|(e, val)| if *val < half { Some(e) } else { None })
        .collect();
    for e in to_despawn {
        world.despawn(e);
    }

    let stats = world.stats();
    println!(
        "After despawn: {} entities, {:.1} MB used / {:.1} MB total",
        stats.entity_count,
        stats.pool_used.unwrap() as f64 / 1_048_576.0,
        stats.pool_capacity.unwrap() as f64 / 1_048_576.0,
    );

    // Show that the default builder (no budget) reports no pool stats
    println!("\n--- Default World (system allocator, no pool) ---");
    let default_world = World::new();
    let ds = default_world.stats();
    println!(
        "pool_capacity: {:?}, pool_used: {:?}",
        ds.pool_capacity, ds.pool_used
    );

    println!("\nDone.");
}
