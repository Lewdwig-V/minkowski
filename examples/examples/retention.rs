//! Retention: dispatch-count entity cleanup via Expiry + RetentionReducer.
//!
//! Spawns entities with varying lifetimes. Each frame runs the retention
//! reducer once, decrementing Expiry counters and despawning entities
//! that reach zero.
//!
//! Run: cargo run -p minkowski-examples --example retention --release

use minkowski::{Expiry, ReducerRegistry, World};

fn main() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    // Spawn entities that expire after different numbers of retention dispatches.
    // Expiry::after(n) means "survive n dispatches, despawn on the nth."
    let lifetimes: &[u32] = &[1, 2, 3, 5, 8];

    println!(
        "--- Spawning {} entities with lifetimes: {lifetimes:?} dispatches ---",
        lifetimes.len()
    );
    println!();

    for &n in lifetimes {
        world.spawn((Expiry::after(n), n));
    }

    // Each loop iteration is one "frame" — run retention once per frame.
    for frame in 1..=10 {
        let before = world.entity_count();
        registry.run(&mut world, retention_id, ()).unwrap();
        let after = world.entity_count();
        let despawned = before - after;

        if despawned > 0 {
            println!("Frame {frame}: {before} -> {after} entities ({despawned} despawned)");
        } else {
            println!("Frame {frame}: {after} entities");
        }

        if after == 0 {
            break;
        }
    }

    println!();
    println!("--- All entities expired ---");
}
