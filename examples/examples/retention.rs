//! Retention: tick-based entity cleanup via Expiry + RetentionReducer.
//!
//! Spawns entities with varying TTLs. Each "frame" advances the world tick
//! by running dummy mutations, then runs the retention reducer to despawn
//! expired entities.
//!
//! Run: cargo run -p minkowski-examples --example retention --release

use minkowski::{Expiry, ReducerRegistry, World};

fn main() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    // Spawn entities with different TTLs (in ticks from now).
    // Each mutable operation (spawn, query, despawn) advances the internal
    // tick by 1, so 100 dummy queries per frame ≈ 100 ticks per frame.
    let ttls: &[u64] = &[10, 50, 100, 200, 500];
    let now = world.change_tick();

    println!(
        "--- Spawning {} entities with TTLs: {ttls:?} ---",
        ttls.len()
    );
    println!("    base tick: {}", now.to_raw());
    println!();

    for &ttl in ttls {
        world.spawn((Expiry::with_ttl(now, ttl), ttl as u32));
    }

    // Simulate frames — each iteration advances the tick via dummy queries.
    for frame in 0..6 {
        // Advance tick by doing work (each query call advances tick by 1).
        for _ in 0..100 {
            world.query::<(&u32,)>().for_each(|_| {});
        }

        let before = world.entity_count();
        registry.run(&mut world, retention_id, ()).unwrap();
        let after = world.entity_count();
        let tick = world.change_tick().to_raw();

        let despawned = before - after;
        println!(
            "Frame {frame}: tick={tick}, entities {before} -> {after} ({despawned} despawned)"
        );
    }

    println!();
    println!("--- Final entity count: {} ---", world.entity_count());
}
