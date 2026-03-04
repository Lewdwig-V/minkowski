//! Reducer system demo: typed handles enforce declared access at compile time.
//!
//! Demonstrates:
//! 1. Entity reducer (heal) — reads & writes Health via EntityMut
//! 2. Query reducer (gravity) — mutates Velocity via QueryMut
//! 3. Read-only query reducer (logger) — reads via QueryRef
//! 4. Spawner reducer — creates new entities via Spawner
//! 5. Name-based lookup
//! 6. Access conflict detection between registered reducers

use minkowski::{
    Entity, EntityMut, Optimistic, QueryMut, QueryRef, ReducerRegistry, Spawner, World,
};

#[derive(Clone, Copy, Debug)]
struct Health(u32);

#[derive(Clone, Copy, Debug)]
struct Velocity(f32);

fn main() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    // ── 1. Entity reducer: heal ──────────────────────────────────────
    let heal_id = registry.register_entity::<(Health,), u32, _>(
        &mut world,
        "heal",
        |mut entity: EntityMut<'_, (Health,)>, amount: u32| {
            let hp = entity.get::<Health, 0>().0;
            entity.set::<Health, 0>(Health(hp + amount));
        },
    );
    println!("Registered 'heal' reducer (id={:?})", heal_id);

    // ── 2. Entity reducer: damage ────────────────────────────────────
    let damage_id = registry.register_entity::<(Health,), u32, _>(
        &mut world,
        "damage",
        |mut entity: EntityMut<'_, (Health,)>, amount: u32| {
            let hp = entity.get::<Health, 0>().0;
            entity.set::<Health, 0>(Health(hp.saturating_sub(amount)));
        },
    );
    println!("Registered 'damage' reducer (id={:?})", damage_id);

    // ── 3. Query reducer: gravity ────────────────────────────────────
    let gravity_id = registry.register_query::<(&mut Velocity,), f32, _>(
        &mut world,
        "gravity",
        |mut query: QueryMut<'_, (&mut Velocity,)>, dt: f32| {
            query.for_each(|(vel,)| {
                vel.0 -= 9.81 * dt;
            });
        },
    );
    println!("Registered 'gravity' query reducer (id={:?})", gravity_id);

    // ── 4. Read-only query reducer: logger ───────────────────────────
    let logger_id = registry.register_query_ref::<(&Health, &Velocity), (), _>(
        &mut world,
        "logger",
        |mut query: QueryRef<'_, (&Health, &Velocity)>, ()| {
            let count = query.count();
            println!("  [logger] {} entities with Health + Velocity", count);
        },
    );
    println!("Registered 'logger' query_ref reducer (id={:?})", logger_id);

    // ── 5. Spawner reducer ───────────────────────────────────────────
    let spawn_id = registry.register_spawner::<(Health, Velocity), u32, _>(
        &mut world,
        "spawn_unit",
        |mut spawner: Spawner<'_, (Health, Velocity)>, hp: u32| {
            let e = spawner.spawn((Health(hp), Velocity(0.0)));
            println!("  [spawner] created entity {:?} with {}hp", e, hp);
        },
    );
    println!(
        "Registered 'spawn_unit' spawner reducer (id={:?})",
        spawn_id
    );

    // ── Setup: create some entities ──────────────────────────────────
    let hero = world.spawn((Health(100), Velocity(0.0)));
    let enemy = world.spawn((Health(50), Velocity(0.0)));
    println!(
        "\nSpawned hero={:?} (100hp), enemy={:?} (50hp)",
        hero, enemy
    );

    // ── Dispatch via Optimistic strategy ─────────────────────────────
    let strategy = Optimistic::new(&world);
    println!("\n--- Dispatching reducers ---");

    // Heal hero
    registry
        .call_entity(&strategy, &mut world, heal_id, hero, 25u32)
        .unwrap();
    println!(
        "After heal: hero hp={}",
        world.get::<Health>(hero).unwrap().0
    );

    // Damage enemy
    registry
        .call_entity(&strategy, &mut world, damage_id, enemy, 30u32)
        .unwrap();
    println!(
        "After damage: enemy hp={}",
        world.get::<Health>(enemy).unwrap().0
    );

    // Apply gravity
    registry.run(&mut world, gravity_id, 0.1f32);
    println!(
        "After gravity: hero vel={:.2}",
        world.get::<Velocity>(hero).unwrap().0
    );

    // Log state
    registry.run(&mut world, logger_id, ());

    // Spawn a new unit via spawner reducer
    registry
        .call_entity(&strategy, &mut world, spawn_id, Entity::DANGLING, 75u32)
        .unwrap();

    // ── 6. Name-based lookup ─────────────────────────────────────────
    println!("\n--- Name-based lookup ---");
    let found_heal = registry.reducer_id_by_name("heal").unwrap();
    println!("'heal' -> {:?}", found_heal);
    let found_gravity = registry.query_reducer_id_by_name("gravity").unwrap();
    println!("'gravity' -> {:?}", found_gravity);

    // ── 7. Access conflict detection ─────────────────────────────────
    println!("\n--- Access conflict detection ---");
    let heal_access = registry.reducer_access(heal_id);
    let damage_access = registry.reducer_access(damage_id);
    let gravity_access = registry.query_reducer_access(gravity_id);

    println!(
        "heal vs damage: {}",
        if heal_access.conflicts_with(damage_access) {
            "CONFLICT (both write Health)"
        } else {
            "compatible"
        }
    );
    println!(
        "heal vs gravity: {}",
        if heal_access.conflicts_with(gravity_access) {
            "CONFLICT"
        } else {
            "compatible (disjoint components)"
        }
    );

    // Final state
    println!("\n--- Final state ---");
    for (health, vel) in world.query::<(&Health, &Velocity)>() {
        println!("  hp={}, vel={:.2}", health.0, vel.0);
    }

    println!("\nDone.");
}
