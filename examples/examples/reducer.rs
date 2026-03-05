//! Reducer system demo: typed handles enforce declared access at compile time,
//! dynamic reducers provide runtime-flexible access with builder-declared bounds.
//!
//! Demonstrates:
//! 1. Entity reducer (heal) — reads & writes Health via EntityMut
//! 2. Query reducer (gravity) — mutates Velocity via QueryMut
//! 3. Read-only query reducer (logger) — reads via QueryRef
//! 4. Spawner reducer — creates new entities via Spawner
//! 5. Query writer reducer (drag) — buffered bulk updates via QueryWriter
//! 6. Name-based lookup
//! 7. Access conflict detection between registered reducers
//! 8. Dynamic reducer — conditional access based on runtime state

use minkowski::{
    DynamicCtx, Entity, EntityMut, Optimistic, QueryMut, QueryRef, QueryWriter, ReducerRegistry,
    Spawner, World,
};

#[derive(Clone, Copy, Debug)]
struct Health(u32);

#[derive(Clone, Copy, Debug)]
struct Velocity(f32);

#[derive(Clone, Copy, Debug)]
struct Energy(f32);

#[derive(Clone, Copy, Debug)]
struct Shield(f32);

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
        .call(&strategy, &mut world, heal_id, (hero, 25u32))
        .unwrap();
    println!(
        "After heal: hero hp={}",
        world.get::<Health>(hero).unwrap().0
    );

    // Damage enemy
    registry
        .call(&strategy, &mut world, damage_id, (enemy, 30u32))
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
        .call(&strategy, &mut world, spawn_id, 75u32)
        .unwrap();

    // ── 6. Query writer reducer: persistent bulk update ────────────
    let drag_id = registry.register_query_writer::<(&mut Velocity,), f32, _>(
        &mut world,
        "apply_drag",
        |mut query: QueryWriter<'_, (&mut Velocity,)>, drag: f32| {
            query.for_each(|(mut vel,)| {
                vel.modify(|v| v.0 *= drag);
            });
        },
    );
    println!(
        "Registered 'apply_drag' query writer reducer (id={:?})",
        drag_id
    );

    // Dispatch — goes through strategy.transact(), compatible with Durable
    registry
        .call(&strategy, &mut world, drag_id, 0.9f32)
        .unwrap();
    println!(
        "After drag: hero vel={:.2}",
        world.get::<Velocity>(hero).unwrap().0
    );

    // ── 7. Name-based lookup ─────────────────────────────────────────
    println!("\n--- Name-based lookup ---");
    let found_heal = registry.reducer_id_by_name("heal").unwrap();
    println!("'heal' -> {:?}", found_heal);
    let found_gravity = registry.query_reducer_id_by_name("gravity").unwrap();
    println!("'gravity' -> {:?}", found_gravity);
    let found_drag = registry.reducer_id_by_name("apply_drag").unwrap();
    println!("'apply_drag' -> {:?}", found_drag);

    // ── 8. Dynamic reducer — conditional access ──────────────────
    println!("\n--- Dynamic reducers ---");

    // Give hero and enemy Energy + Shield components for dynamic reducer demo
    world.insert(hero, Energy(80.0));
    world.insert(hero, Shield(0.0));
    world.insert(enemy, Energy(30.0));
    world.insert(enemy, Shield(0.0));

    // Register a dynamic reducer that conditionally applies a shield
    // based on HP and Energy — can't express this with static types
    // because the Shield write only happens when HP < 50.
    let shield_id = registry
        .dynamic("conditional_shield", &mut world)
        .can_read::<Health>()
        .can_read::<Energy>()
        .can_write::<Energy>()
        .can_write::<Shield>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            let hp = ctx.read::<Health>(*entity).0;
            let energy = ctx.read::<Energy>(*entity).0;
            if energy >= 50.0 {
                ctx.write(*entity, Energy(energy - 50.0));
                if hp < 50 {
                    // Conditional write — only when HP is low
                    ctx.write(*entity, Shield(100.0));
                    println!(
                        "  [shield] entity {:?}: low HP ({}) — shield activated!",
                        entity, hp
                    );
                } else {
                    println!(
                        "  [shield] entity {:?}: HP fine ({}) — energy spent, no shield",
                        entity, hp
                    );
                }
            } else {
                println!(
                    "  [shield] entity {:?}: not enough energy ({:.0})",
                    entity, energy
                );
            }
        });
    println!(
        "Registered 'conditional_shield' dynamic reducer (id={:?})",
        shield_id
    );

    // Dispatch via Optimistic strategy (dynamic reducers buffer writes,
    // so they need a real transactional strategy — not Sequential)
    registry
        .dynamic_call(&strategy, &mut world, shield_id, &hero)
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, shield_id, &enemy)
        .unwrap();

    println!(
        "After shields: hero energy={:.0} shield={:.0}, enemy energy={:.0} shield={:.0}",
        world.get::<Energy>(hero).unwrap().0,
        world.get::<Shield>(hero).unwrap().0,
        world.get::<Energy>(enemy).unwrap().0,
        world.get::<Shield>(enemy).unwrap().0,
    );

    // Name-based lookup works for dynamic reducers too
    let found_shield = registry.dynamic_id_by_name("conditional_shield");
    println!("Dynamic lookup 'conditional_shield' -> {:?}", found_shield);

    // ── 9. Access conflict detection (static vs dynamic) ───────────
    println!("\n--- Access conflict detection ---");
    let heal_access = registry.reducer_access(heal_id);
    let damage_access = registry.reducer_access(damage_id);
    let gravity_access = registry.query_reducer_access(gravity_id);
    let drag_access = registry.reducer_access(drag_id);
    let shield_access = registry.dynamic_access(shield_id);

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
    println!(
        "gravity vs apply_drag: {}",
        if gravity_access.conflicts_with(drag_access) {
            "CONFLICT (both write Velocity)"
        } else {
            "compatible"
        }
    );
    println!(
        "heal vs apply_drag: {}",
        if heal_access.conflicts_with(drag_access) {
            "CONFLICT"
        } else {
            "compatible (disjoint components)"
        }
    );
    println!(
        "heal vs conditional_shield: {}",
        if heal_access.conflicts_with(shield_access) {
            "CONFLICT (both touch Health)"
        } else {
            "compatible"
        }
    );
    println!(
        "gravity vs conditional_shield: {}",
        if gravity_access.conflicts_with(shield_access) {
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
