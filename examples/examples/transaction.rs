//! Transaction strategies — raw Tx building blocks vs reducer-based dispatch.
//!
//! Run: cargo run -p minkowski-examples --example transaction --release
//!
//! Part 1: Raw Tx API — Sequential begin/commit + Optimistic transact closure.
//!         Shows the low-level building blocks that reducers abstract over.
//!
//! Part 2: Reducer API — same logic registered as query reducers, dispatched
//!         via `registry.run()`. Strategy-agnostic, with free conflict detection.
//!
//! Part 3: Join plans inside transactions — demonstrates that query planner
//!         joins work through `for_each_raw`/`execute_raw` with `&World`.
//!         The plan's internal scratch buffer is invisible to the conflict
//!         model — it's computation, not state.

use minkowski::{
    Access, Entity, JoinKind, Optimistic, QueryMut, QueryPlanResult, QueryPlanner, QueryReducerId,
    ReducerRegistry, Sequential, Transact, World,
};

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
struct Vel {
    dx: f32,
    dy: f32,
}

#[derive(Clone, Copy)]
struct Health(u32);

// ── Helpers ─────────────────────────────────────────────────────────

fn spawn_world() -> (World, Vec<Entity>) {
    let mut world = World::new();
    let mut entities = Vec::with_capacity(100);
    for i in 0..100 {
        let e = world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
            Health(100),
        ));
        entities.push(e);
    }
    (world, entities)
}

fn avg_pos(world: &mut World) -> (f32, f32) {
    let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0u32);
    for pos in world.query::<(&Pos,)>() {
        sx += pos.0.x;
        sy += pos.0.y;
        n += 1;
    }
    (sx / n as f32, sy / n as f32)
}

fn avg_health(world: &mut World) -> f32 {
    let (mut total, mut n) = (0u32, 0u32);
    for hp in world.query::<(&Health,)>() {
        total += hp.0.0;
        n += 1;
    }
    total as f32 / n as f32
}

// ════════════════════════════════════════════════════════════════════
// Part 1: Raw Tx API
// ════════════════════════════════════════════════════════════════════

fn run_raw_sequential(world: &mut World) {
    let access = Access::of::<(&mut Pos, &Vel)>(world);
    let strategy = Sequential;

    for _ in 0..10 {
        let tx = strategy.begin(world, &access);
        for (pos, vel) in tx.query::<(&mut Pos, &Vel)>(world) {
            pos.x += vel.dx;
            pos.y += vel.dy;
        }
        let result = tx.commit(world);
        assert!(result.is_ok(), "sequential commit always succeeds");
    }

    let (ax, ay) = avg_pos(world);
    println!("  after 10 steps: avg pos = ({ax:.1}, {ay:.1})");
}

fn run_raw_optimistic(world: &mut World) {
    let access = Access::of::<(&Health, &mut Health)>(world);
    let strategy = Optimistic::new(world);

    for _ in 0..10 {
        strategy
            .transact(world, &access, |tx, world| {
                let healths: Vec<(Entity, u32)> = tx
                    .query::<(Entity, &Health)>(world)
                    .map(|(e, hp)| (e, hp.0))
                    .collect();
                for (e, hp) in healths {
                    tx.write::<Health>(world, e, Health(hp.saturating_sub(3)));
                }
            })
            .expect("no concurrent modification = clean commit");
    }

    let avg = avg_health(world);
    println!("  after 10 steps: avg health = {avg:.0}");
}

// ════════════════════════════════════════════════════════════════════
// Part 2: Reducer API — same logic, strategy-agnostic
// ════════════════════════════════════════════════════════════════════

fn register_reducers(
    registry: &mut ReducerRegistry,
    world: &mut World,
) -> (QueryReducerId, QueryReducerId) {
    let move_id = registry
        .register_query::<(&mut Pos, &Vel), f32, _>(
            world,
            "move_entities",
            |mut query: QueryMut<'_, (&mut Pos, &Vel)>, dt: f32| {
                query.for_each(|(positions, velocities)| {
                    for i in 0..positions.len() {
                        positions[i].x += velocities[i].dx * dt;
                        positions[i].y += velocities[i].dy * dt;
                    }
                });
            },
        )
        .unwrap();

    let decay_id = registry
        .register_query::<(&mut Health,), u32, _>(
            world,
            "decay_health",
            |mut query: QueryMut<'_, (&mut Health,)>, amount: u32| {
                query.for_each(|(healths,)| {
                    for h in healths.iter_mut() {
                        h.0 = h.0.saturating_sub(amount);
                    }
                });
            },
        )
        .unwrap();

    (move_id, decay_id)
}

fn run_reducer_demo(
    registry: &ReducerRegistry,
    world: &mut World,
    move_id: QueryReducerId,
    decay_id: QueryReducerId,
) {
    // Query reducers use registry.run() — direct &mut World access.
    // The same reducer IDs work regardless of which strategy you'd use
    // for transactional reducers; query reducers are always scheduled.
    for _ in 0..10 {
        registry.run(world, move_id, 1.0f32).unwrap();
        registry.run(world, decay_id, 3u32).unwrap();
    }

    let (ax, ay) = avg_pos(world);
    let avg_hp = avg_health(world);
    println!("  after 10 steps: avg pos = ({ax:.1}, {ay:.1}), avg health = {avg_hp:.0}");
}

fn show_conflict_detection(
    registry: &ReducerRegistry,
    move_id: QueryReducerId,
    decay_id: QueryReducerId,
) {
    let move_access = registry.query_reducer_access(move_id);
    let decay_access = registry.query_reducer_access(decay_id);

    println!(
        "  move vs decay: {}",
        if move_access.conflicts_with(decay_access) {
            "CONFLICT"
        } else {
            "compatible (disjoint components: Pos+Vel vs Health)"
        }
    );
    println!(
        "  move vs move: {}",
        if move_access.conflicts_with(move_access) {
            "CONFLICT (both write Pos)"
        } else {
            "compatible"
        }
    );
    println!(
        "  decay vs decay: {}",
        if decay_access.conflicts_with(decay_access) {
            "CONFLICT (both write Health)"
        } else {
            "compatible"
        }
    );
}

// ════════════════════════════════════════════════════════════════════
// Part 3: Join plans inside transactions
// ════════════════════════════════════════════════════════════════════

/// Build a join plan that finds entities with both Pos and Health.
fn build_join_plan(world: &World) -> QueryPlanResult {
    let planner = QueryPlanner::new(world);
    planner
        .scan::<(&Pos,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build()
}

fn run_transactional_join(world: &mut World) {
    let strategy = Optimistic::new(world);
    let access = Access::of::<(&Pos, &Health, &mut Health)>(world);

    // Build the join plan outside the transaction — plans are reusable.
    let mut plan = build_join_plan(world);

    // Use execute_raw inside a transaction to get joined entity list.
    strategy
        .transact(world, &access, |tx, world| {
            let joined = plan
                .execute_raw(world)
                .expect("same world, plan supports joins");

            println!(
                "    execute_raw: {count} entities in join result",
                count = joined.len()
            );

            // Apply decay to every entity in the join result.
            for &entity in joined {
                let hp = tx
                    .query::<(Entity, &Health)>(world)
                    .find(|(e, _)| *e == entity)
                    .map_or(0, |(_, h)| h.0);
                tx.write::<Health>(world, entity, Health(hp.saturating_sub(5)));
            }
        })
        .expect("clean commit");

    // Use for_each_raw inside a transaction — collect join results, then mutate.
    // for_each_raw borrows &World, so we collect entities first, then process.
    strategy
        .transact(world, &access, |tx, world| {
            let mut joined = Vec::new();
            plan.for_each_raw(world, |entity| {
                joined.push(entity);
            })
            .expect("for_each_raw supports joins");

            println!(
                "    for_each_raw: collected {count} joined entities",
                count = joined.len()
            );

            for entity in joined {
                let hp = tx
                    .query::<(Entity, &Health)>(world)
                    .find(|(e, _)| *e == entity)
                    .map_or(0, |(_, h)| h.0);
                tx.write::<Health>(world, entity, Health(hp.saturating_sub(5)));
            }
        })
        .expect("clean commit");

    let avg = avg_health(world);
    println!("    after 2 transactional join-driven decays: avg health = {avg:.0}");
}

fn run_left_join_demo(world: &mut World) {
    // Spawn some entities without Health to demonstrate left join semantics.
    for i in 0..20 {
        world.spawn((Pos {
            x: 200.0 + i as f32,
            y: 0.0,
        },));
    }

    let planner = QueryPlanner::new(world);
    let mut inner_plan = planner
        .scan::<(&Pos,)>()
        .join::<(&Health,)>(JoinKind::Inner)
        .build();
    let mut left_plan = planner
        .scan::<(&Pos,)>()
        .join::<(&Health,)>(JoinKind::Left)
        .build();

    let inner = inner_plan.execute_raw(world).unwrap().len();
    let left = left_plan.execute_raw(world).unwrap().len();
    println!("    inner join: {inner} entities (Pos ∩ Health)");
    println!("    left join:  {left} entities (all Pos, including those without Health)");
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    println!("Transaction demo — 100 entities with (Pos, Vel, Health)");
    println!();

    // ── Part 1: Raw Tx API ──────────────────────────────────────────

    println!("=== Part 1: Raw Tx API ===");
    println!();

    println!("1. Sequential (begin/commit, zero-cost passthrough)");
    let (mut world, _entities) = spawn_world();
    run_raw_sequential(&mut world);
    println!();

    println!("2. Optimistic (transact closure, tick-based validation)");
    run_raw_optimistic(&mut world);
    println!();

    // ── Part 2: Reducer API ─────────────────────────────────────────

    println!("=== Part 2: Reducer API (same logic, less boilerplate) ===");
    println!();

    // Fresh world for each strategy to show identical results.
    let mut registry = ReducerRegistry::new();

    // Register once — dispatched identically regardless of strategy context.
    let (mut world, _) = spawn_world();
    let (move_id, decay_id) = register_reducers(&mut registry, &mut world);

    // 3. Query reducers run directly on &mut World via registry.run().
    //    No transaction needed — the Access metadata is still tracked.
    println!("3. Query reducers (direct &mut World, all strategies equivalent)");
    run_reducer_demo(&registry, &mut world, move_id, decay_id);
    println!();

    // 4. Conflict detection — free from the Access metadata recorded at registration.
    println!("4. Conflict detection via query_reducer_access()");
    show_conflict_detection(&registry, move_id, decay_id);
    println!();

    // 5. Same reducers, fresh world — demonstrates reproducibility.
    println!("5. Same reducers, fresh world (strategy-agnostic reproducibility)");
    let (mut world2, _) = spawn_world();
    run_reducer_demo(&registry, &mut world2, move_id, decay_id);
    println!();

    // ── Part 3: Join plans inside transactions ──────────────────────

    println!("=== Part 3: Join plans inside transactions ===");
    println!();

    println!("6. execute_raw + for_each_raw with join plans inside Optimistic Tx");
    let (mut world3, _) = spawn_world();
    run_transactional_join(&mut world3);
    println!();

    println!("7. Inner vs Left join comparison (execute_raw)");
    run_left_join_demo(&mut world3);
    println!();

    println!("Done.");
}
