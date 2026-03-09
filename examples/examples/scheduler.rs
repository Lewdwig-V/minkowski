//! Minimal parallel scheduler — exercises `ReducerRegistry` for access metadata
//! and conflict detection.
//!
//! Run: cargo run -p minkowski-examples --example scheduler --release
//!
//! Six systems registered as query reducers. The scheduler uses
//! `registry.query_reducer_access(id)` to build a conflict matrix and assign
//! systems to batches via greedy graph coloring.
//!
//! Three batches (greedy coloring):
//! - Batch 0: movement (writes Pos, reads Vel) + health_regen (writes Health)
//! - Batch 1: gravity (writes Vel) + apply_damage (writes Health) + log_positions (reads Pos)
//! - Batch 2: log_health (reads Health)
//!
//! Within each batch, systems touch disjoint component sets and could run
//! in parallel. Across batches, conflicts require sequential execution.

use minkowski::{QueryMut, QueryReducerId, QueryRef, ReducerRegistry, World};

// -- Components ---------------------------------------------------------------

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

// -- Greedy batch scheduler ---------------------------------------------------

/// Assign systems to batches using greedy graph coloring.
/// Systems in the same batch have no conflicts and could run in parallel.
fn assign_batches(registry: &ReducerRegistry, ids: &[QueryReducerId]) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();

    for i in 0..ids.len() {
        let access_i = registry.query_reducer_access(ids[i]);
        let mut assigned = false;
        for (b, batch) in batches.iter().enumerate() {
            let conflicts_with_batch = batch.iter().any(|&j| {
                let access_j = registry.query_reducer_access(ids[j]);
                access_i.conflicts_with(access_j)
            });
            if !conflicts_with_batch {
                batches[b].push(i);
                assigned = true;
                break;
            }
        }
        if !assigned {
            batches.push(vec![i]);
        }
    }

    batches
}

// -- Main ---------------------------------------------------------------------

fn main() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    // Spawn some entities
    for i in 0..100 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.0 },
            Health(100),
        ));
    }

    // -- Register the 6 systems as query reducers -----------------------------

    let movement_id = registry
        .register_query::<(&mut Pos, &Vel), (), _>(
            &mut world,
            "movement",
            |mut query: QueryMut<'_, (&mut Pos, &Vel)>, ()| {
                query.for_each(|(pos, vel)| {
                    pos.x += vel.dx;
                    pos.y += vel.dy;
                });
            },
        )
        .unwrap();

    let gravity_id = registry
        .register_query::<(&mut Vel,), (), _>(
            &mut world,
            "gravity",
            |mut query: QueryMut<'_, (&mut Vel,)>, ()| {
                query.for_each(|(vel,)| {
                    vel.dy -= 9.8;
                });
            },
        )
        .unwrap();

    let health_regen_id = registry
        .register_query::<(&mut Health,), (), _>(
            &mut world,
            "health_regen",
            |mut query: QueryMut<'_, (&mut Health,)>, ()| {
                query.for_each(|(hp,)| {
                    hp.0 = hp.0.saturating_add(1);
                });
            },
        )
        .unwrap();

    let apply_damage_id = registry
        .register_query::<(&mut Health,), (), _>(
            &mut world,
            "apply_damage",
            |mut query: QueryMut<'_, (&mut Health,)>, ()| {
                query.for_each(|(hp,)| {
                    hp.0 = hp.0.saturating_sub(5);
                });
            },
        )
        .unwrap();

    let log_positions_id = registry
        .register_query_ref::<(&Pos,), (), _>(
            &mut world,
            "log_positions",
            |mut query: QueryRef<'_, (&Pos,)>, ()| {
                let count = query.count();
                println!("    log_positions: {count} entities");
            },
        )
        .unwrap();

    let log_health_id = registry
        .register_query_ref::<(&Health,), (), _>(
            &mut world,
            "log_health",
            |mut query: QueryRef<'_, (&Health,)>, ()| {
                let mut total: u32 = 0;
                query.for_each(|(h,)| {
                    total += h.0;
                });
                println!("    log_health: total HP = {total}");
            },
        )
        .unwrap();

    // Ordered list of system IDs (names retrieved via registry for display)
    let system_names = [
        "movement",
        "gravity",
        "health_regen",
        "apply_damage",
        "log_positions",
        "log_health",
    ];
    let system_ids = [
        movement_id,
        gravity_id,
        health_regen_id,
        apply_damage_id,
        log_positions_id,
        log_health_id,
    ];

    // -- Conflict matrix ------------------------------------------------------

    println!("Conflict matrix:");
    println!();
    let max_name = system_names.iter().map(|s| s.len()).max().unwrap_or(0);
    for i in 0..system_ids.len() {
        let access_i = registry.query_reducer_access(system_ids[i]);
        for j in (i + 1)..system_ids.len() {
            let access_j = registry.query_reducer_access(system_ids[j]);
            let tag = if access_i.conflicts_with(access_j) {
                "CONFLICT"
            } else {
                "independent"
            };
            println!(
                "  {:>width$} <-> {:<width$}  {}",
                system_names[i],
                system_names[j],
                tag,
                width = max_name
            );
        }
    }
    println!();

    // -- Batch assignment -----------------------------------------------------

    let batches = assign_batches(&registry, &system_ids);

    println!("Batch assignment ({} batches):", batches.len());
    for (b, batch) in batches.iter().enumerate() {
        let names: Vec<_> = batch.iter().map(|&i| system_names[i]).collect();
        println!("  batch {b}: [{}]", names.join(", "));
    }
    println!();

    // -- Execute --------------------------------------------------------------

    println!("Running 10 frames:");
    for frame in 0..10 {
        for (b, batch) in batches.iter().enumerate() {
            // Within a batch, systems could run in parallel — they touch
            // disjoint component sets. A real framework would use rayon or
            // scoped threads here. We run sequentially to keep the example
            // focused on ReducerRegistry, not unsafe parallel execution.
            for &i in batch {
                registry.run(&mut world, system_ids[i], ()).unwrap();
            }

            if frame == 0 {
                let names: Vec<_> = batch.iter().map(|&i| system_names[i]).collect();
                println!("  frame {frame}, batch {b}: ran [{}]", names.join(", "));
            }
        }
    }

    // Final state
    let pos_sum: f32 = world.query::<&Pos>().map(|p| p.x).sum();
    let vel_sum: f32 = world.query::<&Vel>().map(|v| v.dy).sum();
    let hp_sum: u32 = world.query::<&Health>().map(|h| h.0).sum();

    println!();
    println!("After 10 frames:");
    println!("  avg pos.x:  {:.1}", pos_sum / 100.0);
    println!("  avg vel.dy: {:.1}", vel_sum / 100.0);
    println!("  avg hp:     {:.1}", hp_sum as f32 / 100.0);
    println!();
    println!("Done.");
}
