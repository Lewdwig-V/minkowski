//! Minimal parallel scheduler — exercises `Access` for conflict detection.
//!
//! Run: cargo run -p minkowski-examples --example scheduler --release
//!
//! This is NOT a real framework scheduler. It demonstrates how a framework
//! author would use `Access::of` and `conflicts_with` to build one.
//!
//! Six systems, three batches:
//! - Batch 0: movement (writes Pos, reads Vel) + health_regen (writes Health)
//! - Batch 1: gravity (writes Vel) + apply_damage (writes Health)
//! - Batch 2: log_positions (reads Pos) + log_health (reads Health)
//!
//! Within each batch, systems touch disjoint component sets and could run
//! in parallel. Across batches, conflicts require sequential execution.

use minkowski::{Access, World};

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

// ── Systems ─────────────────────────────────────────────────────────

fn movement(world: &mut World) {
    for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
        pos.x += vel.dx;
        pos.y += vel.dy;
    }
}

fn gravity(world: &mut World) {
    for vel in world.query::<(&mut Vel,)>() {
        vel.0.dy -= 9.8;
    }
}

fn health_regen(world: &mut World) {
    for hp in world.query::<(&mut Health,)>() {
        hp.0 .0 = hp.0 .0.saturating_add(1);
    }
}

fn apply_damage(world: &mut World) {
    for hp in world.query::<(&mut Health,)>() {
        hp.0 .0 = hp.0 .0.saturating_sub(5);
    }
}

fn log_positions(world: &mut World) {
    let count = world.query::<(&Pos,)>().count();
    println!("    log_positions: {count} entities");
}

fn log_health(world: &mut World) {
    let total: u32 = world.query::<(&Health,)>().map(|h| h.0 .0).sum();
    println!("    log_health: total HP = {total}");
}

// ── Greedy batch scheduler ──────────────────────────────────────────

struct SystemEntry {
    name: &'static str,
    access: Access,
    run: fn(&mut World),
}

/// Assign systems to batches using greedy graph coloring.
/// Systems in the same batch have no conflicts and could run in parallel.
fn assign_batches(systems: &[SystemEntry]) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();

    for i in 0..systems.len() {
        // Find the first batch where system i has no conflicts
        let mut assigned = false;
        for (b, batch) in batches.iter().enumerate() {
            let conflicts_with_batch = batch
                .iter()
                .any(|&j| systems[i].access.conflicts_with(&systems[j].access));
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

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let mut world = World::new();

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

    // Register systems with their access metadata
    let systems = vec![
        SystemEntry {
            name: "movement",
            access: Access::of::<(&mut Pos, &Vel)>(&mut world),
            run: movement,
        },
        SystemEntry {
            name: "gravity",
            access: Access::of::<(&mut Vel,)>(&mut world),
            run: gravity,
        },
        SystemEntry {
            name: "health_regen",
            access: Access::of::<(&mut Health,)>(&mut world),
            run: health_regen,
        },
        SystemEntry {
            name: "apply_damage",
            access: Access::of::<(&mut Health,)>(&mut world),
            run: apply_damage,
        },
        SystemEntry {
            name: "log_positions",
            access: Access::of::<(&Pos,)>(&mut world),
            run: log_positions,
        },
        SystemEntry {
            name: "log_health",
            access: Access::of::<(&Health,)>(&mut world),
            run: log_health,
        },
    ];

    // ── Conflict matrix ─────────────────────────────────────────────

    println!("Conflict matrix:");
    println!();
    let max_name = systems.iter().map(|s| s.name.len()).max().unwrap_or(0);
    for (i, a) in systems.iter().enumerate() {
        for (_j, b) in systems.iter().enumerate().skip(i + 1) {
            let tag = if a.access.conflicts_with(&b.access) {
                "CONFLICT"
            } else {
                "independent"
            };
            println!(
                "  {:>width$} <-> {:<width$}  {}",
                a.name,
                b.name,
                tag,
                width = max_name
            );
        }
    }
    println!();

    // ── Batch assignment ────────────────────────────────────────────

    let batches = assign_batches(&systems);

    println!("Batch assignment ({} batches):", batches.len());
    for (b, batch) in batches.iter().enumerate() {
        let names: Vec<_> = batch.iter().map(|&i| systems[i].name).collect();
        println!("  batch {b}: [{}]", names.join(", "));
    }
    println!();

    // ── Execute ─────────────────────────────────────────────────────

    println!("Running 10 frames:");
    for frame in 0..10 {
        for (b, batch) in batches.iter().enumerate() {
            // Within a batch, systems could run in parallel — they touch
            // disjoint component sets. A real framework would use rayon or
            // scoped threads here. We run sequentially to keep the example
            // focused on Access, not unsafe parallel execution.
            for &i in batch {
                (systems[i].run)(&mut world);
            }

            if frame == 0 {
                let names: Vec<_> = batch.iter().map(|&i| systems[i].name).collect();
                println!("  frame {frame}, batch {b}: ran [{}]", names.join(", "));
            }
        }
    }

    // Final state
    let pos_sum: f32 = world.query::<(&Pos,)>().map(|p| p.0.x).sum();
    let vel_sum: f32 = world.query::<(&Vel,)>().map(|v| v.0.dy).sum();
    let hp_sum: u32 = world.query::<(&Health,)>().map(|h| h.0 .0).sum();

    println!();
    println!("After 10 frames:");
    println!("  avg pos.x:  {:.1}", pos_sum / 100.0);
    println!("  avg vel.dy: {:.1}", vel_sum / 100.0);
    println!("  avg hp:     {:.1}", hp_sum as f32 / 100.0);
    println!();
    println!("Done.");
}
