//! Multi-threaded arena battle — stress-tests optimistic vs pessimistic
//! transaction strategies under tunable conflict rates.
//!
//! Run: cargo run -p minkowski-examples --example battle --release
//!
//! Features exercised:
//! - `TransactionStrategy` with `Optimistic` and `Pessimistic` strategies
//! - `rayon::join` for parallel read phases (concurrent `&World` access)
//! - Split-phase execution: parallel reads, sequential writes, sequential commit
//! - Conflict detection under disjoint vs overlapping write-sets
//!
//! Two modes:
//! - **Low conflict**: combat writes `Damage`, healing writes `Healing` — disjoint
//!   columns, so optimistic commits succeed without conflicts. A separate
//!   `apply_effects` step merges both into `Health`.
//! - **High conflict**: both combat and healing write `Health` directly. Both
//!   transactions read and write the same column, so the first commit advances
//!   the Health column tick, causing the second commit to detect a conflict.
//!   The conflicted transaction's writes are discarded (optimistic abort).

use minkowski::{Access, Entity, Optimistic, Pessimistic, TransactionStrategy, World};
use std::time::Instant;

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Health(u32);

#[derive(Clone, Copy)]
struct Team(u8);

#[derive(Clone, Copy)]
struct Damage(u32);

#[derive(Clone, Copy)]
struct Healing(u32);

// ── Conflict mode ───────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConflictMode {
    /// Combat writes Damage, healing writes Healing. Disjoint columns.
    Low,
    /// Both combat and healing write Health directly. Overlapping.
    High,
}

// ── System computations ─────────────────────────────────────────────
//
// Pure functions: read entity snapshots, return intended mutations.

/// Compute damage for opposing-team entities.
fn compute_combat(entities: &[(Entity, u8, u32)], mode: ConflictMode) -> Vec<(Entity, u32)> {
    entities
        .iter()
        .filter(|&&(_, team, _)| match mode {
            ConflictMode::Low => team == 1,
            ConflictMode::High => true,
        })
        .map(|&(e, _, hp)| {
            let dmg = 5u32;
            match mode {
                ConflictMode::Low => (e, dmg),
                ConflictMode::High => (e, hp.saturating_sub(dmg)),
            }
        })
        .collect()
}

/// Compute healing for friendly-team entities.
fn compute_healing(entities: &[(Entity, u8, u32)], mode: ConflictMode) -> Vec<(Entity, u32)> {
    entities
        .iter()
        .filter(|&&(_, team, _)| match mode {
            ConflictMode::Low => team == 0,
            ConflictMode::High => true,
        })
        .map(|&(e, _, hp)| {
            let heal = 3u32;
            match mode {
                ConflictMode::Low => (e, heal),
                ConflictMode::High => (e, hp.saturating_add(heal).min(100)),
            }
        })
        .collect()
}

/// Low-conflict only: merge Damage + Healing into Health.
fn apply_effects(world: &mut World) {
    let effects: Vec<(Entity, u32)> = world
        .query::<(Entity, &Health, &Damage, &Healing)>()
        .map(|(e, hp, dmg, heal)| {
            let new_hp = hp.0.saturating_sub(dmg.0).saturating_add(heal.0).min(100);
            (e, new_hp)
        })
        .collect();

    for (e, new_hp) in effects {
        world.insert(e, Health(new_hp));
    }
}

// ── Stats ───────────────────────────────────────────────────────────

struct FrameStats {
    commits: u32,
    conflicts: u32,
    total_time: std::time::Duration,
    frames: u32,
}

impl FrameStats {
    fn new() -> Self {
        Self {
            commits: 0,
            conflicts: 0,
            total_time: std::time::Duration::ZERO,
            frames: 0,
        }
    }

    fn print(&self, label: &str) {
        let avg_ms = if self.frames > 0 {
            self.total_time.as_secs_f64() * 1000.0 / self.frames as f64
        } else {
            0.0
        };
        println!("{label}");
        println!(
            "  commits: {} | conflicts: {} | avg frame: {:.2}ms",
            self.commits, self.conflicts, avg_ms,
        );
    }
}

// ── Optimistic frame ────────────────────────────────────────────────

fn run_frame_optimistic(
    world: &mut World,
    mode: ConflictMode,
    combat_access: &Access,
    healing_access: &Access,
    stats: &mut FrameStats,
) {
    let frame_start = Instant::now();

    // 1. Begin phase (sequential -- needs &mut World).
    let strategy = Optimistic::new(world);
    let mut tx_combat = strategy.begin(world, combat_access);
    let mut tx_healing = strategy.begin(world, healing_access);

    // 2. Parallel read phase.
    //    tx.query(&self, &World) takes shared refs -- safe for concurrent reads.
    //    Each closure captures &tx (Sync) and &world (Sync), both shareable.
    let (combat_results, healing_results) = rayon::join(
        || {
            let entities: Vec<(Entity, u8, u32)> = tx_combat
                .query::<(Entity, &Team, &Health)>(world)
                .map(|(e, t, h)| (e, t.0, h.0))
                .collect();
            compute_combat(&entities, mode)
        },
        || {
            let entities: Vec<(Entity, u8, u32)> = tx_healing
                .query::<(Entity, &Team, &Health)>(world)
                .map(|(e, t, h)| (e, t.0, h.0))
                .collect();
            compute_healing(&entities, mode)
        },
    );

    // 3. Sequential write phase -- buffer results into changesets.
    match mode {
        ConflictMode::Low => {
            for &(entity, dmg) in &combat_results {
                tx_combat.insert::<Damage>(world, entity, Damage(dmg));
            }
            for &(entity, heal) in &healing_results {
                tx_healing.insert::<Healing>(world, entity, Healing(heal));
            }
        }
        ConflictMode::High => {
            for &(entity, new_hp) in &combat_results {
                tx_combat.insert::<Health>(world, entity, Health(new_hp));
            }
            for &(entity, new_hp) in &healing_results {
                tx_healing.insert::<Health>(world, entity, Health(new_hp));
            }
        }
    }

    // 4. Commit phase (sequential).
    //    In high-conflict mode, the first commit writes Health and advances
    //    the column tick. The second commit's validation sees this and fails.
    match tx_combat.commit(world) {
        Ok(_) => stats.commits += 1,
        Err(_) => stats.conflicts += 1,
    }
    match tx_healing.commit(world) {
        Ok(_) => stats.commits += 1,
        Err(_) => stats.conflicts += 1,
    }

    // 5. Low-conflict mode: merge Damage + Healing into Health.
    if mode == ConflictMode::Low {
        apply_effects(world);
    }

    stats.total_time += frame_start.elapsed();
    stats.frames += 1;
}

// ── Pessimistic frame ───────────────────────────────────────────────

fn run_frame_pessimistic(
    world: &mut World,
    mode: ConflictMode,
    combat_access: &Access,
    healing_access: &Access,
    stats: &mut FrameStats,
) {
    let frame_start = Instant::now();

    // Single strategy instance — both transactions share the same lock table,
    // so conflicting locks are correctly detected across concurrent systems.
    let strategy = Pessimistic::new(world);

    // 1. Begin phase (sequential).
    let mut tx_combat = strategy.begin(world, combat_access);
    let mut tx_healing = strategy.begin(world, healing_access);

    // If the second tx couldn't acquire locks (write-write conflict on same
    // columns), fall back to sequential execution: commit combat first, then
    // re-begin healing with locks now available.
    if !tx_healing.has_locks() {
        stats.conflicts += 1;
        // Run combat to completion
        let entities: Vec<(Entity, u8, u32)> = tx_combat
            .query::<(Entity, &Team, &Health)>(world)
            .map(|(e, t, h)| (e, t.0, h.0))
            .collect();
        let combat_results = compute_combat(&entities, mode);
        match mode {
            ConflictMode::Low => {
                for &(entity, dmg) in &combat_results {
                    tx_combat.insert::<Damage>(world, entity, Damage(dmg));
                }
            }
            ConflictMode::High => {
                for &(entity, new_hp) in &combat_results {
                    tx_combat.insert::<Health>(world, entity, Health(new_hp));
                }
            }
        }
        let _ = tx_combat.commit(world);
        stats.commits += 1;
        drop(tx_healing); // release (no locks to release)

        // Re-begin healing — locks now available
        let mut tx_healing = strategy.begin(world, healing_access);
        let entities: Vec<(Entity, u8, u32)> = tx_healing
            .query::<(Entity, &Team, &Health)>(world)
            .map(|(e, t, h)| (e, t.0, h.0))
            .collect();
        let healing_results = compute_healing(&entities, mode);
        match mode {
            ConflictMode::Low => {
                for &(entity, heal) in &healing_results {
                    tx_healing.insert::<Healing>(world, entity, Healing(heal));
                }
            }
            ConflictMode::High => {
                for &(entity, new_hp) in &healing_results {
                    tx_healing.insert::<Health>(world, entity, Health(new_hp));
                }
            }
        }
        let _ = tx_healing.commit(world);
        stats.commits += 1;
        if mode == ConflictMode::Low {
            apply_effects(world);
        }
        stats.total_time += frame_start.elapsed();
        stats.frames += 1;
        return;
    }

    // 2. Parallel read phase — both txs have locks, safe to read concurrently.
    let (combat_results, healing_results) = rayon::join(
        || {
            let entities: Vec<(Entity, u8, u32)> = tx_combat
                .query::<(Entity, &Team, &Health)>(world)
                .map(|(e, t, h)| (e, t.0, h.0))
                .collect();
            compute_combat(&entities, mode)
        },
        || {
            let entities: Vec<(Entity, u8, u32)> = tx_healing
                .query::<(Entity, &Team, &Health)>(world)
                .map(|(e, t, h)| (e, t.0, h.0))
                .collect();
            compute_healing(&entities, mode)
        },
    );

    // 3. Sequential write phase.
    match mode {
        ConflictMode::Low => {
            for &(entity, dmg) in &combat_results {
                tx_combat.insert::<Damage>(world, entity, Damage(dmg));
            }
            for &(entity, heal) in &healing_results {
                tx_healing.insert::<Healing>(world, entity, Healing(heal));
            }
        }
        ConflictMode::High => {
            for &(entity, new_hp) in &combat_results {
                tx_combat.insert::<Health>(world, entity, Health(new_hp));
            }
            for &(entity, new_hp) in &healing_results {
                tx_healing.insert::<Health>(world, entity, Health(new_hp));
            }
        }
    }

    // 4. Commit phase — locks guarantee success.
    let _ = tx_combat.commit(world);
    stats.commits += 1;
    let _ = tx_healing.commit(world);
    stats.commits += 1;

    // 5. Low-conflict mode: merge effects.
    if mode == ConflictMode::Low {
        apply_effects(world);
    }

    stats.total_time += frame_start.elapsed();
    stats.frames += 1;
}

// ── World setup ─────────────────────────────────────────────────────

fn spawn_arena(world: &mut World, count: usize) {
    let per_team = count / 2;
    for i in 0..count {
        let team = if i < per_team { 0u8 } else { 1u8 };
        world.spawn((Health(100), Team(team), Damage(0), Healing(0)));
    }
}

fn avg_health(world: &mut World) -> f32 {
    let (mut total, mut n) = (0u64, 0u32);
    for hp in world.query::<(&Health,)>() {
        total += hp.0 .0 as u64;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    total as f32 / n as f32
}

// ── Main ────────────────────────────────────────────────────────────

fn run_scenario(
    label: &str,
    mode: ConflictMode,
    entity_count: usize,
    frames: usize,
    use_optimistic: bool,
) {
    let mut world = World::new();
    spawn_arena(&mut world, entity_count);

    // Access descriptors must be created per-world (component IDs are per-registry).
    let (combat_access, healing_access) = match mode {
        ConflictMode::Low => (
            Access::of::<(Entity, &Team, &Health, &mut Damage)>(&mut world),
            Access::of::<(Entity, &Team, &Health, &mut Healing)>(&mut world),
        ),
        ConflictMode::High => (
            Access::of::<(Entity, &Team, &Health, &mut Health)>(&mut world),
            Access::of::<(Entity, &Team, &Health, &mut Health)>(&mut world),
        ),
    };

    let mut stats = FrameStats::new();
    for _ in 0..frames {
        if use_optimistic {
            run_frame_optimistic(
                &mut world,
                mode,
                &combat_access,
                &healing_access,
                &mut stats,
            );
        } else {
            run_frame_pessimistic(
                &mut world,
                mode,
                &combat_access,
                &healing_access,
                &mut stats,
            );
        }
    }

    let health = avg_health(&mut world);
    stats.print(label);
    println!("  avg health: {health:.0}");
}

fn main() {
    let entity_count = 500;
    let frames = 100;

    println!(
        "Battle simulation: {entity_count} entities ({} per team), {frames} frames",
        entity_count / 2,
    );
    println!();

    // ── Low conflict mode ───────────────────────────────────────────

    println!("=== Low conflict mode (disjoint targets) ===");
    println!();

    run_scenario("Optimistic:", ConflictMode::Low, entity_count, frames, true);
    println!();
    run_scenario(
        "Pessimistic:",
        ConflictMode::Low,
        entity_count,
        frames,
        false,
    );
    println!();

    // ── High conflict mode ──────────────────────────────────────────

    println!("=== High conflict mode (overlapping targets) ===");
    println!();

    run_scenario(
        "Optimistic:",
        ConflictMode::High,
        entity_count,
        frames,
        true,
    );
    println!();
    run_scenario(
        "Pessimistic:",
        ConflictMode::High,
        entity_count,
        frames,
        false,
    );
    println!();

    println!("Done.");
}
