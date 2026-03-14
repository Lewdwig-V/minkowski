//! Multi-threaded arena battle — stress-tests optimistic vs pessimistic
//! transaction strategies under tunable conflict rates using EntityMut reducers.
//!
//! Run: cargo run -p minkowski-examples --example battle --release
//!
//! Features exercised:
//! - `ReducerRegistry` with `register_entity` for typed per-entity mutations
//! - `EntityMut<C>` handles with compile-time access enforcement
//! - `registry.call()` dispatching through `Optimistic` / `Pessimistic` strategies
//! - `rayon::join` for parallel snapshot computation, sequential reducer dispatch
//! - Conflict detection under disjoint vs overlapping write-sets
//!
//! Two modes:
//! - **Low conflict**: attack writes `Damage`, heal writes `Healing` — disjoint
//!   component sets, so optimistic commits succeed without conflicts. A separate
//!   `apply_effects` step merges both into `Health`.
//! - **High conflict**: both attack and heal write `Health` directly. Both
//!   reducers touch the same column, so optimistic conflicts are expected.
//!   The strategy's internal retry loop handles them.

use minkowski::{
    Entity, EntityMut, Optimistic, Pessimistic, ReducerId, ReducerRegistry, Transact, World,
};
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
    /// Attack writes Damage, heal writes Healing. Disjoint columns.
    Low,
    /// Both attack and heal write Health directly. Overlapping.
    High,
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
            self.commits, self.conflicts, avg_ms
        );
    }
}

// ── Reducer IDs for each scenario ───────────────────────────────────

struct ReducerIds {
    /// Low-conflict: writes Damage component
    attack_low: ReducerId,
    /// Low-conflict: writes Healing component
    heal_low: ReducerId,
    /// High-conflict: writes Health directly
    attack_high: ReducerId,
    /// High-conflict: writes Health directly
    heal_high: ReducerId,
}

// ── System computations ─────────────────────────────────────────────
//
// Pure functions: read entity snapshots, return intended targets + amounts.

/// Compute attack targets for opposing-team entities.
/// Returns (entity, delta) — always a relative amount, never absolute.
fn compute_combat_targets(
    entities: &[(Entity, u8, u32)],
    mode: ConflictMode,
) -> Vec<(Entity, u32)> {
    let dmg = 5u32;
    entities
        .iter()
        .filter(|&&(_, team, _)| match mode {
            ConflictMode::Low => team == 1,
            ConflictMode::High => true,
        })
        .map(|&(e, _, _)| (e, dmg))
        .collect()
}

/// Compute healing targets for friendly-team entities.
/// Returns (entity, delta) — always a relative amount, never absolute.
fn compute_healing_targets(
    entities: &[(Entity, u8, u32)],
    mode: ConflictMode,
) -> Vec<(Entity, u32)> {
    let heal = 3u32;
    entities
        .iter()
        .filter(|&&(_, team, _)| match mode {
            ConflictMode::Low => team == 0,
            ConflictMode::High => true,
        })
        .map(|&(e, _, _)| (e, heal))
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
        world.insert(e, (Health(new_hp),)).unwrap();
    }
}

// ── Frame execution ─────────────────────────────────────────────────

fn run_frame<S: Transact>(
    world: &mut World,
    registry: &ReducerRegistry,
    strategy: &S,
    mode: ConflictMode,
    ids: &ReducerIds,
    stats: &mut FrameStats,
) {
    let frame_start = Instant::now();

    // 1. Snapshot phase: read all entity state for target computation.
    let snapshot: Vec<(Entity, u8, u32)> = world
        .query::<(Entity, &Team, &Health)>()
        .map(|(e, t, h)| (e, t.0, h.0))
        .collect();

    // 2. Parallel computation: determine targets + amounts using rayon.
    //    This is pure computation on snapshot data — no World access needed.
    let (combat_targets, healing_targets) = rayon::join(
        || compute_combat_targets(&snapshot, mode),
        || compute_healing_targets(&snapshot, mode),
    );

    // 3. Sequential dispatch: apply mutations via registry.call().
    //    Each call() runs through the strategy's transact() with internal retry.
    let (attack_id, heal_id) = match mode {
        ConflictMode::Low => (ids.attack_low, ids.heal_low),
        ConflictMode::High => (ids.attack_high, ids.heal_high),
    };

    for &(entity, amount) in &combat_targets {
        match registry.call(strategy, world, attack_id, (entity, amount)) {
            Ok(()) => stats.commits += 1,
            Err(_) => stats.conflicts += 1,
        }
    }

    for &(entity, amount) in &healing_targets {
        match registry.call(strategy, world, heal_id, (entity, amount)) {
            Ok(()) => stats.commits += 1,
            Err(_) => stats.conflicts += 1,
        }
    }

    // 4. Low-conflict mode: merge Damage + Healing into Health.
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
        let team = u8::from(i >= per_team);
        world.spawn((Health(100), Team(team), Damage(0), Healing(0)));
    }
}

fn avg_health(world: &mut World) -> f32 {
    let (mut total, mut n) = (0u64, 0u32);
    for hp in world.query::<&Health>() {
        total += hp.0 as u64;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    total as f32 / n as f32
}

// ── Main ────────────────────────────────────────────────────────────

fn register_reducers(world: &mut World, registry: &mut ReducerRegistry) -> ReducerIds {
    // Low-conflict: attack writes Damage (disjoint from Healing)
    let attack_low = registry
        .register_entity::<(Damage,), u32, _>(
            world,
            "attack_low",
            |mut entity: EntityMut<'_, (Damage,)>, dmg: u32| {
                entity.set::<Damage, 0>(Damage(dmg));
            },
        )
        .unwrap();

    // Low-conflict: heal writes Healing (disjoint from Damage)
    let heal_low = registry
        .register_entity::<(Healing,), u32, _>(
            world,
            "heal_low",
            |mut entity: EntityMut<'_, (Healing,)>, heal: u32| {
                entity.set::<Healing, 0>(Healing(heal));
            },
        )
        .unwrap();

    // High-conflict: attack reads + writes Health (read-modify-write)
    let attack_high = registry
        .register_entity::<(Health,), u32, _>(
            world,
            "attack_high",
            |mut entity: EntityMut<'_, (Health,)>, dmg: u32| {
                let hp = entity.get::<Health, 0>().0;
                entity.set::<Health, 0>(Health(hp.saturating_sub(dmg)));
            },
        )
        .unwrap();

    // High-conflict: heal reads + writes Health (read-modify-write)
    let heal_high = registry
        .register_entity::<(Health,), u32, _>(
            world,
            "heal_high",
            |mut entity: EntityMut<'_, (Health,)>, heal: u32| {
                let hp = entity.get::<Health, 0>().0;
                entity.set::<Health, 0>(Health(hp.saturating_add(heal).min(100)));
            },
        )
        .unwrap();

    // Verify access conflicts match expectations
    let attack_low_access = registry.reducer_access(attack_low);
    let heal_low_access = registry.reducer_access(heal_low);
    let attack_high_access = registry.reducer_access(attack_high);
    let heal_high_access = registry.reducer_access(heal_high);

    assert!(
        !attack_low_access.conflicts_with(heal_low_access),
        "low-conflict reducers should not conflict (disjoint components)"
    );
    assert!(
        attack_high_access.conflicts_with(heal_high_access),
        "high-conflict reducers should conflict (both write Health)"
    );

    println!("  attack_low vs heal_low: compatible (disjoint components)");
    println!("  attack_high vs heal_high: CONFLICT (both write Health)");

    ReducerIds {
        attack_low,
        heal_low,
        attack_high,
        heal_high,
    }
}

fn run_scenario(
    label: &str,
    mode: ConflictMode,
    entity_count: usize,
    frames: usize,
    use_optimistic: bool,
) {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let ids = register_reducers(&mut world, &mut registry);

    spawn_arena(&mut world, entity_count);

    let mut stats = FrameStats::new();

    if use_optimistic {
        let strategy = Optimistic::new(&world);
        for _ in 0..frames {
            run_frame(&mut world, &registry, &strategy, mode, &ids, &mut stats);
        }
    } else {
        let strategy = Pessimistic::new(&world);
        for _ in 0..frames {
            run_frame(&mut world, &registry, &strategy, mode, &ids, &mut stats);
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
