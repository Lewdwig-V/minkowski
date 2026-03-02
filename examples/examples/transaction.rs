//! Transaction strategies — demonstrates Sequential, Optimistic, and Pessimistic
//! transaction semantics on the same workload.
//!
//! Run: cargo run -p minkowski-examples --example transaction --release
//!
//! Features exercised:
//! - `TransactionStrategy` trait with three built-in strategies
//! - `SequentialTx` — zero-overhead passthrough, commit always succeeds
//! - `OptimisticTx` — live reads, buffered writes, tick-based validation at commit
//! - `PessimisticTx` — cooperative column locks, guaranteed commit success
//! - `Access` — component-level read/write metadata for transaction begin
//! - `EnumChangeSet` — reverse changeset returned on successful commit

use minkowski::{
    Access, Entity, Optimistic, OptimisticTx, Pessimistic, PessimisticTx, Sequential, SequentialTx,
    TransactionStrategy, World,
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
        total += hp.0 .0;
        n += 1;
    }
    total as f32 / n as f32
}

// ── Sequential ──────────────────────────────────────────────────────

fn movement_system(tx: &mut SequentialTx, world: &mut World) {
    for (pos, vel) in tx.query::<(&mut Pos, &Vel)>(world) {
        pos.x += vel.dx;
        pos.y += vel.dy;
    }
}

fn run_sequential(world: &mut World) {
    let access = Access::of::<(&mut Pos, &Vel)>(world);
    let strategy = Sequential;

    for _ in 0..10 {
        let mut tx = strategy.begin(world, &access);
        movement_system(&mut tx, world);
        let result = tx.commit(world);
        assert!(result.is_ok(), "sequential commit always succeeds");
    }

    let (ax, ay) = avg_pos(world);
    println!("  after 10 steps: avg pos = ({ax:.1}, {ay:.1})");
    println!("  commit result:  Ok (always succeeds, zero overhead)");
}

// ── Optimistic (clean commit) ───────────────────────────────────────

fn health_decay_optimistic(tx: &mut OptimisticTx, world: &mut World, entities: &[Entity]) {
    // Read current health values (live read from World via query_raw)
    let healths: Vec<(Entity, u32)> = tx
        .query::<(Entity, &Health)>(world)
        .map(|(e, hp)| (e, hp.0))
        .collect();

    // Buffer writes into changeset
    for (e, hp) in healths {
        if entities.contains(&e) {
            tx.insert::<Health>(world, e, Health(hp.saturating_sub(3)));
        }
    }
}

fn run_optimistic_clean(world: &mut World, entities: &[Entity]) {
    let access = Access::of::<(&Health, &mut Health)>(world);
    let strategy = Optimistic::new(world);

    for _ in 0..10 {
        let mut tx = strategy.begin(world, &access);
        health_decay_optimistic(&mut tx, world, entities);
        let result = tx.commit(world);
        assert!(result.is_ok(), "no concurrent modification = clean commit");
    }

    let avg = avg_health(world);
    println!("  after 10 steps: avg health = {avg:.0}");
    println!("  commit result:  Ok (no conflicts detected)");
}

// ── Optimistic (conflict demonstration) ─────────────────────────────

fn run_optimistic_conflict(world: &mut World) {
    // Declare access: reads Pos, writes Health.
    // The optimistic strategy snapshots read-column ticks at begin.
    let access = Access::of::<(&Pos, &mut Health)>(world);
    let strategy = Optimistic::new(world);

    let tx = strategy.begin(world, &access);

    // Mutate the Pos column directly through World (outside the tx).
    // This advances the Pos column tick, invalidating the snapshot.
    for pos in world.query::<(&mut Pos,)>() {
        pos.0.x += 999.0;
    }

    // Commit detects that the Pos column tick advanced past the snapshot.
    let result = tx.commit(world);
    match result {
        Err(_conflict) => {
            println!("  commit result:  Err(Conflict) — read column was modified");
            println!("  buffered writes discarded (transaction aborted)");
        }
        Ok(_) => {
            println!("  commit result:  Ok (unexpected — conflict should have been detected)");
        }
    }
}

// ── Pessimistic ─────────────────────────────────────────────────────

fn health_decay_pessimistic(tx: &mut PessimisticTx<'_>, world: &mut World, entities: &[Entity]) {
    let healths: Vec<(Entity, u32)> = tx
        .query::<(Entity, &Health)>(world)
        .map(|(e, hp)| (e, hp.0))
        .collect();

    for (e, hp) in healths {
        if entities.contains(&e) {
            tx.insert::<Health>(world, e, Health(hp.saturating_sub(3)));
        }
    }
}

fn run_pessimistic(world: &mut World, entities: &[Entity]) {
    let access = Access::of::<(&Health, &mut Health)>(world);
    let strategy = Pessimistic::new(world);

    for _ in 0..10 {
        let mut tx = strategy.begin(world, &access);
        health_decay_pessimistic(&mut tx, world, entities);
        let result = tx.commit(world);
        assert!(result.is_ok(), "pessimistic commit always succeeds");
    }

    let avg = avg_health(world);
    println!("  after 10 steps: avg health = {avg:.0}");
    println!("  commit result:  Ok (locks guarantee success)");
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    println!("Transaction strategies — 100 entities with (Pos, Vel, Health)");
    println!();

    // ── 1. Sequential ───────────────────────────────────────────────

    println!("1. Sequential (zero-cost passthrough)");
    println!("   Movement system: writes Pos, reads Vel");
    let (mut world, entities) = spawn_world();
    run_sequential(&mut world);
    println!();

    // ── 2. Optimistic (clean commit) ────────────────────────────────

    println!("2. Optimistic (clean commit)");
    println!("   Health decay system: reads Health, buffers writes, validates at commit");
    run_optimistic_clean(&mut world, &entities);
    println!();

    // ── 3. Optimistic (conflict) ────────────────────────────────────

    println!("3. Optimistic (conflict demonstration)");
    println!("   Declared access: reads Pos, writes Health");
    println!("   But Pos column is modified externally — tick advances past snapshot");
    run_optimistic_conflict(&mut world);
    println!();

    // ── 4. Pessimistic (guaranteed commit) ──────────────────────────

    println!("4. Pessimistic (guaranteed commit)");
    println!("   Same health decay, but with cooperative column locks");
    run_pessimistic(&mut world, &entities);
    println!();

    println!("Done.");
}
