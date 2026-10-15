#![no_main]

//! Column growth against a small, non-unit-multiple pool budget.
//!
//! Drives the two-frontier `SlabPool` through `World`: wide components push
//! columns past 64 KiB (large runs) and past 1 MiB (multi-unit `grow`
//! widening / shifting), bulk spawns hit frontier collision, and despawning
//! whole archetypes frees runs to leave interior holes and edge retreats.
//! A shadow model checks every live value after each op, so a bad memmove,
//! side table route, or overlapping chunk shows up as a value mismatch.

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use minkowski::{Entity, HugePages, World};

/// 3 MiB + change: three 1 MiB units plus a sub-unit tail that only the
/// bottom frontier can use while the top region is empty.
const BUDGET: usize = 3 * 1024 * 1024 + 12_345;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Small(u64);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Wide([u64; 64]); // 512 B: 128 rows per 64 KiB, 2048 rows per MiB

#[derive(Clone, Copy, Debug, PartialEq)]
struct Huge([u64; 512]); // 4 KiB: crosses 64 KiB at 16 rows

#[derive(Clone, Copy, Debug, Arbitrary)]
enum Kind {
    Small,
    Wide,
    Huge,
    SmallWide,
}

#[derive(Clone, Debug, Arbitrary)]
enum Op {
    /// Spawn up to `count` entities of `kind`, stopping at PoolExhausted.
    SpawnMany(Kind, u16),
    /// Despawn every live entity of `kind`.
    DespawnKind(Kind),
    /// Despawn one live entity (any kind).
    DespawnOne(u16),
}

fn spawn(world: &mut World, kind: Kind, v: u64) -> Option<Entity> {
    match kind {
        Kind::Small => world.try_spawn((Small(v),)).ok(),
        Kind::Wide => world.try_spawn((Wide([v; 64]),)).ok(),
        Kind::Huge => world.try_spawn((Huge([v; 512]),)).ok(),
        Kind::SmallWide => world.try_spawn((Small(v), Wide([v; 64]))).ok(),
    }
}

fn check(world: &World, e: Entity, kind: Kind, v: u64) {
    let ok = match kind {
        Kind::Small => world.get::<Small>(e) == Some(&Small(v)),
        Kind::Wide => world.get::<Wide>(e) == Some(&Wide([v; 64])),
        Kind::Huge => world.get::<Huge>(e) == Some(&Huge([v; 512])),
        Kind::SmallWide => {
            world.get::<Small>(e) == Some(&Small(v)) && world.get::<Wide>(e) == Some(&Wide([v; 64]))
        }
    };
    assert!(ok, "{kind:?} entity {e:?} lost value {v}");
}

fn same_kind(a: Kind, b: Kind) -> bool {
    std::mem::discriminant(&a) == std::mem::discriminant(&b)
}

fuzz_target!(|ops: Vec<Op>| {
    let mut world = World::builder()
        .memory_budget(BUDGET)
        .hugepages(HugePages::Off)
        .build()
        .expect("pool fits in the fuzz environment");
    // BTreeMap: deterministic iteration order for despawns and checks.
    // Keyed by Entity::to_bits (Entity is not Ord).
    let mut live: BTreeMap<u64, (Entity, Kind, u64)> = BTreeMap::new();
    let mut next = 1u64;

    for op in ops.iter().take(64) {
        match *op {
            Op::SpawnMany(kind, count) => {
                for _ in 0..count.min(4096) {
                    let Some(e) = spawn(&mut world, kind, next) else {
                        break; // PoolExhausted is an accepted outcome.
                    };
                    live.insert(e.to_bits(), (e, kind, next));
                    next += 1;
                }
            }
            Op::DespawnKind(kind) => {
                let doomed: Vec<Entity> = live
                    .values()
                    .filter(|(_, k, _)| same_kind(*k, kind))
                    .map(|(e, _, _)| *e)
                    .collect();
                for e in doomed {
                    assert!(world.despawn(e));
                    live.remove(&e.to_bits());
                }
            }
            Op::DespawnOne(i) => {
                if let Some(&(e, _, _)) = live.values().nth(i as usize % live.len().max(1)) {
                    assert!(world.despawn(e));
                    live.remove(&e.to_bits());
                }
            }
        }

        for &(e, kind, v) in live.values() {
            check(&world, e, kind, v);
        }
        world.flush_pool_caches();
        let stats = world.stats();
        assert!(stats.pool_used <= stats.pool_capacity);
    }

    // Dropping the world returns every column to the pool; it must not trip
    // a deallocation assert (foreign pointer, bad side table entry, run).
    drop(world);
});
