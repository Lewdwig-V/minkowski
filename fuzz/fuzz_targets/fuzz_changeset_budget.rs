#![no_main]

//! Changeset apply against a small pool budget (#294).
//!
//! Random batches of migrations, sparse writes and despawns are applied as one
//! `EnumChangeSet` each. Applying must either succeed, or fail with
//! `ApplyError::PoolExhausted` and leave the world exactly as before (the
//! reservation runs before any mutation). A shadow model checks every live
//! value after each op, so a reservation that misses a growth site shows up as
//! an abort or a value mismatch.

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use minkowski::{ApplyError, EnumChangeSet, Entity, HugePages, World};

/// 1 MiB + change: enough for a few thousand entities, not for every batch.
const BUDGET: usize = (1 << 20) + 4_321;

#[derive(Clone, Copy, Debug, PartialEq)]
struct A(u32);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Wide([u64; 64]); // 512 B: migrations into this archetype grow fast

#[derive(Clone, Copy, Debug, PartialEq)]
struct Tag(u64); // stored sparse

#[derive(Clone, Copy, Debug, Arbitrary)]
enum Action {
    InsertWide,
    RemoveWide,
    InsertTag,
    RemoveTag,
    Despawn,
}

#[derive(Clone, Debug, Arbitrary)]
enum Op {
    /// Spawn up to `count` entities directly, stopping at PoolExhausted.
    Spawn(u16),
    /// One changeset of `(first entity, span, action)` entries: the action
    /// applies to `span + 1` consecutive live entities.
    Batch(Vec<(u16, u8, Action)>),
}

#[derive(Clone, Default)]
struct Model {
    a: u32,
    wide: Option<u64>,
    tag: Option<u64>,
}

fn check(world: &World, live: &BTreeMap<u64, (Entity, Model)>) {
    assert_eq!(world.entity_count(), live.len(), "entity count");
    for (e, m) in live.values() {
        assert_eq!(world.get::<A>(*e), Some(&A(m.a)), "{e:?} A");
        assert_eq!(world.get::<Wide>(*e).map(|w| w.0[0]), m.wide, "{e:?} Wide");
        assert_eq!(world.get::<Tag>(*e).map(|t| t.0), m.tag, "{e:?} Tag");
    }
}

#[derive(Clone, Debug, Arbitrary)]
struct Input {
    /// Initial fill: `fill * 64` entities, so the fuzzer can start anywhere
    /// from an empty pool to one close to full.
    fill: u8,
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    let Input { fill, ops } = input;
    let mut world = World::builder()
        .memory_budget(BUDGET)
        .hugepages(HugePages::Off)
        .build()
        .expect("pool fits in the fuzz environment");
    world.register_component::<A>();
    world.register_component::<Wide>();
    // Mark Tag sparse up front (first sparse write would do it too).
    let seed = world.spawn((A(0),));
    world.insert_sparse(seed, Tag(0));
    world.despawn(seed);

    let mut live: BTreeMap<u64, (Entity, Model)> = BTreeMap::new();
    let mut next = 1u64;
    for _ in 0..usize::from(fill) * 64 {
        let a = next as u32;
        let Ok(e) = world.try_spawn((A(a),)) else { break };
        live.insert(e.to_bits(), (e, Model { a, ..Model::default() }));
        next += 1;
    }

    for op in ops.iter().take(48) {
        match op {
            Op::Spawn(count) => {
                for _ in 0..*count {
                    let a = next as u32;
                    let Ok(e) = world.try_spawn((A(a),)) else { break };
                    live.insert(e.to_bits(), (e, Model { a, ..Model::default() }));
                    next += 1;
                }
            }
            Op::Batch(actions) => {
                if live.is_empty() {
                    continue;
                }
                let keys: Vec<u64> = live.keys().copied().collect();
                let mut planned = live.clone();
                let mut cs = EnumChangeSet::new();
                let targets = actions.iter().take(256).flat_map(|&(pick, span, action)| {
                    (0..=usize::from(span)).map(move |k| (pick as usize + k, action))
                });
                for (pick, action) in targets {
                    let key = keys[pick % keys.len()];
                    let Some((e, m)) = planned.get_mut(&key) else {
                        continue; // despawned earlier in this batch
                    };
                    let e = *e;
                    match action {
                        Action::InsertWide => {
                            cs.insert(&mut world, e, Wide([next; 64]));
                            m.wide = Some(next);
                        }
                        Action::RemoveWide => {
                            cs.remove::<Wide>(&mut world, e);
                            m.wide = None;
                        }
                        Action::InsertTag => {
                            cs.insert_sparse(&mut world, e, Tag(next));
                            m.tag = Some(next);
                        }
                        Action::RemoveTag => {
                            cs.remove_sparse::<Tag>(&mut world, e);
                            m.tag = None;
                        }
                        Action::Despawn => {
                            cs.record_despawn(e);
                            planned.remove(&key);
                        }
                    }
                    next += 1;
                }
                match cs.apply(&mut world) {
                    Ok(()) => live = planned,
                    Err(ApplyError::PoolExhausted(_)) => {} // world unchanged
                    Err(e) => panic!("unexpected apply error: {e}"),
                }
            }
        }
        check(&world, &live);
    }
});
