#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::World;

#[derive(Clone, Debug, Arbitrary)]
enum Op {
    Spawn(BundleKind),
    Despawn(u8),
    Insert(u8, CompVal),
    Remove(u8, CompKind),
    Get(u8, CompKind),
    GetMut(u8, CompKind),
    Query(QueryShape),
    DespawnBatch(Vec<u8>),
}

fuzz_target!(|ops: Vec<Op>| {
    let mut world = World::new();
    let mut live: Vec<minkowski::Entity> = Vec::new();

    for op in &ops {
        match op {
            Op::Spawn(bundle) => {
                let e = spawn_bundle(&mut world, bundle);
                live.push(e);
            }
            Op::Despawn(idx) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live.swap_remove(i);
                    assert!(world.despawn(e));
                }
            }
            Op::Insert(idx, val) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match val {
                        CompVal::A(v) => world.insert(e, *v),
                        CompVal::B(v) => world.insert(e, *v),
                        CompVal::C(v) => world.insert(e, *v),
                        CompVal::D(v) => world.insert(e, *v),
                    }
                }
            }
            Op::Remove(idx, kind) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match kind {
                        CompKind::A => {
                            world.remove::<A>(e);
                        }
                        CompKind::B => {
                            world.remove::<B>(e);
                        }
                        CompKind::C => {
                            world.remove::<C>(e);
                        }
                        CompKind::D => {
                            world.remove::<D>(e);
                        }
                    }
                }
            }
            Op::Get(idx, kind) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match kind {
                        CompKind::A => {
                            world.get::<A>(e);
                        }
                        CompKind::B => {
                            world.get::<B>(e);
                        }
                        CompKind::C => {
                            world.get::<C>(e);
                        }
                        CompKind::D => {
                            world.get::<D>(e);
                        }
                    }
                }
            }
            Op::GetMut(idx, kind) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match kind {
                        CompKind::A => {
                            world.get_mut::<A>(e);
                        }
                        CompKind::B => {
                            world.get_mut::<B>(e);
                        }
                        CompKind::C => {
                            world.get_mut::<C>(e);
                        }
                        CompKind::D => {
                            world.get_mut::<D>(e);
                        }
                    }
                }
            }
            Op::Query(shape) => {
                run_query(&mut world, *shape);
            }
            Op::DespawnBatch(indices) => {
                if !live.is_empty() {
                    let entities: Vec<_> = indices
                        .iter()
                        .map(|idx| live[*idx as usize % live.len()])
                        .collect();
                    world.despawn_batch(&entities);
                    live.retain(|e| world.is_alive(*e));
                }
            }
        }

        assert_invariants(&world, &live);
    }

    // Cleanup: despawn everything
    for e in &live {
        world.despawn(*e);
    }
    assert_eq!(world.entity_count(), 0);
});
