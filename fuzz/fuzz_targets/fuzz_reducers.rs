#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::World;

#[derive(Clone, Debug, Arbitrary)]
enum ReducerOp {
    QueryMut(QueryShape),
    QueryRef(QueryShape),
    ForEachChunk(QueryShape),
    SpawnMore(BundleKind),
    DespawnSome(u8),
}

/// Run a query using for_each_chunk to exercise typed slice construction.
fn run_for_each_chunk(world: &mut World, shape: QueryShape) {
    match shape {
        QueryShape::RefA => {
            world.query::<(&A,)>().for_each_chunk(|(_,)| {});
        }
        QueryShape::RefAB => {
            world.query::<(&A, &B)>().for_each_chunk(|(_, _)| {});
        }
        QueryShape::RefABC => {
            world.query::<(&A, &B, &C)>().for_each_chunk(|(_, _, _)| {});
        }
        QueryShape::MutA => {
            world.query::<(&mut A,)>().for_each_chunk(|(a,)| {
                for v in a.iter_mut() {
                    v.0 = v.0.wrapping_add(1);
                }
            });
        }
        QueryShape::MutARefB => {
            world.query::<(&mut A, &B)>().for_each_chunk(|(a, _b)| {
                for v in a.iter_mut() {
                    v.0 = v.0.wrapping_add(1);
                }
            });
        }
        QueryShape::MutARefBC => {
            world
                .query::<(&mut A, &B, &C)>()
                .for_each_chunk(|(a, _, _)| {
                    for v in a.iter_mut() {
                        v.0 = v.0.wrapping_add(1);
                    }
                });
        }
        QueryShape::RefB => {
            world.query::<(&B,)>().for_each_chunk(|(_,)| {});
        }
        QueryShape::MutD => {
            world.query::<(&mut D,)>().for_each_chunk(|(d,)| {
                for v in d.iter_mut() {
                    v.0 += 1.0;
                }
            });
        }
    }
}

fuzz_target!(|ops: Vec<ReducerOp>| {
    let mut world = World::new();
    let mut live: Vec<minkowski::Entity> = Vec::new();

    for op in &ops {
        match op {
            ReducerOp::QueryMut(shape) => match shape {
                QueryShape::MutA => {
                    world.query::<(&mut A,)>().for_each(|(a,)| {
                        a.0 = a.0.wrapping_add(1);
                    });
                }
                QueryShape::MutARefB => {
                    world.query::<(&mut A, &B)>().for_each(|(a, _)| {
                        a.0 = a.0.wrapping_add(1);
                    });
                }
                QueryShape::MutARefBC => {
                    world.query::<(&mut A, &B, &C)>().for_each(|(a, _, _)| {
                        a.0 = a.0.wrapping_add(1);
                    });
                }
                QueryShape::MutD => {
                    world.query::<(&mut D,)>().for_each(|(d,)| {
                        d.0 += 1.0;
                    });
                }
                _ => {
                    run_query(&mut world, *shape);
                }
            },
            ReducerOp::QueryRef(shape) => {
                run_query(&mut world, *shape);
            }
            ReducerOp::ForEachChunk(shape) => {
                run_for_each_chunk(&mut world, *shape);
            }
            ReducerOp::SpawnMore(bundle) => {
                let e = spawn_bundle(&mut world, bundle);
                live.push(e);
            }
            ReducerOp::DespawnSome(idx) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live.swap_remove(i);
                    world.despawn(e);
                }
            }
        }

        assert_invariants(&world, &live);
    }
});
