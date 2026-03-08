#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::{EnumChangeSet, World};
use minkowski_persist::{CodecRegistry, Wal, WalConfig};

fn make_codecs(world: &mut World) -> CodecRegistry {
    let mut codecs = CodecRegistry::new();
    codecs.register::<A>(world);
    codecs.register::<B>(world);
    codecs.register::<C>(world);
    codecs.register::<D>(world);
    codecs
}

#[derive(Clone, Debug, Arbitrary)]
enum WalOp {
    SpawnA(A),
    SpawnAB(A, B),
    InsertB(u8, B),
    InsertD(u8, D),
}

#[derive(Clone, Debug, Arbitrary)]
struct WalRoundTrip {
    ops: Vec<WalOp>,
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mode = data[0];
    let rest = &data[1..];

    let dir = tempfile::tempdir().unwrap();

    match mode {
        // Mode 0: write raw bytes as a WAL file, try to open + replay
        0 => {
            let wal_dir = dir.path().join("raw");
            std::fs::create_dir_all(&wal_dir).unwrap();
            let seg_path = wal_dir.join("00000000.wal");
            std::fs::write(&seg_path, rest).unwrap();

            let mut world = World::new();
            let codecs = make_codecs(&mut world);
            if let Ok(mut wal) = Wal::open(&wal_dir, &codecs, WalConfig::default()) {
                let _ = wal.replay_from(0, &mut world, &codecs);
            }
        }
        // Mode 1+: round-trip — build valid changesets, append, replay, verify
        _ => {
            let Ok(rt) = arbitrary::Unstructured::new(rest).arbitrary::<WalRoundTrip>() else {
                return;
            };
            if rt.ops.is_empty() {
                return;
            }

            let wal_dir = dir.path().join("rt");
            std::fs::create_dir_all(&wal_dir).unwrap();

            let mut world = World::new();
            let codecs = make_codecs(&mut world);
            let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

            let mut live: Vec<minkowski::Entity> = Vec::new();

            for op in &rt.ops {
                let mut cs = EnumChangeSet::new();
                match op {
                    WalOp::SpawnA(a) => {
                        let e = world.alloc_entity();
                        cs.spawn_bundle(&mut world, e, (*a,));
                        wal.append(&cs, &codecs).unwrap();
                        cs.apply(&mut world);
                        live.push(e);
                    }
                    WalOp::SpawnAB(a, b) => {
                        let e = world.alloc_entity();
                        cs.spawn_bundle(&mut world, e, (*a, *b));
                        wal.append(&cs, &codecs).unwrap();
                        cs.apply(&mut world);
                        live.push(e);
                    }
                    WalOp::InsertB(idx, b) => {
                        if !live.is_empty() {
                            let i = *idx as usize % live.len();
                            let e = live[i];
                            cs.insert::<B>(&mut world, e, *b);
                            wal.append(&cs, &codecs).unwrap();
                            cs.apply(&mut world);
                        }
                    }
                    WalOp::InsertD(idx, d) => {
                        if !live.is_empty() {
                            let i = *idx as usize % live.len();
                            let e = live[i];
                            cs.insert::<D>(&mut world, e, *d);
                            wal.append(&cs, &codecs).unwrap();
                            cs.apply(&mut world);
                        }
                    }
                }
            }

            // Replay into fresh world
            let mut world2 = World::new();
            let codecs2 = make_codecs(&mut world2);
            let mut wal2 = Wal::open(&wal_dir, &codecs2, WalConfig::default()).unwrap();
            let _ = wal2.replay_from(0, &mut world2, &codecs2);

            assert_eq!(world.entity_count(), world2.entity_count());
        }
    }
});
