#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::World;
use minkowski_persist::{CodecRegistry, Snapshot};

fn make_codecs(world: &mut World) -> CodecRegistry {
    let mut codecs = CodecRegistry::new();
    codecs.register::<A>(world).unwrap();
    codecs.register::<B>(world).unwrap();
    codecs.register::<C>(world).unwrap();
    codecs.register::<D>(world).unwrap();
    codecs
}

#[derive(Clone, Debug, Arbitrary)]
struct RoundTripOps {
    spawns: Vec<BundleKind>,
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mode = data[0];
    let rest = &data[1..];

    match mode {
        // Mode 0: raw bytes — must not panic
        0 => {
            let mut world = World::new();
            let codecs = make_codecs(&mut world);
            let snap = Snapshot::new();
            let _ = snap.load_from_bytes(rest, &codecs);
        }
        // Mode 1+: round-trip — build valid world, save, load, verify
        _ => {
            let Ok(ops) = arbitrary::Unstructured::new(rest).arbitrary::<RoundTripOps>() else {
                return;
            };
            if ops.spawns.is_empty() {
                return;
            }

            let mut world = World::new();
            let codecs = make_codecs(&mut world);

            for bundle in &ops.spawns {
                spawn_bundle(&mut world, bundle);
            }

            let snap = Snapshot::new();
            let Ok((header, bytes)) = snap.save_to_bytes(&world, &codecs, 0) else {
                return;
            };
            assert_eq!(header.entity_count, ops.spawns.len());

            // Load into fresh world with fresh codecs
            let mut world2 = World::new();
            let codecs2 = make_codecs(&mut world2);
            let Ok((mut world2, seq)) = snap.load_from_bytes(&bytes, &codecs2) else {
                panic!("round-trip load failed on bytes we just saved");
            };
            assert_eq!(seq, 0);
            assert_eq!(world.entity_count(), world2.entity_count());

            // Verify component A values match
            let mut orig: Vec<u32> = world.query::<(&A,)>().map(|(a,)| a.0).collect();
            let mut loaded: Vec<u32> = world2.query::<(&A,)>().map(|(a,)| a.0).collect();
            orig.sort();
            loaded.sort();
            assert_eq!(orig, loaded);
        }
    }
});
