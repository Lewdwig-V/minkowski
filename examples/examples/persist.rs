//! Persistence — WAL + snapshot save/load/recovery.
//!
//! Demonstrates the full persistence lifecycle:
//! 1. Create a world and spawn entities
//! 2. Save a snapshot
//! 3. Apply mutations via durable transactions (WAL-backed)
//! 4. Recover from snapshot + WAL replay
//!
//! Run: cargo run -p minkowski-examples --example persist --release

use minkowski::{Access, Optimistic, Transact, World};
use minkowski_persist::{Bincode, CodecRegistry, Durable, Snapshot, Wal};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Vel {
    dx: f32,
    dy: f32,
}

fn main() {
    let dir = std::env::temp_dir().join("minkowski-persist-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("example.wal");
    let snap_path = dir.join("example.snap");

    // Clean up from previous runs
    let _ = std::fs::remove_file(&wal_path);
    let _ = std::fs::remove_file(&snap_path);

    // -- Phase 1: Create world --
    println!("Phase 1: Creating world with 100 entities...");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);
    codecs.register::<Vel>(&mut world);

    for i in 0..100 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }

    // -- Phase 2: Snapshot (before creating Durable, which takes ownership of codecs) --
    let snap = Snapshot::new(Bincode);
    let wal = Wal::create(&wal_path, Bincode).unwrap();
    let header = snap
        .save(&snap_path, &world, &codecs, wal.next_seq())
        .unwrap();
    println!(
        "Phase 2: Snapshot saved — {} entities, {} archetypes, WAL seq {}",
        header.entity_count, header.archetype_count, header.wal_seq
    );

    // -- Phase 3: Durable transactions --
    println!("Phase 3: Simulating 10 frames (durable transactions)...");
    let strategy = Optimistic::new(&world);
    let durable = Durable::new(strategy, wal, codecs);

    let access = Access::of::<(minkowski::Entity, &Pos, &Vel, &mut Pos)>(&mut world);
    for frame in 0..10 {
        durable
            .transact(&mut world, &access, |tx, world| {
                let updates: Vec<_> = tx
                    .query::<(minkowski::Entity, &Pos, &Vel)>(world)
                    .map(|(e, p, v)| {
                        (
                            e,
                            Pos {
                                x: p.x + v.dx,
                                y: p.y + v.dy,
                            },
                        )
                    })
                    .collect();

                for (e, new_pos) in updates {
                    tx.write::<Pos>(world, e, new_pos);
                }
            })
            .unwrap();

        if frame == 0 || frame == 9 {
            let sample = world.query::<(&Pos,)>().next().unwrap();
            println!(
                "  frame {}: first entity at ({:.1}, {:.1})",
                frame, sample.0.x, sample.0.y
            );
        }
    }
    println!("  WAL has {} records", durable.wal_seq());

    // -- Phase 4: Recovery --
    println!("Phase 4: Recovering from snapshot + WAL...");
    let mut load_codecs = CodecRegistry::new();
    let mut load_world_tmp = World::new();
    load_codecs.register::<Pos>(&mut load_world_tmp);
    load_codecs.register::<Vel>(&mut load_world_tmp);

    let (mut recovered, snap_seq) = snap.load(&snap_path, &load_codecs).unwrap();
    let mut replay_wal = Wal::open(&wal_path, Bincode).unwrap();
    let last_seq = replay_wal
        .replay_from(snap_seq, &mut recovered, &load_codecs)
        .unwrap();
    println!(
        "  Loaded snapshot (seq {}), replayed WAL to seq {}",
        snap_seq, last_seq
    );

    let count = recovered.query::<(&Pos,)>().count();
    let sample = recovered.query::<(&Pos,)>().next().unwrap();
    println!(
        "  Recovered world: {} entities, first at ({:.1}, {:.1})",
        count, sample.0.x, sample.0.y
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone.");
}
