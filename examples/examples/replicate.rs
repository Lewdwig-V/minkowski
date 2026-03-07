//! Replication — pull-based WAL cursor for read replicas.
//!
//! Demonstrates the full replication lifecycle:
//! 1. Create a source world and spawn entities
//! 2. Take a snapshot
//! 3. Write post-snapshot mutations to WAL
//! 4. Open a WalCursor at the snapshot seq
//! 5. Pull a ReplicationBatch and apply to a replica world
//! 6. Verify source and replica converge
//!
//! Run: cargo run -p minkowski-examples --example replicate --release

use minkowski::{EnumChangeSet, World};
use minkowski_persist::{
    apply_batch, CodecRegistry, ReplicationBatch, Snapshot, Wal, WalConfig, WalCursor,
};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Vel {
    dx: f32,
    dy: f32,
}

fn main() {
    let dir = std::env::temp_dir().join("minkowski-replicate-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_dir = dir.join("source.wal");
    let snap_path = dir.join("source.snap");

    // Clean up from previous runs
    let _ = std::fs::remove_dir_all(&wal_dir);
    let _ = std::fs::remove_file(&snap_path);

    // -- Phase 1: Source world --
    println!("Phase 1: Creating source world...");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Vel>("vel", &mut world);

    for i in 0..20 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }
    println!("  Spawned {} entities", world.entity_count());

    // -- Phase 2: Snapshot --
    println!("Phase 2: Taking snapshot...");
    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
    let snap = Snapshot::new();
    let header = snap
        .save(&snap_path, &world, &codecs, wal.next_seq())
        .unwrap();
    println!(
        "  Snapshot: {} entities, WAL seq {}",
        header.entity_count, header.wal_seq
    );

    // -- Phase 3: Post-snapshot mutations --
    println!("Phase 3: Writing post-snapshot mutations...");
    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (
                Pos {
                    x: 100.0 + i as f32,
                    y: 100.0,
                },
                Vel { dx: -1.0, dy: -0.5 },
            ),
        );
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }
    println!(
        "  Source now has {} entities, WAL seq {}",
        world.entity_count(),
        wal.next_seq()
    );

    drop(wal);

    // -- Phase 4: Replica via snapshot + cursor --
    println!("Phase 4: Replicating...");
    let mut replica_codecs = CodecRegistry::new();
    let mut tmp = World::new();
    replica_codecs.register_as::<Pos>("pos", &mut tmp);
    replica_codecs.register_as::<Vel>("vel", &mut tmp);

    let (mut replica, snap_seq) = snap.load(&snap_path, &replica_codecs).unwrap();
    println!(
        "  Loaded snapshot: {} entities at seq {}",
        replica.query::<(&Pos,)>().count(),
        snap_seq
    );

    let mut cursor = WalCursor::open(&wal_dir, snap_seq).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    println!(
        "  Pulled batch: {} records, schema has {} components",
        batch.records.len(),
        batch.schema.components.len()
    );

    // Demonstrate wire format round-trip
    let wire_bytes = batch.to_bytes().unwrap();
    println!("  Wire format: {} bytes", wire_bytes.len());
    let batch = ReplicationBatch::from_bytes(&wire_bytes).unwrap();

    let last_seq = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();
    println!("  Applied up to seq {:?}", last_seq);

    // -- Phase 5: Verify convergence --
    println!("Phase 5: Verifying convergence...");
    let source_count = world.entity_count();
    let replica_count = replica.query::<(&Pos,)>().count();
    println!(
        "  Source: {} entities, Replica: {} entities",
        source_count, replica_count
    );
    assert_eq!(source_count, replica_count, "replica should match source");

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone. Source and replica converged.");
}
