//! Replication — pull-based WAL cursor for read replicas.
//!
//! Simulates a network replication flow using channels as the wire:
//! - `source_side` spawns entities, takes a snapshot, writes post-snapshot
//!   mutations to WAL, then sends snapshot bytes + WAL batch bytes over a channel.
//! - `replica_side` receives the bytes, reconstructs a World from the snapshot,
//!   applies the WAL batch, and returns the replica World.
//! - The two sides share no World state — only serialized bytes cross the boundary.
//!
//! Run: cargo run -p minkowski-examples --example replicate --release

use minkowski::{EnumChangeSet, World};
use minkowski_persist::{
    CodecRegistry, ReplicationBatch, Snapshot, Wal, WalConfig, WalCursor, apply_batch,
};
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::mpsc;

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

/// Messages sent from source to replica over the "network" (channel).
enum WireMessage {
    /// Full snapshot bytes — sent once at the start.
    Snapshot(Vec<u8>),
    /// Incremental WAL batch bytes — sent for each batch of mutations.
    WalBatch(Vec<u8>),
}

/// Source side: owns the authoritative World and WAL.
/// Sends snapshot + WAL batches over the channel.
fn source_side(tx: &mpsc::Sender<WireMessage>) {
    let dir = std::env::temp_dir().join("minkowski-replicate-source");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_dir = dir.join("source.wal");
    let _ = std::fs::remove_dir_all(&wal_dir);

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Vel>("vel", &mut world);

    // Spawn initial entities
    for i in 0..20 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }
    println!("[source] Spawned {} entities", world.entity_count());

    // Take snapshot and send bytes over the wire
    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
    let snap = Snapshot::new();
    let (header, snap_bytes) = snap.save_to_bytes(&world, &codecs, wal.next_seq()).unwrap();
    println!(
        "[source] Snapshot: {} entities, {} bytes, WAL seq {}",
        header.entity_count,
        snap_bytes.len(),
        header.wal_seq
    );
    tx.send(WireMessage::Snapshot(snap_bytes)).unwrap();

    // Post-snapshot mutations via WAL
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
        cs.apply(&mut world).expect("replicate apply");
    }
    println!(
        "[source] After mutations: {} entities, WAL seq {}",
        world.entity_count(),
        wal.next_seq()
    );

    drop(wal);

    // Read WAL batch from cursor and send over the wire
    let mut cursor = WalCursor::open(&wal_dir, header.wal_seq).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    let batch_bytes = batch.to_bytes().unwrap();
    println!(
        "[source] WAL batch: {} records, {} bytes",
        batch.records.len(),
        batch_bytes.len()
    );
    tx.send(WireMessage::WalBatch(batch_bytes)).unwrap();

    let _ = std::fs::remove_dir_all(&dir);
    println!("[source] Done — {} entities total", world.entity_count());
}

/// Replica side: reconstructs World from bytes received over the channel.
/// Has its own CodecRegistry — no shared state with source.
fn replica_side(rx: &mpsc::Receiver<WireMessage>) -> World {
    // Replica registers codecs independently (could be different order)
    let mut codecs = CodecRegistry::new();
    let mut tmp = World::new();
    codecs.register_as::<Vel>("vel", &mut tmp);
    codecs.register_as::<Pos>("pos", &mut tmp);
    drop(tmp);

    // Receive and load snapshot
    let WireMessage::Snapshot(snap_bytes) = rx.recv().unwrap() else {
        panic!("expected snapshot")
    };
    println!("[replica] Received snapshot: {} bytes", snap_bytes.len());

    let snap = Snapshot::new();
    let (mut world, snap_seq) = snap.load_from_bytes(&snap_bytes, &codecs).unwrap();
    println!(
        "[replica] Loaded snapshot: {} entities at seq {}",
        world.entity_count(),
        snap_seq
    );

    // Receive and apply WAL batch
    let WireMessage::WalBatch(batch_bytes) = rx.recv().unwrap() else {
        panic!("expected WAL batch")
    };
    println!("[replica] Received WAL batch: {} bytes", batch_bytes.len());

    let batch = ReplicationBatch::from_bytes(&batch_bytes).unwrap();
    let last_seq = apply_batch(&batch, &mut world, &codecs).unwrap();
    println!(
        "[replica] Applied {} records up to seq {:?} — {} entities",
        batch.records.len(),
        last_seq,
        world.entity_count()
    );

    world
}

fn main() {
    let (tx, rx) = mpsc::channel();

    // Run source and replica — channel is the "network"
    source_side(&tx);
    let replica = replica_side(&rx);

    // Verify convergence
    println!("\n=== Convergence check ===");
    println!("Replica entities: {}", replica.entity_count());
    assert_eq!(
        replica.entity_count(),
        30,
        "replica should have 20 (snapshot) + 10 (WAL) = 30 entities"
    );
    println!("Source and replica converged.");
}
