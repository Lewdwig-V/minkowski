//! Observability — capture, diff, and display engine metrics.
//!
//! Demonstrates MetricsSnapshot capture at two points in time, diffing
//! to compute entity churn, tick velocity, and archetype changes.
//!
//! Run: cargo run -p minkowski-examples --example observe --release

use minkowski::{EnumChangeSet, World};
use minkowski_observe::{MetricsDiff, MetricsSnapshot};
use minkowski_persist::{CodecRegistry, Wal, WalConfig};
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
    let dir = std::env::temp_dir().join("minkowski-observe-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_dir = dir.join("observe.wal");
    let _ = std::fs::remove_dir_all(&wal_dir);

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Vel>("vel", &mut world);

    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

    // Phase 1: spawn 100 entities with Pos + Vel
    for i in 0..100 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Vel { dx: 1.0, dy: 0.5 },
            ),
        );
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let snap1 = MetricsSnapshot::capture(&world, Some(&wal));
    println!("=== Snapshot 1 ===");
    println!("{snap1}");

    // Phase 2: churn — despawn 20, spawn 50 new (Pos-only archetype)
    let entities: Vec<_> = world
        .query::<(minkowski::Entity,)>()
        .take(20)
        .map(|e| e.0)
        .collect();
    for e in entities {
        world.despawn(e);
    }

    for i in 100..150 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (Pos {
                x: i as f32,
                y: 10.0,
            },),
        );
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let snap2 = MetricsSnapshot::capture(&world, Some(&wal));
    println!("=== Snapshot 2 ===");
    println!("{snap2}");

    let diff = MetricsDiff::compute(&snap1, &snap2);
    println!("=== Diff ===");
    println!("{diff}");

    let _ = std::fs::remove_dir_all(&dir);
    println!("Done.");
}
