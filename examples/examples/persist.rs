//! Persistence — WAL + snapshot save/load/recovery + persistent indexes.
//!
//! Demonstrates the full persistence lifecycle:
//! 1. Create a world and spawn entities
//! 2. Save a snapshot
//! 3. Apply mutations via a durable QueryWriter reducer (WAL-backed)
//! 4. Recover from snapshot + WAL replay
//! 5. Persistent index: save, load, catch-up vs full rebuild
//!
//! Run: cargo run -p minkowski-examples --example persist --release

use minkowski::{BTreeIndex, Optimistic, QueryWriter, ReducerRegistry, SpatialIndex, World};
use minkowski_persist::{
    CodecRegistry, Durable, PersistentIndex, Snapshot, Wal, WalConfig, load_btree_index,
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

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Health(u32);

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Archive, Serialize, Deserialize,
)]
#[repr(C)]
struct Score(u32);

fn main() {
    let dir = std::env::temp_dir().join("minkowski-persist-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_dir = dir.join("example.wal");
    let snap_path = dir.join("example.snap");

    // Clean up from previous runs
    let _ = std::fs::remove_dir_all(&wal_dir);
    let _ = std::fs::remove_file(&snap_path);

    // -- Phase 1: Create world with multiple archetypes --
    println!("Phase 1: Creating world with 100 entities across 3 archetypes...");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world).unwrap();
    codecs.register_as::<Vel>("vel", &mut world).unwrap();
    codecs.register_as::<Health>("health", &mut world).unwrap();
    codecs.register_as::<Score>("score", &mut world).unwrap();

    // Archetype 1: (Pos, Vel) — moving entities
    for i in 0..50 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }
    // Archetype 2: (Pos, Health) — static entities with health
    for i in 50..80 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 100.0,
            },
            Health(100),
        ));
    }
    // Archetype 3: (Pos, Vel, Health, Score) — full-featured entities
    for i in 80..100 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 50.0,
            },
            Vel { dx: 0.5, dy: -0.5 },
            Health(200),
            Score(i),
        ));
    }
    // Sparse: add Score to some Archetype-1 entities
    let moving: Vec<_> = world
        .query::<(minkowski::Entity, &Vel)>()
        .filter(|(_, v)| v.dx > 0.9)
        .map(|(e, _)| e)
        .take(10)
        .collect();
    for (i, e) in moving.iter().enumerate() {
        world.insert_sparse::<Score>(*e, Score(1000 + i as u32));
    }

    // -- Phase 2: Snapshot --
    let snap = Snapshot::new();
    let wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
    let header = snap
        .save(&snap_path, &world, &codecs, wal.next_seq())
        .unwrap();
    println!(
        "Phase 2: Snapshot saved — {} entities, {} archetypes, WAL seq {}",
        header.entity_count, header.archetype_count, header.wal_seq
    );

    // -- Phase 3: Durable transactions via QueryWriter reducer --
    println!("Phase 3: Simulating 10 frames (durable QueryWriter reducer)...");
    let strategy = Optimistic::new(&world);
    let durable = Durable::new(strategy, wal, codecs);

    let mut registry = ReducerRegistry::new();
    let writer_id = registry
        .register_query_writer::<(&mut Pos, &Vel), f32, _>(
            &mut world,
            "apply_velocity",
            |mut query: QueryWriter<'_, (&mut Pos, &Vel)>, dt: f32| {
                query.for_each(|(mut pos, vel)| {
                    pos.modify(|p| {
                        p.x += vel.dx * dt;
                        p.y += vel.dy * dt;
                    });
                });
            },
        )
        .unwrap();

    for frame in 0..10 {
        registry
            .call(&durable, &mut world, writer_id, 1.0f32)
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
    load_codecs
        .register_as::<Pos>("pos", &mut load_world_tmp)
        .unwrap();
    load_codecs
        .register_as::<Vel>("vel", &mut load_world_tmp)
        .unwrap();
    load_codecs
        .register_as::<Health>("health", &mut load_world_tmp)
        .unwrap();
    load_codecs
        .register_as::<Score>("score", &mut load_world_tmp)
        .unwrap();

    let (mut recovered, snap_seq) = snap.load(&snap_path, &load_codecs).unwrap();
    let mut replay_wal = Wal::open(&wal_dir, &load_codecs, WalConfig::default()).unwrap();
    let last_seq = replay_wal
        .replay_from(snap_seq, &mut recovered, &load_codecs)
        .unwrap();
    println!(
        "  Loaded snapshot (seq {}), replayed WAL to seq {}",
        snap_seq, last_seq
    );

    let pos_count = recovered.query::<(&Pos,)>().count();
    let vel_count = recovered.query::<(&Vel,)>().count();
    let health_count = recovered.query::<(&Health,)>().count();
    println!(
        "  Recovered: {} with Pos, {} with Vel, {} with Health",
        pos_count, vel_count, health_count
    );
    let score_id = recovered.component_id::<Score>().unwrap();
    let sparse_scores: Vec<_> = recovered.iter_sparse::<Score>(score_id).unwrap().collect();
    println!("  Sparse scores recovered: {}", sparse_scores.len());

    // -- Phase 5: Persistent index — save, load, catch-up --
    println!("Phase 5: Persistent index recovery...");
    let idx_path = dir.join("score.idx");

    // Full rebuild (baseline)
    let t0 = std::time::Instant::now();
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut recovered);
    let rebuild_time = t0.elapsed();
    let idx_len = idx.len();
    println!("  rebuild():   {} entries in {:?}", idx_len, rebuild_time);

    // Save index and capture the tick at save time
    idx.save(&idx_path).unwrap();
    let save_tick = recovered.change_tick();
    println!("  Saved index to {}", idx_path.display());

    // Simulate: mutate world after index save (WAL tail equivalent)
    for i in 0..5 {
        recovered.spawn((Pos { x: 999.0, y: 999.0 }, Score(9000 + i)));
    }

    // Load + update (recovery path) — use save-time tick so update()
    // catches up with the post-save mutations
    let t1 = std::time::Instant::now();
    let mut loaded_idx = load_btree_index::<Score>(&idx_path, save_tick).unwrap();
    loaded_idx.update(&mut recovered);
    let load_time = t1.elapsed();
    println!(
        "  load+update: {} entries in {:?}",
        loaded_idx.len(),
        load_time
    );
    println!(
        "  Index recovery is {:.0}x faster than rebuild",
        if load_time.as_nanos() > 0 {
            rebuild_time.as_nanos() as f64 / load_time.as_nanos() as f64
        } else {
            f64::INFINITY
        }
    );

    // Verify correctness: loaded index matches world
    let fresh_count = recovered.query::<(minkowski::Entity, &Score)>().count();
    assert_eq!(
        loaded_idx.len(),
        fresh_count,
        "loaded index entity count should match world"
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone.");
}
