//! Persistence — WAL + LSM flush/recovery + persistent indexes.
//!
//! Run: cargo run -p minkowski-examples --example persist --release

use minkowski::{BTreeIndex, Optimistic, QueryWriter, ReducerRegistry, SpatialIndex, World};
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::types::{SeqNo, SeqRange};
use minkowski_persist::{
    AutoCheckpoint, CodecRegistry, Durable, PersistentIndex, Wal, WalConfig, load_btree_index,
    recover_world,
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
    let lsm_dir = dir.join("lsm");
    let manifest_log = lsm_dir.join("manifest.log");
    std::fs::create_dir_all(&lsm_dir).unwrap();

    let _ = std::fs::remove_dir_all(&wal_dir);
    let _ = std::fs::remove_dir_all(&lsm_dir);
    std::fs::create_dir_all(&lsm_dir).unwrap();

    println!("Phase 1: Creating world with 100 entities across 3 archetypes...");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world).unwrap();
    codecs.register_as::<Vel>("vel", &mut world).unwrap();
    codecs.register_as::<Health>("health", &mut world).unwrap();
    codecs.register_as::<Score>("score", &mut world).unwrap();

    for i in 0..50 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }
    for i in 50..80 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 100.0,
            },
            Health(100),
        ));
    }
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

    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
    let flush_seq = wal.next_seq();
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&manifest_log).unwrap();
    flush_and_record(
        &world,
        SeqRange::new(SeqNo::from(0u64), SeqNo::from(flush_seq)).unwrap(),
        &mut manifest,
        &mut log,
        &lsm_dir,
    )
    .unwrap()
    .expect("baseline flush");
    wal.acknowledge_flush(flush_seq).unwrap();

    println!(
        "Phase 2: LSM baseline flushed ({} entities)",
        world.query::<(&Pos,)>().count()
    );

    println!("Phase 3: Simulating 10 frames (durable QueryWriter reducer)...");
    let strategy = Optimistic::new(&world);
    let checkpoint = AutoCheckpoint::new(&lsm_dir);
    let durable = Durable::with_checkpoint(strategy, wal, codecs, checkpoint);

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

    for _frame in 0..10 {
        registry
            .call(&durable, &mut world, writer_id, 1.0f32)
            .unwrap();
    }
    println!("  WAL seq after simulation: {}", durable.wal_seq());

    println!("Phase 4: Recovering from LSM + WAL...");
    let mut load_codecs = CodecRegistry::new();
    let mut tmp = World::new();
    load_codecs.register_as::<Pos>("pos", &mut tmp).unwrap();
    load_codecs.register_as::<Vel>("vel", &mut tmp).unwrap();
    load_codecs
        .register_as::<Health>("health", &mut tmp)
        .unwrap();
    load_codecs.register_as::<Score>("score", &mut tmp).unwrap();

    let mut replay_wal = Wal::open(&wal_dir, &load_codecs, WalConfig::default()).unwrap();
    let mut recovered =
        recover_world(&lsm_dir, &manifest_log, &mut replay_wal, &load_codecs).unwrap();

    println!(
        "  Recovered: {} Pos, {} Vel, {} Health",
        recovered.query::<(&Pos,)>().count(),
        recovered.query::<(&Vel,)>().count(),
        recovered.query::<(&Health,)>().count(),
    );

    println!("Phase 5: Persistent index recovery...");
    let idx_path = dir.join("score.idx");
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut recovered);
    idx.save(&idx_path).unwrap();
    let save_tick = recovered.change_tick();

    recovered.spawn((Pos { x: 999.0, y: 0.0 }, Score(42)));
    let mut loaded_idx = load_btree_index::<Score>(&idx_path, save_tick).unwrap();
    loaded_idx.update(&mut recovered);
    println!("  Index entries after catch-up: {}", loaded_idx.len());

    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone.");
}
