use criterion::{criterion_group, criterion_main, Criterion};
use minkowski::EnumChangeSet;
use minkowski_bench::{register_codecs, spawn_world, Position};
use minkowski_persist::{CodecRegistry, Snapshot, Wal, WalConfig};

fn serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize");

    // -- snapshot_save --
    group.bench_function("snapshot_save", |b| {
        let world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        let mut w = world;
        register_codecs(&mut codecs, &mut w);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.snap");

        b.iter(|| {
            Snapshot::new().save(&path, &w, &codecs, 0).unwrap();
        });
    });

    // -- snapshot_load --
    group.bench_function("snapshot_load", |b| {
        let mut world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        register_codecs(&mut codecs, &mut world);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.snap");
        Snapshot::new().save(&path, &world, &codecs, 0).unwrap();

        b.iter(|| {
            let (_w, _seq) = Snapshot::new().load(&path, &codecs).unwrap();
        });
    });

    // -- wal_append --
    group.bench_function("wal_append", |b| {
        let mut world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        register_codecs(&mut codecs, &mut world);
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();

        // Grab the first entity for insert mutations
        let entity = world.query::<minkowski::Entity>().next().unwrap();

        b.iter(|| {
            let mut cs = EnumChangeSet::new();
            cs.insert(
                &mut world,
                entity,
                Position {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
            );
            wal.append(&cs, &codecs).unwrap();
        });
    });

    // -- wal_replay --
    group.bench_function("wal_replay", |b| {
        // One-time setup: create world, save snapshot, populate WAL with 1K mutations
        let mut world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        register_codecs(&mut codecs, &mut world);

        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("bench.snap");
        Snapshot::new()
            .save(&snap_path, &world, &codecs, 0)
            .unwrap();

        // Create WAL and write 1K insert mutations
        let wal_dir = dir.path().join("wal");
        let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
        let entities: Vec<_> = world.query::<minkowski::Entity>().collect();
        for &entity in &entities {
            let mut cs = EnumChangeSet::new();
            cs.insert(
                &mut world,
                entity,
                Position {
                    x: 9.0,
                    y: 8.0,
                    z: 7.0,
                },
            );
            wal.append(&cs, &codecs).unwrap();
        }
        drop(wal);

        b.iter_batched(
            || {
                // Setup: load snapshot to get fresh world, open WAL
                let (w, _seq) = Snapshot::new().load(&snap_path, &codecs).unwrap();
                let wal = Wal::open(&wal_dir, &codecs, WalConfig::default()).unwrap();
                (w, wal)
            },
            |(mut w, mut wal)| {
                wal.replay(&mut w, &codecs).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, serialize);
criterion_main!(benches);
