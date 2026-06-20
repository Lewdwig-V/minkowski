use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::EnumChangeSet;
use minkowski_bench::{Position, register_codecs, spawn_world};
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::types::{SeqNo, SeqRange};
use minkowski_persist::{CodecRegistry, Wal, WalConfig, recover_world};

fn serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize");

    group.bench_function("lsm_flush", |b| {
        let world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        let mut w = world;
        register_codecs(&mut codecs, &mut w);
        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        std::fs::create_dir_all(&lsm_dir).unwrap();
        let log_path = lsm_dir.join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        b.iter(|| {
            flush_and_record(
                &w,
                SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
                &mut manifest,
                &mut log,
                &lsm_dir,
                &codecs,
            )
            .unwrap();
        });
    });

    group.bench_function("lsm_recover", |b| {
        let mut world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        register_codecs(&mut codecs, &mut world);
        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&lsm_dir).unwrap();
        let log_path = lsm_dir.join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        b.iter(|| {
            let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
            let _w = recover_world(&lsm_dir, &log_path, &mut wal, &codecs).unwrap();
        });
    });

    group.bench_function("wal_append", |b| {
        let mut world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        register_codecs(&mut codecs, &mut world);
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();

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

    group.bench_function("wal_replay", |b| {
        let mut world = spawn_world(1_000);
        let mut codecs = CodecRegistry::new();
        register_codecs(&mut codecs, &mut world);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&lsm_dir).unwrap();
        let log_path = lsm_dir.join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

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
                let mut wal = Wal::open(&wal_dir, &codecs, WalConfig::default()).unwrap();
                let w = recover_world(&lsm_dir, &log_path, &mut wal, &codecs).unwrap();
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
