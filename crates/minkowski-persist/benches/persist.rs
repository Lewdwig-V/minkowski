use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::World;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::recovery::LsmRecovery;
use minkowski_lsm::types::{SeqNo, SeqRange};
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

fn setup() -> (World, CodecRegistry) {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world).unwrap();
    codecs.register::<Vel>(&mut world).unwrap();
    for i in 0..1_000 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.0 },
        ));
    }
    (world, codecs)
}

fn bench_lsm_flush(c: &mut Criterion) {
    let (world, codecs) = setup();
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");

    c.bench_function("persist/lsm_flush_1k", |b| {
        b.iter(|| {
            let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
            flush_and_record(
                &world,
                SeqRange::new(SeqNo::from(0u64), SeqNo::from(100u64)).unwrap(),
                &mut manifest,
                &mut log,
                dir.path(),
                &codecs,
            )
            .unwrap();
        });
    });
}

fn bench_lsm_recover(c: &mut Criterion) {
    let (world, codecs) = setup();
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
    flush_and_record(
        &world,
        SeqRange::new(SeqNo::from(0u64), SeqNo::from(100u64)).unwrap(),
        &mut manifest,
        &mut log,
        dir.path(),
        &codecs,
    )
    .unwrap()
    .expect("flush");

    c.bench_function("persist/lsm_recover_1k", |b| {
        b.iter(|| {
            let (mut result, _, _) =
                LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
            assert_eq!(result.world.query::<(&Pos,)>().count(), 1_000);
        });
    });
}

fn bench_wal_append(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("bench.wal");
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world).unwrap();

    c.bench_function("persist/wal_append", |b| {
        let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
        b.iter(|| {
            let cs = minkowski::EnumChangeSet::new();
            wal.append(&cs, &codecs).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_lsm_flush,
    bench_lsm_recover,
    bench_wal_append
);
criterion_main!(benches);
