use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::World;
use minkowski_persist::{CodecRegistry, Snapshot, Wal, WalConfig};
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

fn bench_snapshot_save(c: &mut Criterion) {
    let (world, codecs) = setup();
    let dir = tempfile::tempdir().unwrap();

    c.bench_function("persist/snapshot_save_1k", |b| {
        let path = dir.path().join("bench.snap");
        b.iter(|| {
            let snap = Snapshot::new();
            snap.save(&path, &world, &codecs, 0).unwrap();
        });
    });
}

fn bench_snapshot_load(c: &mut Criterion) {
    let (world, codecs) = setup();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.snap");
    let snap = Snapshot::new();
    snap.save(&path, &world, &codecs, 0).unwrap();

    c.bench_function("persist/snapshot_load_1k", |b| {
        b.iter(|| {
            let snap = Snapshot::new();
            let (_world, _seq) = snap.load(&path, &codecs).unwrap();
        });
    });
}

fn bench_wal_append(c: &mut Criterion) {
    let (mut world, codecs) = setup();
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("bench.wal");
    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

    // Single-mutation changeset — WAL append cost scales with changeset size,
    // not world size, so this isolates serialization + I/O overhead.
    let entity = world.spawn((Pos { x: 99.0, y: 99.0 }, Vel { dx: 0.0, dy: 0.0 }));
    let mut cs = minkowski::EnumChangeSet::new();
    cs.insert::<Pos>(&mut world, entity, Pos { x: 100.0, y: 100.0 });

    c.bench_function("persist/wal_append", |b| {
        b.iter(|| {
            wal.append(&cs, &codecs).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_snapshot_save,
    bench_snapshot_load,
    bench_wal_append,
);
criterion_main!(benches);
