//! Replication — pull-based WAL cursor for read replicas.
//!
//! Run: cargo run -p minkowski-examples --example replicate --release

use minkowski::{EnumChangeSet, World};
use minkowski_lsm::manifest::SortedRunMeta;
use minkowski_lsm::manifest_log::{ManifestEntry, ManifestLog};
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::reader::SortedRunReader;
use minkowski_lsm::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};
use minkowski_persist::{
    CodecRegistry, ReplicationBatch, Wal, WalConfig, WalCursor, apply_batch, recover_world,
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

enum WireMessage {
    /// LSM baseline: sorted run bytes + WAL sequence covered by the flush.
    LsmBaseline {
        run_name: String,
        run: Vec<u8>,
        flush_seq: u64,
    },
    WalBatch(Vec<u8>),
}

fn source_side(tx: &mpsc::Sender<WireMessage>) {
    let dir = std::env::temp_dir().join("minkowski-replicate-source");
    let lsm_dir = dir.join("lsm");
    let wal_dir = dir.join("source.wal");
    let manifest_log = lsm_dir.join("manifest.log");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&lsm_dir).unwrap();

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world).unwrap();
    codecs.register_as::<Vel>("vel", &mut world).unwrap();

    for i in 0..20 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel { dx: 1.0, dy: 0.5 },
        ));
    }

    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
    let flush_seq = wal.next_seq();
    let (mut manifest, mut log) = ManifestLog::recover::<4>(&manifest_log).unwrap();
    let run_path = flush_and_record(
        &world,
        SeqRange::new(SeqNo::from(0u64), SeqNo::from(flush_seq)).unwrap(),
        &mut manifest,
        &mut log,
        &lsm_dir,
    )
    .unwrap()
    .expect("flush");

    let run_name = run_path.file_name().unwrap().to_string_lossy().into_owned();
    let run_bytes = std::fs::read(&run_path).unwrap();
    tx.send(WireMessage::LsmBaseline {
        run_name,
        run: run_bytes,
        flush_seq,
    })
    .unwrap();

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
        )
        .unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();
    }

    drop(wal);

    let mut cursor = WalCursor::open(&wal_dir, flush_seq).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    tx.send(WireMessage::WalBatch(batch.to_bytes().unwrap()))
        .unwrap();

    let _ = std::fs::remove_dir_all(&dir);
}

fn install_baseline(
    lsm_dir: &std::path::Path,
    manifest_log: &std::path::Path,
    run_name: &str,
    run: &[u8],
    flush_seq: u64,
) {
    let run_path = lsm_dir.join(run_name);
    std::fs::write(&run_path, run).unwrap();
    let reader = SortedRunReader::open(&run_path).unwrap();
    let file_size = std::fs::metadata(&run_path).unwrap().len();
    let meta = SortedRunMeta::new(
        run_path,
        reader.sequence_range(),
        reader.archetype_ids(),
        PageCount::new(reader.page_count()).expect("page count"),
        SizeBytes::new(file_size),
    )
    .expect("run meta");
    let (_, mut log) = ManifestLog::recover::<4>(manifest_log).unwrap();
    log.append(&ManifestEntry::AddRunAndSequence {
        level: Level::L0,
        meta,
        next_sequence: SeqNo::from(flush_seq),
    })
    .unwrap();
}

fn replica_side(rx: &mpsc::Receiver<WireMessage>) -> World {
    let dir = std::env::temp_dir().join("minkowski-replicate-replica");
    let lsm_dir = dir.join("lsm");
    let wal_dir = dir.join("replica.wal");
    let manifest_log = lsm_dir.join("manifest.log");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&lsm_dir).unwrap();

    let mut codecs = CodecRegistry::new();
    let mut tmp = World::new();
    codecs.register_as::<Vel>("vel", &mut tmp).unwrap();
    codecs.register_as::<Pos>("pos", &mut tmp).unwrap();
    drop(tmp);

    let WireMessage::LsmBaseline {
        run_name,
        run,
        flush_seq,
    } = rx.recv().unwrap()
    else {
        panic!("expected LSM baseline");
    };

    install_baseline(&lsm_dir, &manifest_log, &run_name, &run, flush_seq);

    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
    let mut world = recover_world(&lsm_dir, &manifest_log, &mut wal, &codecs).unwrap();

    let WireMessage::WalBatch(batch_bytes) = rx.recv().unwrap() else {
        panic!("expected WAL batch");
    };
    let batch = ReplicationBatch::from_bytes(&batch_bytes).unwrap();
    apply_batch(&batch, &mut world, &codecs).unwrap();

    let _ = std::fs::remove_dir_all(&dir);
    world
}

fn main() {
    let (tx, rx) = mpsc::channel();
    source_side(&tx);
    drop(tx);
    let world = replica_side(&rx);
    println!("Replica converged with {} entities", world.entity_count());
}
