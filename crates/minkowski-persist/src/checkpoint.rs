use std::path::{Path, PathBuf};
use std::sync::Arc;

use minkowski::World;
use minkowski_lsm::compactor::{COMPACTION_TRIGGER, compact_one};
use minkowski_lsm::manifest::DefaultManifest;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::types::{SeqNo, SeqRange};
use parking_lot::Mutex;

use crate::index::PersistentIndex;
use crate::wal::Wal;
use minkowski_lsm::codec::CodecRegistry;

/// Callback invoked when the WAL has accumulated more mutation bytes than
/// the configured `max_bytes_between_checkpoints` threshold without a
/// flush acknowledgment. The default consumer is [`Durable`](crate::Durable).
///
/// Implementations should call [`Wal::acknowledge_flush`] on success to
/// reset the byte counter. If they do not, `checkpoint_needed()` will
/// remain true and the handler will fire again on the next commit.
///
/// Returning `Err` is non-fatal: the transaction that triggered the
/// checkpoint has already been committed and applied. The engine will
/// retry on the next commit that exceeds the threshold.
pub trait CheckpointHandler: Send {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Default checkpoint handler: LSM flush of dirty pages + optional compaction.
pub struct AutoCheckpoint {
    lsm_dir: PathBuf,
    manifest: Mutex<DefaultManifest>,
    manifest_log: Mutex<ManifestLog>,
    indexes: Vec<(PathBuf, Arc<Mutex<dyn PersistentIndex>>)>,
}

impl AutoCheckpoint {
    pub fn new(lsm_dir: &Path) -> Self {
        std::fs::create_dir_all(lsm_dir).expect("create LSM directory");
        let manifest_log_path = lsm_dir.join("manifest.log");
        let (manifest, log) =
            ManifestLog::recover::<4>(&manifest_log_path).expect("recover manifest log");
        Self {
            lsm_dir: lsm_dir.to_path_buf(),
            manifest: Mutex::new(manifest),
            manifest_log: Mutex::new(log),
            indexes: Vec::new(),
        }
    }

    /// Register a persistent index to be saved on each checkpoint.
    ///
    /// Index save failures are non-fatal — they are logged to stderr
    /// but do not fail the checkpoint. The LSM baseline and WAL are the
    /// source of truth; indexes are a performance optimization that
    /// can always be rebuilt.
    pub fn register_index(&mut self, path: PathBuf, index: Arc<Mutex<dyn PersistentIndex>>) {
        self.indexes.push((path, index));
    }
}

impl CheckpointHandler for AutoCheckpoint {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let flush_seq = wal.next_seq();
        let lo = wal.last_checkpoint_seq().unwrap_or(0);
        let seq_range = SeqRange::new(SeqNo::from(lo), SeqNo::from(flush_seq))
            .map_err(|e| format!("invalid flush sequence range: {e}"))?;

        let mut manifest = self.manifest.lock();
        let mut log = self.manifest_log.lock();
        flush_and_record(
            world,
            seq_range,
            &mut manifest,
            &mut log,
            &self.lsm_dir,
            codecs,
        )?;

        // Best-effort compaction when L0 is over threshold.
        while manifest
            .runs_at_level(minkowski_lsm::types::Level::L0)
            .len()
            >= COMPACTION_TRIGGER
        {
            if compact_one(&mut manifest, &mut log, &self.lsm_dir)?.is_none() {
                break;
            }
        }

        for (idx_path, index) in &self.indexes {
            let mut guard = index.lock();
            guard.update(world);
            if let Err(e) = guard.save(idx_path) {
                eprintln!("warning: index save failed for {}: {e}", idx_path.display());
            }
        }

        wal.acknowledge_flush(flush_seq)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{Wal, WalConfig};
    use minkowski::World;
    use minkowski_lsm::codec::CodecRegistry;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    #[repr(C)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[test]
    fn auto_checkpoint_creates_lsm_run() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");
        let lsm_dir = dir.path().join("lsm");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        for i in 0..10 {
            let e = world.alloc_entity();
            let mut cs = minkowski::EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world).unwrap();
        }

        assert!(wal.checkpoint_needed());

        let mut handler = AutoCheckpoint::new(&lsm_dir);
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        assert!(!wal.checkpoint_needed());
        assert!(wal.last_checkpoint_seq().is_some());

        let runs: Vec<_> = std::fs::read_dir(&lsm_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "run"))
            .collect();
        assert_eq!(runs.len(), 1);
    }

    /// (f) Regression test (Codex-#5): sparse components must survive a full
    /// `AutoCheckpoint` → `recover_world` round-trip. Before the sparse
    /// durability feature, `acknowledge_flush` advanced past sparse state that
    /// was never written to LSM, silently losing it on recovery.
    #[test]
    fn checkpoint_then_recover_preserves_sparse() {
        use crate::recover::recover_world;
        use crate::wal::WalConfig;

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag7f(u32);

        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test7f.wal");
        let lsm_dir = dir.path().join("lsm7f");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Tag7f>("tag7f", &mut world).unwrap();

        // Low threshold so that a handful of WAL appends trigger a checkpoint.
        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        // Spawn an entity and record it in the WAL.
        let e = world.alloc_entity();
        let mut cs = minkowski::EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();

        // Attach a sparse component directly to the world (not via WAL, so it
        // only exists in the in-memory world and must be captured by the LSM
        // flush triggered by the checkpoint).
        world.insert_sparse::<Tag7f>(e, Tag7f(42));

        // Append more records until checkpoint is needed.
        for i in 0..10u32 {
            let e2 = world.alloc_entity();
            let mut cs2 = minkowski::EnumChangeSet::new();
            cs2.spawn_bundle(
                &mut world,
                e2,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs2, &codecs).unwrap();
            cs2.apply(&mut world).unwrap();
        }

        assert!(wal.checkpoint_needed(), "checkpoint must be needed");

        // Run the checkpoint: flushes world (including sparse) to LSM.
        let mut handler = AutoCheckpoint::new(&lsm_dir);
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        assert!(!wal.checkpoint_needed(), "checkpoint must be acknowledged");

        // Recover via a fresh WAL handle — simulates a crash-reopen.
        let log_path = lsm_dir.join("manifest.log");
        let mut wal2 = Wal::open(&wal_dir, &codecs, WalConfig::default()).unwrap();
        let recovered = recover_world(&lsm_dir, &log_path, &mut wal2, &codecs).unwrap();

        assert_eq!(
            recovered.get::<Tag7f>(e).copied(),
            Some(Tag7f(42)),
            "sparse component must survive checkpoint → recover round-trip"
        );
    }

    #[test]
    fn auto_checkpoint_saves_registered_index() {
        use crate::index::load_btree_index;
        use minkowski::{BTreeIndex, SpatialIndex};
        use parking_lot::Mutex;
        use std::sync::Arc;

        #[derive(
            Clone,
            Copy,
            Debug,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            rkyv::Archive,
            rkyv::Serialize,
            rkyv::Deserialize,
        )]
        #[repr(C)]
        struct Score(u32);

        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("idx_ckpt.wal");
        let lsm_dir = dir.path().join("lsm");
        let idx_path = dir.path().join("score.idx");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Score>("score", &mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        for i in 0..10 {
            let e = world.alloc_entity();
            let mut cs = minkowski::EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world).unwrap();
        }

        world.spawn((Score(100),));
        world.spawn((Score(200),));

        let idx = {
            let mut idx = BTreeIndex::<Score>::new();
            idx.rebuild(&mut world);
            Arc::new(Mutex::new(idx))
        };

        let mut handler = AutoCheckpoint::new(&lsm_dir);
        handler.register_index(idx_path.clone(), idx.clone());

        assert!(wal.checkpoint_needed());
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        assert!(idx_path.exists());
        let loaded = load_btree_index::<Score>(&idx_path, world.change_tick()).unwrap();
        assert_eq!(loaded.get(&Score(100)).len(), 1);
        assert_eq!(loaded.get(&Score(200)).len(), 1);
    }
}
