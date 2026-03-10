use std::path::{Path, PathBuf};
use std::sync::Arc;

use minkowski::World;
use parking_lot::Mutex;

use crate::codec::CodecRegistry;
use crate::index::PersistentIndex;
use crate::snapshot::Snapshot;
use crate::wal::Wal;

/// Callback invoked when the WAL has accumulated more mutation bytes than
/// the configured `max_bytes_between_checkpoints` threshold without a
/// snapshot acknowledgment. The default consumer is [`Durable`](crate::Durable).
///
/// Implementations should call [`Wal::acknowledge_snapshot`] on success to
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

/// Default checkpoint handler: saves a snapshot and acknowledges it.
///
/// Snapshots are written to `snap_dir/checkpoint-{seq:06}.snap`.
pub struct AutoCheckpoint {
    snap_dir: PathBuf,
    indexes: Vec<(PathBuf, Arc<Mutex<dyn PersistentIndex>>)>,
}

impl AutoCheckpoint {
    pub fn new(snap_dir: &Path) -> Self {
        Self {
            snap_dir: snap_dir.to_path_buf(),
            indexes: Vec::new(),
        }
    }

    /// Register a persistent index to be saved on each checkpoint.
    ///
    /// Index save failures are non-fatal — they are logged to stderr
    /// but do not fail the checkpoint. The snapshot and WAL are the
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
        let seq = wal.next_seq();
        let path = self.snap_dir.join(format!("checkpoint-{seq:06}.snap"));
        let snap = Snapshot::new();
        snap.save(&path, world, codecs, seq)?;

        // Sync and save registered indexes (non-fatal on failure).
        // update() ensures the index reflects all mutations up to this
        // checkpoint — without it, the saved index could be stale.
        for (idx_path, index) in &self.indexes {
            let mut guard = index.lock();
            guard.update(world);
            if let Err(e) = guard.save(idx_path) {
                eprintln!("warning: index save failed for {}: {e}", idx_path.display());
            }
        }

        wal.acknowledge_snapshot(seq)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::wal::{Wal, WalConfig};
    use minkowski::World;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    #[repr(C)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[test]
    fn auto_checkpoint_creates_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");
        let snap_dir = dir.path().join("snaps");
        std::fs::create_dir_all(&snap_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

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
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world).unwrap();
        }

        assert!(wal.checkpoint_needed());

        let mut handler = AutoCheckpoint::new(&snap_dir);
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        assert!(!wal.checkpoint_needed());
        assert!(wal.last_checkpoint_seq().is_some());

        // Verify snapshot file was created
        let snaps: Vec<_> = std::fs::read_dir(&snap_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "snap"))
            .collect();
        assert_eq!(snaps.len(), 1);
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
        let snap_dir = dir.path().join("snaps");
        let idx_path = dir.path().join("score.idx");
        std::fs::create_dir_all(&snap_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);
        codecs.register_as::<Score>("score", &mut world);

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        // Spawn entities with Pos (for WAL) and Score (for index)
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
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world).unwrap();
        }

        // Also spawn some Score entities (not in WAL, just in world)
        world.spawn((Score(100),));
        world.spawn((Score(200),));

        // Build and register index
        let idx = {
            let mut idx = BTreeIndex::<Score>::new();
            idx.rebuild(&mut world);
            Arc::new(Mutex::new(idx))
        };

        let mut handler = AutoCheckpoint::new(&snap_dir);
        handler.register_index(idx_path.clone(), idx.clone());

        assert!(wal.checkpoint_needed());
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        // Verify index file was created and is loadable
        assert!(idx_path.exists());
        let loaded = load_btree_index::<Score>(&idx_path, world.change_tick()).unwrap();
        assert_eq!(loaded.get(&Score(100)).len(), 1);
        assert_eq!(loaded.get(&Score(200)).len(), 1);
    }
}
