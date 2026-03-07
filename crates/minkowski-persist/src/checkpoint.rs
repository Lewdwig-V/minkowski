use std::path::{Path, PathBuf};

use minkowski::World;

use crate::codec::CodecRegistry;
use crate::snapshot::Snapshot;
use crate::wal::Wal;

/// Callback invoked by [`Durable`](crate::Durable) when the WAL exceeds
/// `max_bytes_between_checkpoints` without a snapshot acknowledgment.
pub trait CheckpointHandler: Send {
    fn on_checkpoint_needed(&mut self, world: &mut World, wal: &mut Wal, codecs: &CodecRegistry);
}

/// Default checkpoint handler: saves a snapshot and acknowledges it.
///
/// Snapshots are written to `snap_dir/checkpoint-{seq:06}.snap`.
pub struct AutoCheckpoint {
    snap_dir: PathBuf,
}

impl AutoCheckpoint {
    pub fn new(snap_dir: &Path) -> Self {
        Self {
            snap_dir: snap_dir.to_path_buf(),
        }
    }
}

impl CheckpointHandler for AutoCheckpoint {
    fn on_checkpoint_needed(&mut self, world: &mut World, wal: &mut Wal, codecs: &CodecRegistry) {
        let seq = wal.next_seq();
        let path = self.snap_dir.join(format!("checkpoint-{seq:06}.snap"));
        let snap = Snapshot::new();
        snap.save(&path, world, codecs, seq)
            .expect("checkpoint snapshot failed");
        wal.acknowledge_snapshot(seq)
            .expect("checkpoint acknowledge failed");
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
            cs.apply(&mut world);
        }

        assert!(wal.checkpoint_needed());

        let mut handler = AutoCheckpoint::new(&snap_dir);
        handler.on_checkpoint_needed(&mut world, &mut wal, &codecs);

        assert!(!wal.checkpoint_needed());
        assert!(wal.last_checkpoint_seq().is_some());

        // Verify snapshot file was created
        let snaps: Vec<_> = std::fs::read_dir(&snap_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "snap").unwrap_or(false))
            .collect();
        assert_eq!(snaps.len(), 1);
    }
}
