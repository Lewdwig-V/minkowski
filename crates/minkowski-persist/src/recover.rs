//! World recovery from LSM sorted runs + WAL tail replay.

use std::path::Path;

use minkowski::World;
use minkowski_lsm::codec::CodecRegistry;
use minkowski_lsm::error::LsmError;
use minkowski_lsm::recovery::LsmRecovery;

use crate::wal::{Wal, WalError};

/// Errors during world recovery.
#[derive(Debug, thiserror::Error)]
pub enum RecoverError {
    #[error("LSM recovery error: {0}")]
    Lsm(#[from] LsmError),
    #[error("WAL replay error: {0}")]
    Wal(#[from] WalError),
}

/// Recover a [`World`] from on-disk LSM state and replay the WAL tail.
///
/// If no LSM manifest exists yet, returns an empty world and replays the WAL
/// from sequence 0.
pub fn recover_world(
    lsm_dir: &Path,
    manifest_log_path: &Path,
    wal: &mut Wal,
    codecs: &CodecRegistry,
) -> Result<World, RecoverError> {
    if manifest_log_path.exists() {
        let (result, _, _) = LsmRecovery::recover::<4>(lsm_dir, manifest_log_path, codecs)?;
        let mut world = result.world;
        for id in codecs.registered_ids() {
            codecs.register_one(id, &mut world);
        }
        // INVARIANT (replay floor): we replay from the newest LSM run's seq_hi,
        // NOT the WAL's last acknowledged flush_seq. `AutoCheckpoint` may
        // acknowledge a flush_seq ahead of the LSM baseline when a checkpoint
        // had nothing to persist (e.g. a sparse-only removal → `flush` returns
        // None). That is benign ONLY because the WAL is never truncated below
        // this floor, so every mutation in `[result.flush_seq, end)` — including
        // such a removal — is still replayed on top of the baseline. If WAL
        // truncation is ever added it MUST NOT delete records at or above the
        // newest run's seq_hi, or removed state would resurrect. Covered by
        // `checkpoint::tests::sparse_only_removal_checkpoint_does_not_resurrect`.
        wal.replay_from(result.flush_seq, &mut world, codecs)?;
        Ok(world)
    } else {
        // No LSM baseline yet (crash before first flush). Register components
        // before replaying — otherwise spawn replay records ComponentIds into an
        // empty registry and EnumChangeSet::apply panics resolving their layouts.
        let mut world = World::new();
        for id in codecs.registered_ids() {
            codecs.register_one(id, &mut world);
        }
        wal.replay_from(0, &mut world, codecs)?;
        Ok(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski_lsm::manifest_log::ManifestLog;
    use minkowski_lsm::manifest_ops::flush_and_record;
    use minkowski_lsm::types::{SeqNo, SeqRange};
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    struct Health(u32);

    #[test]
    fn recover_world_replays_wal_tail() {
        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        let wal_dir = dir.path().join("test.wal");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();

        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(0u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let mut wal = Wal::create(&wal_dir, &codecs, crate::wal::WalConfig::default()).unwrap();
        let e = world.alloc_entity();
        let mut cs = minkowski::EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Health(42),)).unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();

        let mut wal2 = Wal::open(&wal_dir, &codecs, crate::wal::WalConfig::default()).unwrap();
        let mut recovered = recover_world(&lsm_dir, &log_path, &mut wal2, &codecs).unwrap();

        assert_eq!(recovered.query::<(&Pos,)>().count(), 1);
        assert_eq!(recovered.query::<(&Health,)>().count(), 1);
        assert_eq!(recovered.query::<(&Health,)>().next().unwrap().0.0, 42);
    }

    /// (e) Sparse component in LSM baseline + sparse mutation appended to WAL
    /// tail after the flush_seq both appear in the recovered world.
    #[test]
    fn recover_world_restores_sparse_then_wal_tail() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Pos7e {
            x: f32,
            y: f32,
        }

        // Baseline sparse component — written by flush.
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct BaseTag7e(u32);

        // WAL-tail sparse component — written after the flush.
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct TailTag7e(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        let wal_dir = dir.path().join("test7e.wal");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos7e>("pos7e", &mut world).unwrap();
        codecs
            .register_as::<BaseTag7e>("base_tag7e", &mut world)
            .unwrap();
        codecs
            .register_as::<TailTag7e>("tail_tag7e", &mut world)
            .unwrap();

        // Spawn an entity, attach a baseline sparse component.
        let e_base = world.spawn((Pos7e { x: 1.0, y: 0.0 },));
        world.insert_sparse::<BaseTag7e>(e_base, BaseTag7e(100));

        // Flush baseline to LSM.
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(0u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("baseline flush");

        // Post-flush: append a WAL record that inserts a sparse component on a
        // new entity. This is the "WAL tail" that replay_from must pick up.
        let mut wal = Wal::create(&wal_dir, &codecs, crate::wal::WalConfig::default()).unwrap();

        let e_tail = world.alloc_entity();
        let mut cs = minkowski::EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e_tail, (Pos7e { x: 2.0, y: 0.0 },))
            .unwrap();
        cs.insert_sparse::<TailTag7e>(&mut world, e_tail, TailTag7e(200));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();

        // Open a second WAL handle (simulates crash-reopen) and recover.
        let mut wal2 = Wal::open(&wal_dir, &codecs, crate::wal::WalConfig::default()).unwrap();
        let recovered = recover_world(&lsm_dir, &log_path, &mut wal2, &codecs).unwrap();

        // Baseline sparse survives via apply_sparse.
        assert_eq!(
            recovered.get::<BaseTag7e>(e_base).copied(),
            Some(BaseTag7e(100)),
            "baseline sparse must survive recover"
        );

        // WAL-tail sparse survives via replay_from.
        assert_eq!(
            recovered.get::<TailTag7e>(e_tail).copied(),
            Some(TailTag7e(200)),
            "WAL-tail sparse must survive replay"
        );
    }
}
