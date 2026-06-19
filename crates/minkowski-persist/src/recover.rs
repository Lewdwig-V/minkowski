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
}
