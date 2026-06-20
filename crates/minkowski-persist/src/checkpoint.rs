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

/// Monitoring callback fired when a checkpoint flush fails. Receives the
/// error and runs on the committing thread, inside the checkpoint handler
/// lock — must be cheap (increment a metric, send a log line) and must not
/// re-enter `AutoCheckpoint` or the WAL.
pub type FailureCallback = Box<dyn FnMut(&dyn std::error::Error) + Send + Sync>;

/// Default checkpoint handler: LSM flush of dirty pages + optional compaction.
///
/// # Operational monitoring
/// A flush failure (full disk, bad perms) is non-fatal at the call site — the
/// transaction is already durable in the WAL — but it means the recovery
/// baseline is not advancing. Use [`AutoCheckpoint::consecutive_failures`] to
/// detect sustained degradation, or install an [`AutoCheckpoint::on_failure`]
/// callback to surface failures to monitoring (metric, log aggregator, alert).
/// A rising counter signals the WAL is growing unbounded and recovery time is
/// degrading; sustained failures eventually prevent recovery from completing.
pub struct AutoCheckpoint {
    lsm_dir: PathBuf,
    manifest: Mutex<DefaultManifest>,
    manifest_log: Mutex<ManifestLog>,
    indexes: Vec<(PathBuf, Arc<Mutex<dyn PersistentIndex>>)>,
    consecutive_failures: u64,
    on_failure: Option<FailureCallback>,
}

impl AutoCheckpoint {
    /// Create a new auto-checkpoint handler rooted at `lsm_dir`.
    ///
    /// # Panics
    /// Panics if the LSM directory cannot be created or the manifest log
    /// cannot be recovered. These are non-recoverable operational conditions:
    /// an unwritable directory or a corrupt manifest log means no checkpoint
    /// can ever succeed, so the recovery baseline can never advance. Fail fast
    /// at construction rather than silently degrading on every commit.
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
            consecutive_failures: 0,
            on_failure: None,
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

    /// Install a callback fired when a checkpoint flush fails. The callback
    /// receives the error and is invoked on the committing thread, inside the
    /// checkpoint handler lock — it must be cheap (e.g. increment a metric,
    /// send a log line) and must not re-enter `AutoCheckpoint` or the WAL.
    /// Use this to surface baseline-advancement failures to monitoring so a
    /// silently-growing WAL is caught before recovery can no longer complete.
    pub fn on_failure<F>(&mut self, handler: F)
    where
        F: FnMut(&dyn std::error::Error) + Send + Sync + 'static,
    {
        self.on_failure = Some(Box::new(handler));
    }

    /// Number of consecutive checkpoint failures since the last successful
    /// flush. Resets to 0 on success. A rising value signals the recovery
    /// baseline is not advancing (e.g. full disk, bad perms); sustained
    /// failures cause unbounded WAL growth and eventually prevent recovery.
    pub fn consecutive_failures(&self) -> u64 {
        self.consecutive_failures
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
        let flush_result = flush_and_record(
            world,
            seq_range,
            &mut manifest,
            &mut log,
            &self.lsm_dir,
            codecs,
        );

        // On a flush error, count it and fire the monitoring callback before
        // returning. The error is still returned so `Durable` can log it; the
        // counter lets operators detect sustained degradation via a polling
        // check, and the callback lets them push it to a metric/log sink.
        if let Err(ref e) = flush_result {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            if let Some(ref mut cb) = self.on_failure {
                cb(e as &dyn std::error::Error);
            }
        }
        flush_result?;

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
        // Success: reset the consecutive-failure counter.
        self.consecutive_failures = 0;
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

    /// A failed checkpoint flush increments `consecutive_failures` and fires
    /// the `on_failure` callback. A subsequent successful checkpoint resets
    /// the counter to 0. This covers the monitoring path: operators detect a
    /// silently-degrading recovery baseline via the counter/callback rather
    /// than scraping stderr.
    #[test]
    fn checkpoint_failure_counter_and_callback() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("fc.wal");
        let lsm_dir = dir.path().join("lsm_fc");

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

        // Construct the handler, then make the LSM dir read-only so the flush
        // cannot write a new run file. The dir exists (created by `new`), so
        // the error comes from the flush itself, not from construction.
        let mut handler = AutoCheckpoint::new(&lsm_dir);
        let failures = Arc::new(AtomicU64::new(0));
        let failures_cb = Arc::clone(&failures);
        handler.on_failure(move |_e| {
            failures_cb.fetch_add(1, Ordering::SeqCst);
        });

        // Remove write permission from the LSM dir to force flush failure.
        // On unix, chmod 555 (r-x). On Windows this is a no-op; the test
        // still validates the counter stays 0 on success there.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o555);
            std::fs::set_permissions(&lsm_dir, perms).unwrap();
        }

        let result = handler.on_checkpoint_needed(&mut world, &mut wal, &codecs);

        #[cfg(unix)]
        {
            assert!(result.is_err(), "flush must fail on a read-only dir");
            assert_eq!(handler.consecutive_failures(), 1, "counter must increment");
            assert_eq!(
                failures.load(Ordering::SeqCst),
                1,
                "on_failure callback must fire once"
            );

            // Restore write permission and retry — the next success resets the
            // counter to 0.
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            std::fs::set_permissions(&lsm_dir, perms).unwrap();
            handler
                .on_checkpoint_needed(&mut world, &mut wal, &codecs)
                .unwrap();
            assert_eq!(
                handler.consecutive_failures(),
                0,
                "counter must reset on success"
            );
        }
        #[cfg(not(unix))]
        {
            // On non-unix we can't easily force a flush failure; just confirm
            // the success path resets the counter and the callback did not fire.
            let _ = result;
            assert_eq!(handler.consecutive_failures(), 0);
            assert_eq!(failures.load(Ordering::SeqCst), 0);
        }
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

    /// Codex PR #199 P1 scenario: a checkpoint triggered by a sparse-only
    /// removal writes NO run (flush returns None) yet still acknowledges the
    /// flush. Verify the removed sparse component does NOT resurrect on recovery.
    #[test]
    fn sparse_only_removal_checkpoint_does_not_resurrect() {
        use crate::recover::recover_world;
        use crate::wal::WalConfig;

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct TagR(u32);

        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("testr.wal");
        let lsm_dir = dir.path().join("lsmr");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<TagR>("tag_r", &mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();
        let mut handler = AutoCheckpoint::new(&lsm_dir);

        // Spawn e (recorded in WAL) and attach a sparse component in-memory.
        let e = world.alloc_entity();
        let mut cs = minkowski::EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world).unwrap();
        world.insert_sparse::<TagR>(e, TagR(7));

        // Checkpoint 1: writes run A containing the sparse TagR.
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        // Force the exact Codex condition: no archetype page is dirty at the
        // next checkpoint (a complete flush would clear dirty bits), and the
        // only mutation since is a sparse removal.
        world.clear_all_dirty_pages();
        let mut cs2 = minkowski::EnumChangeSet::new();
        cs2.remove_sparse::<TagR>(&mut world, e);
        wal.append(&cs2, &codecs).unwrap();
        cs2.apply(&mut world).unwrap();

        // Checkpoint 2: flush finds nothing to persist (no dirty pages, no sparse
        // entries) and returns None — but on_checkpoint_needed still acks.
        handler
            .on_checkpoint_needed(&mut world, &mut wal, &codecs)
            .unwrap();

        // FAITHFULNESS CHECK: checkpoint 2 must have written NO run (flush None),
        // so only run A exists — this is Codex's exact premise.
        let run_count = std::fs::read_dir(&lsm_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "run"))
            .count();
        assert_eq!(run_count, 1, "checkpoint 2 must write no run (flush None)");

        // Recover from scratch. The removal lives only in the WAL tail; recovery
        // must apply it on top of run A's (stale) sparse baseline.
        let log_path = lsm_dir.join("manifest.log");
        let mut wal2 = Wal::open(&wal_dir, &codecs, WalConfig::default()).unwrap();
        let recovered = recover_world(&lsm_dir, &log_path, &mut wal2, &codecs).unwrap();

        assert_eq!(
            recovered.get::<TagR>(e),
            None,
            "a sparse component removed via a no-run checkpoint must not resurrect"
        );
        // The entity itself must still be alive (only its sparse component went).
        assert_eq!(recovered.get::<Pos>(e).map(|p| p.x), Some(1.0));
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
