use parking_lot::Mutex;

use minkowski::{Access, EnumChangeSet, Transact, TransactError, Tx, World, WorldMismatch};

use crate::checkpoint::CheckpointHandler;
use crate::wal::{Wal, WalError, WalFrameRange, WalRangeLimits};
use minkowski_lsm::codec::CodecRegistry;

/// Wraps any [`Transact`] strategy to guarantee WAL logging on commit.
///
/// Every successful `try_commit` writes the forward changeset to the WAL
/// before returning it. The closure API then applies it to World. Failed
/// attempts (retries) are not logged.
///
/// WAL write failure panics — the durability invariant is non-negotiable.
pub struct Durable<S: Transact> {
    inner: S,
    wal: Mutex<Wal>,
    codecs: CodecRegistry,
    checkpoint_handler: Option<Mutex<Box<dyn CheckpointHandler>>>,
}

impl<S: Transact> Durable<S> {
    pub fn new(strategy: S, wal: Wal, codecs: CodecRegistry) -> Self {
        Self {
            inner: strategy,
            wal: Mutex::new(wal),
            codecs,
            checkpoint_handler: None,
        }
    }

    /// Create a `Durable` wrapper with a checkpoint handler that fires
    /// when the WAL exceeds `max_bytes_between_checkpoints`.
    pub fn with_checkpoint(
        strategy: S,
        wal: Wal,
        codecs: CodecRegistry,
        handler: impl CheckpointHandler + 'static,
    ) -> Self {
        Self {
            inner: strategy,
            wal: Mutex::new(wal),
            codecs,
            checkpoint_handler: Some(Mutex::new(Box::new(handler))),
        }
    }

    /// Current WAL sequence number (next append will use this).
    pub fn wal_seq(&self) -> u64 {
        self.wal.lock().next_seq()
    }

    /// Copy synchronized raw frames, releasing the WAL lock before returning.
    /// Transport may retain or send the detached range without blocking commits.
    pub fn records_from(
        &self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<WalFrameRange, WalError> {
        self.wal.lock().records_from(from_seq, limits)
    }

    pub(crate) fn pin_replication_prefix(&self) {
        self.wal.lock().pin_replication_prefix();
    }

    pub(crate) fn durable_seq(&self) -> u64 {
        self.wal.lock().durable_seq()
    }
}

impl<S: Transact> Transact for Durable<S> {
    fn begin(&self, world: &mut World, access: &Access) -> Result<Tx<'_>, WorldMismatch> {
        self.inner.begin(world, access)
    }

    fn try_commit(
        &self,
        tx: &mut Tx<'_>,
        world: &mut World,
    ) -> Result<EnumChangeSet, TransactError> {
        let mut forward = self.inner.try_commit(tx, world)?;
        // Shared by building-block commits and all closure entry points.
        // The WAL serializer requires ordinary mutations, not fast-lane batches.
        forward.drain_fast_lane_to_mutations();
        // WAL chokepoint (#294): never log a changeset the world cannot hold.
        // The built-in strategies already reserved; this also covers custom
        // `Transact` implementations and mutations added after delegating.
        forward
            .try_reserve(world)
            .map_err(TransactError::PoolExhausted)?;
        self.wal
            .lock()
            .append(&forward, &self.codecs, world.current_tick())
            .expect("WAL write failed — durable commit impossible");
        Ok(forward)
    }

    fn max_retries(&self) -> usize {
        self.inner.max_retries()
    }

    fn transact<R>(
        &self,
        world: &mut World,
        access: &Access,
        f: impl FnMut(&mut Tx<'_>, &mut World) -> R,
    ) -> Result<R, TransactError> {
        let mut f = f;
        let mut last_conflict = None;
        for _attempt in 0..self.max_retries() {
            let mut tx = self.begin(world, access)?;
            if !tx.is_ready() {
                drop(tx);
                continue;
            }
            let value = f(&mut tx, world);
            let commit_result = self.try_commit(&mut tx, world);
            match commit_result {
                Ok(forward) => {
                    tx.mark_committed();
                    drop(tx);
                    forward
                        .apply(world)
                        .expect("changeset apply after successful commit");

                    // Fire checkpoint handler if threshold exceeded.
                    // Checkpoint is best-effort — the transaction is already
                    // committed and applied. Errors are non-fatal; the handler
                    // will be retried on the next commit that exceeds the threshold.
                    if let Some(ref handler_mutex) = self.checkpoint_handler {
                        let mut wal = self.wal.lock();
                        if wal.checkpoint_needed() {
                            let mut handler = handler_mutex.lock();
                            // Operational, not fatal: the transaction is already
                            // committed and durable in the WAL. But a failed flush
                            // is the recovery baseline failing to advance — surface
                            // it rather than swallow it (e.g. full disk, bad perms).
                            if let Err(e) =
                                handler.on_checkpoint_needed(world, &mut wal, &self.codecs)
                            {
                                eprintln!(
                                    "warning: LSM checkpoint failed (commit is durable in WAL, \
                                     recovery baseline did not advance): {e}"
                                );
                            }
                        }
                    }

                    return Ok(value);
                }
                Err(TransactError::Conflict(conflict)) => {
                    last_conflict = Some(conflict);
                }
                Err(e @ (TransactError::WorldMismatch(_) | TransactError::PoolExhausted(_))) => {
                    return Err(e);
                }
            }
        }
        Err(TransactError::Conflict(last_conflict.unwrap_or_else(
            || minkowski::Conflict {
                component_ids: fixedbitset::FixedBitSet::new(),
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointHandler;
    use crate::wal::{Wal, WalConfig};
    use minkowski::{Optimistic, Pessimistic};
    use minkowski_lsm::codec::CodecRegistry;
    use rkyv::{Archive, Deserialize, Serialize};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Health(u32);

    #[test]
    fn durable_optimistic_transact_logs_to_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let strategy = Optimistic::new(&world);
        let durable = Durable::new(strategy, wal, codecs);

        let access = Access::of::<(&mut Pos,)>(&mut world);
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        durable
            .transact(&mut world, &access, |tx, world| {
                tx.write::<Pos>(world, e, Pos { x: 10.0, y: 20.0 });
            })
            .unwrap();

        assert_eq!(durable.wal_seq(), 1);
        assert_eq!(world.get::<Pos>(e).unwrap().x, 10.0);
    }

    #[test]
    fn durable_pessimistic_transact_logs_to_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Health>(&mut world).unwrap();

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let strategy = Pessimistic::new(&world);
        let durable = Durable::new(strategy, wal, codecs);

        let access = Access::of::<(&mut Health,)>(&mut world);
        let e = world.spawn((Health(100),));

        durable
            .transact(&mut world, &access, |tx, world| {
                tx.write::<Health>(world, e, Health(90));
            })
            .unwrap();

        assert_eq!(durable.wal_seq(), 1);
    }

    #[test]
    fn durable_commit_pool_exhausted_writes_no_wal_record() {
        // #294 / A3: the inner strategy reserves the commit's pool growth
        // before Durable appends to the WAL, so a transaction that cannot fit
        // returns PoolExhausted and logs nothing (the WAL never runs ahead of
        // the world).
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::builder()
            .memory_budget(512 << 10)
            .hugepages(minkowski::HugePages::Off)
            .build()
            .unwrap();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();
        let mut entities = Vec::new();
        while let Ok(e) = world.try_spawn((Pos { x: 0.0, y: 0.0 },)) {
            entities.push(e);
        }

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let durable = Durable::new(Optimistic::new(&world), wal, codecs);
        let access = Access::of::<(&mut Health,)>(&mut world);
        let result = durable.transact(&mut world, &access, |tx, world| {
            for &e in &entities {
                tx.write::<Health>(world, e, Health(1)); // migrates every entity
            }
        });

        assert!(
            matches!(result, Err(TransactError::PoolExhausted(_))),
            "{result:?}"
        );
        assert_eq!(durable.wal_seq(), 0, "nothing logged");
        assert!(world.query::<(&Health,)>().next().is_none());
    }

    /// Codex P1 on #296: a custom strategy can hand Durable an unreserved
    /// changeset (here: mutations appended after delegating to Optimistic).
    /// Durable reserves at its own WAL chokepoint, so it still logs nothing.
    #[test]
    fn durable_reserves_custom_strategy_changesets_before_wal() {
        struct AppendsAfterCommit {
            inner: Optimistic,
            entities: Vec<minkowski::Entity>,
        }
        impl Transact for AppendsAfterCommit {
            fn begin(&self, world: &mut World, access: &Access) -> Result<Tx<'_>, WorldMismatch> {
                self.inner.begin(world, access)
            }
            fn try_commit(
                &self,
                tx: &mut Tx<'_>,
                world: &mut World,
            ) -> Result<EnumChangeSet, TransactError> {
                let mut forward = self.inner.try_commit(tx, world)?;
                for &e in &self.entities {
                    forward.insert(world, e, Health(1)); // never reserved
                }
                Ok(forward)
            }
            fn max_retries(&self) -> usize {
                1
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let mut world = World::builder()
            .memory_budget(512 << 10)
            .hugepages(minkowski::HugePages::Off)
            .build()
            .unwrap();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Health>(&mut world).unwrap();
        let mut entities = Vec::new();
        while let Ok(e) = world.try_spawn((Pos { x: 0.0, y: 0.0 },)) {
            entities.push(e);
        }
        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let strategy = AppendsAfterCommit {
            inner: Optimistic::new(&world),
            entities,
        };
        let durable = Durable::new(strategy, wal, codecs);
        let access = Access::of::<(&mut Health,)>(&mut world);
        let result = durable.transact(&mut world, &access, |_tx, _world| {});
        assert!(
            matches!(result, Err(TransactError::PoolExhausted(_))),
            "{result:?}"
        );
        assert_eq!(durable.wal_seq(), 0, "nothing logged");
    }

    struct CountingHandler {
        count: Arc<AtomicU32>,
    }

    impl CheckpointHandler for CountingHandler {
        fn on_checkpoint_needed(
            &mut self,
            _world: &mut World,
            wal: &mut Wal,
            _codecs: &CodecRegistry,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.count.fetch_add(1, Ordering::SeqCst);
            wal.acknowledge_flush(wal.next_seq())?;
            Ok(())
        }
    }

    #[test]
    fn durable_fires_checkpoint_handler() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(64),
        };
        let wal = Wal::create(&wal_path, &codecs, config).unwrap();
        let strategy = Optimistic::new(&world);

        let count = Arc::new(AtomicU32::new(0));
        let handler = CountingHandler {
            count: count.clone(),
        };
        let durable = Durable::with_checkpoint(strategy, wal, codecs, handler);

        let access = Access::of::<(&mut Pos,)>(&mut world);
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        for _ in 0..20 {
            durable
                .transact(&mut world, &access, |tx, world| {
                    tx.write::<Pos>(world, e, Pos { x: 10.0, y: 20.0 });
                })
                .unwrap();
        }

        assert!(
            count.load(Ordering::SeqCst) >= 1,
            "handler should have fired"
        );
    }

    #[test]
    fn durable_no_handler_backward_compat() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(64),
        };
        let wal = Wal::create(&wal_path, &codecs, config).unwrap();
        let strategy = Optimistic::new(&world);
        let durable = Durable::new(strategy, wal, codecs);

        let access = Access::of::<(&mut Pos,)>(&mut world);
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        // Should work fine without handler, even if threshold is exceeded
        for _ in 0..20 {
            durable
                .transact(&mut world, &access, |tx, world| {
                    tx.write::<Pos>(world, e, Pos { x: 10.0, y: 20.0 });
                })
                .unwrap();
        }
    }

    #[test]
    fn durable_failed_attempt_not_logged() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let strategy = Optimistic::with_retries(&world, 3);
        let durable = Durable::new(strategy, wal, codecs);

        let access = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let e = world.spawn((Pos { x: 1.0, y: 1.0 },));

        let mut attempt = 0u32;
        durable
            .transact(&mut world, &access, |tx, world| {
                attempt += 1;
                if attempt == 1 {
                    // Force conflict by mutating the read column
                    for pos in world.query::<(&mut Pos,)>() {
                        pos.0.x = 99.0;
                    }
                }
                tx.write::<Pos>(world, e, Pos { x: 42.0, y: 42.0 });
            })
            .unwrap();

        // Only the successful attempt should be logged
        assert_eq!(durable.wal_seq(), 1);
        assert!(attempt >= 2);
    }
}
