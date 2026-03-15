use parking_lot::Mutex;

use minkowski::{Access, EnumChangeSet, Transact, TransactError, Tx, World, WorldMismatch};

use crate::checkpoint::CheckpointHandler;
use crate::codec::CodecRegistry;
use crate::wal::Wal;

/// Wraps any [`Transact`] strategy to guarantee WAL logging on commit.
///
/// Every successful `transact` writes the forward changeset to the WAL
/// before applying it to World. Failed attempts (retries) are not logged.
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
        self.inner.try_commit(tx, world)
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
                Ok(mut forward) => {
                    tx.mark_committed();
                    drop(tx);
                    // Drain fast-lane archetype batches into regular mutations
                    // so the WAL serializer can iterate them.
                    forward.drain_fast_lane_to_mutations();
                    // WAL write BEFORE apply — durable commit point
                    self.wal
                        .lock()
                        .append(&forward, &self.codecs)
                        .expect("WAL write failed — durable commit impossible");
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
                            let _ = handler.on_checkpoint_needed(world, &mut wal, &self.codecs);
                        }
                    }

                    return Ok(value);
                }
                Err(TransactError::Conflict(conflict)) => {
                    last_conflict = Some(conflict);
                }
                Err(e @ TransactError::WorldMismatch(_)) => return Err(e),
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
    use crate::codec::CodecRegistry;
    use crate::wal::{Wal, WalConfig};
    use minkowski::{Optimistic, Pessimistic};
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
            wal.acknowledge_snapshot(wal.next_seq())?;
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
