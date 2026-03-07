use parking_lot::Mutex;

use minkowski::{Access, Conflict, EnumChangeSet, Transact, Tx, World};

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
}

impl<S: Transact> Durable<S> {
    pub fn new(strategy: S, wal: Wal, codecs: CodecRegistry) -> Self {
        Self {
            inner: strategy,
            wal: Mutex::new(wal),
            codecs,
        }
    }

    /// Current WAL sequence number (next append will use this).
    pub fn wal_seq(&self) -> u64 {
        self.wal.lock().next_seq()
    }
}

impl<S: Transact> Transact for Durable<S> {
    fn begin(&self, world: &mut World, access: &Access) -> Tx<'_> {
        self.inner.begin(world, access)
    }

    fn try_commit(&self, tx: &mut Tx<'_>, world: &mut World) -> Result<EnumChangeSet, Conflict> {
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
    ) -> Result<R, Conflict> {
        let mut f = f;
        let mut last_conflict = None;
        for _attempt in 0..self.max_retries() {
            let mut tx = self.begin(world, access);
            if !tx.is_ready() {
                drop(tx);
                continue;
            }
            let value = f(&mut tx, world);
            match self.try_commit(&mut tx, world) {
                Ok(forward) => {
                    tx.mark_committed();
                    drop(tx);
                    // WAL write BEFORE apply — durable commit point
                    self.wal
                        .lock()
                        .append(&forward, &self.codecs)
                        .expect("WAL write failed — durable commit impossible");
                    forward.apply(world);
                    return Ok(value);
                }
                Err(conflict) => {
                    last_conflict = Some(conflict);
                    continue;
                }
            }
        }
        Err(last_conflict.unwrap_or_else(|| Conflict {
            component_ids: fixedbitset::FixedBitSet::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::wal::Wal;
    use minkowski::{Optimistic, Pessimistic};
    use rkyv::{Archive, Deserialize, Serialize};

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
        codecs.register::<Pos>(&mut world);

        let wal = Wal::create(&wal_path, &codecs).unwrap();
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
        codecs.register::<Health>(&mut world);

        let wal = Wal::create(&wal_path, &codecs).unwrap();
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
    fn durable_failed_attempt_not_logged() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let wal = Wal::create(&wal_path, &codecs).unwrap();
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
