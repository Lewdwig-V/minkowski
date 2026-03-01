use fixedbitset::FixedBitSet;

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
use crate::world::World;

/// Conflict information returned when a transaction commit fails.
pub struct Conflict {
    /// Which component columns had conflicting concurrent modifications.
    pub component_ids: FixedBitSet,
}

/// Strategy for transaction concurrency control.
///
/// The trait provides `begin()` which returns a strategy-specific transaction
/// object. The caller reads and writes through the transaction, then calls
/// `commit()` to apply changes atomically — or drops the transaction to abort.
///
/// Three built-in strategies:
/// - [`Sequential`] — zero-cost passthrough, no conflict detection
/// - [`Optimistic`] — live reads with tick-based validation at commit
/// - [`Pessimistic`] — cooperative column locks, guaranteed commit success
pub trait TransactionStrategy {
    /// The transaction object returned by `begin()`.
    type Tx<'w>;

    /// Begin a transaction. `access` declares which components will be
    /// read and written — used by Optimistic for tick snapshotting and
    /// by Pessimistic for lock acquisition.
    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> Self::Tx<'w>;
}

// ── Sequential ──────────────────────────────────────────────────────

/// Zero-cost transaction strategy. All operations delegate directly to World.
/// Commit always succeeds. No read-set, no changeset buffering, no validation.
///
/// Use when systems run sequentially and no conflict detection is needed.
pub struct Sequential;

impl TransactionStrategy for Sequential {
    type Tx<'w> = SequentialTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, _access: &Access) -> SequentialTx<'w> {
        SequentialTx { world }
    }
}

/// Transaction object for the [`Sequential`] strategy.
/// Transparent wrapper around `&mut World`.
pub struct SequentialTx<'w> {
    world: &'w mut World,
}

impl<'w> SequentialTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        self.world.spawn(bundle)
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.world.despawn(entity)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.world.insert(entity, component);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.world.remove::<T>(entity)
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.world.get_mut::<T>(entity)
    }

    /// Commit the transaction. Always succeeds for Sequential.
    /// Returns an empty reverse changeset (mutations went directly to World).
    pub fn commit(self) -> Result<EnumChangeSet, Conflict> {
        Ok(EnumChangeSet::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::Access;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);

    #[test]
    fn sequential_query_reads_spawned_entities() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &Vel)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn sequential_mutation_is_immediate() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn((Pos(42.0),));
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        let _ = tx.commit();
    }

    #[test]
    fn sequential_commit_always_ok() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn((Pos(1.0),));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn sequential_drop_is_noop() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let mut strategy = Sequential;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.spawn((Pos(1.0),));
            // drop without commit — but mutations already applied
        }
        assert_eq!(world.query::<(&Pos,)>().count(), 1);
    }
}
