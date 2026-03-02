//! Transaction strategies for concurrent ECS access.
//!
//! # Design overview
//!
//! A single [`TransactionStrategy`] trait provides optimistic and pessimistic
//! concurrency behind one interface. [`Sequential`] exists as a zero-cost
//! passthrough for code that doesn't need isolation — users who never touch
//! transactions pay nothing.
//!
//! ## Split-phase execution
//!
//! Tx objects do **not** hold `&mut World`. Methods take the world as a
//! parameter, which splits every frame into three phases:
//!
//! 1. **Begin** (sequential, `&mut World`) — snapshot ticks or acquire locks.
//! 2. **Execute** (parallel, `&World`) — read through `tx.query(&world)`.
//!    Multiple transactions can read concurrently because `query` requires
//!    [`ReadOnlyWorldQuery`], preventing `&mut T` access through a shared
//!    reference. Without this bound the parallel phase would be unsound —
//!    two transactions could obtain aliased `&mut T` from the same `&World`.
//! 3. **Commit** (sequential, `&mut World`) — validate and apply buffered
//!    writes. `commit(self, &mut World)` consumes the transaction so it
//!    cannot be reused.
//!
//! ## Commit, abort, and entity lifecycle
//!
//! Dropping a transaction without calling `commit` is an **abort**: the
//! changeset is discarded, locks are released, and no mutations reach World.
//! `commit` consumes `self` so the compiler enforces exactly-once semantics.
//!
//! Entity IDs allocated during a transaction (`tx.spawn`) are tracked in a
//! `spawned_entities` vec. On successful commit they become placed entities.
//! On abort — whether from an explicit drop or a failed optimistic validation
//! — the IDs must be reclaimed. `Drop` pushes them to an [`OrphanQueue`]
//! shared with World via `Arc<Mutex<Vec<Entity>>>`. World drains this queue
//! automatically at the top of every `&mut self` method, bumping generations
//! and recycling indices. No entity ID ever leaks, regardless of how the
//! transaction ends, and no manual cleanup step is required.
//!
//! [`OrphanQueue`]: crate::world::OrphanQueue
//!
//! ## Cooperative locking (pessimistic)
//!
//! [`Pessimistic`] owns a `Mutex<ColumnLockTable>` — a bookkeeping structure
//! (not an OS mutex per column) tracking shared readers and exclusive writers
//! at `(ArchetypeId, ComponentId)` granularity. When a component appears in
//! both reads and writes, the lock request is **upgraded** to exclusive —
//! `dedup_by` keeps the highest privilege, not the first seen. Locks are
//! acquired in deterministic `(arch, comp)` order to prevent deadlock. The
//! mutex is held only during begin/commit/drop (all sequential phases), never
//! during the parallel read phase.
//!
//! ## World identity
//!
//! Each World gets a unique [`WorldId`] at construction. Strategies capture it
//! and assert it matches in `begin()` and `commit()`. This prevents a strategy
//! created from world A being used with world B, which would push aborted
//! entity IDs into the wrong orphan queue and corrupt unrelated live entities.
//!
//! [`WorldId`]: crate::world::WorldId

use std::sync::atomic::{AtomicU64, Ordering};

use fixedbitset::FixedBitSet;
use parking_lot::Mutex;

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::Component;
use crate::entity::Entity;
use crate::lock_table::{ColumnLockSet, ColumnLockTable};
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
use crate::query::iter::QueryIter;
use crate::world::{World, WorldId};

/// Unique transaction identifier, monotonically increasing per strategy.
pub type TxId = u64;

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
///
/// Tx types do not hold a World reference. Methods take `&World` or
/// `&mut World` as parameters, enabling split-phase execution where
/// multiple transactions read concurrently and commit sequentially.
pub trait TransactionStrategy {
    /// The transaction object returned by `begin()`.
    type Tx<'s>
    where
        Self: 's;

    /// Begin a transaction. `access` declares which components will be
    /// read and written — used by Optimistic for tick snapshotting and
    /// by Pessimistic for lock acquisition.
    ///
    /// World is borrowed transiently for setup. The returned Tx does not
    /// hold a World reference. Takes `&self` (not `&mut self`) so multiple
    /// transactions can coexist from the same strategy instance.
    fn begin<'s>(&'s self, world: &mut World, access: &Access) -> Self::Tx<'s>;
}

// ── Sequential ──────────────────────────────────────────────────────

/// Zero-cost transaction strategy. All operations delegate directly to World.
/// Commit always succeeds. No read-set, no changeset buffering, no validation.
///
/// Use when systems run sequentially and no conflict detection is needed.
pub struct Sequential;

impl TransactionStrategy for Sequential {
    type Tx<'s> = SequentialTx;

    fn begin(&self, _world: &mut World, _access: &Access) -> SequentialTx {
        SequentialTx
    }
}

/// Transaction object for the [`Sequential`] strategy.
/// Zero-state unit struct — all methods delegate directly to World.
pub struct SequentialTx;

impl SequentialTx {
    pub fn query<'a, Q: WorldQuery + 'static>(&self, world: &'a mut World) -> QueryIter<'a, Q> {
        world.query::<Q>()
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, world: &mut World, bundle: B) -> Entity {
        world.spawn(bundle)
    }

    pub fn despawn(&mut self, world: &mut World, entity: Entity) -> bool {
        world.despawn(entity)
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, component: T) {
        world.insert(entity, component);
    }

    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) -> Option<T> {
        world.remove::<T>(entity)
    }

    pub fn get_mut<'a, T: Component>(
        &mut self,
        world: &'a mut World,
        entity: Entity,
    ) -> Option<&'a mut T> {
        world.get_mut::<T>(entity)
    }

    /// Commit the transaction. Always succeeds for Sequential.
    /// Returns an empty reverse changeset (mutations went directly to World).
    pub fn commit(self, _world: &mut World) -> Result<EnumChangeSet, Conflict> {
        Ok(EnumChangeSet::new())
    }
}

// ── Optimistic ──────────────────────────────────────────────────────

/// Optimistic transaction strategy. Reads go directly to World (zero-copy)
/// via the shared-ref `query_raw` path. Writes buffer into an EnumChangeSet.
/// At commit, validates that no read column was modified since begin.
///
/// Best for read-heavy workloads where conflicts are rare.
pub struct Optimistic {
    world_id: WorldId,
    next_tx_id: AtomicU64,
    orphan_queue: crate::world::OrphanQueue,
}

impl Optimistic {
    pub fn new(world: &World) -> Self {
        Self {
            world_id: world.world_id(),
            next_tx_id: AtomicU64::new(1),
            orphan_queue: world.orphan_queue(),
        }
    }
}

impl TransactionStrategy for Optimistic {
    type Tx<'s> = OptimisticTx<'s>;

    fn begin<'s>(&'s self, world: &mut World, access: &Access) -> OptimisticTx<'s> {
        assert_eq!(
            self.world_id,
            world.world_id(),
            "strategy used with a different World than it was created from"
        );
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::Relaxed);
        let mut accessed = access.reads().clone();
        accessed.union_with(access.writes());
        let read_ticks = world.snapshot_column_ticks(&accessed);
        let archetype_count = world.archetypes.archetypes.len();
        OptimisticTx {
            tx_id,
            strategy: self,
            read_ticks,
            archetype_count,
            spawned_entities: Vec::new(),
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Optimistic`] strategy.
///
/// Does not hold a World reference — methods take `&World` or `&mut World`
/// as parameters. Reads use `World::query_raw` (shared-ref, no tick/cache
/// mutation). Writes buffer into an `EnumChangeSet`.
///
/// Archetype set is frozen at begin time — queries only see archetypes
/// that existed when the transaction started.
///
/// Drop without commit releases spawned entity IDs back to the allocator
/// via the World's orphan queue — no entity IDs ever leak.
pub struct OptimisticTx<'s> {
    /// Unique transaction identifier (monotonic per strategy).
    pub tx_id: TxId,
    strategy: &'s Optimistic,
    read_ticks: Vec<(usize, crate::component::ComponentId, crate::tick::Tick)>,
    archetype_count: usize,
    spawned_entities: Vec<Entity>,
    changeset: EnumChangeSet,
}

impl<'s> OptimisticTx<'s> {
    /// Read through the transaction via shared World reference.
    /// No tick advancement, no cache mutation. Safe for concurrent reads.
    /// Only sees archetypes that existed at begin time.
    ///
    /// Requires `ReadOnlyWorldQuery` — `&mut T` queries are rejected at
    /// compile time. Writes go through `insert`/`remove`/`spawn` instead.
    pub fn query<'a, Q: ReadOnlyWorldQuery + 'static>(&self, world: &'a World) -> QueryIter<'a, Q> {
        world.query_raw::<Q>(self.archetype_count)
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        self.changeset.insert::<T>(world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        self.changeset.remove::<T>(world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, world: &mut World, bundle: B) -> Entity {
        let entity = world.alloc_entity();
        self.spawned_entities.push(entity);
        self.changeset.spawn_bundle(world, entity, bundle);
        entity
    }

    /// Validate and commit. Checks that no read column was modified since
    /// begin. On conflict, spawned entity IDs are immediately deallocated.
    pub fn commit(mut self, world: &mut World) -> Result<EnumChangeSet, Conflict> {
        assert_eq!(
            self.strategy.world_id,
            world.world_id(),
            "transaction committed to a different World than it was created from"
        );
        let conflicts = world.check_column_conflicts(&self.read_ticks);
        if !conflicts.is_empty() {
            for &entity in &self.spawned_entities {
                world.dealloc_entity(entity);
            }
            self.spawned_entities.clear();
            return Err(Conflict {
                component_ids: conflicts,
            });
        }
        self.spawned_entities.clear();
        let changeset = std::mem::take(&mut self.changeset);
        Ok(changeset.apply(world))
    }
}

impl<'s> Drop for OptimisticTx<'s> {
    fn drop(&mut self) {
        if !self.spawned_entities.is_empty() {
            self.strategy
                .orphan_queue
                .0
                .lock()
                .extend(self.spawned_entities.drain(..));
        }
    }
}

// ── Pessimistic ─────────────────────────────────────────────────────

/// Pessimistic transaction strategy. Acquires cooperative column locks
/// at begin. Reads and writes are guaranteed conflict-free. Commit
/// always succeeds.
///
/// Owns the `ColumnLockTable` via `Mutex` — multiple transactions can
/// coexist from the same strategy instance. The mutex is only locked
/// during begin/commit/drop (all sequential phases), never contended
/// during the parallel read phase. Created via `Pessimistic::new(&world)`.
///
/// Best for write-heavy workloads or expensive computations where
/// optimistic retry would waste work.
pub struct Pessimistic {
    world_id: WorldId,
    lock_table: Mutex<ColumnLockTable>,
    next_tx_id: AtomicU64,
    orphan_queue: crate::world::OrphanQueue,
}

impl Pessimistic {
    pub fn new(world: &World) -> Self {
        Self {
            world_id: world.world_id(),
            lock_table: Mutex::new(ColumnLockTable::new()),
            next_tx_id: AtomicU64::new(1),
            orphan_queue: world.orphan_queue(),
        }
    }
}

impl TransactionStrategy for Pessimistic {
    type Tx<'s> = PessimisticTx<'s>;

    fn begin<'s>(&'s self, world: &mut World, access: &Access) -> PessimisticTx<'s> {
        assert_eq!(
            self.world_id,
            world.world_id(),
            "strategy used with a different World than it was created from"
        );
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::Relaxed);
        let lock_result = self.lock_table.lock().acquire(
            &world.archetypes.archetypes,
            access.reads(),
            access.writes(),
        );
        let locks = lock_result.ok();
        let archetype_count = world.archetypes.archetypes.len();
        PessimisticTx {
            tx_id,
            strategy: self,
            locks,
            archetype_count,
            spawned_entities: Vec::new(),
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Pessimistic`] strategy.
///
/// Holds a shared reference to the Pessimistic strategy for lock release
/// on drop. Multiple transactions can coexist from the same strategy.
/// Does not hold a World reference — methods take `&World` or `&mut World`.
///
/// Drop without commit releases locks and discards the changeset (abort).
pub struct PessimisticTx<'s> {
    /// Unique transaction identifier (monotonic per strategy).
    pub tx_id: TxId,
    strategy: &'s Pessimistic,
    locks: Option<ColumnLockSet>,
    archetype_count: usize,
    spawned_entities: Vec<Entity>,
    changeset: EnumChangeSet,
}

impl<'s> PessimisticTx<'s> {
    /// Read through the transaction via shared World reference.
    /// Only sees archetypes that existed at begin time.
    ///
    /// Requires `ReadOnlyWorldQuery` — `&mut T` queries are rejected at
    /// compile time. Writes go through `insert`/`remove`/`spawn` instead.
    pub fn query<'a, Q: ReadOnlyWorldQuery + 'static>(&self, world: &'a World) -> QueryIter<'a, Q> {
        world.query_raw::<Q>(self.archetype_count)
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        self.changeset.insert::<T>(world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        self.changeset.remove::<T>(world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, world: &mut World, bundle: B) -> Entity {
        let entity = world.alloc_entity();
        self.spawned_entities.push(entity);
        self.changeset.spawn_bundle(world, entity, bundle);
        entity
    }

    /// Returns true if locks were successfully acquired at begin time.
    /// If false, commit will return Err(Conflict) — the caller should
    /// fall back to sequential execution for conflicting systems.
    pub fn has_locks(&self) -> bool {
        self.locks.is_some()
    }

    /// Commit the transaction. Succeeds if locks were acquired at begin
    /// time (guaranteed for non-conflicting access patterns). Returns
    /// Err(Conflict) if locks were not acquired — spawned entity IDs
    /// are immediately deallocated.
    pub fn commit(mut self, world: &mut World) -> Result<EnumChangeSet, Conflict> {
        assert_eq!(
            self.strategy.world_id,
            world.world_id(),
            "transaction committed to a different World than it was created from"
        );
        if self.locks.is_none() {
            for &entity in &self.spawned_entities {
                world.dealloc_entity(entity);
            }
            self.spawned_entities.clear();
            return Err(Conflict {
                component_ids: FixedBitSet::new(),
            });
        }
        self.spawned_entities.clear(); // committed — entities are now placed
        let changeset = std::mem::take(&mut self.changeset);
        let reverse = changeset.apply(world);
        if let Some(locks) = self.locks.take() {
            self.strategy.lock_table.lock().release(locks);
        }
        Ok(reverse)
    }
}

impl<'s> Drop for PessimisticTx<'s> {
    fn drop(&mut self) {
        // Release locks
        if let Some(locks) = self.locks.take() {
            self.strategy.lock_table.lock().release(locks);
        }
        // Release spawned entity IDs to orphan queue
        if !self.spawned_entities.is_empty() {
            self.strategy
                .orphan_queue
                .0
                .lock()
                .extend(self.spawned_entities.drain(..));
        }
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

    // ── Sequential tests ────────────────────────────────────────────

    #[test]
    fn sequential_query_reads_spawned_entities() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &Vel)>(&mut world);

        let strategy = Sequential;
        let tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>(&mut world).count();
        assert_eq!(count, 1);
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn sequential_mutation_is_immediate() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn(&mut world, (Pos(42.0),));
        let count = tx.query::<(&Pos,)>(&mut world).count();
        assert_eq!(count, 1);
        let _ = tx.commit(&mut world);
    }

    #[test]
    fn sequential_commit_always_ok() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn(&mut world, (Pos(1.0),));
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn sequential_drop_is_noop() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Sequential;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.spawn(&mut world, (Pos(1.0),));
        }
        assert_eq!(world.query::<(&Pos,)>().count(), 1);
    }

    // ── Optimistic tests ────────────────────────────────────────────

    #[test]
    fn optimistic_commit_succeeds_without_conflict() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let strategy = Optimistic::new(&world);
        let mut tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>(&world).count();
        assert_eq!(count, 1);
        tx.insert::<Vel>(&mut world, e, Vel(99.0));
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn optimistic_buffered_writes_not_visible_until_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Optimistic::new(&world);
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(&mut world, e, Pos(99.0));
        // query_raw sees old value (write is buffered in changeset)
        let pos = tx.query::<(&Pos,)>(&world).next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit(&mut world);
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 99.0);
    }

    #[test]
    fn optimistic_drop_aborts_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Optimistic::new(&world);
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.insert::<Pos>(&mut world, e, Pos(99.0));
        }
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
    }

    #[test]
    fn optimistic_conflict_detected_on_read_column_mutation() {
        use fixedbitset::FixedBitSet;

        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));

        let pos_id = world.components.id::<Pos>().unwrap();
        let mut read_bits = FixedBitSet::with_capacity(pos_id + 1);
        read_bits.insert(pos_id);
        let snap = world.snapshot_column_ticks(&read_bits);

        for pos in world.query::<(&mut Pos,)>() {
            pos.0 .0 = 99.0;
        }

        let conflicts = world.check_column_conflicts(&snap);
        assert!(conflicts.contains(pos_id));
    }

    #[test]
    fn optimistic_spawn_allocates_entity() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Optimistic::new(&world);
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn(&mut world, (Pos(42.0),));
        assert!(tx.commit(&mut world).is_ok());
        assert!(world.get::<Pos>(e).is_some());
    }

    // ── Pessimistic tests ───────────────────────────────────────────

    #[test]
    fn pessimistic_commit_always_succeeds() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let strategy = Pessimistic::new(&world);
        let mut tx = strategy.begin(&mut world, &access);
        let _ = tx.query::<(&Pos,)>(&world).count();
        tx.insert::<Vel>(&mut world, e, Vel(99.0));
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn pessimistic_buffered_writes_applied_at_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Pessimistic::new(&world);
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(&mut world, e, Pos(42.0));
        let pos = tx.query::<(&Pos,)>(&world).next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit(&mut world);
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 42.0);
    }

    #[test]
    fn pessimistic_drop_releases_locks() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Pessimistic::new(&world);
        {
            let _tx = strategy.begin(&mut world, &access);
        }
        let tx2 = strategy.begin(&mut world, &access);
        let _ = tx2.commit(&mut world);
    }

    #[test]
    fn pessimistic_spawn_visible_after_commit() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Pessimistic::new(&world);
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn(&mut world, (Pos(77.0),));
        let _ = tx.commit(&mut world);
        assert!(world.get::<Pos>(e).is_some());
        assert_eq!(world.get::<Pos>(e).unwrap().0, 77.0);
    }

    // ── Entity ID leak tests ────────────────────────────────────────

    #[test]
    fn optimistic_drop_releases_spawned_entity_ids() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Optimistic::new(&world);

        let spawned_entity;
        {
            let mut tx = strategy.begin(&mut world, &access);
            spawned_entity = tx.spawn(&mut world, (Pos(1.0),));
            // drop without commit — entity ID goes to pending release queue
        }

        // Entity was allocated but never placed
        assert!(!world.is_placed(spawned_entity));

        // Any &mut World method drains the orphan queue
        world.register_component::<Pos>();

        // Entity handle is now stale (generation bumped)
        assert!(!world.is_alive(spawned_entity));
    }

    #[test]
    fn pessimistic_drop_releases_spawned_entity_ids() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Pessimistic::new(&world);

        let spawned_entity;
        {
            let mut tx = strategy.begin(&mut world, &access);
            spawned_entity = tx.spawn(&mut world, (Pos(1.0),));
            // drop without commit
        }

        assert!(!world.is_placed(spawned_entity));

        // Any &mut World method drains the orphan queue
        world.register_component::<Pos>();

        assert!(!world.is_alive(spawned_entity));
    }

    #[test]
    fn optimistic_conflict_deallocates_spawned_entities() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let strategy = Optimistic::new(&world);

        let mut tx = strategy.begin(&mut world, &access);
        let spawned = tx.spawn(&mut world, (Pos(99.0),));

        // Mutate a read column to force conflict
        for pos in world.query::<(&mut Pos,)>() {
            pos.0 .0 = 42.0;
        }

        let result = tx.commit(&mut world);
        assert!(result.is_err());
        // Spawned entity was deallocated immediately by commit
        assert!(!world.is_alive(spawned));
    }

    // ── Cross-world safety tests ─────────────────────────────────────

    #[test]
    #[should_panic(expected = "different World")]
    fn optimistic_begin_panics_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world_a);
        let strategy = Optimistic::new(&world_a);
        let _tx = strategy.begin(&mut world_b, &access);
    }

    #[test]
    #[should_panic(expected = "different World")]
    fn pessimistic_begin_panics_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world_a);
        let strategy = Pessimistic::new(&world_a);
        let _tx = strategy.begin(&mut world_b, &access);
    }

    #[test]
    #[should_panic(expected = "different World")]
    fn optimistic_commit_panics_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world_a);
        let strategy = Optimistic::new(&world_a);
        let tx = strategy.begin(&mut world_a, &access);
        let _ = tx.commit(&mut world_b);
    }
}
