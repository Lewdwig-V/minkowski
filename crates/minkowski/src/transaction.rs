//! Transaction strategies for concurrent ECS access.
//!
//! # Design overview
//!
//! The [`Transact`] trait provides optimistic and pessimistic concurrency
//! behind one closure-based interface with automatic retry. [`Sequential`]
//! exists as a zero-cost passthrough for code that doesn't need isolation —
//! users who never touch transactions pay nothing.
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
//! — the IDs must be reclaimed. `Drop` pushes them to an `OrphanQueue`
//! shared with World via `Arc<Mutex<Vec<Entity>>>`. World drains this queue
//! automatically at the top of every `&mut self` method, bumping generations
//! and recycling indices. No entity ID ever leaks, regardless of how the
//! transaction ends, and no manual cleanup step is required.
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
//! Each World gets a unique `WorldId` at construction. Strategies capture it
//! and assert it matches in `begin()` and `commit()`. This prevents a strategy
//! created from world A being used with world B, which would push aborted
//! entity IDs into the wrong orphan queue and corrupt unrelated live entities.

use crate::sync::{AtomicU64, Mutex};

use fixedbitset::FixedBitSet;

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::lock_table::{ColumnLockSet, ColumnLockTable};
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
use crate::query::iter::QueryIter;
use crate::world::{World, WorldId};

/// Unique transaction identifier, monotonically increasing per strategy.
pub type TxId = u64;

/// Returned when an optimistic transaction commit fails validation.
///
/// Contains the set of component columns that had conflicting concurrent
/// modifications. Returned by [`Transact::transact`] after exhausting retries,
/// or by [`Transact::try_commit`] on a single failed attempt. See [`Optimistic`].
#[derive(Debug)]
pub struct Conflict {
    /// Which component columns had conflicting concurrent modifications.
    pub component_ids: FixedBitSet,
}

impl Conflict {
    /// Format the conflict with resolved component names from the World.
    ///
    /// Produces a human-readable message like:
    /// `"transaction conflict on [Pos, Vel] — try increasing retries or switching to Pessimistic"`
    pub fn display_with(&self, world: &World) -> String {
        let names: Vec<&str> = self
            .component_ids
            .ones()
            .filter_map(|id| world.component_name(id))
            .collect();
        if names.is_empty() {
            return format!("{self}");
        }
        format!(
            "transaction conflict on [{}] — \
             try increasing retries or switching to Pessimistic strategy",
            names.join(", ")
        )
    }
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ids: Vec<usize> = self.component_ids.ones().collect();
        write!(
            f,
            "transaction conflict on component columns {ids:?} — \
             try increasing retries or switching to Pessimistic strategy"
        )
    }
}

impl std::error::Error for Conflict {}

// ── TxCleanup + Tx ─────────────────────────────────────────────────

/// Strategy-specific teardown logic invoked when a [`Tx`] is dropped
/// without committing (abort path). Strategies implement this to release
/// resources they acquired at begin time (e.g. column locks).
///
/// Also provides extraction methods for strategy-specific validation state
/// so that [`Transact::try_commit`] can access it without `Any` downcasting
/// (which would require `'static` and conflict with borrowed cleanup types).
pub(crate) trait TxCleanup {
    fn abort(&mut self);

    /// Extract the read tick snapshot (optimistic validation).
    /// Only `OptimisticCleanup` overrides this; others return `None`.
    fn take_read_ticks(
        &mut self,
    ) -> Option<Vec<(usize, crate::component::ComponentId, crate::tick::Tick)>> {
        None
    }

    /// Extract acquired column locks (pessimistic commit).
    /// Only `PessimisticCleanup` overrides this; others return `None`.
    fn take_locks(&mut self) -> Option<ColumnLockSet> {
        None
    }

    /// Check whether locks were acquired without consuming them.
    /// Only `PessimisticCleanup` overrides this; others return `false`.
    fn has_locks(&self) -> bool {
        false
    }

    /// Whether the transaction is ready to execute the closure.
    /// Returns `true` for optimistic (validation at commit) and `false`
    /// for pessimistic when lock acquisition failed. Callers should skip
    /// the closure and retry when this returns `false`.
    fn is_ready(&self) -> bool {
        true
    }
}

/// No-op cleanup for strategies that have nothing to release on abort.
#[allow(dead_code)]
pub(crate) struct NoopCleanup;

impl TxCleanup for NoopCleanup {
    fn abort(&mut self) {}
}

/// Cleanup for [`Optimistic`] transactions via the [`Transact`] trait.
/// Stores the read tick snapshot captured at begin time. Abort is a no-op
/// (no locks to release). `take_read_ticks` extracts the snapshot for
/// validation at commit time.
pub(crate) struct OptimisticCleanup {
    read_ticks: Option<Vec<(usize, crate::component::ComponentId, crate::tick::Tick)>>,
}

impl TxCleanup for OptimisticCleanup {
    fn abort(&mut self) {}

    fn take_read_ticks(
        &mut self,
    ) -> Option<Vec<(usize, crate::component::ComponentId, crate::tick::Tick)>> {
        self.read_ticks.take()
    }
}

/// Cleanup for [`Pessimistic`] transactions via the [`Transact`] trait.
/// Stores acquired column locks and a reference to the lock table.
/// Abort releases locks back to the table. `take_locks` extracts
/// the lock set for the commit path.
pub(crate) struct PessimisticCleanup<'a> {
    locks: Option<ColumnLockSet>,
    lock_table: &'a Mutex<ColumnLockTable>,
}

impl TxCleanup for PessimisticCleanup<'_> {
    fn abort(&mut self) {
        if let Some(locks) = self.locks.take() {
            self.lock_table.lock().release(locks);
        }
    }

    fn take_locks(&mut self) -> Option<ColumnLockSet> {
        self.locks.take()
    }

    fn has_locks(&self) -> bool {
        self.locks.is_some()
    }

    fn is_ready(&self) -> bool {
        self.locks.is_some()
    }
}

/// Unified transaction handle. Holds a buffered [`EnumChangeSet`] and tracks
/// entity IDs allocated during the transaction.
///
/// Does **not** hold `&mut World` — methods take `&World` or `&mut World`
/// as parameters. This split-phase design enables concurrent reads:
/// [`query(&world)`](Tx::query) uses `World::query_raw` (shared-ref, no ticks/cache)
/// and requires [`ReadOnlyWorldQuery`] to prevent aliased `&mut T`.
/// Writes go through [`write`](Tx::write), [`remove`](Tx::remove), and
/// [`spawn`](Tx::spawn), which buffer into the internal changeset.
///
/// Drop without [`mark_committed`](Tx::mark_committed) triggers abort:
/// the cleanup callback runs and spawned entity IDs are pushed to the
/// orphan queue for reclamation. Strategy-specific teardown (e.g. lock
/// release for [`Pessimistic`]) is handled by an internal cleanup trait.
pub struct Tx<'a> {
    archetype_count: usize,
    changeset: EnumChangeSet,
    allocated: Vec<Entity>,
    orphan_queue: crate::world::OrphanQueue,
    cleanup: Box<dyn TxCleanup + Send + Sync + 'a>,
    committed: bool,
}

impl<'a> Tx<'a> {
    /// Create a new transaction with the given archetype snapshot size,
    /// orphan queue handle, and strategy-specific cleanup.
    #[allow(dead_code)]
    pub(crate) fn new(
        archetype_count: usize,
        orphan_queue: crate::world::OrphanQueue,
        cleanup: Box<dyn TxCleanup + Send + Sync + 'a>,
    ) -> Self {
        Self {
            archetype_count,
            changeset: EnumChangeSet::new(),
            allocated: Vec::new(),
            orphan_queue,
            cleanup,
            committed: false,
        }
    }

    /// Read through the transaction via shared World reference.
    /// No tick advancement, no cache mutation. Safe for concurrent reads.
    /// Only sees archetypes that existed at begin time.
    ///
    /// Requires [`ReadOnlyWorldQuery`] — `&mut T` queries are rejected at
    /// compile time. Writes go through [`write`](Tx::write)/[`remove`](Tx::remove)/[`spawn`](Tx::spawn) instead.
    pub fn query<'w, Q: ReadOnlyWorldQuery + 'static>(&self, world: &'w World) -> QueryIter<'w, Q> {
        world.query_raw::<Q>(self.archetype_count)
    }

    /// Read a single component from an entity via shared World reference.
    pub fn read<'w, T: Component>(&self, world: &'w World, entity: Entity) -> Option<&'w T> {
        world.get::<T>(entity)
    }

    /// Buffer a component write. The mutation is recorded in the internal
    /// changeset and will be applied when the strategy commits the transaction.
    pub fn write<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        self.changeset.insert::<T>(world, entity, value);
    }

    /// Buffer a component removal. The mutation is recorded in the internal
    /// changeset and will be applied when the strategy commits the transaction.
    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        self.changeset.remove::<T>(world, entity);
    }

    /// Buffer a sparse component write. The mutation is recorded in the internal
    /// changeset and will be applied when the strategy commits the transaction.
    pub fn write_sparse<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        self.changeset.insert_sparse::<T>(world, entity, value);
    }

    /// Buffer a sparse component removal.
    pub fn remove_sparse<T: Component>(&mut self, world: &mut World, entity: Entity) {
        self.changeset.remove_sparse::<T>(world, entity);
    }

    /// Allocate a new entity and buffer its initial components. The entity ID
    /// is tracked so it can be reclaimed on abort (drop without commit).
    pub fn spawn<B: crate::bundle::Bundle>(&mut self, world: &mut World, bundle: B) -> Entity {
        let entity = world.alloc_entity();
        self.allocated.push(entity);
        self.changeset.spawn_bundle(world, entity, bundle);
        entity
    }

    /// Check whether column locks were acquired at begin time.
    ///
    /// Only meaningful for [`Pessimistic`] transactions — returns `true`
    /// if locks were successfully acquired, `false` otherwise. For
    /// [`Optimistic`] transactions this always returns `false`.
    pub fn has_locks(&self) -> bool {
        self.cleanup.has_locks()
    }

    /// Whether the transaction is ready to execute the closure.
    /// Returns `false` for pessimistic transactions that failed to acquire
    /// locks — the closure should be skipped and the transaction retried.
    /// Always returns `true` for optimistic transactions.
    pub fn is_ready(&self) -> bool {
        self.cleanup.is_ready()
    }

    /// Track an externally-allocated entity ID for orphan reclamation on abort.
    #[allow(dead_code)]
    pub(crate) fn track_allocated(&mut self, entity: Entity) {
        self.allocated.push(entity);
    }

    /// Mark this transaction as committed, preventing abort cleanup on drop.
    /// Clears the allocated entity list (they are now placed).
    ///
    /// Call this after a successful [`Transact::try_commit`] when using
    /// the building-block API, before dropping the `Tx`.
    pub fn mark_committed(&mut self) {
        self.allocated.clear();
        self.committed = true;
    }

    /// Borrow the internal changeset.
    #[allow(dead_code)]
    pub(crate) fn changeset(&self) -> &EnumChangeSet {
        &self.changeset
    }

    /// Take ownership of the internal changeset, leaving an empty one in place.
    pub fn take_changeset(&mut self) -> EnumChangeSet {
        std::mem::take(&mut self.changeset)
    }

    /// Mutable borrow of the internal changeset. Used by typed reducer
    /// handles that buffer writes directly into the changeset.
    #[allow(dead_code)]
    pub(crate) fn changeset_mut(&mut self) -> &mut EnumChangeSet {
        &mut self.changeset
    }

    /// Mutable borrow of the allocated entity tracking list. Used by
    /// Spawner to register entity IDs for orphan reclamation on abort.
    #[allow(dead_code)]
    pub(crate) fn allocated_mut(&mut self) -> &mut Vec<Entity> {
        &mut self.allocated
    }

    /// Split borrow: returns mutable refs to both changeset and allocated
    /// list simultaneously. Needed by reducer adapters that construct both
    /// EntityMut (changeset) and Spawner (allocated) handles.
    #[allow(dead_code)]
    pub(crate) fn reducer_parts(&mut self) -> (&mut EnumChangeSet, &mut Vec<Entity>) {
        (&mut self.changeset, &mut self.allocated)
    }

    // ── Pre-resolved raw methods (pub(crate)) ────────────────────

    /// Write a component using a pre-resolved ComponentId.
    /// No registration, no `&mut World`.
    #[allow(dead_code)]
    pub(crate) fn write_raw<T: Component>(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
        value: T,
    ) {
        self.changeset.insert_raw(entity, comp_id, value);
    }

    /// Remove a component using a pre-resolved ComponentId.
    #[allow(dead_code)]
    pub(crate) fn remove_raw(&mut self, entity: Entity, comp_id: ComponentId) {
        self.changeset.record_remove(entity, comp_id);
    }

    /// Sparse write with a pre-resolved ComponentId.
    #[allow(dead_code)]
    pub(crate) fn write_sparse_raw<T: Component>(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
        value: T,
    ) {
        self.changeset.insert_sparse_raw(entity, comp_id, value);
    }

    /// Sparse remove with a pre-resolved ComponentId.
    #[allow(dead_code)]
    pub(crate) fn remove_sparse_raw(&mut self, entity: Entity, comp_id: ComponentId) {
        self.changeset.remove_sparse_raw(entity, comp_id);
    }

    /// Spawn an entity with a bundle using pre-resolved ComponentIds.
    /// Atomically reserves an entity ID via `EntityAllocator::reserve()`.
    #[allow(dead_code)]
    pub(crate) fn spawn_raw<B: crate::bundle::Bundle>(
        &mut self,
        world: &World,
        bundle: B,
    ) -> Entity {
        let entity = world.entities.reserve();
        self.track_allocated(entity);
        self.changeset
            .spawn_bundle_raw(entity, &world.components, bundle);
        entity
    }
}

impl Drop for Tx<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.cleanup.abort();
            if !self.allocated.is_empty() {
                self.orphan_queue.0.lock().extend(self.allocated.drain(..));
            }
        }
    }
}

// ── Sequential ──────────────────────────────────────────────────────

/// Zero-cost transaction strategy. All operations delegate directly to World.
/// Commit always succeeds. No read-set, no changeset buffering, no validation.
///
/// Use when systems run sequentially and no conflict detection is needed.
pub struct Sequential;

impl Sequential {
    /// Begin a sequential transaction. Returns a zero-state unit struct
    /// that delegates all operations directly to World.
    pub fn begin(&self, _world: &mut World, _access: &Access) -> SequentialTx {
        SequentialTx
    }
}

/// Transaction handle for the [`Sequential`] strategy.
///
/// Zero-state unit struct — all methods delegate directly to [`World`].
/// No buffering, no validation, no conflict detection. Commit always succeeds
/// and returns an empty reverse changeset.
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

    pub fn insert<B: crate::bundle::Bundle>(
        &mut self,
        world: &mut World,
        entity: Entity,
        bundle: B,
    ) {
        world.insert(entity, bundle);
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
    #[allow(dead_code)]
    next_tx_id: AtomicU64,
    orphan_queue: crate::world::OrphanQueue,
    max_retries: usize,
}

impl Optimistic {
    pub fn new(world: &World) -> Self {
        Self::with_retries(world, 3)
    }

    pub fn with_retries(world: &World, max_retries: usize) -> Self {
        Self {
            world_id: world.world_id(),
            next_tx_id: AtomicU64::new(1),
            orphan_queue: world.orphan_queue(),
            max_retries,
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
    #[allow(dead_code)]
    next_tx_id: AtomicU64,
    orphan_queue: crate::world::OrphanQueue,
    max_retries: usize,
}

impl Pessimistic {
    pub fn new(world: &World) -> Self {
        Self::with_retries(world, 64)
    }

    pub fn with_retries(world: &World, max_retries: usize) -> Self {
        Self {
            world_id: world.world_id(),
            lock_table: Mutex::new(ColumnLockTable::new()),
            next_tx_id: AtomicU64::new(1),
            orphan_queue: world.orphan_queue(),
            max_retries,
        }
    }
}

// ── Transact (closure-based API) ─────────────────────────────────

/// Exponential backoff for retry loops. Spin-waits for small attempt
/// counts, yields the thread for larger ones.
fn backoff(attempt: usize) {
    if attempt < 6 {
        for _ in 0..(4 << attempt) {
            std::hint::spin_loop();
        }
    } else {
        crate::sync::yield_now();
    }
}

/// Closure-based transaction API. Wraps the begin/execute/commit cycle
/// in a retry loop, handling abort cleanup automatically.
///
/// Strategies implement [`begin`](Transact::begin) (setup) and
/// [`try_commit`](Transact::try_commit) (validate + extract changeset).
/// The default [`transact`](Transact::transact) method runs the closure
/// in a retry loop, applying the changeset on success.
///
/// # Example
///
/// ```ignore
/// let strategy = Optimistic::new(&world);
/// let access = Access::of::<(&Pos, &mut Vel)>(&mut world);
/// strategy.transact(&mut world, &access, |tx, world| {
///     for (pos,) in tx.query::<(&Pos,)>(world) {
///         // read ...
///     }
///     tx.write::<Vel>(world, entity, Vel(1.0));
/// }).unwrap();
/// ```
pub trait Transact {
    /// Run a closure inside a transaction, retrying on conflict up to
    /// [`max_retries`](Transact::max_retries) times.
    ///
    /// On success: drops the transaction (committed), applies the forward
    /// changeset to the world, and returns `Ok(value)`.
    /// On conflict: drops the transaction (abort cleanup runs), retries.
    /// After exhausting retries: returns `Err(Conflict)`.
    fn transact<R>(
        &self,
        world: &mut World,
        access: &Access,
        f: impl FnMut(&mut Tx<'_>, &mut World) -> R,
    ) -> Result<R, Conflict> {
        self.transact_inner(world, access, f)
    }

    /// Default retry loop. Strategies that need custom behavior between
    /// retries (e.g. backoff) override [`transact`](Transact::transact) instead.
    fn transact_inner<R>(
        &self,
        world: &mut World,
        access: &Access,
        mut f: impl FnMut(&mut Tx<'_>, &mut World) -> R,
    ) -> Result<R, Conflict> {
        let mut last_conflict = None;
        for _ in 0..self.max_retries() {
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
                    forward
                        .apply(world)
                        .expect("changeset apply after successful commit");
                    return Ok(value);
                }
                Err(conflict) => {
                    last_conflict = Some(conflict);
                    drop(tx);
                    continue;
                }
            }
        }
        Err(last_conflict.unwrap_or_else(|| Conflict {
            component_ids: FixedBitSet::new(),
        }))
    }

    /// Begin a transaction. Snapshots ticks (optimistic) or acquires
    /// locks (pessimistic). Returns a [`Tx`] with strategy-specific cleanup.
    fn begin(&self, world: &mut World, access: &Access) -> Tx<'_>;

    /// Validate and extract the changeset. Does NOT apply it — the caller
    /// (typically the default `transact` impl) applies the returned
    /// changeset after dropping the `Tx`.
    ///
    /// On success: returns the forward changeset (marks the tx for commit
    /// is the caller's responsibility via `tx.mark_committed()`).
    /// On failure: returns `Err(Conflict)`. The tx will abort on drop.
    fn try_commit(&self, tx: &mut Tx<'_>, world: &mut World) -> Result<EnumChangeSet, Conflict>;

    /// Maximum number of retry attempts before returning `Err(Conflict)`.
    fn max_retries(&self) -> usize;
}

impl Transact for Optimistic {
    fn begin(&self, world: &mut World, access: &Access) -> Tx<'_> {
        assert_eq!(
            self.world_id,
            world.world_id(),
            "strategy used with a different World than it was created from"
        );
        let mut accessed = access.reads().clone();
        accessed.union_with(access.writes());
        let read_ticks = world.snapshot_column_ticks(&accessed);
        let archetype_count = world.archetypes.archetypes.len();
        Tx::new(
            archetype_count,
            self.orphan_queue.clone(),
            Box::new(OptimisticCleanup {
                read_ticks: Some(read_ticks),
            }),
        )
    }

    fn try_commit(&self, tx: &mut Tx<'_>, world: &mut World) -> Result<EnumChangeSet, Conflict> {
        assert_eq!(
            self.world_id,
            world.world_id(),
            "strategy used with a different World than it was created from"
        );
        let read_ticks = tx
            .cleanup
            .take_read_ticks()
            .expect("OptimisticCleanup missing read_ticks");
        let conflicts = world.check_column_conflicts(&read_ticks);
        if !conflicts.is_empty() {
            return Err(Conflict {
                component_ids: conflicts,
            });
        }
        Ok(tx.take_changeset())
    }

    fn max_retries(&self) -> usize {
        self.max_retries
    }
}

impl Transact for Pessimistic {
    fn transact<R>(
        &self,
        world: &mut World,
        access: &Access,
        mut f: impl FnMut(&mut Tx<'_>, &mut World) -> R,
    ) -> Result<R, Conflict> {
        let mut last_conflict = None;
        for attempt in 0..self.max_retries() {
            let mut tx = Transact::begin(self, world, access);
            if !tx.is_ready() {
                // Lock acquisition failed — abort and retry with backoff
                drop(tx);
                backoff(attempt);
                continue;
            }
            let value = f(&mut tx, world);
            match self.try_commit(&mut tx, world) {
                Ok(forward) => {
                    tx.mark_committed();
                    drop(tx);
                    forward
                        .apply(world)
                        .expect("changeset apply after successful commit");
                    return Ok(value);
                }
                Err(conflict) => {
                    last_conflict = Some(conflict);
                    drop(tx);
                    backoff(attempt);
                    continue;
                }
            }
        }
        Err(last_conflict.unwrap_or_else(|| Conflict {
            component_ids: FixedBitSet::new(),
        }))
    }

    fn begin(&self, world: &mut World, access: &Access) -> Tx<'_> {
        assert_eq!(
            self.world_id,
            world.world_id(),
            "strategy used with a different World than it was created from"
        );
        let lock_result = self.lock_table.lock().acquire(
            &world.archetypes.archetypes,
            access.reads(),
            access.writes(),
        );
        let locks = lock_result.ok();
        let archetype_count = world.archetypes.archetypes.len();
        Tx::new(
            archetype_count,
            self.orphan_queue.clone(),
            Box::new(PessimisticCleanup {
                locks,
                lock_table: &self.lock_table,
            }),
        )
    }

    fn try_commit(&self, tx: &mut Tx<'_>, world: &mut World) -> Result<EnumChangeSet, Conflict> {
        assert_eq!(
            self.world_id,
            world.world_id(),
            "strategy used with a different World than it was created from"
        );
        let locks = tx.cleanup.take_locks();
        if let Some(locks) = locks {
            let changeset = tx.take_changeset();
            self.lock_table.lock().release(locks);
            Ok(changeset)
        } else {
            Err(Conflict {
                component_ids: FixedBitSet::new(),
            })
        }
    }

    fn max_retries(&self) -> usize {
        self.max_retries
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
    #[derive(Clone, Copy, PartialEq, Debug)]
    #[allow(dead_code)]
    struct Health(u32);

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

    // ── Entity ID leak tests ────────────────────────────────────────

    #[test]
    fn optimistic_drop_releases_spawned_entity_ids() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Optimistic::new(&world);

        let spawned_entity;
        {
            let mut tx = Transact::begin(&strategy, &mut world, &access);
            spawned_entity = tx.spawn(&mut world, (Pos(1.0),));
            // drop without commit — entity ID goes to orphan queue via Tx Drop
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
            let mut tx = Transact::begin(&strategy, &mut world, &access);
            spawned_entity = tx.spawn(&mut world, (Pos(1.0),));
            // drop without commit — entity ID goes to orphan queue via Tx Drop
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

        let mut tx = Transact::begin(&strategy, &mut world, &access);
        let spawned = tx.spawn(&mut world, (Pos(99.0),));

        // Mutate a read column to force conflict
        for pos in world.query::<(&mut Pos,)>() {
            pos.0 .0 = 42.0;
        }

        let result = strategy.try_commit(&mut tx, &mut world);
        assert!(result.is_err());
        // Tx will abort on drop, pushing spawned entity to orphan queue
        drop(tx);
        // Drain orphan queue
        world.register_component::<Pos>();
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
        let _tx = Transact::begin(&strategy, &mut world_b, &access);
    }

    #[test]
    #[should_panic(expected = "different World")]
    fn pessimistic_begin_panics_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world_a);
        let strategy = Pessimistic::new(&world_a);
        let _tx = Transact::begin(&strategy, &mut world_b, &access);
    }

    #[test]
    #[should_panic(expected = "different World")]
    fn optimistic_commit_panics_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world_a);
        let strategy = Optimistic::new(&world_a);
        let mut tx = Transact::begin(&strategy, &mut world_a, &access);
        let _ = strategy.try_commit(&mut tx, &mut world_b);
    }

    // ── Tx (unified) tests ──────────────────────────────────────────

    #[test]
    fn tx_query_reads_via_query_raw() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let archetype_count = world.archetypes.archetypes.len();
        let orphan_queue = world.orphan_queue();

        let tx = Tx::new(archetype_count, orphan_queue, Box::new(NoopCleanup));
        let count = tx.query::<(&Pos,)>(&world).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn tx_write_buffers_in_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let archetype_count = world.archetypes.archetypes.len();
        let orphan_queue = world.orphan_queue();

        let mut tx = Tx::new(archetype_count, orphan_queue, Box::new(NoopCleanup));
        tx.write::<Pos>(&mut world, e, Pos(99.0));
        // The write is buffered — world still has old value
        assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
        // Changeset is non-empty
        assert!(tx.changeset().iter_mutations().count() > 0);
        tx.mark_committed();
    }

    #[test]
    fn tx_drop_pushes_allocated_to_orphan_queue() {
        let mut world = World::new();
        let archetype_count = world.archetypes.archetypes.len();
        let orphan_queue = world.orphan_queue();

        let spawned;
        {
            let mut tx = Tx::new(archetype_count, orphan_queue, Box::new(NoopCleanup));
            spawned = tx.spawn(&mut world, (Pos(1.0),));
            // drop without mark_committed — entity goes to orphan queue
        }

        assert!(!world.is_placed(spawned));
        // Any &mut World method drains the orphan queue
        world.register_component::<Pos>();
        assert!(!world.is_alive(spawned));
    }

    #[test]
    fn tx_committed_flag_prevents_orphan_push() {
        let mut world = World::new();
        let archetype_count = world.archetypes.archetypes.len();
        let orphan_queue = world.orphan_queue();

        let spawned;
        {
            let mut tx = Tx::new(archetype_count, orphan_queue, Box::new(NoopCleanup));
            spawned = tx.spawn(&mut world, (Pos(1.0),));
            tx.mark_committed();
            // drop after mark_committed — entity NOT pushed to orphan queue
        }

        // Entity was allocated (alloc_entity) so it exists in the allocator.
        // Since we marked committed, the orphan queue should be empty.
        // Drain orphans and confirm the entity is still alive.
        world.register_component::<Vel>();
        assert!(world.is_alive(spawned));
    }

    // ── Transact (closure API) tests ──────────────────────────────────

    #[test]
    fn optimistic_transact_returns_closure_value() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);
        let strategy = Optimistic::new(&world);
        let count = strategy.transact(&mut world, &access, |tx, world| {
            tx.query::<(&Pos,)>(world).count()
        });
        assert_eq!(count.unwrap(), 1);
    }

    #[test]
    fn optimistic_transact_applies_writes() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Optimistic::new(&world);
        strategy
            .transact(&mut world, &access, |tx, world| {
                tx.write::<Pos>(world, e, Pos(42.0));
            })
            .unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 42.0);
    }

    #[test]
    fn optimistic_transact_retries_on_conflict() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let strategy = Optimistic::with_retries(&world, 3);
        let mut attempt = 0u32;
        let result = strategy.transact(&mut world, &access, |tx, world| {
            attempt += 1;
            if attempt == 1 {
                // Mutate a read column to force conflict on first attempt
                for pos in world.query::<(&mut Pos,)>() {
                    pos.0 .0 = 99.0;
                }
            }
            tx.write::<Pos>(world, e, Pos(42.0));
        });
        assert!(result.is_ok());
        assert!(attempt >= 2, "should have retried at least once");
    }

    #[test]
    fn optimistic_transact_exhausted_retries() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let strategy = Optimistic::with_retries(&world, 2);
        let result = strategy.transact(&mut world, &access, |_tx, world| {
            // Every attempt mutates the read column — conflict every time
            for pos in world.query::<(&mut Pos,)>() {
                pos.0 .0 += 1.0;
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn pessimistic_transact_returns_closure_value() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);
        let strategy = Pessimistic::new(&world);
        let count = strategy.transact(&mut world, &access, |tx, world| {
            tx.query::<(&Pos,)>(world).count()
        });
        assert_eq!(count.unwrap(), 1);
    }

    #[test]
    fn pessimistic_transact_applies_writes() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Pessimistic::new(&world);
        strategy
            .transact(&mut world, &access, |tx, world| {
                tx.write::<Pos>(world, e, Pos(77.0));
            })
            .unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 77.0);
    }

    #[test]
    fn transact_entity_cleanup_on_retry() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let strategy = Optimistic::with_retries(&world, 3);
        let mut spawned_on_first = None;
        let mut attempt = 0u32;
        let _result = strategy.transact(&mut world, &access, |tx, world| {
            attempt += 1;
            let e = tx.spawn(world, (Pos(0.0),));
            if attempt == 1 {
                spawned_on_first = Some(e);
                // Mutate a read column to force conflict on first attempt
                for pos in world.query::<(&mut Pos,)>() {
                    pos.0 .0 = 99.0;
                }
            }
        });
        if let Some(e) = spawned_on_first {
            assert!(
                !world.is_alive(e),
                "entity from failed attempt should be deallocated"
            );
        }
    }

    #[test]
    #[should_panic(expected = "different World")]
    fn transact_panics_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world_a);
        let strategy = Optimistic::new(&world_a);
        let _ = strategy.transact(&mut world_b, &access, |_tx, _world| {});
    }

    // ── Conflict display tests ──────────────────────────────────────

    #[test]
    fn conflict_display_shows_component_ids() {
        let mut bits = FixedBitSet::with_capacity(8);
        bits.insert(1);
        bits.insert(5);
        let conflict = Conflict {
            component_ids: bits,
        };
        let msg = format!("{conflict}");
        assert!(msg.contains("[1, 5]"), "should list component IDs: {msg}");
        assert!(
            msg.contains("Pessimistic"),
            "should suggest alternative: {msg}"
        );
    }

    #[test]
    fn conflict_display_with_resolves_names() {
        let mut world = World::new();
        world.register_component::<Pos>();
        world.register_component::<Vel>();
        let mut bits = FixedBitSet::with_capacity(2);
        bits.insert(0);
        bits.insert(1);
        let conflict = Conflict {
            component_ids: bits,
        };
        let msg = conflict.display_with(&world);
        assert!(msg.contains("Pos"), "should resolve Pos: {msg}");
        assert!(msg.contains("Vel"), "should resolve Vel: {msg}");
        assert!(
            msg.contains("Pessimistic"),
            "should suggest alternative: {msg}"
        );
    }

    #[test]
    fn conflict_is_std_error() {
        let conflict = Conflict {
            component_ids: FixedBitSet::new(),
        };
        // Verify Conflict implements std::error::Error (compiles = passes)
        let _: &dyn std::error::Error = &conflict;
    }

    #[test]
    fn conflict_display_with_empty_names() {
        let world = World::new();
        let conflict = Conflict {
            component_ids: FixedBitSet::new(),
        };
        let msg = conflict.display_with(&world);
        // No registered components → falls back to Display impl
        assert!(
            msg.contains("component columns"),
            "should use Display fallback: {msg}"
        );
    }

    // ── Tx method coverage ──────────────────────────────────────────

    #[test]
    fn tx_read_returns_component() {
        let mut world = World::new();
        let e = world.spawn((Pos(42.0),));
        let access = Access::of::<(&Pos,)>(&mut world);
        let strategy = Optimistic::new(&world);
        strategy
            .transact(&mut world, &access, |tx, world| {
                let p = tx.read::<Pos>(world, e);
                assert_eq!(p.unwrap().0, 42.0);
            })
            .unwrap();
    }

    #[test]
    fn tx_remove_buffers_removal() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&mut Pos, &mut Vel)>(&mut world);
        let strategy = Optimistic::new(&world);
        strategy
            .transact(&mut world, &access, |tx, world| {
                tx.remove::<Vel>(world, e);
            })
            .unwrap();
        assert!(world.get::<Vel>(e).is_none());
        assert!(world.get::<Pos>(e).is_some());
    }

    #[test]
    fn tx_has_locks_false_for_optimistic() {
        let mut world = World::new();
        let access = Access::of::<(&Pos,)>(&mut world);
        let strategy = Optimistic::new(&world);
        strategy
            .transact(&mut world, &access, |tx, _world| {
                assert!(!tx.has_locks());
            })
            .unwrap();
    }

    #[test]
    fn tx_has_locks_true_for_pessimistic() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Pessimistic::new(&world);
        strategy
            .transact(&mut world, &access, |tx, _world| {
                assert!(tx.has_locks());
            })
            .unwrap();
    }

    // ── SequentialTx method coverage ────────────────────────────────

    #[test]
    fn sequential_tx_insert_remove_get_mut() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos, &mut Vel)>(&mut world);

        let strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert(&mut world, e, (Vel(5.0),));
        assert!(world.get::<Vel>(e).is_some());

        {
            let p = tx.get_mut::<Pos>(&mut world, e).unwrap();
            p.0 = 99.0;
        }
        assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);

        tx.remove::<Vel>(&mut world, e);
        assert!(world.get::<Vel>(e).is_none());

        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn sequential_tx_despawn() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        assert!(tx.despawn(&mut world, e));
        assert!(!world.is_alive(e));
        assert!(tx.commit(&mut world).is_ok());
    }

    // ── Pessimistic retry coverage ──────────────────────────────────

    #[test]
    fn pessimistic_exhausted_retries_returns_conflict() {
        // Single retry with a failing lock scenario: two overlapping accesses
        // on the same strategy instance. We simulate exhaustion by using
        // max_retries=1 and forcing the lock to be held.
        use std::sync::Arc;

        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let strategy = Arc::new(Pessimistic::with_retries(&world, 1));

        // Hold locks via begin(), then try to transact — should exhaust retries
        let tx = strategy.begin(&mut world, &access);
        assert!(tx.has_locks());

        let result = strategy.transact(&mut world, &access, |_tx, _world| {});
        assert!(result.is_err(), "should fail with locks held");
        drop(tx);
    }

    #[test]
    fn optimistic_sparse_write_commits() {
        let mut world = World::new();
        let strategy = Optimistic::new(&world);
        let e = world.spawn((Vel(1.0),));
        let access = Access::empty();

        let result = strategy.transact(&mut world, &access, |tx, world| {
            tx.write_sparse::<Health>(world, e, Health(99));
        });
        assert!(result.is_ok());
        assert_eq!(world.get::<Health>(e), Some(&Health(99)));
    }

    #[test]
    fn pessimistic_sparse_remove_commits() {
        let mut world = World::new();
        let strategy = Pessimistic::new(&world);
        let e = world.spawn((Vel(1.0),));
        world.insert_sparse(e, Health(50));
        let access = Access::empty();

        let result = strategy.transact(&mut world, &access, |tx, world| {
            tx.remove_sparse::<Health>(world, e);
        });
        assert!(result.is_ok());
        assert_eq!(world.get::<Health>(e), None);
    }
}
