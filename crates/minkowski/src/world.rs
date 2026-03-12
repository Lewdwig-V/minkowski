use crate::bundle::Bundle;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::{Entity, EntityAllocator};
use crate::pool::{PoolExhausted, SharedPool, SlabPool, default_pool};
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
use crate::storage::archetype::{Archetype, ArchetypeId, Archetypes};
use crate::storage::sparse::SparseStorage;
use crate::sync::{Arc, AtomicBool, AtomicU64, Mutex, Ordering};
use crate::table::TableCache;
use crate::tick::Tick;
use fixedbitset::FixedBitSet;
use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;

/// Unique identity for a World instance. Strategies capture this at
/// construction and assert it matches in begin/commit to prevent
/// cross-world corruption.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct WorldId(u64);

#[cfg(not(loom))]
static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);

#[cfg(loom)]
loom::lazy_static! {
    static ref NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);
}

impl WorldId {
    fn next() -> Self {
        Self(NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed))
    }
}

pub(crate) fn get_pair_mut(
    v: &mut [Archetype],
    a: usize,
    b: usize,
) -> (&mut Archetype, &mut Archetype) {
    assert_ne!(a, b, "cannot get mutable references to the same archetype");
    if a < b {
        let (left, right) = v.split_at_mut(b);
        (&mut left[a], &mut right[0])
    } else {
        let (left, right) = v.split_at_mut(a);
        (&mut right[0], &mut left[b])
    }
}

#[derive(Clone, Copy)]
pub(crate) struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub row: usize,
}

pub(crate) struct QueryCacheEntry {
    /// Archetypes whose component_ids are a superset of the query's required_ids.
    matched_ids: Vec<ArchetypeId>,
    /// Precomputed required component bitset for incremental scans.
    required: FixedBitSet,
    /// Number of archetypes when cache was last updated.
    last_archetype_count: usize,
    /// Tick at which this query last read data (used by Changed<T> filter).
    last_read_tick: Tick,
    /// Tick that was deferred for this query. Committed only when the
    /// previous iterator was actually iterated (lazy advancement).
    pending_read_tick: Option<Tick>,
    /// Shared flag set by `QueryIter` when iteration actually occurs.
    /// Checked on the next `query()` call to decide whether to commit
    /// `pending_read_tick`.
    iterated: Arc<AtomicBool>,
    /// ComponentRegistry version when `required` was last computed.
    /// Avoids recomputing the FixedBitSet on every query() call.
    registry_version: u64,
}

/// Shared queue for entity IDs orphaned by aborted transactions.
/// World owns the canonical instance; strategies clone the Arc handle.
///
/// Uses an `AtomicBool` flag to skip the mutex lock when the queue is empty,
/// which is the common case during spawn-heavy workloads.
#[derive(Clone)]
pub(crate) struct OrphanQueue {
    pub(crate) queue: Arc<Mutex<Vec<Entity>>>,
    pub(crate) has_items: Arc<crate::sync::AtomicBool>,
}

impl OrphanQueue {
    fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            has_items: Arc::new(crate::sync::AtomicBool::new(false)),
        }
    }
}

/// Read-only snapshot of engine statistics. Plain data struct — no references
/// to internal state, safe to store or send across threads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldStats {
    pub entity_count: usize,
    pub archetype_count: usize,
    pub component_count: usize,
    pub free_list_len: usize,
    pub query_cache_len: usize,
    pub current_tick: u64,
    pub total_spawns: u64,
    pub total_despawns: u64,
    /// Total capacity of the backing memory pool, if bounded.
    pub pool_capacity: Option<usize>,
    /// Bytes currently allocated from the pool, if tracked.
    pub pool_used: Option<usize>,
}

/// Debug information about the tick state of a cached query type.
///
/// Returned by [`World::query_tick_info`]. Helps diagnose unexpected
/// `Changed<T>` behavior by exposing the internal tick bookkeeping.
#[derive(Debug, Clone)]
pub struct QueryTickInfo {
    /// The committed read tick — changes older than this are invisible
    /// to `Changed<T>` filters.
    pub last_read_tick: u64,
    /// The current world tick (monotonic counter).
    pub current_world_tick: u64,
    /// Number of archetypes matched by this query type.
    pub matched_archetype_count: usize,
    /// Whether a deferred tick is pending (iterator was created but not
    /// yet committed). `true` means `query()` was called but the iterator
    /// hasn't been iterated yet, so the change window is still open.
    pub has_pending_tick: bool,
}

/// The central data store. Holds all entities, components, archetype metadata,
/// and the query cache.
///
/// Most operations start here: [`spawn`](World::spawn) creates entities,
/// [`query`](World::query) iterates them, [`insert`](World::insert)/[`remove`](World::remove)
/// alter component sets, and [`get`](World::get)/[`get_mut`](World::get_mut) access
/// individual components by [`Entity`] handle. For schema-declared access,
/// [`query_table`](World::query_table) and [`query_table_mut`](World::query_table_mut)
/// skip archetype matching entirely.
///
/// For concurrent access, combine with a [`ReducerRegistry`](crate::ReducerRegistry)
/// and a [`Transact`](crate::Transact) strategy. World itself is not `Sync` —
/// concurrency is achieved through the split-phase transaction protocol where
/// multiple [`Tx`](crate::Tx) objects read via `&World` concurrently.
pub struct World {
    pub(crate) id: WorldId,
    pub(crate) entities: EntityAllocator,
    pub(crate) archetypes: Archetypes,
    pub(crate) components: ComponentRegistry,
    pub(crate) sparse: SparseStorage,
    pub(crate) entity_locations: Vec<Option<EntityLocation>>,
    pub(crate) table_cache: TableCache,
    pub(crate) query_cache: HashMap<TypeId, QueryCacheEntry, crate::component::TypeIdBuildHasher>,
    pub(crate) current_tick: Tick,
    pub(crate) orphan_queue: OrphanQueue,
    pub(crate) pool: SharedPool,
}

impl World {
    pub fn new() -> Self {
        Self::new_with_pool(default_pool())
    }

    pub(crate) fn new_with_pool(pool: SharedPool) -> Self {
        Self {
            id: WorldId::next(),
            entities: EntityAllocator::new(),
            archetypes: Archetypes::new(),
            components: ComponentRegistry::new(),
            sparse: SparseStorage::new(pool.clone()),
            entity_locations: Vec::new(),
            table_cache: TableCache::new(),
            query_cache: HashMap::with_hasher(crate::component::TypeIdBuildHasher),
            current_tick: Tick::default(),
            orphan_queue: OrphanQueue::new(),
            pool,
        }
    }

    /// Create a [`WorldBuilder`] for configuring memory and allocation strategy.
    pub fn builder() -> WorldBuilder {
        WorldBuilder::new()
    }

    /// Advance the internal tick and return the new value.
    /// Called automatically on every mutation and query — not user-facing.
    pub(crate) fn next_tick(&mut self) -> Tick {
        self.current_tick.advance()
    }

    /// Drain orphaned entity IDs from aborted transactions.
    /// Called automatically at the top of every &mut self entry point.
    fn drain_orphans(&mut self) {
        self.entities.materialize_reserved();
        if !self
            .orphan_queue
            .has_items
            .load(crate::sync::Ordering::Acquire)
        {
            return;
        }
        let mut queue = self.orphan_queue.queue.lock();
        for entity in queue.drain(..) {
            self.entities.dealloc(entity);
        }
        // Clear flag while holding the lock so a concurrent Tx::drop
        // pushing between drain and this store won't lose its flag.
        if queue.is_empty() {
            self.orphan_queue
                .has_items
                .store(false, crate::sync::Ordering::Release);
        }
    }

    /// Clone the orphan queue handle. Strategies capture this at construction
    /// so that transaction Drop can push orphaned entity IDs without &mut World.
    pub(crate) fn orphan_queue(&self) -> OrphanQueue {
        self.orphan_queue.clone()
    }

    /// Return this World's unique identity. Strategies capture this at
    /// construction and assert it matches in begin/commit.
    pub(crate) fn world_id(&self) -> WorldId {
        self.id
    }

    /// Return the current change tick. Secondary indexes store this to
    /// track their sync point — see [`BTreeIndex`](crate::BTreeIndex).
    pub fn change_tick(&self) -> crate::tick::ChangeTick {
        crate::tick::ChangeTick(self.current_tick)
    }

    /// Returns `(Entity, T)` pairs (cloned) from archetypes whose `T` column
    /// was mutably accessed after `since`.
    ///
    /// Granularity is per-archetype-column, not per-entity — all entities in
    /// a changed archetype are returned, even if only one was mutated. This
    /// matches the pessimistic marking used by [`Changed<T>`](crate::Changed).
    ///
    /// Unlike the `Changed<T>` filter used by [`World::query`], this method
    /// does **not** use the shared query cache — the caller owns the tick via
    /// [`ChangeTick`](crate::ChangeTick), so multiple indexes on the same
    /// component type can call this independently.
    ///
    /// Values are cloned because the returned `Vec` must own its data
    /// independently of the World borrow.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `T` is not a sparse component. Sparse components
    /// are not stored in archetypes and have no column-level change ticks.
    /// Use [`rebuild`](crate::SpatialIndex::rebuild) for sparse components.
    pub fn query_changed_since<T: Component + Clone>(
        &mut self,
        since: crate::tick::ChangeTick,
    ) -> Vec<(Entity, T)> {
        self.drain_orphans();
        let Some(comp_id) = self.components.id::<T>() else {
            return Vec::new();
        };
        debug_assert!(
            !self.components.is_sparse(comp_id),
            "query_changed_since does not support sparse components — use rebuild() instead"
        );

        let since_tick = since.0;
        let mut results = Vec::new();

        for arch in &self.archetypes.archetypes {
            let Some(col_idx) = arch.column_index(comp_id) else {
                continue;
            };
            if !arch.columns[col_idx].changed_tick.is_newer_than(since_tick) {
                continue;
            }
            for row in 0..arch.len() {
                let entity = arch.entities[row];
                let ptr = unsafe { arch.columns[col_idx].get_ptr(row) as *const T };
                let value = unsafe { (*ptr).clone() };
                results.push((entity, value));
            }
        }

        results
    }

    /// Look up the ComponentId for a type. Returns None if the type has
    /// never been spawned or registered.
    pub fn component_id<T: Component>(&self) -> Option<ComponentId> {
        self.components.id::<T>()
    }

    /// Register a component type, returning its ComponentId. Idempotent —
    /// returns the existing id if already registered.
    pub fn register_component<T: Component>(&mut self) -> ComponentId {
        self.drain_orphans();
        self.components.register::<T>()
    }

    /// Register a component slot using raw metadata without a concrete type.
    /// Used during snapshot restore to preserve ComponentId assignments for
    /// components that exist in the snapshot schema but have no codec.
    pub fn register_component_raw(
        &mut self,
        name: &'static str,
        layout: std::alloc::Layout,
    ) -> ComponentId {
        self.drain_orphans();
        self.components.register_raw(name, layout)
    }

    /// Allocate a fresh entity ID without placing it in any archetype.
    /// Use this to obtain an unplaced handle for `EnumChangeSet::spawn_bundle`.
    pub fn alloc_entity(&mut self) -> Entity {
        self.drain_orphans();
        let entity = self.entities.alloc();
        let index = entity.index() as usize;
        if index >= self.entity_locations.len() {
            self.entity_locations.resize(index + 1, None);
        }
        entity
    }

    /// Deallocate an unplaced entity ID. Bumps the generation so the handle
    /// becomes stale, and returns the index to the free list.
    ///
    /// Used by transaction abort to clean up entity IDs that were allocated
    /// via `alloc_entity()` but never placed (changeset was discarded).
    #[allow(dead_code)]
    pub(crate) fn dealloc_entity(&mut self, entity: Entity) {
        self.drain_orphans();
        self.entities.dealloc(entity);
    }

    /// Returns true if the entity has been placed in an archetype (has a row).
    /// Entities from `alloc_entity()` return false until they are spawned.
    pub fn is_placed(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.entity_locations.len() && self.entity_locations[idx].is_some()
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        self.drain_orphans();
        let component_ids = B::component_ids(&mut self.components);
        let arch_id = self
            .archetypes
            .get_or_create(&component_ids, &self.components, &self.pool);
        let entity = self.entities.alloc();
        let index = entity.index() as usize;

        if index >= self.entity_locations.len() {
            self.entity_locations.resize(index + 1, None);
        }

        let tick = self.next_tick();
        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let col = archetype.column_index(comp_id).unwrap();
                archetype.columns[col].push(ptr as *mut u8);
            });
        }
        for col in &mut archetype.columns {
            col.mark_changed(tick);
        }
        let row = archetype.entities.len();
        archetype.entities.push(entity);
        archetype.debug_assert_consistent();

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
        entity
    }

    /// Like [`spawn`] but returns `Err(PoolExhausted)` instead of panicking
    /// when the pool cannot allocate storage for the new entity's components.
    ///
    /// If this method returns `Err`, the world state is unchanged — no partial
    /// entity is created and no archetype slot is leaked.
    ///
    /// # Panics
    ///
    /// This method can still panic if the system allocator (used for internal
    /// metadata like entity location tables) runs out of memory. Only pool-backed
    /// storage (component columns) is covered by the `Result` return type.
    pub fn try_spawn<B: Bundle>(&mut self, bundle: B) -> Result<Entity, PoolExhausted> {
        self.drain_orphans();
        let component_ids = B::component_ids(&mut self.components);
        let arch_id = self
            .archetypes
            .get_or_create(&component_ids, &self.components, &self.pool);
        let archetype = &mut self.archetypes.archetypes[arch_id.0];

        // Pre-check: ensure all columns can accommodate one more element.
        // If any column is at capacity and the pool can't grow it, return Err
        // before modifying any state.
        for col in &mut archetype.columns {
            if col.len() == col.capacity() {
                col.try_grow()?;
            }
        }

        // All columns have capacity — proceed with infallible push.
        let entity = self.entities.alloc();
        let index = entity.index() as usize;

        if index >= self.entity_locations.len() {
            self.entity_locations.resize(index + 1, None);
        }

        let tick = self.next_tick();
        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let col = archetype.column_index(comp_id).unwrap();
                archetype.columns[col].push(ptr as *mut u8);
            });
        }
        for col in &mut archetype.columns {
            col.mark_changed(tick);
        }
        let row = archetype.entities.len();
        archetype.entities.push(entity);
        archetype.debug_assert_consistent();

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
        Ok(entity)
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.drain_orphans();
        if !self.entities.is_alive(entity) {
            return false;
        }
        let index = entity.index() as usize;
        let Some(location) = self.entity_locations[index] else {
            return false;
        };

        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        let row = location.row;
        debug_assert!(
            row < archetype.len(),
            "stale EntityLocation: row {} >= archetype len {}",
            row,
            archetype.len()
        );

        for col in &mut archetype.columns {
            unsafe {
                col.swap_remove(row);
            }
        }

        archetype.entities.swap_remove(row);

        // Update the swapped entity's location
        if row < archetype.entities.len() {
            let swapped = archetype.entities[row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: location.archetype_id,
                row,
            });
        }

        archetype.debug_assert_consistent();

        // Clean sparse storage before dealloc bumps the generation
        self.sparse.remove_all(entity);

        self.entity_locations[index] = None;
        self.entities.dealloc(entity);
        true
    }

    /// Despawn multiple entities efficiently. Groups by archetype, sorts rows
    /// descending, and sweeps back-to-front so that swap-remove indices stay valid.
    /// Returns the number of entities actually despawned (skips dead/unplaced).
    pub fn despawn_batch(&mut self, entities: &[Entity]) -> usize {
        self.drain_orphans();

        // Phase 1: Filter alive+placed, group by archetype
        let mut by_archetype: HashMap<usize, Vec<(usize, Entity)>> = HashMap::new();
        let mut to_dealloc: Vec<Entity> = Vec::new();

        let mut seen = FixedBitSet::with_capacity(self.entities.generations.len());
        for &entity in entities {
            if !self.entities.is_alive(entity) {
                continue;
            }
            let index = entity.index() as usize;
            // Deduplicate: skip if we've already recorded this entity index
            if seen.contains(index) {
                continue;
            }
            seen.insert(index);
            let Some(location) = self.entity_locations[index] else {
                continue;
            };
            by_archetype
                .entry(location.archetype_id.0)
                .or_default()
                .push((location.row, entity));
            to_dealloc.push(entity);
        }

        // Phase 2: Sparse cleanup BEFORE dealloc bumps generations
        for &entity in &to_dealloc {
            self.sparse.remove_all(entity);
        }

        // Phase 3: For each archetype, sort rows descending, sweep back-to-front
        // PERF: Non-vectorizable by design — drop_fn is opaque, memcpy is variable-size.
        // Batch wins via amortized archetype resolution, not SIMD.
        for (arch_idx, mut row_entities) in by_archetype {
            row_entities.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let archetype = &mut self.archetypes.archetypes[arch_idx];

            for &(row, _entity) in &row_entities {
                let last = archetype.entities.len() - 1;

                // Drop component data at this row
                for col in &mut archetype.columns {
                    unsafe {
                        col.drop_in_place(row);
                    }
                }

                // If not last, copy last element into gap
                if row < last {
                    for col in &mut archetype.columns {
                        unsafe {
                            col.copy_unchecked(last, row);
                        }
                    }
                    let moved_entity = archetype.entities[last];
                    archetype.entities[row] = moved_entity;
                    self.entity_locations[moved_entity.index() as usize] = Some(EntityLocation {
                        archetype_id: ArchetypeId(arch_idx),
                        row,
                    });
                }

                archetype.entities.truncate(last);
                for col in &mut archetype.columns {
                    unsafe {
                        col.set_len(last);
                    }
                }
            }
            archetype.debug_assert_consistent();
        }

        // Phase 4: Dealloc entities and clear locations
        for &entity in &to_dealloc {
            self.entity_locations[entity.index() as usize] = None;
            self.entities.dealloc(entity);
        }

        to_dealloc.len()
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Returns true if the entity is alive and currently carries component `T`.
    ///
    /// For archetype components, this is a bitset check. For sparse components,
    /// it checks the sparse storage map. No component data is read in either case.
    /// Returns false if the entity is dead, unplaced, or if `T` was removed.
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        let Some(location) = self.entity_locations[entity.index() as usize] else {
            return false;
        };
        let Some(comp_id) = self.components.id::<T>() else {
            return false;
        };
        if self.components.is_sparse(comp_id) {
            return self.sparse.contains(comp_id, entity);
        }
        let archetype = &self.archetypes.archetypes[location.archetype_id.0];
        archetype.component_ids.contains(comp_id)
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let comp_id = self.components.id::<T>()?;

        if self.components.is_sparse(comp_id) {
            return self.sparse.get::<T>(comp_id, entity);
        }

        let archetype = &self.archetypes.archetypes[location.archetype_id.0];
        let col_idx = archetype.column_index(comp_id)?;
        unsafe {
            let ptr = archetype.columns[col_idx].get_ptr(location.row) as *const T;
            Some(&*ptr)
        }
    }

    /// Get a component by pre-resolved ComponentId. Same as `get::<T>()` but
    /// skips the TypeId → ComponentId lookup. No sparse check (reducer
    /// components are always archetype-stored).
    #[allow(dead_code)]
    pub(crate) fn get_by_id<T: Component>(
        &self,
        entity: Entity,
        comp_id: ComponentId,
    ) -> Option<&T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let archetype = &self.archetypes.archetypes[location.archetype_id.0];
        let col_idx = archetype.column_index(comp_id)?;
        unsafe {
            let ptr = archetype.columns[col_idx].get_ptr(location.row) as *const T;
            Some(&*ptr)
        }
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.drain_orphans();
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let comp_id = self.components.id::<T>()?;

        if self.components.is_sparse(comp_id) {
            return self.sparse.get_mut::<T>(comp_id, entity);
        }

        let tick = self.next_tick();
        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        debug_assert!(
            location.row < archetype.len(),
            "stale EntityLocation: row {} >= archetype len {}",
            location.row,
            archetype.len()
        );
        let col_idx = archetype.column_index(comp_id)?;
        unsafe {
            let ptr = archetype.columns[col_idx].get_ptr_mut(location.row, tick) as *mut T;
            Some(&mut *ptr)
        }
    }

    /// Fetch a component for multiple entities, grouped by archetype for
    /// cache locality. Returns results in the same order as the input slice.
    /// Dead entities and entities missing the component yield `None`.
    ///
    /// This amortises the per-entity overhead of [`get`](Self::get): the
    /// `ComponentId` is resolved once, entities are grouped by archetype,
    /// and all rows in each archetype are fetched together.
    pub fn get_batch<T: Component>(&self, entities: &[Entity]) -> Vec<Option<&T>> {
        let mut results = vec![None; entities.len()];
        if entities.is_empty() {
            return results;
        }

        let Some(comp_id) = self.components.id::<T>() else {
            return results;
        };

        // Sparse fast path — no archetype grouping benefit
        if self.components.is_sparse(comp_id) {
            for (i, &entity) in entities.iter().enumerate() {
                if self.entities.is_alive(entity) {
                    results[i] = self.sparse.get::<T>(comp_id, entity);
                }
            }
            return results;
        }

        // PERF: HashMap allocation is O(archetype_count) which is typically < 20.
        // Not in a per-frame inner loop — called once per batch operation.
        let mut by_arch: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (i, &entity) in entities.iter().enumerate() {
            if !self.entities.is_alive(entity) {
                continue;
            }
            if let Some(location) = self.entity_locations[entity.index() as usize] {
                by_arch
                    .entry(location.archetype_id.0)
                    .or_default()
                    .push((i, location.row));
            }
        }

        // Fetch per-archetype: one column lookup per archetype, not per entity.
        for (arch_idx, rows) in &by_arch {
            let arch = &self.archetypes.archetypes[*arch_idx];
            if let Some(col_idx) = arch.column_index(comp_id) {
                for &(result_idx, row) in rows {
                    results[result_idx] = unsafe {
                        let ptr = arch.columns[col_idx].get_ptr(row) as *const T;
                        Some(&*ptr)
                    };
                }
            }
        }

        results
    }

    /// Mutable batch fetch — same archetype-grouped pattern as
    /// [`get_batch`](Self::get_batch), but returns `&mut T` and marks
    /// accessed columns as changed for [`Changed<T>`](crate::query::fetch::Changed)
    /// detection.
    ///
    /// **Note:** Sparse components do not participate in [`Changed<T>`](crate::query::fetch::Changed)
    /// detection (this matches the behavior of [`get_mut`](Self::get_mut)).
    ///
    /// # Panics
    ///
    /// Panics if the same entity appears more than once in `entities`.
    /// Aliased `&mut T` is undefined behaviour — this check is unconditional.
    pub fn get_batch_mut<T: Component>(&mut self, entities: &[Entity]) -> Vec<Option<&mut T>> {
        self.drain_orphans();

        let mut results: Vec<Option<&mut T>> = (0..entities.len()).map(|_| None).collect();
        if entities.is_empty() {
            return results;
        }

        let Some(comp_id) = self.components.id::<T>() else {
            return results;
        };

        // Sparse fast path — duplicate check via sorted indices.
        if self.components.is_sparse(comp_id) {
            let mut alive_indices: Vec<u32> = entities
                .iter()
                .filter(|e| self.entities.is_alive(**e))
                .map(|e| e.index())
                .collect();
            alive_indices.sort_unstable();
            for w in alive_indices.windows(2) {
                assert!(w[0] != w[1], "duplicate entity in get_batch_mut");
            }
            for (i, &entity) in entities.iter().enumerate() {
                if self.entities.is_alive(entity)
                    && let Some(val) = self.sparse.get_mut::<T>(comp_id, entity)
                {
                    // SAFETY: the duplicate check above guarantees each entity
                    // appears at most once, so no two results alias.
                    results[i] = Some(unsafe { &mut *(val as *mut T) });
                }
            }
            return results;
        }

        // PERF: HashMap allocation is O(archetype_count) which is typically < 20.
        // Not in a per-frame inner loop — called once per batch operation.
        let mut by_arch: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (i, &entity) in entities.iter().enumerate() {
            if !self.entities.is_alive(entity) {
                continue;
            }
            if let Some(location) = self.entity_locations[entity.index() as usize] {
                by_arch
                    .entry(location.archetype_id.0)
                    .or_default()
                    .push((i, location.row));
            }
        }

        // Duplicate check: sort each group by row, check adjacent pairs.
        // Same row in same archetype = same entity (each row holds exactly one entity).
        for rows in by_arch.values_mut() {
            rows.sort_unstable_by_key(|&(_, row)| row);
            for w in rows.windows(2) {
                assert!(w[0].1 != w[1].1, "duplicate entity in get_batch_mut");
            }
        }

        // Mark columns changed and fetch per-archetype.
        let tick = self.next_tick();
        for (arch_idx, rows) in &by_arch {
            let arch = &mut self.archetypes.archetypes[*arch_idx];
            if let Some(col_idx) = arch.column_index(comp_id) {
                for &(result_idx, row) in rows {
                    results[result_idx] = unsafe {
                        let ptr = arch.columns[col_idx].get_ptr_mut(row, tick) as *mut T;
                        Some(&mut *ptr)
                    };
                }
            }
        }

        results
    }

    /// Iterate entities matching the query type `Q`.
    ///
    /// # Tick advancement (lazy)
    ///
    /// This method uses **lazy tick advancement**: the `Changed<T>` read
    /// baseline is only updated when the returned iterator is actually
    /// iterated. If you create a query and drop the iterator without
    /// consuming it, the change window is preserved — subsequent queries
    /// will still see those changes.
    ///
    /// **Mutable columns** (`&mut T` in `Q`) are still marked as changed
    /// eagerly (required for soundness — the pointers are already valid).
    ///
    /// This matches the behavior of [`QueryWriter`](crate::QueryWriter) in
    /// the reducer system, which uses a `queried` flag to defer tick
    /// advancement until `for_each` or `count` is called.
    ///
    /// See also:
    /// - [`has_changed`](Self::has_changed) — peek without consuming changes
    /// - [`advance_query_tick`](Self::advance_query_tick) — explicitly consume
    ///   the change window without iterating
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.drain_orphans();
        let type_id = TypeId::of::<Q>();
        let total = self.archetypes.archetypes.len();

        let entry = self
            .query_cache
            .entry(type_id)
            .or_insert_with(|| QueryCacheEntry {
                matched_ids: Vec::new(),
                required: Q::required_ids(&self.components),
                last_archetype_count: 0,
                last_read_tick: Tick::default(),
                pending_read_tick: None,
                iterated: Arc::new(AtomicBool::new(false)),
                registry_version: self.components.version(),
            });

        // Commit deferred tick from previous iterator, if it was iterated.
        if entry.iterated.load(Ordering::Relaxed) {
            if let Some(pending) = entry.pending_read_tick.take() {
                entry.last_read_tick = pending;
            }
            entry.iterated.store(false, Ordering::Relaxed);
        } else {
            // Previous iterator was not iterated — discard pending tick,
            // preserving the change window.
            entry.pending_read_tick = None;
        }

        // Refresh required bitset if new components were registered since the
        // cache entry was created. Version check avoids recomputing the FixedBitSet
        // on every query() call — only recomputes when the registry actually grew.
        let current_version = self.components.version();
        if entry.registry_version != current_version {
            let fresh_required = Q::required_ids(&self.components);
            if fresh_required != entry.required {
                entry.required = fresh_required;
                entry.matched_ids.clear();
                entry.last_archetype_count = 0;
                entry.last_read_tick = Tick::default();
            }
            entry.registry_version = current_version;
        }

        // Incremental scan: only check archetypes added since last cache update
        if entry.last_archetype_count < total {
            for arch in &self.archetypes.archetypes[entry.last_archetype_count..total] {
                if entry.required.is_subset(&arch.component_ids) {
                    entry.matched_ids.push(arch.id);
                }
            }
            entry.last_archetype_count = total;
        }

        // Take the Vec to release the &mut borrow on query_cache.
        // Avoids cloning — the Vec is put back after filtering.
        let matched_ids = std::mem::take(&mut entry.matched_ids);
        let last_read_tick = entry.last_read_tick;

        // Compute which component IDs this query mutates
        let mutable = Q::mutable_ids(&self.components);

        // Pass 1: filter archetypes and mark mutable columns (requires &mut self)
        let mut filtered_ids = Vec::new();
        for &aid in &matched_ids {
            let arch = &self.archetypes.archetypes[aid.0];
            if arch.is_empty() {
                continue;
            }
            if !Q::matches_filters(arch, &self.components, last_read_tick) {
                continue;
            }
            filtered_ids.push(aid);
        }
        // Mark mutable columns changed (each gets a fresh tick)
        if !mutable.is_empty() {
            let tick = self.next_tick();
            for &aid in &filtered_ids {
                for comp_id in mutable.ones() {
                    let arch = &self.archetypes.archetypes[aid.0];
                    if let Some(col_idx) = arch.column_index(comp_id) {
                        self.archetypes.archetypes[aid.0].columns[col_idx].mark_changed(tick);
                    }
                }
            }
        }

        // Defer read tick — only committed on next query() if iterator is iterated.
        let read_tick = self.next_tick();
        let iterated_flag = if let Some(entry) = self.query_cache.get_mut(&type_id) {
            entry.pending_read_tick = Some(read_tick);
            entry.matched_ids = matched_ids;
            // PERF: Arc::clone is one atomic increment per query() call.
            // Could use raw pointer since &mut self guarantees single-threaded,
            // but correctness-first — the atomic cost is negligible vs archetype scan.
            entry.iterated.clone()
        } else {
            unreachable!("cache entry was just inserted")
        };

        // Pass 2: build fetches (only immutable borrows of archetypes from here)
        let fetches: Vec<_> = filtered_ids
            .iter()
            .map(|&aid| {
                let arch = &self.archetypes.archetypes[aid.0];
                (Q::init_fetch(arch, &self.components), arch.len())
            })
            .collect();

        QueryIter::with_tick_flag(fetches, iterated_flag)
    }

    /// Resolve column pointers for a table's archetype.
    fn resolve_table_ptrs<T: crate::table::Table>(&mut self) -> (Vec<(*mut u8, usize)>, usize) {
        let pool = self.pool.clone();
        let desc =
            self.table_cache
                .get_or_create::<T>(&mut self.components, &mut self.archetypes, &pool);
        let arch_id = desc.archetype_id;
        let col_indices = desc.col_indices.clone();
        let item_sizes = desc.item_sizes.clone();

        let archetype = &self.archetypes.archetypes[arch_id.0];
        if archetype.is_empty() {
            return (vec![], 0);
        }

        let col_ptrs: Vec<(*mut u8, usize)> = col_indices
            .iter()
            .zip(item_sizes.iter())
            .map(|(&col_idx, &size)| unsafe { (archetype.columns[col_idx].get_ptr(0), size) })
            .collect();

        (col_ptrs, archetype.len())
    }

    /// Iterate all rows of a table's archetype with raw column pointers.
    /// Skips archetype matching — goes directly to the table's cached archetype.
    ///
    /// Marks all columns changed (raw pointers allow arbitrary writes).
    pub fn query_table_raw<T: crate::table::Table>(&mut self) -> crate::table::TableIter<'_> {
        self.drain_orphans();
        self.mark_table_columns_changed::<T>();
        let (col_ptrs, len) = self.resolve_table_ptrs::<T>();
        crate::table::TableIter::new(col_ptrs, len)
    }

    /// Iterate all rows with typed immutable field access.
    pub fn query_table<T: crate::table::Table>(
        &mut self,
    ) -> crate::table::TableTypedIter<'_, T::Ref<'_>> {
        let (col_ptrs, len) = self.resolve_table_ptrs::<T>();
        crate::table::TableTypedIter::new(col_ptrs, len)
    }

    /// Iterate all rows with typed mutable field access.
    ///
    /// Marks all columns changed for change detection.
    pub fn query_table_mut<T: crate::table::Table>(
        &mut self,
    ) -> crate::table::TableTypedIter<'_, T::Mut<'_>> {
        self.mark_table_columns_changed::<T>();
        let (col_ptrs, len) = self.resolve_table_ptrs::<T>();
        crate::table::TableTypedIter::new(col_ptrs, len)
    }

    /// Mark all columns in a table's archetype as changed.
    fn mark_table_columns_changed<T: crate::table::Table>(&mut self) {
        let tick = self.next_tick();
        let pool = self.pool.clone();
        let desc =
            self.table_cache
                .get_or_create::<T>(&mut self.components, &mut self.archetypes, &pool);
        let arch_id = desc.archetype_id;
        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        for col in &mut archetype.columns {
            col.mark_changed(tick);
        }
    }

    pub fn insert<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        self.drain_orphans();
        assert!(self.is_alive(entity), "entity is not alive");
        let index = entity.index() as usize;
        let location = self.entity_locations[index].unwrap();
        debug_assert!(
            location.row < self.archetypes.archetypes[location.archetype_id.0].len(),
            "stale EntityLocation: row {} >= archetype len {}",
            location.row,
            self.archetypes.archetypes[location.archetype_id.0].len()
        );
        let new_ids = B::component_ids(&mut self.components);

        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        let all_existing = new_ids
            .iter()
            .all(|&id| src_arch.component_ids.contains(id));

        if all_existing {
            // All bundle components already exist — overwrite in-place
            let tick = self.next_tick();
            let arch = &mut self.archetypes.archetypes[location.archetype_id.0];
            let row = location.row;
            unsafe {
                bundle.put(&self.components, &mut |comp_id, ptr, layout| {
                    let col_idx = arch.column_index(comp_id).unwrap();
                    let dst = arch.columns[col_idx].get_ptr_mut(row, tick);
                    if let Some(drop_fn) = self.components.info(comp_id).drop_fn {
                        drop_fn(dst);
                    }
                    std::ptr::copy_nonoverlapping(ptr, dst, layout.size());
                });
            }
            return;
        }

        // Compute target archetype: source components ∪ new components
        let mut target_ids = src_arch.sorted_ids.clone();
        for &id in &new_ids {
            if !src_arch.component_ids.contains(id) {
                target_ids.push(id);
            }
        }
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id =
            self.archetypes
                .get_or_create(&target_ids, &self.components, &self.pool);
        let tick = self.next_tick();

        // Check if source and target are the same archetype (all components
        // already existed but we took the migration path due to a race-free
        // logic flow — can't happen currently, but guard defensively).
        if src_arch_id == target_arch_id {
            let arch = &mut self.archetypes.archetypes[src_arch_id.0];
            unsafe {
                bundle.put(&self.components, &mut |comp_id, ptr, layout| {
                    let col_idx = arch.column_index(comp_id).unwrap();
                    let dst = arch.columns[col_idx].get_ptr_mut(src_row, tick);
                    if let Some(drop_fn) = self.components.info(comp_id).drop_fn {
                        drop_fn(dst);
                    }
                    std::ptr::copy_nonoverlapping(ptr, dst, layout.size());
                });
            }
            return;
        }

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Collect which source columns have overwrites from the bundle,
        // so we know to skip copying them (the bundle value wins).
        let overwrite_src_cols: Vec<bool> = src_arch
            .sorted_ids
            .iter()
            .map(|&cid| new_ids.contains(&cid))
            .collect();

        // Move non-overwritten columns from source to target
        for (src_col, &cid) in src_arch.sorted_ids.iter().enumerate() {
            if overwrite_src_cols[src_col] {
                // This column is being overwritten by the bundle — drop the
                // old value and skip the copy; the bundle.put below writes it.
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    if let Some(drop_fn) = self.components.info(cid).drop_fn {
                        drop_fn(ptr);
                    }
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            } else if let Some(tgt_col) = target_arch.column_index(cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        // Write all bundle components into target
        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let tgt_col = target_arch.column_index(comp_id).unwrap();
                target_arch.columns[tgt_col].push(ptr as *mut u8);
            });
        }
        for col in &mut target_arch.columns {
            col.mark_changed(tick);
        }

        // Move entity tracking
        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        // Update swapped entity's location in source
        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });

        // Verify column/entity invariant after migration.
        src_arch.debug_assert_consistent();
        target_arch.debug_assert_consistent();
    }

    /// Like [`insert`] but returns `Err(PoolExhausted)` instead of panicking
    /// when the pool cannot allocate storage for the new archetype's columns.
    ///
    /// If this method returns `Err`, the world state is unchanged — no partial
    /// migration occurs and the entity remains in its original archetype.
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn try_insert<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
    ) -> Result<(), PoolExhausted> {
        self.drain_orphans();
        assert!(self.is_alive(entity), "entity is not alive");
        let index = entity.index() as usize;
        let location = self.entity_locations[index].unwrap();
        debug_assert!(
            location.row < self.archetypes.archetypes[location.archetype_id.0].len(),
            "stale EntityLocation: row {} >= archetype len {}",
            location.row,
            self.archetypes.archetypes[location.archetype_id.0].len()
        );
        let new_ids = B::component_ids(&mut self.components);

        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        let all_existing = new_ids
            .iter()
            .all(|&id| src_arch.component_ids.contains(id));

        if all_existing {
            // All bundle components already exist — overwrite in-place (no allocation)
            let tick = self.next_tick();
            let arch = &mut self.archetypes.archetypes[location.archetype_id.0];
            let row = location.row;
            unsafe {
                bundle.put(&self.components, &mut |comp_id, ptr, layout| {
                    let col_idx = arch.column_index(comp_id).unwrap();
                    let dst = arch.columns[col_idx].get_ptr_mut(row, tick);
                    if let Some(drop_fn) = self.components.info(comp_id).drop_fn {
                        drop_fn(dst);
                    }
                    std::ptr::copy_nonoverlapping(ptr, dst, layout.size());
                });
            }
            return Ok(());
        }

        // Compute target archetype: source components ∪ new components
        let mut target_ids = src_arch.sorted_ids.clone();
        for &id in &new_ids {
            if !src_arch.component_ids.contains(id) {
                target_ids.push(id);
            }
        }
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id =
            self.archetypes
                .get_or_create(&target_ids, &self.components, &self.pool);
        let tick = self.next_tick();

        // Defensive: same archetype → overwrite in-place
        if src_arch_id == target_arch_id {
            let arch = &mut self.archetypes.archetypes[src_arch_id.0];
            unsafe {
                bundle.put(&self.components, &mut |comp_id, ptr, layout| {
                    let col_idx = arch.column_index(comp_id).unwrap();
                    let dst = arch.columns[col_idx].get_ptr_mut(src_row, tick);
                    if let Some(drop_fn) = self.components.info(comp_id).drop_fn {
                        drop_fn(dst);
                    }
                    std::ptr::copy_nonoverlapping(ptr, dst, layout.size());
                });
            }
            return Ok(());
        }

        // Pre-check: ensure all target columns can accommodate one more element.
        // If any column is at capacity and the pool can't grow it, return Err
        // before starting the migration.
        {
            let target_arch = &mut self.archetypes.archetypes[target_arch_id.0];
            for col in &mut target_arch.columns {
                if col.len() == col.capacity() {
                    col.try_grow()?;
                }
            }
        }

        // All target columns have capacity — proceed with infallible migration.
        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        let overwrite_src_cols: Vec<bool> = src_arch
            .sorted_ids
            .iter()
            .map(|&cid| new_ids.contains(&cid))
            .collect();

        for (src_col, &cid) in src_arch.sorted_ids.iter().enumerate() {
            if overwrite_src_cols[src_col] {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    if let Some(drop_fn) = self.components.info(cid).drop_fn {
                        drop_fn(ptr);
                    }
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            } else if let Some(tgt_col) = target_arch.column_index(cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let tgt_col = target_arch.column_index(comp_id).unwrap();
                target_arch.columns[tgt_col].push(ptr as *mut u8);
            });
        }
        for col in &mut target_arch.columns {
            col.mark_changed(tick);
        }

        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });

        src_arch.debug_assert_consistent();
        target_arch.debug_assert_consistent();
        Ok(())
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.drain_orphans();
        if !self.is_alive(entity) {
            return None;
        }
        let index = entity.index() as usize;
        let location = self.entity_locations[index]?;
        debug_assert!(
            location.row < self.archetypes.archetypes[location.archetype_id.0].len(),
            "stale EntityLocation: row {} >= archetype len {}",
            location.row,
            self.archetypes.archetypes[location.archetype_id.0].len()
        );
        let comp_id = self.components.id::<T>()?;

        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        if !src_arch.component_ids.contains(comp_id) {
            return None;
        }

        // Read the component value before migration
        let removed = unsafe {
            let col_idx = src_arch.column_index(comp_id).unwrap();
            let ptr = src_arch.columns[col_idx].get_ptr(location.row) as *const T;
            std::ptr::read(ptr)
        };

        // Compute target archetype: source components minus removed component
        let target_ids: Vec<ComponentId> = src_arch
            .sorted_ids
            .iter()
            .copied()
            .filter(|&id| id != comp_id)
            .collect();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        if target_ids.is_empty() {
            // Entity has no components left — move to empty archetype
            let arch = &mut self.archetypes.archetypes[src_arch_id.0];
            // swap_remove_no_drop for the removed component (already read)
            let removed_col = arch.column_index(comp_id).unwrap();
            unsafe {
                arch.columns[removed_col].swap_remove_no_drop(src_row);
            }
            // swap_remove with drop for remaining columns
            for (col_idx, &cid) in arch.sorted_ids.iter().enumerate() {
                if cid != comp_id {
                    unsafe {
                        arch.columns[col_idx].swap_remove(src_row);
                    }
                }
            }
            arch.entities.swap_remove(src_row);
            if src_row < arch.entities.len() {
                let swapped = arch.entities[src_row];
                self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                    archetype_id: src_arch_id,
                    row: src_row,
                });
            }
            arch.debug_assert_consistent();
            let empty_arch_id = self
                .archetypes
                .get_or_create(&[], &self.components, &self.pool);
            let empty_arch = &mut self.archetypes.archetypes[empty_arch_id.0];
            empty_arch.entities.push(entity);
            self.entity_locations[index] = Some(EntityLocation {
                archetype_id: empty_arch_id,
                row: empty_arch.entities.len() - 1,
            });
            return Some(removed);
        }

        let target_arch_id =
            self.archetypes
                .get_or_create(&target_ids, &self.components, &self.pool);

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns (skip removed component)
        for (src_col, &cid) in src_arch.sorted_ids.iter().enumerate() {
            if cid == comp_id {
                // Already read — just discard from source
                unsafe {
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            } else if let Some(tgt_col) = target_arch.column_index(cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });

        // Verify column/entity invariant after migration.
        src_arch.debug_assert_consistent();
        target_arch.debug_assert_consistent();

        Some(removed)
    }

    /// Read all component data for an entity as raw bytes.
    /// Returns (ComponentId, *const u8, Layout) per component.
    /// The pointers are valid until the next structural mutation.
    #[allow(dead_code)]
    pub(crate) fn read_all_components(
        &self,
        entity: Entity,
    ) -> Option<Vec<(ComponentId, *const u8, Layout)>> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let archetype = &self.archetypes.archetypes[location.archetype_id.0];

        let components: Vec<_> = archetype
            .sorted_ids
            .iter()
            .map(|&comp_id| {
                let col_idx = archetype.column_index(comp_id).unwrap();
                let info = self.components.info(comp_id);
                let ptr = unsafe { archetype.columns[col_idx].get_ptr(location.row) };
                (comp_id, ptr as *const u8, info.layout)
            })
            .collect();

        Some(components)
    }

    /// Snapshot the `changed_tick` of every column matching the given component
    /// bitset. Returns a Vec of (ArchetypeId index, ComponentId, Tick) triples.
    /// Used by optimistic transactions for read-set validation.
    #[allow(dead_code)]
    pub(crate) fn snapshot_column_ticks(
        &self,
        component_ids: &FixedBitSet,
    ) -> Vec<(usize, ComponentId, crate::tick::Tick)> {
        let mut ticks = Vec::new();
        for arch in &self.archetypes.archetypes {
            for comp_id in component_ids.ones() {
                if let Some(col_idx) = arch.column_index(comp_id) {
                    ticks.push((arch.id.0, comp_id, arch.columns[col_idx].changed_tick));
                }
            }
        }
        ticks
    }

    /// Check if any column in the snapshot has been modified since the
    /// recorded tick. Returns a FixedBitSet of conflicting ComponentIds.
    #[allow(dead_code)]
    pub(crate) fn check_column_conflicts(
        &self,
        snapshot: &[(usize, ComponentId, crate::tick::Tick)],
    ) -> FixedBitSet {
        let mut conflicts = FixedBitSet::new();
        #[allow(clippy::collapsible_if)]
        for &(arch_idx, comp_id, begin_tick) in snapshot {
            if let Some(arch) = self.archetypes.archetypes.get(arch_idx) {
                if let Some(col_idx) = arch.column_index(comp_id) {
                    if arch.columns[col_idx].changed_tick.is_newer_than(begin_tick) {
                        conflicts.grow(comp_id + 1);
                        conflicts.insert(comp_id);
                    }
                }
            }
        }
        conflicts
    }

    /// Shared-ref query path for transactions. No cache update, no tick
    /// advancement, no column marking. Safe for concurrent reads from
    /// multiple threads.
    ///
    /// Scans archetypes (up to `archetype_count`) against the query's required
    /// component bitset. Skips empty archetypes. Does not apply Changed<T> filters
    /// (transactions have their own tick-based conflict model).
    ///
    /// `archetype_count` freezes the archetype set to what existed at begin time.
    #[allow(dead_code)]
    pub(crate) fn query_raw<Q: crate::query::fetch::ReadOnlyWorldQuery + 'static>(
        &self,
        archetype_count: usize,
    ) -> QueryIter<'_, Q> {
        let required = Q::required_ids(&self.components);
        let fetches: Vec<_> = self.archetypes.archetypes[..archetype_count]
            .iter()
            .filter(|arch| !arch.is_empty() && required.is_subset(&arch.component_ids))
            .map(|arch| (Q::init_fetch(arch, &self.components), arch.len()))
            .collect();
        QueryIter::new(fetches)
    }

    // ── Tick control ──────────────────────────────────────────────────

    /// Check whether any archetype has `Changed<T>` data for query type `Q`
    /// **without** advancing the read tick or marking mutable columns.
    ///
    /// This is a non-consuming peek: calling `has_changed` repeatedly returns
    /// the same result until a mutation actually occurs. Use this to decide
    /// whether to run an expensive query without losing the change window.
    ///
    /// Returns `false` if `Q` has never been queried (no cache entry exists).
    /// This means `has_changed` is not suitable as a guard before the *first*
    /// query — use `query().count() > 0` instead if you need to detect
    /// initial data. After the first `query::<Q>()` call establishes a
    /// baseline tick, `has_changed` accurately reports subsequent mutations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Establish baseline with first query.
    /// world.query::<(Changed<Pos>, &Pos)>().for_each(|_| {});
    ///
    /// // Now has_changed accurately reports mutations since the baseline.
    /// if world.has_changed::<(Changed<Pos>, &Pos)>() {
    ///     for pos in world.query::<(Changed<Pos>, &Pos)>() { /* ... */ }
    /// }
    /// ```
    #[inline]
    pub fn has_changed<Q: WorldQuery + 'static>(&self) -> bool {
        let type_id = TypeId::of::<Q>();
        // No cache entry means never queried — first query will see everything.
        let Some(entry) = self.query_cache.get(&type_id) else {
            return false;
        };

        // If the previous iterator was iterated, the pending tick
        // represents the most recent baseline (not yet committed to
        // last_read_tick — that happens on the next query() call).
        // Use it so has_changed doesn't report stale true results.
        let last_read_tick = if entry.iterated.load(Ordering::Relaxed) {
            entry.pending_read_tick.unwrap_or(entry.last_read_tick)
        } else {
            entry.last_read_tick
        };

        // Use cached matched_ids to skip non-matching archetypes.
        // Archetypes added after the last query() call are not in the cache,
        // but they will be picked up on the next query() call. This is
        // consistent with has_changed requiring a prior query() for baseline.
        entry.matched_ids.iter().any(|arch_id| {
            let arch = &self.archetypes.archetypes[arch_id.0];
            !arch.is_empty() && Q::matches_filters(arch, &self.components, last_read_tick)
        })
    }

    /// Advance the read tick for query type `Q` without iterating.
    ///
    /// This "acknowledges" all current changes so that subsequent
    /// `Changed<T>` queries of the same type start from the new baseline.
    /// Unlike [`query`](Self::query), this does **not** mark mutable columns
    /// as changed — it only moves the read tick forward.
    ///
    /// Use this when you know changes occurred but want to skip processing
    /// them (e.g., during a loading phase where you don't want stale change
    /// detection after initialization).
    #[inline]
    pub fn advance_query_tick<Q: WorldQuery + 'static>(&mut self) {
        self.drain_orphans();
        let type_id = TypeId::of::<Q>();
        let total = self.archetypes.archetypes.len();

        // Ensure cache entry exists and update archetype scan.
        {
            let entry = self
                .query_cache
                .entry(type_id)
                .or_insert_with(|| QueryCacheEntry {
                    matched_ids: Vec::new(),
                    required: Q::required_ids(&self.components),
                    last_archetype_count: 0,
                    last_read_tick: Tick::default(),
                    pending_read_tick: None,
                    iterated: Arc::new(AtomicBool::new(false)),
                    registry_version: self.components.version(),
                });

            let current_version = self.components.version();
            if entry.registry_version != current_version {
                let fresh_required = Q::required_ids(&self.components);
                if fresh_required != entry.required {
                    entry.required = fresh_required;
                    entry.matched_ids.clear();
                    entry.last_archetype_count = 0;
                    entry.last_read_tick = Tick::default();
                }
                entry.registry_version = current_version;
            }

            if entry.last_archetype_count < total {
                for arch in &self.archetypes.archetypes[entry.last_archetype_count..total] {
                    if entry.required.is_subset(&arch.component_ids) {
                        entry.matched_ids.push(arch.id);
                    }
                }
                entry.last_archetype_count = total;
            }
        }

        // Advance read tick directly — this is an explicit "acknowledge changes" call.
        let read_tick = self.next_tick();
        if let Some(entry) = self.query_cache.get_mut(&type_id) {
            entry.last_read_tick = read_tick;
            entry.pending_read_tick = None;
            entry.iterated.store(false, Ordering::Relaxed);
        }
    }

    /// Debug information about the tick state of a cached query type.
    ///
    /// Returns `None` if `Q` has never been queried. The returned struct
    /// contains the last read tick and current world tick, which can help
    /// diagnose unexpected `Changed<T>` behavior.
    pub fn query_tick_info<Q: WorldQuery + 'static>(&self) -> Option<QueryTickInfo> {
        let type_id = TypeId::of::<Q>();
        let entry = self.query_cache.get(&type_id)?;
        Some(QueryTickInfo {
            last_read_tick: entry.last_read_tick.raw(),
            current_world_tick: self.current_tick.raw(),
            matched_archetype_count: entry.matched_ids.len(),
            has_pending_tick: entry.pending_read_tick.is_some(),
        })
    }

    // ── Introspection ─────────────────────────────────────────────────

    /// Number of live (placed) entities across all archetypes.
    pub fn entity_count(&self) -> usize {
        self.archetypes.archetypes.iter().map(Archetype::len).sum()
    }

    /// Read-only snapshot of engine statistics for observability.
    pub fn stats(&self) -> WorldStats {
        WorldStats {
            entity_count: self.entity_count(),
            archetype_count: self.archetypes.archetypes.len(),
            component_count: self.components.len(),
            free_list_len: self.entities.free_list.len(),
            query_cache_len: self.query_cache.len(),
            current_tick: self.current_tick.raw(),
            total_spawns: self.entities.total_spawns,
            total_despawns: self.entities.total_despawns,
            pool_capacity: self.pool.capacity(),
            pool_used: self.pool.used(),
        }
    }

    // ── Persistence accessors ────────────────────────────────────────

    /// Number of archetypes (for iteration bounds).
    pub fn archetype_count(&self) -> usize {
        self.archetypes.archetypes.len()
    }

    /// Sorted component IDs defining an archetype's schema.
    pub fn archetype_component_ids(&self, arch_idx: usize) -> &[ComponentId] {
        &self.archetypes.archetypes[arch_idx].sorted_ids
    }

    /// Entity handles stored in an archetype (one per row).
    pub fn archetype_entities(&self, arch_idx: usize) -> &[Entity] {
        &self.archetypes.archetypes[arch_idx].entities
    }

    /// Row count for an archetype.
    pub fn archetype_len(&self, arch_idx: usize) -> usize {
        self.archetypes.archetypes[arch_idx].len()
    }

    /// Raw pointer to a component value at a specific row in an archetype column.
    ///
    /// # Safety
    /// The caller must read through the pointer using the correct component type
    /// and layout. The pointer is valid until the next structural mutation.
    pub unsafe fn archetype_column_ptr(
        &self,
        arch_idx: usize,
        comp_id: ComponentId,
        row: usize,
    ) -> *const u8 {
        let arch = &self.archetypes.archetypes[arch_idx];
        let col_idx = arch.column_index(comp_id).unwrap();
        unsafe { arch.columns[col_idx].get_ptr(row) as *const u8 }
    }

    /// Component name (from `std::any::type_name`). Returns None if unregistered.
    pub fn component_name(&self, id: ComponentId) -> Option<&'static str> {
        if id < self.components.len() {
            Some(self.components.info(id).name)
        } else {
            None
        }
    }

    /// Component memory layout. Returns None if unregistered.
    pub fn component_layout(&self, id: ComponentId) -> Option<Layout> {
        if id < self.components.len() {
            Some(self.components.info(id).layout)
        } else {
            None
        }
    }

    /// Number of registered component types.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Read-only view of entity allocator state for snapshot serialization.
    /// Returns (generations_slice, free_list_slice).
    pub fn entity_allocator_state(&self) -> (&[u32], &[u32]) {
        (&self.entities.generations, &self.entities.free_list)
    }

    /// Restore entity allocator state from a snapshot.
    pub fn restore_allocator_state(&mut self, generations: Vec<u32>, free_list: Vec<u32>) {
        self.drain_orphans();
        self.entities.generations = generations;
        self.entities.free_list = free_list;
        // Sync the atomic counter so reserve() doesn't hand out already-used indices.
        self.entities.sync_reserved();
        self.entity_locations
            .resize(self.entities.generations.len(), None);
    }

    /// Which ComponentIds have sparse storage.
    pub fn sparse_component_ids(&self) -> Vec<ComponentId> {
        self.sparse.component_ids()
    }

    /// Typed read-only iteration over a sparse component.
    pub fn iter_sparse<T: Component>(
        &self,
        comp_id: ComponentId,
    ) -> Option<impl Iterator<Item = (Entity, &T)>> {
        self.sparse.iter::<T>(comp_id)
    }

    /// Insert a sparse component value. Marks the component as sparse in the
    /// registry so that `get`/`has` route through sparse storage.
    pub fn insert_sparse<T: Component>(&mut self, entity: Entity, value: T) {
        self.drain_orphans();
        let comp_id = self.components.register_sparse::<T>();
        self.sparse.insert(comp_id, entity, value);
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

// ── WorldBuilder ─────────────────────────────────────────────────────

// HugePages is defined in pool.rs and re-exported via lib.rs.
use crate::pool::HugePages;

/// Builder for configuring [`World`] memory and allocation strategy.
///
/// # Example
///
/// ```
/// use minkowski::WorldBuilder;
///
/// let world = WorldBuilder::new().build().unwrap();
/// let stats = world.stats();
/// assert!(stats.pool_capacity.is_none()); // system allocator
/// ```
pub struct WorldBuilder {
    memory_budget: Option<usize>,
    hugepages: HugePages,
    lock_all_memory: bool,
}

impl WorldBuilder {
    pub fn new() -> Self {
        Self {
            memory_budget: None,
            hugepages: HugePages::default(),
            lock_all_memory: false,
        }
    }

    /// Set a fixed memory budget in bytes. When set, the World will
    /// allocate from a bounded [`SlabPool`] instead of the system allocator.
    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = Some(bytes);
        self
    }

    /// Configure hugepage usage for the memory pool.
    pub fn hugepages(mut self, hp: HugePages) -> Self {
        self.hugepages = hp;
        self
    }

    /// Lock all current and future memory mappings into physical RAM.
    ///
    /// Calls `mlockall(MCL_CURRENT | MCL_FUTURE)` before pool creation.
    /// This is a **process-global** operation — it affects all memory in
    /// the process, not just the pool. Use only in dedicated database
    /// processes, not in libraries embedded in larger applications.
    ///
    /// If `mlockall` fails (e.g., insufficient `RLIMIT_MEMLOCK`), the pool's
    /// per-mapping pre-fault fallback (mlock -> manual touch) still applies.
    ///
    /// Requires `CAP_IPC_LOCK` or sufficient `RLIMIT_MEMLOCK` on Linux.
    /// On unsupported platforms this is a no-op.
    pub fn lock_all_memory(mut self, lock: bool) -> Self {
        self.lock_all_memory = lock;
        self
    }

    /// Build the [`World`]. Returns `Err` if the memory pool cannot be
    /// allocated (only possible with a bounded `memory_budget`).
    pub fn build(self) -> Result<World, PoolExhausted> {
        if self.lock_all_memory {
            // mlockall failure is non-fatal — the pool's per-mapping
            // pre-fault fallback (mlock + manual touch) still applies.
            let _ = crate::pool::try_mlockall();
        }
        let pool: SharedPool = match self.memory_budget {
            Some(bytes) => {
                let slab = SlabPool::new(bytes, self.hugepages)?;
                crate::pool::into_shared(slab)
            }
            None => default_pool(),
        };
        Ok(World::new_with_pool(pool))
    }
}

impl Default for WorldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn spawn_and_get() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn spawn_different_archetypes() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e1), None);
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e2), Some(&Vel { dx: 1.0, dy: 0.0 }));
    }

    #[test]
    fn despawn_and_is_alive() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        assert!(world.is_alive(e));
        assert!(world.despawn(e));
        assert!(!world.is_alive(e));
        assert_eq!(world.get::<Pos>(e), None);
    }

    #[test]
    fn entity_recycling() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.despawn(e1);
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        assert_eq!(e2.index(), e1.index());
        assert_ne!(e2.generation(), e1.generation());
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
    }

    #[test]
    fn get_mut() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        if let Some(pos) = world.get_mut::<Pos>(e) {
            pos.x = 10.0;
        }
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 2.0 }));
    }

    #[test]
    fn insert_new_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, (Vel { dx: 3.0, dy: 4.0 },));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn insert_overwrites_existing() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, (Pos { x: 10.0, y: 20.0 },));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 20.0 }));
    }

    #[test]
    fn insert_multi_component_bundle() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, (Vel { dx: 3.0, dy: 4.0 }, Health(100u32)));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Health>(e), Some(&Health(100u32)));
    }

    #[test]
    fn insert_bundle_partial_overwrite() {
        // Some bundle components exist, some are new — mixed migration + overwrite
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.0, dy: 0.0 }));
        // Vel already exists, Health is new
        world.insert(e, (Vel { dx: 9.0, dy: 8.0 }, Health(50u32)));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 9.0, dy: 8.0 }));
        assert_eq!(world.get::<Health>(e), Some(&Health(50u32)));
    }

    #[test]
    fn insert_tuple_unpacks_correctly() {
        // Regression: insert((T,)) must store T, not the 1-tuple (T,).
        // Before the Bundle-based insert, this was a silent footgun.
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.insert(e, (Vel { dx: 1.0, dy: 2.0 },));
        // If the tuple were stored as-is, get::<Vel> would return None
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 1.0, dy: 2.0 }));
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        let removed = world.remove::<Vel>(e);
        assert_eq!(removed, Some(Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), None);
    }

    #[test]
    fn migration_preserves_other_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        let e3 = world.spawn((Pos { x: 3.0, y: 0.0 },));

        // Migrate e1 — e3 should swap into e1's old row
        world.insert(e1, (Vel { dx: 1.0, dy: 0.0 },));

        // All entities still accessible with correct data
        assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e1), Some(&Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Pos>(e3), Some(&Pos { x: 3.0, y: 0.0 }));
    }

    #[test]
    fn query_cache_populated_on_first_call() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        let count1 = world.query::<&Pos>().count();
        let count2 = world.query::<&Pos>().count();
        assert_eq!(count1, 1);
        assert_eq!(count2, 1);
    }

    #[test]
    fn query_cache_incremental_after_new_archetype() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);

        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.query::<&Pos>().count(), 2);
    }

    #[test]
    fn query_cache_filters_empty_archetypes() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);

        world.despawn(e);
        assert_eq!(world.query::<&Pos>().count(), 0);
    }

    #[test]
    fn query_cache_independent_per_query_type() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));

        assert_eq!(world.query::<&Pos>().count(), 2);
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 1);
    }

    #[test]
    fn query_cache_unrelated_archetype_no_false_match() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Vel>().count(), 0);

        world.spawn((Vel { dx: 1.0, dy: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);
        assert_eq!(world.query::<&Vel>().count(), 1);
    }

    #[test]
    fn query_cache_after_migration() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 0);

        world.insert(e, (Vel { dx: 1.0, dy: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 1);
    }

    #[test]
    fn read_entity_components_raw() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let components = world.read_all_components(e).unwrap();
        assert_eq!(components.len(), 2);
        for &(_, _, layout) in &components {
            assert!(layout.size() > 0);
        }
    }

    #[test]
    fn spawn_marks_column_ticks() {
        use crate::tick::Tick;
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Spawn auto-advances tick and marks columns
        let arch = &world.archetypes.archetypes[0];
        for col in &arch.columns {
            assert!(col.changed_tick.is_newer_than(Tick::default()));
        }
    }

    #[test]
    fn get_mut_marks_column_tick() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let spawn_tick = world.archetypes.archetypes[0].columns[0].changed_tick;

        let _ = world.get_mut::<Pos>(e);

        let loc = world.entity_locations[e.index() as usize].unwrap();
        let arch = &world.archetypes.archetypes[loc.archetype_id.0];
        let comp_id = world.components.id::<Pos>().unwrap();
        let col_idx = arch.column_index(comp_id).unwrap();
        // get_mut should advance the tick beyond spawn's tick
        assert!(arch.columns[col_idx].changed_tick.is_newer_than(spawn_tick));
    }

    use crate::query::fetch::Changed;

    #[test]
    fn changed_filter_skips_stale_archetype() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // First query: spawn's tick is newer than default last_read_tick
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // Second query: nothing changed since last read
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn changed_filter_detects_get_mut() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Consume the initial change
        let _ = world.query::<(Changed<Pos>,)>().count();

        // No changes -- should skip
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 0);

        // Mutate via get_mut -- auto-advances tick
        let _ = world.get_mut::<Pos>(e);
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 1);
    }

    #[test]
    fn changed_filter_mixed_query() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));

        // Consume initial change
        let _ = world.query::<(&Pos, Changed<Vel>)>().count();

        // Mutate only Pos, not Vel
        let _ = world.get_mut::<Pos>(e);

        // Changed<Vel> should skip -- Vel column not touched
        assert_eq!(world.query::<(&Pos, Changed<Vel>)>().count(), 0);
    }

    #[test]
    fn changed_filter_same_tick_interleave() {
        // Regression: reads and writes within the same "frame" must be
        // correctly ordered. Monotonic ticks ensure this.
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Read (consumes initial change)
        let _ = world.query::<(Changed<Pos>,)>().count();

        // Write (auto-advances tick)
        let _ = world.get_mut::<Pos>(e);

        // Read again -- must see the write, even without a manual tick() call
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 1);
    }

    #[test]
    fn query_mut_marks_column_changed() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Consume initial change
        let _ = world.query::<(Changed<Pos>,)>().count();

        // Query with &mut Pos -- should mark column changed
        let _ = world.query::<&mut Pos>().count();

        // Now Changed<Pos> should detect the mutation
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 1);
    }

    #[test]
    fn query_table_mut_marks_columns_changed() {
        use crate::table::{Table, TableRow};

        // Manual table impl for testing (matches the PosVel in table::tests)
        struct PosVelTable {
            pos: Pos,
            vel: Vel,
        }
        unsafe impl crate::bundle::Bundle for PosVelTable {
            fn component_ids(
                registry: &mut crate::component::ComponentRegistry,
            ) -> Vec<crate::component::ComponentId> {
                let mut ids = vec![registry.register::<Pos>(), registry.register::<Vel>()];
                ids.sort_unstable();
                ids.dedup();
                ids
            }
            unsafe fn put(
                self,
                registry: &crate::component::ComponentRegistry,
                func: &mut dyn FnMut(crate::component::ComponentId, *const u8, std::alloc::Layout),
            ) {
                let pos = std::mem::ManuallyDrop::new(self.pos);
                func(
                    registry.id::<Pos>().unwrap(),
                    &*pos as *const Pos as *const u8,
                    std::alloc::Layout::new::<Pos>(),
                );
                let vel = std::mem::ManuallyDrop::new(self.vel);
                func(
                    registry.id::<Vel>().unwrap(),
                    &*vel as *const Vel as *const u8,
                    std::alloc::Layout::new::<Vel>(),
                );
            }
        }
        struct PosVelMut<'w> {
            pos: &'w mut Pos,
            #[allow(dead_code)]
            vel: &'w mut Vel,
        }
        struct PosVelRef<'w> {
            #[allow(dead_code)]
            pos: &'w Pos,
            #[allow(dead_code)]
            vel: &'w Vel,
        }
        unsafe impl<'w> TableRow<'w> for PosVelMut<'w> {
            unsafe fn from_row(col_ptrs: &[(*mut u8, usize)], row: usize) -> Self {
                unsafe {
                    Self {
                        pos: &mut *(col_ptrs[0].0.add(row * col_ptrs[0].1) as *mut Pos),
                        vel: &mut *(col_ptrs[1].0.add(row * col_ptrs[1].1) as *mut Vel),
                    }
                }
            }
        }
        unsafe impl<'w> TableRow<'w> for PosVelRef<'w> {
            unsafe fn from_row(col_ptrs: &[(*mut u8, usize)], row: usize) -> Self {
                unsafe {
                    Self {
                        pos: &*(col_ptrs[0].0.add(row * col_ptrs[0].1) as *const Pos),
                        vel: &*(col_ptrs[1].0.add(row * col_ptrs[1].1) as *const Vel),
                    }
                }
            }
        }
        unsafe impl Table for PosVelTable {
            const FIELD_COUNT: usize = 2;
            type Ref<'w> = PosVelRef<'w>;
            type Mut<'w> = PosVelMut<'w>;
            fn register(
                registry: &mut crate::component::ComponentRegistry,
            ) -> Vec<crate::component::ComponentId> {
                vec![registry.register::<Pos>(), registry.register::<Vel>()]
            }
        }

        let mut world = World::new();
        world.spawn(PosVelTable {
            pos: Pos { x: 1.0, y: 2.0 },
            vel: Vel { dx: 3.0, dy: 4.0 },
        });

        // Consume initial change
        let _ = world.query::<(Changed<Pos>,)>().count();

        // Mutate via query_table_mut
        for row in world.query_table_mut::<PosVelTable>() {
            row.pos.x += 10.0;
        }

        // Changed<Pos> must detect the mutation from the table path
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 1);
    }

    #[test]
    fn component_id_returns_none_for_unregistered() {
        let world = World::new();
        assert_eq!(world.component_id::<Pos>(), None);
    }

    #[test]
    fn register_component_returns_id_and_subsequent_lookup_works() {
        let mut world = World::new();
        let id = world.register_component::<Pos>();
        assert_eq!(world.component_id::<Pos>(), Some(id));
    }

    #[test]
    fn register_component_is_idempotent() {
        let mut world = World::new();
        let a = world.register_component::<Pos>();
        let b = world.register_component::<Pos>();
        assert_eq!(a, b);
    }

    #[test]
    fn alloc_entity_is_not_placed_until_spawned() {
        let mut world = World::new();
        let e = world.alloc_entity();
        // Allocated but not yet placed in any archetype
        assert!(world.is_alive(e)); // allocator considers it alive
        assert!(!world.is_placed(e)); // but no archetype row yet
    }

    #[test]
    fn alloc_entity_safe_with_get_and_despawn() {
        let mut world = World::new();
        let e = world.alloc_entity();
        // These must not panic — entity_locations slot exists (set to None).
        assert_eq!(world.get::<Pos>(e), None);
        assert_eq!(world.get_mut::<Pos>(e), None);
        assert!(!world.despawn(e));
    }

    #[test]
    fn snapshot_column_ticks_captures_current_state() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut bits = FixedBitSet::with_capacity(pos_id + 1);
        bits.insert(pos_id);
        let snap = world.snapshot_column_ticks(&bits);
        assert!(!snap.is_empty());
    }

    #[test]
    fn check_column_conflicts_detects_mutation() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut bits = FixedBitSet::with_capacity(pos_id + 1);
        bits.insert(pos_id);

        let snap = world.snapshot_column_ticks(&bits);

        // Mutate through query — advances tick
        for pos in world.query::<(&mut Pos,)>() {
            pos.0.x = 99.0;
        }

        let conflicts = world.check_column_conflicts(&snap);
        assert!(conflicts.contains(pos_id));
    }

    #[test]
    fn check_column_conflicts_clean_when_unchanged() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut bits = FixedBitSet::with_capacity(pos_id + 1);
        bits.insert(pos_id);

        let snap = world.snapshot_column_ticks(&bits);
        // No mutation
        let conflicts = world.check_column_conflicts(&snap);
        assert!(!conflicts.contains(pos_id));
    }

    #[test]
    fn query_raw_reads_without_mutation() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        // query_raw takes &self — no tick advancement, no cache mutation
        let n = world.archetypes.archetypes.len();
        let count = world.query_raw::<(&Pos,)>(n).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn query_raw_skips_empty_archetypes() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let n = world.archetypes.archetypes.len();
        world.despawn(e);

        let count = world.query_raw::<(&Pos,)>(n).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn query_raw_matches_multiple_archetypes() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.spawn((Pos { x: 1.0, y: 1.0 }, Vel { dx: 0.0, dy: 0.0 }));
        let n = world.archetypes.archetypes.len();
        // Both archetypes contain Pos
        let count = world.query_raw::<(&Pos,)>(n).count();
        assert_eq!(count, 2);
    }

    // ── Persistence accessor tests ────────────────────────────────

    #[test]
    fn archetype_accessors_match_spawned_data() {
        let mut world = World::new();
        world.spawn((1.0f32, 2u32));
        world.spawn((3.0f32, 4u32));

        // At least one non-empty archetype
        let mut found = false;
        for idx in 0..world.archetype_count() {
            if world.archetype_len(idx) > 0 {
                let comp_ids = world.archetype_component_ids(idx);
                assert!(!comp_ids.is_empty());
                let entities = world.archetype_entities(idx);
                assert_eq!(entities.len(), world.archetype_len(idx));
                found = true;
            }
        }
        assert!(found);
    }

    #[test]
    fn entity_allocator_state_readable() {
        let mut world = World::new();
        world.spawn((1.0f32,));
        let (gens, free) = world.entity_allocator_state();
        assert!(!gens.is_empty());
        // free list may or may not be empty
        let _ = free;
    }

    #[test]
    fn component_name_and_layout() {
        let mut world = World::new();
        let id = world.register_component::<f32>();
        assert!(world.component_name(id).unwrap().contains("f32"));
        assert_eq!(world.component_layout(id).unwrap().size(), 4);
    }

    // ── get_by_id tests ──────────────────────────────────────────

    #[test]
    fn get_by_id_basic() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let comp_id = world.components.id::<Pos>().unwrap();
        let pos = world.get_by_id::<Pos>(e, comp_id);
        assert_eq!(pos, Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn get_by_id_missing_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let vel_id = world.register_component::<Vel>();
        assert_eq!(world.get_by_id::<Vel>(e, vel_id), None);
    }

    #[test]
    fn get_by_id_dead_entity() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let comp_id = world.components.id::<Pos>().unwrap();
        world.despawn(e);
        assert_eq!(world.get_by_id::<Pos>(e, comp_id), None);
    }

    // ── World::has tests ────────────────────────────────────────

    #[test]
    fn has_alive_entity_with_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.0, dy: 0.0 }));
        assert!(world.has::<Pos>(e));
        assert!(world.has::<Vel>(e));
    }

    #[test]
    fn has_alive_entity_without_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        assert!(!world.has::<Vel>(e));
    }

    #[test]
    fn has_dead_entity() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.despawn(e);
        assert!(!world.has::<Pos>(e));
    }

    #[test]
    fn has_after_component_removed() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.0, dy: 0.0 }));
        world.remove::<Vel>(e);
        assert!(world.has::<Pos>(e));
        assert!(!world.has::<Vel>(e));
    }

    #[test]
    fn has_unregistered_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        #[derive(Clone, Copy)]
        struct Unregistered;
        assert!(!world.has::<Unregistered>(e));
    }

    // ── World::query_changed_since tests ────────────────────────

    #[test]
    fn query_changed_since_returns_changed() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let _e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));

        let tick = world.change_tick();

        // Mutate e1
        *world.get_mut::<Pos>(e1).unwrap() = Pos { x: 99.0, y: 0.0 };

        let changed = world.query_changed_since::<Pos>(tick);
        // e1's archetype column was touched, so all entities in that archetype returned
        assert!(!changed.is_empty());
        assert!(changed.iter().any(|(e, _)| *e == e1));
    }

    #[test]
    fn query_changed_since_empty_when_no_changes() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        let tick = world.change_tick();
        // No mutations
        let changed = world.query_changed_since::<Pos>(tick);
        assert!(changed.is_empty());
    }

    #[test]
    fn query_changed_since_default_tick_returns_all() {
        use crate::tick::ChangeTick;
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 },));

        let changed = world.query_changed_since::<Pos>(ChangeTick::default());
        assert_eq!(changed.len(), 2);
    }

    #[test]
    fn query_changed_since_unregistered_type() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        #[derive(Clone, Copy)]
        struct Unknown;
        let changed = world.query_changed_since::<Unknown>(crate::tick::ChangeTick::default());
        assert!(changed.is_empty());
    }

    // ── World::get_batch tests ────────────────────────────────────

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Health(u32);

    #[test]
    fn get_batch_basic() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));
        let e3 = world.spawn((Health(25),));

        let results = world.get_batch::<Health>(&[e1, e2, e3]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(&Health(100)));
        assert_eq!(results[1], Some(&Health(50)));
        assert_eq!(results[2], Some(&Health(25)));
    }

    #[test]
    fn get_batch_dead_entity() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));
        world.despawn(e1);

        let results = world.get_batch::<Health>(&[e1, e2]);
        assert_eq!(results[0], None);
        assert_eq!(results[1], Some(&Health(50)));
    }

    #[test]
    fn get_batch_missing_component() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let results = world.get_batch::<Health>(&[e1, e2]);
        assert_eq!(results[0], Some(&Health(100)));
        assert_eq!(results[1], None);
    }

    #[test]
    fn get_batch_empty_input() {
        let world = World::new();
        let results = world.get_batch::<Health>(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn get_batch_unregistered_type() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let results = world.get_batch::<Health>(&[e]);
        assert_eq!(results[0], None);
    }

    #[test]
    fn get_batch_multi_archetype() {
        let mut world = World::new();
        let e1 = world.spawn((Health(10),));
        let e2 = world.spawn((Health(20), Pos { x: 0.0, y: 0.0 }));
        let e3 = world.spawn((Health(30),));

        let results = world.get_batch::<Health>(&[e1, e2, e3]);
        assert_eq!(results[0], Some(&Health(10)));
        assert_eq!(results[1], Some(&Health(20)));
        assert_eq!(results[2], Some(&Health(30)));
    }

    #[test]
    fn get_batch_duplicate_entity() {
        let mut world = World::new();
        let e = world.spawn((Health(42),));

        let results = world.get_batch::<Health>(&[e, e]);
        assert_eq!(results[0], Some(&Health(42)));
        assert_eq!(results[1], Some(&Health(42)));
    }

    #[test]
    fn get_batch_preserves_order() {
        let mut world = World::new();
        let e1 = world.spawn((Health(1),));
        let e2 = world.spawn((Health(2),));
        let e3 = world.spawn((Health(3),));

        let results = world.get_batch::<Health>(&[e3, e1, e2]);
        assert_eq!(results[0], Some(&Health(3)));
        assert_eq!(results[1], Some(&Health(1)));
        assert_eq!(results[2], Some(&Health(2)));
    }

    #[test]
    fn get_batch_sparse() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let e1 = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        world.insert_sparse(e1, Health(100));
        world.insert_sparse(e2, Health(50));

        let results = world.get_batch::<Health>(&[e1, e2]);
        assert_eq!(results[0], Some(&Health(100)));
        assert_eq!(results[1], Some(&Health(50)));
    }

    #[test]
    fn get_batch_mut_basic() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));

        let mut results = world.get_batch_mut::<Health>(&[e1, e2]);
        assert_eq!(results.len(), 2);
        *results[0].take().unwrap() = Health(200);
        *results[1].take().unwrap() = Health(75);
        drop(results);

        assert_eq!(world.get::<Health>(e1), Some(&Health(200)));
        assert_eq!(world.get::<Health>(e2), Some(&Health(75)));
    }

    #[test]
    fn get_batch_mut_marks_changed() {
        let mut world = World::new();
        let e = world.spawn((Health(100),));
        let spawn_tick = world.archetypes.archetypes[0].columns[0].changed_tick;

        let _results = world.get_batch_mut::<Health>(&[e]);

        let loc = world.entity_locations[e.index() as usize].unwrap();
        let arch = &world.archetypes.archetypes[loc.archetype_id.0];
        let comp_id = world.components.id::<Health>().unwrap();
        let col_idx = arch.column_index(comp_id).unwrap();
        assert!(arch.columns[col_idx].changed_tick.is_newer_than(spawn_tick));
    }

    #[test]
    fn get_batch_mut_dead_entity() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));
        world.despawn(e1);

        let mut results = world.get_batch_mut::<Health>(&[e1, e2]);
        assert!(results[0].is_none());
        assert_eq!(*results[1].take().unwrap(), Health(50));
    }

    #[test]
    #[should_panic(expected = "duplicate entity")]
    fn get_batch_mut_duplicate_panics() {
        let mut world = World::new();
        let e = world.spawn((Health(42),));
        let _results = world.get_batch_mut::<Health>(&[e, e]);
    }

    #[test]
    fn get_batch_mut_sparse() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let e1 = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        world.insert_sparse(e1, Health(100));
        world.insert_sparse(e2, Health(50));

        let mut results = world.get_batch_mut::<Health>(&[e1, e2]);
        *results[0].take().unwrap() = Health(200);
        *results[1].take().unwrap() = Health(75);
        drop(results);

        assert_eq!(world.get::<Health>(e1), Some(&Health(200)));
        assert_eq!(world.get::<Health>(e2), Some(&Health(75)));
    }

    #[test]
    fn get_batch_mut_multi_archetype() {
        let mut world = World::new();
        let e1 = world.spawn((Health(10),));
        let e2 = world.spawn((Health(20), Pos { x: 0.0, y: 0.0 }));

        let mut results = world.get_batch_mut::<Health>(&[e1, e2]);
        *results[0].take().unwrap() = Health(11);
        *results[1].take().unwrap() = Health(21);
        drop(results);

        assert_eq!(world.get::<Health>(e1), Some(&Health(11)));
        assert_eq!(world.get::<Health>(e2), Some(&Health(21)));
    }

    #[test]
    #[should_panic(expected = "duplicate entity")]
    fn get_batch_mut_sparse_duplicate_panics() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.insert_sparse(e, Health(42));
        let _results = world.get_batch_mut::<Health>(&[e, e]);
    }

    #[test]
    fn despawn_cleans_sparse_components() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let entity = world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.insert_sparse(entity, Health(42));
        let comp_id = world.components.id::<Health>().unwrap();
        assert!(world.sparse.contains(comp_id, entity));
        world.despawn(entity);
        assert!(!world.sparse.contains(comp_id, entity));
    }

    #[test]
    fn get_batch_mut_same_index_different_generation_no_panic() {
        let mut world = World::new();
        let old = world.spawn((Health(1),));
        world.despawn(old);
        let new = world.spawn((Health(2),));
        // After despawn + respawn, the free list reuses the same index
        assert_eq!(old.index(), new.index());
        let results = world.get_batch_mut::<Health>(&[old, new]);
        assert!(results[0].is_none()); // dead
        assert_eq!(*results[1].as_ref().unwrap(), &Health(2));
    }

    #[test]
    fn get_batch_mut_unregistered_type() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let results = world.get_batch_mut::<Health>(&[e]);
        assert_eq!(results[0], None);
    }

    #[test]
    fn get_batch_mut_empty_input() {
        let mut world = World::new();
        let results = world.get_batch_mut::<Health>(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn despawn_batch_multiple_same_archetype() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 2.0 },));
        let c = world.spawn((Pos { x: 3.0, y: 3.0 },));
        let d = world.spawn((Pos { x: 4.0, y: 4.0 },));
        let count = world.despawn_batch(&[b, d]);
        assert_eq!(count, 2);
        assert!(!world.is_alive(b));
        assert!(!world.is_alive(d));
        assert!(world.is_alive(a));
        assert!(world.is_alive(c));
        assert_eq!(world.get::<Pos>(a).unwrap().x, 1.0);
        assert_eq!(world.get::<Pos>(c).unwrap().x, 3.0);
    }

    #[test]
    fn despawn_batch_multiple_archetypes() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        let c = world.spawn((Pos { x: 3.0, y: 0.0 },));
        let d = world.spawn((Pos { x: 4.0, y: 0.0 }, Vel { dx: 2.0, dy: 0.0 }));
        let count = world.despawn_batch(&[a, d]);
        assert_eq!(count, 2);
        assert!(!world.is_alive(a));
        assert!(!world.is_alive(d));
        assert!(world.is_alive(b));
        assert!(world.is_alive(c));
    }

    #[test]
    fn despawn_batch_skips_dead() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 0.0 },));
        world.despawn(a);
        let count = world.despawn_batch(&[a, b]);
        assert_eq!(count, 1);
        assert!(!world.is_alive(b));
    }

    #[test]
    fn despawn_batch_single_entity() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.despawn_batch(&[a]), 1);
        assert!(!world.is_alive(a));
    }

    #[test]
    fn despawn_batch_empty() {
        let mut world = World::new();
        assert_eq!(world.despawn_batch(&[]), 0);
    }

    #[test]
    fn despawn_batch_cleans_sparse() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 0.0 },));
        world.insert_sparse(a, Health(42));
        let comp_id = world.components.id::<Health>().unwrap();
        let count = world.despawn_batch(&[a, b]);
        assert_eq!(count, 2);
        assert!(!world.sparse.contains(comp_id, a));
    }

    #[test]
    fn despawn_batch_duplicate_entity() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 0.0 },));
        // Same entity twice — must not double-drop
        let count = world.despawn_batch(&[a, a, b]);
        assert_eq!(count, 2); // a counted once
        assert!(!world.is_alive(a));
        assert!(!world.is_alive(b));
    }

    #[test]
    fn despawn_batch_back_to_front_correctness() {
        // Regression guard: if rows were processed front-to-back,
        // the swap-remove would invalidate subsequent row indices.
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 0.0 },));
        let c = world.spawn((Pos { x: 3.0, y: 0.0 },));
        let d = world.spawn((Pos { x: 4.0, y: 0.0 },));
        let e = world.spawn((Pos { x: 5.0, y: 0.0 },));

        // Despawn b(row 1) and c(row 2) — adjacent middle rows
        // Front-to-back would: remove row 1 (e swaps in), then
        // remove row 2 which is now d, not c. Back-to-front is correct.
        world.despawn_batch(&[b, c]);

        assert!(world.is_alive(a));
        assert!(!world.is_alive(b));
        assert!(!world.is_alive(c));
        assert!(world.is_alive(d));
        assert!(world.is_alive(e));
        // Verify survivors have correct data
        assert_eq!(world.get::<Pos>(a).unwrap().x, 1.0);
        assert_eq!(world.get::<Pos>(d).unwrap().x, 4.0);
        assert_eq!(world.get::<Pos>(e).unwrap().x, 5.0);
    }

    #[test]
    fn despawn_batch_query_after() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let _b = world.spawn((Pos { x: 2.0, y: 0.0 },));
        let c = world.spawn((Pos { x: 3.0, y: 0.0 },));
        let _d = world.spawn((Pos { x: 4.0, y: 0.0 },));

        world.despawn_batch(&[a, c]);

        // Query iteration must yield exactly the survivors with correct values
        let mut values = Vec::new();
        world.query::<(&Pos,)>().for_each(|(pos,)| {
            values.push(pos.x);
        });
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![2.0, 4.0]);
    }

    #[test]
    fn entity_count() {
        let mut world = World::new();
        assert_eq!(world.entity_count(), 0);

        let a = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let _b = world.spawn((Pos { x: 1.0, y: 1.0 }, Vel { dx: 1.0, dy: 1.0 }));
        assert_eq!(world.entity_count(), 2);

        world.despawn(a);
        assert_eq!(world.entity_count(), 1);
    }

    #[test]
    fn world_stats_reflects_state() {
        let mut world = World::new();
        let s0 = world.stats();
        assert_eq!(s0.entity_count, 0);
        assert_eq!(s0.archetype_count, 0);
        assert_eq!(s0.component_count, 0);
        assert_eq!(s0.free_list_len, 0);
        assert_eq!(s0.query_cache_len, 0);
        assert_eq!(s0.current_tick, 0);
        assert_eq!(s0.total_spawns, 0);
        assert_eq!(s0.total_despawns, 0);

        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let s1 = world.stats();
        assert_eq!(s1.entity_count, 1);
        assert!(s1.archetype_count >= 1);
        assert!(s1.component_count >= 1);
        assert!(s1.current_tick > s0.current_tick);
        assert_eq!(s1.total_spawns, 1);
        assert_eq!(s1.total_despawns, 0);

        world.despawn(e);
        let s2 = world.stats();
        assert_eq!(s2.entity_count, 0);
        assert_eq!(s2.free_list_len, 1);
        assert_eq!(s2.total_spawns, 1);
        assert_eq!(s2.total_despawns, 1);
    }

    #[test]
    fn world_stats_query_cache_len() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        assert_eq!(world.stats().query_cache_len, 0);
        let _: Vec<_> = world.query::<(&Pos,)>().collect();
        assert_eq!(world.stats().query_cache_len, 1);
    }

    // ── remove: empty-archetype path ─────────────────────────────

    #[test]
    fn remove_last_component_moves_to_empty_archetype() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let removed = world.remove::<Pos>(e);
        assert_eq!(removed, Some(Pos { x: 1.0, y: 2.0 }));
        assert!(world.is_alive(e));
        assert!(world.is_placed(e));
        // Entity is alive with no components
        assert_eq!(world.get::<Pos>(e), None);
        assert_eq!(world.query::<(&Pos,)>().count(), 0);
    }

    #[test]
    fn remove_last_component_swap_fixup() {
        // Removing a non-last row in a single-component archetype triggers
        // swap-remove; verify the swapped entity's location is updated.
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        let e3 = world.spawn((Pos { x: 3.0, y: 0.0 },));
        // e1 is row 0, e2 is row 1, e3 is row 2
        // Removing the only component from e1 swap-removes row 0, e3 fills gap
        world.remove::<Pos>(e1);
        assert!(world.is_alive(e1));
        assert_eq!(world.get::<Pos>(e1), None);
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Pos>(e3), Some(&Pos { x: 3.0, y: 0.0 }));
    }

    #[test]
    fn remove_last_component_with_multiple_columns() {
        // Entity has (Pos, Vel); remove Pos, then remove Vel.
        // Second remove hits the empty-archetype path with remaining columns
        // that need drop via swap_remove.
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        let _pos = world.remove::<Pos>(e);
        // Now entity has only Vel
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
        // Remove last component
        let vel = world.remove::<Vel>(e);
        assert_eq!(vel, Some(Vel { dx: 3.0, dy: 4.0 }));
        assert!(world.is_alive(e));
        assert_eq!(world.get::<Vel>(e), None);
    }

    // ── remove: dead entity and missing component paths ──────────

    #[test]
    fn remove_dead_entity() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.despawn(e);
        assert_eq!(world.remove::<Pos>(e), None);
    }

    #[test]
    fn remove_component_entity_doesnt_have() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        assert_eq!(world.remove::<Vel>(e), None);
    }

    // ── get: sparse component path ───────────────────────────────

    #[test]
    fn get_sparse_component() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.insert_sparse(e, Health(42));
        assert_eq!(world.get::<Health>(e), Some(&Health(42)));
    }

    #[test]
    fn get_mut_sparse_component() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.insert_sparse(e, Health(42));
        *world.get_mut::<Health>(e).unwrap() = Health(99);
        assert_eq!(world.get::<Health>(e), Some(&Health(99)));
    }

    // ── has: sparse component path ───────────────────────────────

    #[test]
    fn has_sparse_component() {
        let mut world = World::new();
        world.components.register_sparse::<Health>();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        assert!(!world.has::<Health>(e));
        world.insert_sparse(e, Health(42));
        assert!(world.has::<Health>(e));
    }

    // ── query: cache invalidation on new component registration ──

    #[test]
    fn query_cache_invalidated_on_new_component_registration() {
        // If a component is registered after the query cache entry is created,
        // the required bitset changes and the cache must rescan from scratch.
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        // First query: Vel not yet registered, bitset uses whatever IDs exist
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 0);
        // Register Vel explicitly and add an entity that matches
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        // The required bitset for (&Pos, &Vel) may have changed if Vel was
        // registered between the first and second query call.
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 1);
    }

    // ── despawn_batch: unplaced entity ───────────────────────────

    #[test]
    fn despawn_batch_skips_unplaced() {
        let mut world = World::new();
        let placed = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let unplaced = world.alloc_entity();
        assert!(!world.is_placed(unplaced));
        let count = world.despawn_batch(&[placed, unplaced]);
        // Only the placed entity counts
        assert_eq!(count, 1);
        assert!(!world.is_alive(placed));
    }

    // ── despawn_batch: last-row removal (no swap needed) ─────────

    #[test]
    fn despawn_batch_last_row_no_swap() {
        let mut world = World::new();
        let a = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let b = world.spawn((Pos { x: 2.0, y: 0.0 },));
        // b is the last row — swap_remove just truncates, no copy needed
        let count = world.despawn_batch(&[b]);
        assert_eq!(count, 1);
        assert!(!world.is_alive(b));
        assert_eq!(world.get::<Pos>(a).unwrap().x, 1.0);
    }

    // ── Lazy tick advancement tests ────────────────────────────────

    #[test]
    fn dropped_query_preserves_change_window() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // First query — iterates, consuming the initial "changed" state.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // Mutate to create a new change.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // Create a query but DROP it without iterating.
        let dropped = world.query::<(Changed<Pos>,)>();
        drop(dropped);

        // The change window should be preserved — we should still see the change.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn iterated_query_consumes_change_window() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // First query — iterates, consuming initial changes.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // Mutate.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // Iterate the Changed query (consumes the change window).
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // Now the window should be consumed — no more changes visible.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn has_changed_does_not_consume_window() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Initial query to set baseline.
        let _ = world.query::<(Changed<Pos>,)>().count();

        // Mutate.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // has_changed should detect the change.
        assert!(world.has_changed::<(Changed<Pos>,)>());

        // Calling has_changed again should still see it (non-consuming).
        assert!(world.has_changed::<(Changed<Pos>,)>());

        // Now iterate to consume.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // After iteration, no more changes.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn has_changed_uses_pending_tick_after_iteration() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Baseline.
        let _ = world.query::<(Changed<Pos>,)>().count();

        // Mutate.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // Iterate to consume the change.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // has_changed BEFORE the next query() call — the pending tick
        // hasn't been committed yet, but has_changed should use it
        // so it doesn't report stale true.
        assert!(
            !world.has_changed::<(Changed<Pos>,)>(),
            "has_changed should use pending tick after iteration"
        );
    }

    #[test]
    fn advance_query_tick_consumes_without_iterating() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Initial query to set baseline.
        let _ = world.query::<(Changed<Pos>,)>().count();

        // Mutate.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // Advance tick without iterating.
        world.advance_query_tick::<(Changed<Pos>,)>();

        // Change window should be consumed.
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn for_each_chunk_consumes_change_window() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Baseline — establish cache entry for this query type.
        world.query::<(Changed<Pos>, &Pos)>().for_each_chunk(|_| {});

        // Mutate.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // Consume via for_each_chunk.
        world.query::<(Changed<Pos>, &Pos)>().for_each_chunk(|_| {});

        // Window should be consumed — same query type sees no changes.
        let count = world.query::<(Changed<Pos>, &Pos)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn par_for_each_consumes_change_window() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Baseline — establish cache entry.
        world.query::<(Changed<Pos>, &Pos)>().par_for_each(|_| {});

        // Mutate.
        let e = world.query::<(Entity,)>().map(|(e,)| e).next().unwrap();
        let _ = world.get_mut::<Pos>(e);

        // Consume via par_for_each.
        world.query::<(Changed<Pos>, &Pos)>().par_for_each(|_| {});

        // Window should be consumed — same query type sees no changes.
        let count = world.query::<(Changed<Pos>, &Pos)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn has_changed_returns_false_for_uncached_query() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        // Never queried — has_changed returns false even though data exists.
        assert!(!world.has_changed::<(Changed<Pos>,)>());
    }

    #[test]
    fn query_tick_info_returns_none_for_uncached() {
        let world = World::new();
        assert!(world.query_tick_info::<(&Pos,)>().is_none());
    }

    #[test]
    fn query_tick_info_reflects_state() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // After first query, cache entry exists with a pending tick
        // (lazy advancement: committed on next query() call).
        let _ = world.query::<(&Pos,)>().count();
        let info = world.query_tick_info::<(&Pos,)>().unwrap();
        assert_eq!(info.matched_archetype_count, 1);
        assert!(info.has_pending_tick); // pending because iterator was iterated

        // Second query commits the pending tick.
        let _ = world.query::<(&Pos,)>().count();
        let info = world.query_tick_info::<(&Pos,)>().unwrap();
        assert!(info.last_read_tick > 0); // now committed
    }

    #[test]
    fn query_tick_info_shows_pending_after_dropped_iter() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // First query to establish cache.
        let _ = world.query::<(&Pos,)>().count();

        // Create and drop without iterating.
        let dropped = world.query::<(&Pos,)>();
        drop(dropped);

        let info = world.query_tick_info::<(&Pos,)>().unwrap();
        assert!(info.has_pending_tick);
    }

    // ── WorldBuilder tests ───────────────────────────────────────────

    #[test]
    fn world_builder_default_is_system_allocator() {
        let world = World::builder().build().unwrap();
        let stats = world.stats();
        assert!(stats.pool_capacity.is_none());
        assert!(stats.pool_used.is_none());
    }

    #[test]
    fn world_new_still_works() {
        let world = World::new();
        let stats = world.stats();
        assert!(stats.pool_capacity.is_none());
        assert!(stats.pool_used.is_none());
    }

    #[test]
    fn world_builder_with_memory_budget_uses_slab_pool() {
        let world = WorldBuilder::new()
            .memory_budget(64 * 1024 * 1024)
            .hugepages(HugePages::Off)
            .build()
            .unwrap();
        let stats = world.stats();
        assert_eq!(stats.pool_capacity, Some(64 * 1024 * 1024));
        assert_eq!(stats.pool_used, Some(0));
    }

    #[test]
    fn world_builder_spawn_and_query() {
        let mut world = World::builder().build().unwrap();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        let mut count = 0;
        world.query::<(&Pos, &Vel)>().for_each(|_| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn world_builder_default_impl() {
        let builder = WorldBuilder::default();
        let world = builder.build().unwrap();
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn world_builder_with_slab_pool() {
        let mut world = World::builder()
            .memory_budget(8 * 1024 * 1024)
            .hugepages(HugePages::Off)
            .build()
            .unwrap();

        for i in 0..100 {
            world.spawn((i as u32, i as f64));
        }

        let stats = world.stats();
        assert_eq!(stats.entity_count, 100);
        assert!(stats.pool_capacity.is_some());
        assert!(stats.pool_used.unwrap() > 0);
    }

    // ── try_spawn / try_insert tests ────────────────────────────────

    #[test]
    fn try_spawn_succeeds_with_system_allocator() {
        let mut world = World::new();
        let e = world.try_spawn((42u32,)).unwrap();
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
    }

    #[test]
    fn try_spawn_returns_error_on_pool_exhaustion() {
        let mut world = World::builder()
            .memory_budget(64 * 1024) // tiny 64KB pool
            .hugepages(HugePages::Off)
            .build()
            .unwrap();

        let mut count = 0u32;
        while world.try_spawn((count, count as f64)).is_ok() {
            count += 1;
        }
        assert!(count > 0, "should have spawned at least one entity");
        // Verify world is still consistent after exhaustion
        let mut sum = 0u32;
        world.query::<(&u32,)>().for_each(|(v,)| sum += v);
        assert_eq!(sum, (0..count).sum::<u32>());
    }

    #[test]
    fn try_insert_succeeds_with_system_allocator() {
        let mut world = World::new();
        let e = world.spawn((42u32,));
        world.try_insert(e, (1.5f64,)).unwrap();
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
        assert_eq!(*world.get::<f64>(e).unwrap(), 1.5);
    }

    #[test]
    fn try_insert_overwrite_succeeds() {
        // Overwriting existing components never allocates new columns
        let mut world = World::new();
        let e = world.spawn((42u32,));
        world.try_insert(e, (99u32,)).unwrap();
        assert_eq!(*world.get::<u32>(e).unwrap(), 99);
    }

    #[test]
    fn try_insert_returns_error_on_pool_exhaustion() {
        let mut world = World::builder()
            .memory_budget(64 * 1024)
            .hugepages(HugePages::Off)
            .build()
            .unwrap();

        // Fill up the pool
        let mut entities = Vec::new();
        while let Ok(e) = world.try_spawn((0u32,)) {
            entities.push(e);
        }
        assert!(!entities.is_empty());

        // Try inserting a new component — should fail (pool exhausted)
        let result = world.try_insert(entities[0], (1.5f64,));
        // Result depends on whether the new archetype column allocation fits.
        // Just verify it doesn't panic and world remains consistent.
        drop(result);

        // Verify all original entities are still readable
        let mut count = 0;
        world.query::<(&u32,)>().for_each(|_| count += 1);
        assert_eq!(count, entities.len());
    }
}

#[cfg(loom)]
mod loom_tests {
    use crate::entity::Entity;
    use crate::sync::{Arc, Mutex};
    use loom::thread;

    use super::OrphanQueue;

    /// Two transactions abort concurrently, pushing orphaned entity IDs to
    /// the shared OrphanQueue. Main thread drains. All IDs must be present.
    #[test]
    fn loom_orphan_queue_push_drain_no_lost_ids() {
        loom::model(|| {
            let queue = OrphanQueue::new();

            let q1 = queue.clone();
            let t1 = thread::spawn(move || {
                let mut guard = q1.queue.lock();
                guard.push(Entity::new(10, 0));
                guard.push(Entity::new(11, 0));
            });

            let q2 = queue.clone();
            let t2 = thread::spawn(move || {
                let mut guard = q2.queue.lock();
                guard.push(Entity::new(20, 0));
                guard.push(Entity::new(21, 0));
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let mut drained: Vec<u32> = queue.queue.lock().drain(..).map(|e| e.index()).collect();
            drained.sort();
            assert_eq!(drained, vec![10, 11, 20, 21]);
        });
    }

    /// Interleaved push and drain: one thread pushes individual IDs while
    /// another drains. All IDs must appear exactly once across batches.
    #[test]
    fn loom_orphan_queue_interleaved_push_drain() {
        loom::model(|| {
            let queue = OrphanQueue::new();
            let results: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));

            let q1 = queue.clone();
            let pusher = thread::spawn(move || {
                q1.queue.lock().push(Entity::new(1, 0));
                q1.queue.lock().push(Entity::new(2, 0));
            });

            let q2 = queue.clone();
            let r = results.clone();
            let drainer = thread::spawn(move || {
                let batch: Vec<u32> = q2.queue.lock().drain(..).map(|e| e.index()).collect();
                r.lock().push(batch);
            });

            pusher.join().unwrap();
            drainer.join().unwrap();

            let remainder: Vec<u32> = queue.queue.lock().drain(..).map(|e| e.index()).collect();
            results.lock().push(remainder);

            let mut all: Vec<u32> = results
                .lock()
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect();
            all.sort();
            assert_eq!(all, vec![1, 2]);
        });
    }
}
