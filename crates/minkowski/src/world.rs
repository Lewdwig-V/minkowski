use crate::bundle::Bundle;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::{Entity, EntityAllocator};
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
use crate::storage::archetype::{Archetype, ArchetypeId, Archetypes};
use crate::storage::sparse::SparseStorage;
use crate::table::TableCache;
use crate::tick::Tick;
use fixedbitset::FixedBitSet;
use parking_lot::Mutex;
use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Unique identity for a World instance. Strategies capture this at
/// construction and assert it matches in begin/commit to prevent
/// cross-world corruption.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct WorldId(u64);

static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);

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
}

/// Shared queue for entity IDs orphaned by aborted transactions.
/// World owns the canonical instance; strategies clone the Arc handle.
#[derive(Clone)]
pub(crate) struct OrphanQueue(pub(crate) Arc<Mutex<Vec<Entity>>>);

impl OrphanQueue {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}

pub struct World {
    pub(crate) id: WorldId,
    pub(crate) entities: EntityAllocator,
    pub(crate) archetypes: Archetypes,
    pub(crate) components: ComponentRegistry,
    pub(crate) sparse: SparseStorage,
    pub(crate) entity_locations: Vec<Option<EntityLocation>>,
    pub(crate) table_cache: TableCache,
    pub(crate) query_cache: HashMap<TypeId, QueryCacheEntry>,
    pub(crate) current_tick: Tick,
    pub(crate) orphan_queue: OrphanQueue,
}

impl World {
    pub fn new() -> Self {
        Self {
            id: WorldId::next(),
            entities: EntityAllocator::new(),
            archetypes: Archetypes::new(),
            components: ComponentRegistry::new(),
            sparse: SparseStorage::new(),
            entity_locations: Vec::new(),
            table_cache: TableCache::new(),
            query_cache: HashMap::new(),
            current_tick: Tick::default(),
            orphan_queue: OrphanQueue::new(),
        }
    }

    /// Advance the internal tick and return the new value.
    /// Called automatically on every mutation and query — not user-facing.
    pub(crate) fn next_tick(&mut self) -> Tick {
        self.current_tick.advance()
    }

    /// Drain orphaned entity IDs from aborted transactions.
    /// Called automatically at the top of every &mut self entry point.
    fn drain_orphans(&mut self) {
        let mut queue = self.orphan_queue.0.lock();
        for entity in queue.drain(..) {
            self.entities.dealloc(entity);
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
            .get_or_create(&component_ids, &self.components);
        let entity = self.entities.alloc();
        let index = entity.index() as usize;

        if index >= self.entity_locations.len() {
            self.entity_locations.resize(index + 1, None);
        }

        let tick = self.next_tick();
        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let col = archetype.component_index[&comp_id];
                archetype.columns[col].push(ptr as *mut u8);
            });
        }
        for col in &mut archetype.columns {
            col.mark_changed(tick);
        }
        let row = archetype.entities.len();
        archetype.entities.push(entity);

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
        entity
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.drain_orphans();
        if !self.entities.is_alive(entity) {
            return false;
        }
        let index = entity.index() as usize;
        let location = match self.entity_locations[index] {
            Some(loc) => loc,
            None => return false,
        };

        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        let row = location.row;

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

        self.entity_locations[index] = None;
        self.entities.dealloc(entity);
        true
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
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
        let col_idx = archetype.component_index.get(&comp_id)?;
        unsafe {
            let ptr = archetype.columns[*col_idx].get_ptr(location.row) as *const T;
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
        let col_idx = *archetype.component_index.get(&comp_id)?;
        unsafe {
            let ptr = archetype.columns[col_idx].get_ptr_mut(location.row, tick) as *mut T;
            Some(&mut *ptr)
        }
    }

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
            });

        // Refresh required bitset in case new components were registered since
        // the cache entry was created. If the bitset changed, rescan from scratch.
        let fresh_required = Q::required_ids(&self.components);
        if fresh_required != entry.required {
            entry.required = fresh_required;
            entry.matched_ids.clear();
            entry.last_archetype_count = 0;
            entry.last_read_tick = Tick::default();
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

        // Extract what we need from the cache entry, dropping the borrow
        let matched_ids = entry.matched_ids.clone();
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
                    if let Some(&col_idx) = arch.component_index.get(&comp_id) {
                        self.archetypes.archetypes[aid.0].columns[col_idx].mark_changed(tick);
                    }
                }
            }
        }

        // Update last_read_tick with a fresh tick (distinct from any mutation tick)
        let read_tick = self.next_tick();
        if let Some(entry) = self.query_cache.get_mut(&type_id) {
            entry.last_read_tick = read_tick;
        }

        // Pass 2: build fetches (only immutable borrows of archetypes from here)
        let fetches: Vec<_> = filtered_ids
            .iter()
            .map(|&aid| {
                let arch = &self.archetypes.archetypes[aid.0];
                (Q::init_fetch(arch, &self.components), arch.len())
            })
            .collect();

        QueryIter::new(fetches)
    }

    /// Resolve column pointers for a table's archetype.
    fn resolve_table_ptrs<T: crate::table::Table>(&mut self) -> (Vec<(*mut u8, usize)>, usize) {
        let desc = self
            .table_cache
            .get_or_create::<T>(&mut self.components, &mut self.archetypes);
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
        let desc = self
            .table_cache
            .get_or_create::<T>(&mut self.components, &mut self.archetypes);
        let arch_id = desc.archetype_id;
        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        for col in &mut archetype.columns {
            col.mark_changed(tick);
        }
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.drain_orphans();
        assert!(self.is_alive(entity), "entity is not alive");
        let index = entity.index() as usize;
        let location = self.entity_locations[index].unwrap();
        let comp_id = self.components.register::<T>();

        // If entity already has this component, overwrite in-place
        {
            let has_component = self.archetypes.archetypes[location.archetype_id.0]
                .component_ids
                .contains(comp_id);
            if has_component {
                let tick = self.next_tick();
                let src_arch = &mut self.archetypes.archetypes[location.archetype_id.0];
                let col_idx = src_arch.component_index[&comp_id];
                unsafe {
                    let ptr = src_arch.columns[col_idx].get_ptr_mut(location.row, tick) as *mut T;
                    std::ptr::drop_in_place(ptr);
                    std::ptr::write(ptr, component);
                }
                return;
            }
        }

        // Compute target archetype: source components + new component
        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        let mut target_ids = src_arch.sorted_ids.clone();
        target_ids.push(comp_id);
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id = self.archetypes.get_or_create(&target_ids, &self.components);
        let tick = self.next_tick();

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns: read ptr from source, push to target, swap_remove_no_drop source
        for (&cid, &src_col) in &src_arch.component_index {
            if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        // Write the new component into target
        let tgt_col = target_arch.component_index[&comp_id];
        unsafe {
            let comp = std::mem::ManuallyDrop::new(component);
            target_arch.columns[tgt_col].push(&*comp as *const T as *mut u8);
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
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.drain_orphans();
        if !self.is_alive(entity) {
            return None;
        }
        let index = entity.index() as usize;
        let location = self.entity_locations[index]?;
        let comp_id = self.components.id::<T>()?;

        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        if !src_arch.component_ids.contains(comp_id) {
            return None;
        }

        // Read the component value before migration
        let removed = unsafe {
            let col_idx = src_arch.component_index[&comp_id];
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
            let removed_col = arch.component_index[&comp_id];
            unsafe {
                arch.columns[removed_col].swap_remove_no_drop(src_row);
            }
            // swap_remove with drop for remaining columns
            for (&cid, &col_idx) in &arch.component_index {
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
            let empty_arch_id = self.archetypes.get_or_create(&[], &self.components);
            let empty_arch = &mut self.archetypes.archetypes[empty_arch_id.0];
            empty_arch.entities.push(entity);
            self.entity_locations[index] = Some(EntityLocation {
                archetype_id: empty_arch_id,
                row: empty_arch.entities.len() - 1,
            });
            return Some(removed);
        }

        let target_arch_id = self.archetypes.get_or_create(&target_ids, &self.components);

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns (skip removed component)
        for (&cid, &src_col) in &src_arch.component_index {
            if cid == comp_id {
                // Already read — just discard from source
                unsafe {
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            } else if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
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
                let col_idx = archetype.component_index[&comp_id];
                let info = self.components.info(comp_id);
                let ptr = unsafe { archetype.columns[col_idx].get_ptr(location.row) };
                (comp_id, ptr as *const u8, info.layout)
            })
            .collect();

        Some(components)
    }

    /// Snapshot the `changed_tick` of every column matching the given component
    /// bitset. Returns a Vec of (ArchetypeId index, ComponentId, Tick) triples.
    /// Used by OptimisticTx for read-set validation.
    #[allow(dead_code)]
    pub(crate) fn snapshot_column_ticks(
        &self,
        component_ids: &FixedBitSet,
    ) -> Vec<(usize, ComponentId, crate::tick::Tick)> {
        let mut ticks = Vec::new();
        for arch in &self.archetypes.archetypes {
            for comp_id in component_ids.ones() {
                if let Some(&col_idx) = arch.component_index.get(&comp_id) {
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
        for &(arch_idx, comp_id, begin_tick) in snapshot {
            if let Some(arch) = self.archetypes.archetypes.get(arch_idx) {
                if let Some(&col_idx) = arch.component_index.get(&comp_id) {
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
}

impl Default for World {
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
        world.insert(e, Vel { dx: 3.0, dy: 4.0 });
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn insert_overwrites_existing() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, Pos { x: 10.0, y: 20.0 });
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 20.0 }));
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
        world.insert(e1, Vel { dx: 1.0, dy: 0.0 });

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

        world.insert(e, Vel { dx: 1.0, dy: 0.0 });
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
        let col_idx = arch.component_index[&comp_id];
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
                Self {
                    pos: &mut *(col_ptrs[0].0.add(row * col_ptrs[0].1) as *mut Pos),
                    vel: &mut *(col_ptrs[1].0.add(row * col_ptrs[1].1) as *mut Vel),
                }
            }
        }
        unsafe impl<'w> TableRow<'w> for PosVelRef<'w> {
            unsafe fn from_row(col_ptrs: &[(*mut u8, usize)], row: usize) -> Self {
                Self {
                    pos: &*(col_ptrs[0].0.add(row * col_ptrs[0].1) as *const Pos),
                    vel: &*(col_ptrs[1].0.add(row * col_ptrs[1].1) as *const Vel),
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
}
