use std::alloc::Layout;

use crate::entity::Entity;
use crate::storage::blob_vec::BlobVec;

const PAGE_SIZE: usize = 4096;
const EMPTY: u32 = u32::MAX;

/// A paged sparse set mapping `Entity` → dense index, backed by `BlobVec`
/// for type-erased value storage. Pages are lazily allocated.
///
/// The sparse array is indexed by `entity.index()`, split into pages of
/// [`PAGE_SIZE`]. Each page entry stores a dense index (`u32`) or [`EMPTY`].
/// Generation is validated against `dense_entities` to reject stale handles.
pub(crate) struct PagedSparseSet {
    // PERF: Two pointer chases per lookup (Vec data → Box page → array slot).
    // Inherent to paged design — flat array would be 16 GiB for u32 index space.
    // Prefer archetype-stored components for hot-path queries.
    pages: Vec<Option<Box<[u32; PAGE_SIZE]>>>,
    dense_entities: Vec<Entity>,
    pub(crate) dense_values: BlobVec,
}

#[allow(dead_code)]
impl PagedSparseSet {
    /// Creates a new empty `PagedSparseSet` for components with the given layout.
    pub fn new(item_layout: Layout, drop_fn: Option<unsafe fn(*mut u8)>) -> Self {
        Self {
            pages: Vec::new(),
            dense_entities: Vec::new(),
            dense_values: BlobVec::new(item_layout, drop_fn, 0),
        }
    }

    /// Number of entities stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense_entities.len()
    }

    /// Whether the set is empty.
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.dense_entities.is_empty()
    }

    /// Returns `true` if the entity is present (index in range, page allocated,
    /// dense index valid, and generation matches).
    pub fn contains(&self, entity: Entity) -> bool {
        self.dense_index(entity).is_some()
    }

    /// Returns a pointer to the entity's value, or `None` if absent.
    pub fn get(&self, entity: Entity) -> Option<*mut u8> {
        let dense = self.dense_index(entity)?;
        Some(unsafe { self.dense_values.get_ptr(dense) })
    }

    /// Returns a mutable pointer to the entity's value, or `None` if absent.
    /// Takes `&mut self` to prevent aliased `&mut T` from concurrent callers.
    pub fn get_mut(&mut self, entity: Entity) -> Option<*mut u8> {
        let dense = self.dense_index(entity)?;
        Some(unsafe { self.dense_values.get_ptr(dense) })
    }

    /// Inserts a value for the given entity. If the entity already has a value,
    /// the old value is dropped and overwritten.
    ///
    /// # Safety
    /// `ptr` must point to a valid, initialized value matching this set's layout.
    /// Caller must not double-drop the source value.
    pub unsafe fn insert(&mut self, entity: Entity, ptr: *const u8) {
        if let Some(dense) = self.dense_index(entity) {
            // Overwrite: drop old, copy new
            let dst = self.dense_values.get_ptr(dense);
            if let Some(drop_fn) = self.dense_values.drop_fn {
                drop_fn(dst);
            }
            let size = self.dense_values.item_layout.size();
            if size > 0 {
                std::ptr::copy_nonoverlapping(ptr, dst, size);
            }
        } else {
            // New entry
            debug_assert!(
                self.dense_entities.len() < EMPTY as usize,
                "PagedSparseSet overflow: {} entries exceeds u32 index space",
                self.dense_entities.len()
            );
            let dense = self.dense_entities.len() as u32;
            let idx = entity.index() as usize;
            let page_idx = idx / PAGE_SIZE;
            let slot = idx % PAGE_SIZE;

            // Grow pages vec if needed
            if page_idx >= self.pages.len() {
                self.pages.resize_with(page_idx + 1, || None);
            }
            // Allocate page if needed
            let page = self.pages[page_idx].get_or_insert_with(|| Box::new([EMPTY; PAGE_SIZE]));
            page[slot] = dense;

            self.dense_entities.push(entity);
            self.dense_values.push(ptr as *mut u8);
        }
    }

    /// Inserts a value, overwriting any existing value WITHOUT dropping the old
    /// one. Returns `true` if an overwrite occurred (caller owns the old bytes).
    ///
    /// # Safety
    /// Same as [`insert`]. If this returns `true`, the caller is responsible for
    /// ensuring the old value's destructor runs (e.g., via a `DropEntry` in the
    /// reverse changeset).
    pub unsafe fn insert_no_drop(&mut self, entity: Entity, ptr: *const u8) -> bool {
        if let Some(dense) = self.dense_index(entity) {
            // Overwrite: copy new bytes WITHOUT dropping old.
            let dst = self.dense_values.get_ptr(dense);
            let size = self.dense_values.item_layout.size();
            if size > 0 {
                std::ptr::copy_nonoverlapping(ptr, dst, size);
            }
            true
        } else {
            // New entry — same as insert.
            debug_assert!(
                self.dense_entities.len() < EMPTY as usize,
                "PagedSparseSet overflow: {} entries exceeds u32 index space",
                self.dense_entities.len()
            );
            let dense = self.dense_entities.len() as u32;
            let idx = entity.index() as usize;
            let page_idx = idx / PAGE_SIZE;
            let slot = idx % PAGE_SIZE;

            if page_idx >= self.pages.len() {
                self.pages.resize_with(page_idx + 1, || None);
            }
            let page = self.pages[page_idx].get_or_insert_with(|| Box::new([EMPTY; PAGE_SIZE]));
            page[slot] = dense;

            self.dense_entities.push(entity);
            self.dense_values.push(ptr as *mut u8);
            false
        }
    }

    /// Removes the entity's value, dropping it. Returns `true` if the entity
    /// was present.
    pub fn remove(&mut self, entity: Entity) -> bool {
        self.remove_internal(entity, true)
    }

    /// Removes the entity's value without dropping it. Returns `true` if the
    /// entity was present. Used when the caller has already extracted the value.
    pub fn remove_no_drop(&mut self, entity: Entity) -> bool {
        self.remove_internal(entity, false)
    }

    /// Iterates over all (entity, value_ptr) pairs.
    pub fn for_each(&self, mut f: impl FnMut(Entity, *mut u8)) {
        for (i, &entity) in self.dense_entities.iter().enumerate() {
            let ptr = unsafe { self.dense_values.get_ptr(i) };
            f(entity, ptr);
        }
    }

    // ── internals ───────────────────────────────────────────

    /// Resolves an entity to its dense index, validating generation.
    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let idx = entity.index() as usize;
        let page_idx = idx / PAGE_SIZE;
        let slot = idx % PAGE_SIZE;

        let page = self.pages.get(page_idx)?.as_ref()?;
        let dense = page[slot];
        if dense == EMPTY {
            return None;
        }
        let dense = dense as usize;
        // Generation check: the entity stored at this dense index must match.
        if dense < self.dense_entities.len() && self.dense_entities[dense] == entity {
            Some(dense)
        } else {
            None
        }
    }

    /// Shared remove logic. If `do_drop` is true, the removed value is dropped.
    fn remove_internal(&mut self, entity: Entity, do_drop: bool) -> bool {
        let Some(dense) = self.dense_index(entity) else {
            return false;
        };

        let idx = entity.index() as usize;
        let page_idx = idx / PAGE_SIZE;
        let slot = idx % PAGE_SIZE;

        // Clear the sparse entry for the removed entity.
        self.pages[page_idx].as_mut().unwrap()[slot] = EMPTY;

        let last = self.dense_entities.len() - 1;

        if dense != last {
            // Swap-remove: the last entity moves into the removed slot.
            let swapped_entity = self.dense_entities[last];
            let sw_idx = swapped_entity.index() as usize;
            let sw_page_idx = sw_idx / PAGE_SIZE;
            let sw_slot = sw_idx % PAGE_SIZE;

            // Update the swapped entity's sparse entry to point to the new dense index.
            self.pages[sw_page_idx].as_mut().unwrap()[sw_slot] = dense as u32;
        }

        // Remove from dense arrays.
        self.dense_entities.swap_remove(dense);
        if do_drop {
            unsafe { self.dense_values.swap_remove(dense) };
        } else {
            unsafe { self.dense_values.swap_remove_no_drop(dense) };
        }

        true
    }
}

// ── SparseStorage: typed wrapper around PagedSparseSet ───────────────

use std::collections::HashMap;
use std::mem::ManuallyDrop;

use crate::component::{Component, ComponentId};

unsafe fn drop_ptr<T>(ptr: *mut u8) {
    std::ptr::drop_in_place(ptr as *mut T);
}

/// Typed sparse component storage backed by `PagedSparseSet`.
/// Each `ComponentId` maps to a `PagedSparseSet` with the correct layout
/// and drop function. No `dyn Any`, no `Box<HashMap>`.
pub(crate) struct SparseStorage {
    storages: HashMap<ComponentId, PagedSparseSet>,
}

#[allow(dead_code)]
impl SparseStorage {
    pub fn new() -> Self {
        Self {
            storages: HashMap::new(),
        }
    }

    pub fn insert<T: Component>(&mut self, comp_id: ComponentId, entity: Entity, value: T) {
        let set = self.storages.entry(comp_id).or_insert_with(|| {
            let drop_fn = if std::mem::needs_drop::<T>() {
                Some(drop_ptr::<T> as unsafe fn(*mut u8))
            } else {
                None
            };
            PagedSparseSet::new(Layout::new::<T>(), drop_fn)
        });
        let mut value = ManuallyDrop::new(value);
        unsafe {
            set.insert(entity, &mut *value as *mut T as *const u8);
        }
    }

    pub fn get<T: Component>(&self, comp_id: ComponentId, entity: Entity) -> Option<&T> {
        let set = self.storages.get(&comp_id)?;
        debug_assert_eq!(
            set.dense_values.item_layout,
            Layout::new::<T>(),
            "SparseStorage type mismatch for ComponentId {comp_id}"
        );
        let ptr = set.get(entity)?;
        Some(unsafe { &*(ptr as *const T) })
    }

    pub fn get_mut<T: Component>(
        &mut self,
        comp_id: ComponentId,
        entity: Entity,
    ) -> Option<&mut T> {
        let set = self.storages.get_mut(&comp_id)?;
        debug_assert_eq!(
            set.dense_values.item_layout,
            Layout::new::<T>(),
            "SparseStorage type mismatch for ComponentId {comp_id}"
        );
        let ptr = set.get_mut(entity)?;
        Some(unsafe { &mut *(ptr as *mut T) })
    }

    pub fn contains(&self, comp_id: ComponentId, entity: Entity) -> bool {
        self.storages
            .get(&comp_id)
            .is_some_and(|set| set.contains(entity))
    }

    pub fn remove<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<T> {
        let set = self.storages.get_mut(&comp_id)?;
        debug_assert_eq!(
            set.dense_values.item_layout,
            Layout::new::<T>(),
            "SparseStorage type mismatch for ComponentId {comp_id}"
        );
        let ptr = set.get(entity)?;
        let value = unsafe { std::ptr::read(ptr as *const T) };
        set.remove_no_drop(entity);
        Some(value)
    }

    /// Returns the ComponentIds that have sparse storage allocated.
    pub fn component_ids(&self) -> Vec<ComponentId> {
        self.storages.keys().copied().collect()
    }

    /// Typed read-only iteration over a sparse component's entries.
    #[allow(clippy::iter_not_returning_iterator)]
    pub fn iter<T: Component>(
        &self,
        comp_id: ComponentId,
    ) -> Option<impl Iterator<Item = (Entity, &T)>> {
        let set = self.storages.get(&comp_id)?;
        debug_assert_eq!(
            set.dense_values.item_layout,
            Layout::new::<T>(),
            "SparseStorage type mismatch for ComponentId {comp_id}"
        );
        let mut items = Vec::with_capacity(set.len());
        set.for_each(|entity, ptr| {
            items.push((entity, unsafe { &*(ptr as *const T) }));
        });
        Some(items.into_iter())
    }

    /// Raw insert: copies bytes into sparse storage. Creates the
    /// `PagedSparseSet` if it doesn't exist for this `ComponentId`.
    ///
    /// # Safety
    /// `ptr` must point to a valid, initialized value matching `layout`.
    pub unsafe fn insert_raw(
        &mut self,
        comp_id: ComponentId,
        entity: Entity,
        ptr: *const u8,
        layout: Layout,
        drop_fn: Option<unsafe fn(*mut u8)>,
    ) {
        let set = self
            .storages
            .entry(comp_id)
            .or_insert_with(|| PagedSparseSet::new(layout, drop_fn));
        debug_assert_eq!(
            set.dense_values.item_layout, layout,
            "insert_raw layout mismatch for ComponentId {comp_id}"
        );
        set.insert(entity, ptr);
    }

    /// Like [`insert_raw`](Self::insert_raw), but does NOT drop the old value
    /// on overwrite. Returns `true` if an overwrite occurred. The caller is
    /// responsible for the old value's destructor (e.g., via a `DropEntry`).
    ///
    /// # Safety
    /// Same as `insert_raw`.
    pub unsafe fn insert_raw_no_drop(
        &mut self,
        comp_id: ComponentId,
        entity: Entity,
        ptr: *const u8,
        layout: Layout,
        drop_fn: Option<unsafe fn(*mut u8)>,
    ) -> bool {
        let set = self
            .storages
            .entry(comp_id)
            .or_insert_with(|| PagedSparseSet::new(layout, drop_fn));
        debug_assert_eq!(
            set.dense_values.item_layout, layout,
            "insert_raw_no_drop layout mismatch for ComponentId {comp_id}"
        );
        set.insert_no_drop(entity, ptr)
    }

    /// Raw read: returns a pointer to the sparse component data, or `None`.
    pub fn get_raw(&self, comp_id: ComponentId, entity: Entity) -> Option<*const u8> {
        let set = self.storages.get(&comp_id)?;
        set.get(entity).map(|p| p as *const u8)
    }

    /// Raw remove: removes and drops the sparse component. Returns `true`
    /// if the entity had the component.
    pub fn remove_raw(&mut self, comp_id: ComponentId, entity: Entity) -> bool {
        match self.storages.get_mut(&comp_id) {
            Some(set) => set.remove(entity),
            None => false,
        }
    }

    /// Like [`remove_raw`](Self::remove_raw), but does NOT drop the old value.
    /// Returns `true` if the entity had the component. The caller is
    /// responsible for the old value's destructor (e.g., via a `DropEntry`).
    pub fn remove_raw_no_drop(&mut self, comp_id: ComponentId, entity: Entity) -> bool {
        match self.storages.get_mut(&comp_id) {
            Some(set) => set.remove_no_drop(entity),
            None => false,
        }
    }

    /// Removes the entity from all sparse component sets.
    // PERF: O(sparse_types) per entity. Sparse types typically < 5; per-probe
    // cost is one page lookup (~3ns). Not worth per-entity tracking overhead.
    pub fn remove_all(&mut self, entity: Entity) {
        for set in self.storages.values_mut() {
            set.remove(entity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    /// Creates a `PagedSparseSet` for the given type `T`.
    fn make_set<T: 'static>() -> PagedSparseSet {
        let drop_fn = if std::mem::needs_drop::<T>() {
            Some(
                unsafe { |ptr: *mut u8| std::ptr::drop_in_place(ptr as *mut T) }
                    as unsafe fn(*mut u8),
            )
        } else {
            None
        };
        PagedSparseSet::new(Layout::new::<T>(), drop_fn)
    }

    /// Helper: insert a typed value into the set.
    unsafe fn insert_val<T>(set: &mut PagedSparseSet, entity: Entity, mut val: T) {
        set.insert(entity, &mut val as *mut T as *const u8);
        std::mem::forget(val);
    }

    /// Helper: read a typed value from a raw pointer.
    unsafe fn read_ptr<T: Copy>(ptr: *mut u8) -> T {
        *(ptr as *const T)
    }

    #[test]
    fn empty_set() {
        let set = make_set::<u32>();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        let e = Entity::new(0, 0);
        assert!(!set.contains(e));
        assert!(set.get(e).is_none());
    }

    #[test]
    fn insert_and_get() {
        let mut set = make_set::<u32>();
        let e = Entity::new(5, 1);
        unsafe { insert_val(&mut set, e, 42u32) };
        assert_eq!(set.len(), 1);
        assert!(set.contains(e));
        let ptr = set.get(e).unwrap();
        assert_eq!(unsafe { read_ptr::<u32>(ptr) }, 42);
    }

    #[test]
    fn cross_page_lookup() {
        let mut set = make_set::<u32>();
        let e0 = Entity::new(0, 0);
        let e5000 = Entity::new(5000, 0);
        unsafe {
            insert_val(&mut set, e0, 10u32);
            insert_val(&mut set, e5000, 20u32);
        }
        assert_eq!(set.len(), 2);
        assert_eq!(unsafe { read_ptr::<u32>(set.get(e0).unwrap()) }, 10);
        assert_eq!(unsafe { read_ptr::<u32>(set.get(e5000).unwrap()) }, 20);
    }

    #[test]
    fn generation_rejection() {
        let mut set = make_set::<u32>();
        let e_gen0 = Entity::new(7, 0);
        let e_gen1 = Entity::new(7, 1);
        unsafe { insert_val(&mut set, e_gen0, 99u32) };

        // Same index, different generation — must not find.
        assert!(!set.contains(e_gen1));
        assert!(set.get(e_gen1).is_none());

        // Original entity still accessible.
        assert!(set.contains(e_gen0));
    }

    #[test]
    fn remove_and_compact() {
        let mut set = make_set::<u32>();
        let e0 = Entity::new(0, 0);
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);
        unsafe {
            insert_val(&mut set, e0, 100u32);
            insert_val(&mut set, e1, 200u32);
            insert_val(&mut set, e2, 300u32);
        }
        assert_eq!(set.len(), 3);

        // Remove the middle entity.
        assert!(set.remove(e1));
        assert_eq!(set.len(), 2);

        // Removed entity is gone.
        assert!(!set.contains(e1));
        assert!(set.get(e1).is_none());

        // Remaining entities still accessible with correct values.
        assert!(set.contains(e0));
        assert!(set.contains(e2));
        assert_eq!(unsafe { read_ptr::<u32>(set.get(e0).unwrap()) }, 100);
        assert_eq!(unsafe { read_ptr::<u32>(set.get(e2).unwrap()) }, 300);
    }

    #[test]
    fn remove_absent_returns_false() {
        let mut set = make_set::<u32>();
        let e = Entity::new(42, 0);
        assert!(!set.remove(e));
    }

    #[test]
    fn iteration() {
        let mut set = make_set::<u32>();
        let entities: Vec<Entity> = (0..5).map(|i| Entity::new(i, 0)).collect();
        for (i, &e) in entities.iter().enumerate() {
            unsafe { insert_val(&mut set, e, (i as u32) * 10) };
        }

        let mut collected = Vec::new();
        set.for_each(|entity, ptr| {
            let val = unsafe { read_ptr::<u32>(ptr) };
            collected.push((entity, val));
        });

        assert_eq!(collected.len(), 5);
        // All entities must appear (order matches dense array = insertion order).
        for (i, &e) in entities.iter().enumerate() {
            assert_eq!(collected[i].0, e);
            assert_eq!(collected[i].1, (i as u32) * 10);
        }
    }

    #[test]
    fn insert_overwrite_drops_old_value() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        #[allow(dead_code)]
        struct Tracked(u32);
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let mut set = make_set::<Tracked>();
            let e = Entity::new(0, 0);
            unsafe {
                insert_val(&mut set, e, Tracked(1));
                insert_val(&mut set, e, Tracked(2)); // overwrite — old dropped
            }
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1); // old Tracked(1) dropped
            let ptr = set.get(e).unwrap();
            assert_eq!(unsafe { (*(ptr as *const Tracked)).0 }, 2);
            // set drops here — Tracked(2) dropped
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);
    }

    // ── SparseStorage tests ─────────────────────────────────────────

    #[derive(Debug, PartialEq)]
    struct Marker(u32);

    #[test]
    fn sparse_storage_insert_and_get() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert::<Marker>(0, e, Marker(42));
        assert_eq!(storage.get::<Marker>(0, e), Some(&Marker(42)));
    }

    #[test]
    fn sparse_storage_get_missing() {
        let storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn sparse_storage_remove() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert::<Marker>(0, e, Marker(42));
        let removed = storage.remove::<Marker>(0, e);
        assert_eq!(removed, Some(Marker(42)));
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn sparse_storage_component_ids_and_iter() {
        let mut storage = SparseStorage::new();
        let e1 = Entity::new(0, 0);
        let e2 = Entity::new(1, 0);
        storage.insert::<u32>(0, e1, 42u32);
        storage.insert::<u32>(0, e2, 99u32);
        let ids = storage.component_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 0);
        let entries: Vec<_> = storage.iter::<u32>(0).unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn sparse_storage_remove_all() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert::<u32>(0, e, 42u32);
        storage.insert::<Marker>(1, e, Marker(99));
        storage.remove_all(e);
        assert_eq!(storage.get::<u32>(0, e), None);
        assert_eq!(storage.get::<Marker>(1, e), None);
    }
}
