use std::alloc::Layout;
use std::ptr::NonNull;

use crate::entity::Entity;
use crate::sync::Arc;

use crate::pool::{PoolExhausted, SlabPool};
use crate::storage::blob_vec::BlobVec;
use crate::storage::pool_vec::PoolVec;

/// A sparse page: dense indices for `PAGE_SIZE` consecutive entity indices.
type Page = [u32; PAGE_SIZE];

/// Entries per sparse page. 1024 × `u32` = 4 KiB, exactly the pool's 4 KiB
/// slab class (a 16 KiB page would round up to the 64 KiB class, 4× waste).
const PAGE_SIZE: usize = 1024;
const EMPTY: u32 = u32::MAX;

const _: () = assert!(size_of::<Option<NonNull<Page>>>() == size_of::<usize>());

/// A paged sparse set mapping `Entity` → dense index, backed by `BlobVec`
/// for type-erased value storage. Pages are lazily allocated.
///
/// The sparse array is indexed by `entity.index()`, split into pages of
/// [`PAGE_SIZE`]. Each page entry stores a dense index (`u32`) or [`EMPTY`].
/// Generation is validated against `dense_entities` to reject stale handles.
///
/// # Pool backing
///
/// Everything per entry lives in the world's pool, so sparse storage counts
/// against the memory budget: the dense `BlobVec`, `dense_entities`, and the
/// 4 KiB sparse pages (one per 1024 entity indices touched). Only the page
/// directory (`pages`, 8 bytes per 1024 indices) stays on the heap. `insert`
/// aborts on exhaustion like other infallible paths; `try_insert` reserves
/// the page and both dense vectors first and returns `Err` instead.
pub(crate) struct PagedSparseSet {
    // PERF: Two pointer chases per lookup (directory → page → slot).
    // Inherent to paged design — flat array would be 16 GiB for u32 index space.
    // Prefer archetype-stored components for hot-path queries.
    /// Page handles: one pointer each (`Option<NonNull<_>>` uses the niche),
    /// so the heap directory costs 8 B per 1024 indices. Pages are pool
    /// blocks owned by this set and freed in `Drop`.
    pages: Vec<Option<NonNull<Page>>>,
    dense_entities: PoolVec<Entity>,
    pub(crate) dense_values: BlobVec,
    pool: Arc<SlabPool>,
}

impl PagedSparseSet {
    /// Creates a new empty `PagedSparseSet` for components with the given layout.
    pub fn new(
        item_layout: Layout,
        drop_fn: Option<unsafe fn(*mut u8)>,
        pool: Arc<SlabPool>,
    ) -> Self {
        Self {
            pages: Vec::new(),
            dense_entities: PoolVec::new(Arc::clone(&pool)),
            dense_values: BlobVec::new(item_layout, drop_fn, 0, Arc::clone(&pool)),
            pool,
        }
    }

    /// Number of entities stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense_entities.len()
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
            // Overwrite: panic-safe replacement (#180) — the new value is
            // committed to the slot before the old value's destructor runs.
            // SAFETY: `dense` is a valid index; `ptr` is valid per caller
            // contract and does not alias our BlobVec.
            unsafe {
                self.dense_values
                    .replace_protected(dense, ptr)
                    .expect("pool exhausted during sparse overwrite");
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

            if let Err(e) = self.try_ensure_page(page_idx) {
                std::alloc::handle_alloc_error(e.requested);
            }
            self.page_mut(page_idx).expect("page ensured above")[slot] = dense;

            self.dense_entities.push(entity);
            // SAFETY: `ptr` points to a valid value matching this set's layout
            // per caller contract. BlobVec::push copies the bytes.
            unsafe { self.dense_values.push(ptr as *mut u8) };
        }
    }

    /// Like [`insert`](Self::insert), returning `Err(PoolExhausted)` instead
    /// of aborting when the pool cannot hold the value. On `Err` the set is
    /// unchanged and the caller still owns the value at `ptr`.
    ///
    /// # Safety
    /// Same contract as [`insert`](Self::insert).
    pub unsafe fn try_insert(
        &mut self,
        entity: Entity,
        ptr: *const u8,
    ) -> Result<(), PoolExhausted> {
        if let Some(dense) = self.dense_index(entity) {
            // SAFETY: as in insert's overwrite path. On Err nothing was written.
            return unsafe { self.dense_values.replace_protected(dense, ptr) };
        }
        // New entry: ensure the sparse page and reserve both dense vectors,
        // then the plain insert cannot allocate in the pool. A page created
        // here on a later Err holds only EMPTY slots (spare capacity, no
        // observable entry).
        self.try_ensure_page(entity.index() as usize / PAGE_SIZE)?;
        self.dense_entities.try_reserve(1)?;
        self.dense_values.try_reserve(1)?;
        // SAFETY: forwarded caller contract.
        unsafe { self.insert(entity, ptr) };
        Ok(())
    }

    /// Removes the entity's value, dropping it. Returns `true` if the entity
    /// was present.
    pub fn remove(&mut self, entity: Entity) -> bool {
        self.remove_internal(entity)
    }

    /// Iterates over all (entity, value_ptr) pairs.
    pub fn for_each(&self, mut f: impl FnMut(Entity, *mut u8)) {
        for (i, &entity) in self.dense_entities.iter().enumerate() {
            let ptr = unsafe { self.dense_values.get_ptr(i) };
            f(entity, ptr);
        }
    }

    // ── internals ───────────────────────────────────────────

    /// Make sure page `page_idx` exists, allocating it (all `EMPTY`) from the
    /// pool. On `Err` no page was added. The directory grows on the heap.
    fn try_ensure_page(&mut self, page_idx: usize) -> Result<(), PoolExhausted> {
        if self.pages.get(page_idx).is_some_and(Option::is_some) {
            return Ok(());
        }
        let page: NonNull<Page> = self.pool.allocate(Layout::new::<Page>())?.cast();
        // SAFETY: freshly allocated block of Page's size and alignment; every
        // byte 0xFF makes every u32 slot EMPTY (u32::MAX).
        unsafe { page.cast::<u8>().write_bytes(0xFF, size_of::<Page>()) };
        if page_idx >= self.pages.len() {
            self.pages.resize_with(page_idx + 1, || None);
        }
        self.pages[page_idx] = Some(page);
        Ok(())
    }

    fn page(&self, page_idx: usize) -> Option<&Page> {
        let page = (*self.pages.get(page_idx)?)?;
        // SAFETY: page handles are live pool blocks owned by this set,
        // initialized on allocation; &self gives shared access.
        Some(unsafe { page.as_ref() })
    }

    fn page_mut(&mut self, page_idx: usize) -> Option<&mut Page> {
        let mut page = (*self.pages.get(page_idx)?)?;
        // SAFETY: as in page(); &mut self gives exclusive access.
        Some(unsafe { page.as_mut() })
    }

    /// Resolves an entity to its dense index, validating generation.
    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let idx = entity.index() as usize;
        let page_idx = idx / PAGE_SIZE;
        let slot = idx % PAGE_SIZE;

        let page = self.page(page_idx)?;
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

    /// Shared remove logic. The removed value is dropped.
    fn remove_internal(&mut self, entity: Entity) -> bool {
        let Some(dense) = self.dense_index(entity) else {
            return false;
        };

        let idx = entity.index() as usize;
        let page_idx = idx / PAGE_SIZE;
        let slot = idx % PAGE_SIZE;

        // Clear the sparse entry for the removed entity.
        self.page_mut(page_idx).expect("entity's page exists")[slot] = EMPTY;

        let last = self.dense_entities.len() - 1;

        if dense != last {
            // Swap-remove: the last entity moves into the removed slot.
            let swapped_entity = self.dense_entities[last];
            let sw_idx = swapped_entity.index() as usize;
            let sw_page_idx = sw_idx / PAGE_SIZE;
            let sw_slot = sw_idx % PAGE_SIZE;

            // Update the swapped entity's sparse entry to point to the new dense index.
            self.page_mut(sw_page_idx)
                .expect("swapped entity's page exists")[sw_slot] = dense as u32;
        }

        // Remove from dense arrays.
        self.dense_entities.swap_remove(dense);
        unsafe { self.dense_values.swap_remove(dense) };

        true
    }
}

impl Drop for PagedSparseSet {
    fn drop(&mut self) {
        for page in self.pages.iter().flatten() {
            // SAFETY: each handle is a live block allocated from self.pool with
            // Page's layout and is not used after this point.
            unsafe { self.pool.deallocate(page.cast(), Layout::new::<Page>()) };
        }
    }
}

// SAFETY: the set exclusively owns its page blocks (like Box<Page>) and only
// mutates them through &mut self; every other field is Send + Sync.
unsafe impl Send for PagedSparseSet {}
// SAFETY: &PagedSparseSet only reads pages; see Send.
unsafe impl Sync for PagedSparseSet {}

// ── SparseStorage: typed wrapper around PagedSparseSet ───────────────

use std::collections::HashMap;
use std::mem::ManuallyDrop;

use crate::component::{Component, ComponentId};

use crate::component::drop_ptr;

/// Typed sparse component storage backed by `PagedSparseSet`.
/// Each `ComponentId` maps to a `PagedSparseSet` with the correct layout
/// and drop function. No `dyn Any`, no `Box<HashMap>`.
pub(crate) struct SparseStorage {
    storages: HashMap<ComponentId, PagedSparseSet>,
    pool: Arc<SlabPool>,
}

impl SparseStorage {
    pub fn new(pool: Arc<SlabPool>) -> Self {
        Self {
            storages: HashMap::new(),
            pool,
        }
    }

    pub fn insert<T: Component>(&mut self, comp_id: ComponentId, entity: Entity, value: T) {
        let pool = self.pool.clone();
        let set = self.storages.entry(comp_id).or_insert_with(|| {
            let drop_fn = if std::mem::needs_drop::<T>() {
                Some(drop_ptr::<T> as unsafe fn(*mut u8))
            } else {
                None
            };
            PagedSparseSet::new(Layout::new::<T>(), drop_fn, pool)
        });
        let mut value = ManuallyDrop::new(value);
        unsafe {
            set.insert(entity, &mut *value as *mut T as *const u8);
        }
    }

    /// Like [`insert`](Self::insert), returning `Err(PoolExhausted)` instead
    /// of aborting. On `Err` the storage is unchanged and `value` is dropped.
    pub fn try_insert<T: Component>(
        &mut self,
        comp_id: ComponentId,
        entity: Entity,
        value: T,
    ) -> Result<(), PoolExhausted> {
        let mut value = ManuallyDrop::new(value);
        let ptr = &mut *value as *mut T as *const u8;
        // SAFETY (both arms): value is a valid T matching the set's layout; on
        // Ok ownership moved into the set, on Err nothing was written and we
        // drop it below.
        let result = if let Some(set) = self.storages.get_mut(&comp_id) {
            unsafe { set.try_insert(entity, ptr) }
        } else {
            // First value for this component: build the set off to the side
            // and publish it only on success, so Err leaves no empty set
            // behind (a failed set drops here, returning any page it took).
            let drop_fn = if std::mem::needs_drop::<T>() {
                Some(drop_ptr::<T> as unsafe fn(*mut u8))
            } else {
                None
            };
            let mut set = PagedSparseSet::new(Layout::new::<T>(), drop_fn, self.pool.clone());
            let r = unsafe { set.try_insert(entity, ptr) };
            if r.is_ok() {
                self.storages.insert(comp_id, set);
            }
            r
        };
        if result.is_err() {
            // SAFETY: the set did not take ownership.
            unsafe { ManuallyDrop::drop(&mut value) };
        }
        result
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
        let pool = self.pool.clone();
        let set = self
            .storages
            .entry(comp_id)
            .or_insert_with(|| PagedSparseSet::new(layout, drop_fn, pool));
        debug_assert_eq!(
            set.dense_values.item_layout, layout,
            "insert_raw layout mismatch for ComponentId {comp_id}"
        );
        // SAFETY: Caller guarantees `ptr` is valid and matches `layout`.
        unsafe { set.insert(entity, ptr) };
    }

    /// Raw remove: removes and drops the sparse component. Returns `true`
    /// if the entity had the component.
    pub fn remove_raw(&mut self, comp_id: ComponentId, entity: Entity) -> bool {
        match self.storages.get_mut(&comp_id) {
            Some(set) => set.remove(entity),
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
    use crate::pool::default_pool;
    use std::alloc::Layout;

    #[test]
    fn sparse_pages_live_in_the_pool() {
        let pool = Arc::new(SlabPool::new(1 << 20, crate::pool::HugePages::Off, false).unwrap());
        let mut set = PagedSparseSet::new(Layout::new::<u64>(), None, Arc::clone(&pool));
        let v = 5u64;
        unsafe { set.insert(Entity::new(0, 0), &v as *const u64 as *const u8) };
        // Pre-size the dense vectors so only the new page touches the pool.
        set.dense_entities.reserve(8);
        set.dense_values.reserve(8);
        pool.flush_caches();
        let before = pool.used();
        // A second page costs exactly one 4 KiB slab block (no class waste).
        unsafe {
            set.insert(
                Entity::new(PAGE_SIZE as u32, 0),
                &v as *const u64 as *const u8,
            );
        }
        pool.flush_caches();
        assert_eq!(pool.used() - before, PAGE_SIZE * 4);
        drop(set);
        pool.flush_caches();
        assert_eq!(pool.used(), 0, "pages are returned on drop");
    }

    #[test]
    fn try_insert_fails_on_page_allocation_and_leaves_set_unchanged() {
        let pool = Arc::new(SlabPool::new(1 << 20, crate::pool::HugePages::Off, false).unwrap());
        let mut set = PagedSparseSet::new(Layout::new::<u64>(), None, Arc::clone(&pool));
        let v = 9u64;
        unsafe { set.try_insert(Entity::new(1, 0), &v as *const u64 as *const u8) }.unwrap();
        // Dense vectors have room; only a new page would need the pool.
        set.dense_entities.reserve(8);
        set.dense_values.reserve(8);
        for size in [1 << 20, 65_536, 16_384, 4096, 1024, 256, 64] {
            let layout = Layout::from_size_align(size, 8).unwrap();
            while pool.allocate(layout).is_ok() {}
        }
        let far = Entity::new(3 * PAGE_SIZE as u32, 0);
        let r = unsafe { set.try_insert(far, &v as *const u64 as *const u8) };
        assert!(r.is_err(), "the page cannot be allocated");
        assert_eq!(set.len(), 1);
        assert!(!set.contains(far));
        // Same page as an existing entry: no page needed, succeeds.
        unsafe { set.try_insert(Entity::new(2, 0), &v as *const u64 as *const u8) }.unwrap();
        assert_eq!(set.len(), 2);
    }

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
        PagedSparseSet::new(Layout::new::<T>(), drop_fn, default_pool())
    }

    /// Helper: insert a typed value into the set.
    unsafe fn insert_val<T>(set: &mut PagedSparseSet, entity: Entity, mut val: T) {
        // SAFETY: `&mut val` points to a valid `T`; caller ensures the set's
        // layout matches `T`. We `forget` to avoid double-drop since the set
        // now owns the bytes.
        unsafe { set.insert(entity, &mut val as *mut T as *const u8) };
        std::mem::forget(val);
    }

    /// Helper: read a typed value from a raw pointer.
    unsafe fn read_ptr<T: Copy>(ptr: *mut u8) -> T {
        // SAFETY: Caller guarantees `ptr` points to a valid, aligned `T`.
        unsafe { *(ptr as *const T) }
    }

    #[test]
    fn empty_set() {
        let set = make_set::<u32>();
        assert_eq!(set.len(), 0);
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
        let mut storage = SparseStorage::new(default_pool());
        let e = Entity::new(0, 0);
        storage.insert::<Marker>(0, e, Marker(42));
        assert_eq!(storage.get::<Marker>(0, e), Some(&Marker(42)));
    }

    #[test]
    fn sparse_storage_get_missing() {
        let storage = SparseStorage::new(default_pool());
        let e = Entity::new(0, 0);
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn sparse_storage_remove() {
        let mut storage = SparseStorage::new(default_pool());
        let e = Entity::new(0, 0);
        storage.insert::<Marker>(0, e, Marker(42));
        assert!(storage.remove_raw(0, e));
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn sparse_storage_component_ids_and_iter() {
        let mut storage = SparseStorage::new(default_pool());
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
        let mut storage = SparseStorage::new(default_pool());
        let e = Entity::new(0, 0);
        storage.insert::<u32>(0, e, 42u32);
        storage.insert::<Marker>(1, e, Marker(99));
        storage.remove_all(e);
        assert_eq!(storage.get::<u32>(0, e), None);
        assert_eq!(storage.get::<Marker>(1, e), None);
    }
}
