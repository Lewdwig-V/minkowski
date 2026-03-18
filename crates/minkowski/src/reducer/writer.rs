use std::alloc::Layout;
use std::marker::PhantomData;

use crate::sync::{Arc, AtomicBool, AtomicU64, Ordering};

use crate::changeset::{DropEntry, EnumChangeSet};
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::query::fetch::{Changed, ThinSlicePtr, WorldQuery};
use crate::storage::archetype::Archetype;
use crate::tick::Tick;
use crate::world::World;

// ── WritableRef ─────────────────────────────────────────────────────

/// Per-component buffered write handle. Reads the current value from the
/// archetype column; writes are buffered into an `EnumChangeSet` and only
/// applied on commit.
///
/// Uses a raw pointer to `EnumChangeSet` because multiple `WritableRef`s
/// in a tuple query need shared write access to the same changeset.
/// `PhantomData<&'a EnumChangeSet>` ties the lifetime without falsely
/// claiming exclusive access (multiple WritableRefs coexist).
pub struct WritableRef<'a, T: Component> {
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: *mut EnumChangeSet,
    row: usize,
    column_slot: usize,
    _marker: PhantomData<&'a EnumChangeSet>,
}

impl<'a, T: Component> WritableRef<'a, T> {
    pub(crate) fn new(
        entity: Entity,
        current: &'a T,
        comp_id: ComponentId,
        changeset: *mut EnumChangeSet,
        row: usize,
        column_slot: usize,
    ) -> Self {
        Self {
            entity,
            current,
            comp_id,
            changeset,
            row,
            column_slot,
            _marker: PhantomData,
        }
    }

    /// Read the current (pre-transaction) value.
    pub fn get(&self) -> &T {
        self.current
    }

    /// Buffer a write. The value is stored in the changeset's fast-lane
    /// archetype batch and applied on commit.
    #[inline]
    pub fn set(&mut self, value: T) {
        // Safety: the raw pointer is valid for the lifetime of the transaction.
        // Multiple WritableRefs in a tuple query share this pointer, but the
        // temporary `&mut EnumChangeSet` does not outlive this method call,
        // and `&mut self` prevents re-entrant access — no overlapping
        // mutable references.
        let cs = unsafe { &mut *self.changeset };
        let batch = cs
            .archetype_batches
            .last_mut()
            .expect("WritableRef::set called without an open archetype batch");
        let col_batch = &mut batch.columns[self.column_slot];
        debug_assert_eq!(col_batch.comp_id, self.comp_id);
        debug_assert_eq!(col_batch.layout, Layout::new::<T>());

        let value = std::mem::ManuallyDrop::new(value);
        let offset = cs
            .arena
            .alloc(&*value as *const T as *const u8, Layout::new::<T>());
        col_batch.entries.push(crate::changeset::BatchEntry {
            row: self.row,
            entity: self.entity,
            arena_offset: offset,
        });

        if std::mem::needs_drop::<T>() {
            cs.drop_entries.push(DropEntry {
                offset,
                drop_fn: crate::component::drop_ptr::<T>,
                mutation_idx: usize::MAX,
            });
        }
    }

    /// Clone the current value, apply `f`, and buffer the result.
    #[inline]
    pub fn modify(&mut self, f: impl FnOnce(&mut T))
    where
        T: Clone,
    {
        let mut val = self.current.clone();
        f(&mut val);
        self.set(val);
    }
}

// ── WriterQuery ─────────────────────────────────────────────────────

/// Maps a `WorldQuery` to a buffered-write variant. For `&T` this is a
/// passthrough; for `&mut T` it produces `WritableRef<T>` which reads
/// from the archetype column but writes into an `EnumChangeSet`.
///
/// # Safety
/// Implementors must guarantee that `init_writer_fetch` returns valid state
/// for the archetype, and `fetch_writer` returns valid items for any
/// row < archetype.len().
pub unsafe trait WriterQuery: WorldQuery {
    type WriterItem<'a>;
    type WriterFetch<'a>: Send + Sync;

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w>;

    /// Add an offset to the column slot index for fast-lane archetype batches.
    /// `&mut T` adds to its slot; tuples propagate to sub-elements.
    /// Other impls use the default no-op.
    fn set_column_slot(_fetch: &mut Self::WriterFetch<'_>, _offset: usize) {}

    /// # Safety
    /// `row` must be less than `archetype.len()`. `changeset` must be valid.
    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        entity: Entity,
        changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w>;
}

// --- &T: passthrough ---
// Safety: delegates to WorldQuery::fetch which produces &'w T.
unsafe impl<T: Component> WriterQuery for &T {
    type WriterItem<'a> = &'a T;
    type WriterFetch<'a> = ThinSlicePtr<T>;

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
        <&T as WorldQuery>::init_fetch(archetype, registry)
    }

    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        _entity: Entity,
        _changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        unsafe { <&T as WorldQuery>::fetch(fetch, row) }
    }
}

// --- &mut T: WritableRef ---
// Safety: reads from the column pointer (valid for archetype lifetime),
// writes are buffered into the changeset.
unsafe impl<T: Component> WriterQuery for &mut T {
    type WriterItem<'a> = WritableRef<'a, T>;
    type WriterFetch<'a> = (ThinSlicePtr<T>, ComponentId, usize);

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
        let id = registry.id::<T>().expect("component not registered");
        let ptr = <&T as WorldQuery>::init_fetch(archetype, registry);
        (ptr, id, 0) // column_slot set by tuple or defaults to 0 for single
    }

    fn set_column_slot(fetch: &mut Self::WriterFetch<'_>, offset: usize) {
        fetch.2 += offset;
    }

    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        entity: Entity,
        changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        unsafe {
            let (ptr, comp_id, column_slot) = fetch;
            let current: &T = &*ptr.ptr.add(row);
            WritableRef::new(entity, current, *comp_id, changeset, row, *column_slot)
        }
    }
}

// --- Entity: passthrough ---
// Safety: entity is Copy, no pointer dereference.
unsafe impl WriterQuery for Entity {
    type WriterItem<'a> = Entity;
    type WriterFetch<'a> = ();

    fn init_writer_fetch<'w>(
        _archetype: &'w Archetype,
        _registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
    }

    unsafe fn fetch_writer<'w>(
        _fetch: &Self::WriterFetch<'w>,
        _row: usize,
        entity: Entity,
        _changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        entity
    }
}

// --- Option<&T>: passthrough ---
// Safety: delegates to WorldQuery::fetch which produces Option<&'w T>.
unsafe impl<T: Component> WriterQuery for Option<&T> {
    type WriterItem<'a> = Option<&'a T>;
    type WriterFetch<'a> = Option<ThinSlicePtr<T>>;

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
        <Option<&T> as WorldQuery>::init_fetch(archetype, registry)
    }

    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        _entity: Entity,
        _changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        unsafe { <Option<&T> as WorldQuery>::fetch(fetch, row) }
    }
}

// --- Changed<T>: filter only ---
// Safety: produces (), no pointer dereference.
unsafe impl<T: Component> WriterQuery for Changed<T> {
    type WriterItem<'a> = ();
    type WriterFetch<'a> = ();

    fn init_writer_fetch<'w>(
        _archetype: &'w Archetype,
        _registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
    }

    unsafe fn fetch_writer<'w>(
        _fetch: &Self::WriterFetch<'w>,
        _row: usize,
        _entity: Entity,
        _changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
    }
}

// --- WriterQuery tuple impls ---
macro_rules! impl_writer_query_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        // Safety: delegates to each element's WriterQuery impl.
        unsafe impl<$($name: WriterQuery),*> WriterQuery for ($($name,)*) {
            type WriterItem<'a> = ($($name::WriterItem<'a>,)*);
            type WriterFetch<'a> = ($($name::WriterFetch<'a>,)*);

            fn init_writer_fetch<'w>(
                archetype: &'w Archetype,
                registry: &ComponentRegistry,
            ) -> Self::WriterFetch<'w> {
                let mut fetch = ($($name::init_writer_fetch(archetype, registry),)*);
                // Assign column_slot by finding each element's position in
                // ascending ComponentId order (matching open_archetype_batch's
                // ColumnBatch creation order via mutable_ids.ones()).
                let _mutable = <Self as WorldQuery>::mutable_ids(registry);
                let mut _assigned = 0usize;
                let ($($name,)*) = &mut fetch;
                $(
                    let sub_mutable = <$name as WorldQuery>::mutable_ids(registry);
                    if sub_mutable.count_ones(..) > 0 {
                        let first_id = sub_mutable.ones().next()
                            .expect("mutable_ids count_ones > 0 but ones() empty");
                        let slot = _mutable.ones().position(|id| id == first_id)
                            .expect("sub-element mutable ID not in tuple mutable_ids");
                        <$name as WriterQuery>::set_column_slot($name, slot);
                        _assigned += sub_mutable.count_ones(..);
                    }
                )*
                debug_assert_eq!(
                    _assigned, _mutable.count_ones(..),
                    "column_slot assignment out of sync with mutable_ids"
                );
                fetch
            }

            unsafe fn fetch_writer<'w>(
                fetch: &Self::WriterFetch<'w>,
                row: usize,
                entity: Entity,
                changeset: *mut EnumChangeSet,
            ) -> Self::WriterItem<'w> { unsafe {
                let ($($name,)*) = fetch;
                ($(<$name as WriterQuery>::fetch_writer($name, row, entity, changeset),)*)
            }}

            fn set_column_slot(fetch: &mut Self::WriterFetch<'_>, offset: usize) {
                let ($($name,)*) = fetch;
                $(<$name as WriterQuery>::set_column_slot($name, offset);)*
            }
        }
    };
}

impl_writer_query_tuple!(A);
impl_writer_query_tuple!(A, B);
impl_writer_query_tuple!(A, B, C);
impl_writer_query_tuple!(A, B, C, D);
impl_writer_query_tuple!(A, B, C, D, E);
impl_writer_query_tuple!(A, B, C, D, E, F);
impl_writer_query_tuple!(A, B, C, D, E, F, G);
impl_writer_query_tuple!(A, B, C, D, E, F, G, H);
impl_writer_query_tuple!(A, B, C, D, E, F, G, H, I);
impl_writer_query_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_writer_query_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_writer_query_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

// ── QueryWriter (transactional, buffered) ────────────────────────────

/// Transactional query iteration with buffered writes.
///
/// Iterates matching archetypes **without** marking columns as changed
/// (unlike [`World::query`]). `&T` items are read directly from archetype
/// columns; `&mut T` items become [`WritableRef<T>`] handles whose
/// [`set`](WritableRef::set)/[`modify`](WritableRef::modify) methods buffer
/// writes into an [`EnumChangeSet`] applied atomically on commit. This avoids
/// self-conflict with optimistic tick-based validation.
///
/// Compatible with `minkowski_persist::Durable` for WAL logging — the
/// motivating use case for buffered iteration.
///
/// Each `QueryWriter` reducer stores a per-reducer `last_read_tick` in an
/// `Arc<AtomicU64>` for `Changed<T>` filter support. Registered via
/// [`ReducerRegistry::register_query_writer`], dispatched via
/// [`ReducerRegistry::call`].
pub struct QueryWriter<'a, Q: WriterQuery> {
    world: &'a mut World,
    changeset: *mut EnumChangeSet,
    last_read_tick: &'a Arc<AtomicU64>,
    queried: &'a AtomicBool,
    _cs: PhantomData<&'a EnumChangeSet>,
    _query: PhantomData<Q>,
}

impl<'a, Q: WriterQuery + 'static> QueryWriter<'a, Q> {
    pub(crate) fn new(
        world: &'a mut World,
        changeset: *mut EnumChangeSet,
        last_read_tick: &'a Arc<AtomicU64>,
        queried: &'a AtomicBool,
    ) -> Self {
        Self {
            world,
            changeset,
            last_read_tick,
            queried,
            _cs: PhantomData,
            _query: PhantomData,
        }
    }

    /// Iterate all matching entities, yielding buffered writer items.
    ///
    /// `&T` components are read directly from archetype columns.
    /// `&mut T` components produce `WritableRef<T>` — reads from the column,
    /// writes buffer into the changeset.
    ///
    /// Iteration visits archetypes in creation order and rows within each
    /// archetype in insertion order. This is deterministic given identical
    /// world state but is not a stability guarantee.
    ///
    /// Advances the change detection tick: entities matched here will NOT
    /// be matched again on the next call unless their columns are modified.
    ///
    // PERF: Per-item iteration only — WritableRef indirection is inherent
    // to buffered writes. A slice API would imply contiguous-slice performance
    // characteristics that the changeset buffering cannot deliver.
    pub fn for_each(&mut self, mut f: impl FnMut(Q::WriterItem<'_>)) {
        self.queried.store(true, Ordering::Relaxed);
        let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));

        let required = Q::required_ids(&self.world.components);
        let mutable = Q::mutable_ids(&self.world.components);
        let cs_ptr = self.changeset;

        // Pre-allocate arena capacity based on matching entity count,
        // capped to avoid worst-case overallocation when only a fraction of
        // matched entities are actually written (conditional-update reducers).
        {
            const MAX_PREALLOC_MUTATIONS: usize = 64 * 1024;
            let mut entity_count = 0;
            for arch in &self.world.archetypes.archetypes {
                if !arch.is_empty()
                    && required.is_subset(&arch.component_ids)
                    && Q::matches_filters(arch, &self.world.components, last_tick)
                {
                    entity_count += arch.len();
                }
            }
            if entity_count > 0 {
                let cs = unsafe { &mut *cs_ptr };
                let mutable_count = mutable.count_ones(..);
                let mutations_needed = (entity_count * mutable_count).min(MAX_PREALLOC_MUTATIONS);
                cs.arena.reserve(mutations_needed * 64);
            }
        }

        for (arch_idx, arch) in self.world.archetypes.archetypes.iter().enumerate() {
            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                continue;
            }
            if !Q::matches_filters(arch, &self.world.components, last_tick) {
                continue;
            }

            // Open a fast-lane batch for this archetype
            let cs = unsafe { &mut *cs_ptr };
            crate::changeset::open_archetype_batch(
                cs,
                arch_idx,
                arch,
                &self.world.components,
                &mutable,
            );

            let fetch = Q::init_writer_fetch(arch, &self.world.components);
            for row in 0..arch.len() {
                let entity = arch.entities[row];
                let item = unsafe { Q::fetch_writer(&fetch, row, entity, cs_ptr) };
                f(item);
            }
        }

        // last_read_tick is updated by call() AFTER the changeset is applied,
        // only if this flag was set (i.e., for_each or count was actually called).
    }

    /// Count matching entities (respects `Changed<T>` filters).
    ///
    /// `last_read_tick` is updated by `call()` after the changeset is applied,
    /// so entities counted here will NOT be matched again unless their columns
    /// are modified externally.
    pub fn count(&mut self) -> usize {
        self.queried.store(true, Ordering::Relaxed);
        let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));

        let required = Q::required_ids(&self.world.components);
        let mut total = 0;
        for arch in &self.world.archetypes.archetypes {
            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                continue;
            }
            if !Q::matches_filters(arch, &self.world.components, last_tick) {
                continue;
            }
            total += arch.len();
        }

        // last_read_tick is updated by call() AFTER the changeset is applied,
        // only if this flag was set.
        total
    }
}
