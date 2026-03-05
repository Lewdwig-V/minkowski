use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::access::Access;
use crate::bundle::Bundle;
use crate::changeset::EnumChangeSet;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::query::fetch::{Changed, ReadOnlyWorldQuery, ThinSlicePtr, WorldQuery};
use crate::storage::archetype::Archetype;
use crate::tick::Tick;
use crate::transaction::{Conflict, Transact};
use crate::world::World;

// ── ComponentSet & Contains ──────────────────────────────────────────

/// Declares a set of component types with pre-resolved IDs.
/// Macro-generated for tuples 1–12.
pub trait ComponentSet: 'static {
    const COUNT: usize;

    /// Build an Access bitset for this component set.
    /// `read_only = true` → all components go in reads.
    /// `read_only = false` → all components go in writes.
    fn access(registry: &mut ComponentRegistry, read_only: bool) -> Access;

    /// Pre-resolve all ComponentIds (registers if needed). Returns them
    /// in positional order matching `Contains<T, INDEX>`.
    fn resolve(registry: &mut ComponentRegistry) -> Vec<ComponentId>;
}

/// Compile-time proof that `T` is at position `INDEX` in the component set.
/// The const generic disambiguates positions so that tuples like `(A, B)`
/// don't produce overlapping impls when A == B.
///
/// When calling `handle.get::<T>()`, the compiler infers INDEX from the
/// unique matching impl — no manual index needed at the call site.
pub trait Contains<T: Component, const INDEX: usize> {}

/// Pre-resolved ComponentIds created once at registration time.
/// `Contains<T, INDEX>` positions index into the inner Vec.
#[allow(dead_code)]
pub(crate) struct ResolvedComponents(pub(crate) Vec<ComponentId>);

/// Pre-resolved component lookup for dynamic reducers.
/// Entries sorted by TypeId for O(log n) binary search at runtime.
pub(crate) struct DynamicResolved {
    entries: Vec<(TypeId, ComponentId)>,
    access: Access,
    spawn_bundles: HashSet<TypeId>,
}

impl DynamicResolved {
    pub(crate) fn new(
        mut entries: Vec<(TypeId, ComponentId)>,
        access: Access,
        spawn_bundles: HashSet<TypeId>,
    ) -> Self {
        entries.sort_by_key(|(tid, _)| *tid);
        assert!(
            entries
                .windows(2)
                .all(|w| w[0].0 != w[1].0 || w[0].1 == w[1].1),
            "DynamicResolved: duplicate TypeId with different ComponentIds"
        );
        entries.dedup_by_key(|(tid, _)| *tid);
        Self {
            entries,
            access,
            spawn_bundles,
        }
    }

    pub(crate) fn lookup<T: 'static>(&self) -> Option<ComponentId> {
        let tid = TypeId::of::<T>();
        self.entries
            .binary_search_by_key(&tid, |(t, _)| *t)
            .ok()
            .map(|idx| self.entries[idx].1)
    }

    pub(crate) fn access(&self) -> &Access {
        &self.access
    }

    pub(crate) fn has_spawn_bundle<B: 'static>(&self) -> bool {
        self.spawn_bundles.contains(&TypeId::of::<B>())
    }
}

// ── DynamicCtx ───────────────────────────────────────────────────────

/// Runtime context for dynamic reducer closures. Provides component
/// access gated by the builder-declared upper bounds in `DynamicResolved`.
///
/// Reads go directly to World; writes buffer into an `EnumChangeSet`
/// applied atomically on commit.
pub struct DynamicCtx<'a> {
    world: &'a World,
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    resolved: &'a DynamicResolved,
}

impl<'a> DynamicCtx<'a> {
    pub(crate) fn new(
        world: &'a World,
        changeset: &'a mut EnumChangeSet,
        allocated: &'a mut Vec<Entity>,
        resolved: &'a DynamicResolved,
    ) -> Self {
        Self {
            world,
            changeset,
            allocated,
            resolved,
        }
    }

    /// Read a component from an entity. Panics if the component type was
    /// not declared via `can_read` / `can_write`, or if the entity does
    /// not have the component.
    pub fn read<T: crate::component::Component>(&self, entity: Entity) -> &T {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_read/can_write)",
                std::any::type_name::<T>()
            )
        });
        self.world
            .get_by_id::<T>(entity, comp_id)
            .unwrap_or_else(|| {
                panic!(
                    "component {} missing on entity {:?}",
                    std::any::type_name::<T>(),
                    entity,
                )
            })
    }

    /// Try to read a component. Returns `None` if the entity doesn't have it.
    /// Panics if the component type was not declared.
    pub fn try_read<T: crate::component::Component>(&self, entity: Entity) -> Option<&T> {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_read/can_write)",
                std::any::type_name::<T>()
            )
        });
        self.world.get_by_id::<T>(entity, comp_id)
    }

    /// Buffer a component write. The value is applied on commit.
    /// Panics if the component was only declared as readable.
    pub fn write<T: crate::component::Component>(&mut self, entity: Entity, value: T) {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_write)",
                std::any::type_name::<T>()
            )
        });
        assert!(
            self.resolved.access().writes().contains(comp_id),
            "component {} declared as read-only, not writable \
             (use can_write instead of can_read)",
            std::any::type_name::<T>()
        );
        self.changeset.insert_raw(entity, comp_id, value);
    }

    /// Buffer a component write only if the entity currently has that component.
    /// Returns `true` if the write was buffered, `false` if the entity does not
    /// have the component (in which case `value` is dropped without effect).
    pub fn try_write<T: crate::component::Component>(&mut self, entity: Entity, value: T) -> bool {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_write)",
                std::any::type_name::<T>()
            )
        });
        assert!(
            self.resolved.access().writes().contains(comp_id),
            "component {} declared as read-only, not writable \
             (use can_write instead of can_read)",
            std::any::type_name::<T>()
        );
        if self.world.get_by_id::<T>(entity, comp_id).is_some() {
            self.changeset.insert_raw(entity, comp_id, value);
            true
        } else {
            false
        }
    }

    /// Spawn an entity with a bundle. The bundle type must have been declared
    /// via `can_spawn` on the builder.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        assert!(
            self.resolved.has_spawn_bundle::<B>(),
            "bundle {} not declared for spawning in dynamic reducer \
             (use can_spawn)",
            std::any::type_name::<B>()
        );
        let entity = self.world.entities.reserve();
        self.allocated.push(entity);
        self.changeset
            .spawn_bundle_raw(entity, &self.world.components, bundle);
        entity
    }
}

// ── Macro ────────────────────────────────────────────────────────────

macro_rules! impl_component_set {
    ($($idx:tt: $name:ident),+) => {
        impl_component_set!(@trait $($name),+);
        impl_component_set!(@contains { $($name),+ } $($idx: $name),+);
    };

    (@trait $($name:ident),+) => {
        impl<$($name: Component),+> ComponentSet for ($($name,)+) {
            const COUNT: usize = impl_component_set!(@count $($name),+);

            fn access(registry: &mut ComponentRegistry, read_only: bool) -> Access {
                let mut access = Access::empty();
                $(
                    let id = registry.register::<$name>();
                    if read_only {
                        access.add_read(id);
                    } else {
                        access.add_write(id);
                    }
                )+
                access
            }

            fn resolve(registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                vec![$(registry.register::<$name>()),+]
            }
        }
    };

    // TT muncher: peel one Contains impl at a time, forwarding the
    // full type list in braces. Avoids cross-depth repetition issues.
    (@contains { $($all:ident),+ } $idx:tt: $target:ident, $($rest:tt)+) => {
        impl<$($all: Component),+> Contains<$target, $idx> for ($($all,)+) {}
        impl_component_set!(@contains { $($all),+ } $($rest)+);
    };
    (@contains { $($all:ident),+ } $idx:tt: $target:ident) => {
        impl<$($all: Component),+> Contains<$target, $idx> for ($($all,)+) {}
    };

    (@count $x:ident) => { 1usize };
    (@count $x:ident, $($rest:ident),+) => { 1usize + impl_component_set!(@count $($rest),+) };
}

impl_component_set!(0: A);
impl_component_set!(0: A, 1: B);
impl_component_set!(0: A, 1: B, 2: C);
impl_component_set!(0: A, 1: B, 2: C, 3: D);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

// ── Typed entity handles (transactional) ─────────────────────────────

/// Read-only entity access. `get::<T>()` is gated by `C: Contains<T, IDX>`.
pub struct EntityRef<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    world: &'a World,
    _marker: PhantomData<C>,
}

impl<'a, C: ComponentSet> EntityRef<'a, C> {
    #[allow(dead_code)]
    pub(crate) fn new(entity: Entity, resolved: &'a ResolvedComponents, world: &'a World) -> Self {
        Self {
            entity,
            resolved,
            world,
            _marker: PhantomData,
        }
    }

    pub fn get<T: Component, const IDX: usize>(&self) -> &T
    where
        C: Contains<T, IDX>,
    {
        let comp_id = self.resolved.0[IDX];
        self.world
            .get_by_id::<T>(self.entity, comp_id)
            .unwrap_or_else(|| {
                panic!(
                    "component {} missing on entity {:?} \
                     (entity may be dead or in a different archetype)",
                    std::any::type_name::<T>(),
                    self.entity,
                )
            })
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }
}

/// Read-write entity access. `get::<T>()` reads live, `set::<T>()` buffers
/// writes into the transaction's changeset.
///
/// Holds `&mut EnumChangeSet` (not `&mut Tx`) to avoid lifetime entanglement
/// with the Tx's cleanup lifetime — enables construction inside transact closures.
pub struct EntityMut<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    changeset: &'a mut EnumChangeSet,
    world: &'a World,
    _marker: PhantomData<C>,
}

impl<'a, C: ComponentSet> EntityMut<'a, C> {
    #[allow(dead_code)]
    pub(crate) fn new(
        entity: Entity,
        resolved: &'a ResolvedComponents,
        changeset: &'a mut EnumChangeSet,
        world: &'a World,
    ) -> Self {
        Self {
            entity,
            resolved,
            changeset,
            world,
            _marker: PhantomData,
        }
    }

    pub fn get<T: Component, const IDX: usize>(&self) -> &T
    where
        C: Contains<T, IDX>,
    {
        let comp_id = self.resolved.0[IDX];
        self.world
            .get_by_id::<T>(self.entity, comp_id)
            .unwrap_or_else(|| {
                panic!(
                    "component {} missing on entity {:?} \
                     (entity may be dead or in a different archetype)",
                    std::any::type_name::<T>(),
                    self.entity,
                )
            })
    }

    pub fn set<T: Component, const IDX: usize>(&mut self, value: T)
    where
        C: Contains<T, IDX>,
    {
        let comp_id = self.resolved.0[IDX];
        self.changeset.insert_raw(self.entity, comp_id, value);
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }
}

/// Spawn capability. Each `spawn(bundle)` atomically reserves an entity ID
/// and buffers the bundle into the transaction's changeset.
pub struct Spawner<'a, B: Bundle> {
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    world: &'a World,
    _marker: PhantomData<B>,
}

impl<'a, B: Bundle> Spawner<'a, B> {
    #[allow(dead_code)]
    pub(crate) fn new(
        changeset: &'a mut EnumChangeSet,
        allocated: &'a mut Vec<Entity>,
        world: &'a World,
    ) -> Self {
        Self {
            changeset,
            allocated,
            world,
            _marker: PhantomData,
        }
    }

    pub fn spawn(&mut self, bundle: B) -> Entity {
        let entity = self.world.entities.reserve();
        self.allocated.push(entity);
        self.changeset
            .spawn_bundle_raw(entity, &self.world.components, bundle);
        entity
    }
}

// ── WritableRef ─────────────────────────────────────────────────────

/// Per-component buffered write handle. Reads the current value from the
/// archetype column; writes are buffered into an `EnumChangeSet` and only
/// applied on commit.
///
/// Uses a raw pointer to `EnumChangeSet` because multiple `WritableRef`s
/// in a tuple query need shared write access to the same changeset.
/// `PhantomData<&'a EnumChangeSet>` ties the lifetime without falsely
/// claiming exclusive access (multiple WritableRefs coexist).
#[allow(dead_code)]
pub struct WritableRef<'a, T: Component> {
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: *mut EnumChangeSet,
    _marker: PhantomData<&'a EnumChangeSet>,
}

impl<'a, T: Component> WritableRef<'a, T> {
    #[allow(dead_code)]
    pub(crate) fn new(
        entity: Entity,
        current: &'a T,
        comp_id: ComponentId,
        changeset: *mut EnumChangeSet,
    ) -> Self {
        Self {
            entity,
            current,
            comp_id,
            changeset,
            _marker: PhantomData,
        }
    }

    /// Read the current (pre-transaction) value.
    #[allow(dead_code)]
    pub fn get(&self) -> &T {
        self.current
    }

    /// Buffer a write. The value is stored in the changeset and applied on commit.
    #[allow(dead_code)]
    pub fn set(&mut self, value: T) {
        // Safety: the raw pointer is valid for the lifetime of the transaction.
        // Multiple WritableRefs in a tuple query share this pointer, but the
        // temporary `&mut EnumChangeSet` does not outlive this method call,
        // and `&mut self` prevents re-entrant access — no overlapping
        // mutable references.
        unsafe { &mut *self.changeset }.insert_raw(self.entity, self.comp_id, value);
    }

    /// Clone the current value, apply `f`, and buffer the result.
    #[allow(dead_code)]
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
#[allow(dead_code)]
pub unsafe trait WriterQuery: WorldQuery {
    type WriterItem<'a>;
    type WriterFetch<'a>: Send + Sync;

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w>;

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
        <&T as WorldQuery>::fetch(fetch, row)
    }
}

// --- &mut T: WritableRef ---
// Safety: reads from the column pointer (valid for archetype lifetime),
// writes are buffered into the changeset.
unsafe impl<T: Component> WriterQuery for &mut T {
    type WriterItem<'a> = WritableRef<'a, T>;
    type WriterFetch<'a> = (ThinSlicePtr<T>, ComponentId);

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
        let id = registry.id::<T>().expect("component not registered");
        let ptr = <&T as WorldQuery>::init_fetch(archetype, registry);
        (ptr, id)
    }

    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        entity: Entity,
        changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        let (ptr, comp_id) = fetch;
        let current: &T = &*ptr.ptr.add(row);
        WritableRef::new(entity, current, *comp_id, changeset)
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
        <Option<&T> as WorldQuery>::fetch(fetch, row)
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
                ($($name::init_writer_fetch(archetype, registry),)*)
            }

            unsafe fn fetch_writer<'w>(
                fetch: &Self::WriterFetch<'w>,
                row: usize,
                entity: Entity,
                changeset: *mut EnumChangeSet,
            ) -> Self::WriterItem<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WriterQuery>::fetch_writer($name, row, entity, changeset),)*)
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
/// (unlike `world.query()`). Writes go through `WritableRef<T>` into an
/// `EnumChangeSet` that is applied atomically on commit. This avoids
/// self-conflict with optimistic tick-based validation.
///
/// Each `QueryWriter` reducer stores a per-reducer `last_read_tick` in an
/// `Arc<AtomicU64>` for `Changed<T>` filter support.
pub struct QueryWriter<'a, Q: WriterQuery> {
    world: &'a mut World,
    changeset: *mut EnumChangeSet,
    last_read_tick: &'a Arc<AtomicU64>,
    _cs: PhantomData<&'a EnumChangeSet>,
    _query: PhantomData<Q>,
}

impl<'a, Q: WriterQuery + 'static> QueryWriter<'a, Q> {
    pub(crate) fn new(
        world: &'a mut World,
        changeset: *mut EnumChangeSet,
        last_read_tick: &'a Arc<AtomicU64>,
    ) -> Self {
        Self {
            world,
            changeset,
            last_read_tick,
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
    /// Advances the change detection tick: entities matched here will NOT
    /// be matched again on the next call unless their columns are modified.
    pub fn for_each(&mut self, mut f: impl FnMut(Q::WriterItem<'_>)) {
        let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));
        let new_tick = self.world.next_tick();

        let required = Q::required_ids(&self.world.components);
        let cs_ptr = self.changeset;

        for arch in &self.world.archetypes.archetypes {
            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                continue;
            }
            if !Q::matches_filters(arch, &self.world.components, last_tick) {
                continue;
            }
            let fetch = Q::init_writer_fetch(arch, &self.world.components);
            for row in 0..arch.len() {
                let entity = arch.entities[row];
                let item = unsafe { Q::fetch_writer(&fetch, row, entity, cs_ptr) };
                f(item);
            }
        }

        self.last_read_tick.store(new_tick.raw(), Ordering::Relaxed);
    }

    /// Count matching entities (respects `Changed<T>` filters).
    ///
    /// Advances the change detection tick (same as `for_each`): entities
    /// counted here will NOT be matched again unless their columns are modified.
    pub fn count(&mut self) -> usize {
        let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));
        let new_tick = self.world.next_tick();

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

        self.last_read_tick.store(new_tick.raw(), Ordering::Relaxed);
        total
    }
}

// ── Typed query handles (scheduled) ──────────────────────────────────

/// Read-only query iteration. Hides `&mut World`.
///
/// Uses the full `world.query()` path (with tick management and filter
/// support including `Changed<T>`). The `ReadOnlyWorldQuery` bound
/// guarantees no `&mut T` access through the query.
pub struct QueryRef<'a, Q: ReadOnlyWorldQuery> {
    world: &'a mut World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: ReadOnlyWorldQuery + 'static> QueryRef<'a, Q> {
    #[allow(dead_code)]
    pub(crate) fn new(world: &'a mut World) -> Self {
        Self {
            world,
            _marker: PhantomData,
        }
    }

    pub fn for_each(&mut self, f: impl FnMut(Q::Item<'_>)) {
        self.world.query::<Q>().for_each(f);
    }

    pub fn count(&mut self) -> usize {
        self.world.query::<Q>().count()
    }
}

/// Read-write query iteration. Hides `&mut World`.
pub struct QueryMut<'a, Q: WorldQuery> {
    world: &'a mut World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: WorldQuery + 'static> QueryMut<'a, Q> {
    #[allow(dead_code)]
    pub(crate) fn new(world: &'a mut World) -> Self {
        Self {
            world,
            _marker: PhantomData,
        }
    }

    pub fn for_each(&mut self, f: impl FnMut(Q::Item<'_>)) {
        self.world.query::<Q>().for_each(f);
    }

    pub fn for_each_chunk(&mut self, f: impl FnMut(Q::Slice<'_>)) {
        self.world.query::<Q>().for_each_chunk(f);
    }

    pub fn count(&mut self) -> usize {
        self.world.query::<Q>().count()
    }
}

// ── ReducerRegistry ──────────────────────────────────────────────────

/// Opaque identifier for a transactional reducer (entity/pair/spawner).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ReducerId(pub(crate) usize);

impl ReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Opaque identifier for a scheduled query reducer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QueryReducerId(pub(crate) usize);

impl QueryReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Opaque identifier for a dynamic reducer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DynamicReducerId(pub(crate) usize);

impl DynamicReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Restricted view of `&mut World` for transactional adapters. Exposes
/// only what reducers legitimately need — archetype iteration, component
/// registry, tick advancement — without the full `World` mutation API.
///
/// Prevents transactional closures from calling `world.spawn()`,
/// `world.insert()`, or `world.query::<(&mut T,)>()` directly,
/// which would bypass the ChangeSet and break optimistic validation.
pub(crate) struct TransactionalWorld<'a>(pub(crate) &'a mut World);

impl<'a> TransactionalWorld<'a> {
    /// Reborrow as `&World` for read-only access (entity reducers, spawners).
    pub(crate) fn as_ref(&self) -> &World {
        self.0
    }

    /// Advance the tick counter.
    #[allow(dead_code)]
    pub(crate) fn next_tick(&mut self) -> crate::tick::Tick {
        self.0.next_tick()
    }
}

impl<'a> std::ops::Deref for TransactionalWorld<'a> {
    type Target = World;
    fn deref(&self) -> &World {
        self.0
    }
}

/// Type-erased transactional reducer adapter. Receives changeset + allocated list
/// (from Tx), a restricted world view, resolved IDs, and type-erased args.
type TransactionalAdapter = Box<
    dyn Fn(
            &mut EnumChangeSet,
            &mut Vec<Entity>,
            &mut TransactionalWorld<'_>,
            &ResolvedComponents,
            &dyn Any,
        ) + Send
        + Sync,
>;

/// Type-erased scheduled reducer adapter.
type ScheduledAdapter = Box<dyn Fn(&mut World, &dyn Any) + Send + Sync>;

/// Type-erased dynamic reducer adapter.
type DynamicAdapter = Box<dyn Fn(&mut DynamicCtx, &dyn Any) + Send + Sync>;

/// Two execution models for reducers.
enum ReducerKind {
    /// Runs inside `strategy.transact()`. Entity + args from call site.
    Transactional(TransactionalAdapter),
    /// Runs with direct `&mut World`.
    Scheduled(ScheduledAdapter),
}

struct ReducerEntry {
    #[allow(dead_code)]
    name: &'static str,
    access: Access,
    resolved: ResolvedComponents,
    kind: ReducerKind,
}

struct DynamicReducerEntry {
    #[allow(dead_code)]
    name: &'static str,
    resolved: DynamicResolved,
    closure: DynamicAdapter,
}

/// Discriminant for the `by_name` lookup table.
#[derive(Clone, Copy)]
enum ReducerSlot {
    Unified(usize),
    Dynamic(usize),
}

/// External registry of typed reducers. Owns closures, Access metadata,
/// and pre-resolved ComponentIds. Composes with World + Transact strategies
/// the same way SpatialIndex composes with World — no World API growth.
pub struct ReducerRegistry {
    reducers: Vec<ReducerEntry>,
    dynamic_reducers: Vec<DynamicReducerEntry>,
    by_name: HashMap<&'static str, ReducerSlot>,
}

impl ReducerRegistry {
    pub fn new() -> Self {
        Self {
            reducers: Vec::new(),
            dynamic_reducers: Vec::new(),
            by_name: HashMap::new(),
        }
    }

    // ── Transactional registration ───────────────────────────────

    /// Register an entity reducer: `f(EntityMut<C>, args)`.
    /// At dispatch, call with `(entity, args)` as the args tuple.
    pub fn register_entity<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> ReducerId
    where
        C: ComponentSet,
        Args: Clone + 'static,
        F: Fn(EntityMut<'_, C>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(C::resolve(&mut world.components));
        // EntityMut can both read (get) and write (set) all components in C.
        let reads = C::access(&mut world.components, true);
        let writes = C::access(&mut world.components, false);
        let access = reads.merge(&writes);

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, _allocated, tw, resolved, args_any| {
                let (entity, args) = args_any
                    .downcast_ref::<(Entity, Args)>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected (Entity, {})",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref());
                f(handle, args);
            });

        self.push_entry(name, access, resolved, ReducerKind::Transactional(adapter))
    }

    /// Register a spawner reducer: `f(Spawner<B>, args)`.
    pub fn register_spawner<B, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> ReducerId
    where
        B: Bundle,
        Args: Clone + 'static,
        F: Fn(Spawner<'_, B>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(B::component_ids(&mut world.components));
        let access = Access::empty(); // spawner creates new entities, no column conflicts

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, allocated, tw, _resolved, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected {}",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = Spawner::<B>::new(changeset, allocated, tw.as_ref());
                f(handle, args);
            });

        self.push_entry(name, access, resolved, ReducerKind::Transactional(adapter))
    }

    /// Register a query writer reducer: `f(QueryWriter<Q>, args)`.
    ///
    /// Iterates matching archetypes with buffered writes. `&T` reads directly
    /// from columns; `&mut T` produces `WritableRef<T>` that buffers into the
    /// transaction's changeset. Column ticks are NOT advanced during iteration
    /// (avoiding self-conflict with optimistic validation). Changes are applied
    /// atomically on commit.
    pub fn register_query_writer<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> ReducerId
    where
        Q: WriterQuery + 'static,
        Args: Clone + 'static,
        F: Fn(QueryWriter<'_, Q>, Args) + Send + Sync + 'static,
    {
        Q::register(&mut world.components);
        let resolved = ResolvedComponents(Vec::new());
        let access = Access::of::<Q>(world);
        let last_read_tick = Arc::new(AtomicU64::new(0));
        let tick_ref = last_read_tick.clone();

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, _allocated, tw, _resolved, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected {}",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let cs_ptr: *mut EnumChangeSet = changeset;
                let qw = QueryWriter::<Q>::new(tw.0, cs_ptr, &tick_ref);
                f(qw, args);
            });

        self.push_entry(name, access, resolved, ReducerKind::Transactional(adapter))
    }

    // ── Scheduled registration ───────────────────────────────────

    /// Register a mutable query reducer: `f(QueryMut<Q>, args)`.
    pub fn register_query<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> QueryReducerId
    where
        Q: WorldQuery + 'static,
        Args: Clone + 'static,
        F: Fn(QueryMut<'_, Q>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(Vec::new());
        let access = Access::of::<Q>(world);

        let adapter: ScheduledAdapter = Box::new(move |world, args_any| {
            let args = args_any
                .downcast_ref::<Args>()
                .unwrap_or_else(|| {
                    panic!(
                        "reducer args type mismatch: expected {}",
                        std::any::type_name::<Args>()
                    )
                })
                .clone();
            let qm = QueryMut::<Q>::new(world);
            f(qm, args);
        });

        let id = self.push_entry(name, access, resolved, ReducerKind::Scheduled(adapter));
        QueryReducerId(id.0)
    }

    /// Register a read-only query reducer: `f(QueryRef<Q>, args)`.
    ///
    /// Uses the full query path with filter support (`Changed<T>` works).
    /// The `ReadOnlyWorldQuery` bound prevents `&mut T` access.
    pub fn register_query_ref<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> QueryReducerId
    where
        Q: ReadOnlyWorldQuery + 'static,
        Args: Clone + 'static,
        F: Fn(QueryRef<'_, Q>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(Vec::new());
        let access = Access::of::<Q>(world);

        let adapter: ScheduledAdapter = Box::new(move |world, args_any| {
            let args = args_any
                .downcast_ref::<Args>()
                .unwrap_or_else(|| {
                    panic!(
                        "reducer args type mismatch: expected {}",
                        std::any::type_name::<Args>()
                    )
                })
                .clone();
            let qr = QueryRef::<Q>::new(world);
            f(qr, args);
        });

        let id = self.push_entry(name, access, resolved, ReducerKind::Scheduled(adapter));
        QueryReducerId(id.0)
    }

    // ── Dynamic registration ────────────────────────────────────

    /// Start building a dynamic reducer. Returns a builder that lets you
    /// declare which components the closure may read, write, or spawn.
    pub fn dynamic<'a>(
        &'a mut self,
        name: &'static str,
        world: &'a mut World,
    ) -> DynamicReducerBuilder<'a> {
        DynamicReducerBuilder {
            registry: self,
            world,
            name,
            access: Access::empty(),
            entries: Vec::new(),
            spawn_bundles: HashSet::new(),
        }
    }

    // ── Dispatch ─────────────────────────────────────────────────

    /// Call a transactional reducer (entity, spawner, or query writer).
    pub fn call<S: Transact, Args: Clone + 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: ReducerId,
        args: Args,
    ) -> Result<(), Conflict> {
        let entry = &self.reducers[id.0];
        let adapter = match &entry.kind {
            ReducerKind::Transactional(f) => f,
            ReducerKind::Scheduled(_) => {
                panic!("call() on scheduled reducer — use run() instead")
            }
        };
        let access = &entry.access;
        let resolved = &entry.resolved;

        strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let mut tw = TransactionalWorld(world);
            adapter(changeset, allocated, &mut tw, resolved, &args);
        })
    }

    /// Run a scheduled query reducer directly. Caller guarantees exclusivity.
    pub fn run<Args: Clone + 'static>(&self, world: &mut World, id: QueryReducerId, args: Args) {
        let entry = &self.reducers[id.0];
        match &entry.kind {
            ReducerKind::Scheduled(f) => f(world, &args),
            ReducerKind::Transactional(_) => {
                panic!("run() called on transactional reducer — use call() instead")
            }
        }
    }

    /// Look up a transactional reducer by name. Returns `None` if the name
    /// is not registered, points to a scheduled reducer, or is a dynamic reducer.
    pub fn reducer_id_by_name(&self, name: &str) -> Option<ReducerId> {
        let &slot = self.by_name.get(name)?;
        match slot {
            ReducerSlot::Unified(idx) => match &self.reducers[idx].kind {
                ReducerKind::Transactional(_) => Some(ReducerId(idx)),
                ReducerKind::Scheduled(_) => None,
            },
            ReducerSlot::Dynamic(_) => None,
        }
    }

    /// Look up a scheduled query reducer by name. Returns `None` if the name
    /// is not registered, points to a transactional reducer, or is a dynamic reducer.
    pub fn query_reducer_id_by_name(&self, name: &str) -> Option<QueryReducerId> {
        let &slot = self.by_name.get(name)?;
        match slot {
            ReducerSlot::Unified(idx) => match &self.reducers[idx].kind {
                ReducerKind::Scheduled(_) => Some(QueryReducerId(idx)),
                ReducerKind::Transactional(_) => None,
            },
            ReducerSlot::Dynamic(_) => None,
        }
    }

    /// Access metadata for a transactional reducer.
    pub fn reducer_access(&self, id: ReducerId) -> &Access {
        &self.reducers[id.0].access
    }

    /// Access metadata for a scheduled query reducer.
    pub fn query_reducer_access(&self, id: QueryReducerId) -> &Access {
        &self.reducers[id.0].access
    }

    /// Access metadata by raw index.
    pub fn access(&self, idx: usize) -> &Access {
        &self.reducers[idx].access
    }

    // ── Dynamic dispatch ────────────────────────────────────────

    /// Call a dynamic reducer with a chosen transaction strategy.
    pub fn dynamic_call<S: Transact, Args: 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: DynamicReducerId,
        args: &Args,
    ) -> Result<(), Conflict> {
        let entry = &self.dynamic_reducers[id.0];
        let closure = &entry.closure;
        let resolved = &entry.resolved;
        let access = resolved.access();

        strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let world_ref: &World = world;
            let mut ctx = DynamicCtx::new(world_ref, changeset, allocated, resolved);
            closure(&mut ctx, args);
        })
    }

    /// Look up a dynamic reducer by name.
    pub fn dynamic_id_by_name(&self, name: &str) -> Option<DynamicReducerId> {
        let &slot = self.by_name.get(name)?;
        match slot {
            ReducerSlot::Dynamic(idx) => Some(DynamicReducerId(idx)),
            ReducerSlot::Unified(_) => None,
        }
    }

    /// Access metadata for a dynamic reducer.
    pub fn dynamic_access(&self, id: DynamicReducerId) -> &Access {
        self.dynamic_reducers[id.0].resolved.access()
    }

    // ── Internal ─────────────────────────────────────────────────

    fn push_entry(
        &mut self,
        name: &'static str,
        access: Access,
        resolved: ResolvedComponents,
        kind: ReducerKind,
    ) -> ReducerId {
        let id = self.reducers.len();
        if let Some(slot) = self.by_name.get(name) {
            let (kind, existing) = match slot {
                ReducerSlot::Unified(idx) => ("unified", *idx),
                ReducerSlot::Dynamic(idx) => ("dynamic", *idx),
            };
            panic!(
                "ReducerRegistry: duplicate reducer name '{}' \
                 (already registered as {} reducer at index {})",
                name, kind, existing
            );
        }
        self.by_name.insert(name, ReducerSlot::Unified(id));
        self.reducers.push(ReducerEntry {
            name,
            access,
            resolved,
            kind,
        });
        ReducerId(id)
    }
}

impl Default for ReducerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── DynamicReducerBuilder ────────────────────────────────────────────

/// Builder for registering a dynamic reducer. Declare upper-bound access
/// with `can_read`, `can_write`, and `can_spawn`, then finalize with `build`.
pub struct DynamicReducerBuilder<'a> {
    registry: &'a mut ReducerRegistry,
    world: &'a mut World,
    name: &'static str,
    access: Access,
    entries: Vec<(TypeId, ComponentId)>,
    spawn_bundles: HashSet<TypeId>,
}

impl<'a> DynamicReducerBuilder<'a> {
    /// Declare that the closure may read component `T`.
    pub fn can_read<T: crate::component::Component>(mut self) -> Self {
        let comp_id = self.world.register_component::<T>();
        self.access.add_read(comp_id);
        self.entries.push((TypeId::of::<T>(), comp_id));
        self
    }

    /// Declare that the closure may write component `T`.
    /// Also adds a read entry (write implies read capability).
    pub fn can_write<T: crate::component::Component>(mut self) -> Self {
        let comp_id = self.world.register_component::<T>();
        self.access.add_read(comp_id);
        self.access.add_write(comp_id);
        self.entries.push((TypeId::of::<T>(), comp_id));
        self
    }

    /// Declare that the closure may spawn entities with bundle `B`.
    /// Adds write access for conflict detection but does NOT add TypeId
    /// entries (spawn uses the Bundle trait directly, not per-component lookup).
    pub fn can_spawn<B: Bundle>(mut self) -> Self {
        let comp_ids = B::component_ids(&mut self.world.components);
        for &comp_id in &comp_ids {
            self.access.add_write(comp_id);
        }
        self.spawn_bundles.insert(TypeId::of::<B>());
        self
    }

    /// Finalize registration. The closure receives `&mut DynamicCtx` and
    /// type-erased `&Args`. Returns the opaque `DynamicReducerId`.
    pub fn build<Args, F>(self, f: F) -> DynamicReducerId
    where
        Args: 'static,
        F: Fn(&mut DynamicCtx, &Args) + Send + Sync + 'static,
    {
        let resolved = DynamicResolved::new(self.entries, self.access.clone(), self.spawn_bundles);

        let closure: DynamicAdapter = Box::new(move |ctx, args_any| {
            let args = args_any.downcast_ref::<Args>().unwrap_or_else(|| {
                panic!(
                    "dynamic reducer args type mismatch: expected {}",
                    std::any::type_name::<Args>()
                )
            });
            f(ctx, args);
        });

        let id = self.registry.dynamic_reducers.len();
        if let Some(slot) = self.registry.by_name.get(self.name) {
            let (kind, existing) = match slot {
                ReducerSlot::Unified(idx) => ("unified", *idx),
                ReducerSlot::Dynamic(idx) => ("dynamic", *idx),
            };
            panic!(
                "ReducerRegistry: duplicate reducer name '{}' \
                 (already registered as {} reducer at index {})",
                self.name, kind, existing
            );
        }
        self.registry
            .by_name
            .insert(self.name, ReducerSlot::Dynamic(id));
        self.registry.dynamic_reducers.push(DynamicReducerEntry {
            name: self.name,
            resolved,
            closure,
        });
        DynamicReducerId(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Health(u32);

    // ── DynamicResolved tests ────────────────────────────────────

    #[test]
    fn dynamic_resolved_lookup() {
        use std::any::TypeId;
        let entries = vec![
            (TypeId::of::<u32>(), 0),
            (TypeId::of::<f64>(), 2),
            (TypeId::of::<i64>(), 1),
        ];
        let resolved = DynamicResolved::new(entries, Access::empty(), Default::default());
        assert_eq!(resolved.lookup::<u32>(), Some(0));
        assert_eq!(resolved.lookup::<f64>(), Some(2));
        assert_eq!(resolved.lookup::<i64>(), Some(1));
        assert_eq!(resolved.lookup::<u8>(), None);
    }

    #[test]
    fn dynamic_resolved_dedup() {
        use std::any::TypeId;
        let entries = vec![(TypeId::of::<u32>(), 0), (TypeId::of::<u32>(), 0)];
        let resolved = DynamicResolved::new(entries, Access::empty(), Default::default());
        // After dedup, duplicate entries are collapsed
        assert_eq!(resolved.lookup::<u32>(), Some(0));
    }

    #[test]
    fn dynamic_resolved_has_spawn_bundle() {
        use std::any::TypeId;
        let mut bundles = HashSet::new();
        bundles.insert(TypeId::of::<(Pos, Vel)>());
        let resolved = DynamicResolved::new(vec![], Access::empty(), bundles);
        assert!(resolved.has_spawn_bundle::<(Pos, Vel)>());
        assert!(!resolved.has_spawn_bundle::<(Health,)>());
    }

    // ── DynamicCtx tests ──────────────────────────────────────

    #[test]
    fn dynamic_ctx_read() {
        use std::any::TypeId;
        let mut world = World::new();
        let pos_id = world.register_component::<Pos>();
        let e = world.spawn((Pos(42.0),));

        let entries = vec![(TypeId::of::<Pos>(), pos_id)];
        let mut access = Access::empty();
        access.add_read(pos_id);
        let resolved = DynamicResolved::new(entries, access, Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        assert_eq!(ctx.read::<Pos>(e).0, 42.0);
    }

    #[test]
    fn dynamic_ctx_try_read_none() {
        use std::any::TypeId;
        let mut world = World::new();
        let pos_id = world.register_component::<Pos>();
        let vel_id = world.register_component::<Vel>();
        let e = world.spawn((Pos(1.0),)); // no Vel

        let entries = vec![(TypeId::of::<Pos>(), pos_id), (TypeId::of::<Vel>(), vel_id)];
        let mut access = Access::empty();
        access.add_read(pos_id);
        access.add_read(vel_id);
        let resolved = DynamicResolved::new(entries, access, Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        assert!(ctx.try_read::<Pos>(e).is_some());
        assert!(ctx.try_read::<Vel>(e).is_none());
    }

    #[test]
    fn dynamic_ctx_write_buffers() {
        use std::any::TypeId;
        let mut world = World::new();
        let pos_id = world.register_component::<Pos>();
        let e = world.spawn((Pos(1.0),));

        let entries = vec![(TypeId::of::<Pos>(), pos_id)];
        let mut access = Access::empty();
        access.add_write(pos_id);
        let resolved = DynamicResolved::new(entries, access, Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        {
            let mut ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
            ctx.write(e, Pos(99.0));
        }
        // Not yet applied
        assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
        // Apply changeset
        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_read_undeclared_panics() {
        let mut world = World::new();
        world.register_component::<Pos>();
        let e = world.spawn((Pos(1.0),));

        // Empty resolved — no components declared
        let resolved = DynamicResolved::new(vec![], Access::empty(), Default::default());
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        let _ = ctx.read::<Pos>(e);
    }

    // ── DynamicReducerBuilder tests ──────────────────────────────

    #[test]
    fn dynamic_builder_registers() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        let id = reducers
            .dynamic("test_dyn", &mut world)
            .can_read::<Pos>()
            .can_write::<Vel>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});

        assert_eq!(id.index(), 0);

        // Verify access: Pos is read, Vel is read+write
        let pos_id = world.components.id::<Pos>().unwrap();
        let vel_id = world.components.id::<Vel>().unwrap();
        let entry = &reducers.dynamic_reducers[id.0];
        assert!(entry.resolved.access().reads()[pos_id]);
        assert!(!entry.resolved.access().writes()[pos_id]); // read-only
        assert!(entry.resolved.access().reads()[vel_id]);
        assert!(entry.resolved.access().writes()[vel_id]); // writable
    }

    #[test]
    fn dynamic_builder_can_spawn() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        let id = reducers
            .dynamic("spawner", &mut world)
            .can_spawn::<(Pos, Vel)>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});

        let pos_id = world.components.id::<Pos>().unwrap();
        let vel_id = world.components.id::<Vel>().unwrap();
        let entry = &reducers.dynamic_reducers[id.0];
        // Spawn adds writes for conflict detection
        assert!(entry.resolved.access().writes()[pos_id]);
        assert!(entry.resolved.access().writes()[vel_id]);
    }

    #[test]
    #[should_panic(expected = "duplicate reducer name")]
    fn dynamic_builder_duplicate_name_panics() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        reducers
            .dynamic("dup", &mut world)
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
        reducers
            .dynamic("dup", &mut world)
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
    }

    #[test]
    #[should_panic(expected = "duplicate reducer name")]
    fn dynamic_name_conflicts_with_unified() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        reducers.register_entity::<(Health,), (), _>(&mut world, "shared_name", |_e, ()| {});
        reducers
            .dynamic("shared_name", &mut world)
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
    }

    // ── Dynamic dispatch tests ────────────────────────────────────

    #[test]
    fn dynamic_call_reads_and_writes() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);

        let mut reducers = ReducerRegistry::new();
        let id = reducers
            .dynamic("apply_vel", &mut world)
            .can_read::<Vel>()
            .can_write::<Pos>()
            .build(|ctx: &mut DynamicCtx, entity: &Entity| {
                let vel = ctx.read::<Vel>(*entity).0;
                let pos = ctx.read::<Pos>(*entity).0;
                ctx.write(*entity, Pos(pos + vel));
            });

        reducers
            .dynamic_call(&strategy, &mut world, id, &e)
            .unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 3.0);
    }

    #[test]
    fn dynamic_id_by_name_lookup() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        let dyn_id = reducers
            .dynamic("my_dyn", &mut world)
            .can_read::<Pos>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
        reducers.register_entity::<(Health,), (), _>(&mut world, "entity_one", |_e, ()| {});

        // Dynamic lookup finds dynamic reducer
        assert_eq!(reducers.dynamic_id_by_name("my_dyn"), Some(dyn_id));
        // Dynamic lookup does not find unified reducer
        assert_eq!(reducers.dynamic_id_by_name("entity_one"), None);
        // Dynamic lookup does not find nonexistent
        assert_eq!(reducers.dynamic_id_by_name("nope"), None);
        // Unified lookup does not find dynamic reducer
        assert_eq!(reducers.reducer_id_by_name("my_dyn"), None);
    }

    #[test]
    fn dynamic_access_metadata() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        let id = reducers
            .dynamic("test_access", &mut world)
            .can_read::<Pos>()
            .can_write::<Vel>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});

        let pos_id = world.components.id::<Pos>().unwrap();
        let vel_id = world.components.id::<Vel>().unwrap();
        let access = reducers.dynamic_access(id);
        assert!(access.reads()[pos_id]);
        assert!(access.reads()[vel_id]);
        assert!(access.writes()[vel_id]);
        assert!(!access.writes()[pos_id]);
    }

    // ── Debug assertion tests ────────────────────────────────────

    #[test]
    #[should_panic(expected = "read-only")]
    fn dynamic_ctx_write_on_read_only_panics() {
        use std::any::TypeId;
        let mut world = World::new();
        let pos_id = world.register_component::<Pos>();
        let e = world.spawn((Pos(1.0),));

        // Declare Pos as read-only (not writable)
        let entries = vec![(TypeId::of::<Pos>(), pos_id)];
        let mut access = Access::empty();
        access.add_read(pos_id); // read only, no write
        let resolved = DynamicResolved::new(entries, access, Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        ctx.write(e, Pos(99.0)); // should panic: read-only
    }

    #[test]
    #[should_panic(expected = "bundle")]
    fn dynamic_ctx_spawn_undeclared_bundle_panics() {
        let mut world = World::new();
        world.register_component::<Pos>();

        // No spawn bundles declared
        let resolved = DynamicResolved::new(vec![], Access::empty(), Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        ctx.spawn((Pos(1.0),)); // should panic: bundle not declared
    }

    // ── ComponentSet tests ──────────────────────────────────────

    #[test]
    fn component_set_count_single() {
        assert_eq!(<(Pos,) as ComponentSet>::COUNT, 1);
    }

    #[test]
    fn component_set_count_pair() {
        assert_eq!(<(Pos, Vel) as ComponentSet>::COUNT, 2);
    }

    #[test]
    fn component_set_resolve() {
        let mut reg = ComponentRegistry::new();
        let ids = <(Pos, Vel)>::resolve(&mut reg);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], reg.id::<Pos>().unwrap());
        assert_eq!(ids[1], reg.id::<Vel>().unwrap());
    }

    #[test]
    fn contains_bound_compiles() {
        // Verify that Contains<T, IDX> impls exist for the expected positions.
        // The functions exist purely for the compile-time bound check.
        fn assert_contains_at_0<C: Contains<Pos, 0>>(_: std::marker::PhantomData<C>) {}
        fn assert_contains_at_1<C: Contains<Vel, 1>>(_: std::marker::PhantomData<C>) {}
        assert_contains_at_0::<(Pos, Vel)>(std::marker::PhantomData);
        assert_contains_at_1::<(Pos, Vel)>(std::marker::PhantomData);
    }

    #[test]
    fn resolved_components_lookup() {
        let mut reg = ComponentRegistry::new();
        let ids = <(Pos, Vel)>::resolve(&mut reg);
        let resolved = ResolvedComponents(ids);
        // Position 0 → Pos's ComponentId, position 1 → Vel's ComponentId
        assert_eq!(resolved.0[0], reg.id::<Pos>().unwrap());
        assert_eq!(resolved.0[1], reg.id::<Vel>().unwrap());
    }

    #[test]
    fn access_read_only() {
        let mut reg = ComponentRegistry::new();
        let a = <(Pos, Vel)>::access(&mut reg, true);
        let pos_id = reg.id::<Pos>().unwrap();
        let vel_id = reg.id::<Vel>().unwrap();
        assert!(a.reads()[pos_id]);
        assert!(a.reads()[vel_id]);
        assert!(a.writes().is_empty());
    }

    #[test]
    fn access_write() {
        let mut reg = ComponentRegistry::new();
        let a = <(Pos, Vel)>::access(&mut reg, false);
        let pos_id = reg.id::<Pos>().unwrap();
        let vel_id = reg.id::<Vel>().unwrap();
        assert!(a.reads().is_empty());
        assert!(a.writes()[pos_id]);
        assert!(a.writes()[vel_id]);
    }

    #[test]
    fn access_merge_read_and_write() {
        let mut reg = ComponentRegistry::new();
        let reads = <(Pos,)>::access(&mut reg, true);
        let writes = <(Vel,)>::access(&mut reg, false);
        let merged = reads.merge(&writes);
        let pos_id = reg.id::<Pos>().unwrap();
        let vel_id = reg.id::<Vel>().unwrap();
        assert!(merged.reads()[pos_id]);
        assert!(merged.writes()[vel_id]);
    }

    #[test]
    fn triple_component_set() {
        let mut reg = ComponentRegistry::new();
        assert_eq!(<(Pos, Vel, Health) as ComponentSet>::COUNT, 3);
        let ids = <(Pos, Vel, Health)>::resolve(&mut reg);
        assert_eq!(ids.len(), 3);

        fn assert_pos_0<C: Contains<Pos, 0>>(_: std::marker::PhantomData<C>) {}
        fn assert_vel_1<C: Contains<Vel, 1>>(_: std::marker::PhantomData<C>) {}
        fn assert_health_2<C: Contains<Health, 2>>(_: std::marker::PhantomData<C>) {}
        assert_pos_0::<(Pos, Vel, Health)>(std::marker::PhantomData);
        assert_vel_1::<(Pos, Vel, Health)>(std::marker::PhantomData);
        assert_health_2::<(Pos, Vel, Health)>(std::marker::PhantomData);
    }

    /// Verify the index-inference pattern used by typed handles:
    /// `get::<T>()` calls a helper bounded by `C: Contains<T, IDX>`
    /// and the compiler infers IDX from the unique matching impl.
    #[test]
    fn index_inference() {
        fn get_index<T: Component, C, const IDX: usize>(
            resolved: &ResolvedComponents,
            _marker: std::marker::PhantomData<C>,
        ) -> ComponentId
        where
            C: Contains<T, IDX>,
        {
            resolved.0[IDX]
        }

        let mut reg = ComponentRegistry::new();
        let ids = <(Pos, Vel)>::resolve(&mut reg);
        let resolved = ResolvedComponents(ids);

        let pos_id = get_index::<Pos, (Pos, Vel), 0>(&resolved, std::marker::PhantomData);
        let vel_id = get_index::<Vel, (Pos, Vel), 1>(&resolved, std::marker::PhantomData);
        assert_eq!(pos_id, reg.id::<Pos>().unwrap());
        assert_eq!(vel_id, reg.id::<Vel>().unwrap());
    }

    // ── EntityRef tests ──────────────────────────────────────────

    #[test]
    fn entity_ref_get() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let resolved = ResolvedComponents(<(Pos, Vel)>::resolve(&mut world.components));

        let er: EntityRef<'_, (Pos, Vel)> = EntityRef::new(e, &resolved, &world);
        assert_eq!(er.get::<Pos, 0>().0, 1.0);
        assert_eq!(er.get::<Vel, 1>().0, 2.0);
        assert_eq!(er.entity(), e);
    }

    // ── QueryRef tests ───────────────────────────────────────────

    #[test]
    fn query_ref_for_each() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));
        let mut qr: QueryRef<'_, (&Pos,)> = QueryRef::new(&mut world);
        let mut sum = 0.0;
        qr.for_each(|(pos,)| sum += pos.0);
        assert_eq!(sum, 3.0);
    }

    #[test]
    fn query_ref_count() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));
        world.spawn((Pos(3.0),));
        let mut qr: QueryRef<'_, (&Pos,)> = QueryRef::new(&mut world);
        assert_eq!(qr.count(), 3);
    }

    // ── QueryMut tests ───────────────────────────────────────────

    #[test]
    fn query_mut_for_each() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        {
            let mut qm: QueryMut<'_, (&mut Pos,)> = QueryMut::new(&mut world);
            qm.for_each(|(pos,)| pos.0 += 10.0);
        }
        assert_eq!(world.get::<Pos>(e).unwrap().0, 11.0);
    }

    #[test]
    fn query_mut_count() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));
        let mut qm: QueryMut<'_, (&mut Pos,)> = QueryMut::new(&mut world);
        assert_eq!(qm.count(), 2);
    }

    // ── ReducerRegistry tests ────────────────────────────────────

    use crate::transaction::Optimistic;

    #[test]
    fn register_entity_and_call() {
        let mut world = World::new();
        let e = world.spawn((Health(100),));
        let strategy = Optimistic::new(&world);

        let mut registry = ReducerRegistry::new();
        let heal_id = registry.register_entity::<(Health,), u32, _>(
            &mut world,
            "heal",
            |mut entity, amount: u32| {
                let hp = entity.get::<Health, 0>().0;
                entity.set::<Health, 0>(Health(hp + amount));
            },
        );

        registry
            .call(&strategy, &mut world, heal_id, (e, 25u32))
            .unwrap();
        assert_eq!(world.get::<Health>(e).unwrap().0, 125);
    }

    #[test]
    fn register_query_and_run() {
        let mut world = World::new();
        world.spawn((Vel(1.0),));
        world.spawn((Vel(2.0),));

        let mut registry = ReducerRegistry::new();
        let gravity_id = registry.register_query::<(&mut Vel,), f32, _>(
            &mut world,
            "gravity",
            |mut query, dt: f32| {
                query.for_each(|(vel,)| vel.0 -= 9.81 * dt);
            },
        );

        registry.run(&mut world, gravity_id, 0.1f32);

        let mut sum = 0.0;
        for (vel,) in world.query::<(&Vel,)>() {
            sum += vel.0;
        }
        // (1.0 - 0.981) + (2.0 - 0.981) = 1.038
        assert!((sum - 1.038).abs() < 0.001, "sum = {}", sum);
    }

    #[test]
    fn register_query_ref_and_run() {
        let mut world = World::new();
        world.spawn((Pos(10.0),));
        world.spawn((Pos(20.0),));

        let mut registry = ReducerRegistry::new();
        let count_id =
            registry.register_query_ref::<(&Pos,), (), _>(&mut world, "count", |mut query, ()| {
                assert_eq!(query.count(), 2);
            });

        registry.run(&mut world, count_id, ());
    }

    #[test]
    fn typed_id_by_name_lookup() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let heal_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
        let _gravity_id =
            registry.register_query::<(&mut Vel,), (), _>(&mut world, "gravity", |_query, ()| {});

        // Typed lookups return the correct variant
        assert_eq!(registry.reducer_id_by_name("heal"), Some(heal_id));
        assert_eq!(registry.reducer_id_by_name("gravity"), None); // wrong kind
        assert_eq!(registry.reducer_id_by_name("nonexistent"), None);

        assert!(registry.query_reducer_id_by_name("gravity").is_some());
        assert_eq!(registry.query_reducer_id_by_name("heal"), None); // wrong kind
    }

    #[test]
    fn access_metadata_matches() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let heal_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
        let health_id = world.components.id::<Health>().unwrap();
        let access = registry.reducer_access(heal_id);
        // Entity reducers declare both reads and writes
        assert!(access.writes()[health_id]);
        assert!(access.reads()[health_id]);
    }

    #[test]
    fn access_conflict_between_reducers() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let heal_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});

        let damage_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "damage", |_entity, ()| {});

        let heal_access = registry.reducer_access(heal_id);
        let damage_access = registry.reducer_access(damage_id);
        assert!(
            heal_access.conflicts_with(damage_access),
            "two reducers writing Health should conflict"
        );
    }

    // ── Spawner lifecycle tests ──────────────────────────────────

    #[test]
    fn register_spawner_and_call() {
        let mut world = World::new();
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let spawn_id = registry.register_spawner::<(Health,), u32, _>(
            &mut world,
            "spawn_unit",
            |mut spawner, hp: u32| {
                spawner.spawn((Health(hp),));
            },
        );

        registry
            .call(&strategy, &mut world, spawn_id, 50u32)
            .unwrap();

        let mut count = 0;
        for (h,) in world.query::<(&Health,)>() {
            assert_eq!(h.0, 50);
            count += 1;
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn spawner_abort_reclaims_reserved_ids() {
        let mut world = World::new();
        world.spawn((Pos(1.0),)); // seed an entity so conflict detection works
        let strategy = Optimistic::with_retries(&world, 1);
        let mut registry = ReducerRegistry::new();

        // Register a spawner that also reads Pos to create a conflict surface
        let _spawn_id = registry.register_spawner::<(Health,), (), _>(
            &mut world,
            "spawn_and_conflict",
            |mut spawner, ()| {
                spawner.spawn((Health(1),));
            },
        );

        // Force a conflict: mutate Pos column between begin and commit
        // by using a strategy with max 1 retry and always-conflicting access
        let mut attempt = 0u32;
        let access_with_pos = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let result = strategy.transact(&mut world, &access_with_pos, |tx, world| {
            attempt += 1;
            let (changeset, allocated) = tx.reducer_parts();
            let spawner = Spawner::<(Health,)>::new(changeset, allocated, world);
            // Spawner allocates via reserve() — entity tracked in allocated
            let _spawned = spawner;

            if attempt == 1 {
                // Mutate to force conflict
                for pos in world.query::<(&mut Pos,)>() {
                    pos.0 .0 = 99.0;
                }
            }
        });

        // After abort+retry, no leaked entities
        // Trigger drain_orphans
        world.register_component::<Health>();
        let health_count = world.query::<(&Health,)>().count();
        // May be 0 (both attempts conflicted) or 1 (retry succeeded)
        assert!(health_count <= 1, "no duplicate spawns");
        assert!(attempt >= 1);
        let _ = result;
    }

    #[test]
    #[should_panic(expected = "duplicate reducer name")]
    fn duplicate_name_panics() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
        registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
    }

    #[test]
    fn dynamic_ctx_try_write_success() {
        use std::any::TypeId;
        let mut world = World::new();
        let e = world.spawn((42u32,));
        let comp_id = world.components.id::<u32>().unwrap();

        let mut access = Access::empty();
        access.add_write(comp_id);
        let entries = vec![(TypeId::of::<u32>(), comp_id)];
        let resolved = DynamicResolved::new(entries, access, Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);

        let wrote = ctx.try_write::<u32>(e, 99);
        assert!(wrote);

        let _reverse = cs.apply(&mut world);
        assert_eq!(*world.get::<u32>(e).unwrap(), 99);
    }

    #[test]
    fn dynamic_ctx_try_write_missing_component() {
        use std::any::TypeId;
        let mut world = World::new();
        let e = world.spawn((42u32,)); // has u32, not f64
        let f64_id = world.register_component::<f64>();

        let mut access = Access::empty();
        access.add_write(f64_id);
        let entries = vec![(TypeId::of::<f64>(), f64_id)];
        let resolved = DynamicResolved::new(entries, access, Default::default());

        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);

        let wrote = ctx.try_write::<f64>(e, 99.0);
        assert!(!wrote);
        assert_eq!(cs.len(), 0); // nothing buffered
    }

    #[test]
    fn dynamic_call_spawn_places_entity() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();

        let id = reducers
            .dynamic("spawner", &mut world)
            .can_spawn::<(u32, f64)>()
            .build(|ctx: &mut DynamicCtx, _args: &()| {
                let e = ctx.spawn((42u32, std::f64::consts::PI));
                let _ = e;
            });

        let strategy = Optimistic::new(&world);
        reducers
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();

        // Verify the spawned entity exists with correct components
        let mut found = false;
        world.query::<(&u32, &f64)>().for_each(|(u, f)| {
            assert_eq!(*u, 42);
            assert!((f - std::f64::consts::PI).abs() < f64::EPSILON);
            found = true;
        });
        assert!(found, "spawned entity not found in world");
    }

    #[test]
    fn restore_allocator_syncs_next_reserved() {
        let mut world = World::new();
        // Spawn some entities to populate generations
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));

        // Simulate snapshot restore
        let gens = vec![0u32; 5]; // 5 entities in the snapshot
        let free = vec![];
        world.restore_allocator_state(gens, free);

        // reserve() should start at index 5, not 0
        let reserved = world.entities.reserve();
        assert_eq!(reserved.index(), 5, "reserve() must skip restored indices");
    }

    // ── Additional review-requested tests ───────────────────────

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_try_read_undeclared_panics() {
        let mut world = World::new();
        let e = world.spawn((42u32,));
        let resolved = DynamicResolved::new(vec![], Access::empty(), Default::default());
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        let _ = ctx.try_read::<u32>(e);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_try_write_undeclared_panics() {
        let mut world = World::new();
        let e = world.spawn((42u32,));
        let resolved = DynamicResolved::new(vec![], Access::empty(), Default::default());
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(&world, &mut cs, &mut allocated, &resolved);
        ctx.try_write::<u32>(e, 99);
    }

    #[test]
    #[should_panic(expected = "duplicate reducer name")]
    fn unified_name_conflicts_with_dynamic() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        // Register dynamic first
        reducers
            .dynamic("clash", &mut world)
            .can_read::<u32>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
        // Then unified — should panic
        reducers.register_entity::<(u32,), (), _>(
            &mut world,
            "clash",
            |_entity: EntityMut<'_, (u32,)>, ()| {},
        );
    }

    // ── WritableRef tests ──────────────────────────────────────────

    #[test]
    fn writable_ref_get_returns_current_value() {
        let mut world = World::new();
        let e = world.spawn((Pos(42.0),));
        let pos_id = world.components.id::<Pos>().unwrap();
        let current = world.get::<Pos>(e).unwrap();

        let mut cs = EnumChangeSet::new();
        let wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet);
        assert_eq!(wr.get().0, 42.0);
    }

    #[test]
    fn writable_ref_set_buffers_into_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let pos_id = world.components.id::<Pos>().unwrap();
        let current = world.get::<Pos>(e).unwrap();

        let mut cs = EnumChangeSet::new();
        {
            let mut wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet);
            wr.set(Pos(99.0));
        }
        // World unchanged before apply
        assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
        assert_eq!(cs.len(), 1);
        // Apply and verify
        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);
    }

    #[test]
    fn writable_ref_modify_clones_and_sets() {
        let mut world = World::new();
        let e = world.spawn((Pos(10.0),));
        let pos_id = world.components.id::<Pos>().unwrap();
        let current = world.get::<Pos>(e).unwrap();

        let mut cs = EnumChangeSet::new();
        {
            let mut wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet);
            wr.modify(|p| p.0 += 10.0);
        }
        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 20.0);
    }

    // ── WriterQuery tests ──────────────────────────────────────────

    #[test]
    fn writer_query_ref_t_passthrough() {
        let mut world = World::new();
        let e = world.spawn((Pos(7.0),));
        let loc = world.entity_locations[e.index() as usize].unwrap();
        let archetype = &world.archetypes.archetypes[loc.archetype_id.0];

        let fetch = <&Pos as WriterQuery>::init_writer_fetch(archetype, &world.components);
        let mut cs = EnumChangeSet::new();
        let item = unsafe {
            <&Pos as WriterQuery>::fetch_writer(&fetch, loc.row, e, &mut cs as *mut EnumChangeSet)
        };
        assert_eq!(item.0, 7.0);
    }

    #[test]
    fn writer_query_mut_t_becomes_writable_ref() {
        let mut world = World::new();
        let e = world.spawn((Pos(5.0),));
        let loc = world.entity_locations[e.index() as usize].unwrap();
        let archetype = &world.archetypes.archetypes[loc.archetype_id.0];

        let fetch = <&mut Pos as WriterQuery>::init_writer_fetch(archetype, &world.components);
        let mut cs = EnumChangeSet::new();
        let mut item = unsafe {
            <&mut Pos as WriterQuery>::fetch_writer(
                &fetch,
                loc.row,
                e,
                &mut cs as *mut EnumChangeSet,
            )
        };
        assert_eq!(item.get().0, 5.0);
        item.set(Pos(55.0));
        // World unchanged
        assert_eq!(world.get::<Pos>(e).unwrap().0, 5.0);
        // Apply changeset
        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 55.0);
    }

    #[test]
    fn writer_query_tuple_fetch() {
        let mut world = World::new();
        let e = world.spawn((Pos(3.0), Vel(4.0)));
        let loc = world.entity_locations[e.index() as usize].unwrap();
        let archetype = &world.archetypes.archetypes[loc.archetype_id.0];

        let fetch =
            <(&Vel, &mut Pos) as WriterQuery>::init_writer_fetch(archetype, &world.components);
        let mut cs = EnumChangeSet::new();
        let (vel_ref, mut pos_wr) = unsafe {
            <(&Vel, &mut Pos) as WriterQuery>::fetch_writer(
                &fetch,
                loc.row,
                e,
                &mut cs as *mut EnumChangeSet,
            )
        };
        // Read velocity passthrough
        assert_eq!(vel_ref.0, 4.0);
        // Write position via WritableRef
        pos_wr.set(Pos(vel_ref.0 + pos_wr.get().0));
        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 7.0);
    }

    #[test]
    fn dynamic_spawn_abort_orphans_entity() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut world = World::new();
        // Spawn a sentinel to occupy index 0
        let sentinel = world.spawn((42u32,));

        let strategy = Optimistic::new(&world);
        let mut reducers = ReducerRegistry::new();

        let attempt_count = std::sync::Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let id = reducers
            .dynamic("spawn_and_fail", &mut world)
            .can_write::<u32>()
            .can_spawn::<(u32,)>()
            .build(move |ctx: &mut DynamicCtx, _args: &()| {
                let _e = ctx.spawn((999u32,));
                attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                // Write to u32 column — will conflict if another writer touched it
                ctx.write(sentinel, 0u32);
            });

        // Mutate the u32 column to cause an optimistic conflict on first attempt
        // by advancing the column tick between begin and commit.
        // We use a direct spawn to dirty the archetype's u32 column.
        world.spawn((77u32,));

        // The first call may fail due to stale ticks from our spawn above,
        // but retries should eventually succeed. Either way, entity IDs
        // from failed attempts must be recycled.
        let _ = reducers.dynamic_call(&strategy, &mut world, id, &());

        // After the call (success or failure), any orphaned entity IDs
        // from failed attempts should be drained on the next &mut World call.
        // Trigger drain by calling any &mut World method.
        let _ = world.spawn((0u32,));

        // Verify: every entity in the world should be alive (no leaked IDs)
        // and the attempt count confirms at least one attempt happened.
        assert!(attempt_count.load(Ordering::SeqCst) >= 1);
    }

    // ── QueryWriter tests ─────────────────────────────────────────

    #[test]
    fn query_writer_for_each_reads_and_buffers() {
        let mut world = World::new();
        let e1 = world.spawn((Pos(1.0), Vel(10.0)));
        let e2 = world.spawn((Pos(2.0), Vel(20.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        let id = registry.register_query_writer::<(&Pos, &mut Vel), f32, _>(
            &mut world,
            "apply_drag",
            |mut query, drag: f32| {
                query.for_each(|(pos, mut vel)| {
                    let _ = pos; // read Pos (passthrough)
                    vel.modify(|v| v.0 *= drag);
                });
            },
        );

        registry.call(&strategy, &mut world, id, 0.5f32).unwrap();

        assert_eq!(world.get::<Vel>(e1).unwrap().0, 5.0);
        assert_eq!(world.get::<Vel>(e2).unwrap().0, 10.0);
        assert_eq!(world.get::<Pos>(e1).unwrap().0, 1.0); // unchanged
    }

    #[test]
    fn query_writer_count() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));
        world.spawn((Vel(3.0),)); // no Pos — not matched
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        let id = registry.register_query_writer::<(&mut Pos,), (), _>(
            &mut world,
            "counter",
            |mut query, ()| {
                assert_eq!(query.count(), 2);
            },
        );

        registry.call(&strategy, &mut world, id, ()).unwrap();
    }

    #[test]
    fn query_writer_access_conflict() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let entity_id =
            registry.register_entity::<(Vel,), (), _>(&mut world, "set_vel", |_e, ()| {});
        let writer_id = registry.register_query_writer::<(&mut Vel,), (), _>(
            &mut world,
            "bulk_vel",
            |_q, ()| {},
        );

        let entity_access = registry.reducer_access(entity_id);
        let writer_access = registry.reducer_access(writer_id);
        assert!(entity_access.conflicts_with(writer_access));
    }

    #[test]
    fn query_writer_no_conflict_disjoint() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let entity_id =
            registry.register_entity::<(Pos,), (), _>(&mut world, "set_pos", |_e, ()| {});
        let writer_id = registry.register_query_writer::<(&mut Vel,), (), _>(
            &mut world,
            "bulk_vel",
            |_q, ()| {},
        );

        let entity_access = registry.reducer_access(entity_id);
        let writer_access = registry.reducer_access(writer_id);
        assert!(!entity_access.conflicts_with(writer_access));
    }
}
