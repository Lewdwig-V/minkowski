use fixedbitset::FixedBitSet;
use std::marker::PhantomData;

use crate::component::{Component, ComponentRegistry};
use crate::entity::Entity;
use crate::storage::archetype::Archetype;

/// Send + Sync wrapper for raw pointers used in query fetches.
pub struct ThinSlicePtr<T> {
    pub(crate) ptr: *mut T,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for ThinSlicePtr<T> {}
unsafe impl<T: Sync> Sync for ThinSlicePtr<T> {}

impl<T> ThinSlicePtr<T> {
    /// # Safety
    /// `ptr` must be valid for reads/writes and properly aligned for `T`.
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }
}

/// # Safety
/// Implementors must guarantee that `init_fetch` returns valid state for the
/// archetype, and `fetch` returns valid items for any row < archetype.len().
pub unsafe trait WorldQuery {
    type Item<'w>;
    type Fetch<'w>: Send + Sync;
    type Slice<'w>;

    /// Returns a FixedBitSet with bits set for each required ComponentId.
    /// Archetypes missing any required component are skipped during query matching.
    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet;

    /// Initialize fetch state for the given archetype.
    fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Self::Fetch<'w>;

    /// Fetch the item at the given row.
    ///
    /// # Safety
    /// `row` must be less than `archetype.len()`, and the caller must ensure
    /// no aliasing violations occur.
    unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w>;

    /// Construct a typed slice over the entire column for this archetype.
    ///
    /// # Safety
    /// `len` must equal the archetype length. Caller must ensure no aliasing violations.
    unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> Self::Slice<'w>;

    /// Register all component types referenced by this query.
    /// Called by `Access::of` to ensure components have stable IDs before
    /// `required_ids` / `mutable_ids` are queried.
    fn register(_registry: &mut ComponentRegistry) {}

    /// Returns ComponentIds that this query accesses in any way (read or write),
    /// including optional components. Used by `Access` for conflict detection.
    ///
    /// Defaults to `required_ids` — correct for non-optional query terms.
    /// `Option<&T>` overrides to include T even though it's not required for
    /// archetype matching.
    fn accessed_ids(registry: &ComponentRegistry) -> FixedBitSet {
        Self::required_ids(registry)
    }

    /// Returns ComponentIds that this query accesses mutably.
    /// Used by change detection to mark columns as changed before iteration.
    fn mutable_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new()
    }

    /// Archetype-level filter. Returns false to skip this archetype entirely.
    /// Used by `Changed<T>` to skip archetypes whose column tick is stale.
    fn matches_filters(
        _archetype: &Archetype,
        _registry: &ComponentRegistry,
        _last_read_tick: crate::tick::Tick,
    ) -> bool {
        true
    }
}

/// Marker trait for query types that only read components.
///
/// Required by [`Tx::query`](crate::Tx::query) to prevent aliased `&mut T`
/// during concurrent read phases. Implemented for `&T`, `Entity`,
/// `Option<&T>`, `Changed<T>`, and tuples of read-only queries.
///
/// # Safety
/// Implementors must guarantee that `fetch` and `as_slice` never produce
/// mutable references. `&mut T` intentionally does NOT implement this trait.
pub unsafe trait ReadOnlyWorldQuery: WorldQuery {}

// Safety: &T produces &'w T — shared reference only.
unsafe impl<T: Component> ReadOnlyWorldQuery for &T {}

// Safety: Entity produces Entity (Copy) — no references at all.
unsafe impl ReadOnlyWorldQuery for Entity {}

// Safety: Option<&T> produces Option<&'w T> — shared reference only.
unsafe impl<T: Component> ReadOnlyWorldQuery for Option<&T> {}

// Safety: Changed<T> produces () — no references at all.
unsafe impl<T: Component> ReadOnlyWorldQuery for Changed<T> {}

// --- &T ---
unsafe impl<T: Component> WorldQuery for &T {
    type Item<'w> = &'w T;
    type Fetch<'w> = ThinSlicePtr<T>;
    type Slice<'w> = &'w [T];

    fn register(registry: &mut ComponentRegistry) {
        registry.register::<T>();
    }

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        let mut bits = FixedBitSet::new();
        let id = match registry.id::<T>() {
            Some(id) => id,
            // Component not yet registered — use a sentinel bit that no
            // archetype can contain, so the subset check correctly fails.
            None => registry.len(),
        };
        bits.grow(id + 1);
        bits.insert(id);
        bits
    }

    fn init_fetch(archetype: &Archetype, registry: &ComponentRegistry) -> ThinSlicePtr<T> {
        let id = registry.id::<T>().expect("component not registered");
        let col_idx = archetype.component_index[&id];
        unsafe { ThinSlicePtr::new(archetype.columns[col_idx].get_ptr(0) as *mut T) }
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w T {
        &*fetch.ptr.add(row)
    }

    unsafe fn as_slice<'w>(fetch: &ThinSlicePtr<T>, len: usize) -> &'w [T] {
        std::slice::from_raw_parts(fetch.ptr as *const T, len)
    }
}

// --- &mut T ---
unsafe impl<T: Component> WorldQuery for &mut T {
    type Item<'w> = &'w mut T;
    type Fetch<'w> = ThinSlicePtr<T>;
    type Slice<'w> = &'w mut [T];

    fn register(registry: &mut ComponentRegistry) {
        registry.register::<T>();
    }

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry)
    }

    fn mutable_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry)
    }

    fn init_fetch(archetype: &Archetype, registry: &ComponentRegistry) -> ThinSlicePtr<T> {
        <&T>::init_fetch(archetype, registry)
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w mut T {
        &mut *fetch.ptr.add(row)
    }

    unsafe fn as_slice<'w>(fetch: &ThinSlicePtr<T>, len: usize) -> &'w mut [T] {
        std::slice::from_raw_parts_mut(fetch.ptr, len)
    }
}

// --- Entity ---
unsafe impl WorldQuery for Entity {
    type Item<'w> = Entity;
    type Fetch<'w> = ThinSlicePtr<Entity>;
    type Slice<'w> = &'w [Entity];

    fn required_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new()
    }

    fn init_fetch(archetype: &Archetype, _registry: &ComponentRegistry) -> ThinSlicePtr<Entity> {
        unsafe { ThinSlicePtr::new(archetype.entities.as_ptr() as *mut Entity) }
    }

    unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        *fetch.ptr.add(row)
    }

    unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> &'w [Entity] {
        std::slice::from_raw_parts(fetch.ptr as *const Entity, len)
    }
}

// --- Option<&T> ---
unsafe impl<T: Component> WorldQuery for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type Fetch<'w> = Option<ThinSlicePtr<T>>;
    type Slice<'w> = Option<&'w [T]>;

    fn register(registry: &mut ComponentRegistry) {
        registry.register::<T>();
    }

    fn required_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new() // optional — does not filter archetypes
    }

    fn accessed_ids(registry: &ComponentRegistry) -> FixedBitSet {
        // Option<&T> reads T when the archetype contains it.
        // Not required for matching, but accessed for conflict detection.
        <&T>::required_ids(registry)
    }

    fn init_fetch(archetype: &Archetype, registry: &ComponentRegistry) -> Option<ThinSlicePtr<T>> {
        let id = registry.id::<T>()?;
        let col_idx = archetype.component_index.get(&id)?;
        Some(unsafe { ThinSlicePtr::new(archetype.columns[*col_idx].get_ptr(0) as *mut T) })
    }

    unsafe fn fetch<'w>(fetch: &Option<ThinSlicePtr<T>>, row: usize) -> Option<&'w T> {
        fetch.as_ref().map(|f| &*f.ptr.add(row))
    }

    unsafe fn as_slice<'w>(fetch: &Option<ThinSlicePtr<T>>, len: usize) -> Option<&'w [T]> {
        fetch
            .as_ref()
            .map(|f| std::slice::from_raw_parts(f.ptr as *const T, len))
    }
}

// --- Changed<T> ---

/// Query filter that skips archetypes unchanged since the last query evaluation.
///
/// "Changed" means "since the last time **this query** observed this column" —
/// it has no concept of frames or simulation time. The tick is per-query-type
/// (stored in the query cache) and advances automatically on each mutable access.
/// Marking is pessimistic: any mutable access path marks the column, even if
/// no bytes actually changed.
pub struct Changed<T: Component>(std::marker::PhantomData<T>);

unsafe impl<T: Component> WorldQuery for Changed<T> {
    type Item<'w> = ();
    type Fetch<'w> = ();
    type Slice<'w> = ();

    fn register(registry: &mut ComponentRegistry) {
        registry.register::<T>();
    }

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry)
    }

    fn init_fetch(_archetype: &Archetype, _registry: &ComponentRegistry) {}

    unsafe fn fetch<'w>(_fetch: &Self::Fetch<'w>, _row: usize) -> Self::Item<'w> {}

    unsafe fn as_slice<'w>(_fetch: &Self::Fetch<'w>, _len: usize) -> Self::Slice<'w> {}

    fn matches_filters(
        archetype: &Archetype,
        registry: &ComponentRegistry,
        last_read_tick: crate::tick::Tick,
    ) -> bool {
        let comp_id = match registry.id::<T>() {
            Some(id) => id,
            None => return false,
        };
        let col_idx = match archetype.component_index.get(&comp_id) {
            Some(&idx) => idx,
            None => return false,
        };
        archetype.columns[col_idx]
            .changed_tick
            .is_newer_than(last_read_tick)
    }
}

// --- Tuple impls ---
macro_rules! impl_world_query_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: WorldQuery),*> WorldQuery for ($($name,)*) {
            type Item<'w> = ($($name::Item<'w>,)*);
            type Fetch<'w> = ($($name::Fetch<'w>,)*);
            type Slice<'w> = ($($name::Slice<'w>,)*);

            fn register(registry: &mut ComponentRegistry) {
                $($name::register(registry);)*
            }

            fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
                let mut bits = FixedBitSet::new();
                $(
                    let sub = $name::required_ids(registry);
                    bits.grow(sub.len());
                    bits.union_with(&sub);
                )*
                bits
            }

            fn accessed_ids(registry: &ComponentRegistry) -> FixedBitSet {
                let mut bits = FixedBitSet::new();
                $(
                    let sub = $name::accessed_ids(registry);
                    bits.grow(sub.len());
                    bits.union_with(&sub);
                )*
                bits
            }

            fn mutable_ids(registry: &ComponentRegistry) -> FixedBitSet {
                let mut bits = FixedBitSet::new();
                $(
                    let sub = $name::mutable_ids(registry);
                    bits.grow(sub.len());
                    bits.union_with(&sub);
                )*
                bits
            }

            fn matches_filters(
                archetype: &Archetype,
                registry: &ComponentRegistry,
                last_read_tick: crate::tick::Tick,
            ) -> bool {
                $($name::matches_filters(archetype, registry, last_read_tick))&&*
            }

            fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Self::Fetch<'w> {
                ($($name::init_fetch(archetype, registry),)*)
            }

            unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WorldQuery>::fetch($name, row),)*)
            }

            unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WorldQuery>::as_slice($name, len),)*)
            }
        }
    };
}

macro_rules! impl_read_only_world_query_tuple {
    ($($name:ident),*) => {
        // Safety: all elements are ReadOnlyWorldQuery, so the tuple only produces shared refs.
        unsafe impl<$($name: ReadOnlyWorldQuery),*> ReadOnlyWorldQuery for ($($name,)*) {}
    };
}

impl_world_query_tuple!(A);
impl_world_query_tuple!(A, B);
impl_world_query_tuple!(A, B, C);
impl_world_query_tuple!(A, B, C, D);
impl_world_query_tuple!(A, B, C, D, E);
impl_world_query_tuple!(A, B, C, D, E, F);
impl_world_query_tuple!(A, B, C, D, E, F, G);
impl_world_query_tuple!(A, B, C, D, E, F, G, H);
impl_world_query_tuple!(A, B, C, D, E, F, G, H, I);
impl_world_query_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_world_query_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_world_query_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

impl_read_only_world_query_tuple!(A);
impl_read_only_world_query_tuple!(A, B);
impl_read_only_world_query_tuple!(A, B, C);
impl_read_only_world_query_tuple!(A, B, C, D);
impl_read_only_world_query_tuple!(A, B, C, D, E);
impl_read_only_world_query_tuple!(A, B, C, D, E, F);
impl_read_only_world_query_tuple!(A, B, C, D, E, F, G);
impl_read_only_world_query_tuple!(A, B, C, D, E, F, G, H);
impl_read_only_world_query_tuple!(A, B, C, D, E, F, G, H, I);
impl_read_only_world_query_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_read_only_world_query_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_read_only_world_query_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;
    use crate::entity::Entity;
    use crate::storage::archetype::{Archetype, ArchetypeId};

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }
    fn make_archetype_with_data(
        reg: &ComponentRegistry,
        ids: &[crate::component::ComponentId],
    ) -> Archetype {
        Archetype::new(ArchetypeId(0), ids, reg)
    }

    #[test]
    fn fetch_ref() {
        let mut reg = ComponentRegistry::new();
        let pos_id = reg.register::<Pos>();
        let mut arch = make_archetype_with_data(&reg, &[pos_id]);

        let mut pos = Pos { x: 1.0, y: 2.0 };
        unsafe {
            let col = arch.component_index[&pos_id];
            arch.columns[col].push(&mut pos as *mut Pos as *mut u8);
            let _ = pos;
        }
        arch.entities.push(Entity::new(0, 0));

        let fetch = <&Pos>::init_fetch(&arch, &reg);
        let item: &Pos = unsafe { <&Pos>::fetch(&fetch, 0) };
        assert_eq!(item, &Pos { x: 1.0, y: 2.0 });
    }

    #[test]
    fn fetch_mut() {
        let mut reg = ComponentRegistry::new();
        let pos_id = reg.register::<Pos>();
        let mut arch = make_archetype_with_data(&reg, &[pos_id]);

        let mut pos = Pos { x: 1.0, y: 2.0 };
        unsafe {
            let col = arch.component_index[&pos_id];
            arch.columns[col].push(&mut pos as *mut Pos as *mut u8);
            let _ = pos;
        }
        arch.entities.push(Entity::new(0, 0));

        let fetch = <&mut Pos>::init_fetch(&arch, &reg);
        unsafe {
            let item: &mut Pos = <&mut Pos>::fetch(&fetch, 0);
            item.x = 10.0;
        }
        unsafe {
            let ptr = arch.columns[0].get_ptr(0) as *const Pos;
            assert_eq!((*ptr).x, 10.0);
        }
    }

    #[test]
    fn fetch_entity() {
        let mut reg = ComponentRegistry::new();
        let pos_id = reg.register::<Pos>();
        let mut arch = make_archetype_with_data(&reg, &[pos_id]);

        let entity = Entity::new(42, 7);
        let mut pos = Pos { x: 0.0, y: 0.0 };
        unsafe {
            arch.columns[0].push(&mut pos as *mut Pos as *mut u8);
            let _ = pos;
        }
        arch.entities.push(entity);

        let fetch = Entity::init_fetch(&arch, &reg);
        let item = unsafe { Entity::fetch(&fetch, 0) };
        assert_eq!(item, entity);
    }

    #[test]
    fn required_ids_for_ref() {
        let mut reg = ComponentRegistry::new();
        reg.register::<Pos>();
        let bits = <&Pos>::required_ids(&reg);
        assert!(bits.contains(0));
    }

    #[test]
    fn required_ids_for_option_is_empty() {
        let mut reg = ComponentRegistry::new();
        reg.register::<Pos>();
        let bits = <Option<&Pos>>::required_ids(&reg);
        assert_eq!(bits.len(), 0);
    }
}
