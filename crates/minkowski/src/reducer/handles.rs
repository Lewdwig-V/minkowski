use std::marker::PhantomData;

use crate::bundle::Bundle;
use crate::changeset::EnumChangeSet;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
use crate::world::World;

use super::{ComponentSet, Contains, ResolvedComponents};

/// Read-only single-entity handle for transactional reducers.
///
/// Provides [`get::<T>()`](EntityRef::get) to read a component, gated by
/// `C: Contains<T, IDX>` so only components in the declared set are accessible.
/// Created inside [`ReducerRegistry::register_entity`] closures. For read-write
/// access, see [`EntityMut`].
pub struct EntityRef<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    world: &'a World,
    _marker: PhantomData<C>,
}

impl<'a, C: ComponentSet> EntityRef<'a, C> {
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

/// Read-write single-entity handle for transactional reducers.
///
/// [`get::<T>()`](EntityMut::get) reads the live value from the archetype column.
/// [`set::<T>()`](EntityMut::set) buffers a write into the transaction's
/// [`EnumChangeSet`], applied atomically on commit. [`remove::<T>()`](EntityMut::remove)
/// buffers a component removal, and [`despawn()`](EntityMut::despawn) buffers entity
/// destruction (requires [`register_entity_despawn`](ReducerRegistry::register_entity_despawn)).
///
/// All operations are gated by [`Contains<T, IDX>`](Contains) so only components
/// in the declared set `C` are accessible. Holds `&mut EnumChangeSet` (not
/// `&mut Tx`) for clean borrow splitting inside transact closures. For
/// read-only access, see [`EntityRef`].
pub struct EntityMut<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    changeset: &'a mut EnumChangeSet,
    world: &'a World,
    can_despawn: bool,
    _marker: PhantomData<C>,
}

impl<'a, C: ComponentSet> EntityMut<'a, C> {
    pub(crate) fn new(
        entity: Entity,
        resolved: &'a ResolvedComponents,
        changeset: &'a mut EnumChangeSet,
        world: &'a World,
        can_despawn: bool,
    ) -> Self {
        Self {
            entity,
            resolved,
            changeset,
            world,
            can_despawn,
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

    /// Buffer a component removal. Bounded by the declared component set C.
    pub fn remove<T: Component, const IDX: usize>(&mut self)
    where
        C: Contains<T, IDX>,
    {
        let comp_id = self.resolved.0[IDX];
        self.changeset.record_remove(self.entity, comp_id);
    }

    /// Buffer an entity despawn. Panics if the reducer was not registered
    /// with `register_entity_despawn`.
    pub fn despawn(&mut self) {
        assert!(
            self.can_despawn,
            "despawn not declared (use register_entity_despawn)"
        );
        self.changeset.record_despawn(self.entity);
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }
}

/// Entity creation handle for transactional reducers.
///
/// Each call to [`spawn(bundle)`](Spawner::spawn) atomically reserves an entity
/// ID via lock-free `EntityAllocator::reserve` and buffers the bundle's
/// components into the transaction's [`EnumChangeSet`]. On successful commit the
/// entities are placed; on abort their IDs are reclaimed via the orphan queue.
///
/// Created inside [`ReducerRegistry::register_spawner`] closures.
pub struct Spawner<'a, B: Bundle> {
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    world: &'a World,
    _marker: PhantomData<B>,
}

impl<'a, B: Bundle> Spawner<'a, B> {
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

/// Read-only query iteration handle for scheduled reducers.
///
/// Uses the full [`World::query`] path with tick management and filter
/// support (including `Changed<T>`). The [`ReadOnlyWorldQuery`] bound
/// guarantees no `&mut T` access through the query. Provides
/// [`for_each`](QueryRef::for_each) and [`count`](QueryRef::count).
/// For read-write iteration, see [`QueryMut`].
///
/// Registered via [`ReducerRegistry::register_query_ref`], dispatched
/// via [`ReducerRegistry::run`].
pub struct QueryRef<'a, Q: ReadOnlyWorldQuery> {
    world: &'a mut World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: ReadOnlyWorldQuery + 'static> QueryRef<'a, Q> {
    pub(crate) fn new(world: &'a mut World) -> Self {
        Self {
            world,
            _marker: PhantomData,
        }
    }

    /// Iterate matching entities in contiguous typed slices per archetype.
    ///
    /// Iteration visits archetypes in creation order and rows within each
    /// archetype in insertion order. This is deterministic given identical
    /// world state but is not a stability guarantee.
    pub fn for_each(&mut self, f: impl FnMut(Q::Slice<'_>)) {
        self.world.query::<Q>().for_each_chunk(f);
    }

    pub fn count(&mut self) -> usize {
        self.world.query::<Q>().count()
    }
}

/// Read-write query iteration handle for scheduled reducers.
///
/// Same as [`QueryRef`] but allows `&mut T` in the query type, enabling
/// direct in-place mutation during iteration. Provides
/// [`for_each`](QueryMut::for_each) and [`count`](QueryMut::count).
///
/// Registered via [`ReducerRegistry::register_query`], dispatched
/// via [`ReducerRegistry::run`].
pub struct QueryMut<'a, Q: WorldQuery> {
    world: &'a mut World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: WorldQuery + 'static> QueryMut<'a, Q> {
    pub(crate) fn new(world: &'a mut World) -> Self {
        Self {
            world,
            _marker: PhantomData,
        }
    }

    /// Iterate matching entities in contiguous typed slices per archetype.
    ///
    /// Iteration visits archetypes in creation order and rows within each
    /// archetype in insertion order. This is deterministic given identical
    /// world state but is not a stability guarantee.
    pub fn for_each(&mut self, f: impl FnMut(Q::Slice<'_>)) {
        self.world.query::<Q>().for_each_chunk(f);
    }

    pub fn count(&mut self) -> usize {
        self.world.query::<Q>().count()
    }
}
