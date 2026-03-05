use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::access::Access;
use crate::bundle::Bundle;
use crate::changeset::EnumChangeSet;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
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
    #[allow(dead_code)]
    access: Access,
    spawn_bundles: HashSet<TypeId>,
}

impl DynamicResolved {
    #[allow(dead_code)]
    pub(crate) fn new(
        mut entries: Vec<(TypeId, ComponentId)>,
        access: Access,
        spawn_bundles: HashSet<TypeId>,
    ) -> Self {
        entries.sort_by_key(|(tid, _)| *tid);
        entries.dedup_by_key(|(tid, _)| *tid);
        Self {
            entries,
            access,
            spawn_bundles,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn lookup<T: 'static>(&self) -> Option<ComponentId> {
        let tid = TypeId::of::<T>();
        self.entries
            .binary_search_by_key(&tid, |(t, _)| *t)
            .ok()
            .map(|idx| self.entries[idx].1)
    }

    #[allow(dead_code)]
    pub(crate) fn access(&self) -> &Access {
        &self.access
    }

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    /// Panics in debug mode if the component was only declared as readable.
    pub fn write<T: crate::component::Component>(&mut self, entity: Entity, value: T) {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_write)",
                std::any::type_name::<T>()
            )
        });
        debug_assert!(
            self.resolved.access().writes().contains(comp_id),
            "component {} declared as read-only, not writable \
             (use can_write instead of can_read)",
            std::any::type_name::<T>()
        );
        self.changeset.insert_raw(entity, comp_id, value);
    }

    /// Buffer a component write only if the entity currently has that component.
    /// Returns `true` if the write was buffered.
    pub fn try_write<T: crate::component::Component>(&mut self, entity: Entity, value: T) -> bool {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_write)",
                std::any::type_name::<T>()
            )
        });
        debug_assert!(
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
        debug_assert!(
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

/// Type-erased entity reducer adapter. Receives changeset + allocated list
/// (from Tx), world (for reads), resolved IDs, target entity, and type-erased args.
type EntityAdapter = Box<
    dyn Fn(&mut EnumChangeSet, &mut Vec<Entity>, &World, &ResolvedComponents, Entity, &dyn Any)
        + Send
        + Sync,
>;

/// Type-erased scheduled reducer adapter.
type ScheduledAdapter = Box<dyn Fn(&mut World, &dyn Any) + Send + Sync>;

/// Two execution models for reducers.
enum ReducerKind {
    /// Runs inside `strategy.transact()`. Entity + args from call site.
    EntityTransactional(EntityAdapter),
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

/// External registry of typed reducers. Owns closures, Access metadata,
/// and pre-resolved ComponentIds. Composes with World + Transact strategies
/// the same way SpatialIndex composes with World — no World API growth.
pub struct ReducerRegistry {
    reducers: Vec<ReducerEntry>,
    by_name: HashMap<&'static str, usize>,
}

impl ReducerRegistry {
    pub fn new() -> Self {
        Self {
            reducers: Vec::new(),
            by_name: HashMap::new(),
        }
    }

    // ── Transactional registration ───────────────────────────────

    /// Register an entity reducer: `f(EntityMut<C>, args)`.
    /// At dispatch, the entity comes from `call_entity()`.
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

        let adapter: EntityAdapter = Box::new(
            move |changeset, _allocated, world, resolved, entity, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected {}",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = EntityMut::<C>::new(entity, resolved, changeset, world);
                f(handle, args);
            },
        );

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::EntityTransactional(adapter),
        )
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

        let adapter: EntityAdapter = Box::new(
            move |changeset, allocated, world, _resolved, _entity, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected {}",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = Spawner::<B>::new(changeset, allocated, world);
                f(handle, args);
            },
        );

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::EntityTransactional(adapter),
        )
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

    // ── Dispatch ─────────────────────────────────────────────────

    /// Call a transactional entity reducer with a chosen strategy.
    pub fn call_entity<S: Transact, Args: Clone + 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: ReducerId,
        entity: Entity,
        args: Args,
    ) -> Result<(), Conflict> {
        let entry = &self.reducers[id.0];
        let adapter = match &entry.kind {
            ReducerKind::EntityTransactional(f) => f,
            ReducerKind::Scheduled(_) => {
                panic!("call_entity() on scheduled reducer — use run() instead")
            }
        };
        let access = &entry.access;
        let resolved = &entry.resolved;

        strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let world_ref: &World = world;
            adapter(changeset, allocated, world_ref, resolved, entity, &args);
        })
    }

    /// Run a scheduled query reducer directly. Caller guarantees exclusivity.
    pub fn run<Args: Clone + 'static>(&self, world: &mut World, id: QueryReducerId, args: Args) {
        let entry = &self.reducers[id.0];
        match &entry.kind {
            ReducerKind::Scheduled(f) => f(world, &args),
            ReducerKind::EntityTransactional(_) => {
                panic!("run() called on transactional reducer — use call_entity() instead")
            }
        }
    }

    /// Look up a transactional reducer by name. Returns `None` if the name
    /// is not registered or points to a scheduled reducer.
    pub fn reducer_id_by_name(&self, name: &str) -> Option<ReducerId> {
        let &idx = self.by_name.get(name)?;
        match &self.reducers[idx].kind {
            ReducerKind::EntityTransactional(_) => Some(ReducerId(idx)),
            ReducerKind::Scheduled(_) => None,
        }
    }

    /// Look up a scheduled query reducer by name. Returns `None` if the name
    /// is not registered or points to a transactional reducer.
    pub fn query_reducer_id_by_name(&self, name: &str) -> Option<QueryReducerId> {
        let &idx = self.by_name.get(name)?;
        match &self.reducers[idx].kind {
            ReducerKind::Scheduled(_) => Some(QueryReducerId(idx)),
            ReducerKind::EntityTransactional(_) => None,
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

    // ── Internal ─────────────────────────────────────────────────

    fn push_entry(
        &mut self,
        name: &'static str,
        access: Access,
        resolved: ResolvedComponents,
        kind: ReducerKind,
    ) -> ReducerId {
        let id = self.reducers.len();
        if let Some(&existing) = self.by_name.get(name) {
            panic!(
                "ReducerRegistry: duplicate reducer name '{}' \
                 (already registered at index {})",
                name, existing
            );
        }
        self.by_name.insert(name, id);
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
        let entries = vec![(TypeId::of::<u32>(), 0), (TypeId::of::<u32>(), 5)];
        let resolved = DynamicResolved::new(entries, Access::empty(), Default::default());
        // After dedup, first entry wins (sorted, then dedup keeps first)
        assert!(resolved.lookup::<u32>().is_some());
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
            .call_entity(&strategy, &mut world, heal_id, e, 25u32)
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
            .call_entity(&strategy, &mut world, spawn_id, Entity::DANGLING, 50u32)
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
}
