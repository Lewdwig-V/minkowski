use std::any::Any;
use std::collections::HashMap;
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
            .expect("component missing on entity — archetype mismatch")
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
            .expect("component missing on entity — archetype mismatch")
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

/// Read-only query iteration. Hides `&World`.
pub struct QueryRef<'a, Q: ReadOnlyWorldQuery> {
    world: &'a World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: ReadOnlyWorldQuery + 'static> QueryRef<'a, Q> {
    #[allow(dead_code)]
    pub(crate) fn new(world: &'a World) -> Self {
        Self {
            world,
            _marker: PhantomData,
        }
    }

    pub fn for_each(&self, f: impl FnMut(Q::Item<'_>)) {
        let count = self.world.archetype_count();
        self.world.query_raw::<Q>(count).for_each(f);
    }

    pub fn count(&self) -> usize {
        let count = self.world.archetype_count();
        self.world.query_raw::<Q>(count).count()
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
pub struct ReducerId(pub usize);

/// Opaque identifier for a scheduled query reducer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QueryReducerId(usize);

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
        let access = C::access(&mut world.components, false);

        let adapter: EntityAdapter = Box::new(
            move |changeset, _allocated, world, resolved, entity, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .expect("reducer args type mismatch")
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
                    .expect("reducer args type mismatch")
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
                .expect("reducer args type mismatch")
                .clone();
            let qm = QueryMut::<Q>::new(world);
            f(qm, args);
        });

        let id = self.push_entry(name, access, resolved, ReducerKind::Scheduled(adapter));
        QueryReducerId(id.0)
    }

    /// Register a read-only query reducer: `f(QueryRef<Q>, args)`.
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
                .expect("reducer args type mismatch")
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

    /// Name-based lookup for network dispatch.
    pub fn id_by_name(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// Access metadata for a reducer, useful for scheduler integration.
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
        let qr: QueryRef<'_, (&Pos,)> = QueryRef::new(&world);
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
        let qr: QueryRef<'_, (&Pos,)> = QueryRef::new(&world);
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
            registry.register_query_ref::<(&Pos,), (), _>(&mut world, "count", |query, ()| {
                assert_eq!(query.count(), 2);
            });

        registry.run(&mut world, count_id, ());
    }

    #[test]
    fn id_by_name_lookup() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let _id =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});

        assert_eq!(registry.id_by_name("heal"), Some(0));
        assert_eq!(registry.id_by_name("nonexistent"), None);
    }

    #[test]
    fn access_metadata_matches() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let heal_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
        let health_id = world.components.id::<Health>().unwrap();
        let access = registry.access(heal_id.0);
        assert!(access.writes()[health_id]);
    }

    #[test]
    fn access_conflict_between_reducers() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let heal_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});

        let damage_id =
            registry.register_entity::<(Health,), (), _>(&mut world, "damage", |_entity, ()| {});

        let heal_access = registry.access(heal_id.0);
        let damage_access = registry.access(damage_id.0);
        assert!(
            heal_access.conflicts_with(damage_access),
            "two reducers writing Health should conflict"
        );
    }
}
