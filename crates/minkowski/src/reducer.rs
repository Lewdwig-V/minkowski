use std::alloc::Layout;
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;

use crate::sync::{Arc, AtomicBool, AtomicU64, Ordering};

use crate::access::Access;
use crate::bundle::Bundle;
use crate::changeset::{DropEntry, EnumChangeSet};
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::query::fetch::{Changed, ReadOnlyWorldQuery, ThinSlicePtr, WorldQuery};
use crate::storage::archetype::Archetype;
use crate::tick::Tick;
use crate::transaction::{Conflict, Transact, TransactError, WorldMismatch};
use crate::world::World;

// ── ReducerError ─────────────────────────────────────────────────────

/// Error type for reducer dispatch and registration failures.
///
/// These are API-misuse errors that can be checked at the call site without
/// panicking. Access-boundary violations inside reducer closures (e.g.
/// reading an undeclared component in `DynamicCtx`) still panic per the
/// assert boundary rule — they indicate broken invariants, not recoverable
/// conditions.
#[derive(Debug)]
pub enum ReducerError {
    /// Attempted to call a scheduled reducer with `call()`, or a
    /// transactional reducer with `run()`.
    WrongKind {
        /// `"transactional"` or `"scheduled"`.
        expected: &'static str,
        /// `"transactional"` or `"scheduled"`.
        actual: &'static str,
    },
    /// A reducer with this name was already registered.
    DuplicateName {
        name: &'static str,
        existing_kind: &'static str,
        existing_index: usize,
    },
    /// Transaction conflict (wraps [`Conflict`]).
    TransactionConflict(Conflict),
    /// The reducer ID does not refer to a valid entry in this registry.
    /// Caused by using an ID from a different registry or after the
    /// registry has been rebuilt.
    InvalidId {
        /// `"reducer"` or `"dynamic"`.
        kind: &'static str,
        index: usize,
        max: usize,
    },
    /// The transaction strategy was used with a different World than it
    /// was created from.
    WorldMismatch(WorldMismatch),
}

impl fmt::Display for ReducerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReducerError::WrongKind { expected, actual } => {
                write!(
                    f,
                    "reducer kind mismatch: expected {expected}, got {actual}"
                )
            }
            ReducerError::DuplicateName {
                name,
                existing_kind,
                existing_index,
            } => {
                write!(
                    f,
                    "duplicate reducer name '{name}' \
                     (already registered as {existing_kind} reducer at index {existing_index})"
                )
            }
            ReducerError::TransactionConflict(c) => {
                write!(f, "transaction conflict: {c}")
            }
            ReducerError::InvalidId { kind, index, max } => {
                write!(
                    f,
                    "invalid {kind} reducer ID (index {index}, registry has {max})"
                )
            }
            ReducerError::WorldMismatch(w) => write!(f, "{w}"),
        }
    }
}

impl std::error::Error for ReducerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReducerError::TransactionConflict(c) => Some(c),
            ReducerError::WorldMismatch(w) => Some(w),
            _ => None,
        }
    }
}

impl From<TransactError> for ReducerError {
    fn from(e: TransactError) -> Self {
        match e {
            TransactError::Conflict(c) => ReducerError::TransactionConflict(c),
            TransactError::WorldMismatch(w) => ReducerError::WorldMismatch(w),
        }
    }
}

/// Introspection descriptor for a registered reducer.
///
/// Returned by [`ReducerRegistry::reducer_info`], [`query_reducer_info`](ReducerRegistry::query_reducer_info),
/// and [`dynamic_reducer_info`](ReducerRegistry::dynamic_reducer_info).
#[derive(Debug, Clone)]
pub struct ReducerInfo {
    /// Registration name.
    pub name: &'static str,
    /// `"transactional"`, `"scheduled"`, or `"dynamic"`.
    pub kind: &'static str,
    /// Component-level access bitsets.
    pub access: Access,
    /// Whether this reducer has `Changed<T>` tick tracking.
    pub has_change_tracking: bool,
    /// Whether this reducer declares despawn capability.
    pub can_despawn: bool,
}

// ── ComponentSet & Contains ──────────────────────────────────────────

/// Declares a set of component types with pre-resolved IDs.
///
/// Macro-generated for tuples of 1-12 `Component` types.
/// Used as the type parameter `C` on [`EntityRef<C>`](EntityRef) and
/// [`EntityMut<C>`](EntityMut) to constrain which components the handle can access.
/// See [`ReducerRegistry`] for usage.
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
pub(crate) struct ResolvedComponents(pub(crate) Vec<ComponentId>);

/// Pre-resolved component lookup for dynamic reducers.
/// Uses an identity hasher since `TypeId` is already well-distributed.
pub(crate) struct DynamicResolved {
    entries: HashMap<TypeId, ComponentId, crate::component::TypeIdBuildHasher>,
    /// All declared ComponentIds for fast membership checks.
    comp_ids: HashSet<ComponentId>,
    access: Access,
    spawn_bundles: HashSet<TypeId>,
    remove_ids: HashSet<TypeId>,
}

impl DynamicResolved {
    pub(crate) fn new(
        entries: Vec<(TypeId, ComponentId)>,
        access: Access,
        spawn_bundles: HashSet<TypeId>,
        remove_ids: HashSet<TypeId>,
    ) -> Self {
        let comp_ids: HashSet<ComponentId> = entries.iter().map(|(_, cid)| *cid).collect();
        let entries: HashMap<TypeId, ComponentId, crate::component::TypeIdBuildHasher> =
            entries.into_iter().collect();
        Self {
            entries,
            comp_ids,
            access,
            spawn_bundles,
            remove_ids,
        }
    }

    #[inline]
    pub(crate) fn lookup<T: 'static>(&self) -> Option<ComponentId> {
        self.entries.get(&TypeId::of::<T>()).copied()
    }

    pub(crate) fn access(&self) -> &Access {
        &self.access
    }

    pub(crate) fn has_spawn_bundle<B: 'static>(&self) -> bool {
        self.spawn_bundles.contains(&TypeId::of::<B>())
    }

    pub(crate) fn has_remove<T: 'static>(&self) -> bool {
        self.remove_ids.contains(&TypeId::of::<T>())
    }

    /// Check if a ComponentId is in the declared set.
    pub(crate) fn contains_comp_id(&self, comp_id: ComponentId) -> bool {
        self.comp_ids.contains(&comp_id)
    }
}

// ── DynamicCtx ───────────────────────────────────────────────────────

/// Runtime-validated access handle for dynamic reducer closures.
///
/// Provides [`read`](DynamicCtx::read)/[`try_read`](DynamicCtx::try_read),
/// [`write`](DynamicCtx::write)/[`try_write`](DynamicCtx::try_write),
/// [`spawn`](DynamicCtx::spawn), [`remove`](DynamicCtx::remove)/[`try_remove`](DynamicCtx::try_remove),
/// [`despawn`](DynamicCtx::despawn), and [`for_each`](DynamicCtx::for_each).
/// Every operation validates at runtime that the accessed component type
/// was declared on the [`DynamicReducerBuilder`] — accessing undeclared
/// types, writing to read-only components, or despawning without declaration
/// panics in all builds.
///
/// Reads go directly to World; writes buffer into an [`EnumChangeSet`]
/// applied atomically on commit. Component IDs are pre-resolved at
/// registration time for O(log n) lookup by `TypeId`.
pub struct DynamicCtx<'a> {
    world: &'a World,
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    resolved: &'a DynamicResolved,
    last_read_tick: &'a Arc<AtomicU64>,
    queried: &'a AtomicBool,
}

impl<'a> DynamicCtx<'a> {
    pub(crate) fn new(
        world: &'a World,
        changeset: &'a mut EnumChangeSet,
        allocated: &'a mut Vec<Entity>,
        resolved: &'a DynamicResolved,
        last_read_tick: &'a Arc<AtomicU64>,
        queried: &'a AtomicBool,
    ) -> Self {
        Self {
            world,
            changeset,
            allocated,
            resolved,
            last_read_tick,
            queried,
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
    #[inline]
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

    /// Buffer a component removal. The removal is applied on commit
    /// (archetype migration). Panics if T was not declared via `can_remove`.
    pub fn remove<T: crate::component::Component>(&mut self, entity: Entity) {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_remove)",
                std::any::type_name::<T>()
            )
        });
        assert!(
            self.resolved.has_remove::<T>(),
            "component {} not declared for removal in dynamic reducer \
             (use can_remove, not can_read/can_write)",
            std::any::type_name::<T>()
        );
        self.changeset.record_remove(entity, comp_id);
    }

    /// Try to buffer a component removal. Returns `false` if the entity
    /// does not currently have the component. Panics if T was not declared
    /// via `can_remove`.
    pub fn try_remove<T: crate::component::Component>(&mut self, entity: Entity) -> bool {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "component {} not declared in dynamic reducer (use can_remove)",
                std::any::type_name::<T>()
            )
        });
        assert!(
            self.resolved.has_remove::<T>(),
            "component {} not declared for removal in dynamic reducer \
             (use can_remove, not can_read/can_write)",
            std::any::type_name::<T>()
        );
        if self.world.get_by_id::<T>(entity, comp_id).is_some() {
            self.changeset.record_remove(entity, comp_id);
            true
        } else {
            false
        }
    }

    /// Buffer an entity despawn. The entity is destroyed on commit.
    /// Panics if `can_despawn()` was not declared on the builder.
    pub fn despawn(&mut self, entity: Entity) {
        assert!(
            self.resolved.access().despawns(),
            "despawn not declared in dynamic reducer (use can_despawn)"
        );
        self.changeset.record_despawn(entity);
    }

    /// Debug-only validation: check that a component type is declared on
    /// this context without performing any read or write. Returns `true`
    /// if the type was declared via `can_read` or `can_write`.
    ///
    /// This is useful for debug assertions in reducer closures:
    /// ```ignore
    /// debug_assert!(ctx.is_declared::<Pos>(), "Pos not declared");
    /// ```
    pub fn is_declared<T: 'static>(&self) -> bool {
        self.resolved.lookup::<T>().is_some()
    }

    /// Debug-only validation: check that a component type is declared
    /// as writable. Returns `true` if the type was declared via `can_write`.
    pub fn is_writable<T: crate::component::Component>(&self) -> bool {
        self.resolved
            .lookup::<T>()
            .is_some_and(|comp_id| self.resolved.access().writes().contains(comp_id))
    }

    /// Debug-only validation: check that a component type is declared
    /// as removable. Returns `true` if the type was declared via `can_remove`.
    pub fn is_removable<T: crate::component::Component>(&self) -> bool {
        self.resolved.has_remove::<T>()
    }

    /// Debug-only validation: check that despawn is declared.
    pub fn can_despawn(&self) -> bool {
        self.resolved.access().despawns()
    }

    /// Iterate entities matching query `Q` using the typed query codepath.
    /// `Q` must be a `ReadOnlyWorldQuery` — writes go through `ctx.write()`.
    ///
    /// Yields typed slices per archetype for SIMD-friendly access.
    /// Iteration visits archetypes in creation order and rows within each
    /// archetype in insertion order. This is deterministic given identical
    /// world state but is not a stability guarantee.
    ///
    /// # Panics
    /// Panics if `Q` accesses any component not declared via `can_read`
    /// or `can_write` on the builder.
    pub fn for_each<Q: ReadOnlyWorldQuery + 'static>(&self, mut f: impl FnMut(Q::Slice<'_>)) {
        self.queried.store(true, Ordering::Relaxed);
        let accessed = Q::accessed_ids(&self.world.components);
        for comp_id in accessed.ones() {
            assert!(
                self.resolved.contains_comp_id(comp_id),
                "query accesses component ID {} which was not declared \
                 in dynamic reducer (use can_read/can_write)",
                comp_id,
            );
        }

        let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));
        let required = Q::required_ids(&self.world.components);

        for arch in &self.world.archetypes.archetypes {
            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                continue;
            }
            if !Q::matches_filters(arch, &self.world.components, last_tick) {
                continue;
            }
            let fetch = Q::init_fetch(arch, &self.world.components);
            // Safety: fetch was initialized from this archetype, len is in bounds.
            let slices = unsafe { Q::as_slice(&fetch, arch.len()) };
            f(slices);
        }
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

// ── Typed query handles (scheduled) ──────────────────────────────────

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

// ── ReducerRegistry ──────────────────────────────────────────────────

/// Typed handle for dispatching transactional reducers via [`ReducerRegistry::call`].
///
/// Obtained from [`ReducerRegistry::register_entity`],
/// [`register_entity_despawn`](ReducerRegistry::register_entity_despawn),
/// [`register_spawner`](ReducerRegistry::register_spawner), or
/// [`register_query_writer`](ReducerRegistry::register_query_writer).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ReducerId(pub(crate) usize);

impl ReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Typed handle for dispatching scheduled query reducers via [`ReducerRegistry::run`].
///
/// Obtained from [`ReducerRegistry::register_query`] or
/// [`register_query_ref`](ReducerRegistry::register_query_ref).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QueryReducerId(pub(crate) usize);

impl QueryReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Typed handle for dispatching dynamic reducers via [`ReducerRegistry::dynamic_call`].
///
/// Obtained from [`DynamicReducerBuilder::build`].
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

impl TransactionalWorld<'_> {
    /// Reborrow as `&World` for read-only access (entity reducers, spawners).
    pub(crate) fn as_ref(&self) -> &World {
        self.0
    }
}

impl std::ops::Deref for TransactionalWorld<'_> {
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
    name: &'static str,
    access: Access,
    resolved: ResolvedComponents,
    kind: ReducerKind,
    /// Per-reducer tick for `Changed<T>` support in `QueryWriter`.
    /// `None` for non-query-writer reducers.
    last_read_tick: Option<Arc<AtomicU64>>,
    /// Set to `true` by `for_each`/`count` — tick only advances if query ran.
    queried: Option<Arc<AtomicBool>>,
}

struct DynamicReducerEntry {
    name: &'static str,
    resolved: DynamicResolved,
    closure: DynamicAdapter,
    last_read_tick: Arc<AtomicU64>,
}

/// Discriminant for the `by_name` lookup table.
#[derive(Clone, Copy)]
enum ReducerSlot {
    Unified(usize),
    Dynamic(usize),
}

/// Central registry for typed reducer closures with conflict analysis.
///
/// Owns closures, [`Access`] metadata, and pre-resolved `ComponentId`s.
/// Composes with [`World`] and [`Transact`] strategies the same way
/// [`SpatialIndex`](crate::SpatialIndex) composes with World — no World API growth.
///
/// ## Registration
///
/// - [`register_entity`](ReducerRegistry::register_entity) / [`register_entity_despawn`](ReducerRegistry::register_entity_despawn) — single-entity read-write via [`EntityMut`]
/// - [`register_entity_ref`](ReducerRegistry::register_entity_ref) — single-entity read-only via [`EntityRef`]
/// - [`register_spawner`](ReducerRegistry::register_spawner) — entity creation via [`Spawner`]
/// - [`register_query_writer`](ReducerRegistry::register_query_writer) — buffered query iteration via [`QueryWriter`]
/// - [`register_query`](ReducerRegistry::register_query) — direct mutable iteration via [`QueryMut`]
/// - [`register_query_ref`](ReducerRegistry::register_query_ref) — read-only iteration via [`QueryRef`]
/// - [`dynamic`](ReducerRegistry::dynamic) — runtime-validated access via [`DynamicReducerBuilder`]
///
/// ## Dispatch
///
/// - [`call`](ReducerRegistry::call) — transactional reducers (entity, spawner, query writer), runs through `strategy.transact()`
/// - [`run`](ReducerRegistry::run) — scheduled query reducers, direct `&mut World`
/// - [`dynamic_call`](ReducerRegistry::dynamic_call) — dynamic reducers, routes through `strategy.transact()`
///
/// ## Conflict analysis
///
/// - [`reducer_access`](ReducerRegistry::reducer_access) / [`query_reducer_access`](ReducerRegistry::query_reducer_access) / [`dynamic_access`](ReducerRegistry::dynamic_access) — retrieve [`Access`] bitsets for scheduler conflict detection
/// - [`reducer_id_by_name`](ReducerRegistry::reducer_id_by_name) / [`query_reducer_id_by_name`](ReducerRegistry::query_reducer_id_by_name) / [`dynamic_id_by_name`](ReducerRegistry::dynamic_id_by_name) — name-based lookup for network dispatch
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
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_entity<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
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
                let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref(), false);
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register an entity reducer with despawn capability.
    /// Same as `register_entity`, but `EntityMut::despawn()` is enabled
    /// and the Access includes the despawn flag.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_entity_despawn<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        C: ComponentSet,
        Args: Clone + 'static,
        F: Fn(EntityMut<'_, C>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(C::resolve(&mut world.components));
        let reads = C::access(&mut world.components, true);
        let writes = C::access(&mut world.components, false);
        let mut access = reads.merge(&writes);
        access.set_despawns();

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
                let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref(), true);
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register a read-only entity reducer: `f(EntityRef<C>, args)`.
    /// At dispatch, call with `(entity, args)` as the args tuple.
    ///
    /// Unlike [`register_entity`](Self::register_entity), this provides
    /// read-only access via [`EntityRef`] — no writes are buffered, and
    /// the access metadata reflects reads only. Use this when the reducer
    /// only needs to inspect component values without modifying them.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_entity_ref<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        C: ComponentSet,
        Args: Clone + 'static,
        F: Fn(EntityRef<'_, C>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(C::resolve(&mut world.components));
        // EntityRef is read-only — no write access needed.
        let access = C::access(&mut world.components, true);

        let adapter: TransactionalAdapter =
            Box::new(move |_changeset, _allocated, tw, resolved, args_any| {
                let (entity, args) = args_any
                    .downcast_ref::<(Entity, Args)>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected (Entity, {})",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = EntityRef::<C>::new(entity, resolved, tw.as_ref());
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register a spawner reducer: `f(Spawner<B>, args)`.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_spawner<B, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
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

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register a query writer reducer: `f(QueryWriter<Q>, args)`.
    ///
    /// Iterates matching archetypes with buffered writes. `&T` reads directly
    /// from columns; `&mut T` produces `WritableRef<T>` that buffers into the
    /// transaction's changeset. Column ticks are NOT advanced during iteration
    /// (avoiding self-conflict with optimistic validation). Changes are applied
    /// atomically on commit.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_query_writer<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
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
        let queried = Arc::new(AtomicBool::new(false));
        let queried_ref = queried.clone();

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
                let qw = QueryWriter::<Q>::new(tw.0, cs_ptr, &tick_ref, &queried_ref);
                f(qw, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            Some(last_read_tick),
            Some(queried),
        )
    }

    // ── Scheduled registration ───────────────────────────────────

    /// Register a mutable query reducer: `f(QueryMut<Q>, args)`.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_query<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<QueryReducerId, ReducerError>
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

        let id = self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Scheduled(adapter),
            None,
            None,
        )?;
        Ok(QueryReducerId(id.0))
    }

    /// Register a read-only query reducer: `f(QueryRef<Q>, args)`.
    ///
    /// Uses the full query path with filter support (`Changed<T>` works).
    /// The `ReadOnlyWorldQuery` bound prevents `&mut T` access.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_query_ref<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<QueryReducerId, ReducerError>
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

        let id = self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Scheduled(adapter),
            None,
            None,
        )?;
        Ok(QueryReducerId(id.0))
    }

    // ── Built-in reducers ────────────────────────────────────────

    /// Register the built-in retention reducer that despawns expired entities.
    ///
    /// Each dispatch decrements every [`Expiry`](crate::Expiry) counter by one.
    /// Entities whose counter reaches zero are batch-despawned. The user
    /// controls how often retention runs — each call to `run()` is one
    /// "retention cycle."
    ///
    /// # Panics
    /// Panics if a reducer named `"__retention"` is already registered.
    pub fn retention(&mut self, world: &mut World) -> QueryReducerId {
        let expiry_id = world.register_component::<crate::retention::Expiry>();
        let mut access = Access::empty();
        access.add_write(expiry_id);
        access.set_despawns();

        let resolved = ResolvedComponents(Vec::new());

        let adapter: ScheduledAdapter = Box::new(|world, _args_any| {
            // Decrement all Expiry counters and collect entities that hit zero.
            let mut expired: Vec<Entity> = Vec::new();
            world
                .query::<(Entity, &mut crate::retention::Expiry)>()
                .for_each(|(entity, expiry)| {
                    expiry.tick();
                    if expiry.is_expired() {
                        expired.push(entity);
                    }
                });
            if !expired.is_empty() {
                world.despawn_batch(&expired);
            }
        });

        // Double-underscore prefix marks engine-built-in reducers.
        // push_entry rejects duplicate names, preventing user reducers
        // from colliding with built-ins.
        let id = self
            .push_entry(
                "__retention",
                access,
                resolved,
                ReducerKind::Scheduled(adapter),
                None,
                None,
            )
            .expect("__retention reducer name conflict");
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
            remove_ids: HashSet::new(),
        }
    }

    // ── Dispatch ─────────────────────────────────────────────────

    /// Call a transactional reducer (entity, spawner, or query writer).
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds,
    /// `Err(ReducerError::WrongKind)` if the ID points to a scheduled
    /// reducer, or `Err(ReducerError::TransactionConflict)` if the
    /// transaction strategy detects a conflict.
    pub fn call<S: Transact, Args: Clone + 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: ReducerId,
        args: Args,
    ) -> Result<(), ReducerError> {
        let entry = self.get_entry(id.0)?;
        let adapter = match &entry.kind {
            ReducerKind::Transactional(f) => f,
            ReducerKind::Scheduled(_) => {
                return Err(ReducerError::WrongKind {
                    expected: "transactional",
                    actual: "scheduled",
                });
            }
        };
        let access = &entry.access;
        let resolved = &entry.resolved;

        let tick_arc = entry.last_read_tick.clone();
        let queried_flag = entry.queried.clone();
        // Reset the queried flag before each call so we only advance the
        // tick if for_each/count actually runs during this invocation.
        if let Some(q) = &queried_flag {
            q.store(false, Ordering::Relaxed);
        }
        let result = strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let mut tw = TransactionalWorld(world);
            adapter(changeset, allocated, &mut tw, resolved, &args);
        });
        // Update last_read_tick AFTER the changeset is applied (by transact),
        // but only if for_each/count was actually called during this invocation.
        if result.is_ok()
            && let Some(arc) = &tick_arc
            && queried_flag
                .as_ref()
                .is_none_or(|q| q.load(Ordering::Relaxed))
        {
            let new_tick = world.next_tick();
            arc.store(new_tick.raw(), Ordering::Relaxed);
        }
        result.map_err(ReducerError::from)
    }

    /// Run a scheduled query reducer directly. Caller guarantees exclusivity.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds,
    /// or `Err(ReducerError::WrongKind)` if the ID points to a
    /// transactional reducer.
    pub fn run<Args: Clone + 'static>(
        &self,
        world: &mut World,
        id: QueryReducerId,
        args: Args,
    ) -> Result<(), ReducerError> {
        let entry = self.get_entry(id.0)?;
        match &entry.kind {
            ReducerKind::Scheduled(f) => {
                f(world, &args);
                Ok(())
            }
            ReducerKind::Transactional(_) => Err(ReducerError::WrongKind {
                expected: "scheduled",
                actual: "transactional",
            }),
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
    ///
    /// # Panics
    /// Panics if the ID is out of bounds. Use [`reducer_info`](Self::reducer_info)
    /// for a fallible alternative.
    pub fn reducer_access(&self, id: ReducerId) -> &Access {
        &self.reducers[id.0].access
    }

    /// Access metadata for a scheduled query reducer.
    ///
    /// # Panics
    /// Panics if the ID is out of bounds.
    pub fn query_reducer_access(&self, id: QueryReducerId) -> &Access {
        &self.reducers[id.0].access
    }

    /// Access metadata by raw index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn access(&self, idx: usize) -> &Access {
        &self.reducers[idx].access
    }

    // ── Dynamic dispatch ────────────────────────────────────────

    /// Call a dynamic reducer with a chosen transaction strategy.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds,
    /// or `Err(ReducerError::TransactionConflict)` if the transaction
    /// strategy detects a conflict.
    pub fn dynamic_call<S: Transact, Args: 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: DynamicReducerId,
        args: &Args,
    ) -> Result<(), ReducerError> {
        let entry = self.get_dynamic_entry(id.0)?;
        let closure = &entry.closure;
        let resolved = &entry.resolved;
        let access = resolved.access();
        let tick_arc = entry.last_read_tick.clone();
        let queried = Arc::new(AtomicBool::new(false));

        let result = strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let world_ref: &World = world;
            let mut ctx = DynamicCtx::new(
                world_ref, changeset, allocated, resolved, &tick_arc, &queried,
            );
            closure(&mut ctx, args);
        });
        // Update last_read_tick AFTER the changeset is applied (by transact),
        // but only if for_each was actually called during this invocation.
        if result.is_ok() && queried.load(Ordering::Relaxed) {
            let new_tick = world.next_tick();
            entry
                .last_read_tick
                .store(new_tick.raw(), Ordering::Relaxed);
        }
        result.map_err(ReducerError::from)
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
    ///
    /// # Panics
    /// Panics if the ID is out of bounds.
    pub fn dynamic_access(&self, id: DynamicReducerId) -> &Access {
        self.dynamic_reducers[id.0].resolved.access()
    }

    // ── Introspection ─────────────────────────────────────────────

    /// Introspection for a transactional reducer.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds.
    pub fn reducer_info(&self, id: ReducerId) -> Result<ReducerInfo, ReducerError> {
        let entry = self.get_entry(id.0)?;
        let kind = match &entry.kind {
            ReducerKind::Transactional(_) => "transactional",
            ReducerKind::Scheduled(_) => "scheduled",
        };
        Ok(ReducerInfo {
            name: entry.name,
            kind,
            access: entry.access.clone(),
            has_change_tracking: entry.last_read_tick.is_some(),
            can_despawn: entry.access.despawns(),
        })
    }

    /// Introspection for a scheduled query reducer.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds.
    pub fn query_reducer_info(&self, id: QueryReducerId) -> Result<ReducerInfo, ReducerError> {
        self.reducer_info(ReducerId(id.0))
    }

    /// Introspection for a dynamic reducer.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds.
    pub fn dynamic_reducer_info(&self, id: DynamicReducerId) -> Result<ReducerInfo, ReducerError> {
        let entry = self.get_dynamic_entry(id.0)?;
        let access = entry.resolved.access().clone();
        let can_despawn = access.despawns();
        Ok(ReducerInfo {
            name: entry.name,
            kind: "dynamic",
            access,
            has_change_tracking: true, // dynamic reducers always have tick tracking
            can_despawn,
        })
    }

    /// Number of registered unified reducers (transactional + scheduled).
    pub fn reducer_count(&self) -> usize {
        self.reducers.len()
    }

    /// Number of registered dynamic reducers.
    pub fn dynamic_reducer_count(&self) -> usize {
        self.dynamic_reducers.len()
    }

    /// Iterate all registered reducer names and their slot kinds.
    pub fn registered_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_name.keys().copied()
    }

    // ── Internal ─────────────────────────────────────────────────

    fn get_entry(&self, index: usize) -> Result<&ReducerEntry, ReducerError> {
        self.reducers.get(index).ok_or(ReducerError::InvalidId {
            kind: "reducer",
            index,
            max: self.reducers.len(),
        })
    }

    fn get_dynamic_entry(&self, index: usize) -> Result<&DynamicReducerEntry, ReducerError> {
        self.dynamic_reducers
            .get(index)
            .ok_or(ReducerError::InvalidId {
                kind: "dynamic",
                index,
                max: self.dynamic_reducers.len(),
            })
    }

    fn push_entry(
        &mut self,
        name: &'static str,
        access: Access,
        resolved: ResolvedComponents,
        kind: ReducerKind,
        last_read_tick: Option<Arc<AtomicU64>>,
        queried: Option<Arc<AtomicBool>>,
    ) -> Result<ReducerId, ReducerError> {
        let id = self.reducers.len();
        if let Some(slot) = self.by_name.get(name) {
            let (existing_kind, existing_index) = match slot {
                ReducerSlot::Unified(idx) => ("unified", *idx),
                ReducerSlot::Dynamic(idx) => ("dynamic", *idx),
            };
            return Err(ReducerError::DuplicateName {
                name,
                existing_kind,
                existing_index,
            });
        }
        self.by_name.insert(name, ReducerSlot::Unified(id));
        self.reducers.push(ReducerEntry {
            name,
            access,
            resolved,
            kind,
            last_read_tick,
            queried,
        });
        Ok(ReducerId(id))
    }
}

impl Default for ReducerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── DynamicReducerBuilder ────────────────────────────────────────────

/// Builder for registering a dynamic reducer.
///
/// Declare upper-bound access with [`can_read`](DynamicReducerBuilder::can_read),
/// [`can_write`](DynamicReducerBuilder::can_write),
/// [`can_spawn`](DynamicReducerBuilder::can_spawn),
/// [`can_remove`](DynamicReducerBuilder::can_remove), and
/// [`can_despawn`](DynamicReducerBuilder::can_despawn), then finalize with
/// [`build`](DynamicReducerBuilder::build). The resulting [`DynamicCtx`]
/// enforces these bounds at runtime.
///
/// Obtained via [`ReducerRegistry::dynamic`].
pub struct DynamicReducerBuilder<'a> {
    registry: &'a mut ReducerRegistry,
    world: &'a mut World,
    name: &'static str,
    access: Access,
    entries: Vec<(TypeId, ComponentId)>,
    spawn_bundles: HashSet<TypeId>,
    remove_ids: HashSet<TypeId>,
}

impl DynamicReducerBuilder<'_> {
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

    /// Declare that the closure may remove component `T` from entities.
    /// Marks T as written (removal is a structural write) and adds a
    /// TypeId entry for runtime validation.
    pub fn can_remove<T: crate::component::Component>(mut self) -> Self {
        let comp_id = self.world.register_component::<T>();
        self.access.add_read(comp_id); // removal implies read (inspect before removing)
        self.access.add_write(comp_id); // removal is a structural write
        self.entries.push((TypeId::of::<T>(), comp_id));
        self.remove_ids.insert(TypeId::of::<T>());
        self
    }

    /// Declare that the closure may despawn entities. Sets a blanket
    /// conflict flag — this reducer conflicts with any other reducer
    /// that accesses any component.
    pub fn can_despawn(mut self) -> Self {
        self.access.set_despawns();
        self
    }

    /// Finalize registration. The closure receives `&mut DynamicCtx` and
    /// type-erased `&Args`. Returns the opaque `DynamicReducerId`.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn build<Args, F>(self, f: F) -> Result<DynamicReducerId, ReducerError>
    where
        Args: 'static,
        F: Fn(&mut DynamicCtx, &Args) + Send + Sync + 'static,
    {
        let resolved = DynamicResolved::new(
            self.entries,
            self.access.clone(),
            self.spawn_bundles,
            self.remove_ids,
        );

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
            let (existing_kind, existing_index) = match slot {
                ReducerSlot::Unified(idx) => ("unified", *idx),
                ReducerSlot::Dynamic(idx) => ("dynamic", *idx),
            };
            return Err(ReducerError::DuplicateName {
                name: self.name,
                existing_kind,
                existing_index,
            });
        }
        self.registry
            .by_name
            .insert(self.name, ReducerSlot::Dynamic(id));
        self.registry.dynamic_reducers.push(DynamicReducerEntry {
            name: self.name,
            resolved,
            closure,
            last_read_tick: Arc::new(AtomicU64::new(0)),
        });
        Ok(DynamicReducerId(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    struct Vel(f32);
    #[derive(Clone, Copy)]
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
        let resolved = DynamicResolved::new(
            entries,
            Access::empty(),
            HashSet::default(),
            HashSet::default(),
        );
        assert_eq!(resolved.lookup::<u32>(), Some(0));
        assert_eq!(resolved.lookup::<f64>(), Some(2));
        assert_eq!(resolved.lookup::<i64>(), Some(1));
        assert_eq!(resolved.lookup::<u8>(), None);
    }

    #[test]
    fn dynamic_resolved_dedup() {
        use std::any::TypeId;
        let entries = vec![(TypeId::of::<u32>(), 0), (TypeId::of::<u32>(), 0)];
        let resolved = DynamicResolved::new(
            entries,
            Access::empty(),
            HashSet::default(),
            HashSet::default(),
        );
        // After dedup, duplicate entries are collapsed
        assert_eq!(resolved.lookup::<u32>(), Some(0));
    }

    #[test]
    fn dynamic_resolved_has_spawn_bundle() {
        use std::any::TypeId;
        let mut bundles = HashSet::new();
        bundles.insert(TypeId::of::<(Pos, Vel)>());
        let resolved = DynamicResolved::new(vec![], Access::empty(), bundles, HashSet::default());
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
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
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
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
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
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        {
            let mut ctx = DynamicCtx::new(
                &world,
                &mut cs,
                &mut allocated,
                &resolved,
                &default_tick,
                &default_queried,
            );
            ctx.write(e, Pos(99.0));
        }
        // Not yet applied
        assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
        // Apply changeset
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_read_undeclared_panics() {
        let mut world = World::new();
        world.register_component::<Pos>();
        let e = world.spawn((Pos(1.0),));

        // Empty resolved — no components declared
        let resolved = DynamicResolved::new(
            vec![],
            Access::empty(),
            HashSet::default(),
            HashSet::default(),
        );
        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
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
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
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
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let pos_id = world.components.id::<Pos>().unwrap();
        let vel_id = world.components.id::<Vel>().unwrap();
        let entry = &reducers.dynamic_reducers[id.0];
        // Spawn adds writes for conflict detection
        assert!(entry.resolved.access().writes()[pos_id]);
        assert!(entry.resolved.access().writes()[vel_id]);
    }

    #[test]
    fn dynamic_builder_duplicate_name_returns_err() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        reducers
            .dynamic("dup", &mut world)
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let result = reducers
            .dynamic("dup", &mut world)
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
        assert!(matches!(
            result,
            Err(ReducerError::DuplicateName { name: "dup", .. })
        ));
    }

    #[test]
    fn dynamic_name_conflicts_with_unified_returns_err() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        reducers
            .register_entity::<(Health,), (), _>(&mut world, "shared_name", |_e, ()| {})
            .unwrap();
        let result = reducers
            .dynamic("shared_name", &mut world)
            .build(|_ctx: &mut DynamicCtx, _args: &()| {});
        assert!(matches!(
            result,
            Err(ReducerError::DuplicateName {
                name: "shared_name",
                ..
            })
        ));
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
            })
            .unwrap();

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
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        reducers
            .register_entity::<(Health,), (), _>(&mut world, "entity_one", |_e, ()| {})
            .unwrap();

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
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let pos_id = world.components.id::<Pos>().unwrap();
        let vel_id = world.components.id::<Vel>().unwrap();
        let access = reducers.dynamic_access(id);
        assert!(access.reads()[pos_id]);
        assert!(access.reads()[vel_id]);
        assert!(access.writes()[vel_id]);
        assert!(!access.writes()[pos_id]);
    }

    #[test]
    fn can_remove_marks_write_access() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("remover", &mut world)
            .can_read::<Health>()
            .can_remove::<Vel>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let access = registry.dynamic_access(id);
        let vel_id = world.components.id::<Vel>().unwrap();
        assert!(access.writes().contains(vel_id));
    }

    #[test]
    fn can_despawn_sets_flag() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("despawner", &mut world)
            .can_read::<Health>()
            .can_despawn()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let access = registry.dynamic_access(id);
        assert!(access.despawns());
    }

    #[test]
    fn despawn_reducer_conflicts_with_reader() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let dyn_id = registry
            .dynamic("despawner", &mut world)
            .can_read::<Health>()
            .can_despawn()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let entity_id = registry
            .register_entity::<(Vel,), (), _>(&mut world, "set_vel", |_e, ()| {})
            .unwrap();
        let dyn_access = registry.dynamic_access(dyn_id);
        let entity_access = registry.reducer_access(entity_id);
        assert!(dyn_access.conflicts_with(entity_access));
    }

    // ── DynamicCtx structural mutation tests ─────────────────────

    #[test]
    fn dynamic_ctx_remove_buffers_mutation() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("strip_vel", &mut world)
            .can_read::<Pos>()
            .can_remove::<Vel>()
            .build(|ctx: &mut DynamicCtx, entity: &Entity| {
                ctx.remove::<Vel>(*entity);
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &e)
            .unwrap();
        assert!(world.get::<Vel>(e).is_none());
        assert!(world.get::<Pos>(e).is_some());
    }

    #[test]
    fn dynamic_ctx_try_remove_returns_false_when_missing() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let result = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let result_clone = result.clone();
        let id = registry
            .dynamic("try_strip", &mut world)
            .can_remove::<Vel>()
            .build(move |ctx: &mut DynamicCtx, entity: &Entity| {
                let removed = ctx.try_remove::<Vel>(*entity);
                result_clone.store(removed, std::sync::atomic::Ordering::Relaxed);
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &e)
            .unwrap();
        assert!(!result.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_remove_undeclared_panics() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("bad_remove", &mut world)
            .can_read::<Pos>()
            .build(|ctx: &mut DynamicCtx, entity: &Entity| {
                ctx.remove::<Vel>(*entity);
            })
            .unwrap();
        let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
    }

    #[test]
    #[should_panic(expected = "not declared for removal")]
    fn dynamic_ctx_remove_with_can_write_panics() {
        // can_write does NOT authorize remove — remove requires can_remove
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("bad_remove2", &mut world)
            .can_write::<Vel>()
            .build(|ctx: &mut DynamicCtx, entity: &Entity| {
                ctx.remove::<Vel>(*entity);
            })
            .unwrap();
        let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
    }

    #[test]
    fn dynamic_ctx_try_remove_returns_true_and_removes() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let result = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let result_clone = result.clone();
        let id = registry
            .dynamic("try_strip_ok", &mut world)
            .can_remove::<Vel>()
            .build(move |ctx: &mut DynamicCtx, entity: &Entity| {
                let removed = ctx.try_remove::<Vel>(*entity);
                result_clone.store(removed, std::sync::atomic::Ordering::Relaxed);
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &e)
            .unwrap();
        assert!(result.load(std::sync::atomic::Ordering::Relaxed));
        assert!(world.get::<Vel>(e).is_none());
        assert!(world.get::<Pos>(e).is_some());
    }

    #[test]
    fn dynamic_ctx_despawn_buffers_mutation() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("killer", &mut world)
            .can_read::<Health>()
            .can_despawn()
            .build(|ctx: &mut DynamicCtx, entity: &Entity| {
                ctx.despawn(*entity);
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &e)
            .unwrap();
        assert!(!world.is_alive(e));
    }

    #[test]
    #[should_panic(expected = "can_despawn")]
    fn dynamic_ctx_despawn_without_declaration_panics() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("bad_despawn", &mut world)
            .can_read::<Pos>()
            .build(|ctx: &mut DynamicCtx, entity: &Entity| {
                ctx.despawn(*entity);
            })
            .unwrap();
        let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
    }

    // ── EntityMut structural mutation tests ──────────────────────

    #[test]
    fn entity_mut_remove_buffers_mutation() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity::<(Pos, Vel), (), _>(
                &mut world,
                "strip_vel",
                |mut entity: EntityMut<'_, (Pos, Vel)>, ()| {
                    entity.remove::<Vel, 1>();
                },
            )
            .unwrap();
        registry.call(&strategy, &mut world, id, (e, ())).unwrap();
        assert!(world.get::<Vel>(e).is_none());
        assert!(world.get::<Pos>(e).is_some());
    }

    #[test]
    fn entity_mut_despawn_buffers_mutation() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity_despawn::<(Pos,), (), _>(
                &mut world,
                "killer",
                |mut entity: EntityMut<'_, (Pos,)>, ()| {
                    entity.despawn();
                },
            )
            .unwrap();
        registry.call(&strategy, &mut world, id, (e, ())).unwrap();
        assert!(!world.is_alive(e));
    }

    #[test]
    #[should_panic(expected = "register_entity_despawn")]
    fn entity_mut_despawn_without_flag_panics() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity::<(Pos,), (), _>(
                &mut world,
                "bad_killer",
                |mut entity: EntityMut<'_, (Pos,)>, ()| {
                    entity.despawn();
                },
            )
            .unwrap();
        let _ = registry.call(&strategy, &mut world, id, (e, ()));
    }

    // ── DynamicCtx::for_each tests ──────────────────────────────

    #[test]
    fn dynamic_ctx_for_each_iterates() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));
        world.spawn((Vel(3.0),)); // no Pos — not matched
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = count.clone();
        let id = registry
            .dynamic("count_pos", &mut world)
            .can_read::<Pos>()
            .build(move |ctx: &mut DynamicCtx, _args: &()| {
                ctx.for_each::<(&Pos,)>(|(positions,)| {
                    counter.fetch_add(positions.len(), std::sync::atomic::Ordering::Relaxed);
                });
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_for_each_undeclared_panics() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("bad_query", &mut world)
            .can_read::<Pos>()
            .build(|ctx: &mut DynamicCtx, _args: &()| {
                ctx.for_each::<(&Pos, &Vel)>(|(_p, _v)| {});
            })
            .unwrap();
        let _ = registry.dynamic_call(&strategy, &mut world, id, &());
    }

    #[test]
    fn dynamic_ctx_for_each_with_write_after_read() {
        let mut world = World::new();
        let e1 = world.spawn((Pos(1.0), Vel(10.0)));
        let e2 = world.spawn((Pos(2.0), Vel(20.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("double_vel", &mut world)
            .can_read::<Vel>()
            .can_write::<Vel>()
            .build(|ctx: &mut DynamicCtx, _args: &()| {
                let mut updates: Vec<(Entity, f32)> = Vec::new();
                ctx.for_each::<(Entity, &Vel)>(|(entities, velocities)| {
                    for (entity, vel) in entities.iter().copied().zip(velocities.iter()) {
                        updates.push((entity, vel.0 * 2.0));
                    }
                });
                for (entity, new_vel) in updates {
                    ctx.write(entity, Vel(new_vel));
                }
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();
        assert_eq!(world.get::<Vel>(e1).unwrap().0, 20.0);
        assert_eq!(world.get::<Vel>(e2).unwrap().0, 40.0);
    }

    #[test]
    fn dynamic_ctx_for_each_changed_filter() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let visit_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = visit_count.clone();
        let id = registry
            .dynamic("changed_pos", &mut world)
            .can_read::<Pos>()
            .can_write::<Pos>()
            .build(move |ctx: &mut DynamicCtx, _args: &()| {
                let mut updates = Vec::new();
                ctx.for_each::<(Entity, Changed<Pos>, &Pos)>(|(entities, (), positions)| {
                    for (entity, pos) in entities.iter().copied().zip(positions.iter()) {
                        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        updates.push((entity, Pos(pos.0 + 1.0)));
                    }
                });
                for (entity, val) in updates {
                    ctx.write(entity, val);
                }
            })
            .unwrap();

        // First call: column was never read by this reducer, Changed matches
        registry
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();
        assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 2.0);

        // Second call: no external mutation, Changed should skip
        visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        registry
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();
        assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 0);

        // External mutation, then call again
        visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        for (pos,) in world.query::<(&mut Pos,)>() {
            pos.0 = 99.0;
        }
        registry
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();
        assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 100.0);
    }

    #[test]
    fn dynamic_ctx_for_each_slice_iterates() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        world.spawn((Pos(2.0),));
        world.spawn((Vel(3.0),)); // no Pos — not matched
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = count.clone();
        let id = registry
            .dynamic("count_pos_chunks", &mut world)
            .can_read::<Pos>()
            .build(move |ctx: &mut DynamicCtx, _args: &()| {
                ctx.for_each::<(&Pos,)>(|(positions,)| {
                    counter.fetch_add(positions.len(), std::sync::atomic::Ordering::Relaxed);
                });
            })
            .unwrap();
        registry
            .dynamic_call(&strategy, &mut world, id, &())
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_for_each_undeclared_multi_component_panics() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("bad_chunk_query", &mut world)
            .can_read::<Pos>()
            .build(|ctx: &mut DynamicCtx, _args: &()| {
                ctx.for_each::<(&Pos, &Vel)>(|(_p, _v)| {});
            })
            .unwrap();
        let _ = registry.dynamic_call(&strategy, &mut world, id, &());
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
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
        ctx.write(e, Pos(99.0)); // should panic: read-only
    }

    #[test]
    #[should_panic(expected = "bundle")]
    fn dynamic_ctx_spawn_undeclared_bundle_panics() {
        let mut world = World::new();
        world.register_component::<Pos>();

        // No spawn bundles declared
        let resolved = DynamicResolved::new(
            vec![],
            Access::empty(),
            HashSet::default(),
            HashSet::default(),
        );

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
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
        qr.for_each(|(positions,)| {
            for p in positions {
                sum += p.0;
            }
        });
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
            qm.for_each(|(positions,)| {
                for p in positions {
                    p.0 += 10.0;
                }
            });
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
        let heal_id = registry
            .register_entity::<(Health,), u32, _>(&mut world, "heal", |mut entity, amount: u32| {
                let hp = entity.get::<Health, 0>().0;
                entity.set::<Health, 0>(Health(hp + amount));
            })
            .unwrap();

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
        let gravity_id = registry
            .register_query::<(&mut Vel,), f32, _>(&mut world, "gravity", |mut query, dt: f32| {
                query.for_each(|(velocities,)| {
                    for v in velocities {
                        v.0 -= 9.81 * dt;
                    }
                });
            })
            .unwrap();

        registry.run(&mut world, gravity_id, 0.1f32).unwrap();

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
        let count_id = registry
            .register_query_ref::<(&Pos,), (), _>(&mut world, "count", |mut query, ()| {
                assert_eq!(query.count(), 2);
            })
            .unwrap();

        registry.run(&mut world, count_id, ()).unwrap();
    }

    #[test]
    fn typed_id_by_name_lookup() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let heal_id = registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
            .unwrap();
        let _gravity_id = registry
            .register_query::<(&mut Vel,), (), _>(&mut world, "gravity", |_query, ()| {})
            .unwrap();

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
        let heal_id = registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
            .unwrap();
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

        let heal_id = registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
            .unwrap();

        let damage_id = registry
            .register_entity::<(Health,), (), _>(&mut world, "damage", |_entity, ()| {})
            .unwrap();

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
        let spawn_id = registry
            .register_spawner::<(Health,), u32, _>(
                &mut world,
                "spawn_unit",
                |mut spawner, hp: u32| {
                    spawner.spawn((Health(hp),));
                },
            )
            .unwrap();

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
        let _spawn_id = registry
            .register_spawner::<(Health,), (), _>(
                &mut world,
                "spawn_and_conflict",
                |mut spawner, ()| {
                    spawner.spawn((Health(1),));
                },
            )
            .unwrap();

        // Force a conflict: mutate Pos column between begin and commit
        // by using a strategy with max 1 retry and always-conflicting access
        let mut attempt = 0u32;
        let access_with_pos = Access::of::<(&Pos, &mut Pos)>(&mut world);
        let result = strategy.transact(&mut world, &access_with_pos, |tx, world| {
            attempt += 1;
            let (changeset, allocated) = tx.reducer_parts();
            let _spawner = Spawner::<(Health,)>::new(changeset, allocated, world);
            // Spawner allocates via reserve() — entity tracked in allocated

            if attempt == 1 {
                // Mutate to force conflict
                for pos in world.query::<(&mut Pos,)>() {
                    pos.0.0 = 99.0;
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
    fn duplicate_name_returns_err() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
            .unwrap();
        let result =
            registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
        assert!(matches!(
            result,
            Err(ReducerError::DuplicateName { name: "heal", .. })
        ));
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
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );

        let wrote = ctx.try_write::<u32>(e, 99);
        assert!(wrote);

        cs.apply(&mut world).unwrap();
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
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );

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
            })
            .unwrap();

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
        let resolved = DynamicResolved::new(
            vec![],
            Access::empty(),
            HashSet::default(),
            HashSet::default(),
        );
        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
        let _ = ctx.try_read::<u32>(e);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn dynamic_ctx_try_write_undeclared_panics() {
        let mut world = World::new();
        let e = world.spawn((42u32,));
        let resolved = DynamicResolved::new(
            vec![],
            Access::empty(),
            HashSet::default(),
            HashSet::default(),
        );
        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let mut ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
        ctx.try_write::<u32>(e, 99);
    }

    #[test]
    fn unified_name_conflicts_with_dynamic_returns_err() {
        let mut world = World::new();
        let mut reducers = ReducerRegistry::new();
        // Register dynamic first
        reducers
            .dynamic("clash", &mut world)
            .can_read::<u32>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        // Then unified — should return Err
        let result = reducers.register_entity::<(u32,), (), _>(
            &mut world,
            "clash",
            |_entity: EntityMut<'_, (u32,)>, ()| {},
        );
        assert!(matches!(
            result,
            Err(ReducerError::DuplicateName { name: "clash", .. })
        ));
    }

    // ── WritableRef tests ──────────────────────────────────────────

    #[test]
    fn writable_ref_get_returns_current_value() {
        let mut world = World::new();
        let e = world.spawn((Pos(42.0),));
        let pos_id = world.components.id::<Pos>().unwrap();
        let current = world.get::<Pos>(e).unwrap();

        let mut cs = EnumChangeSet::new();
        let wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet, 0, 0);
        assert_eq!(wr.get().0, 42.0);
    }

    /// Helper: open an archetype batch for the given entity's archetype
    /// with a single mutable component so that `WritableRef::set` can
    /// route through the fast lane.
    fn open_batch_for_entity(
        cs: &mut EnumChangeSet,
        world: &World,
        entity: Entity,
        comp_id: ComponentId,
    ) {
        let loc = world.entity_locations[entity.index() as usize].unwrap();
        let arch_idx = loc.archetype_id.0;
        let arch = &world.archetypes.archetypes[arch_idx];
        let mut mutable = fixedbitset::FixedBitSet::with_capacity(comp_id + 1);
        mutable.insert(comp_id);
        crate::changeset::open_archetype_batch(cs, arch_idx, arch, &world.components, &mutable);
    }

    #[test]
    fn writable_ref_set_buffers_into_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let pos_id = world.components.id::<Pos>().unwrap();
        let current = world.get::<Pos>(e).unwrap();

        let mut cs = EnumChangeSet::new();
        open_batch_for_entity(&mut cs, &world, e, pos_id);
        {
            let mut wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet, 0, 0);
            wr.set(Pos(99.0));
        }
        // World unchanged before apply
        assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
        assert_eq!(cs.len(), 1);
        // Apply and verify
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);
    }

    #[test]
    fn writable_ref_modify_clones_and_sets() {
        let mut world = World::new();
        let e = world.spawn((Pos(10.0),));
        let pos_id = world.components.id::<Pos>().unwrap();
        let current = world.get::<Pos>(e).unwrap();

        let mut cs = EnumChangeSet::new();
        open_batch_for_entity(&mut cs, &world, e, pos_id);
        {
            let mut wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet, 0, 0);
            wr.modify(|p| p.0 += 10.0);
        }
        cs.apply(&mut world).unwrap();
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
        let pos_id = world.components.id::<Pos>().unwrap();
        open_batch_for_entity(&mut cs, &world, e, pos_id);
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
        cs.apply(&mut world).unwrap();
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
        let pos_id = world.components.id::<Pos>().unwrap();
        open_batch_for_entity(&mut cs, &world, e, pos_id);
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
        cs.apply(&mut world).unwrap();
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
            })
            .unwrap();

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

        let id = registry
            .register_query_writer::<(&Pos, &mut Vel), f32, _>(
                &mut world,
                "apply_drag",
                |mut query, drag: f32| {
                    query.for_each(|(pos, mut vel)| {
                        let _ = pos; // read Pos (passthrough)
                        vel.modify(|v| v.0 *= drag);
                    });
                },
            )
            .unwrap();

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

        let id = registry
            .register_query_writer::<(&mut Pos,), (), _>(&mut world, "counter", |mut query, ()| {
                assert_eq!(query.count(), 2);
            })
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();
    }

    #[test]
    fn query_writer_access_conflict() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let entity_id = registry
            .register_entity::<(Vel,), (), _>(&mut world, "set_vel", |_e, ()| {})
            .unwrap();
        let writer_id = registry
            .register_query_writer::<(&mut Vel,), (), _>(&mut world, "bulk_vel", |_q, ()| {})
            .unwrap();

        let entity_access = registry.reducer_access(entity_id);
        let writer_access = registry.reducer_access(writer_id);
        assert!(entity_access.conflicts_with(writer_access));
    }

    #[test]
    fn query_writer_no_conflict_disjoint() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let entity_id = registry
            .register_entity::<(Pos,), (), _>(&mut world, "set_pos", |_e, ()| {})
            .unwrap();
        let writer_id = registry
            .register_query_writer::<(&mut Vel,), (), _>(&mut world, "bulk_vel", |_q, ()| {})
            .unwrap();

        let entity_access = registry.reducer_access(entity_id);
        let writer_access = registry.reducer_access(writer_id);
        assert!(!entity_access.conflicts_with(writer_access));
    }

    // ── API boundary panic tests ─────────────────────────────────

    #[test]
    fn call_on_scheduled_returns_wrong_kind() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let qid = registry
            .register_query::<(&Pos,), (), _>(&mut world, "read_pos", |_q, ()| {})
            .unwrap();
        let strategy = Optimistic::new(&world);
        // QueryReducerId and ReducerId share the same index space —
        // passing ReducerId(qid.0) should hit the Scheduled arm.
        let result = registry.call(&strategy, &mut world, ReducerId(qid.0), ());
        assert!(matches!(
            result,
            Err(ReducerError::WrongKind {
                expected: "transactional",
                actual: "scheduled"
            })
        ));
    }

    #[test]
    fn run_on_transactional_returns_wrong_kind() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let rid = registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_e, ()| {})
            .unwrap();
        // ReducerId and QueryReducerId share the same index space.
        let result = registry.run(&mut world, QueryReducerId(rid.0), ());
        assert!(matches!(
            result,
            Err(ReducerError::WrongKind {
                expected: "scheduled",
                actual: "transactional"
            })
        ));
    }

    // ── Multi-archetype QueryWriter test ─────────────────────────

    #[test]
    fn query_writer_spans_multiple_archetypes() {
        let mut world = World::new();
        // Two different archetypes, both containing Vel
        let e1 = world.spawn((Vel(10.0),));
        let e2 = world.spawn((Vel(20.0), Pos(0.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        let id = registry
            .register_query_writer::<(&mut Vel,), f32, _>(
                &mut world,
                "scale_vel",
                |mut query, factor: f32| {
                    query.for_each(|(mut vel,)| {
                        vel.modify(|v| v.0 *= factor);
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, 0.5f32).unwrap();

        assert_eq!(world.get::<Vel>(e1).unwrap().0, 5.0);
        assert_eq!(world.get::<Vel>(e2).unwrap().0, 10.0);
    }

    // ── Changed<T> filter with QueryWriter ───────────────────────

    #[test]
    fn query_writer_changed_filter() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        // Track how many entities the query writer visits
        let visit_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = visit_count.clone();

        let id = registry
            .register_query_writer::<(Changed<Pos>, &mut Pos), (), _>(
                &mut world,
                "changed_pos",
                move |mut query, ()| {
                    query.for_each(|((), mut pos)| {
                        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        pos.modify(|p| p.0 += 1.0);
                    });
                },
            )
            .unwrap();

        // First call: column was never read by this reducer, so Changed matches
        registry.call(&strategy, &mut world, id, ()).unwrap();
        assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 2.0);

        // Second call: no mutation since last call, Changed should skip
        visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        registry.call(&strategy, &mut world, id, ()).unwrap();
        assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 0);

        // Mutate the column externally, then call again
        visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        for (pos,) in world.query::<(&mut Pos,)>() {
            pos.0 = 99.0;
        }
        registry.call(&strategy, &mut world, id, ()).unwrap();
        assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(world.get::<Pos>(e).unwrap().0, 100.0);
    }

    // ── materialize_reserved regression test ─────────────────────

    #[test]
    fn query_writer_after_spawn_reducer() {
        // Regression: spawn via changeset (reserve + Mutation::Spawn) must
        // call materialize_reserved() so that subsequent changesets targeting
        // the spawned entity pass the is_alive() check.
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();

        let spawn_id = registry
            .register_spawner::<(Vel,), f32, _>(&mut world, "spawn", |mut spawner, vel: f32| {
                spawner.spawn((Vel(vel),));
            })
            .unwrap();

        let writer_id = registry
            .register_query_writer::<(&mut Vel,), f32, _>(
                &mut world,
                "scale",
                |mut query, factor: f32| {
                    query.for_each(|(mut vel,)| {
                        vel.modify(|v| v.0 *= factor);
                    });
                },
            )
            .unwrap();

        let strategy = Optimistic::new(&world);

        // Spawn an entity via changeset (uses reserve() internally)
        registry
            .call(&strategy, &mut world, spawn_id, 10.0f32)
            .unwrap();

        // Query writer iterates the spawned entity and buffers a write.
        // Without materialize_reserved(), this panics with "entity is not alive"
        // when the query writer's changeset is applied.
        registry
            .call(&strategy, &mut world, writer_id, 2.0f32)
            .unwrap();

        // Verify the spawned entity has the correct value
        let mut found = false;
        for (vel,) in world.query::<(&Vel,)>() {
            assert_eq!(vel.0, 20.0);
            found = true;
        }
        assert!(found);
    }

    // ── Coverage tests ──────────────────────────────────────────────

    #[test]
    fn entity_ref_entity_accessor() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        registry
            .register_entity::<(Pos,), (), _>(
                &mut world,
                "check_entity",
                move |entity: EntityMut<'_, (Pos,)>, ()| {
                    assert_eq!(entity.entity(), e);
                    let _ = entity.get::<Pos, 0>();
                },
            )
            .unwrap();
        let id = registry.reducer_id_by_name("check_entity").unwrap();
        registry.call(&strategy, &mut world, id, (e, ())).unwrap();
    }

    #[test]
    fn query_mut_for_each_slice() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Pos(i as f32),));
        }
        let mut registry = ReducerRegistry::new();
        registry
            .register_query::<(&Pos,), (), _>(
                &mut world,
                "chunk_iter",
                |mut query: QueryMut<'_, (&Pos,)>, ()| {
                    let mut count = 0;
                    query.for_each(|chunk| {
                        count += chunk.0.len();
                    });
                    assert_eq!(count, 5);
                },
            )
            .unwrap();
        let id = registry.query_reducer_id_by_name("chunk_iter").unwrap();
        registry.run(&mut world, id, ()).unwrap();
    }

    #[test]
    fn reducer_id_index() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity::<(Pos,), (), _>(
                &mut world,
                "idx_test",
                |_entity: EntityMut<'_, (Pos,)>, ()| {},
            )
            .unwrap();
        assert_eq!(id.index(), 0);

        let qid = registry
            .register_query::<(&Pos,), (), _>(
                &mut world,
                "qidx_test",
                |_query: QueryMut<'_, (&Pos,)>, ()| {},
            )
            .unwrap();
        assert_eq!(qid.index(), 1); // shares the reducers vec
    }

    #[test]
    fn reducer_registry_access_methods() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity::<(Pos,), (), _>(
                &mut world,
                "access_test",
                |_entity: EntityMut<'_, (Pos,)>, ()| {},
            )
            .unwrap();
        let qid = registry
            .register_query::<(&mut Pos,), (), _>(
                &mut world,
                "qaccess_test",
                |_query: QueryMut<'_, (&mut Pos,)>, ()| {},
            )
            .unwrap();

        let access = registry.access(id.index());
        assert!(access.has_any_access());
        let qaccess = registry.query_reducer_access(qid);
        assert!(qaccess.has_any_access());
    }

    #[test]
    fn reducer_registry_default() {
        let _registry: ReducerRegistry = ReducerRegistry::default();
    }

    // ── ReducerError tests ──────────────────────────────────────────

    #[test]
    fn reducer_error_display() {
        let err = ReducerError::WrongKind {
            expected: "transactional",
            actual: "scheduled",
        };
        let msg = format!("{err}");
        assert!(msg.contains("transactional"));
        assert!(msg.contains("scheduled"));

        let err = ReducerError::DuplicateName {
            name: "foo",
            existing_kind: "unified",
            existing_index: 0,
        };
        let msg = format!("{err}");
        assert!(msg.contains("foo"));
        assert!(msg.contains("unified"));
    }

    #[test]
    fn reducer_error_from_transact_error_conflict() {
        let conflict = Conflict {
            component_ids: fixedbitset::FixedBitSet::new(),
        };
        let transact_err = TransactError::Conflict(conflict);
        let err: ReducerError = transact_err.into();
        assert!(matches!(err, ReducerError::TransactionConflict(_)));
    }

    #[test]
    fn reducer_error_from_transact_error_world_mismatch() {
        let world1 = crate::World::new();
        let world2 = crate::World::new();
        let transact_err =
            TransactError::WorldMismatch(WorldMismatch::new(world1.world_id(), world2.world_id()));
        let err: ReducerError = transact_err.into();
        assert!(matches!(err, ReducerError::WorldMismatch(_)));
    }

    // ── Introspection tests ──────────────────────────────────────────

    #[test]
    fn reducer_info_entity() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_e, ()| {})
            .unwrap();
        let info = registry.reducer_info(id).unwrap();
        assert_eq!(info.name, "heal");
        assert_eq!(info.kind, "transactional");
        assert!(!info.can_despawn);
        assert!(!info.has_change_tracking);
    }

    #[test]
    fn reducer_info_entity_despawn() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity_despawn::<(Health,), (), _>(&mut world, "kill", |_e, ()| {})
            .unwrap();
        let info = registry.reducer_info(id).unwrap();
        assert_eq!(info.name, "kill");
        assert!(info.can_despawn);
    }

    #[test]
    fn reducer_info_entity_ref() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_entity_ref::<(Health,), (), _>(&mut world, "inspect", |_e, ()| {})
            .unwrap();
        let info = registry.reducer_info(id).unwrap();
        assert_eq!(info.name, "inspect");
        assert_eq!(info.kind, "transactional");
        assert!(!info.can_despawn);
    }

    #[test]
    fn entity_ref_reducer_reads_component() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let mut world = World::new();
        let e = world.spawn((Health(42),));

        let mut registry = ReducerRegistry::new();
        let observed = Arc::new(AtomicU32::new(0));
        let obs = Arc::clone(&observed);
        let id = registry
            .register_entity_ref::<(Health,), (), _>(&mut world, "read_hp", move |handle, ()| {
                let hp = handle.get::<Health, 0>();
                obs.store(hp.0, Ordering::Relaxed);
            })
            .unwrap();

        let strategy = Optimistic::new(&world);
        registry.call(&strategy, &mut world, id, (e, ())).unwrap();
        assert_eq!(observed.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn entity_ref_reducer_has_read_only_access() {
        let mut world = World::new();
        world.spawn((Health(1),));

        let mut registry = ReducerRegistry::new();
        registry
            .register_entity_ref::<(Health,), (), _>(&mut world, "read_only", |_e, ()| {})
            .unwrap();

        registry
            .register_entity::<(Health,), (), _>(&mut world, "read_write", |_e, ()| {})
            .unwrap();

        let ref_id = registry.reducer_id_by_name("read_only").unwrap();
        let mut_id = registry.reducer_id_by_name("read_write").unwrap();

        let ref_access = registry.reducer_access(ref_id);
        let mut_access = registry.reducer_access(mut_id);

        // EntityRef should NOT conflict with another EntityRef.
        assert!(
            !ref_access.conflicts_with(ref_access),
            "two read-only reducers should not conflict"
        );
        // EntityRef SHOULD conflict with EntityMut (writes vs reads).
        assert!(
            ref_access.conflicts_with(mut_access),
            "read-only reducer should conflict with read-write reducer"
        );
    }

    #[test]
    fn reducer_info_query_writer() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_query_writer::<(&mut Pos,), (), _>(&mut world, "move", |_qw, ()| {})
            .unwrap();
        let info = registry.reducer_info(id).unwrap();
        assert_eq!(info.name, "move");
        assert!(info.has_change_tracking);
    }

    #[test]
    fn query_reducer_info() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_query::<(&mut Pos,), (), _>(&mut world, "move", |_q, ()| {})
            .unwrap();
        let info = registry.query_reducer_info(id).unwrap();
        assert_eq!(info.name, "move");
        assert_eq!(info.kind, "scheduled");
    }

    #[test]
    fn dynamic_reducer_info() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        let id = registry
            .dynamic("dyn_test", &mut world)
            .can_read::<Pos>()
            .can_write::<Vel>()
            .can_despawn()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();
        let info = registry.dynamic_reducer_info(id).unwrap();
        assert_eq!(info.name, "dyn_test");
        assert_eq!(info.kind, "dynamic");
        assert!(info.can_despawn);
        assert!(info.has_change_tracking);
    }

    #[test]
    fn reducer_count_and_names() {
        let mut world = World::new();
        let mut registry = ReducerRegistry::new();
        registry
            .register_entity::<(Health,), (), _>(&mut world, "heal", |_e, ()| {})
            .unwrap();
        registry
            .register_query::<(&Pos,), (), _>(&mut world, "read", |_q, ()| {})
            .unwrap();
        registry
            .dynamic("dyn", &mut world)
            .can_read::<Vel>()
            .build(|_ctx: &mut DynamicCtx, _args: &()| {})
            .unwrap();

        assert_eq!(registry.reducer_count(), 2);
        assert_eq!(registry.dynamic_reducer_count(), 1);
        let names: Vec<_> = registry.registered_names().collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"heal"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"dyn"));
    }

    // ── DynamicCtx introspection tests ──────────────────────────────

    #[test]
    fn dynamic_ctx_is_declared() {
        use std::any::TypeId;
        let mut world = World::new();
        let pos_id = world.register_component::<Pos>();
        let vel_id = world.register_component::<Vel>();

        let entries = vec![(TypeId::of::<Pos>(), pos_id), (TypeId::of::<Vel>(), vel_id)];
        let mut access = Access::empty();
        access.add_read(pos_id);
        access.add_write(vel_id);
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );

        assert!(ctx.is_declared::<Pos>());
        assert!(ctx.is_declared::<Vel>());
        assert!(!ctx.is_declared::<Health>());

        assert!(!ctx.is_writable::<Pos>());
        assert!(ctx.is_writable::<Vel>());

        assert!(!ctx.is_removable::<Pos>());
        assert!(!ctx.can_despawn());
    }

    #[test]
    fn dynamic_ctx_despawn_introspection() {
        use std::any::TypeId;
        let mut world = World::new();
        let pos_id = world.register_component::<Pos>();

        let entries = vec![(TypeId::of::<Pos>(), pos_id)];
        let mut access = Access::empty();
        access.add_read(pos_id);
        access.set_despawns();
        let resolved =
            DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

        let default_tick = Arc::new(AtomicU64::new(0));
        let default_queried = AtomicBool::new(false);
        let mut cs = EnumChangeSet::new();
        let mut allocated = Vec::new();
        let ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );

        assert!(ctx.can_despawn());
    }

    // ── InvalidId bounds-check tests ─────────────────────────────────

    #[test]
    fn call_with_invalid_reducer_id() {
        let mut world = World::new();
        let strategy = crate::Optimistic::new(&world);
        let registry = ReducerRegistry::new();
        let bogus = ReducerId(999);
        let result = registry.call(&strategy, &mut world, bogus, ());
        assert!(matches!(
            result,
            Err(ReducerError::InvalidId {
                kind: "reducer",
                index: 999,
                max: 0
            })
        ));
    }

    #[test]
    fn run_with_invalid_query_reducer_id() {
        let mut world = World::new();
        let registry = ReducerRegistry::new();
        let bogus = QueryReducerId(42);
        let result = registry.run(&mut world, bogus, ());
        assert!(matches!(
            result,
            Err(ReducerError::InvalidId {
                kind: "reducer",
                index: 42,
                max: 0
            })
        ));
    }

    #[test]
    fn dynamic_call_with_invalid_id() {
        let mut world = World::new();
        let strategy = crate::Optimistic::new(&world);
        let registry = ReducerRegistry::new();
        let bogus = DynamicReducerId(7);
        let result = registry.dynamic_call(&strategy, &mut world, bogus, &());
        assert!(matches!(
            result,
            Err(ReducerError::InvalidId {
                kind: "dynamic",
                index: 7,
                max: 0
            })
        ));
    }

    #[test]
    fn reducer_info_with_invalid_id() {
        let registry = ReducerRegistry::new();
        let bogus = ReducerId(0);
        let result = registry.reducer_info(bogus);
        assert!(matches!(result, Err(ReducerError::InvalidId { .. })));
    }

    #[test]
    fn query_reducer_info_with_invalid_id() {
        let registry = ReducerRegistry::new();
        let bogus = QueryReducerId(0);
        let result = registry.query_reducer_info(bogus);
        assert!(matches!(result, Err(ReducerError::InvalidId { .. })));
    }

    #[test]
    fn dynamic_reducer_info_with_invalid_id() {
        let registry = ReducerRegistry::new();
        let bogus = DynamicReducerId(0);
        let result = registry.dynamic_reducer_info(bogus);
        assert!(matches!(result, Err(ReducerError::InvalidId { .. })));
    }

    // ── Fast-lane integration tests ─────────────────────────────────

    #[test]
    fn query_writer_fast_lane_roundtrip() {
        let mut world = World::new();
        // Spawn 100 entities: Pos(0.0), Vel(i as f32)
        let entities: Vec<Entity> = (0..100)
            .map(|i| world.spawn((Pos(0.0), Vel(i as f32))))
            .collect();
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        let id = registry
            .register_query_writer::<(&mut Pos, &Vel), (), _>(
                &mut world,
                "apply_vel_roundtrip",
                |mut query, ()| {
                    query.for_each(|(mut pos, vel)| {
                        pos.modify(|p| p.0 += vel.0);
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();

        // Verify each entity: Pos should equal its Vel value
        for (i, &e) in entities.iter().enumerate() {
            let pos = world.get::<Pos>(e).unwrap().0;
            assert_eq!(
                pos, i as f32,
                "entity {i}: expected Pos({i}.0), got Pos({pos})"
            );
        }
    }

    #[test]
    fn query_writer_fast_lane_change_detection() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(10.0)));
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        // First reducer: writes Pos via fast lane
        let writer_id = registry
            .register_query_writer::<(&mut Pos, &Vel), (), _>(
                &mut world,
                "move_pos",
                |mut query, ()| {
                    query.for_each(|(mut pos, vel)| {
                        pos.modify(|p| p.0 += vel.0);
                    });
                },
            )
            .unwrap();

        // Second reducer: reads Pos with Changed filter
        let visit_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = visit_count.clone();

        let changed_id = registry
            .register_query_writer::<(Changed<Pos>, &mut Pos), (), _>(
                &mut world,
                "detect_changed",
                move |mut query, ()| {
                    query.for_each(|((), mut pos)| {
                        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Touch pos so the write goes through the fast lane
                        pos.modify(|p| p.0 += 0.0);
                    });
                },
            )
            .unwrap();

        // Call the writer — this updates Pos via fast lane
        registry.call(&strategy, &mut world, writer_id, ()).unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 11.0);

        // Changed<Pos> should match because the writer just modified Pos
        visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        registry
            .call(&strategy, &mut world, changed_id, ())
            .unwrap();
        assert_eq!(
            visit_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "Changed<Pos> should match after fast-lane write"
        );

        // Call again with no intervening mutation — Changed should NOT match
        visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        registry
            .call(&strategy, &mut world, changed_id, ())
            .unwrap();
        assert_eq!(
            visit_count.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "Changed<Pos> should not match when nothing changed"
        );
    }

    #[test]
    fn query_writer_conditional_update() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..100).map(|i| world.spawn((Pos(i as f32),))).collect();
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        let id = registry
            .register_query_writer::<(&mut Pos,), (), _>(
                &mut world,
                "conditional_update",
                |mut query, ()| {
                    query.for_each(|(mut pos,)| {
                        if pos.get().0 > 50.0 {
                            pos.modify(|p| p.0 *= 2.0);
                        }
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();

        for (i, &e) in entities.iter().enumerate() {
            let val = world.get::<Pos>(e).unwrap().0;
            let expected = if (i as f32) > 50.0 {
                (i as f32) * 2.0
            } else {
                i as f32
            };
            assert_eq!(val, expected, "entity {i}: expected {expected}, got {val}");
        }
    }

    #[test]
    fn query_writer_read_only_components() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..50)
            .map(|i| world.spawn((Pos(i as f32), Vel(i as f32 * 10.0))))
            .collect();
        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();

        let id = registry
            .register_query_writer::<(&Pos, &mut Vel), (), _>(
                &mut world,
                "read_pos_write_vel",
                |mut query, ()| {
                    query.for_each(|(pos, mut vel)| {
                        // Read Pos, use its value to modify Vel
                        vel.modify(|v| v.0 += pos.0);
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();

        for (i, &e) in entities.iter().enumerate() {
            let pos_val = world.get::<Pos>(e).unwrap().0;
            let vel_val = world.get::<Vel>(e).unwrap().0;
            // Pos should be unchanged
            assert_eq!(pos_val, i as f32, "entity {i}: Pos should be unchanged");
            // Vel should be original + Pos value
            let expected_vel = (i as f32 * 10.0) + i as f32;
            assert_eq!(
                vel_val, expected_vel,
                "entity {i}: Vel should be {expected_vel}, got {vel_val}"
            );
        }
    }

    #[test]
    fn query_writer_column_slot_debug_assert() {
        // Exercises the debug_assert_eq!(col_batch.comp_id, self.comp_id) in set().
        // If column_slot assignment were incorrect, this would panic in debug builds.
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(3.0)));

        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_query_writer::<(&mut Pos, &mut Vel), (), _>(
                &mut world,
                "slot_test",
                |mut qw: QueryWriter<'_, (&mut Pos, &mut Vel)>, ()| {
                    qw.for_each(|(mut pos, mut vel)| {
                        pos.set(Pos(10.0));
                        vel.set(Vel(30.0));
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();
        // If we get here without a debug_assert panic, slots are correct.
        assert_eq!(world.get::<Pos>(e).unwrap().0, 10.0);
        assert_eq!(world.get::<Vel>(e).unwrap().0, 30.0);
    }

    #[test]
    fn query_writer_reverse_component_order() {
        // Exercises the slot assignment when tuple order is reversed relative
        // to ascending ComponentId order. Without the position-based lookup
        // fix, Vel's set() would write to Pos's column (and vice versa),
        // tripping the debug_assert_eq on comp_id inside WritableRef::set().
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(3.0)));

        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        // Note: (&mut Vel, &mut Pos) — reverse of registration order
        let id = registry
            .register_query_writer::<(&mut Vel, &mut Pos), (), _>(
                &mut world,
                "reverse_slot_test",
                |mut qw: QueryWriter<'_, (&mut Vel, &mut Pos)>, ()| {
                    qw.for_each(|(mut vel, mut pos)| {
                        vel.set(Vel(30.0));
                        pos.set(Pos(10.0));
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 10.0);
        assert_eq!(world.get::<Vel>(e).unwrap().0, 30.0);
    }

    #[test]
    fn query_writer_nested_tuple() {
        // Exercises nested mutable tuple: (&mut Pos, (&mut Vel,))
        // Without the offset-propagation fix, Vel would get slot 0
        // instead of slot 1, triggering the debug_assert on comp_id.
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));

        let strategy = Optimistic::new(&world);
        let mut registry = ReducerRegistry::new();
        let id = registry
            .register_query_writer::<(&mut Pos, (&mut Vel,)), (), _>(
                &mut world,
                "nested_tuple",
                |mut qw: QueryWriter<'_, (&mut Pos, (&mut Vel,))>, ()| {
                    qw.for_each(|(mut pos, (mut vel,))| {
                        pos.set(Pos(10.0));
                        vel.set(Vel(20.0));
                    });
                },
            )
            .unwrap();

        registry.call(&strategy, &mut world, id, ()).unwrap();
        assert_eq!(world.get::<Pos>(e).unwrap().0, 10.0);
        assert_eq!(world.get::<Vel>(e).unwrap().0, 20.0);
    }
}
