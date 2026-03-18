use std::any::TypeId;
use std::collections::{HashMap, HashSet};

use crate::sync::{Arc, AtomicBool, AtomicU64, Ordering};

use crate::access::Access;
use crate::bundle::Bundle;
use crate::changeset::EnumChangeSet;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::query::fetch::ReadOnlyWorldQuery;
use crate::tick::Tick;
use crate::world::World;

use super::{DynamicReducerEntry, ReducerError, ReducerRegistry, ReducerSlot};

// ── ComponentSet & Contains ──────────────────────────────────────────

/// Declares a set of component types with pre-resolved IDs.
///
/// Macro-generated for tuples of 1-12 `Component` types.
/// Used as the type parameter `C` on [`EntityRef<C>`](super::EntityRef) and
/// [`EntityMut<C>`](super::EntityMut) to constrain which components the handle can access.
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

// ── DynamicResolved ─────────────────────────────────────────────────

/// Pre-resolved component lookup for dynamic reducers.
/// Uses an identity hasher since `TypeId` is already well-distributed.
pub(super) struct DynamicResolved {
    entries: HashMap<TypeId, ComponentId, crate::component::TypeIdBuildHasher>,
    /// All declared ComponentIds for fast membership checks.
    comp_ids: HashSet<ComponentId>,
    access: Access,
    spawn_bundles: HashSet<TypeId>,
    remove_ids: HashSet<TypeId>,
}

impl DynamicResolved {
    pub(super) fn new(
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
    pub(super) fn lookup<T: 'static>(&self) -> Option<ComponentId> {
        self.entries.get(&TypeId::of::<T>()).copied()
    }

    pub(super) fn access(&self) -> &Access {
        &self.access
    }

    pub(super) fn has_spawn_bundle<B: 'static>(&self) -> bool {
        self.spawn_bundles.contains(&TypeId::of::<B>())
    }

    pub(super) fn has_remove<T: 'static>(&self) -> bool {
        self.remove_ids.contains(&TypeId::of::<T>())
    }

    /// Check if a ComponentId is in the declared set.
    pub(super) fn contains_comp_id(&self, comp_id: ComponentId) -> bool {
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
    pub(super) fn new(
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

// ── DynamicReducerBuilder ────────────────────────────────────────────

/// Type-erased dynamic reducer adapter.
pub(super) type DynamicAdapter = Box<dyn Fn(&mut DynamicCtx, &dyn std::any::Any) + Send + Sync>;

/// Typed handle for dispatching dynamic reducers via [`ReducerRegistry::dynamic_call`].
///
/// Obtained from [`DynamicReducerBuilder::build`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DynamicReducerId(pub(super) usize);

impl DynamicReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

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
    pub(super) registry: &'a mut ReducerRegistry,
    pub(super) world: &'a mut World,
    pub(super) name: &'static str,
    pub(super) access: Access,
    pub(super) entries: Vec<(TypeId, ComponentId)>,
    pub(super) spawn_bundles: HashSet<TypeId>,
    pub(super) remove_ids: HashSet<TypeId>,
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
