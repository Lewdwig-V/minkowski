use std::alloc::Layout;
use std::ptr::NonNull;

use crate::bundle::Bundle;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::pool::{SharedPool, default_pool};
use crate::tick::Tick;
use crate::world::{EntityLocation, World, get_pair_mut};

/// Error returned by [`EnumChangeSet::apply`] when a mutation targets
/// an invalid entity.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ApplyError {
    /// Mutation targeted an entity that is no longer alive.
    DeadEntity(Entity),
    /// Spawn targeted an entity already placed in an archetype.
    AlreadyPlaced(Entity),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadEntity(e) => write!(f, "entity {:?} is not alive", e),
            Self::AlreadyPlaced(e) => write!(f, "entity {:?} is already placed", e),
        }
    }
}

impl std::error::Error for ApplyError {}

const ARENA_ALIGN: usize = 16;

/// Contiguous byte arena for component data. Mutations store integer offsets
/// into this arena, avoiding per-mutation heap allocation.
///
/// Backing memory is allocated with alignment 16 to satisfy alignment
/// requirements of any component type stored within.
pub(crate) struct Arena {
    data: NonNull<u8>,
    len: usize,
    capacity: usize,
    pool: SharedPool,
}

impl Arena {
    pub fn new(pool: SharedPool) -> Self {
        Self {
            data: NonNull::dangling(),
            len: 0,
            capacity: 0,
            pool,
        }
    }

    /// Pre-allocate at least `bytes` of arena capacity.
    pub fn reserve(&mut self, bytes: usize) {
        if bytes > self.capacity {
            self.grow(bytes);
        }
    }

    /// Copy `layout.size()` bytes from `src` into the arena.
    /// Returns the byte offset where data was written.
    #[inline]
    pub fn alloc(&mut self, src: *const u8, layout: Layout) -> usize {
        if layout.size() == 0 {
            return 0;
        }
        let align = layout.align();
        let offset = (self.len + align - 1) & !(align - 1);
        let new_len = offset + layout.size();

        if new_len > self.capacity {
            self.grow(new_len);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src, self.data.as_ptr().add(offset), layout.size());
        }
        self.len = new_len;
        offset
    }

    /// Get a raw pointer to data at the given offset.
    #[inline]
    pub fn get(&self, offset: usize) -> *const u8 {
        debug_assert!(
            offset <= self.len,
            "arena read at offset {offset} exceeds arena length {}",
            self.len
        );
        unsafe { self.data.as_ptr().add(offset) }
    }

    fn grow(&mut self, min_capacity: usize) {
        let new_capacity = (min_capacity * 2).max(64);
        let new_layout =
            Layout::from_size_align(new_capacity, ARENA_ALIGN).expect("invalid arena layout");

        // Always use alloc + copy + dealloc instead of realloc.
        // realloc may not preserve alignment > max_align_t (typically 16 bytes).
        // Same reasoning as BlobVec::grow — alignment guarantees are non-negotiable.
        let new_data = self
            .pool
            .allocate(new_layout)
            .unwrap_or_else(|_| std::alloc::handle_alloc_error(new_layout));
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.data.as_ptr(), new_data.as_ptr(), self.len);
                let old_layout = Layout::from_size_align(self.capacity, ARENA_ALIGN).unwrap();
                self.pool.deallocate(self.data, old_layout);
            }
        }
        self.data = new_data;
        self.capacity = new_capacity;
    }
}

// SAFETY: Arena owns its allocation exclusively. All mutation goes through
// &mut self (alloc, grow). Shared references (&Arena) only read. Same
// reasoning as BlobVec's Send + Sync impls.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Drop for Arena {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout =
                Layout::from_size_align(self.capacity, ARENA_ALIGN).expect("invalid arena layout");
            // SAFETY: self.data was returned by a prior call to pool.allocate
            // with this layout, and will not be used after this call.
            unsafe {
                self.pool.deallocate(self.data, layout);
            }
        }
    }
}

/// Tracks an arena-buffered value that needs its destructor run if the
/// changeset is dropped without being applied.
struct DropEntry {
    /// Byte offset within the arena where the owned value starts.
    offset: usize,
    /// Type-erased destructor (`drop_in_place::<T>`).
    drop_fn: unsafe fn(*mut u8),
    /// Index of the mutation that owns this value. Used to partition
    /// drop entries between processed (ownership transferred to World)
    /// and unprocessed (must be dropped on error) during `apply()`.
    mutation_idx: usize,
}

/// A single structural mutation recorded in a ChangeSet.
#[allow(dead_code)]
pub(crate) enum Mutation {
    Spawn {
        entity: Entity,
        /// (ComponentId, arena offset, Layout) per component.
        components: Vec<(ComponentId, usize, Layout)>,
    },
    Despawn {
        entity: Entity,
    },
    Insert {
        entity: Entity,
        component_id: ComponentId,
        offset: usize,
        layout: Layout,
    },
    Remove {
        entity: Entity,
        component_id: ComponentId,
    },
    /// Insert a component into sparse storage (not archetypes).
    SparseInsert {
        entity: Entity,
        component_id: ComponentId,
        offset: usize,
        layout: Layout,
    },
    /// Remove a component from sparse storage.
    SparseRemove {
        entity: Entity,
        component_id: ComponentId,
    },
}

/// Data-driven mutation buffer. Records structural mutations as an enum vec
/// with component bytes stored in a contiguous arena.
///
/// [`apply()`](EnumChangeSet::apply) executes all buffered mutations against a
/// [`World`], returning `Ok(())` on success or `Err(ApplyError)` if a
/// mutation targets a dead or already-placed entity.
///
/// Typed helpers — [`insert`](EnumChangeSet::insert), [`remove`](EnumChangeSet::remove),
/// [`spawn_bundle`](EnumChangeSet::spawn_bundle) — auto-register component types and
/// take ownership via `ManuallyDrop` (drop entries registered for cleanup).
/// Raw methods ([`record_insert`](EnumChangeSet::record_insert),
/// [`record_spawn`](EnumChangeSet::record_spawn)) copy bytes without taking ownership —
/// the caller remains responsible for the source data's lifetime.
///
/// Used internally by [`Tx`](crate::Tx) for transactional writes and by
/// `minkowski_persist::Durable` as the WAL serialization boundary.
/// For closure-based deferred mutations, see [`CommandBuffer`](crate::CommandBuffer).
///
/// Implements `Drop`: if discarded without calling `apply()`, destructors
/// are run for any values whose ownership was transferred via typed helpers.
pub struct EnumChangeSet {
    pub(crate) mutations: Vec<Mutation>,
    pub(crate) arena: Arena,
    /// Drop glue for arena-buffered values whose ownership was transferred
    /// from the caller via typed helpers. Cleared on `apply()` when ownership
    /// moves to the world; remaining entries are cleaned up on drop.
    drop_entries: Vec<DropEntry>,
}

impl EnumChangeSet {
    pub fn new() -> Self {
        Self::new_in(default_pool())
    }

    /// Create a changeset pre-allocated for `mutations` mutations, each up to
    /// `bytes_per_mutation` bytes of component data. Avoids reallocation during
    /// recording when the mutation count is known ahead of time.
    pub fn with_capacity(mutations: usize, bytes_per_mutation: usize) -> Self {
        let pool = default_pool();
        let mut arena = Arena::new(pool.clone());
        arena.reserve(mutations * bytes_per_mutation);
        Self {
            mutations: Vec::with_capacity(mutations),
            arena,
            drop_entries: Vec::new(),
        }
    }

    pub(crate) fn new_in(pool: SharedPool) -> Self {
        Self {
            mutations: Vec::new(),
            arena: Arena::new(pool),
            drop_entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Record a spawn: entity + raw component data.
    ///
    /// The entity must not be placed (`World::is_placed` returns false) when
    /// the changeset is applied — passing an already-placed entity corrupts
    /// archetype storage. Use `World::alloc_entity()` to obtain an unplaced
    /// handle. The `apply` method will panic if this precondition is violated.
    pub fn record_spawn(
        &mut self,
        entity: Entity,
        components: &[(ComponentId, *const u8, Layout)],
    ) {
        let stored: Vec<(ComponentId, usize, Layout)> = components
            .iter()
            .map(|&(id, ptr, layout)| {
                let offset = self.arena.alloc(ptr, layout);
                (id, offset, layout)
            })
            .collect();
        self.mutations.push(Mutation::Spawn {
            entity,
            components: stored,
        });
    }

    /// Record a despawn.
    pub fn record_despawn(&mut self, entity: Entity) {
        self.mutations.push(Mutation::Despawn { entity });
    }

    /// Record inserting a component on an entity. Returns the arena byte offset
    /// where the component data was stored.
    #[inline]
    pub fn record_insert(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        data: *const u8,
        layout: Layout,
    ) -> usize {
        let offset = self.arena.alloc(data, layout);
        self.mutations.push(Mutation::Insert {
            entity,
            component_id,
            offset,
            layout,
        });
        offset
    }

    /// Record removing a component from an entity.
    pub fn record_remove(&mut self, entity: Entity, component_id: ComponentId) {
        self.mutations.push(Mutation::Remove {
            entity,
            component_id,
        });
    }

    /// Record inserting a sparse component on an entity. Returns the arena byte
    /// offset where the component data was stored.
    pub fn record_sparse_insert(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        data: *const u8,
        layout: Layout,
    ) -> usize {
        let offset = self.arena.alloc(data, layout);
        self.mutations.push(Mutation::SparseInsert {
            entity,
            component_id,
            offset,
            layout,
        });
        offset
    }

    /// Record removing a sparse component from an entity.
    pub fn record_sparse_remove(&mut self, entity: Entity, component_id: ComponentId) {
        self.mutations.push(Mutation::SparseRemove {
            entity,
            component_id,
        });
    }
}

// ── Typed safe helpers ─────────────────────────────────────────

impl EnumChangeSet {
    /// Record inserting a component on an entity. Auto-registers the
    /// component type. Safe wrapper over `record_insert`.
    ///
    /// Takes ownership of `value` — the destructor will run either when the
    /// changeset is applied (ownership transfers to the world) or when the
    /// changeset is dropped without applying.
    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        let comp_id = world.register_component::<T>();
        let drop_fn = world.components.info(comp_id).drop_fn;
        let layout = Layout::new::<T>();
        let value = std::mem::ManuallyDrop::new(value);
        let offset = self.record_insert(entity, comp_id, &*value as *const T as *const u8, layout);
        if let Some(drop_fn) = drop_fn {
            let mutation_idx = self.mutations.len() - 1;
            self.drop_entries.push(DropEntry {
                offset,
                drop_fn,
                mutation_idx,
            });
        }
    }

    /// Record removing a component from an entity. Auto-registers the
    /// component type.
    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        let comp_id = world.register_component::<T>();
        self.record_remove(entity, comp_id);
    }

    /// Record inserting a sparse component on an entity. Auto-registers the
    /// component type.
    ///
    /// Takes ownership of `value` — the destructor will run either when the
    /// changeset is applied (ownership transfers to the world) or when the
    /// changeset is dropped without applying.
    pub fn insert_sparse<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        let comp_id = world.register_component::<T>();
        let drop_fn = world.components.info(comp_id).drop_fn;
        let layout = Layout::new::<T>();
        let value = std::mem::ManuallyDrop::new(value);
        let offset =
            self.record_sparse_insert(entity, comp_id, &*value as *const T as *const u8, layout);
        if let Some(drop_fn) = drop_fn {
            let mutation_idx = self.mutations.len() - 1;
            self.drop_entries.push(DropEntry {
                offset,
                drop_fn,
                mutation_idx,
            });
        }
    }

    /// Record removing a sparse component from an entity. Auto-registers the
    /// component type.
    pub fn remove_sparse<T: Component>(&mut self, world: &mut World, entity: Entity) {
        let comp_id = world.register_component::<T>();
        self.record_sparse_remove(entity, comp_id);
    }

    /// Record spawning an entity with a bundle of components. Auto-registers
    /// all component types in the bundle.
    ///
    /// # Panics
    /// Panics if `entity` is already alive. Use `World::alloc_entity()` to
    /// obtain an unplaced entity handle.
    pub fn spawn_bundle<B: Bundle>(&mut self, world: &mut World, entity: Entity, bundle: B) {
        assert!(
            !world.is_placed(entity),
            "spawn_bundle: entity {:?} is already placed in an archetype — \
             use World::alloc_entity() to obtain an unplaced handle",
            entity,
        );
        let _ids = B::component_ids(&mut world.components);
        let mut components = Vec::new();
        unsafe {
            bundle.put(&world.components, &mut |comp_id, ptr, layout| {
                let offset = self.arena.alloc(ptr, layout);
                components.push((comp_id, offset, layout));
            });
        }
        // Register drop entries for components that need cleanup.
        let mutation_idx = self.mutations.len(); // index of the Spawn we're about to push
        for &(comp_id, offset, _) in &components {
            if let Some(drop_fn) = world.components.info(comp_id).drop_fn {
                self.drop_entries.push(DropEntry {
                    offset,
                    drop_fn,
                    mutation_idx,
                });
            }
        }
        self.mutations.push(Mutation::Spawn { entity, components });
    }
}

// ── Pre-resolved helpers (pub(crate)) ────────────────────────────

impl EnumChangeSet {
    /// Insert with a pre-resolved ComponentId. Same as `insert()` but
    /// skips `world.register_component::<T>()`. Used by typed reducer handles.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn insert_raw<T: Component>(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
        value: T,
    ) {
        let layout = Layout::new::<T>();
        let value = std::mem::ManuallyDrop::new(value);
        let offset = self.record_insert(entity, comp_id, &*value as *const T as *const u8, layout);
        if std::mem::needs_drop::<T>() {
            let mutation_idx = self.mutations.len() - 1;
            self.drop_entries.push(DropEntry {
                offset,
                drop_fn: crate::component::drop_ptr::<T>,
                mutation_idx,
            });
        }
    }

    /// Sparse insert with a pre-resolved ComponentId. Same as `insert_sparse()`
    /// but skips `world.register_component::<T>()`. Used by typed reducer handles.
    #[allow(dead_code)]
    pub(crate) fn insert_sparse_raw<T: Component>(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
        value: T,
    ) {
        let layout = Layout::new::<T>();
        let value = std::mem::ManuallyDrop::new(value);
        let offset =
            self.record_sparse_insert(entity, comp_id, &*value as *const T as *const u8, layout);
        if std::mem::needs_drop::<T>() {
            let mutation_idx = self.mutations.len() - 1;
            self.drop_entries.push(DropEntry {
                offset,
                drop_fn: crate::component::drop_ptr::<T>,
                mutation_idx,
            });
        }
    }

    /// Sparse remove with a pre-resolved ComponentId.
    #[allow(dead_code)]
    pub(crate) fn remove_sparse_raw(&mut self, entity: Entity, comp_id: ComponentId) {
        self.record_sparse_remove(entity, comp_id);
    }

    /// Spawn a bundle with pre-resolved ComponentIds. Same as `spawn_bundle()`
    /// but takes `&ComponentRegistry` instead of `&mut World`. Used by Spawner.
    ///
    /// The caller must guarantee that `entity` is unplaced (e.g. via `reserve()`).
    #[allow(dead_code)]
    pub(crate) fn spawn_bundle_raw<B: Bundle>(
        &mut self,
        entity: Entity,
        registry: &ComponentRegistry,
        bundle: B,
    ) {
        let mut components = Vec::new();
        unsafe {
            bundle.put(registry, &mut |comp_id, ptr, layout| {
                let offset = self.arena.alloc(ptr, layout);
                components.push((comp_id, offset, layout));
            });
        }
        let mutation_idx = self.mutations.len();
        for &(comp_id, offset, _) in &components {
            if let Some(drop_fn) = registry.info(comp_id).drop_fn {
                self.drop_entries.push(DropEntry {
                    offset,
                    drop_fn,
                    mutation_idx,
                });
            }
        }
        self.mutations.push(Mutation::Spawn { entity, components });
    }
}

/// Read-only view of a mutation for serialization. Component data is
/// borrowed as byte slices from the changeset's Arena.
pub enum MutationRef<'a> {
    Spawn {
        entity: Entity,
        components: Vec<(ComponentId, &'a [u8])>,
    },
    Despawn {
        entity: Entity,
    },
    Insert {
        entity: Entity,
        component_id: ComponentId,
        data: &'a [u8],
    },
    Remove {
        entity: Entity,
        component_id: ComponentId,
    },
    /// Sparse component insert (not stored in archetypes).
    SparseInsert {
        entity: Entity,
        component_id: ComponentId,
        data: &'a [u8],
    },
    /// Sparse component removal.
    SparseRemove {
        entity: Entity,
        component_id: ComponentId,
    },
}

impl EnumChangeSet {
    /// Iterate mutations as borrowed views. Component data is returned as
    /// byte slices from the Arena.
    pub fn iter_mutations(&self) -> impl Iterator<Item = MutationRef<'_>> + '_ {
        self.mutations.iter().map(|m| match m {
            Mutation::Spawn { entity, components } => MutationRef::Spawn {
                entity: *entity,
                components: components
                    .iter()
                    .map(|(id, offset, layout)| {
                        let ptr = self.arena.get(*offset);
                        let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
                        (*id, bytes)
                    })
                    .collect(),
            },
            Mutation::Despawn { entity } => MutationRef::Despawn { entity: *entity },
            Mutation::Insert {
                entity,
                component_id,
                offset,
                layout,
            } => {
                let ptr = self.arena.get(*offset);
                let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
                MutationRef::Insert {
                    entity: *entity,
                    component_id: *component_id,
                    data: bytes,
                }
            }
            Mutation::Remove {
                entity,
                component_id,
            } => MutationRef::Remove {
                entity: *entity,
                component_id: *component_id,
            },
            Mutation::SparseInsert {
                entity,
                component_id,
                offset,
                layout,
            } => {
                let ptr = self.arena.get(*offset);
                let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
                MutationRef::SparseInsert {
                    entity: *entity,
                    component_id: *component_id,
                    data: bytes,
                }
            }
            Mutation::SparseRemove {
                entity,
                component_id,
            } => MutationRef::SparseRemove {
                entity: *entity,
                component_id: *component_id,
            },
        })
    }
}

impl Default for EnumChangeSet {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EnumChangeSet {
    fn drop(&mut self) {
        // Run destructors for any arena-buffered values that were never
        // transferred to the world via apply(). After apply(), drop_entries
        // is empty so this is a no-op.
        for entry in &self.drop_entries {
            unsafe {
                let ptr = self.arena.get(entry.offset) as *mut u8;
                (entry.drop_fn)(ptr);
            }
        }
    }
}

impl EnumChangeSet {
    /// Apply every mutation in this changeset to `world`.
    ///
    /// Returns `Ok(())` on success or `Err(ApplyError)` if a mutation targets
    /// an invalid entity. A single tick is allocated for the entire batch.
    pub fn apply(mut self, world: &mut World) -> Result<(), ApplyError> {
        let tick = world.next_tick();

        let result = self.apply_mutations(world, tick);

        match result {
            Ok(()) => {
                // All mutations succeeded — disarm all drop entries.
                // Ownership of arena-backed values has transferred to World.
                self.drop_entries.clear();
                Ok(())
            }
            Err((failed_idx, err)) => {
                // Mutations 0..failed_idx succeeded (ownership transferred to World).
                // Mutations failed_idx.. were not processed — their arena values
                // must be dropped. Retain only drop entries for unprocessed mutations
                // so the Drop impl cleans them up.
                self.drop_entries
                    .retain(|entry| entry.mutation_idx >= failed_idx);
                Err(err)
            }
        }
    }

    /// Process all mutations. Returns `Err((mutation_index, error))` on failure,
    /// where `mutation_index` is the index of the mutation that failed.
    fn apply_mutations(
        &self,
        world: &mut World,
        tick: crate::tick::Tick,
    ) -> Result<(), (usize, ApplyError)> {
        for (mutation_idx, mutation) in self.mutations.iter().enumerate() {
            let map_err = |e| (mutation_idx, e);
            match mutation {
                Mutation::Spawn { entity, components } => {
                    // Ensure the entity allocator's generations vec covers
                    // reserved indices — reserve() is lock-free and doesn't
                    // touch generations. Without this, is_alive() returns
                    // false for reserved-then-spawned entities.
                    world.entities.materialize_reserved();
                    if world.is_placed(*entity) {
                        return Err(map_err(ApplyError::AlreadyPlaced(*entity)));
                    }
                    // --- Apply: push raw component data into the right archetype ---
                    let sorted_ids: Vec<ComponentId> = {
                        let mut ids: Vec<_> = components.iter().map(|&(id, _, _)| id).collect();
                        ids.sort_unstable();
                        ids.dedup();
                        ids
                    };
                    let arch_id =
                        world
                            .archetypes
                            .get_or_create(&sorted_ids, &world.components, &world.pool);
                    let index = entity.index() as usize;

                    if index >= world.entity_locations.len() {
                        world.entity_locations.resize(index + 1, None);
                    }

                    let archetype = &mut world.archetypes.archetypes[arch_id.0];
                    for &(comp_id, offset, _) in components {
                        let src = self.arena.get(offset);
                        let col = archetype.column_index(comp_id).unwrap();
                        unsafe {
                            archetype.columns[col].push(src as *mut u8);
                        }
                        archetype.columns[col].mark_changed(tick);
                    }
                    let row = archetype.entities.len();
                    archetype.entities.push(*entity);
                    archetype.debug_assert_consistent();

                    world.entity_locations[index] = Some(EntityLocation {
                        archetype_id: arch_id,
                        row,
                    });
                }

                Mutation::Despawn { entity } => {
                    if !world.despawn(*entity) {
                        return Err(map_err(ApplyError::DeadEntity(*entity)));
                    }
                }

                Mutation::Insert {
                    entity,
                    component_id,
                    offset,
                    layout,
                } => {
                    let data_ptr = self.arena.get(*offset);
                    changeset_insert_raw(world, *entity, *component_id, data_ptr, *layout, tick)
                        .map_err(map_err)?;
                }

                Mutation::Remove {
                    entity,
                    component_id,
                } => {
                    changeset_remove_raw(world, *entity, *component_id, tick).map_err(map_err)?;
                }

                Mutation::SparseInsert {
                    entity,
                    component_id,
                    offset,
                    layout,
                } => {
                    if !world.is_alive(*entity) {
                        return Err(map_err(ApplyError::DeadEntity(*entity)));
                    }
                    let data_ptr = self.arena.get(*offset);
                    let info = world.components.info(*component_id);
                    let drop_fn = info.drop_fn;

                    // Ensure the component is marked as sparse in the registry
                    // (needed for world.get/has routing after WAL replay).
                    world.components.mark_sparse(*component_id);

                    // insert_raw handles both first-insert and overwrite
                    // (drops old value on overwrite).
                    unsafe {
                        world
                            .sparse
                            .insert_raw(*component_id, *entity, data_ptr, *layout, drop_fn);
                    }
                }

                Mutation::SparseRemove {
                    entity,
                    component_id,
                } => {
                    if !world.is_alive(*entity) {
                        return Err(map_err(ApplyError::DeadEntity(*entity)));
                    }
                    // Remove with drop — no reverse to own the old bytes.
                    world.sparse.remove_raw(*component_id, *entity);
                }
            }
        }

        Ok(())
    }
}

/// Raw insert: either overwrites an existing component in-place or performs
/// archetype migration to add a new component.
fn changeset_insert_raw(
    world: &mut World,
    entity: Entity,
    comp_id: ComponentId,
    data_ptr: *const u8,
    layout: Layout,
    tick: Tick,
) -> Result<(), ApplyError> {
    if !world.is_alive(entity) {
        return Err(ApplyError::DeadEntity(entity));
    }
    let index = entity.index() as usize;
    let location = world.entity_locations[index].unwrap();

    let src_arch = &world.archetypes.archetypes[location.archetype_id.0];

    if src_arch.component_ids.contains(comp_id) {
        // Entity already has this component — overwrite in-place.
        let col_idx = src_arch.column_index(comp_id).unwrap();

        // Overwrite with new data (write path — marks column changed).
        let src_arch = &mut world.archetypes.archetypes[location.archetype_id.0];
        unsafe {
            let dst = src_arch.columns[col_idx].get_ptr_mut(location.row, tick);
            let info = world.components.info(comp_id);
            if let Some(drop_fn) = info.drop_fn {
                drop_fn(dst);
            }
            std::ptr::copy_nonoverlapping(data_ptr, dst, layout.size());
        }
    } else {
        // Archetype migration: source components + new component.
        let src_arch = &world.archetypes.archetypes[location.archetype_id.0];
        let mut target_ids = src_arch.sorted_ids.clone();
        target_ids.push(comp_id);
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id =
            world
                .archetypes
                .get_or_create(&target_ids, &world.components, &world.pool);

        let (src_arch, target_arch) = get_pair_mut(
            &mut world.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns from source to target.
        for (src_col, &cid) in src_arch.sorted_ids.iter().enumerate() {
            if let Some(tgt_col) = target_arch.column_index(cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        // Write the new component into target.
        let tgt_col = target_arch.column_index(comp_id).unwrap();
        unsafe {
            target_arch.columns[tgt_col].push(data_ptr as *mut u8);
        }

        // Mark all target columns as changed.
        for col in &mut target_arch.columns {
            col.mark_changed(tick);
        }

        // Move entity tracking.
        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        // Update swapped entity's location in source.
        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            world.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        // Verify column/entity invariant after migration.
        src_arch.debug_assert_consistent();
        target_arch.debug_assert_consistent();

        world.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });
    }

    Ok(())
}

/// Raw remove: performs archetype migration to remove a component, dropping
/// the old value.
fn changeset_remove_raw(
    world: &mut World,
    entity: Entity,
    comp_id: ComponentId,
    tick: Tick,
) -> Result<(), ApplyError> {
    if !world.is_alive(entity) {
        return Err(ApplyError::DeadEntity(entity));
    }
    let index = entity.index() as usize;
    let location = world.entity_locations[index].unwrap();
    let src_arch = &world.archetypes.archetypes[location.archetype_id.0];

    if !src_arch.component_ids.contains(comp_id) {
        return Ok(()); // Component not present — nothing to do.
    }

    // Compute target archetype: source components minus this one.
    let target_ids: Vec<ComponentId> = src_arch
        .sorted_ids
        .iter()
        .copied()
        .filter(|&id| id != comp_id)
        .collect();
    let src_arch_id = location.archetype_id;
    let src_row = location.row;

    if target_ids.is_empty() {
        // Entity has no components left — move to empty archetype.
        let arch = &mut world.archetypes.archetypes[src_arch_id.0];
        // swap_remove with drop for all columns (including the removed one).
        for col_idx_inner in 0..arch.columns.len() {
            unsafe {
                arch.columns[col_idx_inner].swap_remove(src_row);
            }
        }
        arch.entities.swap_remove(src_row);
        if src_row < arch.entities.len() {
            let swapped = arch.entities[src_row];
            world.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }
        arch.debug_assert_consistent();
        let empty_arch_id = world
            .archetypes
            .get_or_create(&[], &world.components, &world.pool);
        let empty_arch = &mut world.archetypes.archetypes[empty_arch_id.0];
        empty_arch.entities.push(entity);
        world.entity_locations[index] = Some(EntityLocation {
            archetype_id: empty_arch_id,
            row: empty_arch.entities.len() - 1,
        });
        return Ok(());
    }

    let target_arch_id =
        world
            .archetypes
            .get_or_create(&target_ids, &world.components, &world.pool);

    let (src_arch, target_arch) = get_pair_mut(
        &mut world.archetypes.archetypes,
        src_arch_id.0,
        target_arch_id.0,
    );

    // Move shared columns (skip removed component — drop it).
    for (src_col, &cid) in src_arch.sorted_ids.iter().enumerate() {
        if cid == comp_id {
            // Drop the removed component's data.
            unsafe {
                src_arch.columns[src_col].swap_remove(src_row);
            }
        } else if let Some(tgt_col) = target_arch.column_index(cid) {
            unsafe {
                let ptr = src_arch.columns[src_col].get_ptr(src_row);
                target_arch.columns[tgt_col].push(ptr);
                src_arch.columns[src_col].swap_remove_no_drop(src_row);
            }
        }
    }

    // Mark target columns as changed.
    for col in &mut target_arch.columns {
        col.mark_changed(tick);
    }

    target_arch.entities.push(entity);
    let target_row = target_arch.entities.len() - 1;
    src_arch.entities.swap_remove(src_row);

    if src_row < src_arch.entities.len() {
        let swapped = src_arch.entities[src_row];
        world.entity_locations[swapped.index() as usize] = Some(EntityLocation {
            archetype_id: src_arch_id,
            row: src_row,
        });
    }

    // Verify column/entity invariant after migration.
    src_arch.debug_assert_consistent();
    target_arch.debug_assert_consistent();

    world.entity_locations[index] = Some(EntityLocation {
        archetype_id: target_arch_id,
        row: target_row,
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_read_back() {
        let mut arena = Arena::new(default_pool());
        let value: u32 = 42;
        let layout = Layout::new::<u32>();
        let offset = arena.alloc(&value as *const u32 as *const u8, layout);
        let ptr = arena.get(offset) as *const u32;
        assert_eq!(unsafe { *ptr }, 42);
    }

    #[test]
    fn arena_alignment() {
        let mut arena = Arena::new(default_pool());
        let byte: u8 = 0xFF;
        let _ = arena.alloc(&byte as *const u8, Layout::new::<u8>());

        let val: u64 = 123456789;
        let offset = arena.alloc(&val as *const u64 as *const u8, Layout::new::<u64>());
        assert_eq!(offset % 8, 0, "u64 offset must be 8-byte aligned");
    }

    #[test]
    fn arena_zst() {
        let mut arena = Arena::new(default_pool());
        let layout = Layout::new::<()>();
        let offset = arena.alloc(std::ptr::null(), layout);
        assert_eq!(offset, 0);
    }

    #[test]
    fn record_and_count() {
        let mut cs = EnumChangeSet::new();
        let e = Entity::new(0, 0);
        cs.record_despawn(e);
        cs.record_remove(e, 1);
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn record_insert_stores_data() {
        let mut cs = EnumChangeSet::new();
        let e = Entity::new(0, 0);
        let value: u32 = 99;
        cs.record_insert(
            e,
            0,
            &value as *const u32 as *const u8,
            Layout::new::<u32>(),
        );
        assert_eq!(cs.len(), 1);
        if let Mutation::Insert { offset, .. } = &cs.mutations[0] {
            let ptr = cs.arena.get(*offset) as *const u32;
            assert_eq!(unsafe { *ptr }, 99);
        } else {
            panic!("expected Insert mutation");
        }
    }

    #[test]
    fn record_spawn_stores_components() {
        let mut cs = EnumChangeSet::new();
        let e = Entity::new(0, 0);
        let a: u32 = 1;
        let b: u64 = 2;
        let components = vec![
            (0, &a as *const u32 as *const u8, Layout::new::<u32>()),
            (1, &b as *const u64 as *const u8, Layout::new::<u64>()),
        ];
        cs.record_spawn(e, &components);
        assert_eq!(cs.len(), 1);
        if let Mutation::Spawn { components, .. } = &cs.mutations[0] {
            assert_eq!(components.len(), 2);
        } else {
            panic!("expected Spawn mutation");
        }
    }

    #[test]
    fn iter_mutations_returns_correct_views() {
        use crate::changeset::MutationRef;
        use crate::world::World;

        let mut world = World::new();
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.insert::<f32>(&mut world, e, 42.0f32);
        cs.record_despawn(Entity::new(99, 0));

        let views: Vec<_> = cs.iter_mutations().collect();
        assert_eq!(views.len(), 2);
        assert!(matches!(views[0], MutationRef::Insert { .. }));
        assert!(matches!(views[1], MutationRef::Despawn { .. }));

        if let MutationRef::Insert { data, .. } = &views[0] {
            assert_eq!(data.len(), std::mem::size_of::<f32>());
            let val = unsafe { *(data.as_ptr() as *const f32) };
            assert_eq!(val, 42.0);
        }
    }

    // ── apply + reverse tests ──────────────────────────────────

    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn apply_spawn() {
        let mut world = World::new();
        let entity = world.alloc_entity();

        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            entity,
            (Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }),
        );

        cs.apply(&mut world).unwrap();
        assert!(world.is_alive(entity));
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_despawn() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 5.0, y: 6.0 }, Vel { dx: 7.0, dy: 8.0 }));

        let mut cs = EnumChangeSet::new();
        cs.record_despawn(e);
        cs.apply(&mut world).unwrap();
        assert!(!world.is_alive(e));
    }

    #[test]
    fn apply_insert_new() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_insert_overwrite() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Pos>(&mut world, e, Pos { x: 99.0, y: 99.0 });

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 99.0, y: 99.0 }));
    }

    #[test]
    fn apply_remove() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let mut cs = EnumChangeSet::new();
        cs.remove::<Vel>(&mut world, e);

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_empty_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let cs = EnumChangeSet::new();
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    // ── typed helper tests ────────────────────────────────────────

    #[test]
    fn typed_insert_and_apply() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn typed_insert_overwrite() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Pos>(&mut world, e, Pos { x: 99.0, y: 99.0 });

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 99.0, y: 99.0 }));
    }

    #[test]
    fn typed_spawn_and_apply() {
        let mut world = World::new();
        let entity = world.alloc_entity();

        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            entity,
            (Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }),
        );

        cs.apply(&mut world).unwrap();
        assert!(world.is_alive(entity));
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(entity), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn typed_remove() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let mut cs = EnumChangeSet::new();
        cs.remove::<Vel>(&mut world, e);

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    #[should_panic(expected = "already placed")]
    fn spawn_bundle_panics_on_live_entity() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Vel { dx: 3.0, dy: 4.0 },));
    }

    #[test]
    fn apply_spawn_returns_error_on_live_entity() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        // Use raw record_spawn to bypass the spawn_bundle guard,
        // then verify apply() returns AlreadyPlaced error.
        let pos_id = world.register_component::<Vel>();
        let vel = Vel { dx: 3.0, dy: 4.0 };
        let mut cs = EnumChangeSet::new();
        cs.record_spawn(
            e,
            &[(
                pos_id,
                &vel as *const Vel as *const u8,
                Layout::new::<Vel>(),
            )],
        );
        let result = cs.apply(&mut world);
        assert!(matches!(result, Err(ApplyError::AlreadyPlaced(_))));
    }

    // ── Drop safety tests ─────────────────────────────────────────

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A non-Copy component that counts destructor calls via a shared counter.
    /// Each test creates its own counter, avoiding races with parallel tests.
    #[derive(Debug)]
    struct Tracked {
        #[allow(dead_code)]
        value: u32,
        counter: Arc<AtomicUsize>,
    }

    impl Tracked {
        fn new(value: u32, counter: &Arc<AtomicUsize>) -> Self {
            Self {
                value,
                counter: counter.clone(),
            }
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn drop_runs_destructor_for_unapplied_insert() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        {
            let mut cs = EnumChangeSet::new();
            cs.insert::<Tracked>(&mut world, e, Tracked::new(42, &drops));
            // cs dropped here without apply()
        }

        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "destructor should run on changeset drop"
        );
    }

    #[test]
    fn drop_runs_destructor_for_unapplied_spawn_bundle() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.alloc_entity();

        {
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Tracked::new(42, &drops),));
            // cs dropped here without apply()
        }

        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "destructor should run on changeset drop"
        );
    }

    #[test]
    fn apply_does_not_double_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        {
            let mut cs = EnumChangeSet::new();
            cs.insert::<Tracked>(&mut world, e, Tracked::new(42, &drops));
            cs.apply(&mut world).unwrap();
            // cs consumed by apply — should not double-drop
        }

        // Value is now owned by the world; no spurious drops from changeset.
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "no drops yet — value owned by world"
        );

        world.despawn(e);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "one drop from despawn");
    }

    #[test]
    fn remove_drops_value() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Tracked::new(99, &drops)));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "spawn transfers ownership, no drop"
        );

        let mut cs = EnumChangeSet::new();
        cs.remove::<Tracked>(&mut world, e);
        cs.apply(&mut world).unwrap();

        // Remove now drops the value directly (no reverse changeset).
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "remove should drop the value"
        );
    }

    // ── insert_raw / spawn_bundle_raw tests ──────────────────────

    #[test]
    fn insert_raw_and_apply() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let vel_id = world.register_component::<Vel>();

        let mut cs = EnumChangeSet::new();
        cs.insert_raw::<Vel>(e, vel_id, Vel { dx: 3.0, dy: 4.0 });
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn insert_raw_drop_on_abort() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let tracked_id = world.register_component::<Tracked>();

        {
            let mut cs = EnumChangeSet::new();
            cs.insert_raw::<Tracked>(e, tracked_id, Tracked::new(42, &drops));
            // drop without apply
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn spawn_bundle_raw_and_apply() {
        let mut world = World::new();
        // Pre-register component types
        world.register_component::<Pos>();
        world.register_component::<Vel>();

        let entity = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle_raw(
            entity,
            &world.components,
            (Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }),
        );
        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(entity), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn apply_batch_despawns() {
        let mut world = World::new();
        let a = world.spawn((10u32,));
        let b = world.spawn((20u32,));
        let c = world.spawn((30u32,));

        let mut cs = EnumChangeSet::new();
        cs.record_despawn(a);
        cs.record_despawn(c);

        cs.apply(&mut world).unwrap();

        assert!(!world.is_alive(a));
        assert!(world.is_alive(b));
        assert!(!world.is_alive(c));
        assert_eq!(*world.get::<u32>(b).unwrap(), 20);
    }

    // ── Sparse mutation tests ────────────────────────────────────────

    #[test]
    fn sparse_insert() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert_sparse::<Vel>(&mut world, e, Vel { dx: 5.0, dy: 6.0 });

        cs.apply(&mut world).unwrap();

        // Sparse component should be readable via world.get (routes through sparse).
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 5.0, dy: 6.0 }));
        // Archetype component still intact.
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn sparse_insert_overwrite() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        // Initial sparse insert.
        world.insert_sparse::<Vel>(e, Vel { dx: 1.0, dy: 2.0 });
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 1.0, dy: 2.0 }));

        // Overwrite via changeset.
        let mut cs = EnumChangeSet::new();
        cs.insert_sparse::<Vel>(&mut world, e, Vel { dx: 99.0, dy: 99.0 });

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 99.0, dy: 99.0 }));
    }

    #[test]
    fn sparse_remove() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Vel>(e, Vel { dx: 3.0, dy: 4.0 });

        let mut cs = EnumChangeSet::new();
        cs.remove_sparse::<Vel>(&mut world, e);

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), None);
    }

    #[test]
    fn sparse_remove_absent_is_noop() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.register_component::<Vel>();

        let mut cs = EnumChangeSet::new();
        cs.remove_sparse::<Vel>(&mut world, e);

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn sparse_insert_drop_on_abort() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        {
            let mut cs = EnumChangeSet::new();
            cs.insert_sparse::<Tracked>(&mut world, e, Tracked::new(42, &drops));
            // drop without apply
        }

        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "destructor should run on changeset drop"
        );
    }

    #[test]
    fn sparse_insert_no_double_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        {
            let mut cs = EnumChangeSet::new();
            cs.insert_sparse::<Tracked>(&mut world, e, Tracked::new(42, &drops));
            cs.apply(&mut world).unwrap();
        }

        // Value owned by sparse storage, no spurious drops.
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        world.despawn(e);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "one drop from despawn");
    }

    #[test]
    fn sparse_overwrite_tracked_drops_old() {
        // Verifies that overwriting a sparse component with a non-trivial
        // destructor drops the old value exactly once during apply.
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        // Insert initial Tracked into sparse storage.
        {
            let mut cs = EnumChangeSet::new();
            cs.insert_sparse::<Tracked>(&mut world, e, Tracked::new(1, &drops));
            cs.apply(&mut world).unwrap();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0, "no drops yet");

        // Overwrite with a new Tracked — old value dropped during apply.
        {
            let mut cs = EnumChangeSet::new();
            cs.insert_sparse::<Tracked>(&mut world, e, Tracked::new(2, &drops));
            cs.apply(&mut world).unwrap();
            assert_eq!(
                drops.load(Ordering::SeqCst),
                1,
                "old value dropped during apply"
            );
        }

        // New value still alive in world.
        world.despawn(e);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "new value dropped by despawn"
        );
    }

    #[test]
    fn sparse_remove_tracked_drops_value() {
        // Verifies that removing a sparse component with a non-trivial
        // destructor via changeset drops the value during apply.
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        // Insert Tracked into sparse storage.
        {
            let mut cs = EnumChangeSet::new();
            cs.insert_sparse::<Tracked>(&mut world, e, Tracked::new(1, &drops));
            cs.apply(&mut world).unwrap();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        // Remove via changeset — value dropped during apply.
        {
            let mut cs = EnumChangeSet::new();
            cs.remove_sparse::<Tracked>(&mut world, e);
            cs.apply(&mut world).unwrap();
            assert_eq!(
                drops.load(Ordering::SeqCst),
                1,
                "value dropped during apply"
            );
        }

        // Entity still alive but no sparse component — despawn should not drop again.
        world.despawn(e);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "no extra drops");
    }

    #[test]
    fn mixed_sparse_and_dense_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));

        let mut cs = EnumChangeSet::new();
        // Dense insert (archetype migration)
        cs.insert::<Vel>(&mut world, e, Vel { dx: 1.0, dy: 2.0 });
        // Sparse insert
        cs.insert_sparse::<f32>(&mut world, e, 42.0f32);

        cs.apply(&mut world).unwrap();

        // Both present
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 1.0, dy: 2.0 }));
        assert_eq!(world.get::<f32>(e), Some(&42.0f32));
    }

    #[test]
    fn sparse_iter_mutations_yields_correct_views() {
        let mut world = World::new();
        let e = world.alloc_entity();
        world.register_component::<f32>();

        let mut cs = EnumChangeSet::new();
        cs.insert_sparse::<f32>(&mut world, e, 42.0f32);
        cs.remove_sparse::<u32>(&mut world, e);

        let views: Vec<_> = cs.iter_mutations().collect();
        assert_eq!(views.len(), 2);
        assert!(matches!(views[0], MutationRef::SparseInsert { .. }));
        assert!(matches!(views[1], MutationRef::SparseRemove { .. }));

        if let MutationRef::SparseInsert { data, .. } = &views[0] {
            assert_eq!(data.len(), std::mem::size_of::<f32>());
            let val = unsafe { *(data.as_ptr() as *const f32) };
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    fn sparse_insert_apply() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert_sparse::<Vel>(&mut world, e, Vel { dx: 10.0, dy: 20.0 });

        cs.apply(&mut world).unwrap();
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 10.0, dy: 20.0 }));
    }

    /// Component with a drop counter — verifies no leaks on partial apply.
    struct DropCounted {
        _value: u32,
        counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Drop for DropCounted {
        fn drop(&mut self) {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn partial_apply_drops_unapplied_values() {
        use std::sync::Arc;
        use std::sync::atomic::Ordering;

        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut world = World::new();
        let live = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let dead = world.spawn((Pos { x: 3.0, y: 4.0 },));
        world.despawn(dead); // dead entity

        // Register the droppable component type.
        let dc_id = world.register_component::<DropCounted>();

        let mut cs = EnumChangeSet::new();
        // Mutation 0: insert on live entity — will succeed
        cs.insert_raw::<DropCounted>(
            live,
            dc_id,
            DropCounted {
                _value: 1,
                counter: drops.clone(),
            },
        );
        // Mutation 1: insert on dead entity — will fail
        cs.insert_raw::<DropCounted>(
            dead,
            dc_id,
            DropCounted {
                _value: 2,
                counter: drops.clone(),
            },
        );

        // Apply should fail on mutation 1.
        let result = cs.apply(&mut world);
        assert!(result.is_err());

        // Mutation 0's value transferred to World (not dropped by changeset).
        // Mutation 1's value should have been dropped by the changeset cleanup.
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "unapplied value must be dropped"
        );

        // Despawn the live entity to drop mutation 0's value from World.
        world.despawn(live);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "both values must be dropped total"
        );
    }
}
