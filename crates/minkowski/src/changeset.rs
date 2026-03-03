use std::alloc::{self, Layout};
use std::ptr::NonNull;

use crate::bundle::Bundle;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::world::{get_pair_mut, EntityLocation, World};

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
}

impl Arena {
    pub fn new() -> Self {
        Self {
            data: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    /// Copy `layout.size()` bytes from `src` into the arena.
    /// Returns the byte offset where data was written.
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
    pub fn get(&self, offset: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(offset) }
    }

    fn grow(&mut self, min_capacity: usize) {
        let new_capacity = (min_capacity * 2).max(64);
        let new_layout =
            Layout::from_size_align(new_capacity, ARENA_ALIGN).expect("invalid arena layout");

        let new_ptr = if self.capacity == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout =
                Layout::from_size_align(self.capacity, ARENA_ALIGN).expect("invalid arena layout");
            unsafe { alloc::realloc(self.data.as_ptr(), old_layout, new_capacity) }
        };

        self.data = NonNull::new(new_ptr).unwrap_or_else(|| alloc::handle_alloc_error(new_layout));
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
            unsafe {
                alloc::dealloc(self.data.as_ptr(), layout);
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
}

/// Data-driven mutation buffer. Records structural mutations as an enum vec
/// with component bytes stored in a contiguous Arena.
///
/// Implements `Drop`: if the changeset is discarded without calling `apply()`,
/// destructors are run for any values whose ownership was transferred in via
/// typed helpers (`insert`, `spawn_bundle`). Values recorded through the raw
/// API (`record_insert`, `record_spawn`) are borrowed, not owned, so their
/// callers remain responsible for cleanup.
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
        Self {
            mutations: Vec::new(),
            arena: Arena::new(),
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
            self.drop_entries.push(DropEntry { offset, drop_fn });
        }
    }

    /// Record removing a component from an entity. Auto-registers the
    /// component type.
    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        let comp_id = world.register_component::<T>();
        self.record_remove(entity, comp_id);
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
        for &(comp_id, offset, _) in &components {
            if let Some(drop_fn) = world.components.info(comp_id).drop_fn {
                self.drop_entries.push(DropEntry { offset, drop_fn });
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
    /// Apply every mutation in this changeset to `world`, returning a reverse
    /// changeset that undoes the changes when applied.
    pub fn apply(mut self, world: &mut World) -> EnumChangeSet {
        // Disarm drop entries — ownership transfers to the world during apply.
        // If apply panics (programming error), arena values may leak rather
        // than double-drop, which is the safer failure mode.
        self.drop_entries.clear();

        let mut reverse = EnumChangeSet::new();

        for mutation in &self.mutations {
            match mutation {
                Mutation::Spawn { entity, components } => {
                    assert!(
                        !world.is_placed(*entity),
                        "EnumChangeSet::apply: cannot spawn entity {:?} — \
                         already placed in an archetype",
                        entity,
                    );
                    // --- Apply: push raw component data into the right archetype ---
                    let sorted_ids: Vec<ComponentId> = {
                        let mut ids: Vec<_> = components.iter().map(|&(id, _, _)| id).collect();
                        ids.sort_unstable();
                        ids.dedup();
                        ids
                    };
                    let arch_id = world
                        .archetypes
                        .get_or_create(&sorted_ids, &world.components);
                    let index = entity.index() as usize;

                    if index >= world.entity_locations.len() {
                        world.entity_locations.resize(index + 1, None);
                    }

                    let tick = world.next_tick();
                    let archetype = &mut world.archetypes.archetypes[arch_id.0];
                    for &(comp_id, offset, layout) in components {
                        let src = self.arena.get(offset);
                        let col = archetype.component_index[&comp_id];
                        unsafe {
                            archetype.columns[col].push(src as *mut u8);
                        }
                        archetype.columns[col].mark_changed(tick);
                        let _ = layout;
                    }
                    let row = archetype.entities.len();
                    archetype.entities.push(*entity);

                    world.entity_locations[index] = Some(EntityLocation {
                        archetype_id: arch_id,
                        row,
                    });

                    // --- Reverse: despawn this entity ---
                    reverse.record_despawn(*entity);
                }

                Mutation::Despawn { entity } => {
                    // --- Capture: read all components before despawning ---
                    let comp_data = world.read_all_components(*entity).unwrap_or_default();
                    let captured: Vec<(ComponentId, *const u8, Layout)> = comp_data;

                    // Record spawn as reverse (copies bytes into reverse arena)
                    reverse.record_spawn(*entity, &captured);

                    // --- Apply ---
                    world.despawn(*entity);
                }

                Mutation::Insert {
                    entity,
                    component_id,
                    offset,
                    layout,
                } => {
                    let data_ptr = self.arena.get(*offset);
                    changeset_insert_raw(
                        world,
                        &mut reverse,
                        *entity,
                        *component_id,
                        data_ptr,
                        *layout,
                    );
                }

                Mutation::Remove {
                    entity,
                    component_id,
                } => {
                    changeset_remove_raw(world, &mut reverse, *entity, *component_id);
                }
            }
        }

        reverse
    }
}

/// Raw insert: either overwrites an existing component in-place (capturing old
/// value for reverse), or performs archetype migration to add a new component.
fn changeset_insert_raw(
    world: &mut World,
    reverse: &mut EnumChangeSet,
    entity: Entity,
    comp_id: ComponentId,
    data_ptr: *const u8,
    layout: Layout,
) {
    assert!(world.is_alive(entity), "entity is not alive");
    let index = entity.index() as usize;
    let location = world.entity_locations[index].unwrap();

    let src_arch = &world.archetypes.archetypes[location.archetype_id.0];

    if src_arch.component_ids.contains(comp_id) {
        // Entity already has this component — overwrite in-place.
        let col_idx = src_arch.component_index[&comp_id];

        // Capture old value for reverse (read path — no tick).
        let old_ptr = unsafe { src_arch.columns[col_idx].get_ptr(location.row) };
        reverse.record_insert(entity, comp_id, old_ptr as *const u8, layout);

        // Overwrite with new data (write path — marks column changed).
        let tick = world.next_tick();
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
        // Entity does not have this component — reverse is Remove.
        reverse.record_remove(entity, comp_id);

        // Archetype migration: source components + new component.
        let src_arch = &world.archetypes.archetypes[location.archetype_id.0];
        let mut target_ids = src_arch.sorted_ids.clone();
        target_ids.push(comp_id);
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id = world
            .archetypes
            .get_or_create(&target_ids, &world.components);
        let tick = world.next_tick();

        let (src_arch, target_arch) = get_pair_mut(
            &mut world.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns from source to target.
        for (&cid, &src_col) in &src_arch.component_index {
            if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        // Write the new component into target.
        let tgt_col = target_arch.component_index[&comp_id];
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

        world.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });
    }
}

/// Raw remove: performs archetype migration to remove a component, capturing its
/// old value for the reverse changeset.
fn changeset_remove_raw(
    world: &mut World,
    reverse: &mut EnumChangeSet,
    entity: Entity,
    comp_id: ComponentId,
) {
    assert!(world.is_alive(entity), "entity is not alive");
    let index = entity.index() as usize;
    let location = world.entity_locations[index].unwrap();
    let src_arch = &world.archetypes.archetypes[location.archetype_id.0];

    if !src_arch.component_ids.contains(comp_id) {
        return; // Component not present — nothing to do.
    }

    // Capture old value for reverse (Insert). The source column will be
    // swap_remove_no_drop'd below, so the reverse arena becomes the sole
    // owner of this value — register a drop entry.
    let info = world.components.info(comp_id);
    let layout = info.layout;
    let drop_fn = info.drop_fn;
    let col_idx = src_arch.component_index[&comp_id];
    let old_ptr = unsafe { src_arch.columns[col_idx].get_ptr(location.row) };
    let offset = reverse.record_insert(entity, comp_id, old_ptr as *const u8, layout);
    if let Some(drop_fn) = drop_fn {
        reverse.drop_entries.push(DropEntry { offset, drop_fn });
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
        // swap_remove_no_drop for the removed component (data already captured)
        let removed_col = arch.component_index[&comp_id];
        unsafe {
            arch.columns[removed_col].swap_remove_no_drop(src_row);
        }
        // swap_remove with drop for remaining columns
        for (&cid, &col_idx_inner) in &arch.component_index {
            if cid != comp_id {
                unsafe {
                    arch.columns[col_idx_inner].swap_remove(src_row);
                }
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
        let empty_arch_id = world.archetypes.get_or_create(&[], &world.components);
        let empty_arch = &mut world.archetypes.archetypes[empty_arch_id.0];
        empty_arch.entities.push(entity);
        world.entity_locations[index] = Some(EntityLocation {
            archetype_id: empty_arch_id,
            row: empty_arch.entities.len() - 1,
        });
        return;
    }

    let target_arch_id = world
        .archetypes
        .get_or_create(&target_ids, &world.components);

    let (src_arch, target_arch) = get_pair_mut(
        &mut world.archetypes.archetypes,
        src_arch_id.0,
        target_arch_id.0,
    );

    // Move shared columns (skip removed component).
    for (&cid, &src_col) in &src_arch.component_index {
        if cid == comp_id {
            // Already captured — just discard from source.
            unsafe {
                src_arch.columns[src_col].swap_remove_no_drop(src_row);
            }
        } else if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
            unsafe {
                let ptr = src_arch.columns[src_col].get_ptr(src_row);
                target_arch.columns[tgt_col].push(ptr);
                src_arch.columns[src_col].swap_remove_no_drop(src_row);
            }
        }
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

    world.entity_locations[index] = Some(EntityLocation {
        archetype_id: target_arch_id,
        row: target_row,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_read_back() {
        let mut arena = Arena::new();
        let value: u32 = 42;
        let layout = Layout::new::<u32>();
        let offset = arena.alloc(&value as *const u32 as *const u8, layout);
        let ptr = arena.get(offset) as *const u32;
        assert_eq!(unsafe { *ptr }, 42);
    }

    #[test]
    fn arena_alignment() {
        let mut arena = Arena::new();
        let byte: u8 = 0xFF;
        let _ = arena.alloc(&byte as *const u8, Layout::new::<u8>());

        let val: u64 = 123456789;
        let offset = arena.alloc(&val as *const u64 as *const u8, Layout::new::<u64>());
        assert_eq!(offset % 8, 0, "u64 offset must be 8-byte aligned");
    }

    #[test]
    fn arena_zst() {
        let mut arena = Arena::new();
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
    fn apply_spawn_and_reverse_despawns() {
        let mut world = World::new();
        let entity = world.alloc_entity();

        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            entity,
            (Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }),
        );

        let reverse = cs.apply(&mut world);
        assert!(world.is_alive(entity));
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));

        // Reverse should despawn
        let _ = reverse.apply(&mut world);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn apply_despawn_and_reverse_respawns() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 5.0, y: 6.0 }, Vel { dx: 7.0, dy: 8.0 }));

        let mut cs = EnumChangeSet::new();
        cs.record_despawn(e);
        let reverse = cs.apply(&mut world);
        assert!(!world.is_alive(e));

        // Reverse should respawn — creates new entity with same data
        let _ = reverse.apply(&mut world);
        let count = world.query::<(&Pos, &Vel)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn apply_insert_new_and_reverse_removes() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));

        let _ = reverse.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_insert_overwrite_and_reverse_restores() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Pos>(&mut world, e, Pos { x: 99.0, y: 99.0 });

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 99.0, y: 99.0 }));

        let _ = reverse.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_remove_and_reverse_reinserts() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let mut cs = EnumChangeSet::new();
        cs.remove::<Vel>(&mut world, e);

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));

        let _ = reverse.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn apply_empty_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let cs = EnumChangeSet::new();
        let reverse = cs.apply(&mut world);
        assert!(reverse.is_empty());
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn round_trip_forward_reverse_forward() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Vel>(&mut world, e, Vel { dx: 10.0, dy: 20.0 });

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 10.0, dy: 20.0 }));

        let forward_again = reverse.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);

        let _ = forward_again.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 10.0, dy: 20.0 }));
    }

    // ── typed helper tests ────────────────────────────────────────

    #[test]
    fn typed_insert_and_apply() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });

        let _reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn typed_insert_overwrite_and_reverse() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        cs.insert::<Pos>(&mut world, e, Pos { x: 99.0, y: 99.0 });

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 99.0, y: 99.0 }));

        let _ = reverse.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn typed_spawn_and_reverse() {
        let mut world = World::new();
        let entity = world.alloc_entity();

        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            entity,
            (Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }),
        );

        let reverse = cs.apply(&mut world);
        assert!(world.is_alive(entity));
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(entity), Some(&Vel { dx: 3.0, dy: 4.0 }));

        let _ = reverse.apply(&mut world);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn typed_remove_and_reverse() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let mut cs = EnumChangeSet::new();
        cs.remove::<Vel>(&mut world, e);

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));

        let _ = reverse.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
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
    #[should_panic(expected = "already placed")]
    fn apply_spawn_panics_on_live_entity() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        // Use raw record_spawn to bypass the spawn_bundle guard,
        // then verify apply() catches it.
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
        let _ = cs.apply(&mut world);
    }

    // ── Drop safety tests ─────────────────────────────────────────

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
            let _reverse = cs.apply(&mut world);
            // cs dropped after apply — should not double-drop
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
    fn reverse_of_remove_drops_owned_value() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Tracked::new(99, &drops)));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "spawn transfers ownership, no drop"
        );

        {
            let mut cs = EnumChangeSet::new();
            cs.remove::<Tracked>(&mut world, e);
            let _reverse = cs.apply(&mut world);
            // _reverse owns the Tracked value (remove used swap_remove_no_drop)
            assert_eq!(drops.load(Ordering::SeqCst), 0, "no drop yet");
            // reverse dropped here without apply()
        }

        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "reverse drop should clean up owned value"
        );
    }
}
