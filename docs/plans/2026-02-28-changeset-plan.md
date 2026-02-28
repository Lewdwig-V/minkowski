# ChangeSet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace opaque closure-based mutations with a data-driven `ChangeSet` abstraction that supports inspection, serialization, and rollback, then refactor `CommandBuffer` as a typed facade on top.

**Architecture:** A `ChangeSet` trait defines the mutation recording + apply + reverse interface. `EnumChangeSet` implements it with a `Vec<Mutation>` enum + a contiguous `Arena` for component byte data. `apply()` returns a reverse `EnumChangeSet` for rollback. `CommandBuffer` is refactored to delegate to `EnumChangeSet` internally.

**Tech Stack:** Rust, `std::alloc::Layout`, raw pointer arithmetic

---

### Task 1: Create `changeset.rs` with Arena

The arena is the foundation — all other types depend on it. Start here.

**Files:**
- Create: `crates/minkowski/src/changeset.rs`
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Write the failing test**

Create `crates/minkowski/src/changeset.rs` with just the test module at the bottom, plus a minimal `Arena` struct stub:

```rust
use std::alloc::Layout;

/// Contiguous byte arena for component data. Mutations store integer offsets
/// into this arena, avoiding per-mutation heap allocation.
pub(crate) struct Arena {
    data: Vec<u8>,
}

impl Arena {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_read_back() {
        let mut arena = Arena::new();
        let val: u32 = 0xDEAD_BEEF;
        let layout = Layout::new::<u32>();
        let offset = arena.alloc(&val as *const u32 as *const u8, layout);
        let ptr = arena.get(offset);
        let read_back = unsafe { std::ptr::read(ptr as *const u32) };
        assert_eq!(read_back, 0xDEAD_BEEF);
    }

    #[test]
    fn arena_alignment() {
        let mut arena = Arena::new();
        // Alloc a u8 first to misalign
        let byte: u8 = 0xFF;
        arena.alloc(&byte as *const u8 as *const u8, Layout::new::<u8>());
        // Now alloc a u64 — must be 8-byte aligned
        let val: u64 = 42;
        let offset = arena.alloc(&val as *const u64 as *const u8, Layout::new::<u64>());
        assert_eq!(offset % 8, 0);
        let read_back = unsafe { std::ptr::read(arena.get(offset) as *const u64) };
        assert_eq!(read_back, 42);
    }

    #[test]
    fn arena_zst() {
        let mut arena = Arena::new();
        let layout = Layout::new::<()>();
        let offset = arena.alloc(std::ptr::NonNull::dangling().as_ptr(), layout);
        assert_eq!(offset, 0);
    }
}
```

**Step 2: Add module to lib.rs**

In `crates/minkowski/src/lib.rs`, add after `pub mod table;`:

```rust
pub mod changeset;
```

**Step 3: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- arena`
Expected: FAIL — `alloc` and `get` methods don't exist yet.

**Step 4: Implement Arena**

Add to `Arena` impl in `crates/minkowski/src/changeset.rs`:

```rust
    /// Copy `layout.size()` bytes from `src` into the arena.
    /// Returns the byte offset where data was written.
    pub fn alloc(&mut self, src: *const u8, layout: Layout) -> usize {
        if layout.size() == 0 {
            return 0;
        }
        let align = layout.align();
        let offset = (self.data.len() + align - 1) & !(align - 1);
        self.data.resize(offset + layout.size(), 0);
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.data.as_mut_ptr().add(offset), layout.size());
        }
        offset
    }

    /// Get a raw pointer to data at the given offset.
    pub fn get(&self, offset: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(offset) }
    }
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- arena`
Expected: 3 tests PASS.

**Step 6: Commit**

```bash
git add crates/minkowski/src/changeset.rs crates/minkowski/src/lib.rs
git commit -m "feat: add Arena byte allocator for ChangeSet component data

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add Mutation enum and EnumChangeSet struct

The data types for recording mutations. No apply logic yet.

**Files:**
- Modify: `crates/minkowski/src/changeset.rs`

**Step 1: Add the types**

Add these to `changeset.rs` above the `#[cfg(test)]` block:

```rust
use crate::component::ComponentId;
use crate::entity::Entity;

/// A single structural mutation recorded in a ChangeSet.
#[derive(Debug)]
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
pub struct EnumChangeSet {
    pub(crate) mutations: Vec<Mutation>,
    pub(crate) arena: Arena,
}

impl EnumChangeSet {
    pub fn new() -> Self {
        Self {
            mutations: Vec::new(),
            arena: Arena::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }
}

impl Default for EnumChangeSet {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 2: Add recording methods to EnumChangeSet**

```rust
impl EnumChangeSet {
    // ... (after is_empty)

    /// Record a spawn: entity + raw component data.
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

    /// Record inserting a component on an entity.
    pub fn record_insert(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        data: *const u8,
        layout: Layout,
    ) {
        let offset = self.arena.alloc(data, layout);
        self.mutations.push(Mutation::Insert {
            entity,
            component_id,
            offset,
            layout,
        });
    }

    /// Record removing a component from an entity.
    pub fn record_remove(&mut self, entity: Entity, component_id: ComponentId) {
        self.mutations.push(Mutation::Remove {
            entity,
            component_id,
        });
    }
}
```

**Step 3: Write tests for recording**

Add to the test module:

```rust
    use crate::entity::Entity;

    #[test]
    fn record_and_count() {
        let mut cs = EnumChangeSet::new();
        assert!(cs.is_empty());

        cs.record_despawn(Entity::new(0, 0));
        cs.record_remove(Entity::new(1, 0), 0);
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn record_insert_stores_data() {
        let mut cs = EnumChangeSet::new();
        let val: u32 = 42;
        cs.record_insert(
            Entity::new(0, 0),
            0, // component_id
            &val as *const u32 as *const u8,
            Layout::new::<u32>(),
        );
        assert_eq!(cs.len(), 1);
        // Verify data is in the arena
        match &cs.mutations[0] {
            Mutation::Insert { offset, layout, .. } => {
                let ptr = cs.arena.get(*offset);
                let read = unsafe { std::ptr::read(ptr as *const u32) };
                assert_eq!(read, 42);
                assert_eq!(layout.size(), 4);
            }
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn record_spawn_stores_components() {
        let mut cs = EnumChangeSet::new();
        let a: u32 = 10;
        let b: f64 = 3.14;
        cs.record_spawn(
            Entity::new(0, 0),
            &[
                (0, &a as *const u32 as *const u8, Layout::new::<u32>()),
                (1, &b as *const f64 as *const u8, Layout::new::<f64>()),
            ],
        );
        assert_eq!(cs.len(), 1);
        match &cs.mutations[0] {
            Mutation::Spawn { components, .. } => {
                assert_eq!(components.len(), 2);
            }
            _ => panic!("expected Spawn"),
        }
    }
```

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- changeset`
Expected: All 6 tests PASS (3 arena + 3 recording).

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 6: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat: add Mutation enum and EnumChangeSet with recording methods

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add a World helper to read all component bytes for an entity

The reverse of `Despawn` needs to capture every component an entity has. Add a helper method on `World` that reads all component bytes for an entity into a list of `(ComponentId, Vec<u8>)`.

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Write the failing test**

Add to `world.rs` test module:

```rust
    #[test]
    fn read_entity_components_raw() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let components = world.read_all_components(e).unwrap();
        assert_eq!(components.len(), 2);
        // Each entry should have correct layout size
        for &(_, _, layout) in &components {
            assert!(layout.size() > 0);
        }
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- read_entity_components_raw`
Expected: FAIL — method doesn't exist.

**Step 3: Implement the method**

Add to `impl World` in `crates/minkowski/src/world.rs`:

```rust
    /// Read all component data for an entity as raw bytes.
    /// Returns (ComponentId, *const u8, Layout) per component.
    /// The pointers are valid until the next structural mutation.
    pub(crate) fn read_all_components(
        &self,
        entity: Entity,
    ) -> Option<Vec<(ComponentId, *const u8, Layout)>> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let archetype = &self.archetypes.archetypes[location.archetype_id.0];

        let components: Vec<_> = archetype
            .sorted_ids
            .iter()
            .map(|&comp_id| {
                let col_idx = archetype.component_index[&comp_id];
                let info = self.components.info(comp_id);
                let ptr = unsafe { archetype.columns[col_idx].get_ptr(location.row) };
                (comp_id, ptr as *const u8, info.layout)
            })
            .collect();

        Some(components)
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- read_entity_components_raw`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add World::read_all_components for entity byte capture

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Implement `EnumChangeSet::apply()` with reverse capture

The core logic: apply each mutation to the world, building a reverse changeset.

**Files:**
- Modify: `crates/minkowski/src/changeset.rs`

**Step 1: Write the failing tests**

Add to the changeset test module:

```rust
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn apply_spawn_and_reverse_despawns() {
        let mut world = World::new();

        let mut cs = EnumChangeSet::new();
        let entity = world.entities.alloc();
        let pos = Pos { x: 1.0, y: 2.0 };
        let vel = Vel { dx: 3.0, dy: 4.0 };

        let pos_id = world.components.register::<Pos>();
        let vel_id = world.components.register::<Vel>();

        cs.record_spawn(
            entity,
            &[
                (pos_id, &pos as *const Pos as *const u8, Layout::new::<Pos>()),
                (vel_id, &vel as *const Vel as *const u8, Layout::new::<Vel>()),
            ],
        );

        let reverse = cs.apply(&mut world);
        assert!(world.is_alive(entity));
        assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));

        // Apply reverse — should despawn
        reverse.apply(&mut world);
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

        // Apply reverse — should respawn with original data
        reverse.apply(&mut world);
        // Entity was recycled with new generation, check the data is back
        // The reverse spawn creates a new entity (same index, new gen)
        let count = world.query::<(&Pos, &Vel)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn apply_insert_new_and_reverse_removes() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        let vel = Vel { dx: 3.0, dy: 4.0 };
        let vel_id = world.components.register::<Vel>();
        cs.record_insert(
            e,
            vel_id,
            &vel as *const Vel as *const u8,
            Layout::new::<Vel>(),
        );

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));

        // Reverse should remove the inserted component
        reverse.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_insert_overwrite_and_reverse_restores() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut cs = EnumChangeSet::new();
        let new_pos = Pos { x: 99.0, y: 99.0 };
        let pos_id = world.components.register::<Pos>();
        cs.record_insert(
            e,
            pos_id,
            &new_pos as *const Pos as *const u8,
            Layout::new::<Pos>(),
        );

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 99.0, y: 99.0 }));

        // Reverse should restore original value
        reverse.apply(&mut world);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn apply_remove_and_reverse_reinserts() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let mut cs = EnumChangeSet::new();
        let vel_id = world.components.register::<Vel>();
        cs.record_remove(e, vel_id);

        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));

        // Reverse should re-insert the removed component
        reverse.apply(&mut world);
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
        let vel = Vel { dx: 10.0, dy: 20.0 };
        let vel_id = world.components.register::<Vel>();
        cs.record_insert(
            e,
            vel_id,
            &vel as *const Vel as *const u8,
            Layout::new::<Vel>(),
        );

        // Forward: insert Vel
        let reverse = cs.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 10.0, dy: 20.0 }));

        // Reverse: remove Vel
        let forward_again = reverse.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), None);

        // Forward again: insert Vel
        let _ = forward_again.apply(&mut world);
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 10.0, dy: 20.0 }));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- changeset::tests::apply`
Expected: FAIL — `apply` method doesn't exist.

**Step 3: Implement `apply()`**

This is the core method. Add to `impl EnumChangeSet` in `changeset.rs`:

```rust
    /// Apply all recorded mutations to the world.
    /// Returns a reverse EnumChangeSet that undoes these changes when applied.
    pub fn apply(self, world: &mut World) -> EnumChangeSet {
        use crate::bundle::Bundle;

        let mut reverse = EnumChangeSet::new();

        for mutation in &self.mutations {
            match mutation {
                Mutation::Spawn {
                    entity,
                    components,
                } => {
                    // Ensure entity_locations is big enough
                    let index = entity.index() as usize;
                    if index >= world.entity_locations.len() {
                        world.entity_locations.resize(index + 1, None);
                    }

                    // Build sorted component IDs for archetype lookup
                    let mut sorted_ids: Vec<ComponentId> =
                        components.iter().map(|&(id, _, _)| id).collect();
                    sorted_ids.sort_unstable();

                    let arch_id = world
                        .archetypes
                        .get_or_create(&sorted_ids, &world.components);
                    let archetype =
                        &mut world.archetypes.archetypes[arch_id.0];

                    // Push each component into its column
                    for &(comp_id, offset, _layout) in components {
                        let col_idx = archetype.component_index[&comp_id];
                        let ptr = self.arena.get(offset) as *mut u8;
                        unsafe {
                            archetype.columns[col_idx].push(ptr);
                        }
                    }

                    let row = archetype.entities.len();
                    archetype.entities.push(*entity);
                    world.entity_locations[index] =
                        Some(crate::world::EntityLocation {
                            archetype_id: arch_id,
                            row,
                        });

                    // Reverse: despawn this entity
                    reverse.record_despawn(*entity);
                }

                Mutation::Despawn { entity } => {
                    // Capture all component data before despawning
                    if let Some(components) = world.read_all_components(*entity) {
                        let stored: Vec<(ComponentId, *const u8, Layout)> = components;
                        reverse.record_spawn(*entity, &stored);
                    }
                    world.despawn(*entity);
                }

                Mutation::Insert {
                    entity,
                    component_id,
                    offset,
                    layout,
                } => {
                    // Check if entity already has this component (for reverse)
                    let has_component = world
                        .entity_locations
                        .get(entity.index() as usize)
                        .and_then(|loc| loc.as_ref())
                        .map(|loc| {
                            world.archetypes.archetypes[loc.archetype_id.0]
                                .component_ids
                                .contains(*component_id)
                        })
                        .unwrap_or(false);

                    if has_component {
                        // Capture old value for reverse
                        let loc =
                            world.entity_locations[entity.index() as usize].unwrap();
                        let archetype =
                            &world.archetypes.archetypes[loc.archetype_id.0];
                        let col_idx = archetype.component_index[component_id];
                        let old_ptr =
                            unsafe { archetype.columns[col_idx].get_ptr(loc.row) };
                        reverse.record_insert(
                            *entity,
                            *component_id,
                            old_ptr as *const u8,
                            *layout,
                        );
                    } else {
                        // No old value — reverse is a remove
                        reverse.record_remove(*entity, *component_id);
                    }

                    // Apply the insert using raw pointer manipulation
                    // We need to use World::insert-like logic but from raw bytes
                    let ptr = self.arena.get(*offset) as *mut u8;
                    changeset_insert_raw(world, *entity, *component_id, ptr, *layout);
                }

                Mutation::Remove {
                    entity,
                    component_id,
                } => {
                    // Capture component data before removing
                    if let Some(loc) =
                        world.entity_locations.get(entity.index() as usize).copied().flatten()
                    {
                        let archetype =
                            &world.archetypes.archetypes[loc.archetype_id.0];
                        if archetype.component_ids.contains(*component_id) {
                            let col_idx = archetype.component_index[component_id];
                            let info = world.components.info(*component_id);
                            let ptr = unsafe {
                                archetype.columns[col_idx].get_ptr(loc.row)
                            };
                            reverse.record_insert(
                                *entity,
                                *component_id,
                                ptr as *const u8,
                                info.layout,
                            );
                        }
                    }

                    changeset_remove_raw(world, *entity, *component_id);
                }
            }
        }

        reverse
    }
```

**Step 4: Implement the raw insert/remove helpers**

These are private functions in `changeset.rs` that perform untyped insert/remove on the world. They duplicate some logic from `World::insert`/`World::remove` but work with `ComponentId` + raw pointers instead of generic `T`:

```rust
/// Insert a component by raw pointer into an entity. Handles archetype migration.
fn changeset_insert_raw(
    world: &mut World,
    entity: Entity,
    comp_id: ComponentId,
    data: *mut u8,
    layout: Layout,
) {
    use crate::world::EntityLocation;

    assert!(world.is_alive(entity), "entity is not alive");
    let index = entity.index() as usize;
    let location = world.entity_locations[index].unwrap();

    // Overwrite in-place if entity already has this component
    let src_arch = &world.archetypes.archetypes[location.archetype_id.0];
    if src_arch.component_ids.contains(comp_id) {
        let col_idx = src_arch.component_index[&comp_id];
        unsafe {
            let dst = src_arch.columns[col_idx].get_ptr(location.row);
            if let Some(drop_fn) = world.components.info(comp_id).drop_fn {
                drop_fn(dst);
            }
            std::ptr::copy_nonoverlapping(data as *const u8, dst, layout.size());
        }
        return;
    }

    // Archetype migration: source components + new component
    let mut target_ids = src_arch.sorted_ids.clone();
    target_ids.push(comp_id);
    target_ids.sort_unstable();
    let src_arch_id = location.archetype_id;
    let src_row = location.row;

    let target_arch_id = world
        .archetypes
        .get_or_create(&target_ids, &world.components);

    let (src_arch, target_arch) = crate::world::get_pair_mut(
        &mut world.archetypes.archetypes,
        src_arch_id.0,
        target_arch_id.0,
    );

    // Move shared columns
    for (&cid, &src_col) in &src_arch.component_index {
        if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
            unsafe {
                let ptr = src_arch.columns[src_col].get_ptr(src_row);
                target_arch.columns[tgt_col].push(ptr);
                src_arch.columns[src_col].swap_remove_no_drop(src_row);
            }
        }
    }

    // Write the new component
    let tgt_col = target_arch.component_index[&comp_id];
    unsafe {
        target_arch.columns[tgt_col].push(data);
    }

    // Move entity tracking
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

/// Remove a component from an entity by ComponentId. Handles archetype migration.
fn changeset_remove_raw(
    world: &mut World,
    entity: Entity,
    comp_id: ComponentId,
) {
    use crate::world::EntityLocation;

    if !world.is_alive(entity) {
        return;
    }
    let index = entity.index() as usize;
    let location = match world.entity_locations[index] {
        Some(loc) => loc,
        None => return,
    };

    let src_arch = &world.archetypes.archetypes[location.archetype_id.0];
    if !src_arch.component_ids.contains(comp_id) {
        return;
    }

    let target_ids: Vec<ComponentId> = src_arch
        .sorted_ids
        .iter()
        .copied()
        .filter(|&id| id != comp_id)
        .collect();
    let src_arch_id = location.archetype_id;
    let src_row = location.row;

    if target_ids.is_empty() {
        let arch = &mut world.archetypes.archetypes[src_arch_id.0];
        for (&_cid, &col_idx) in &arch.component_index {
            unsafe {
                arch.columns[col_idx].swap_remove(src_row);
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

    let (src_arch, target_arch) = crate::world::get_pair_mut(
        &mut world.archetypes.archetypes,
        src_arch_id.0,
        target_arch_id.0,
    );

    for (&cid, &src_col) in &src_arch.component_index {
        if cid == comp_id {
            unsafe {
                src_arch.columns[src_col].swap_remove(src_row);
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
```

NOTE: `get_pair_mut` in `world.rs` is currently a private function. It needs to be made `pub(crate)` so `changeset.rs` can use it. Change line 13 of `world.rs` from:

```rust
fn get_pair_mut(v: &mut [Archetype], a: usize, b: usize) -> (&mut Archetype, &mut Archetype) {
```

to:

```rust
pub(crate) fn get_pair_mut(v: &mut [Archetype], a: usize, b: usize) -> (&mut Archetype, &mut Archetype) {
```

Also make `EntityLocation` usable from changeset: it's already `pub(crate)`.

**Step 5: Run the apply tests**

Run: `cargo test -p minkowski --lib -- changeset::tests::apply`
Expected: All 7 apply tests PASS.

**Step 6: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (existing + new).

**Step 7: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 8: Commit**

```bash
git add crates/minkowski/src/changeset.rs crates/minkowski/src/world.rs
git commit -m "feat: implement EnumChangeSet::apply() with reverse capture

Supports spawn, despawn, insert (new + overwrite), and remove.
Each apply returns a reverse ChangeSet for rollback.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Refactor CommandBuffer to use EnumChangeSet

Replace the closure-based internals with EnumChangeSet delegation.

**Files:**
- Modify: `crates/minkowski/src/command.rs`
- Modify: `crates/minkowski/src/lib.rs` (re-export EnumChangeSet)

**Step 1: Rewrite CommandBuffer**

Replace the contents of `crates/minkowski/src/command.rs` (keep the test module intact):

```rust
use crate::bundle::Bundle;
use crate::changeset::EnumChangeSet;
use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;
use std::alloc::Layout;

/// Deferred mutation buffer. Records commands during iteration,
/// applies them all at once to &mut World.
///
/// Internally uses EnumChangeSet — mutations are data, not closures.
pub struct CommandBuffer {
    changes: EnumChangeSet,
    /// Pending spawns that need entity allocation at apply time.
    deferred_spawns: Vec<Vec<(crate::component::ComponentId, usize, Layout)>>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self {
            changes: EnumChangeSet::new(),
            deferred_spawns: Vec::new(),
        }
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) {
        let mut components = Vec::new();
        let registry_ids = B::component_ids(&mut RegistryStub);
        // We can't register components here (no World access).
        // Store the raw bytes; actual entity allocation + registration happens at apply time.
        unsafe {
            bundle.put_to_arena(&mut self.changes.arena, &mut components);
        }
        self.deferred_spawns.push(components);
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.changes.record_despawn(entity);
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        let component = std::mem::ManuallyDrop::new(component);
        let ptr = &*component as *const T as *const u8;
        let layout = Layout::new::<T>();
        // ComponentId will be resolved at apply time via type
        // For now, store as a typed closure that resolves at apply time
        self.changes.record_insert_deferred::<T>(entity, ptr, layout);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.changes.record_remove_deferred::<T>(entity);
    }

    /// Apply all commands. Returns a reverse changeset for rollback.
    pub fn apply(self, world: &mut World) -> EnumChangeSet {
        // First, handle deferred spawns
        let mut cs = self.changes;
        for spawn_components in self.deferred_spawns {
            let entity = world.entities.alloc();
            // Re-record as a proper spawn with allocated entity
            // Components already have their data in the arena
            cs.mutations.push(crate::changeset::Mutation::Spawn {
                entity,
                components: spawn_components,
            });
        }
        cs.apply(world)
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty() && self.deferred_spawns.is_empty()
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}
```

STOP — this approach has a problem. `CommandBuffer::spawn()` doesn't have access to `World` or `ComponentRegistry`, so it can't resolve `ComponentId`s at recording time. The existing closure-based approach sidestepped this by deferring everything.

**Revised approach**: Keep CommandBuffer's closure-based `spawn` for now (it needs entity allocation + component registration at apply time, which requires `&mut World`). Refactor `despawn`, `insert`, and `remove` to use EnumChangeSet where possible, but `spawn` stays as a closure. OR — simpler: CommandBuffer stores a mix of EnumChangeSet mutations (for despawn/insert/remove where we have entity + component info) and closures (for spawn where we need World access).

Actually, the cleanest approach: `CommandBuffer::apply()` converts closures to changeset entries by executing them through a recording World wrapper. But that's complex.

**Simplest correct approach**: CommandBuffer keeps closures for `spawn` (needs entity allocation), but uses EnumChangeSet for `despawn`/`insert`/`remove`. `apply()` runs spawn closures first, then applies the changeset. The reverse only covers the changeset mutations (spawns are not reversible through this path — they'd need a separate mechanism). This is pragmatic and matches what's actually achievable without a World reference during recording.

Actually, let me reconsider. The simplest approach that maintains all existing tests:

**Keep CommandBuffer exactly as-is for now.** Add an `apply_reversible` method that wraps the closure-based `apply` inside a ChangeSet by snapshotting before/after. OR — just keep the existing closure-based CommandBuffer and demonstrate ChangeSet as a standalone feature that CommandBuffer can be refactored to use later.

For this plan, let's take the pragmatic path:

**Step 1: Add `EnumChangeSet` re-export in lib.rs**

In `crates/minkowski/src/lib.rs`, add:

```rust
pub use changeset::EnumChangeSet;
```

**Step 2: Keep existing CommandBuffer, add `into_changeset` bridge**

Rather than a full refactor (which has the spawn problem), add a method to CommandBuffer that produces an `EnumChangeSet` from `despawn`/`insert`/`remove` operations, keeping spawn as-is. Actually, even simpler: leave CommandBuffer alone entirely. The ChangeSet is a parallel, lower-level API. Users can use either. The refactor to unify them can come later when we have a component registry accessible at recording time.

Update CommandBuffer's `apply` to return `()` as before (no breaking change), and add a note that ChangeSet is the data-driven alternative.

**Step 3: Update tests to use EnumChangeSet directly**

The existing CommandBuffer tests should continue passing unchanged. New tests demonstrate EnumChangeSet for the same operations.

**Step 4: Run all tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 6: Commit**

```bash
git add crates/minkowski/src/lib.rs
git commit -m "feat: re-export EnumChangeSet from crate root

CommandBuffer remains closure-based for now (needs World for spawn).
EnumChangeSet is the data-driven alternative for when you have
ComponentIds and raw pointers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final verification

**Step 1: Full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass including doc tests.

**Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 3: Miri**

Run: `MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib`
Expected: All tests pass under Miri.

**Step 4: Boids example**

Run: `cargo run -p minkowski --example boids --release 2>&1 | tail -5`
Expected: Completes successfully.

**Step 5: Commit if any fixes needed**

If all clean, no commit needed.
