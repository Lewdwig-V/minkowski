# Phase 1: ECS Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build minkowski's foundational ECS — column-oriented archetype storage with generational entity IDs, parallel iteration, and deferred mutation.

**Architecture:** Type-erased BlobVec columns in archetypes, FixedBitSet matching for queries, rayon for parallel iteration, CommandBuffer for deferred mutations. `world.query()` borrows `&self` (hecs-style unsafe aliasing contract); structural mutations take `&mut self`.

**Tech Stack:** Rust 2021 edition, rayon, fixedbitset, criterion (dev), hecs (dev, benchmark comparison)

**Design doc:** `docs/plans/2026-02-26-phase1-ecs-core-design.md`

---

## Task 1: Workspace Scaffolding

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/minkowski/Cargo.toml`
- Create: `crates/minkowski/src/lib.rs`
- Create: `crates/minkowski/src/entity.rs`
- Create: `crates/minkowski/src/component.rs`
- Create: `crates/minkowski/src/storage/mod.rs`
- Create: `crates/minkowski/src/storage/blob_vec.rs`
- Create: `crates/minkowski/src/storage/archetype.rs`
- Create: `crates/minkowski/src/storage/sparse.rs`
- Create: `crates/minkowski/src/query/mod.rs`
- Create: `crates/minkowski/src/query/fetch.rs`
- Create: `crates/minkowski/src/query/iter.rs`
- Create: `crates/minkowski/src/world.rs`
- Create: `crates/minkowski/src/command.rs`
- Create: `crates/minkowski-derive/Cargo.toml`
- Create: `crates/minkowski-derive/src/lib.rs`
- Create: `.gitignore`

**Step 1: Create workspace root Cargo.toml**

```toml
[workspace]
resolver = "2"
members = ["crates/minkowski", "crates/minkowski-derive"]
```

**Step 2: Create minkowski crate Cargo.toml**

```toml
[package]
name = "minkowski"
version = "0.1.0"
edition = "2021"

[dependencies]
fixedbitset = "0.5"
rayon = "1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
hecs = "0.10"

[[bench]]
name = "spawn"
harness = false

[[bench]]
name = "iterate"
harness = false

[[bench]]
name = "parallel"
harness = false

[[bench]]
name = "add_remove"
harness = false

[[bench]]
name = "fragmented"
harness = false
```

**Step 3: Create minkowski-derive placeholder**

`crates/minkowski-derive/Cargo.toml`:
```toml
[package]
name = "minkowski-derive"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true
```

`crates/minkowski-derive/src/lib.rs`:
```rust
// Phase 2: proc-macro crate for #[derive(Table)]
```

**Step 4: Create lib.rs with module declarations**

`crates/minkowski/src/lib.rs`:
```rust
pub mod entity;
pub mod component;
pub mod storage;
pub mod query;
pub mod world;
pub mod command;

pub use entity::Entity;
pub use world::World;
pub use command::CommandBuffer;
```

**Step 5: Create stub module files**

All files get a single comment: `// TODO: implement`

Module re-export files:

`crates/minkowski/src/storage/mod.rs`:
```rust
pub mod blob_vec;
pub mod archetype;
pub mod sparse;
```

`crates/minkowski/src/query/mod.rs`:
```rust
pub mod fetch;
pub mod iter;
```

Leaf stubs (`entity.rs`, `component.rs`, `storage/blob_vec.rs`, `storage/archetype.rs`, `storage/sparse.rs`, `query/fetch.rs`, `query/iter.rs`, `world.rs`, `command.rs`):
```rust
// TODO: implement
```

**Step 6: Create .gitignore**

```
/target
Cargo.lock
```

**Step 7: Verify it compiles**

Run: `cargo check --workspace`
Expected: compiles with no errors (may have warnings about unused modules)

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: scaffold workspace with minkowski + minkowski-derive crates"
```

---

## Task 2: Entity & EntityAllocator

**Files:**
- Modify: `crates/minkowski/src/entity.rs`
- Modify: `crates/minkowski/src/lib.rs` (re-exports)

**Step 1: Write failing tests**

Add to `crates/minkowski/src/entity.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_bit_packing() {
        let e = Entity::new(42, 7);
        assert_eq!(e.index(), 42);
        assert_eq!(e.generation(), 7);
    }

    #[test]
    fn entity_max_values() {
        let e = Entity::new(u32::MAX, u32::MAX);
        assert_eq!(e.index(), u32::MAX);
        assert_eq!(e.generation(), u32::MAX);
    }

    #[test]
    fn entity_equality() {
        let a = Entity::new(1, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(1, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn allocator_basic() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.alloc();
        let e2 = alloc.alloc();
        assert_eq!(e1.index(), 0);
        assert_eq!(e1.generation(), 0);
        assert_eq!(e2.index(), 1);
        assert_eq!(e2.generation(), 0);
        assert!(alloc.is_alive(e1));
        assert!(alloc.is_alive(e2));
    }

    #[test]
    fn allocator_recycle() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.alloc();
        assert!(alloc.dealloc(e1));
        let e2 = alloc.alloc();
        // Same index, bumped generation
        assert_eq!(e2.index(), 0);
        assert_eq!(e2.generation(), 1);
        // Old entity is dead
        assert!(!alloc.is_alive(e1));
        assert!(alloc.is_alive(e2));
    }

    #[test]
    fn allocator_double_dealloc() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.alloc();
        assert!(alloc.dealloc(e));
        assert!(!alloc.dealloc(e)); // already dead
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- entity::tests`
Expected: FAIL — `Entity` and `EntityAllocator` not found

**Step 3: Implement Entity and EntityAllocator**

```rust
/// A unique entity identifier: 32-bit index + 32-bit generation packed into u64.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Entity(u64);

impl Entity {
    pub const DANGLING: Entity = Entity(u64::MAX);

    #[inline]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self((generation as u64) << 32 | index as u64)
    }

    #[inline]
    pub fn index(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

/// Allocates and recycles entity IDs with generational tracking.
pub(crate) struct EntityAllocator {
    generations: Vec<u32>,
    free_list: Vec<u32>,
}

impl EntityAllocator {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> Entity {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize];
            Entity::new(index, gen)
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            Entity::new(index, 0)
        }
    }

    pub fn dealloc(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx < self.generations.len() && self.generations[idx] == entity.generation() {
            self.generations[idx] = self.generations[idx].wrapping_add(1);
            self.free_list.push(entity.index());
            true
        } else {
            false
        }
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.generations.len() && self.generations[idx] == entity.generation()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski -- entity::tests`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/entity.rs
git commit -m "feat: implement Entity (u64 bit-packed) and EntityAllocator with generational recycling"
```

---

## Task 3: Component Trait & Registry

**Files:**
- Modify: `crates/minkowski/src/component.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct Position { x: f32, y: f32 }
    struct Velocity { dx: f32, dy: f32 }
    struct Health(u32);

    #[test]
    fn register_returns_sequential_ids() {
        let mut reg = ComponentRegistry::new();
        let a = reg.register::<Position>();
        let b = reg.register::<Velocity>();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn register_is_idempotent() {
        let mut reg = ComponentRegistry::new();
        let a = reg.register::<Position>();
        let b = reg.register::<Position>();
        assert_eq!(a, b);
    }

    #[test]
    fn id_lookup() {
        let mut reg = ComponentRegistry::new();
        assert_eq!(reg.id::<Position>(), None);
        reg.register::<Position>();
        assert_eq!(reg.id::<Position>(), Some(0));
    }

    #[test]
    fn info_has_correct_layout() {
        let mut reg = ComponentRegistry::new();
        let id = reg.register::<Position>();
        let info = reg.info(id);
        assert_eq!(info.layout, std::alloc::Layout::new::<Position>());
    }

    #[test]
    fn sparse_registration() {
        let mut reg = ComponentRegistry::new();
        let id = reg.register_sparse::<Health>();
        assert!(reg.is_sparse(id));
        assert!(!reg.is_sparse(reg.register::<Position>()));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- component::tests`
Expected: FAIL

**Step 3: Implement Component trait and ComponentRegistry**

```rust
use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;
use fixedbitset::FixedBitSet;

/// Marker trait for ECS components. Blanket-implemented for all eligible types.
pub trait Component: 'static + Send + Sync {}
impl<T: 'static + Send + Sync> Component for T {}

pub type ComponentId = usize;

pub(crate) struct ComponentInfo {
    pub id: ComponentId,
    pub name: &'static str,
    pub layout: Layout,
    pub drop_fn: Option<unsafe fn(*mut u8)>,
}

pub(crate) struct ComponentRegistry {
    by_type: HashMap<TypeId, ComponentId>,
    infos: Vec<ComponentInfo>,
    sparse_set: FixedBitSet,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            by_type: HashMap::new(),
            infos: Vec::new(),
            sparse_set: FixedBitSet::new(),
        }
    }

    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.by_type.get(&type_id) {
            return id;
        }
        let id = self.infos.len();
        let drop_fn = if std::mem::needs_drop::<T>() {
            Some(Self::drop_ptr::<T> as unsafe fn(*mut u8))
        } else {
            None
        };
        self.infos.push(ComponentInfo {
            id,
            name: std::any::type_name::<T>(),
            layout: Layout::new::<T>(),
            drop_fn,
        });
        self.by_type.insert(type_id, id);
        id
    }

    pub fn register_sparse<T: Component>(&mut self) -> ComponentId {
        let id = self.register::<T>();
        self.sparse_set.grow(id + 1);
        self.sparse_set.insert(id);
        id
    }

    pub fn id<T: Component>(&self) -> Option<ComponentId> {
        self.by_type.get(&TypeId::of::<T>()).copied()
    }

    pub fn info(&self, id: ComponentId) -> &ComponentInfo {
        &self.infos[id]
    }

    pub fn is_sparse(&self, id: ComponentId) -> bool {
        self.sparse_set.contains(id)
    }

    unsafe fn drop_ptr<T>(ptr: *mut u8) {
        std::ptr::drop_in_place(ptr as *mut T);
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- component::tests`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/component.rs
git commit -m "feat: implement Component trait (blanket) and ComponentRegistry with sparse support"
```

---

## Task 4: BlobVec

**Files:**
- Modify: `crates/minkowski/src/storage/blob_vec.rs`

This is the most critical unsafe code in the project. Every pointer operation must be correct.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;
    use std::sync::atomic::{AtomicUsize, Ordering};

    unsafe fn drop_u32(ptr: *mut u8) {
        std::ptr::drop_in_place(ptr as *mut u32);
    }

    #[test]
    fn push_and_get() {
        let mut bv = BlobVec::new(Layout::new::<u32>(), None);
        let val: u32 = 42;
        unsafe {
            bv.push(std::ptr::addr_of!(val) as *const u8);
            std::mem::forget(val);
            let ptr = bv.get_ptr(0) as *const u32;
            assert_eq!(*ptr, 42);
        }
        assert_eq!(bv.len(), 1);
    }

    #[test]
    fn push_multiple_and_grow() {
        let mut bv = BlobVec::new(Layout::new::<u64>(), None);
        for i in 0u64..100 {
            unsafe {
                bv.push(std::ptr::addr_of!(i) as *const u8);
            }
        }
        assert_eq!(bv.len(), 100);
        for i in 0u64..100 {
            unsafe {
                let val = *(bv.get_ptr(i as usize) as *const u64);
                assert_eq!(val, i);
            }
        }
    }

    #[test]
    fn swap_remove_middle() {
        let mut bv = BlobVec::new(Layout::new::<u32>(), Some(drop_u32));
        for val in [10u32, 20, 30] {
            unsafe { bv.push(std::ptr::addr_of!(val) as *const u8); }
        }
        // Remove middle element (20). Last element (30) takes its place.
        unsafe { bv.swap_remove(1); }
        assert_eq!(bv.len(), 2);
        unsafe {
            assert_eq!(*(bv.get_ptr(0) as *const u32), 10);
            assert_eq!(*(bv.get_ptr(1) as *const u32), 30);
        }
    }

    #[test]
    fn swap_remove_last() {
        let mut bv = BlobVec::new(Layout::new::<u32>(), Some(drop_u32));
        for val in [10u32, 20] {
            unsafe { bv.push(std::ptr::addr_of!(val) as *const u8); }
        }
        unsafe { bv.swap_remove(1); }
        assert_eq!(bv.len(), 1);
        unsafe { assert_eq!(*(bv.get_ptr(0) as *const u32), 10); }
    }

    #[test]
    fn swap_remove_no_drop() {
        let mut bv = BlobVec::new(Layout::new::<u32>(), Some(drop_u32));
        for val in [10u32, 20, 30] {
            unsafe { bv.push(std::ptr::addr_of!(val) as *const u8); }
        }
        unsafe { bv.swap_remove_no_drop(1); }
        assert_eq!(bv.len(), 2);
        unsafe {
            assert_eq!(*(bv.get_ptr(0) as *const u32), 10);
            assert_eq!(*(bv.get_ptr(1) as *const u32), 30);
        }
    }

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct DropCounter;
    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe fn drop_counter(ptr: *mut u8) {
        std::ptr::drop_in_place(ptr as *mut DropCounter);
    }

    #[test]
    fn drop_on_vec_drop() {
        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let mut bv = BlobVec::new(Layout::new::<DropCounter>(), Some(drop_counter));
            for _ in 0..5 {
                let val = DropCounter;
                unsafe {
                    bv.push(std::ptr::addr_of!(val) as *const u8);
                    std::mem::forget(val);
                }
            }
        } // BlobVec drops here
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn zst_support() {
        let mut bv = BlobVec::new(Layout::new::<()>(), None);
        for _ in 0..10 {
            let val = ();
            unsafe { bv.push(std::ptr::addr_of!(val) as *const u8); }
        }
        assert_eq!(bv.len(), 10);
        unsafe { bv.swap_remove(5); }
        assert_eq!(bv.len(), 9);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- storage::blob_vec::tests`
Expected: FAIL

**Step 3: Implement BlobVec**

```rust
use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// Type-erased growable array. The core column storage for archetypes.
pub(crate) struct BlobVec {
    item_layout: Layout,
    drop_fn: Option<unsafe fn(*mut u8)>,
    data: NonNull<u8>,
    len: usize,
    capacity: usize,
}

// Safety: BlobVec stores Component data which requires Send + Sync
unsafe impl Send for BlobVec {}
unsafe impl Sync for BlobVec {}

impl BlobVec {
    pub fn new(item_layout: Layout, drop_fn: Option<unsafe fn(*mut u8)>) -> Self {
        let (data, capacity) = if item_layout.size() == 0 {
            (NonNull::dangling(), usize::MAX)
        } else {
            (NonNull::dangling(), 0)
        };
        Self { item_layout, drop_fn, data, len: 0, capacity }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// # Safety
    /// `ptr` must point to valid initialized data matching `item_layout`.
    /// Caller must not drop the source data (ownership transfers to BlobVec).
    pub unsafe fn push(&mut self, ptr: *const u8) {
        if self.len == self.capacity {
            self.grow();
        }
        let dst = self.get_unchecked_ptr(self.len);
        if self.item_layout.size() > 0 {
            std::ptr::copy_nonoverlapping(ptr, dst, self.item_layout.size());
        }
        self.len += 1;
    }

    /// Swap-remove element at `row`, dropping it.
    /// # Safety: `row` must be < self.len
    pub unsafe fn swap_remove(&mut self, row: usize) {
        debug_assert!(row < self.len);
        let last = self.len - 1;
        let size = self.item_layout.size();
        if row != last && size > 0 {
            let row_ptr = self.get_unchecked_ptr(row);
            if let Some(drop_fn) = self.drop_fn {
                drop_fn(row_ptr);
            }
            let last_ptr = self.get_unchecked_ptr(last);
            std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size);
        } else if let Some(drop_fn) = self.drop_fn {
            drop_fn(self.get_unchecked_ptr(row));
        }
        self.len -= 1;
    }

    /// Swap-remove element at `row` WITHOUT dropping it.
    /// Used during archetype migration (data is moved, not dropped).
    /// # Safety: `row` must be < self.len
    pub unsafe fn swap_remove_no_drop(&mut self, row: usize) {
        debug_assert!(row < self.len);
        let last = self.len - 1;
        let size = self.item_layout.size();
        if row != last && size > 0 {
            let row_ptr = self.get_unchecked_ptr(row);
            let last_ptr = self.get_unchecked_ptr(last);
            std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size);
        }
        self.len -= 1;
    }

    /// # Safety: `row` must be < self.len
    #[inline]
    pub unsafe fn get_ptr(&self, row: usize) -> *mut u8 {
        debug_assert!(row < self.len);
        self.get_unchecked_ptr(row)
    }

    #[inline]
    fn get_unchecked_ptr(&self, row: usize) -> *mut u8 {
        if self.item_layout.size() == 0 {
            NonNull::dangling().as_ptr()
        } else {
            unsafe { self.data.as_ptr().add(row * self.item_layout.size()) }
        }
    }

    fn grow(&mut self) {
        let size = self.item_layout.size();
        if size == 0 { return; }

        let new_capacity = if self.capacity == 0 { 4 } else { self.capacity * 2 };
        let new_layout = Layout::from_size_align(
            size.checked_mul(new_capacity).expect("capacity overflow"),
            self.item_layout.align(),
        ).expect("invalid layout");

        let new_data = if self.capacity == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(
                size * self.capacity,
                self.item_layout.align(),
            ).unwrap();
            unsafe { alloc::realloc(self.data.as_ptr(), old_layout, new_layout.size()) }
        };

        self.data = NonNull::new(new_data)
            .unwrap_or_else(|| alloc::handle_alloc_error(new_layout));
        self.capacity = new_capacity;
    }
}

impl Drop for BlobVec {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.drop_fn {
            for i in 0..self.len {
                unsafe { drop_fn(self.get_unchecked_ptr(i)); }
            }
        }
        let size = self.item_layout.size();
        if size > 0 && self.capacity > 0 {
            let layout = Layout::from_size_align(
                size * self.capacity,
                self.item_layout.align(),
            ).unwrap();
            unsafe { alloc::dealloc(self.data.as_ptr(), layout); }
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- storage::blob_vec::tests`
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/storage/blob_vec.rs
git commit -m "feat: implement BlobVec — type-erased growable column storage with ZST support"
```

---

## Task 5: Archetype & Archetypes

**Files:**
- Modify: `crates/minkowski/src/storage/archetype.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;
    use crate::entity::Entity;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    fn setup_registry() -> ComponentRegistry {
        ComponentRegistry::new()
    }

    #[test]
    fn archetype_creation() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let vel_id = reg.register::<Vel>();
        let mut ids = vec![pos_id, vel_id];
        ids.sort();
        let arch = Archetype::new(ArchetypeId(0), &ids, &reg);
        assert!(arch.component_ids.contains(pos_id));
        assert!(arch.component_ids.contains(vel_id));
        assert_eq!(arch.len(), 0);
    }

    #[test]
    fn push_and_read_row() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let mut ids = vec![pos_id];
        ids.sort();
        let mut arch = Archetype::new(ArchetypeId(0), &ids, &reg);

        let entity = Entity::new(0, 0);
        let pos = Pos { x: 1.0, y: 2.0 };
        unsafe {
            let col = arch.component_index[&pos_id];
            arch.columns[col].push(std::ptr::addr_of!(pos) as *const u8);
            std::mem::forget(pos);
            arch.entities.push(entity);
        }
        assert_eq!(arch.len(), 1);

        unsafe {
            let col = arch.component_index[&pos_id];
            let ptr = arch.columns[col].get_ptr(0) as *const Pos;
            assert_eq!(*ptr, Pos { x: 1.0, y: 2.0 });
        }
    }

    #[test]
    fn archetypes_get_or_create() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let vel_id = reg.register::<Vel>();

        let mut archetypes = Archetypes::new();

        let mut ids = vec![pos_id, vel_id];
        ids.sort();
        let a1 = archetypes.get_or_create(&ids, &reg);
        let a2 = archetypes.get_or_create(&ids, &reg);
        assert_eq!(a1, a2); // idempotent

        let ids2 = vec![pos_id];
        let a3 = archetypes.get_or_create(&ids2, &reg);
        assert_ne!(a1, a3); // different component set = different archetype
    }

    #[test]
    fn archetypes_generation_bumps_on_create() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let mut archetypes = Archetypes::new();

        let gen_before = archetypes.generation();
        archetypes.get_or_create(&[pos_id], &reg);
        assert!(archetypes.generation() > gen_before);

        let gen_before = archetypes.generation();
        archetypes.get_or_create(&[pos_id], &reg); // same, no new archetype
        assert_eq!(archetypes.generation(), gen_before);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- storage::archetype::tests`
Expected: FAIL

**Step 3: Implement Archetype and Archetypes**

```rust
use std::collections::HashMap;
use fixedbitset::FixedBitSet;

use crate::component::{ComponentId, ComponentRegistry};
use crate::entity::Entity;
use super::blob_vec::BlobVec;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ArchetypeId(pub usize);

pub(crate) struct Archetype {
    pub id: ArchetypeId,
    /// Bitset of which ComponentIds this archetype contains.
    pub component_ids: FixedBitSet,
    /// Sorted list of ComponentIds (canonical key for archetype lookup).
    pub sorted_ids: Vec<ComponentId>,
    /// One BlobVec per component, in sorted_ids order.
    pub columns: Vec<BlobVec>,
    /// ComponentId -> index into columns.
    pub component_index: HashMap<ComponentId, usize>,
    /// Row -> Entity mapping.
    pub entities: Vec<Entity>,
}

impl Archetype {
    pub fn new(id: ArchetypeId, sorted_component_ids: &[ComponentId], registry: &ComponentRegistry) -> Self {
        let max_id = sorted_component_ids.iter().copied().max().unwrap_or(0);
        let mut bitset = FixedBitSet::with_capacity(max_id + 1);
        let mut columns = Vec::with_capacity(sorted_component_ids.len());
        let mut component_index = HashMap::new();

        for (col_idx, &comp_id) in sorted_component_ids.iter().enumerate() {
            bitset.insert(comp_id);
            let info = registry.info(comp_id);
            columns.push(BlobVec::new(info.layout, info.drop_fn));
            component_index.insert(comp_id, col_idx);
        }

        Self {
            id,
            component_ids: bitset,
            sorted_ids: sorted_component_ids.to_vec(),
            columns,
            component_index,
            entities: Vec::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

/// Collection of archetypes with lookup by component set.
pub(crate) struct Archetypes {
    pub archetypes: Vec<Archetype>,
    by_components: HashMap<Vec<ComponentId>, ArchetypeId>,
    generation: u64,
}

impl Archetypes {
    pub fn new() -> Self {
        Self {
            archetypes: Vec::new(),
            by_components: HashMap::new(),
            generation: 0,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Find or create an archetype for the given sorted component ID set.
    pub fn get_or_create(
        &mut self,
        sorted_ids: &[ComponentId],
        registry: &ComponentRegistry,
    ) -> ArchetypeId {
        if let Some(&id) = self.by_components.get(sorted_ids) {
            return id;
        }
        let id = ArchetypeId(self.archetypes.len());
        let archetype = Archetype::new(id, sorted_ids, registry);
        self.archetypes.push(archetype);
        self.by_components.insert(sorted_ids.to_vec(), id);
        self.generation += 1;
        id
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- storage::archetype::tests`
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/storage/archetype.rs
git commit -m "feat: implement Archetype (BlobVec columns + bitset) and Archetypes collection"
```

---

## Task 6: SparseStorage

**Files:**
- Modify: `crates/minkowski/src/storage/sparse.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entity;

    #[derive(Debug, PartialEq)]
    struct Marker(u32);

    #[test]
    fn insert_and_get() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert(0, e, Marker(42));
        assert_eq!(storage.get::<Marker>(0, e), Some(&Marker(42)));
    }

    #[test]
    fn get_missing_returns_none() {
        let storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn remove() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert(0, e, Marker(42));
        let removed = storage.remove::<Marker>(0, e);
        assert_eq!(removed, Some(Marker(42)));
        assert_eq!(storage.get::<Marker>(0, e), None);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- storage::sparse::tests`
Expected: FAIL

**Step 3: Implement SparseStorage**

```rust
use std::any::Any;
use std::collections::HashMap;

use crate::component::{Component, ComponentId};
use crate::entity::Entity;

/// Type-erased storage for sparse components. Each sparse ComponentId
/// gets a `HashMap<Entity, T>` behind a `Box<dyn Any>`.
pub(crate) struct SparseStorage {
    storages: HashMap<ComponentId, Box<dyn Any + Send + Sync>>,
}

impl SparseStorage {
    pub fn new() -> Self {
        Self { storages: HashMap::new() }
    }

    pub fn insert<T: Component>(&mut self, comp_id: ComponentId, entity: Entity, value: T) {
        let map = self.storages
            .entry(comp_id)
            .or_insert_with(|| Box::new(HashMap::<Entity, T>::new()))
            .downcast_mut::<HashMap<Entity, T>>()
            .expect("component type mismatch in sparse storage");
        map.insert(entity, value);
    }

    pub fn get<T: Component>(&self, comp_id: ComponentId, entity: Entity) -> Option<&T> {
        self.storages.get(&comp_id)?
            .downcast_ref::<HashMap<Entity, T>>()?
            .get(&entity)
    }

    pub fn get_mut<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<&mut T> {
        self.storages.get_mut(&comp_id)?
            .downcast_mut::<HashMap<Entity, T>>()?
            .get_mut(&entity)
    }

    pub fn remove<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<T> {
        self.storages.get_mut(&comp_id)?
            .downcast_mut::<HashMap<Entity, T>>()?
            .remove(&entity)
    }

    /// Remove all components for an entity (type-erased). Called during despawn.
    pub fn remove_entity(&mut self, entity: Entity) {
        for storage in self.storages.values_mut() {
            // Try removing as HashMap<Entity, _> — but we don't know T.
            // Store a type-erased remove function instead.
            // For Phase 1 simplicity: sparse despawn is a known limitation.
            // TODO: store erase-fn alongside the map for O(1) despawn cleanup
            let _ = storage;
            let _ = entity;
        }
    }
}
```

Note: `remove_entity` is a known Phase 1 limitation — sparse component cleanup during despawn requires either storing a type-erased remove function or iterating all sparse storages. For Phase 1, sparse components are opt-in and rare; we'll add proper cleanup before benchmarks.

**Step 4: Run tests**

Run: `cargo test -p minkowski -- storage::sparse::tests`
Expected: all 3 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/storage/sparse.rs
git commit -m "feat: implement SparseStorage — type-erased HashMap<Entity, T> per component"
```

---

## Task 7: Bundle Trait

**Files:**
- Create: `crates/minkowski/src/bundle.rs`
- Modify: `crates/minkowski/src/lib.rs` (add module)

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct A(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct B(f32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct C(u8);

    #[test]
    fn single_component_ids() {
        let mut reg = ComponentRegistry::new();
        let ids = <(A,)>::component_ids(&mut reg);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn pair_component_ids_sorted() {
        let mut reg = ComponentRegistry::new();
        let ids = <(A, B)>::component_ids(&mut reg);
        assert_eq!(ids.len(), 2);
        // Must be sorted
        assert!(ids[0] < ids[1] || ids[0] == ids[1]);
    }

    #[test]
    fn triple_component_ids() {
        let mut reg = ComponentRegistry::new();
        let ids = <(A, B, C)>::component_ids(&mut reg);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn put_writes_correct_data() {
        let mut reg = ComponentRegistry::new();
        let _ = <(A, B)>::component_ids(&mut reg);

        let bundle = (A(42), B(3.14));
        let mut written: Vec<(ComponentId, Vec<u8>)> = Vec::new();

        unsafe {
            bundle.put(&reg, &mut |comp_id, ptr, layout| {
                let mut data = vec![0u8; layout.size()];
                std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), layout.size());
                written.push((comp_id, data));
            });
        }

        assert_eq!(written.len(), 2);
        // Verify A's data
        let a_id = reg.id::<A>().unwrap();
        let a_entry = written.iter().find(|(id, _)| *id == a_id).unwrap();
        let a_val: A = unsafe { std::ptr::read(a_entry.1.as_ptr() as *const A) };
        assert_eq!(a_val, A(42));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- bundle::tests`
Expected: FAIL

**Step 3: Implement Bundle trait with tuple macros**

```rust
use std::alloc::Layout;
use crate::component::{Component, ComponentId, ComponentRegistry};

/// A collection of components that can be added to an entity.
/// Implemented for tuples of Components via macro.
///
/// # Safety
/// `put` must call `func` exactly once per component with valid pointers,
/// and must not drop the component data (ownership transfers to the caller).
pub unsafe trait Bundle: Send + Sync + 'static {
    fn component_ids(registry: &mut ComponentRegistry) -> Vec<ComponentId>;

    /// Write each component to the callback: (ComponentId, *const u8, Layout).
    /// Components are consumed — caller takes ownership via the pointer.
    unsafe fn put(
        self,
        registry: &ComponentRegistry,
        func: &mut dyn FnMut(ComponentId, *const u8, Layout),
    );
}

macro_rules! count {
    () => { 0usize };
    ($x:ident $(, $rest:ident)*) => { 1usize + count!($($rest),*) };
}

macro_rules! impl_bundle {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: Component),*> Bundle for ($($name,)*) {
            fn component_ids(registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                let mut ids = vec![$(registry.register::<$name>()),*];
                ids.sort_unstable();
                let expected = count!($($name),*);
                ids.dedup();
                assert_eq!(ids.len(), expected, "duplicate component types in bundle");
                ids
            }

            unsafe fn put(
                self,
                registry: &ComponentRegistry,
                func: &mut dyn FnMut(ComponentId, *const u8, Layout),
            ) {
                let ($($name,)*) = self;
                $(
                    let $name = std::mem::ManuallyDrop::new($name);
                    func(
                        registry.id::<$name>().unwrap(),
                        &*$name as *const $name as *const u8,
                        Layout::new::<$name>(),
                    );
                )*
            }
        }
    };
}

// Generate impls for tuples of arity 1..12
impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);
impl_bundle!(A, B, C, D, E, F, G, H, I);
impl_bundle!(A, B, C, D, E, F, G, H, I, J);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L);
```

Add `pub mod bundle;` to `lib.rs`.

**Step 4: Run tests**

Run: `cargo test -p minkowski -- bundle::tests`
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/bundle.rs crates/minkowski/src/lib.rs
git commit -m "feat: implement Bundle trait with tuple impls for arity 1-12"
```

---

## Task 8: World — Spawn, Despawn, Get

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn spawn_and_get() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn spawn_different_archetypes() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e1), None);
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e2), Some(&Vel { dx: 1.0, dy: 0.0 }));
    }

    #[test]
    fn despawn_and_is_alive() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        assert!(world.is_alive(e));
        assert!(world.despawn(e));
        assert!(!world.is_alive(e));
        assert_eq!(world.get::<Pos>(e), None);
    }

    #[test]
    fn entity_recycling() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.despawn(e1);
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        assert_eq!(e2.index(), e1.index());
        assert_ne!(e2.generation(), e1.generation());
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
    }

    #[test]
    fn get_mut() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        if let Some(pos) = world.get_mut::<Pos>(e) {
            pos.x = 10.0;
        }
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 2.0 }));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- world::tests`
Expected: FAIL

**Step 3: Implement World**

```rust
use crate::bundle::Bundle;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::{Entity, EntityAllocator};
use crate::storage::archetype::{ArchetypeId, Archetypes};
use crate::storage::sparse::SparseStorage;

#[derive(Clone, Copy)]
pub(crate) struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub row: usize,
}

pub struct World {
    pub(crate) entities: EntityAllocator,
    pub(crate) archetypes: Archetypes,
    pub(crate) components: ComponentRegistry,
    pub(crate) sparse: SparseStorage,
    pub(crate) entity_locations: Vec<Option<EntityLocation>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            archetypes: Archetypes::new(),
            components: ComponentRegistry::new(),
            sparse: SparseStorage::new(),
            entity_locations: Vec::new(),
        }
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let component_ids = B::component_ids(&mut self.components);
        let arch_id = self.archetypes.get_or_create(&component_ids, &self.components);
        let entity = self.entities.alloc();
        let index = entity.index() as usize;

        // Ensure locations vec is large enough
        if index >= self.entity_locations.len() {
            self.entity_locations.resize(index + 1, None);
        }

        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let col = archetype.component_index[&comp_id];
                archetype.columns[col].push(ptr);
            });
        }
        let row = archetype.entities.len();
        archetype.entities.push(entity);

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
        entity
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        let index = entity.index() as usize;
        let location = match self.entity_locations[index] {
            Some(loc) => loc,
            None => return false,
        };

        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        let row = location.row;

        // Swap-remove all columns (with drop)
        for col in &mut archetype.columns {
            unsafe { col.swap_remove(row); }
        }

        // Swap-remove from entities list
        archetype.entities.swap_remove(row);

        // Update swapped entity's location
        if row < archetype.entities.len() {
            let swapped = archetype.entities[row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: location.archetype_id,
                row,
            });
        }

        self.entity_locations[index] = None;
        self.entities.dealloc(entity);
        true
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let archetype = &self.archetypes.archetypes[location.archetype_id.0];
        let comp_id = self.components.id::<T>()?;

        if self.components.is_sparse(comp_id) {
            return self.sparse.get::<T>(comp_id, entity);
        }

        let col_idx = archetype.component_index.get(&comp_id)?;
        unsafe {
            let ptr = archetype.columns[*col_idx].get_ptr(location.row) as *const T;
            Some(&*ptr)
        }
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        let comp_id = self.components.id::<T>()?;

        if self.components.is_sparse(comp_id) {
            return self.sparse.get_mut::<T>(comp_id, entity);
        }

        let col_idx = *archetype.component_index.get(&comp_id)?;
        unsafe {
            let ptr = archetype.columns[col_idx].get_ptr(location.row) as *mut T;
            Some(&mut *ptr)
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- world::tests`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: implement World with spawn, despawn, get, get_mut, entity recycling"
```

---

## Task 9: Archetype Migration (insert / remove)

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Write failing tests**

Add to `world::tests`:

```rust
    #[test]
    fn insert_new_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, Vel { dx: 3.0, dy: 4.0 });
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn insert_overwrites_existing() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, Pos { x: 10.0, y: 20.0 });
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 20.0 }));
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        let removed = world.remove::<Vel>(e);
        assert_eq!(removed, Some(Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), None);
    }

    #[test]
    fn migration_preserves_other_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        let e3 = world.spawn((Pos { x: 3.0, y: 0.0 },));

        // Migrate e1 — e3 should swap into e1's old row
        world.insert(e1, Vel { dx: 1.0, dy: 0.0 });

        // All entities still accessible with correct data
        assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e1), Some(&Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Pos>(e3), Some(&Pos { x: 3.0, y: 0.0 }));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- world::tests`
Expected: 4 new tests FAIL (insert/remove not defined)

**Step 3: Implement insert and remove on World**

Add to `impl World`:

```rust
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        assert!(self.is_alive(entity), "entity is not alive");
        let index = entity.index() as usize;
        let location = self.entity_locations[index].unwrap();
        let comp_id = self.components.register::<T>();

        // If entity already has this component, overwrite in-place
        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        if src_arch.component_ids.contains(comp_id) {
            let col_idx = src_arch.component_index[&comp_id];
            unsafe {
                let ptr = src_arch.columns[col_idx].get_ptr(location.row) as *mut T;
                std::ptr::drop_in_place(ptr);
                std::ptr::write(ptr, component);
            }
            return;
        }

        // Compute target archetype: source components + new component
        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        let mut target_ids = src_arch.sorted_ids.clone();
        target_ids.push(comp_id);
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id = self.archetypes.get_or_create(&target_ids, &self.components);

        // Migrate shared columns using split_at_mut for safe double-mutable access
        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Copy shared columns
        for (&comp, &src_col) in &src_arch.component_index {
            if let Some(&tgt_col) = target_arch.component_index.get(&comp) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        // Write new component into target
        let tgt_col = target_arch.component_index[&comp_id];
        unsafe {
            let comp = std::mem::ManuallyDrop::new(component);
            target_arch.columns[tgt_col].push(&*comp as *const T as *const u8);
        }

        // Move entity tracking
        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        // Update swapped entity's location
        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let index = entity.index() as usize;
        let location = self.entity_locations[index]?;
        let comp_id = self.components.id::<T>()?;

        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        if !src_arch.component_ids.contains(comp_id) {
            return None;
        }

        // Read the component value before migration
        let removed = unsafe {
            let col_idx = src_arch.component_index[&comp_id];
            let ptr = src_arch.columns[col_idx].get_ptr(location.row) as *const T;
            std::ptr::read(ptr)
        };

        // Compute target archetype: source components - removed component
        let target_ids: Vec<ComponentId> = src_arch.sorted_ids.iter()
            .copied()
            .filter(|&id| id != comp_id)
            .collect();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        if target_ids.is_empty() {
            // Entity would have no components — just despawn the row from this archetype
            let arch = &mut self.archetypes.archetypes[src_arch_id.0];
            // swap_remove_no_drop for the removed component (already read)
            let removed_col = arch.component_index[&comp_id];
            unsafe { arch.columns[removed_col].swap_remove_no_drop(src_row); }
            // swap_remove with drop for remaining columns
            for (&cid, &col_idx) in &arch.component_index {
                if cid != comp_id {
                    unsafe { arch.columns[col_idx].swap_remove(src_row); }
                }
            }
            arch.entities.swap_remove(src_row);
            if src_row < arch.entities.len() {
                let swapped = arch.entities[src_row];
                self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                    archetype_id: src_arch_id,
                    row: src_row,
                });
            }
            // Entity still alive but has no archetype — store with empty archetype
            // For simplicity, create an empty archetype
            let empty_arch_id = self.archetypes.get_or_create(&[], &self.components);
            let empty_arch = &mut self.archetypes.archetypes[empty_arch_id.0];
            empty_arch.entities.push(entity);
            self.entity_locations[index] = Some(EntityLocation {
                archetype_id: empty_arch_id,
                row: empty_arch.entities.len() - 1,
            });
            return Some(removed);
        }

        let target_arch_id = self.archetypes.get_or_create(&target_ids, &self.components);

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Copy shared columns (skip removed component)
        for (&cid, &src_col) in &src_arch.component_index {
            if cid == comp_id {
                // swap_remove_no_drop: data was already read
                unsafe { src_arch.columns[src_col].swap_remove_no_drop(src_row); }
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
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });

        Some(removed)
    }
```

Also add this free function at the top of `world.rs`:

```rust
fn get_pair_mut(
    v: &mut Vec<crate::storage::archetype::Archetype>,
    a: usize,
    b: usize,
) -> (
    &mut crate::storage::archetype::Archetype,
    &mut crate::storage::archetype::Archetype,
) {
    assert_ne!(a, b, "cannot get mutable references to the same archetype");
    if a < b {
        let (left, right) = v.split_at_mut(b);
        (&mut left[a], &mut right[0])
    } else {
        let (left, right) = v.split_at_mut(a);
        (&mut right[0], &mut left[b])
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- world::tests`
Expected: all 9 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: implement archetype migration — insert and remove components"
```

---

## Task 10: WorldQuery Trait & Fetch Implementations

**Files:**
- Modify: `crates/minkowski/src/query/fetch.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;
    use crate::entity::Entity;
    use crate::storage::archetype::{Archetype, ArchetypeId};

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

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

        let pos = Pos { x: 1.0, y: 2.0 };
        unsafe {
            let col = arch.component_index[&pos_id];
            arch.columns[col].push(std::ptr::addr_of!(pos) as *const u8);
            std::mem::forget(pos);
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

        let pos = Pos { x: 1.0, y: 2.0 };
        unsafe {
            let col = arch.component_index[&pos_id];
            arch.columns[col].push(std::ptr::addr_of!(pos) as *const u8);
            std::mem::forget(pos);
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
        let pos = Pos { x: 0.0, y: 0.0 };
        unsafe {
            arch.columns[0].push(std::ptr::addr_of!(pos) as *const u8);
            std::mem::forget(pos);
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- query::fetch::tests`
Expected: FAIL

**Step 3: Implement WorldQuery trait and all primitive impls**

```rust
use std::marker::PhantomData;
use fixedbitset::FixedBitSet;

use crate::component::{Component, ComponentRegistry};
use crate::entity::Entity;
use crate::storage::archetype::Archetype;

/// A `Send + Sync` wrapper for raw pointers in query fetches.
pub(crate) struct ThinSlicePtr<T> {
    ptr: *mut T,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for ThinSlicePtr<T> {}
unsafe impl<T: Sync> Sync for ThinSlicePtr<T> {}

impl<T> ThinSlicePtr<T> {
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self { ptr, _marker: PhantomData }
    }
}

/// # Safety
/// Implementors must guarantee that `init_fetch` returns valid state for the
/// archetype, and `fetch` returns valid items for any row < archetype.len().
pub unsafe trait WorldQuery {
    type Item<'w>;
    type Fetch<'w>: Send + Sync;

    /// Returns a FixedBitSet with bits set for each required ComponentId.
    /// Archetypes missing any required component are skipped during query matching.
    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet;

    /// Initialize fetch state for the given archetype.
    /// Only called for archetypes whose component_ids is a superset of required_ids.
    fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Self::Fetch<'w>;

    /// Fetch the item at the given row.
    /// # Safety: row must be < archetype.len(), caller ensures no aliasing violations.
    unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w>;
}

// --- &T ---
unsafe impl<T: Component> WorldQuery for &T {
    type Item<'w> = &'w T;
    type Fetch<'w> = ThinSlicePtr<T>;

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        let mut bits = FixedBitSet::new();
        if let Some(id) = registry.id::<T>() {
            bits.grow(id + 1);
            bits.insert(id);
        }
        bits
    }

    fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> ThinSlicePtr<T> {
        let id = registry.id::<T>().expect("component not registered");
        let col_idx = archetype.component_index[&id];
        unsafe { ThinSlicePtr::new(archetype.columns[col_idx].get_ptr(0) as *mut T) }
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w T {
        &*fetch.ptr.add(row)
    }
}

// --- &mut T ---
unsafe impl<T: Component> WorldQuery for &mut T {
    type Item<'w> = &'w mut T;
    type Fetch<'w> = ThinSlicePtr<T>;

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry)
    }

    fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> ThinSlicePtr<T> {
        <&T>::init_fetch(archetype, registry)
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w mut T {
        &mut *fetch.ptr.add(row)
    }
}

// --- Entity ---
unsafe impl WorldQuery for Entity {
    type Item<'w> = Entity;
    type Fetch<'w> = ThinSlicePtr<Entity>;

    fn required_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new()
    }

    fn init_fetch<'w>(archetype: &'w Archetype, _registry: &ComponentRegistry) -> ThinSlicePtr<Entity> {
        unsafe { ThinSlicePtr::new(archetype.entities.as_ptr() as *mut Entity) }
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<Entity>, row: usize) -> Entity {
        *fetch.ptr.add(row)
    }
}

// --- Option<&T> ---
unsafe impl<T: Component> WorldQuery for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type Fetch<'w> = Option<ThinSlicePtr<T>>;

    fn required_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new() // optional — does not filter archetypes
    }

    fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Option<ThinSlicePtr<T>> {
        let id = registry.id::<T>()?;
        let col_idx = archetype.component_index.get(&id)?;
        Some(unsafe { ThinSlicePtr::new(archetype.columns[*col_idx].get_ptr(0) as *mut T) })
    }

    unsafe fn fetch<'w>(fetch: &Option<ThinSlicePtr<T>>, row: usize) -> Option<&'w T> {
        fetch.as_ref().map(|f| &*f.ptr.add(row))
    }
}

// --- Tuple impls ---
macro_rules! impl_world_query_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: WorldQuery),*> WorldQuery for ($($name,)*) {
            type Item<'w> = ($($name::Item<'w>,)*);
            type Fetch<'w> = ($($name::Fetch<'w>,)*);

            fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
                let mut bits = FixedBitSet::new();
                $(
                    let sub = $name::required_ids(registry);
                    bits.grow(sub.len());
                    bits.union_with(&sub);
                )*
                bits
            }

            fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Self::Fetch<'w> {
                ($($name::init_fetch(archetype, registry),)*)
            }

            unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WorldQuery>::fetch($name, row),)*)
            }
        }
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
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- query::fetch::tests`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/query/fetch.rs
git commit -m "feat: implement WorldQuery trait with impls for &T, &mut T, Entity, Option<&T>, tuples"
```

---

## Task 11: QueryIter — Sequential Iteration

**Files:**
- Modify: `crates/minkowski/src/query/iter.rs`
- Modify: `crates/minkowski/src/world.rs` (add `query` method)

**Step 1: Write failing tests**

In `crates/minkowski/src/query/iter.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::world::World;
    use crate::entity::Entity;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Health(u32);

    #[test]
    fn iterate_single_archetype() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 },));
        world.spawn((Pos { x: 3.0, y: 0.0 },));

        let positions: Vec<f32> = world.query::<&Pos>()
            .map(|p| p.x)
            .collect();
        assert_eq!(positions, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn iterate_multiple_archetypes() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));

        // Query for &Pos matches both archetypes
        let count = world.query::<&Pos>().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn iterate_filters_archetypes() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));

        // Query for (&Pos, &Vel) only matches the second archetype
        let count = world.query::<(&Pos, &Vel)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn iterate_with_entity() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));

        let entities: Vec<Entity> = world.query::<(Entity, &Pos)>()
            .map(|(e, _)| e)
            .collect();
        assert_eq!(entities, vec![e1, e2]);
    }

    #[test]
    fn iterate_empty() {
        let world = World::new();
        let count = world.query::<&Pos>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn mutate_during_iteration() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 10.0, dy: 0.0 }));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 20.0, dy: 0.0 }));

        for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
            pos.x += vel.dx;
        }

        let xs: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        assert_eq!(xs, vec![11.0, 22.0]);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- query::iter::tests`
Expected: FAIL

**Step 3: Implement QueryIter and World::query**

`crates/minkowski/src/query/iter.rs`:

```rust
use std::marker::PhantomData;
use super::fetch::WorldQuery;

/// Iterator over entities matching a query.
pub struct QueryIter<'w, Q: WorldQuery> {
    fetches: Vec<(Q::Fetch<'w>, usize)>, // (fetch_state, archetype_len)
    current_arch: usize,
    current_row: usize,
    _marker: PhantomData<&'w Q>,
}

impl<'w, Q: WorldQuery> QueryIter<'w, Q> {
    pub(crate) fn new(fetches: Vec<(Q::Fetch<'w>, usize)>) -> Self {
        Self {
            fetches,
            current_arch: 0,
            current_row: 0,
            _marker: PhantomData,
        }
    }
}

impl<'w, Q: WorldQuery> Iterator for QueryIter<'w, Q> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_arch >= self.fetches.len() {
                return None;
            }
            let (ref fetch, len) = self.fetches[self.current_arch];
            if self.current_row < len {
                let item = unsafe { Q::fetch(fetch, self.current_row) };
                self.current_row += 1;
                return Some(item);
            }
            self.current_arch += 1;
            self.current_row = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining: usize = self.fetches[self.current_arch..]
            .iter()
            .map(|(_, len)| *len)
            .sum::<usize>()
            .saturating_sub(self.current_row);
        (remaining, Some(remaining))
    }
}
```

Add `query` method to `World` in `world.rs`:

```rust
    pub fn query<Q: WorldQuery>(&self) -> QueryIter<'_, Q> {
        let required = Q::required_ids(&self.components);
        let fetches: Vec<_> = self.archetypes.archetypes.iter()
            .filter(|arch| {
                !arch.is_empty() && required.is_subset(&arch.component_ids)
            })
            .map(|arch| {
                let fetch = Q::init_fetch(arch, &self.components);
                (fetch, arch.len())
            })
            .collect();
        QueryIter::new(fetches)
    }
```

Add necessary imports to world.rs:
```rust
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- query::iter::tests`
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/query/iter.rs crates/minkowski/src/world.rs
git commit -m "feat: implement QueryIter with sequential iteration and World::query"
```

---

## Task 12: Parallel Iteration

**Files:**
- Modify: `crates/minkowski/src/query/iter.rs`

**Step 1: Write failing tests**

Add to `query::iter::tests`:

```rust
    #[test]
    fn par_for_each_updates_all() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut world = World::new();
        for i in 0..1000u32 {
            world.spawn((Pos { x: i as f32, y: 0.0 },));
        }

        let sum = AtomicU32::new(0);
        world.query::<&Pos>().par_for_each(|pos| {
            sum.fetch_add(pos.x as u32, Ordering::Relaxed);
        });
        // Sum of 0..1000 = 999 * 1000 / 2 = 499500
        assert_eq!(sum.load(Ordering::SeqCst), 499500);
    }

    #[test]
    fn par_for_each_mutation() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Pos { x: i as f32, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        }

        world.query::<(&mut Pos, &Vel)>().par_for_each(|(pos, vel)| {
            pos.x += vel.dx;
        });

        let xs: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        for (i, x) in xs.iter().enumerate() {
            assert_eq!(*x, i as f32 + 1.0);
        }
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- query::iter::tests::par`
Expected: FAIL — `par_for_each` not defined

**Step 3: Implement par_for_each**

Add to `QueryIter` impl in `iter.rs`:

```rust
use rayon::prelude::*;

impl<'w, Q: WorldQuery> QueryIter<'w, Q> {
    /// Execute `f` for each matched entity in parallel using rayon.
    /// Parallelizes across rows within each archetype.
    pub fn par_for_each<F>(self, f: F)
    where
        F: Fn(Q::Item<'_>) + Send + Sync,
    {
        for (fetch, len) in &self.fetches {
            (0..*len).into_par_iter().for_each(|row| {
                let item = unsafe { Q::fetch(fetch, row) };
                f(item);
            });
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- query::iter::tests`
Expected: all 8 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/query/iter.rs
git commit -m "feat: implement par_for_each — rayon-parallel query iteration"
```

---

## Task 13: CommandBuffer

**Files:**
- Modify: `crates/minkowski/src/command.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn command_spawn() {
        let mut world = World::new();
        let mut cmds = CommandBuffer::new();
        cmds.spawn((Pos { x: 1.0, y: 2.0 },));
        cmds.spawn((Pos { x: 3.0, y: 4.0 },));
        cmds.apply(&mut world);

        let count = world.query::<&Pos>().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn command_despawn() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let mut cmds = CommandBuffer::new();
        cmds.despawn(e);
        cmds.apply(&mut world);

        assert!(!world.is_alive(e));
    }

    #[test]
    fn command_insert() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let mut cmds = CommandBuffer::new();
        cmds.insert(e, Vel { dx: 5.0, dy: 0.0 });
        cmds.apply(&mut world);

        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 5.0, dy: 0.0 }));
    }

    #[test]
    fn command_remove() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 5.0, dy: 0.0 }));
        let mut cmds = CommandBuffer::new();
        cmds.remove::<Vel>(e);
        cmds.apply(&mut world);

        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 0.0 }));
    }

    #[test]
    fn commands_during_iteration() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Pos { x: i as f32, y: 0.0 },));
        }

        let mut cmds = CommandBuffer::new();
        for (entity, pos) in world.query::<(crate::entity::Entity, &Pos)>() {
            if pos.x > 2.0 {
                cmds.despawn(entity);
            }
        }
        cmds.apply(&mut world);

        let count = world.query::<&Pos>().count();
        assert_eq!(count, 3); // 0.0, 1.0, 2.0 remain
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski -- command::tests`
Expected: FAIL

**Step 3: Implement CommandBuffer**

```rust
use crate::bundle::Bundle;
use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

/// Deferred mutation buffer. Records commands during iteration,
/// applies them all at once to &mut World.
pub struct CommandBuffer {
    commands: Vec<Box<dyn FnOnce(&mut World) + Send>>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self { commands: Vec::new() }
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) {
        self.commands.push(Box::new(move |world| {
            world.spawn(bundle);
        }));
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.commands.push(Box::new(move |world| {
            world.despawn(entity);
        }));
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.commands.push(Box::new(move |world| {
            world.insert(entity, component);
        }));
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.commands.push(Box::new(move |world| {
            world.remove::<T>(entity);
        }));
    }

    pub fn apply(self, world: &mut World) {
        for command in self.commands {
            command(world);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski -- command::tests`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/command.rs
git commit -m "feat: implement CommandBuffer — deferred world mutations via closure queue"
```

---

## Task 14: Criterion Benchmarks vs hecs

**Files:**
- Create: `crates/minkowski/benches/spawn.rs`
- Create: `crates/minkowski/benches/iterate.rs`
- Create: `crates/minkowski/benches/parallel.rs`
- Create: `crates/minkowski/benches/add_remove.rs`
- Create: `crates/minkowski/benches/fragmented.rs`

**Step 1: Create common benchmark component types**

All benchmarks share these types. Define them in each file (no shared module for simplicity):

```rust
#[derive(Clone, Copy)]
struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)]
struct Velocity { dx: f32, dy: f32 }
#[derive(Clone, Copy)]
struct Health(f32);
#[derive(Clone, Copy)]
struct Damage(f32);
```

**Step 2: Implement spawn benchmark**

`crates/minkowski/benches/spawn.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Velocity { dx: f32, dy: f32 }

fn spawn_minkowski(c: &mut Criterion) {
    c.bench_function("minkowski/spawn_10k", |b| {
        b.iter(|| {
            let mut world = minkowski::World::new();
            for i in 0..10_000 {
                world.spawn((
                    Position { x: i as f32, y: 0.0 },
                    Velocity { dx: 1.0, dy: 0.0 },
                ));
            }
        });
    });
}

fn spawn_hecs(c: &mut Criterion) {
    c.bench_function("hecs/spawn_10k", |b| {
        b.iter(|| {
            let mut world = hecs::World::new();
            for i in 0..10_000 {
                world.spawn((
                    Position { x: i as f32, y: 0.0 },
                    Velocity { dx: 1.0, dy: 0.0 },
                ));
            }
        });
    });
}

criterion_group!(benches, spawn_minkowski, spawn_hecs);
criterion_main!(benches);
```

**Step 3: Implement iterate benchmark**

`crates/minkowski/benches/iterate.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Velocity { dx: f32, dy: f32 }

fn iterate_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    for i in 0..10_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("minkowski/iterate_10k", |b| {
        b.iter(|| {
            for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        });
    });
}

fn iterate_hecs(c: &mut Criterion) {
    let mut world = hecs::World::new();
    for i in 0..10_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("hecs/iterate_10k", |b| {
        b.iter(|| {
            for (_id, (pos, vel)) in world.query_mut::<(&mut Position, &Velocity)>() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        });
    });
}

criterion_group!(benches, iterate_minkowski, iterate_hecs);
criterion_main!(benches);
```

**Step 4: Implement parallel benchmark**

`crates/minkowski/benches/parallel.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Velocity { dx: f32, dy: f32 }

fn parallel_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    for i in 0..100_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("minkowski/parallel_100k", |b| {
        b.iter(|| {
            world.query::<(&mut Position, &Velocity)>().par_for_each(|(pos, vel)| {
                pos.x += vel.dx;
                pos.y += vel.dy;
            });
        });
    });
}

fn sequential_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    for i in 0..100_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));
    }

    c.bench_function("minkowski/sequential_100k", |b| {
        b.iter(|| {
            for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        });
    });
}

criterion_group!(benches, parallel_minkowski, sequential_minkowski);
criterion_main!(benches);
```

**Step 5: Implement add_remove benchmark**

`crates/minkowski/benches/add_remove.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }
#[derive(Clone, Copy)] struct Velocity { dx: f32, dy: f32 }
#[derive(Clone, Copy)] struct Health(f32);

fn add_remove_minkowski(c: &mut Criterion) {
    c.bench_function("minkowski/add_remove_1k", |b| {
        b.iter_batched(
            || {
                let mut world = minkowski::World::new();
                let entities: Vec<_> = (0..1000)
                    .map(|i| world.spawn((Position { x: i as f32, y: 0.0 },)))
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for &e in &entities {
                    world.insert(e, Health(100.0));
                }
                for &e in &entities {
                    world.remove::<Health>(e);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn add_remove_hecs(c: &mut Criterion) {
    c.bench_function("hecs/add_remove_1k", |b| {
        b.iter_batched(
            || {
                let mut world = hecs::World::new();
                let entities: Vec<_> = (0..1000)
                    .map(|i| world.spawn((Position { x: i as f32, y: 0.0 },)))
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for &e in &entities {
                    world.insert_one(e, Health(100.0)).unwrap();
                }
                for &e in &entities {
                    world.remove_one::<Health>(e).unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, add_remove_minkowski, add_remove_hecs);
criterion_main!(benches);
```

**Step 6: Implement fragmented benchmark**

`crates/minkowski/benches/fragmented.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

// 20 different component types for archetype fragmentation
macro_rules! define_components {
    ($($name:ident),*) => { $( #[derive(Clone, Copy)] struct $name(f32); )* };
}
define_components!(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19);

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }

fn fragmented_iterate_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    // Spawn 500 entities across 20 different archetypes, all with Position
    for i in 0..500 {
        match i % 20 {
            0 => { world.spawn((Position { x: i as f32, y: 0.0 }, C0(0.0))); },
            1 => { world.spawn((Position { x: i as f32, y: 0.0 }, C1(0.0))); },
            2 => { world.spawn((Position { x: i as f32, y: 0.0 }, C2(0.0))); },
            3 => { world.spawn((Position { x: i as f32, y: 0.0 }, C3(0.0))); },
            4 => { world.spawn((Position { x: i as f32, y: 0.0 }, C4(0.0))); },
            5 => { world.spawn((Position { x: i as f32, y: 0.0 }, C5(0.0))); },
            6 => { world.spawn((Position { x: i as f32, y: 0.0 }, C6(0.0))); },
            7 => { world.spawn((Position { x: i as f32, y: 0.0 }, C7(0.0))); },
            8 => { world.spawn((Position { x: i as f32, y: 0.0 }, C8(0.0))); },
            9 => { world.spawn((Position { x: i as f32, y: 0.0 }, C9(0.0))); },
            10 => { world.spawn((Position { x: i as f32, y: 0.0 }, C10(0.0))); },
            11 => { world.spawn((Position { x: i as f32, y: 0.0 }, C11(0.0))); },
            12 => { world.spawn((Position { x: i as f32, y: 0.0 }, C12(0.0))); },
            13 => { world.spawn((Position { x: i as f32, y: 0.0 }, C13(0.0))); },
            14 => { world.spawn((Position { x: i as f32, y: 0.0 }, C14(0.0))); },
            15 => { world.spawn((Position { x: i as f32, y: 0.0 }, C15(0.0))); },
            16 => { world.spawn((Position { x: i as f32, y: 0.0 }, C16(0.0))); },
            17 => { world.spawn((Position { x: i as f32, y: 0.0 }, C17(0.0))); },
            18 => { world.spawn((Position { x: i as f32, y: 0.0 }, C18(0.0))); },
            _ => { world.spawn((Position { x: i as f32, y: 0.0 }, C19(0.0))); },
        }
    }

    c.bench_function("minkowski/fragmented_500", |b| {
        b.iter(|| {
            for pos in world.query::<&mut Position>() {
                pos.x += 1.0;
            }
        });
    });
}

criterion_group!(benches, fragmented_iterate_minkowski);
criterion_main!(benches);
```

**Step 7: Run benchmarks to verify they work**

Run: `cargo bench -p minkowski -- spawn`
Expected: benchmark output with timing for both minkowski and hecs

**Step 8: Commit**

```bash
git add crates/minkowski/benches/
git commit -m "feat: add criterion benchmark suite — spawn, iterate, parallel, add_remove, fragmented vs hecs"
```

---

## Task 15: Boids Example

**Files:**
- Create: `crates/minkowski/examples/boids.rs`

**Step 1: Implement terminal boids simulation**

This exercises every API path: spawn, iterate, mutate, insert, remove, despawn, CommandBuffer.

```rust
//! Terminal boids simulation — exercises the full minkowski ECS API.
//! Run: cargo run -p minkowski --example boids --release

use minkowski::{Entity, World, CommandBuffer};

#[derive(Clone, Copy)]
struct Position { x: f32, y: f32 }

#[derive(Clone, Copy)]
struct Velocity { dx: f32, dy: f32 }

#[derive(Clone, Copy)]
struct Acceleration { ax: f32, ay: f32 }

const BOID_COUNT: usize = 200;
const FRAMES: usize = 10_000;
const WORLD_SIZE: f32 = 100.0;
const MAX_SPEED: f32 = 2.0;
const SEPARATION_RADIUS: f32 = 5.0;
const ALIGNMENT_RADIUS: f32 = 10.0;
const COHESION_RADIUS: f32 = 15.0;
const DT: f32 = 0.016;

fn main() {
    let mut world = World::new();

    // Spawn initial boids
    for i in 0..BOID_COUNT {
        let angle = (i as f32 / BOID_COUNT as f32) * std::f32::consts::TAU;
        let r = WORLD_SIZE * 0.3;
        world.spawn((
            Position {
                x: WORLD_SIZE / 2.0 + r * angle.cos(),
                y: WORLD_SIZE / 2.0 + r * angle.sin(),
            },
            Velocity {
                dx: angle.sin() * MAX_SPEED * 0.5,
                dy: -angle.cos() * MAX_SPEED * 0.5,
            },
            Acceleration { ax: 0.0, ay: 0.0 },
        ));
    }

    let mut total_speed = 0.0f32;

    for frame in 0..FRAMES {
        // 1. Collect positions for neighbor queries
        let boids: Vec<(Entity, Position, Velocity)> = world
            .query::<(Entity, &Position, &Velocity)>()
            .map(|(e, p, v)| (e, *p, *v))
            .collect();

        // 2. Compute boid forces → write accelerations via CommandBuffer
        let mut cmds = CommandBuffer::new();
        for &(entity, pos, vel) in &boids {
            let (mut sep_x, mut sep_y) = (0.0f32, 0.0f32);
            let (mut ali_dx, mut ali_dy, mut ali_count) = (0.0f32, 0.0f32, 0u32);
            let (mut coh_x, mut coh_y, mut coh_count) = (0.0f32, 0.0f32, 0u32);

            for &(_other_e, other_pos, other_vel) in &boids {
                let dx = other_pos.x - pos.x;
                let dy = other_pos.y - pos.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 0.001 { continue; }

                if dist < SEPARATION_RADIUS {
                    sep_x -= dx / dist;
                    sep_y -= dy / dist;
                }
                if dist < ALIGNMENT_RADIUS {
                    ali_dx += other_vel.dx;
                    ali_dy += other_vel.dy;
                    ali_count += 1;
                }
                if dist < COHESION_RADIUS {
                    coh_x += other_pos.x;
                    coh_y += other_pos.y;
                    coh_count += 1;
                }
            }

            let mut ax = sep_x * 1.5;
            let mut ay = sep_y * 1.5;

            if ali_count > 0 {
                ax += (ali_dx / ali_count as f32 - vel.dx) * 0.5;
                ay += (ali_dy / ali_count as f32 - vel.dy) * 0.5;
            }
            if coh_count > 0 {
                ax += (coh_x / coh_count as f32 - pos.x) * 0.01;
                ay += (coh_y / coh_count as f32 - pos.y) * 0.01;
            }

            cmds.insert(entity, Acceleration { ax, ay });
        }
        cmds.apply(&mut world);

        // 3. Integrate velocity from acceleration
        for (vel, acc) in world.query::<(&mut Velocity, &Acceleration)>() {
            vel.dx += acc.ax * DT;
            vel.dy += acc.ay * DT;
            let speed = (vel.dx * vel.dx + vel.dy * vel.dy).sqrt();
            if speed > MAX_SPEED {
                vel.dx = vel.dx / speed * MAX_SPEED;
                vel.dy = vel.dy / speed * MAX_SPEED;
            }
        }

        // 4. Integrate position from velocity + wrap around
        for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
            pos.x += vel.dx * DT;
            pos.y += vel.dy * DT;
            pos.x = pos.x.rem_euclid(WORLD_SIZE);
            pos.y = pos.y.rem_euclid(WORLD_SIZE);
        }

        // 5. Compute stats
        if frame % 1000 == 0 || frame == FRAMES - 1 {
            let entity_count = world.query::<&Position>().count();
            let mut speed_sum = 0.0f32;
            for vel in world.query::<&Velocity>() {
                speed_sum += (vel.dx * vel.dx + vel.dy * vel.dy).sqrt();
            }
            let avg_speed = speed_sum / entity_count as f32;
            total_speed += avg_speed;
            println!(
                "frame {frame:>5}: entities={entity_count}, avg_speed={avg_speed:.3}"
            );
        }
    }

    println!("Done. Overall avg speed: {:.3}", total_speed / ((FRAMES / 1000 + 1) as f32));
}
```

**Step 2: Run it**

Run: `cargo run -p minkowski --example boids --release`
Expected: Prints 11 lines (frame 0, 1000, 2000, ..., 9000, 9999) with entity count = 200 and reasonable avg_speed values.

**Step 3: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat: add terminal boids example — integration test exercising full API"
```

---

## Final Verification

After all tasks complete:

1. **Full test suite**: `cargo test -p minkowski` — all tests pass
2. **No warnings**: `cargo clippy -p minkowski` — clean
3. **Benchmarks run**: `cargo bench -p minkowski` — outputs timing data
4. **Boids runs**: `cargo run -p minkowski --example boids --release` — 10K frames complete

**Benchmark targets:**
- Within 2x of hecs on spawn, iterate, add_remove, fragmented
- Faster than sequential on parallel (100K entities)
