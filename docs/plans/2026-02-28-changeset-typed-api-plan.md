# EnumChangeSet Typed API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add typed safe helper methods to `EnumChangeSet` so external users can record mutations without dealing with raw `ComponentId`, pointers, or `Layout`.

**Architecture:** Add `World::component_id<T>()` and `World::register_component<T>()` as the component-ID access layer. Add `EnumChangeSet::insert<T>()`, `remove<T>()`, and `spawn<B: Bundle>()` as typed wrappers over the existing raw `record_*` methods. Re-export `ComponentId` from `lib.rs`.

**Tech Stack:** Rust, minkowski ECS crate (no new dependencies)

---

### Task 1: Add `World::component_id` and `World::register_component`

**Files:**
- Modify: `crates/minkowski/src/world.rs` (after `next_tick` at line 76, before `spawn` at line 78)

**Step 1: Write failing tests**

Add to the existing `#[cfg(test)] mod tests` in `world.rs`:

```rust
#[test]
fn component_id_returns_none_for_unregistered() {
    let world = World::new();
    assert_eq!(world.component_id::<Position>(), None);
}

#[test]
fn register_component_returns_id_and_subsequent_lookup_works() {
    let mut world = World::new();
    let id = world.register_component::<Position>();
    assert_eq!(world.component_id::<Position>(), Some(id));
}

#[test]
fn register_component_is_idempotent() {
    let mut world = World::new();
    let a = world.register_component::<Position>();
    let b = world.register_component::<Position>();
    assert_eq!(a, b);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- component_id`
Expected: FAIL — `component_id` and `register_component` not found

**Step 3: Implement**

Add to `impl World` in `world.rs`, between `next_tick` and `spawn`:

```rust
/// Look up the `ComponentId` for a type. Returns `None` if the type has
/// never been spawned or registered.
pub fn component_id<T: Component>(&self) -> Option<ComponentId> {
    self.components.id::<T>()
}

/// Register a component type, returning its `ComponentId`. Idempotent —
/// returns the existing id if already registered.
pub fn register_component<T: Component>(&mut self) -> ComponentId {
    self.components.register::<T>()
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- component_id register_component`
Expected: PASS (3 new tests)

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add World::component_id and register_component public API"
```

---

### Task 2: Re-export `ComponentId` from `lib.rs`

**Files:**
- Modify: `crates/minkowski/src/lib.rs` (line 25, after existing `pub use` block)

**Step 1: Add re-export**

Add after `pub use world::World;`:

```rust
pub use component::ComponentId;
```

**Step 2: Verify it compiles**

Run: `cargo test -p minkowski --lib`
Expected: PASS (no regressions)

**Step 3: Commit**

```bash
git add crates/minkowski/src/lib.rs
git commit -m "feat: re-export ComponentId from crate root"
```

---

### Task 3: Add typed `insert` and `remove` helpers on `EnumChangeSet`

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (new `impl` block after line 180, before the `Default` impl)

**Step 1: Write failing tests**

Add to the `#[cfg(test)] mod tests` block in `changeset.rs`, after the existing `round_trip_forward_reverse_forward` test:

```rust
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- typed_insert typed_remove`
Expected: FAIL — methods `insert` and `remove` not found on `EnumChangeSet`

**Step 3: Implement**

Add a new `impl EnumChangeSet` block after line 180 (after `record_remove`), before `impl Default`:

```rust
// ── Typed safe helpers ─────────────────────────────────────────

impl EnumChangeSet {
    /// Record inserting a component on an entity. Auto-registers the
    /// component type. Safe wrapper over `record_insert`.
    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        let comp_id = world.register_component::<T>();
        let layout = Layout::new::<T>();
        let value = std::mem::ManuallyDrop::new(value);
        self.record_insert(
            entity,
            comp_id,
            &*value as *const T as *const u8,
            layout,
        );
    }

    /// Record removing a component from an entity. Auto-registers the
    /// component type.
    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        let comp_id = world.register_component::<T>();
        self.record_remove(entity, comp_id);
    }
}
```

Note: `use crate::component::Component;` is already available via `use crate::component::ComponentId;` being in scope plus `Component` being imported through `World`. Check the imports at the top of `changeset.rs` — add `use crate::component::Component;` if not present.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- typed_insert typed_remove`
Expected: PASS (3 new tests)

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat: add typed insert/remove helpers on EnumChangeSet"
```

---

### Task 4: Add typed `spawn` helper on `EnumChangeSet`

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (add to the typed helpers `impl` block from Task 3)

**Step 1: Write failing test**

Add to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn typed_spawn_and_reverse() {
    let mut world = World::new();
    let entity = world.entities.alloc();

    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world, entity, (Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

    let reverse = cs.apply(&mut world);
    assert!(world.is_alive(entity));
    assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
    assert_eq!(world.get::<Vel>(entity), Some(&Vel { dx: 3.0, dy: 4.0 }));

    let _ = reverse.apply(&mut world);
    assert!(!world.is_alive(entity));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- typed_spawn`
Expected: FAIL — `spawn_bundle` not found

**Step 3: Implement**

Add to the typed helpers `impl` block:

```rust
    /// Record spawning an entity with a bundle of components. Auto-registers
    /// all component types in the bundle.
    pub fn spawn_bundle<B: Bundle>(&mut self, world: &mut World, entity: Entity, bundle: B) {
        let _ids = B::component_ids(&mut world.components);
        let mut components = Vec::new();
        unsafe {
            bundle.put(&world.components, &mut |comp_id, ptr, layout| {
                let offset = self.arena.alloc(ptr, layout);
                components.push((comp_id, offset, layout));
            });
        }
        self.mutations.push(Mutation::Spawn {
            entity,
            components,
        });
    }
```

Add `use crate::bundle::Bundle;` to the imports at the top of `changeset.rs` if not present.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- typed_spawn`
Expected: PASS

**Step 5: Run all tests + clippy**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat: add typed spawn_bundle helper on EnumChangeSet"
```

---

### Task 5: Migrate existing changeset tests to typed API

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (tests starting at line 587)

Migrate these tests to use the typed helpers where possible. Tests that exercise the raw API directly (arena tests, record_and_count, record_insert_stores_data, record_spawn_stores_components) stay as-is since they test the raw internals.

**Step 1: Rewrite apply tests**

Replace these tests with typed equivalents:

- `apply_spawn_and_reverse_despawns` → use `cs.spawn_bundle(&mut world, entity, (Pos{..}, Vel{..}))`
- `apply_insert_new_and_reverse_removes` → use `cs.insert::<Vel>(&mut world, e, Vel{..})`
- `apply_insert_overwrite_and_reverse_restores` → use `cs.insert::<Pos>(&mut world, e, Pos{..})`
- `apply_remove_and_reverse_reinserts` → use `cs.remove::<Vel>(&mut world, e)`
- `round_trip_forward_reverse_forward` → use `cs.insert::<Vel>(&mut world, e, Vel{..})`

Key changes per test:
- Remove `let vel_id = world.components.register::<Vel>();` lines
- Replace `cs.record_insert(e, vel_id, &vel as *const Vel as *const u8, Layout::new::<Vel>())` with `cs.insert::<Vel>(&mut world, e, vel)`
- Replace `cs.record_remove(e, vel_id)` with `cs.remove::<Vel>(&mut world, e)`
- Replace raw `cs.record_spawn(entity, &[...])` with `cs.spawn_bundle(&mut world, entity, (pos, vel))`
- `apply_despawn_and_reverse_respawns` stays as-is (it uses `record_despawn` which doesn't take ComponentId)

**Step 2: Run all tests**

Run: `cargo test -p minkowski --lib`
Expected: PASS (all existing + new tests)

**Step 3: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "refactor: migrate changeset tests to typed API"
```

---

### Task 6: Add external integration test

**Files:**
- Create: `crates/minkowski/tests/changeset_external.rs`

This test lives *outside* the crate — it's the test that would have caught the original visibility bug. It can only access `pub` items.

**Step 1: Write integration test**

```rust
//! Integration test: exercises EnumChangeSet typed API from outside the crate.
//! This test would have caught the original ComponentId visibility bug.

use minkowski::{EnumChangeSet, Entity, World, ComponentId};

#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos { x: f32, y: f32 }

#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel { dx: f32, dy: f32 }

#[test]
fn typed_insert_remove_roundtrip() {
    let mut world = World::new();
    let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

    // Insert via typed API
    let mut cs = EnumChangeSet::new();
    cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });
    let reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));

    // Reverse undoes the insert
    let _ = reverse.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), None);
}

#[test]
fn typed_remove_roundtrip() {
    let mut world = World::new();
    let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

    let mut cs = EnumChangeSet::new();
    cs.remove::<Vel>(&mut world, e);
    let reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), None);

    let _ = reverse.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
}

#[test]
fn component_id_lookup() {
    let mut world = World::new();
    assert_eq!(world.component_id::<Pos>(), None);

    let id: ComponentId = world.register_component::<Pos>();
    assert_eq!(world.component_id::<Pos>(), Some(id));
}
```

**Step 2: Run integration test**

Run: `cargo test -p minkowski --test changeset_external`
Expected: PASS

**Step 3: Run full test suite + clippy**

Run: `cargo test -p minkowski && cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/minkowski/tests/changeset_external.rs
git commit -m "test: add external integration test for EnumChangeSet typed API"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update Key Traits section**

In the `EnumChangeSet` description under Deferred Mutation, add mention of the typed helpers. Update the pub API list in Key Conventions to include `ComponentId`.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for EnumChangeSet typed API and ComponentId export"
```
