# Dynamic Reducers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add runtime-flexible dynamic reducers to ReducerRegistry — builder-declared upper bounds, binary-search component lookup, debug-asserted enforcement.

**Architecture:** Extends the existing unified `Vec<ReducerEntry>` with a new `ReducerKind::Dynamic` variant. DynamicCtx borrows disjoint Tx fields (same pattern as EntityMut/Spawner). Component IDs pre-resolved at registration; binary search by TypeId at runtime. Writes buffered via `EnumChangeSet::insert_raw` (no `&mut World` needed).

**Tech Stack:** Rust, minkowski ECS (same crate — `crates/minkowski/src/reducer.rs`)

**Design Doc:** `docs/plans/2026-03-04-dynamic-reducers-design.md`

---

### Task 1: DynamicResolved — pre-resolved component lookup table

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (after `ResolvedComponents` struct, ~line 42)

**Step 1: Write the failing test**

Add at the bottom of the existing `#[cfg(test)] mod tests` block in `reducer.rs`:

```rust
#[test]
fn dynamic_resolved_lookup() {
    use std::any::TypeId;
    let entries = vec![
        (TypeId::of::<u32>(), ComponentId(0)),
        (TypeId::of::<f64>(), ComponentId(2)),
        (TypeId::of::<i64>(), ComponentId(1)),
    ];
    let resolved = DynamicResolved::new(entries, Access::empty(), Default::default());
    assert_eq!(resolved.lookup::<u32>(), Some(ComponentId(0)));
    assert_eq!(resolved.lookup::<f64>(), Some(ComponentId(2)));
    assert_eq!(resolved.lookup::<i64>(), Some(ComponentId(1)));
    assert_eq!(resolved.lookup::<u8>(), None);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- dynamic_resolved_lookup`
Expected: FAIL — `DynamicResolved` type does not exist.

**Step 3: Write minimal implementation**

Add after `ResolvedComponents` (~line 45):

```rust
use std::any::TypeId;
use std::collections::HashSet;

/// Pre-resolved component lookup for dynamic reducers.
/// Entries sorted by TypeId for O(log n) binary search at runtime.
pub(crate) struct DynamicResolved {
    /// Sorted by TypeId for binary search.
    entries: Vec<(TypeId, ComponentId)>,
    access: Access,
    spawn_bundles: HashSet<TypeId>,
}

impl DynamicResolved {
    pub(crate) fn new(
        mut entries: Vec<(TypeId, ComponentId)>,
        access: Access,
        spawn_bundles: HashSet<TypeId>,
    ) -> Self {
        entries.sort_by_key(|(tid, _)| *tid);
        entries.dedup_by_key(|(tid, _)| *tid);
        Self { entries, access, spawn_bundles }
    }

    /// Look up a component's pre-resolved ID by its TypeId.
    /// Returns None if the type was not declared in the builder.
    pub(crate) fn lookup<T: 'static>(&self) -> Option<ComponentId> {
        let type_id = TypeId::of::<T>();
        self.entries
            .binary_search_by_key(&type_id, |(tid, _)| *tid)
            .ok()
            .map(|idx| self.entries[idx].1)
    }

    pub(crate) fn access(&self) -> &Access {
        &self.access
    }

    pub(crate) fn has_spawn_bundle<B: 'static>(&self) -> bool {
        self.spawn_bundles.contains(&TypeId::of::<B>())
    }
}
```

Note: add `use std::any::TypeId;` and `use std::collections::HashSet;` to the imports at the top of the file if not already present. `TypeId` is already imported if `Any` is — check. `HashSet` needs to be added to the existing `use std::collections::HashMap;` line → `use std::collections::{HashMap, HashSet};`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- dynamic_resolved_lookup`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): add DynamicResolved — pre-resolved lookup table for dynamic reducers"
```

---

### Task 2: DynamicCtx — runtime context for dynamic reducer closures

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (after DynamicResolved)

**Step 1: Write failing tests**

Add three tests:

```rust
#[test]
fn dynamic_ctx_read() {
    let mut world = World::new();
    let e = world.spawn((42u32, 1.0f64));
    let comp_id = world.components.lookup_id(TypeId::of::<u32>()).unwrap();
    let entries = vec![(TypeId::of::<u32>(), comp_id)];
    let resolved = DynamicResolved::new(entries, Access::empty(), Default::default());

    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx {
        world: &world,
        changeset: &mut cs,
        allocated: &mut allocated,
        resolved: &resolved,
    };
    let val = ctx.read::<u32>(e);
    assert_eq!(*val, 42);
}

#[test]
fn dynamic_ctx_write_buffers() {
    let mut world = World::new();
    let e = world.spawn((42u32,));
    let comp_id = world.components.lookup_id(TypeId::of::<u32>()).unwrap();

    let mut access = Access::empty();
    access.add_write(comp_id);
    let entries = vec![(TypeId::of::<u32>(), comp_id)];
    let resolved = DynamicResolved::new(entries, access, Default::default());

    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    {
        let mut ctx = DynamicCtx {
            world: &world,
            changeset: &mut cs,
            allocated: &mut allocated,
            resolved: &resolved,
        };
        ctx.write::<u32>(e, 99);
    }
    // Not yet applied
    assert_eq!(*world.get::<u32>(e).unwrap(), 42);
    // Apply changeset
    let _reverse = cs.apply(&mut world);
    assert_eq!(*world.get::<u32>(e).unwrap(), 99);
}

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_read_undeclared_panics() {
    let mut world = World::new();
    let e = world.spawn((42u32,));
    let resolved = DynamicResolved::new(vec![], Access::empty(), Default::default());
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx {
        world: &world,
        changeset: &mut cs,
        allocated: &mut allocated,
        resolved: &resolved,
    };
    let _ = ctx.read::<u32>(e); // panics: u32 not declared
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- dynamic_ctx`
Expected: FAIL — `DynamicCtx` does not exist.

**Step 3: Write implementation**

```rust
/// Runtime context for dynamic reducer closures.
/// Provides read/write/spawn operations validated against the builder's
/// declared access set. Borrows disjoint Tx fields (same pattern as EntityMut).
pub struct DynamicCtx<'a> {
    world: &'a World,
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    resolved: &'a DynamicResolved,
}

impl<'a> DynamicCtx<'a> {
    /// Read a component value. Panics if the type was not declared in the builder.
    /// Panics if the entity does not have the component.
    pub fn read<T: Component>(&self, entity: Entity) -> &T {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "DynamicCtx::read::<{}>: type not declared in builder",
                std::any::type_name::<T>()
            )
        });
        self.world.get_by_id::<T>(entity, comp_id).unwrap_or_else(|| {
            panic!(
                "DynamicCtx::read::<{}>: entity {:?} does not have component",
                std::any::type_name::<T>(),
                entity
            )
        })
    }

    /// Read a component value, returning None if the entity doesn't have it.
    /// Panics if the type was not declared in the builder.
    pub fn try_read<T: Component>(&self, entity: Entity) -> Option<&T> {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "DynamicCtx::try_read::<{}>: type not declared in builder",
                std::any::type_name::<T>()
            )
        });
        self.world.get_by_id::<T>(entity, comp_id)
    }

    /// Buffer a component write. Panics if the type was not declared in the builder.
    /// In debug builds, also asserts the type was declared with `can_write` (not just `can_read`).
    pub fn write<T: Component>(&mut self, entity: Entity, value: T) {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "DynamicCtx::write::<{}>: type not declared in builder",
                std::any::type_name::<T>()
            )
        });
        debug_assert!(
            self.resolved.access().writes()[comp_id.index()],
            "DynamicCtx::write::<{}>: type declared as read-only (use can_write in builder)",
            std::any::type_name::<T>()
        );
        self.changeset.insert_raw::<T>(entity, comp_id, value);
    }

    /// Buffer a component write only if the entity has the component.
    /// Returns true if the write was buffered.
    /// Panics if the type was not declared in the builder.
    pub fn try_write<T: Component>(&mut self, entity: Entity, value: T) -> bool {
        let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
            panic!(
                "DynamicCtx::try_write::<{}>: type not declared in builder",
                std::any::type_name::<T>()
            )
        });
        debug_assert!(
            self.resolved.access().writes()[comp_id.index()],
            "DynamicCtx::try_write::<{}>: type declared as read-only",
            std::any::type_name::<T>()
        );
        if self.world.get_by_id::<T>(entity, comp_id).is_some() {
            self.changeset.insert_raw::<T>(entity, comp_id, value);
            true
        } else {
            false
        }
    }

    /// Spawn an entity with the given bundle. Panics if the bundle type was
    /// not declared with `can_spawn` in the builder.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        debug_assert!(
            self.resolved.has_spawn_bundle::<B>(),
            "DynamicCtx::spawn::<{}>: bundle not declared with can_spawn in builder",
            std::any::type_name::<B>()
        );
        let entity = self.world.entities.reserve();
        self.allocated.push(entity);
        self.changeset
            .spawn_bundle_raw(entity, &self.world.components, bundle);
        entity
    }
}
```

Note: `ComponentId` needs an `index()` method that returns `usize` for the `FixedBitSet` lookup. Check if this exists. If not, it's just `self.0 as usize` — but since `Access::reads()` / `Access::writes()` return `&FixedBitSet` and `FixedBitSet` indexes by `usize`, we need `comp_id.0 as usize`. The existing code uses `comp_id` directly with `FixedBitSet` — check the convention. If `ComponentId` is `pub struct ComponentId(pub(crate) u32)`, then use `comp_id.0 as usize` since we're in the same crate.

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- dynamic_ctx`
Expected: PASS (3 tests)

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): add DynamicCtx — read/write/spawn with runtime access validation"
```

---

### Task 3: DynamicReducerId and ReducerKind::Dynamic

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (near existing ReducerId/QueryReducerId and ReducerKind)

**Step 1: Write failing test**

```rust
#[test]
fn dynamic_reducer_id_is_distinct_type() {
    let id = DynamicReducerId(0);
    // Ensure it doesn't accidentally convert to ReducerId or QueryReducerId
    let _: DynamicReducerId = id;
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- dynamic_reducer_id`
Expected: FAIL — `DynamicReducerId` does not exist.

**Step 3: Write implementation**

Add after `QueryReducerId` (~line 318):

```rust
/// Opaque identifier for a dynamic reducer in a [`ReducerRegistry`].
/// Indices into the registry's `dynamic_reducers` vec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynamicReducerId(pub(crate) usize);
```

Add the type alias for dynamic adapter closures (near existing EntityAdapter/ScheduledAdapter):

```rust
type DynamicAdapter =
    Box<dyn Fn(&mut DynamicCtx, &dyn Any) + Send + Sync>;
```

Extend `ReducerRegistry` struct to add the dynamic vec:

```rust
pub struct ReducerRegistry {
    reducers: Vec<ReducerEntry>,
    dynamic_reducers: Vec<DynamicReducerEntry>,
    by_name: HashMap<&'static str, ReducerSlot>,
}

enum ReducerSlot {
    Unified(usize),
    Dynamic(usize),
}
```

Update `ReducerRegistry::new()`:

```rust
pub fn new() -> Self {
    Self {
        reducers: Vec::new(),
        dynamic_reducers: Vec::new(),
        by_name: HashMap::new(),
    }
}
```

Add the `DynamicReducerEntry` struct:

```rust
struct DynamicReducerEntry {
    #[allow(dead_code)]
    name: &'static str,
    access: Access,
    resolved: DynamicResolved,
    closure: DynamicAdapter,
}
```

Update all existing `by_name` usage:
- `push_entry`: change `self.by_name.insert(name, id)` → `self.by_name.insert(name, ReducerSlot::Unified(id))`
- `reducer_id_by_name`: change `let &idx = self.by_name.get(name)?` → match on `ReducerSlot::Unified(idx)`
- `query_reducer_id_by_name`: same pattern

**Step 4: Run ALL tests (not just new one)**

Run: `cargo test -p minkowski --lib`
Expected: ALL existing tests still pass, plus the new one.

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): add DynamicReducerId, DynamicReducerEntry, ReducerSlot — infrastructure for dynamic reducers"
```

---

### Task 4: DynamicReducerBuilder — builder pattern for registration

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write failing test**

```rust
#[test]
fn dynamic_builder_registers_and_calls() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();

    let id = reducers
        .dynamic("test_dynamic", &mut world)
        .can_read::<u32>()
        .can_write::<f64>()
        .build(|ctx: &mut DynamicCtx, args: &u32| {
            let val = ctx.read::<u32>(args.clone().into());
            // Just reading — this test verifies registration
            let _ = val;
        });

    // Verify the access set
    let access = reducers.dynamic_access(id);
    let u32_id = world.components.lookup_id(TypeId::of::<u32>()).unwrap();
    let f64_id = world.components.lookup_id(TypeId::of::<f64>()).unwrap();
    assert!(access.reads()[u32_id.0 as usize]);
    assert!(access.writes()[f64_id.0 as usize]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- dynamic_builder`
Expected: FAIL — `dynamic` method does not exist on ReducerRegistry.

**Step 3: Write implementation**

Add the builder struct:

```rust
/// Builder for registering dynamic reducers with runtime-flexible access.
pub struct DynamicReducerBuilder<'a> {
    registry: &'a mut ReducerRegistry,
    world: &'a mut World,
    name: &'static str,
    access: Access,
    entries: Vec<(TypeId, ComponentId)>,
    spawn_bundles: HashSet<TypeId>,
}

impl<'a> DynamicReducerBuilder<'a> {
    /// Declare read access to component T.
    pub fn can_read<T: Component>(mut self) -> Self {
        let comp_id = self.world.register_component::<T>();
        self.entries.push((TypeId::of::<T>(), comp_id));
        self.access.add_read(comp_id);
        self
    }

    /// Declare write access to component T.
    pub fn can_write<T: Component>(mut self) -> Self {
        let comp_id = self.world.register_component::<T>();
        self.entries.push((TypeId::of::<T>(), comp_id));
        self.access.add_write(comp_id);
        self
    }

    /// Declare spawn access for bundle B.
    pub fn can_spawn<B: Bundle>(mut self) -> Self {
        let ids = B::component_ids(&mut self.world.components);
        for &id in &ids {
            self.entries.push((
                // We don't have TypeId per component from Bundle — only ComponentId.
                // Bundle components are already in entries if declared with can_read/can_write.
                // For spawn, we add writes for all bundle components.
                // Note: we can't get TypeId from ComponentId, so we just add the writes.
                // The spawn_bundles HashSet tracks the Bundle TypeId for debug assertion.
                // Skip TypeId tracking per component — spawn is validated at Bundle level.
                std::any::TypeId::of::<()>(), // placeholder, won't match lookups
                id,
            ));
            self.access.add_write(id);
        }
        self.spawn_bundles.insert(TypeId::of::<B>());
        self
    }

    /// Finalize registration with the given closure.
    pub fn build<Args, F>(self, f: F) -> DynamicReducerId
    where
        Args: 'static,
        F: Fn(&mut DynamicCtx, &Args) + Send + Sync + 'static,
    {
        let resolved = DynamicResolved::new(self.entries, self.access.clone(), self.spawn_bundles);
        let adapter: DynamicAdapter = Box::new(move |ctx, args_any| {
            let args = args_any.downcast_ref::<Args>().unwrap_or_else(|| {
                panic!(
                    "dynamic reducer args type mismatch: expected {}",
                    std::any::type_name::<Args>()
                )
            });
            f(ctx, args);
        });

        let id = self.registry.dynamic_reducers.len();
        let name = self.name;
        if let Some(_) = self.registry.by_name.get(name) {
            panic!(
                "ReducerRegistry: duplicate reducer name '{}' (already registered)",
                name
            );
        }
        self.registry.by_name.insert(name, ReducerSlot::Dynamic(id));
        self.registry.dynamic_reducers.push(DynamicReducerEntry {
            name,
            access: self.access,
            resolved,
            closure: adapter,
        });
        DynamicReducerId(id)
    }
}
```

Add the `dynamic` entry-point method on `ReducerRegistry`:

```rust
impl ReducerRegistry {
    /// Start building a dynamic reducer with runtime-flexible access.
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
        }
    }
}
```

Note on `can_spawn`: the approach above adds placeholder TypeIds for spawn components, which won't interfere with binary search since `TypeId::of::<()>()` won't be looked up by read/write. The spawn validation uses `spawn_bundles` HashSet, not the entries vec. A cleaner approach: only add ComponentId writes to Access (for conflict detection), don't push TypeId entries for spawn components.

Revised `can_spawn`:

```rust
pub fn can_spawn<B: Bundle>(mut self) -> Self {
    let ids = B::component_ids(&mut self.world.components);
    for &id in &ids {
        self.access.add_write(id);
    }
    self.spawn_bundles.insert(TypeId::of::<B>());
    self
}
```

(Remove the `entries.push` for spawn components — they don't need TypeId lookup since `DynamicCtx::spawn` uses Bundle trait directly, not per-component lookup.)

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- dynamic_builder`
Expected: PASS

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): add DynamicReducerBuilder — builder pattern for dynamic reducer registration"
```

---

### Task 5: Dispatch — dynamic_call, dynamic_id_by_name, dynamic_access

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn dynamic_call_reads_and_writes() {
    use crate::transaction::Sequential;

    let mut world = World::new();
    let e = world.spawn((42u32, 1.0f64));
    let mut reducers = ReducerRegistry::new();

    let id = reducers
        .dynamic("inc_f64", &mut world)
        .can_read::<u32>()
        .can_write::<f64>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            let u = *ctx.read::<u32>(*entity);
            ctx.write(*entity, u as f64 + 100.0);
        });

    let strategy = Sequential;
    reducers
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();

    assert_eq!(*world.get::<f64>(e).unwrap(), 142.0);
}

#[test]
fn dynamic_id_by_name_lookup() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    let id = reducers
        .dynamic("my_dyn", &mut world)
        .can_read::<u32>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});

    assert_eq!(reducers.dynamic_id_by_name("my_dyn"), Some(id));
    assert_eq!(reducers.dynamic_id_by_name("nonexistent"), None);
    // Static lookup should not find dynamic reducers
    assert_eq!(reducers.reducer_id_by_name("my_dyn"), None);
}

#[test]
fn dynamic_access_for_conflict_detection() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    let id = reducers
        .dynamic("writes_u32", &mut world)
        .can_write::<u32>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});

    let access = reducers.dynamic_access(id);
    let u32_id = world.components.lookup_id(TypeId::of::<u32>()).unwrap();
    assert!(access.writes()[u32_id.0 as usize]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- dynamic_call dynamic_id_by_name dynamic_access_for`
Expected: FAIL — methods don't exist.

**Step 3: Write implementation**

Add to `impl ReducerRegistry`:

```rust
/// Call a dynamic reducer with a chosen transaction strategy.
pub fn dynamic_call<S: Transact, Args: 'static>(
    &self,
    strategy: &S,
    world: &mut World,
    id: DynamicReducerId,
    args: &Args,
) -> Result<(), Conflict> {
    let entry = &self.dynamic_reducers[id.0];
    let closure = &entry.closure;
    let resolved = &entry.resolved;
    let access = &entry.access;

    strategy.transact(world, access, |tx, world| {
        let (changeset, allocated) = tx.reducer_parts();
        let world_ref: &World = world;
        let mut ctx = DynamicCtx {
            world: world_ref,
            changeset,
            allocated,
            resolved,
        };
        closure(&mut ctx, args);
    })
}

/// Look up a dynamic reducer by name.
pub fn dynamic_id_by_name(&self, name: &str) -> Option<DynamicReducerId> {
    match self.by_name.get(name)? {
        ReducerSlot::Dynamic(idx) => Some(DynamicReducerId(*idx)),
        ReducerSlot::Unified(_) => None,
    }
}

/// Access metadata for a dynamic reducer.
pub fn dynamic_access(&self, id: DynamicReducerId) -> &Access {
    &self.dynamic_reducers[id.0].access
}
```

Also update `reducer_id_by_name` and `query_reducer_id_by_name` to use `ReducerSlot::Unified`:

```rust
pub fn reducer_id_by_name(&self, name: &str) -> Option<ReducerId> {
    match self.by_name.get(name)? {
        ReducerSlot::Unified(idx) => {
            let idx = *idx;
            match &self.reducers[idx].kind {
                ReducerKind::EntityTransactional(_) => Some(ReducerId(idx)),
                ReducerKind::Scheduled(_) => None,
            }
        }
        ReducerSlot::Dynamic(_) => None,
    }
}

pub fn query_reducer_id_by_name(&self, name: &str) -> Option<QueryReducerId> {
    match self.by_name.get(name)? {
        ReducerSlot::Unified(idx) => {
            let idx = *idx;
            match &self.reducers[idx].kind {
                ReducerKind::Scheduled(_) => Some(QueryReducerId(idx)),
                ReducerKind::EntityTransactional(_) => None,
            }
        }
        ReducerSlot::Dynamic(_) => None,
    }
}
```

And update `push_entry` to use `ReducerSlot::Unified`:

```rust
fn push_entry(...) -> ReducerId {
    let id = self.reducers.len();
    if let Some(_) = self.by_name.get(name) {
        panic!("ReducerRegistry: duplicate reducer name '{}'...", name);
    }
    self.by_name.insert(name, ReducerSlot::Unified(id));
    // ...rest unchanged
}
```

**Step 4: Run ALL tests**

Run: `cargo test -p minkowski --lib`
Expected: ALL tests pass (existing + new).

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): add dynamic_call, dynamic_id_by_name, dynamic_access — dispatch for dynamic reducers"
```

---

### Task 6: Debug assertion tests

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (tests only)

**Step 1: Write debug-mode tests**

```rust
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "read-only")]
fn dynamic_ctx_write_on_read_only_panics_in_debug() {
    let mut world = World::new();
    let e = world.spawn((42u32,));
    let comp_id = world.components.lookup_id(TypeId::of::<u32>()).unwrap();

    let mut access = Access::empty();
    access.add_read(comp_id); // read only, not write
    let entries = vec![(TypeId::of::<u32>(), comp_id)];
    let resolved = DynamicResolved::new(entries, access, Default::default());

    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx {
        world: &world,
        changeset: &mut cs,
        allocated: &mut allocated,
        resolved: &resolved,
    };
    ctx.write::<u32>(e, 99); // should panic in debug: read-only
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "bundle not declared")]
fn dynamic_ctx_spawn_undeclared_bundle_panics_in_debug() {
    let mut world = World::new();
    world.register_component::<u32>();
    let resolved = DynamicResolved::new(vec![], Access::empty(), Default::default());

    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx {
        world: &world,
        changeset: &mut cs,
        allocated: &mut allocated,
        resolved: &resolved,
    };
    ctx.spawn((42u32,)); // should panic: bundle not declared
}
```

**Step 2: Run tests**

Run: `cargo test -p minkowski --lib -- dynamic_ctx_write_on_read dynamic_ctx_spawn_undeclared`
Expected: PASS (both tests should panic as expected in debug mode).

**Step 3: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "test(reducer): debug assertion tests for DynamicCtx — read-only write, undeclared spawn"
```

---

### Task 7: Public exports

**Files:**
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Add exports**

Add `DynamicCtx`, `DynamicReducerId` to the existing `pub use reducer::{...};` line:

```rust
pub use reducer::{
    ComponentSet, Contains, DynamicCtx, DynamicReducerId, EntityMut, EntityRef,
    QueryMut, QueryReducerId, QueryRef, ReducerId, ReducerRegistry, Spawner,
};
```

Note: `DynamicReducerBuilder` is returned by `ReducerRegistry::dynamic()` so it needs to be pub too (even though users don't name the type directly). Check if it's already pub — yes, the struct is `pub struct DynamicReducerBuilder`. Add to exports:

```rust
pub use reducer::{
    ComponentSet, Contains, DynamicCtx, DynamicReducerBuilder, DynamicReducerId,
    EntityMut, EntityRef, QueryMut, QueryReducerId, QueryRef, ReducerId,
    ReducerRegistry, Spawner,
};
```

**Step 2: Run clippy + tests**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test -p minkowski --lib`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/minkowski/src/lib.rs
git commit -m "feat(reducer): export DynamicCtx, DynamicReducerBuilder, DynamicReducerId"
```

---

### Task 8: External example — dynamic reducer with conditional access

**Files:**
- Modify: `examples/examples/reducer.rs`

**Step 1: Add dynamic reducer demo to existing example**

Add a new section after the existing reducer demos. The example should demonstrate the motivating use case: conditional component access based on runtime state.

```rust
// ── Dynamic reducers: runtime-flexible access ────────────────────────
println!("\n── Dynamic Reducers ──────────────────────────────────");

// Register a dynamic reducer that conditionally writes Shield based on HP
let shield_id = reducers
    .dynamic("conditional_shield", &mut world)
    .can_read::<Health>()
    .can_read::<Energy>()
    .can_write::<Energy>()
    .can_write::<Shield>()
    .build(|ctx: &mut DynamicCtx, entity: &Entity| {
        let hp = ctx.read::<Health>(*entity);
        let energy = ctx.read::<Energy>(*entity);
        if energy.mana >= 50.0 {
            ctx.write(*entity, Energy { mana: energy.mana - 50.0 });
            if hp.hp < 30.0 {
                ctx.write(*entity, Shield { active: true, duration: 5.0 });
            }
        }
    });

// Call via Sequential strategy
let strategy = Sequential;
reducers.dynamic_call(&strategy, &mut world, shield_id, &some_entity).unwrap();

// Name-based lookup for network dispatch
let found = reducers.dynamic_id_by_name("conditional_shield");
assert_eq!(found, Some(shield_id));

// Access metadata for scheduler conflict detection
let access = reducers.dynamic_access(shield_id);
println!("  dynamic reducer 'conditional_shield' registered with access metadata");
println!("  conflicts detectable via access.conflicts_with()");
```

Adapt to use actual component types and entities from the existing example. The example already defines `Health`, `Attack`, `Defense` (or similar) — add `Energy`, `Shield` if not present.

**Step 2: Run the example**

Run: `cargo run -p minkowski-examples --example reducer --release`
Expected: Prints dynamic reducer output without errors.

**Step 3: Run clippy on examples**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS — verifies pub visibility is correct from external crate.

**Step 4: Commit**

```bash
git add examples/examples/reducer.rs
git commit -m "feat(examples): dynamic reducer demo — conditional access, name lookup, conflict detection"
```

---

### Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update Key Conventions pub list**

Add `DynamicCtx`, `DynamicReducerBuilder`, `DynamicReducerId` to the `pub` list in the first bullet of Key Conventions.

**Step 2: Update reducer example command**

Update the reducer example description if needed.

**Step 3: Run format check**

No code changes, just documentation.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add dynamic reducer types to CLAUDE.md pub list"
```

---

## Verification Checklist

After all tasks, run the full suite:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p minkowski
cargo run -p minkowski-examples --example reducer --release
```

All four must pass before creating a PR.

## Semantic Review (per CLAUDE.md checklist)

1. **Can this be called with the wrong World?** No — DynamicCtx borrows `&World` from the strategy.transact closure, same World that was passed in. Builder takes `&mut World` at registration time.

2. **Can Drop observe inconsistent state?** No new Drop impls. DynamicCtx is a borrowed view, drops are trivial. Changeset drops are handled by existing EnumChangeSet drop safety.

3. **Can two threads reach this through `&self`?** No — `dynamic_call` takes `&self` on ReducerRegistry (immutable ref to entry) but routes through `strategy.transact()` which manages concurrency. DynamicCtx borrows the Tx's disjoint fields inside the closure.

4. **Does dedup/merge/collapse preserve the strongest invariant?** DynamicResolved::new sorts and dedups by TypeId. If the same TypeId appears with different ComponentIds (impossible — ComponentRegistry is 1:1), dedup keeps the first. The Access bitset merges correctly via `add_read`/`add_write` (idempotent on FixedBitSet).

5. **What happens if this is abandoned halfway through?** If the builder is dropped before `build()`, no entry is registered — safe. If the transact closure panics mid-execution, the Tx's Drop pushes allocated entities to the orphan queue — existing safety net.

6. **Can a type bound be violated by a legal generic instantiation?** `Args: 'static` prevents non-static references. `F: Fn(&mut DynamicCtx, &Args) + Send + Sync + 'static` prevents non-Send/Sync closures. Component bound on read/write ensures T: Send + Sync.

7. **Does the API surface of this handle permit any operation not covered by the Access bitset?** `read` checks `lookup` (type must be declared). `write` additionally debug_asserts `writes()[comp_id]`. `spawn` debug_asserts `spawn_bundles`. All operations are bounded by what the builder declared, which is exactly what the Access bitset reflects.
