# Extended Reducers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structural mutations (despawn, remove) and dynamic iteration (for_each) to the reducer system.

**Architecture:** Structural mutations buffer into EnumChangeSet — the scheduler treats them as column writes. `can_despawn()` adds a blanket conflict flag to Access. Dynamic iteration via `DynamicCtx::for_each<Q>()` uses typed ReadOnlyWorldQuery codepaths with runtime access validation against builder-declared bounds. Changed<T> support via per-reducer `Arc<AtomicU64>` tick state, updated post-commit.

**Tech Stack:** Rust, fixedbitset (Access bitsets), std::sync::atomic (tick state)

---

### Task 1: Access despawn flag

Add a `despawns: bool` field to `Access` and update `conflicts_with` to implement the blanket conflict rule.

**Files:**
- Modify: `crates/minkowski/src/access.rs`

**Step 1: Write the failing tests**

Add these tests to the existing `mod tests` in `access.rs`:

```rust
#[test]
fn despawn_flag_default_false() {
    let a = Access::empty();
    assert!(!a.despawns());
}

#[test]
fn set_despawns() {
    let mut a = Access::empty();
    a.set_despawns();
    assert!(a.despawns());
}

#[test]
fn has_any_access_empty() {
    let a = Access::empty();
    assert!(!a.has_any_access());
}

#[test]
fn has_any_access_with_read() {
    let mut a = Access::empty();
    a.add_read(0);
    assert!(a.has_any_access());
}

#[test]
fn has_any_access_with_write() {
    let mut a = Access::empty();
    a.add_write(0);
    assert!(a.has_any_access());
}

#[test]
fn despawn_conflicts_with_reader() {
    let mut a = Access::empty();
    a.set_despawns();
    let mut b = Access::empty();
    b.add_read(0);
    assert!(a.conflicts_with(&b));
    assert!(b.conflicts_with(&a));
}

#[test]
fn despawn_conflicts_with_writer() {
    let mut a = Access::empty();
    a.set_despawns();
    let mut b = Access::empty();
    b.add_write(0);
    assert!(a.conflicts_with(&b));
    assert!(b.conflicts_with(&a));
}

#[test]
fn despawn_no_conflict_with_empty() {
    let mut a = Access::empty();
    a.set_despawns();
    let b = Access::empty();
    assert!(!a.conflicts_with(&b));
    assert!(!b.conflicts_with(&a));
}

#[test]
fn two_despawners_with_disjoint_reads_conflict() {
    // Both despawn + read something — each side's despawn flag
    // hits the other side's non-empty component access.
    let mut a = Access::empty();
    a.set_despawns();
    a.add_read(0);
    let mut b = Access::empty();
    b.set_despawns();
    b.add_read(1);
    assert!(a.conflicts_with(&b));
    assert!(b.conflicts_with(&a));
}

#[test]
fn despawn_of_preserves_flag() {
    let mut world = World::new();
    let a = Access::of::<(&Pos,)>(&mut world);
    assert!(!a.despawns());
    // Access::of can't set despawns — it's a builder-only operation
}

#[test]
fn despawn_merge_preserves_flag() {
    let mut a = Access::empty();
    a.set_despawns();
    let b = Access::empty();
    let merged = a.merge(&b);
    assert!(merged.despawns());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- access::tests 2>&1 | head -30`
Expected: compilation errors — `despawns()`, `set_despawns()`, `has_any_access()` don't exist yet.

**Step 3: Implement**

In `crates/minkowski/src/access.rs`, add the `despawns` field and methods:

```rust
#[derive(Clone)]
pub struct Access {
    reads: FixedBitSet,
    writes: FixedBitSet,
    despawns: bool,
}
```

Update `Access::of`:
```rust
pub fn of<Q: WorldQuery + 'static>(world: &mut World) -> Self {
    // ... existing code ...
    Self { reads, writes, despawns: false }
}
```

Update `Access::empty`:
```rust
pub fn empty() -> Self {
    Self {
        reads: FixedBitSet::new(),
        writes: FixedBitSet::new(),
        despawns: false,
    }
}
```

Add methods:
```rust
/// True if this access set includes the despawn capability.
pub fn despawns(&self) -> bool {
    self.despawns
}

/// Mark this access set as including despawn capability.
pub fn set_despawns(&mut self) {
    self.despawns = true;
}

/// True if this access set touches any component (reads or writes).
pub fn has_any_access(&self) -> bool {
    self.reads.ones().next().is_some() || self.writes.ones().next().is_some()
}
```

Update `merge`:
```rust
pub fn merge(&self, other: &Access) -> Access {
    let mut reads = self.reads.clone();
    let mut writes = self.writes.clone();
    reads.union_with(&other.reads);
    writes.union_with(&other.writes);
    Access {
        reads,
        writes,
        despawns: self.despawns || other.despawns,
    }
}
```

Update `conflicts_with`:
```rust
pub fn conflicts_with(&self, other: &Access) -> bool {
    // Column-level read-write / write-write conflicts
    if self.writes.intersection(&other.reads).next().is_some() {
        return true;
    }
    if self.writes.intersection(&other.writes).next().is_some() {
        return true;
    }
    if other.writes.intersection(&self.reads).next().is_some() {
        return true;
    }
    // Despawn conflicts with any component access on the other side
    if self.despawns && other.has_any_access() {
        return true;
    }
    if other.despawns && self.has_any_access() {
        return true;
    }
    false
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- access::tests`
Expected: all access tests pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/access.rs
git commit -m "feat: Access despawn flag with blanket conflict rule"
```

---

### Task 2: DynamicReducerBuilder can_remove + can_despawn

Add `can_remove::<T>()` and `can_despawn()` to `DynamicReducerBuilder`. `can_remove` marks the component as written (removal is a write). `can_despawn` sets the despawn flag on Access. Track remove declarations in `DynamicResolved` so DynamicCtx can validate at runtime.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write the failing tests**

Add to the existing `mod tests` in `reducer.rs`:

```rust
#[test]
fn can_remove_marks_write_access() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("remover", &mut world)
        .can_read::<Health>()
        .can_remove::<Vel>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});
    let access = registry.dynamic_access(id);
    // can_remove marks Vel as written
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
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});
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
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});
    let entity_id = registry.register_entity::<(Vel,), (), _>(
        &mut world, "set_vel", |_e, ()| {},
    );
    let dyn_access = registry.dynamic_access(dyn_id);
    let entity_access = registry.reducer_access(entity_id);
    // Despawn conflicts with any non-empty access
    assert!(dyn_access.conflicts_with(entity_access));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- reducer::tests::can_remove 2>&1 | head -10`
Expected: compilation error — `can_remove` and `can_despawn` don't exist.

**Step 3: Implement**

In `DynamicResolved`, add a `remove_set` field to track which components were declared removable:

```rust
pub(crate) struct DynamicResolved {
    entries: Vec<(TypeId, ComponentId)>,
    access: Access,
    spawn_bundles: HashSet<TypeId>,
    remove_ids: HashSet<TypeId>,   // NEW: components declared via can_remove
}
```

Update `DynamicResolved::new` to accept `remove_ids`:
```rust
pub(crate) fn new(
    mut entries: Vec<(TypeId, ComponentId)>,
    access: Access,
    spawn_bundles: HashSet<TypeId>,
    remove_ids: HashSet<TypeId>,
) -> Self {
    // ... existing sort/dedup ...
    Self { entries, access, spawn_bundles, remove_ids }
}
```

Add a method to check remove permission:
```rust
pub(crate) fn has_remove<T: 'static>(&self) -> bool {
    self.remove_ids.contains(&TypeId::of::<T>())
}
```

In `DynamicReducerBuilder`, add a `remove_ids` field:
```rust
pub struct DynamicReducerBuilder<'a> {
    registry: &'a mut ReducerRegistry,
    world: &'a mut World,
    name: &'static str,
    access: Access,
    entries: Vec<(TypeId, ComponentId)>,
    spawn_bundles: HashSet<TypeId>,
    remove_ids: HashSet<TypeId>,   // NEW
}
```

Add the two builder methods:
```rust
/// Declare that the closure may remove component `T` from entities.
/// Marks T as written (removal is a structural write) and adds
/// a TypeId entry for runtime validation.
pub fn can_remove<T: crate::component::Component>(mut self) -> Self {
    let comp_id = self.world.register_component::<T>();
    self.access.add_write(comp_id);
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
```

Update the `dynamic()` method on `ReducerRegistry` to initialize `remove_ids`:
```rust
// Find where DynamicReducerBuilder is constructed in ReducerRegistry::dynamic()
// and add: remove_ids: HashSet::new(),
```

Update `build()` to pass `remove_ids` to `DynamicResolved::new`:
```rust
let resolved = DynamicResolved::new(
    self.entries,
    self.access.clone(),
    self.spawn_bundles,
    self.remove_ids,
);
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- reducer::tests`
Expected: all pass (existing + new).

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: DynamicReducerBuilder can_remove + can_despawn"
```

---

### Task 3: DynamicCtx remove + despawn methods

Add `remove`, `try_remove`, and `despawn` to `DynamicCtx`. These buffer mutations into the changeset, validated against the builder-declared bounds.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write the failing tests**

```rust
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
        });
    registry
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();
    assert!(world.get::<Vel>(e).is_none());
    assert!(world.get::<Pos>(e).is_some());
}

#[test]
fn dynamic_ctx_try_remove_returns_false_when_missing() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),)); // no Vel
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
        });
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
        // deliberately NOT declaring can_remove::<Vel>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.remove::<Vel>(*entity);
        });
    let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
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
        });
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
        // deliberately NOT declaring can_despawn()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.despawn(*entity);
        });
    let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- dynamic_ctx_remove 2>&1 | head -10`
Expected: compilation error — `remove`, `try_remove`, `despawn` don't exist on DynamicCtx.

**Step 3: Implement**

Add these methods to `impl DynamicCtx`:

```rust
/// Buffer a component removal. The removal is applied on commit
/// (archetype migration). Panics if T was not declared via `can_remove`.
pub fn remove<T: crate::component::Component>(&mut self, entity: Entity) {
    let comp_id = self.resolved.lookup::<T>().unwrap_or_else(|| {
        panic!(
            "component {} not declared in dynamic reducer \
             (use can_remove)",
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
            "component {} not declared in dynamic reducer \
             (use can_remove)",
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
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- reducer::tests`
Expected: all pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: DynamicCtx remove, try_remove, despawn methods"
```

---

### Task 4: EntityMut remove + despawn

Add `remove()` and `despawn()` to `EntityMut`. `remove` is bounded by the component set C (same as `get`/`set`). `despawn` needs the despawn flag — pass it through from registration.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn entity_mut_remove_buffers_mutation() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry.register_entity::<(Pos, Vel), (), _>(
        &mut world,
        "strip_vel",
        |mut entity: EntityMut<'_, (Pos, Vel)>, ()| {
            entity.remove::<Vel, 1>();
        },
    );
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
    let id = registry.register_entity_despawn::<(Pos,), (), _>(
        &mut world,
        "killer",
        |mut entity: EntityMut<'_, (Pos,)>, ()| {
            entity.despawn();
        },
    );
    registry.call(&strategy, &mut world, id, (e, ())).unwrap();
    assert!(!world.is_alive(e));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- entity_mut_remove 2>&1 | head -10`
Expected: compilation error — `remove`, `despawn` don't exist on EntityMut, `register_entity_despawn` doesn't exist.

**Step 3: Implement**

Add `remove` to `EntityMut` — it's already bounded by `Contains<T, IDX>`:

```rust
impl<'a, C: ComponentSet> EntityMut<'a, C> {
    // ... existing get, set, entity methods ...

    /// Buffer a component removal. The component is removed on commit
    /// (archetype migration). Bounded by the declared component set C.
    pub fn remove<T: Component, const IDX: usize>(&mut self)
    where
        C: Contains<T, IDX>,
    {
        let comp_id = self.resolved.0[IDX];
        self.changeset.record_remove(self.entity, comp_id);
    }
}
```

For `despawn`, EntityMut needs to know whether despawn was declared. Add a `can_despawn: bool` field:

```rust
pub struct EntityMut<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    changeset: &'a mut EnumChangeSet,
    world: &'a World,
    can_despawn: bool,   // NEW
    _marker: PhantomData<C>,
}
```

Update `EntityMut::new` to accept `can_despawn`:
```rust
pub(crate) fn new(
    entity: Entity,
    resolved: &'a ResolvedComponents,
    changeset: &'a mut EnumChangeSet,
    world: &'a World,
    can_despawn: bool,
) -> Self {
    Self {
        entity, resolved, changeset, world, can_despawn,
        _marker: PhantomData,
    }
}
```

Add `despawn`:
```rust
pub fn despawn(&mut self) {
    assert!(
        self.can_despawn,
        "despawn not declared (use register_entity_despawn)"
    );
    self.changeset.record_despawn(self.entity);
}
```

Add `register_entity_despawn` to `ReducerRegistry` — identical to `register_entity` but sets despawn flag on Access and passes `can_despawn: true` to EntityMut:

```rust
pub fn register_entity_despawn<C, Args, F>(
    &mut self,
    world: &mut World,
    name: &'static str,
    f: F,
) -> ReducerId
where
    C: ComponentSet + 'static,
    Args: Clone + 'static,
    F: Fn(EntityMut<'_, C>, Args) + Send + Sync + 'static,
{
    let resolved = C::resolve(&mut world.components);
    let mut access = Access::empty();
    for &comp_id in &resolved.0 {
        access.add_read(comp_id);
        access.add_write(comp_id);
    }
    access.set_despawns();  // NEW: set despawn flag

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

    self.push_entry(name, access, ResolvedComponents(resolved.0.clone()), ReducerKind::Transactional(adapter), None)
}
```

Update existing `register_entity` to pass `can_despawn: false` to EntityMut:
```rust
// In the existing register_entity adapter closure, change:
let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref());
// to:
let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref(), false);
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- reducer::tests`
Expected: all pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: EntityMut remove + despawn, register_entity_despawn"
```

---

### Task 5: DynamicCtx::for_each

Add typed read-only iteration to `DynamicCtx`. The query is fully typed (`ReadOnlyWorldQuery`); the dynamic part is runtime validation that `Q::accessed_ids` is a subset of the builder-declared components.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write the failing tests**

```rust
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
            ctx.for_each::<(&Pos,)>(|(pos,)| {
                let _ = pos;
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
        });
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
        // deliberately NOT declaring can_read::<Vel>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            ctx.for_each::<(&Pos, &Vel)>(|(_p, _v)| {});
        });
    let _ = registry.dynamic_call(&strategy, &mut world, id, &());
}

#[test]
fn dynamic_ctx_for_each_with_write_after_read() {
    // Read via typed query, then write via ctx.write()
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
            // Collect entities + values first (can't borrow ctx during for_each)
            let mut updates: Vec<(Entity, f32)> = Vec::new();
            ctx.for_each::<(Entity, &Vel)>(|(entity, vel)| {
                updates.push((entity, vel.0 * 2.0));
            });
            for (entity, new_vel) in updates {
                ctx.write(entity, Vel(new_vel));
            }
        });
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(world.get::<Vel>(e1).unwrap().0, 20.0);
    assert_eq!(world.get::<Vel>(e2).unwrap().0, 40.0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- dynamic_ctx_for_each 2>&1 | head -10`
Expected: compilation error — `for_each` doesn't exist on DynamicCtx.

**Step 3: Implement**

Add `for_each` to `impl DynamicCtx`:

```rust
/// Iterate entities matching query `Q` using the typed query codepath.
/// `Q` must be a `ReadOnlyWorldQuery` — writes go through `ctx.write()`.
///
/// # Panics
/// Panics if `Q` accesses any component not declared via `can_read`
/// or `can_write` on the builder.
pub fn for_each<Q: ReadOnlyWorldQuery + 'static>(
    &self,
    mut f: impl FnMut(Q::Item<'_>),
) {
    // Runtime validation: Q's accessed components must be a subset
    // of the builder-declared components.
    Q::register(&self.world.components);
    let accessed = Q::accessed_ids(&self.world.components);
    for comp_id in accessed.ones() {
        assert!(
            self.resolved.lookup_by_comp_id(comp_id).is_some(),
            "query accesses component ID {} which was not declared \
             in dynamic reducer (use can_read/can_write)",
            comp_id,
        );
    }

    let required = Q::required_ids(&self.world.components);

    for arch in &self.world.archetypes.archetypes {
        if arch.is_empty() || !required.is_subset(&arch.component_ids) {
            continue;
        }
        let fetch = unsafe { Q::init_fetch(arch, &self.world.components) };
        for row in 0..arch.len() {
            let entity = arch.entities[row];
            let item = unsafe { Q::fetch(&fetch, row, entity) };
            f(item);
        }
    }
}
```

Note: `for_each` takes `&self` (not `&mut self`) because it only reads from the world. The user calls `ctx.write()` separately after collecting results. This means `for_each` can't be called while `ctx` is mutably borrowed — the user must collect into a `Vec` first, which is the correct pattern.

**Important**: We need `DynamicResolved::lookup_by_comp_id` — a reverse lookup from ComponentId to check membership. Add it:

```rust
impl DynamicResolved {
    pub(crate) fn lookup_by_comp_id(&self, comp_id: ComponentId) -> Option<TypeId> {
        self.entries.iter().find(|(_, cid)| *cid == comp_id).map(|(tid, _)| *tid)
    }
}
```

**Note on ComponentRegistry::register**: `Q::register` takes `&mut ComponentRegistry` but DynamicCtx only has `&World`. The components in the query should already be registered (they were registered by the builder's `can_read`/`can_write` calls). The accessed_ids check will catch any unregistered component. Skip the `Q::register` call — use `&self.world.components` directly for `accessed_ids` and `required_ids`. If a component isn't registered, its ID won't be in the resolved set, so the assert fires.

Remove the `Q::register` line from the implementation above.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- reducer::tests`
Expected: all pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: DynamicCtx::for_each — typed read-only iteration with runtime access validation"
```

---

### Task 6: DynamicCtx::for_each Changed<T> support

Add per-reducer tick state to dynamic reducers so `Changed<T>` filters work inside `DynamicCtx::for_each`. Same pattern as QueryWriter: `Arc<AtomicU64>` stored on the entry, updated by `dynamic_call` after commit.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write the failing test**

```rust
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
            ctx.for_each::<(Entity, Changed<Pos>, &Pos)>(|(entity, (), pos)| {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                updates.push((entity, Pos(pos.0 + 1.0)));
            });
            for (entity, val) in updates {
                ctx.write(entity, val);
            }
        });

    // First call: column was never read by this reducer, Changed matches
    registry.dynamic_call(&strategy, &mut world, id, &()).unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 2.0);

    // Second call: no external mutation, Changed should skip
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    registry.dynamic_call(&strategy, &mut world, id, &()).unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 0);

    // External mutation, then call again
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    for (pos,) in world.query::<(&mut Pos,)>() {
        pos.0 = 99.0;
    }
    registry.dynamic_call(&strategy, &mut world, id, &()).unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 100.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- dynamic_ctx_for_each_changed_filter`
Expected: FAIL — either compilation error (if `for_each` doesn't accept `Changed<T>` yet because it doesn't call `matches_filters`) or assertion failure on the second call (if filters aren't applied).

**Step 3: Implement**

Add `last_read_tick` to `DynamicCtx`:

```rust
pub struct DynamicCtx<'a> {
    world: &'a World,
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    resolved: &'a DynamicResolved,
    last_read_tick: &'a Arc<AtomicU64>,  // NEW
}
```

Update `DynamicCtx::new` to accept and store `last_read_tick`.

Update `for_each` to use `matches_filters` with `last_read_tick`:

```rust
pub fn for_each<Q: ReadOnlyWorldQuery + 'static>(
    &self,
    mut f: impl FnMut(Q::Item<'_>),
) {
    // ... existing validation ...

    let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));
    let required = Q::required_ids(&self.world.components);

    for arch in &self.world.archetypes.archetypes {
        if arch.is_empty() || !required.is_subset(&arch.component_ids) {
            continue;
        }
        if !Q::matches_filters(arch, &self.world.components, last_tick) {
            continue;
        }
        let fetch = unsafe { Q::init_fetch(arch, &self.world.components) };
        for row in 0..arch.len() {
            let entity = arch.entities[row];
            let item = unsafe { Q::fetch(&fetch, row, entity) };
            f(item);
        }
    }
    // last_read_tick is updated by dynamic_call() AFTER commit
}
```

Add `last_read_tick: Option<Arc<AtomicU64>>` to `DynamicReducerEntry`:

```rust
struct DynamicReducerEntry {
    #[allow(dead_code)]
    name: &'static str,
    resolved: DynamicResolved,
    closure: DynamicAdapter,
    last_read_tick: Option<Arc<AtomicU64>>,  // NEW
}
```

In `DynamicReducerBuilder::build()`, create and store the tick:

```rust
let last_read_tick = Arc::new(AtomicU64::new(0));
// ... pass to DynamicReducerEntry ...
```

Update `DynamicCtx::new` call in `dynamic_call` to pass the tick reference.

Update `dynamic_call` to store tick post-commit (same pattern as `call`):

```rust
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
    let access = resolved.access();
    let tick_arc = entry.last_read_tick.clone();

    let result = strategy.transact(world, access, |tx, world| {
        let (changeset, allocated) = tx.reducer_parts();
        let world_ref: &World = world;
        let mut ctx = DynamicCtx::new(world_ref, changeset, allocated, resolved, &tick_arc.as_ref().unwrap());
        closure(&mut ctx, args);
    });
    if result.is_ok() {
        if let Some(arc) = &tick_arc {
            let new_tick = world.next_tick();
            arc.store(new_tick.raw(), Ordering::Relaxed);
        }
    }
    result
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- reducer::tests`
Expected: all pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: DynamicCtx::for_each Changed<T> support via per-reducer tick"
```

---

### Task 7: Example + CLAUDE.md updates

Update the reducer example to demonstrate structural mutations and dynamic iteration. Update CLAUDE.md with the new API surface.

**Files:**
- Modify: `examples/examples/reducer.rs`
- Modify: `CLAUDE.md`

**Step 1: Add to the example**

Add a section to `examples/examples/reducer.rs` demonstrating:

1. **Dynamic remove**: strip Shield from entities with low energy
2. **Dynamic despawn**: destroy entities with 0 HP
3. **Dynamic for_each**: iterate + write pattern

```rust
// ── 10. Structural mutations — remove + despawn ───────────────
println!("\n--- Structural mutations ---");

// Give enemy 0 HP for the despawn demo
registry
    .call(&strategy, &mut world, damage_id, (enemy, 20u32))
    .unwrap();

let reaper_id = registry
    .dynamic("reaper", &mut world)
    .can_read::<Health>()
    .can_despawn()
    .build(|ctx: &mut DynamicCtx, _args: &()| {
        let mut to_despawn = Vec::new();
        ctx.for_each::<(Entity, &Health)>(|(entity, health)| {
            if health.0 == 0 {
                to_despawn.push(entity);
            }
        });
        for entity in to_despawn {
            ctx.despawn(entity);
            println!("  [reaper] despawned {:?}", entity);
        }
    });

registry
    .dynamic_call(&strategy, &mut world, reaper_id, &())
    .unwrap();

println!(
    "After reaper: enemy alive={}, hero alive={}",
    world.is_alive(enemy),
    world.is_alive(hero),
);
```

**Step 2: Run the example**

Run: `cargo run -p minkowski-examples --example reducer --release`
Expected: runs to completion, shows structural mutation output.

**Step 3: Update CLAUDE.md**

Update the following sections:

1. **Execution models table**: add `DynamicCtx` note about `for_each`, `despawn`, `remove`.
2. **Dynamic reducers paragraph**: mention `can_remove`, `can_despawn`, `for_each`.
3. **Access struct description**: mention `despawns` flag and `has_any_access()`.
4. **Pub list**: add nothing new (existing types, new methods).
5. **EntityMut description**: mention `remove`, `despawn`, `register_entity_despawn`.

**Step 4: Run full test suite + clippy**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass, clean.

**Step 5: Commit**

```bash
git add examples/examples/reducer.rs CLAUDE.md
git commit -m "feat: extended reducers example + CLAUDE.md updates"
```

---

## Semantic Review Checklist

Before each task, verify:

1. **Can this be called with the wrong World?** — DynamicCtx holds `&World` from the transact closure; same-World guarantee comes from strategy's WorldId check.
2. **Can Drop observe inconsistent state?** — No new Drop impls. Changeset owns buffered mutations; existing drop safety applies.
3. **Can two threads reach this through `&self`?** — DynamicCtx is `&mut` — single owner. Access is Clone, used read-only after construction.
4. **Does the API surface permit operations outside the Access bitset?** — `remove` checks `has_remove`, `despawn` checks `access().despawns()`, `for_each` checks `accessed_ids` subset. All `assert!` (not `debug_assert!`).
5. **What happens if this is abandoned halfway through?** — Changeset not applied. Existing Tx Drop handles entity ID cleanup via OrphanQueue.
6. **Can `for_each` be called while `ctx` is mutably borrowed?** — No. `for_each` takes `&self`, `write`/`remove`/`despawn` take `&mut self`. Borrow checker enforces: collect into Vec from `for_each`, then call mutation methods.
