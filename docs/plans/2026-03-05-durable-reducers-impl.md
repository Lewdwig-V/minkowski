# Durable Reducers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `QueryWriter` — a reducer handle that iterates like a query but buffers writes through `ChangeSet` for WAL compatibility — plus unify `call_entity` into a single `call` method.

**Architecture:** `WriterQuery` trait (separate from `WorldQuery`) maps `&mut T` → `WritableRef<T>`. `QueryWriter` does manual archetype scanning (not `world.query()`) to avoid self-conflict with optimistic validation. Unified `call` folds entity into args.

**Tech Stack:** Rust, minkowski ECS core crate

---

### Task 1: Tick visibility

Make `Tick::new` available outside tests and add a `raw()` accessor so reducer code can store/load tick values in `AtomicU64`.

**Files:**
- Modify: `crates/minkowski/src/tick.rs:13-16`

**Step 1: Remove `#[cfg(test)]` from `Tick::new` and add `raw()` accessor**

In `crates/minkowski/src/tick.rs`, change:

```rust
impl Tick {
    #[cfg(test)]
    pub fn new(value: u64) -> Self {
        Self(value)
    }
```

to:

```rust
impl Tick {
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
```

**Step 2: Run tests to verify nothing breaks**

Run: `cargo test -p minkowski --lib -- tick`
Expected: All tick tests pass. Existing tests already use `Tick::new` (in `#[cfg(test)]` contexts), now it's always available.

**Step 3: Commit**

```bash
git add crates/minkowski/src/tick.rs
git commit -m "refactor: make Tick::new and raw() always available (pub(crate))"
```

---

### Task 2: WritableRef struct and tests

The per-component buffered write handle. Uses a raw pointer to `EnumChangeSet` internally because multiple `WritableRef`s in a tuple query (e.g. `(&mut Pos, &mut Vel)`) need shared write access to the same changeset — Rust's borrow rules prevent multiple `&mut` references.

**Safety invariant:** The raw pointer is valid for the duration of `QueryWriter::for_each`. All access is via `insert_raw` (append-only — no aliasing reads). `WritableRef` can't outlive the callback. Constructor is `pub(crate)`.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (add after the `Spawner` impl, before `QueryRef`)

**Step 1: Write the failing tests**

Add at the bottom of `#[cfg(test)] mod tests` in `reducer.rs`:

```rust
// ── WritableRef tests ──────────────────────────────────────────

#[test]
fn writable_ref_get_returns_current_value() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let comp_id = world.register_component::<Pos>();
    let mut cs = EnumChangeSet::new();

    let current = world.get::<Pos>(e).unwrap();
    let cs_ptr: *mut EnumChangeSet = &mut cs;
    let wref = WritableRef::<Pos>::new(e, current, comp_id, cs_ptr);
    assert_eq!(wref.get().0, 1.0);
}

#[test]
fn writable_ref_set_buffers_into_changeset() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let comp_id = world.register_component::<Pos>();
    let mut cs = EnumChangeSet::new();

    let current = world.get::<Pos>(e).unwrap();
    let cs_ptr: *mut EnumChangeSet = &mut cs;
    let mut wref = WritableRef::<Pos>::new(e, current, comp_id, cs_ptr);
    wref.set(Pos(42.0));

    // Value not applied to world yet
    assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
    // Changeset has the buffered write
    assert_eq!(cs.len(), 1);
    let _reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 42.0);
}

#[test]
fn writable_ref_modify_clones_and_sets() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let comp_id = world.register_component::<Pos>();
    let mut cs = EnumChangeSet::new();

    let current = world.get::<Pos>(e).unwrap();
    let cs_ptr: *mut EnumChangeSet = &mut cs;
    let mut wref = WritableRef::<Pos>::new(e, current, comp_id, cs_ptr);
    wref.modify(|p| p.0 += 10.0);

    let _reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 11.0);
}
```

Note: `Pos` is already defined in the test module as `#[derive(Clone, Copy, ...)] struct Pos(f32)`. Verify it derives `Clone` — needed for `modify`. If not, add `Clone` to `Pos` in the test module.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- writable_ref`
Expected: FAIL — `WritableRef` not found.

**Step 3: Implement WritableRef**

Add in `reducer.rs`, after the `Spawner` impl block (around line 403), before the `QueryRef` section:

```rust
// ── WritableRef (buffered per-component write handle) ────────────────

/// Per-component buffered write handle. Reads the current value from the
/// archetype column; writes buffer into an `EnumChangeSet` applied on commit.
///
/// Uses a raw pointer to `EnumChangeSet` internally so that multiple
/// `WritableRef`s in a tuple query can share write access to the same changeset.
///
/// # Safety invariant
/// The raw pointer is valid for the lifetime `'a`. All access is append-only
/// via `insert_raw`. Constructor is `pub(crate)` — users cannot create these.
pub struct WritableRef<'a, T: Component> {
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: *mut EnumChangeSet,
    _marker: std::marker::PhantomData<&'a mut EnumChangeSet>,
}

impl<'a, T: Component> WritableRef<'a, T> {
    pub(crate) fn new(
        entity: Entity,
        current: &'a T,
        comp_id: ComponentId,
        changeset: *mut EnumChangeSet,
    ) -> Self {
        Self {
            entity,
            current,
            comp_id,
            changeset,
            _marker: std::marker::PhantomData,
        }
    }

    /// Read the current value (zero-cost — pointer into archetype column).
    pub fn get(&self) -> &T {
        self.current
    }

    /// Buffer a replacement value into the ChangeSet.
    pub fn set(&mut self, value: T) {
        // Safety: pointer valid for 'a, insert_raw is append-only.
        unsafe { &mut *self.changeset }.insert_raw(self.entity, self.comp_id, value);
    }

    /// Clone-mutate-set in one call.
    pub fn modify(&mut self, f: impl FnOnce(&mut T))
    where
        T: Clone,
    {
        let mut value = self.current.clone();
        f(&mut value);
        self.set(value);
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- writable_ref`
Expected: PASS

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

**Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: WritableRef — per-component buffered write handle"
```

---

### Task 3: WriterQuery trait and primitive impls

Separate trait from `WorldQuery`. Maps `&mut T` → `WritableRef<T>`, passes `&T` through unchanged.

**Key difference from `WorldQuery::Fetch`:** `WriterFetch` for `&mut T` includes a `ComponentId` alongside the column pointer, so `WritableRef` doesn't need a per-row registry lookup.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (add after `WritableRef`, before `QueryRef`)

**Step 1: Write the failing tests**

Add to `#[cfg(test)] mod tests`:

```rust
// ── WriterQuery tests ──────────────────────────────────────────

#[test]
fn writer_query_ref_t_passthrough() {
    // &T WriterItem is just &T
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    let arch = &world.archetypes.archetypes[0];
    let fetch = <&Pos as WriterQuery>::init_writer_fetch(arch, &world.components);
    let mut cs = EnumChangeSet::new();
    let cs_ptr: *mut EnumChangeSet = &mut cs;
    let item: &Pos = unsafe { <&Pos as WriterQuery>::fetch_writer(&fetch, 0, Entity::DANGLING, cs_ptr) };
    assert_eq!(item.0, 1.0);
}

#[test]
fn writer_query_mut_t_becomes_writable_ref() {
    let mut world = World::new();
    let e = world.spawn((Vel(2.0),));
    let arch = &world.archetypes.archetypes[0];
    let entity = arch.entities[0];
    let fetch = <&mut Vel as WriterQuery>::init_writer_fetch(arch, &world.components);
    let mut cs = EnumChangeSet::new();
    let cs_ptr: *mut EnumChangeSet = &mut cs;
    let mut item = unsafe { <&mut Vel as WriterQuery>::fetch_writer(&fetch, 0, entity, cs_ptr) };

    assert_eq!(item.get().0, 2.0);
    item.set(Vel(99.0));
    let _reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Vel>(e).unwrap().0, 99.0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- writer_query`
Expected: FAIL — `WriterQuery` not found.

**Step 3: Implement WriterQuery trait and primitive impls**

Add in `reducer.rs`, after `WritableRef`:

```rust
// ── WriterQuery trait ────────────────────────────────────────────────

/// Maps `WorldQuery` item types to buffered writer equivalents.
/// `&T` passes through unchanged. `&mut T` becomes `WritableRef<T>`.
///
/// Separate from `WorldQuery` — persistence concerns stay out of the
/// storage engine.
///
/// # Safety
/// Same invariants as `WorldQuery`: `init_writer_fetch` must produce valid
/// state, `fetch_writer` must be safe to call for any row < archetype.len().
pub unsafe trait WriterQuery: WorldQuery {
    type WriterItem<'a>;
    type WriterFetch<'a>: Send + Sync;

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w>;

    /// # Safety
    /// `row` must be less than archetype len. `changeset` must be valid.
    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        entity: Entity,
        changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w>;
}

// --- &T: passthrough ---
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
        <&T as WorldQuery>::fetch(fetch, row)
    }
}

// --- &mut T: WritableRef ---
unsafe impl<T: Component> WriterQuery for &mut T {
    type WriterItem<'a> = WritableRef<'a, T>;
    type WriterFetch<'a> = (ThinSlicePtr<T>, ComponentId);

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
        let id = registry.id::<T>().expect("component not registered");
        let col_idx = archetype.component_index[&id];
        let ptr = archetype.columns[col_idx].as_typed_ptr::<T>();
        (ptr, id)
    }

    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        entity: Entity,
        changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        let (ptr, comp_id) = fetch;
        let current: &T = &*ptr.ptr.add(row);
        WritableRef::new(entity, current, *comp_id, changeset)
    }
}

// --- Entity: passthrough ---
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
        <Option<&T> as WorldQuery>::fetch(fetch, row)
    }
}

// --- Option<&mut T>: Option<WritableRef<T>> ---
unsafe impl<T: Component> WriterQuery for Option<&mut T> {
    type WriterItem<'a> = Option<WritableRef<'a, T>>;
    type WriterFetch<'a> = (Option<ThinSlicePtr<T>>, ComponentId);

    fn init_writer_fetch<'w>(
        archetype: &'w Archetype,
        registry: &ComponentRegistry,
    ) -> Self::WriterFetch<'w> {
        let id = registry.id::<T>().expect("component not registered");
        let opt_ptr = <Option<&mut T> as WorldQuery>::init_fetch(archetype, registry);
        (opt_ptr, id)
    }

    unsafe fn fetch_writer<'w>(
        fetch: &Self::WriterFetch<'w>,
        row: usize,
        entity: Entity,
        changeset: *mut EnumChangeSet,
    ) -> Self::WriterItem<'w> {
        let (opt_ptr, comp_id) = fetch;
        opt_ptr.as_ref().map(|ptr| {
            let current: &T = &*ptr.ptr.add(row);
            WritableRef::new(entity, current, *comp_id, changeset)
        })
    }
}

// --- Changed<T>: filter only, no item ---
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
```

Note: This requires adding `use crate::query::fetch::ThinSlicePtr;` and `use crate::storage::archetype::Archetype;` to the imports at the top of `reducer.rs` if not already present. Also add `use crate::component::ComponentRegistry;`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- writer_query`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: WriterQuery trait with impls for &T, &mut T, Entity, Option, Changed"
```

---

### Task 4: WriterQuery tuple macro

Generate `WriterQuery` impls for tuples 1–12, matching the existing `impl_world_query_tuple!` pattern.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write a failing test for 2-tuple**

Add to tests:

```rust
#[test]
fn writer_query_tuple_fetch() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let arch = &world.archetypes.archetypes[0];
    let entity = arch.entities[0];

    let fetch = <(&Pos, &mut Vel) as WriterQuery>::init_writer_fetch(arch, &world.components);
    let mut cs = EnumChangeSet::new();
    let cs_ptr: *mut EnumChangeSet = &mut cs;

    let (pos, mut vel) = unsafe {
        <(&Pos, &mut Vel) as WriterQuery>::fetch_writer(&fetch, 0, entity, cs_ptr)
    };

    assert_eq!(pos.0, 1.0);
    assert_eq!(vel.get().0, 2.0);
    vel.set(Vel(99.0));

    let _reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Vel>(e).unwrap().0, 99.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- writer_query_tuple`
Expected: FAIL — tuple impl not found.

**Step 3: Implement the tuple macro**

Add in `reducer.rs`, after the primitive `WriterQuery` impls:

```rust
macro_rules! impl_writer_query_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: WriterQuery),*> WriterQuery for ($($name,)*) {
            type WriterItem<'a> = ($($name::WriterItem<'a>,)*);
            type WriterFetch<'a> = ($($name::WriterFetch<'a>,)*);

            fn init_writer_fetch<'w>(
                archetype: &'w Archetype,
                registry: &ComponentRegistry,
            ) -> Self::WriterFetch<'w> {
                ($($name::init_writer_fetch(archetype, registry),)*)
            }

            unsafe fn fetch_writer<'w>(
                fetch: &Self::WriterFetch<'w>,
                row: usize,
                entity: Entity,
                changeset: *mut EnumChangeSet,
            ) -> Self::WriterItem<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WriterQuery>::fetch_writer($name, row, entity, changeset),)*)
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
```

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- writer_query`
Expected: All writer_query tests PASS.

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: WriterQuery tuple impls 1-12"
```

---

### Task 5: Unified adapter and `call` method

Refactor: remove `Entity` from the adapter signature, rename `EntityTransactional` → `Transactional`, change `&World` → `&mut World`, replace `call_entity` with `call`.

**This is a breaking change.** Entity reducers now receive entity as part of args. Spawner and query writer reducers don't pass an entity.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (types, registration, dispatch, tests)
- Modify: `examples/examples/reducer.rs` (call sites)

**Step 1: Rename adapter type and ReducerKind variant**

Change the `EntityAdapter` type alias (around line 498):

```rust
// Before:
type EntityAdapter = Box<
    dyn Fn(&mut EnumChangeSet, &mut Vec<Entity>, &World, &ResolvedComponents, Entity, &dyn Any)
        + Send + Sync,
>;

// After:
type TransactionalAdapter = Box<
    dyn Fn(&mut EnumChangeSet, &mut Vec<Entity>, &mut World, &ResolvedComponents, &dyn Any)
        + Send + Sync,
>;
```

Change `ReducerKind`:

```rust
// Before:
enum ReducerKind {
    EntityTransactional(EntityAdapter),
    Scheduled(ScheduledAdapter),
}

// After:
enum ReducerKind {
    Transactional(TransactionalAdapter),
    Scheduled(ScheduledAdapter),
}
```

**Step 2: Update `register_entity` — entity comes from args**

The user's closure signature stays `Fn(EntityMut<C>, Args)`. But at the call site, the user passes `(entity, args)`. The adapter unpacks entity from the args tuple.

Change `register_entity` (around line 564):

```rust
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
    let reads = C::access(&mut world.components, true);
    let writes = C::access(&mut world.components, false);
    let access = reads.merge(&writes);

    let adapter: TransactionalAdapter = Box::new(
        move |changeset, _allocated, world, resolved, args_any| {
            let (entity, args) = args_any
                .downcast_ref::<(Entity, Args)>()
                .unwrap_or_else(|| {
                    panic!(
                        "reducer args type mismatch: expected (Entity, {})",
                        std::any::type_name::<Args>()
                    )
                })
                .clone();
            let world_ref: &World = world;
            let handle = EntityMut::<C>::new(entity, resolved, changeset, world_ref);
            f(handle, args);
        },
    );

    self.push_entry(name, access, resolved, ReducerKind::Transactional(adapter))
}
```

**Step 3: Update `register_spawner` — no entity parameter**

Change `register_spawner` (around line 606):

```rust
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
    let access = Access::empty();

    let adapter: TransactionalAdapter = Box::new(
        move |changeset, allocated, world, _resolved, args_any| {
            let args = args_any
                .downcast_ref::<Args>()
                .unwrap_or_else(|| {
                    panic!(
                        "reducer args type mismatch: expected {}",
                        std::any::type_name::<Args>()
                    )
                })
                .clone();
            let world_ref: &World = world;
            let handle = Spawner::<B>::new(changeset, allocated, world_ref);
            f(handle, args);
        },
    );

    self.push_entry(name, access, resolved, ReducerKind::Transactional(adapter))
}
```

**Step 4: Replace `call_entity` with `call`**

```rust
/// Call a transactional reducer (entity, spawner, or query writer).
pub fn call<S: Transact, Args: Clone + 'static>(
    &self,
    strategy: &S,
    world: &mut World,
    id: ReducerId,
    args: Args,
) -> Result<(), Conflict> {
    let entry = &self.reducers[id.0];
    let adapter = match &entry.kind {
        ReducerKind::Transactional(f) => f,
        ReducerKind::Scheduled(_) => {
            panic!("call() on scheduled reducer — use run() instead")
        }
    };
    let access = &entry.access;
    let resolved = &entry.resolved;

    strategy.transact(world, access, |tx, world| {
        let (changeset, allocated) = tx.reducer_parts();
        adapter(changeset, allocated, world, resolved, &args);
    })
}
```

**Step 5: Update `run()` panic message and `reducer_id_by_name` / `query_reducer_id_by_name`**

In `run()`:
```rust
ReducerKind::Transactional(_) => {
    panic!("run() called on transactional reducer — use call() instead")
}
```

In `reducer_id_by_name`:
```rust
ReducerKind::Transactional(_) => Some(ReducerId(idx)),
ReducerKind::Scheduled(_) => None,
```

In `query_reducer_id_by_name`:
```rust
ReducerKind::Scheduled(_) => Some(QueryReducerId(idx)),
ReducerKind::Transactional(_) => None,
```

**Step 6: Update ALL existing tests in reducer.rs**

Search for `call_entity` in tests and replace. Key patterns:

```rust
// Before:
registry.call_entity(&strategy, &mut world, heal_id, e, 25u32).unwrap();

// After:
registry.call(&strategy, &mut world, heal_id, (e, 25u32)).unwrap();
```

For spawner tests (previously used Entity::DANGLING):
```rust
// Before:
registry.call_entity(&strategy, &mut world, spawn_id, Entity::DANGLING, 50u32).unwrap();

// After:
registry.call(&strategy, &mut world, spawn_id, 50u32).unwrap();
```

**Step 7: Run all tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

**Step 8: Update `examples/examples/reducer.rs`**

Replace all `call_entity` with `call`:

```rust
// Entity reducers — entity is now part of args tuple:
registry.call(&strategy, &mut world, heal_id, (hero, 25u32)).unwrap();
registry.call(&strategy, &mut world, damage_id, (enemy, 30u32)).unwrap();

// Spawner — no entity:
registry.call(&strategy, &mut world, spawn_id, 75u32).unwrap();
```

Run: `cargo run -p minkowski-examples --example reducer --release`
Expected: Same output as before.

**Step 9: Commit**

```bash
git add crates/minkowski/src/reducer.rs examples/examples/reducer.rs
git commit -m "refactor: unified call() — entity is just args, remove call_entity"
```

---

### Task 6: QueryWriter handle, register_query_writer, and integration tests

The core feature. `QueryWriter` does manual archetype scanning (NOT `world.query()`) to avoid marking mutable columns as changed, which would cause self-conflict with optimistic validation. `Changed<T>` filters work via a per-reducer `AtomicU64` tick captured in the adapter closure.

**Why not `world.query()`:** `world.query::<(&mut Vel,)>()` marks Vel columns as changed. Optimistic validation compares column ticks against begin_tick. If we marked Vel ourselves, the commit would see the tick change and conflict with itself. Manual archetype scanning reads column data without marking anything — the changeset apply step handles column marking after commit.

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Write failing tests**

Add to `#[cfg(test)] mod tests`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ── QueryWriter tests ────────────────────────────────────────────

#[test]
fn query_writer_for_each_reads_and_buffers() {
    let mut world = World::new();
    let e1 = world.spawn((Pos(1.0), Vel(10.0)));
    let e2 = world.spawn((Pos(2.0), Vel(20.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry.register_query_writer::<(&Pos, &mut Vel), f32, _>(
        &mut world,
        "apply_drag",
        |mut query, drag: f32| {
            query.for_each(|(pos, mut vel)| {
                vel.modify(|v| v.0 *= drag);
            });
        },
    );

    registry.call(&strategy, &mut world, id, 0.5f32).unwrap();

    assert_eq!(world.get::<Vel>(e1).unwrap().0, 5.0);
    assert_eq!(world.get::<Vel>(e2).unwrap().0, 10.0);
    // Pos unchanged
    assert_eq!(world.get::<Pos>(e1).unwrap().0, 1.0);
}

#[test]
fn query_writer_count() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    world.spawn((Vel(3.0),)); // no Pos — not matched
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry.register_query_writer::<(&mut Pos,), (), _>(
        &mut world,
        "counter",
        |mut query, ()| {
            assert_eq!(query.count(), 2);
        },
    );

    registry.call(&strategy, &mut world, id, ()).unwrap();
}

#[test]
fn query_writer_access_conflict_with_entity_reducer() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    let entity_id = registry.register_entity::<(Vel,), (), _>(
        &mut world, "set_vel", |_e, ()| {},
    );
    let writer_id = registry.register_query_writer::<(&mut Vel,), (), _>(
        &mut world, "bulk_vel", |_q, ()| {},
    );

    let entity_access = registry.reducer_access(entity_id);
    let writer_access = registry.reducer_access(writer_id);
    assert!(entity_access.conflicts_with(writer_access));
}

#[test]
fn query_writer_no_conflict_disjoint_components() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    let entity_id = registry.register_entity::<(Pos,), (), _>(
        &mut world, "set_pos", |_e, ()| {},
    );
    let writer_id = registry.register_query_writer::<(&mut Vel,), (), _>(
        &mut world, "bulk_vel", |_q, ()| {},
    );

    let entity_access = registry.reducer_access(entity_id);
    let writer_access = registry.reducer_access(writer_id);
    assert!(!entity_access.conflicts_with(writer_access));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- query_writer`
Expected: FAIL — `register_query_writer` not found.

**Step 3: Implement QueryWriter handle**

Add in `reducer.rs`, after the `WriterQuery` tuple macro, before the `ReducerRegistry` section:

```rust
// ── QueryWriter (buffered query iteration) ──────────────────────────

/// Query iteration with buffered writes. Reads current values from archetype
/// columns; `&mut T` items become `WritableRef<T>` that buffer into a ChangeSet.
///
/// Uses manual archetype scanning (not `world.query()`) to avoid marking
/// mutable columns as changed, which would cause self-conflict with
/// optimistic validation.
pub struct QueryWriter<'a, Q: WriterQuery> {
    world: &'a mut World,
    changeset: *mut EnumChangeSet,
    last_read_tick: &'a Arc<AtomicU64>,
    _marker: PhantomData<Q>,
}

impl<'a, Q: WriterQuery + 'static> QueryWriter<'a, Q> {
    pub(crate) fn new(
        world: &'a mut World,
        changeset: *mut EnumChangeSet,
        last_read_tick: &'a Arc<AtomicU64>,
    ) -> Self {
        Self {
            world,
            changeset,
            last_read_tick,
            _marker: PhantomData,
        }
    }

    pub fn for_each(&mut self, mut f: impl FnMut(Q::WriterItem<'_>)) {
        use crate::tick::Tick;

        let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));
        let new_tick = self.world.next_tick();

        let required = Q::required_ids(&self.world.components);
        let cs_ptr = self.changeset;

        for arch in &self.world.archetypes.archetypes {
            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                continue;
            }
            if !Q::matches_filters(arch, &self.world.components, last_tick) {
                continue;
            }
            let fetch = Q::init_writer_fetch(arch, &self.world.components);
            for row in 0..arch.len() {
                let entity = arch.entities[row];
                let item = unsafe { Q::fetch_writer(&fetch, row, entity, cs_ptr) };
                f(item);
            }
        }

        self.last_read_tick.store(new_tick.raw(), Ordering::Relaxed);
    }

    pub fn count(&mut self) -> usize {
        use crate::tick::Tick;

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
        total
    }
}
```

**Step 4: Implement `register_query_writer`**

Add to `ReducerRegistry`, in the transactional registration section (after `register_spawner`):

```rust
/// Register a query writer reducer: `f(QueryWriter<Q>, args)`.
///
/// Iterates all matching entities, buffering writes through `ChangeSet`.
/// `&T` items pass through as read-only; `&mut T` items become `WritableRef<T>`.
/// Compatible with `Durable` for WAL logging. `Changed<T>` filters work.
pub fn register_query_writer<Q, Args, F>(
    &mut self,
    world: &mut World,
    name: &'static str,
    f: F,
) -> ReducerId
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

    let adapter: TransactionalAdapter = Box::new(
        move |changeset, _allocated, world, _resolved, args_any| {
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
            let qw = QueryWriter::<Q>::new(world, cs_ptr, &tick_ref);
            f(qw, args);
        },
    );

    self.push_entry(name, access, resolved, ReducerKind::Transactional(adapter))
}
```

Note: add `use std::sync::atomic::{AtomicU64, Ordering};` and `use std::sync::Arc;` to the imports at the top of `reducer.rs`.

**Step 5: Run tests**

Run: `cargo test -p minkowski --lib -- query_writer`
Expected: All query_writer tests PASS.

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

**Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat: QueryWriter handle + register_query_writer — buffered query iteration"
```

---

### Task 7: Update example, exports, and CLAUDE.md

**Files:**
- Modify: `crates/minkowski/src/lib.rs`
- Modify: `examples/examples/reducer.rs`
- Modify: `CLAUDE.md`

**Step 1: Add exports to `lib.rs`**

Update the `pub use reducer::` block to include new types:

```rust
pub use reducer::{
    ComponentSet, Contains, DynamicCtx, DynamicReducerBuilder, DynamicReducerId, EntityMut,
    EntityRef, QueryMut, QueryReducerId, QueryRef, QueryWriter, ReducerId, ReducerRegistry,
    Spawner, WritableRef, WriterQuery,
};
```

**Step 2: Add query writer demo to `examples/examples/reducer.rs`**

Add a new section after the spawner demo and before the name-based lookup:

```rust
// ── 6. Query writer reducer: persistent bulk update ────────────
let drag_id = registry.register_query_writer::<(&Velocity, &mut Velocity), f32, _>(
    &mut world,
    "apply_drag",
    |mut query, drag: f32| {
        query.for_each(|(vel_read, mut vel_write)| {
            vel_write.modify(|v| v.0 *= drag);
        });
    },
);
println!(
    "Registered 'apply_drag' query writer reducer (id={:?})",
    drag_id
);

// Dispatch — goes through strategy.transact(), compatible with Durable
registry.call(&strategy, &mut world, drag_id, 0.9f32).unwrap();
println!(
    "After drag: hero vel={:.2}",
    world.get::<Velocity>(hero).unwrap().0
);
```

Update the access conflict section to include query writer:
```rust
let drag_access = registry.reducer_access(drag_id);
println!(
    "gravity vs apply_drag: {}",
    if gravity_access.conflicts_with(drag_access) {
        "CONFLICT (both write Velocity)"
    } else {
        "compatible"
    }
);
```

Note: `register_query_writer::<(&Velocity, &mut Velocity), ...>` reads Velocity (for iteration matching and read access) and writes Velocity (via `WritableRef`). This means the query type has both `&Velocity` and `&mut Velocity`. Check if this causes issues with `WorldQuery` `required_ids` — both `&Velocity` and `&mut Velocity` produce the same required ID, so archetype matching works. The `Access` will have Velocity in both reads and writes, which is correct.

**Alternative if `(&Velocity, &mut Velocity)` causes trait coherence issues:** Use `(&mut Velocity,)` alone — `WritableRef::get()` provides read access, so no separate `&Velocity` needed:

```rust
let drag_id = registry.register_query_writer::<(&mut Velocity,), f32, _>(
    &mut world,
    "apply_drag",
    |mut query, drag: f32| {
        query.for_each(|(mut vel,)| {
            vel.modify(|v| v.0 *= drag);
        });
    },
);
```

**Step 3: Update CLAUDE.md**

In the Reducer System section, update the execution models table:

```markdown
| Model | Handle types | Isolation | Conflict detection |
|---|---|---|---|
| Transactional | `EntityMut<C>`, `Spawner<B>`, `QueryWriter<Q>` | Buffered writes via EnumChangeSet | Runtime (optimistic ticks or pessimistic locks) |
| Scheduled | `QueryMut<Q>`, `QueryRef<Q>` | Direct `&mut World` (hidden) | Compile-time (Access bitsets) |
| Dynamic | `DynamicCtx` | Buffered writes via EnumChangeSet | Conservative (builder-declared upper bounds) |
```

Add `QueryWriter<Q>` to the typed handles paragraph.

Update the `pub` list to include `QueryWriter`, `WritableRef`, `WriterQuery`.

Update the example command description:
```
cargo run -p minkowski-examples --example reducer --release   # Typed reducer system: entity/query/spawner/query-writer/dynamic handles + conflict detection
```

Update the dispatch description: `call()` for transactional reducers (replaces `call_entity()`).

**Step 4: Run full test suite + example**

Run: `cargo test -p minkowski --lib`
Run: `cargo clippy --workspace --all-targets -- -D warnings`
Run: `cargo run -p minkowski-examples --example reducer --release`
Expected: All pass.

**Step 5: Commit**

```bash
git add crates/minkowski/src/lib.rs examples/examples/reducer.rs CLAUDE.md
git commit -m "feat: QueryWriter exports, example, and documentation"
```

---

## Semantic Review Checklist

Before considering this complete, verify:

1. **Can this be called with the wrong World?** — `call()` passes world to `transact()` which checks WorldId. ✓
2. **Can Drop observe inconsistent state?** — `WritableRef` has no Drop. `QueryWriter` has no Drop. ✓
3. **Can two threads reach this through `&self`?** — `WritableRef` uses raw pointer, but it's ephemeral within `for_each` callback. `AtomicU64` for tick. ✓
4. **Does dedup/merge/collapse preserve the strongest invariant?** — N/A (no dedup here). ✓
5. **What happens if this is abandoned halfway through?** — Transaction abort path handles changeset cleanup. ✓
6. **Can a type bound be violated by a legal generic instantiation?** — `WriterQuery: WorldQuery` ensures only valid query types. ✓
7. **Does the API surface of this handle permit any operation not covered by the Access bitset?** — `WritableRef::set` writes to changeset for components in the query. `Access::of::<Q>()` captures these. No out-of-band access possible. ✓

## Verification Checklist

```bash
cargo test -p minkowski --lib                                    # All unit tests
cargo clippy --workspace --all-targets -- -D warnings            # Lint
cargo run -p minkowski-examples --example reducer --release      # Example
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- --skip par_for_each  # UB check
```
