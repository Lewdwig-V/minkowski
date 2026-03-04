# Reducer System Design

Date: 2026-03-04

## Problem

A transact closure takes `(&mut Tx, &World)` and can do anything — read any component, write any component, spawn entities, query the entire world. The access pattern is invisible to the engine until runtime. Two closures that happen to touch the same component conflict, and the engine can't detect this until commit fails.

Good reducer design means narrowing what a reducer *can* touch so that conflict freedom is provable from the type signature, not discoverable at runtime.

## Design Principles

1. **Type safety at construction, type erasure behind the handle.** The type system verifies invariants at registration time. The runtime representation (ReducerId, ComponentId) is the residue — a small integer that stands in for a proof the type system already verified.

2. **World is a storage engine, not a framework.** The ReducerRegistry lives outside World, same as SpatialIndex and Access. World's API surface doesn't grow.

3. **Make invalid states unrepresentable.** Every handle exposes exactly the operations its Access declaration covers. Undeclared reads, writes through read handles, and undeclared spawns are compile errors.

4. **Handle API surface = Access declaration.** If a handle permits any operation not covered by the Access bitset it was constructed from, the scheduler's conflict analysis is unsound. This is a review checklist item for every new handle type.

5. **Existing primitives compose into the reducer system without retrofit.** Atomic entity allocation (designed for parallel transactions), pre-resolved ComponentIds (designed to avoid registration during &World phase), and ChangeSet buffering (designed for optimistic concurrency) all converge to make reducers work with zero additional machinery.

## Two Execution Models

Reducers split into two execution models based on their conflict scope:

| Model | Shapes | Isolation | Conflict detection |
|---|---|---|---|
| Transactional | Entity, Pair, Spawner | Buffered writes via Tx + changeset | Runtime (optimistic ticks or pessimistic locks) |
| Scheduled | Query | Direct &mut World (hidden behind handle) | Compile-time (Access bitsets, scheduler guarantees exclusivity) |

Entity/pair reducers handle per-entity conflicts at runtime through the transaction layer. Query reducers handle per-column conflicts at batch time through the scheduler. Different mechanisms, same Access metadata.

## Foundation Traits: ComponentSet and Contains

`ComponentSet` declares a set of component types with pre-resolved IDs. `Contains<T>` is a compile-time proof that T is in the set, carrying a positional index for zero-cost ID lookup.

```rust
pub trait ComponentSet: 'static {
    const COUNT: usize;

    fn access(registry: &ComponentRegistry, read_only: bool) -> Access;

    fn resolve(registry: &ComponentRegistry) -> Vec<ComponentId>;
}

pub trait Contains<T: Component> {
    const INDEX: usize;
}
```

Macro-generated for tuples 1–12 (matching Bundle and WorldQuery arity):

```rust
impl<A: Component, B: Component> ComponentSet for (A, B) {
    const COUNT: usize = 2;
    fn access(registry: &ComponentRegistry, read_only: bool) -> Access {
        let mut access = Access::empty();
        let id_a = registry.id::<A>();
        let id_b = registry.id::<B>();
        if read_only {
            access.add_read(id_a);
            access.add_read(id_b);
        } else {
            access.add_write(id_a);
            access.add_write(id_b);
        }
        access
    }
    fn resolve(registry: &ComponentRegistry) -> Vec<ComponentId> {
        vec![registry.id::<A>(), registry.id::<B>()]
    }
}
impl<A: Component, B: Component> Contains<A> for (A, B) { const INDEX: usize = 0; }
impl<A: Component, B: Component> Contains<B> for (A, B) { const INDEX: usize = 1; }
```

`ResolvedComponents` is a `Vec<ComponentId>` created once at registration time. The positional const from `Contains<T>` indexes into it — one array lookup per get/set, no hash map, no type-to-ID resolution at runtime.

```rust
pub(crate) struct ResolvedComponents(Vec<ComponentId>);
```

## Access Builder API

`Access` gains a builder API for programmatic construction (currently only has `Access::of::<Q>(world)`):

```rust
impl Access {
    pub fn empty() -> Self;
    pub fn read(id: ComponentId) -> Self;
    pub fn write(id: ComponentId) -> Self;
    pub fn add_read(&mut self, id: ComponentId);
    pub fn add_write(&mut self, id: ComponentId);
    pub fn merge(&self, other: &Access) -> Access;
}
```

These are small additions. `ComponentSet::access()` and the registration methods use them to build Access from pre-resolved ComponentIds.

## Typed Handles

Five handles, each hiding World behind a facade that exposes exactly the declared operations:

### EntityRef — read-only entity access (transactional)

```rust
pub struct EntityRef<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    world: &'a World,
    _marker: PhantomData<C>,
}

impl<'a, C: ComponentSet> EntityRef<'a, C> {
    pub fn get<T: Component>(&self) -> &T
    where C: Contains<T>
    {
        let comp_id = self.resolved.0[C::INDEX];
        self.world.get_by_id(self.entity, comp_id)
    }

    pub fn entity(&self) -> Entity { self.entity }
}
```

### EntityMut — read-write entity access (transactional)

```rust
pub struct EntityMut<'a, C: ComponentSet> {
    entity: Entity,
    resolved: &'a ResolvedComponents,
    tx: &'a mut Tx<'a>,
    world: &'a World,
    _marker: PhantomData<C>,
}

impl<'a, C: ComponentSet> EntityMut<'a, C> {
    pub fn get<T: Component>(&self) -> &T
    where C: Contains<T>
    {
        let comp_id = self.resolved.0[C::INDEX];
        self.world.get_by_id(self.entity, comp_id)
    }

    pub fn set<T: Component>(&mut self, value: T)
    where C: Contains<T>
    {
        let comp_id = self.resolved.0[C::INDEX];
        self.tx.write_raw(self.entity, comp_id, value);
    }

    pub fn entity(&self) -> Entity { self.entity }
}
```

### Spawner — spawn capability (transactional)

```rust
pub struct Spawner<'a, B: Bundle> {
    resolved: &'a ResolvedComponents,
    tx: &'a mut Tx<'a>,
    world: &'a World,
    _marker: PhantomData<B>,
}

impl<'a, B: Bundle> Spawner<'a, B> {
    pub fn spawn(&mut self, bundle: B) -> Entity {
        self.tx.spawn_raw(self.world, &self.resolved, bundle)
    }
}
```

Entity allocation uses `EntityAllocator::reserve()` which is `&self` via `AtomicU32`. No `&mut World` needed.

### QueryRef — read-only query (scheduled)

```rust
pub struct QueryRef<'a, Q: ReadOnlyWorldQuery> {
    world: &'a World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: ReadOnlyWorldQuery> QueryRef<'a, Q> {
    pub fn for_each(&self, f: impl FnMut(Q::Item<'_>)) {
        self.world.query_raw::<Q>().for_each(f);
    }
}
```

### QueryMut — read-write query (scheduled)

```rust
pub struct QueryMut<'a, Q: WorldQuery> {
    world: &'a mut World,
    _marker: PhantomData<Q>,
}

impl<'a, Q: WorldQuery> QueryMut<'a, Q> {
    pub fn for_each(&mut self, f: impl FnMut(Q::Item<'_>)) {
        self.world.query::<Q>().for_each(f);
    }

    pub fn for_each_chunk(&mut self, f: impl FnMut(Q::Slice<'_>)) {
        self.world.query::<Q>().for_each_chunk(f);
    }
}
```

`&mut World` is inside the handle, unreachable to the closure. No `despawn`, no `insert`, no `query::<SomethingElse>`. The closure receives `QueryMut<(&mut Velocity,)>` and can only iterate Velocity mutably.

### Handle matrix

| Handle | Holds | API surface | Execution |
|---|---|---|---|
| `EntityRef<C>` | `&World` + resolved IDs | `get::<T>()` where `C: Contains<T>` | Transactional |
| `EntityMut<C>` | `&World` + `&mut Tx` + resolved IDs | `get::<T>()` + `set::<T>()` where `C: Contains<T>` | Transactional |
| `Spawner<B>` | `&World` + `&mut Tx` + resolved IDs | `spawn(bundle)` | Transactional |
| `QueryRef<Q>` | `&World` (hidden) | `for_each(f)` | Scheduled |
| `QueryMut<Q>` | `&mut World` (hidden) | `for_each(f)`, `for_each_chunk(f)` | Scheduled |

### Execution environment

```
Transactional (entity/pair/spawner):
  &World       → EntityRef::get()      reads via pre-resolved ComponentId
  &World       → EntityMut::get()      reads via pre-resolved ComponentId
  &mut Tx      → EntityMut::set()      buffers via tx.write_raw()
  &mut Tx      → Spawner::spawn()      reserves via AtomicU32, buffers components

Scheduled (query):
  &World       → QueryRef::for_each()  read-only iteration (hidden world)
  &mut World   → QueryMut::for_each()  read-write iteration (hidden world)
```

No `&mut World` in transactional reducer bodies. No unscoped World access in any reducer body.

## Raw Paths (pub(crate))

Pre-resolved variants of existing methods. The typed handles call these; external users never see them.

### Tx extensions

```rust
impl Tx<'_> {
    /// Write using pre-resolved ComponentId. No registration, no &mut World.
    pub(crate) fn write_raw<T: Component>(
        &mut self, entity: Entity, comp_id: ComponentId, value: T,
    ) {
        self.changeset.insert_raw(entity, comp_id, value);
    }

    /// Remove using pre-resolved ComponentId.
    pub(crate) fn remove_raw(&mut self, entity: Entity, comp_id: ComponentId) {
        self.changeset.record_remove(entity, comp_id);
    }

    /// Spawn with pre-resolved ComponentIds. Atomic entity allocation.
    pub(crate) fn spawn_raw<B: Bundle>(
        &mut self, world: &World, resolved: &ResolvedComponents, bundle: B,
    ) -> Entity {
        let entity = world.entities.reserve();
        self.track_allocated(entity);
        self.changeset.spawn_bundle_raw(entity, resolved, bundle);
        entity
    }
}
```

### EnumChangeSet extensions

```rust
impl EnumChangeSet {
    /// Insert with pre-resolved ComponentId. Handles ManuallyDrop + DropEntry.
    pub(crate) fn insert_raw<T: Component>(
        &mut self, entity: Entity, comp_id: ComponentId, value: T,
    ) {
        let value = ManuallyDrop::new(value);
        let ptr = &*value as *const T as *const u8;
        let layout = Layout::new::<T>();
        let offset = self.arena.len();
        self.drop_entries.push(DropEntry {
            offset, layout,
            drop_fn: needs_drop::<T>().then(|| drop_ptr::<T> as unsafe fn(*mut u8)),
        });
        self.record_insert(entity, comp_id, ptr, layout);
    }

    /// Spawn bundle with pre-resolved ComponentIds.
    pub(crate) fn spawn_bundle_raw<B: Bundle>(
        &mut self, entity: Entity, resolved: &ResolvedComponents, bundle: B,
    );
}
```

### World extensions

```rust
impl World {
    /// Get component by pre-resolved ComponentId.
    pub(crate) fn get_by_id<T: Component>(&self, entity: Entity, comp_id: ComponentId) -> &T;
}
```

### Two-path design

| Path | Resolution | Consumer | Visibility |
|---|---|---|---|
| `tx.write::<T>()` | Runtime lookup/register | Raw transact closures (tier 1) | `pub` |
| `tx.write_raw()` | Pre-resolved ComponentId | EntityMut, Spawner (tier 2) | `pub(crate)` |

## ReducerRegistry

External to World. Owns closures, Access metadata, and ResolvedComponents. Two registration paths for the two execution models, unified Access output.

### Storage

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReducerId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryReducerId(usize);

pub(crate) enum ReducerKind {
    Transactional(Box<dyn Fn(&mut Tx<'_>, &World, &ResolvedComponents, &dyn Any) + Send + Sync>),
    Scheduled(Box<dyn Fn(&mut World, &dyn Any) + Send + Sync>),
}

struct ReducerEntry {
    name: &'static str,
    access: Access,
    resolved: ResolvedComponents,
    kind: ReducerKind,
}

pub struct ReducerRegistry {
    reducers: Vec<ReducerEntry>,
    by_name: HashMap<&'static str, usize>,
}
```

### Transactional registration (entity/pair/spawner)

```rust
impl ReducerRegistry {
    pub fn register_entity<C, Args, F>(
        &mut self, world: &mut World, name: &'static str, f: F,
    ) -> ReducerId
    where
        C: ComponentSet, Args: 'static,
        F: Fn(EntityMut<'_, C>, Args) + Send + Sync + 'static;

    pub fn register_pair<R, W, Args, F>(
        &mut self, world: &mut World, name: &'static str, f: F,
    ) -> ReducerId
    where
        R: ComponentSet, W: ComponentSet, Args: 'static,
        F: Fn(EntityRef<'_, R>, EntityMut<'_, W>, Args) + Send + Sync + 'static;

    pub fn register_spawner<R, B, Args, F>(
        &mut self, world: &mut World, name: &'static str, f: F,
    ) -> ReducerId
    where
        R: ComponentSet, B: Bundle, Args: 'static,
        F: Fn(EntityRef<'_, R>, Spawner<'_, B>, Args) + Send + Sync + 'static;
}
```

Type erasure at registration: the typed closure is wrapped in an adapter that downcasts `&dyn Any` args at dispatch time. The registration function's signature enforces type safety. The ReducerId is opaque — no type parameters.

Component types are pre-registered as a side effect of `ComponentSet::resolve()` / `Access::of()` at registration time. By the time a reducer exists in the registry, every component it touches has a stable ComponentId.

### Scheduled registration (query)

```rust
impl ReducerRegistry {
    pub fn register_query<Q, Args, F>(
        &mut self, world: &mut World, name: &'static str, f: F,
    ) -> QueryReducerId
    where
        Q: WorldQuery + 'static, Args: 'static,
        F: Fn(QueryMut<'_, Q>, Args) + Send + Sync + 'static;

    pub fn register_query_ref<Q, Args, F>(
        &mut self, world: &mut World, name: &'static str, f: F,
    ) -> QueryReducerId
    where
        Q: ReadOnlyWorldQuery + 'static, Args: 'static,
        F: Fn(QueryRef<'_, Q>, Args) + Send + Sync + 'static;
}
```

The scheduler benefits from the `QueryRef`/`QueryMut` distinction — read-only query reducers can be batched together.

### Dispatch

```rust
impl ReducerRegistry {
    /// Call a transactional reducer with a chosen strategy.
    pub fn call<S: Transact, Args: 'static>(
        &self, strategy: &S, world: &mut World, id: ReducerId, args: Args,
    ) -> Result<(), Conflict>;

    /// Run a scheduled reducer directly (caller guarantees exclusivity).
    pub fn run<Args: 'static>(
        &self, world: &mut World, id: QueryReducerId, args: Args,
    );

    /// Name-based lookup for network dispatch.
    pub fn id_by_name(&self, name: &str) -> Option<usize>;

    /// Access metadata for scheduler integration.
    pub fn access(&self, idx: usize) -> &Access;
}
```

The caller composes registry + strategy + world:

```rust
// Registration
let attack_id = registry.register_pair::<(Attack,), (Health,), (), _>(
    &mut world, "attack",
    |attacker, mut target, ()| {
        let damage = attacker.get::<Attack>().power;
        let hp = target.get::<Health>();
        target.set(Health { hp: hp.hp - damage, ..hp });
    },
);

// Local dispatch
registry.call(&optimistic, &mut world, attack_id, (attacker, target, ()))?;

// Network dispatch
let idx = registry.id_by_name("attack").unwrap();
registry.call(&optimistic, &mut world, ReducerId(idx), (attacker, target, ()))?;
```

## API Layering

```
Tier 1: Raw transact — full &World access, no restrictions
  strategy.transact(&mut world, access, |tx, world| { ... })
  For: framework authors, one-off migrations, debug tools

Tier 2: Transactional reducers — declared access, typed handles
  registry.call(&strategy, &mut world, attack_id, args)
  For: game logic, network-delivered actions, replayable commands

Tier 3: Query reducers — declared column access, typed iteration
  registry.run(&mut world, gravity_id, dt)
  For: physics, AI, batch updates
```

Each tier adds constraints that make more bugs impossible. Nobody is forced into higher tiers, but the type system rewards them with stronger guarantees.

## Compile-Time Safety Guarantees

**Reading undeclared components** — compile error:
```rust
registry.register_entity::<(Health,), (), _>(&mut world, "heal", |mut entity, ()| {
    let energy = entity.get::<Energy>();  // ERROR: (Health,) does not implement Contains<Energy>
});
```

**Writing through a read handle** — compile error:
```rust
registry.register_pair::<(Attack,), (Health,), (), _>(&mut world, "attack",
    |attacker, mut target, ()| {
        attacker.set(Attack { power: 999 });  // ERROR: EntityRef has no set method
    }
);
```

**Undeclared column access in query reducer** — impossible, World is hidden:
```rust
registry.register_query::<&mut Velocity, f32, _>(&mut world, "gravity", |mut query, dt| {
    // query.world  ← private field
    // world.despawn(...)  ← no world in scope
    query.for_each(|vel| vel.dy -= 9.81 * dt);  // only this is possible
});
```

## What Reducers Don't Restrict

Entity identity — *which* entities the reducer operates on — is a runtime argument. The type system constrains *what components* can be accessed, not *which entities*. Two attack reducers targeting the same entity conflict at the entity level, detected by optimistic tick validation or pessimistic locks. The type system ensures column-level access is correct; the transaction system ensures entity-level access is consistent.

Column access is static (determined by the reducer's purpose). Entity access is dynamic (determined by game state). Static constraints go in the type system. Dynamic constraints go in the transaction.

## Macro Generation

`impl_component_set!` generates `ComponentSet` and `Contains<T>` impls for tuples 1–12, matching the existing `impl_bundle!` and `impl_world_query_tuple!` pattern. Each invocation generates:

- One `ComponentSet` impl for the tuple
- N `Contains<T>` impls (one per position, each with `const INDEX`)

## Semantic Review Checklist Addition

> Does the API surface of this handle permit any operation not covered by the Access bitset it was constructed from? If yes, the scheduler's conflict analysis is unsound, and any two reducers that the scheduler batches as "compatible" might actually interfere through the undeclared operation.

## Why This Works Without New Primitives

The reducer system composes from existing infrastructure:

- **Atomic entity allocation** (designed for parallel transactions) → Spawner works with `&World`
- **Pre-resolved ComponentId** (designed to avoid registration during &World phase) → typed handles skip runtime lookup
- **ChangeSet buffering** (designed for optimistic concurrency) → EntityMut::set buffers without &mut World
- **Access bitsets** (designed for conflict detection) → same metadata drives both transaction validation and scheduler batching
- **ReadOnlyWorldQuery** (designed for Tx soundness) → QueryRef uses the same gate

Each was justified independently. They compose into the reducer system because they're all instances of the same principle: defer mutation to the commit boundary, keep the execution phase read-only.
