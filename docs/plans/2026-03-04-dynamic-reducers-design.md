# Dynamic Reducers — Design Document

> Third registration method for `ReducerRegistry`. Runtime-flexible access with builder-declared upper bounds and debug-asserted enforcement.

**Date:** 2026-03-04
**Prereq:** Reducer system (PR #21, merged)
**Exploration:** `docs/plans/dynamic-reducers.md`

---

## Motivation

Static reducers (`register_entity`, `register_pair`, `register_spawner`) enforce access at compile time — the type signature is the access declaration. But some game logic touches components conditionally based on runtime state (e.g., "apply shield only when HP < 30"). Static reducers can't express this: they must declare access for all possible paths, and the type system can't distinguish "might read" from "always reads."

Dynamic reducers trade compile-time precision for runtime flexibility. The user declares an upper bound on access, and the runtime validates actual usage in debug builds.

## Cost Matrix

|                    | Static       | Query        | Dynamic           |
|--------------------|-------------|-------------|-------------------|
| Conflict analysis  | Exact        | Exact        | Conservative      |
| Undeclared access  | Compile error| Compile error| Panic (always) + debug_assert (read vs write) |
| SIMD               | No (buffered)| Yes (direct) | No (buffered)     |
| Rollback           | Yes          | No           | Yes               |
| Conditional access | No           | No           | Yes               |
| Overhead           | ChangeSet    | Zero         | ChangeSet + binary search per access |

## Registration: Builder Pattern

```rust
let id = reducers.dynamic("special_ability", &mut world)
    .can_read::<Health>()
    .can_read::<Energy>()
    .can_write::<Energy>()
    .can_write::<Shield>()
    .can_spawn::<(StatusEffect, Duration)>()
    .build(|ctx: &mut DynamicCtx, args: &AbilityArgs| {
        let hp = ctx.read::<Health>(args.caster);
        let energy = ctx.read::<Energy>(args.caster);
        if energy.mana >= 50.0 {
            ctx.write(args.caster, Energy { mana: energy.mana - 50.0 });
            if hp.hp < 30.0 {
                ctx.write(args.caster, Shield { active: true, duration: 5.0 });
            }
        }
    });
```

### `DynamicReducerBuilder`

```rust
pub struct DynamicReducerBuilder<'a> {
    registry: &'a mut ReducerRegistry,
    world: &'a mut World,
    name: &'static str,
    access: Access,
    resolved: Vec<(TypeId, ComponentId)>,
    spawn_bundles: HashSet<TypeId>,
}
```

Takes `&mut World` because `register::<T>()` may allocate new ComponentIds.

**Builder methods:**
- `can_read::<T>()` — registers T, captures `(TypeId, ComponentId)`, adds read to Access
- `can_write::<T>()` — registers T, captures `(TypeId, ComponentId)`, adds write to Access
- `can_spawn::<B: Bundle>()` — registers all bundle components, adds writes to Access, records `TypeId::of::<B>()` for debug assertion
- `build::<Args, F>(f)` — sorts + dedupes resolved entries, constructs `DynamicResolved`, stores type-erased adapter

`can_read` + `can_write` on the same type: both are recorded. Read lookup checks `reads || writes` in Access. Write lookup checks `writes` only.

## Runtime: DynamicCtx

```rust
pub struct DynamicCtx<'a> {
    world: &'a World,
    changeset: &'a mut EnumChangeSet,
    allocated: &'a mut Vec<Entity>,
    resolved: &'a DynamicResolved,
}
```

Borrows disjoint Tx fields (same pattern as EntityMut/Spawner from static path). Does NOT hold `&mut Tx` — avoids lifetime entanglement.

### Pre-resolved Lookup

```rust
pub(crate) struct DynamicResolved {
    entries: Vec<(TypeId, ComponentId)>,  // sorted by TypeId for binary search
    access: Access,
    spawn_bundles: HashSet<TypeId>,
}
```

Component IDs are resolved at registration time in the builder. At runtime, `DynamicCtx` does a binary search by `TypeId` — O(log n) where n is the number of declared components (typically 2-10).

### Methods

| Method | Requires | Behavior |
|--------|----------|----------|
| `read::<T>(entity) -> &T` | `can_read` or `can_write` | Reads from World via `get_by_id`. Panics if entity missing component. |
| `try_read::<T>(entity) -> Option<&T>` | `can_read` or `can_write` | Returns None if entity doesn't have T. |
| `write::<T>(entity, value)` | `can_write` | Buffers into EnumChangeSet via `insert_raw`. |
| `try_write::<T>(entity, value) -> bool` | `can_write` | Buffers write only if entity has T. Returns success. |
| `spawn::<B>(bundle) -> Entity` | `can_spawn::<B>()` | Reserves entity via `entities.reserve()`, buffers spawn. |

### Assertion Model

**Always-on (all builds):**
- Accessing a type not in `resolved` → panic. There is no `ComponentId` to look up — this is not a debug-only check.

**Debug-only (`debug_assert!`):**
- `write::<T>()` on a type declared only with `can_read` → debug_assert failure (write access not declared)
- `spawn::<B>()` with a bundle TypeId not in `spawn_bundles` → debug_assert failure

The always-on panic catches "forgot to declare this type entirely." The debug_assert catches "declared as read but used as write" and "spawned undeclared bundle." In release, the scheduler already uses the declared Access for conflict prevention — under-declaration is a scheduler bug (too much parallelism), not a memory safety issue.

## Registry Integration

```rust
pub struct DynamicReducerId(pub(crate) usize);

struct DynamicReducerEntry {
    name: &'static str,
    access: Access,
    resolved: DynamicResolved,
    closure: Box<dyn Fn(&mut DynamicCtx, &dyn Any) + Send + Sync>,
}
```

ReducerRegistry grows a third vec:

```rust
pub struct ReducerRegistry {
    reducers: Vec<ReducerEntry>,
    query_reducers: Vec<QueryReducerEntry>,
    dynamic_reducers: Vec<DynamicReducerEntry>,
    by_name: HashMap<&'static str, ReducerSlot>,
}

enum ReducerSlot {
    Static(usize),
    Query(usize),
    Dynamic(usize),
}
```

### Dispatch

```rust
impl ReducerRegistry {
    pub fn dynamic_call<S: Transact, Args: 'static>(
        &self, strategy: &S, world: &mut World,
        id: DynamicReducerId, args: &Args,
    ) -> Result<(), Conflict> {
        let entry = &self.dynamic_reducers[id.index()];
        strategy.transact(world, &entry.access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let mut ctx = DynamicCtx {
                world, changeset, allocated,
                resolved: &entry.resolved,
            };
            (entry.closure)(&mut ctx, args);
        })
    }

    pub fn dynamic_id_by_name(&self, name: &str) -> Option<DynamicReducerId> { ... }
    pub fn dynamic_access(&self, id: DynamicReducerId) -> &Access { ... }
}
```

Name registration panics on duplicates (same as static/query reducers).

## What Does NOT Ship

| Feature | Reason |
|---------|--------|
| `DynamicCtx::query()` | Would need QueryMut/QueryRef integration; complex filter semantics |
| Dynamic query reducers | `register_query` is static only |
| Runtime access tracking | No warmup profiling, no access reports |
| `can_despawn` | Despawn access semantics not designed |
| `can_remove` | Removal access semantics not designed |
| `DynamicCtx::remove()` | No `can_remove` to declare it |

## Files

| File | Change |
|------|--------|
| `crates/minkowski/src/reducer.rs` | Add DynamicReducerBuilder, DynamicResolved, DynamicCtx, DynamicReducerEntry, DynamicReducerId; extend ReducerRegistry |
| `crates/minkowski/src/access.rs` | Add convenience `reads_component(id)` / `writes_component(id)` if needed |
| `crates/minkowski/src/lib.rs` | Add DynamicReducerId, DynamicCtx to pub exports |
| `examples/examples/reducer.rs` | Add dynamic reducer demos |

No new files. No new dependencies.

## Testing

1. **Builder correctness**: `can_read`/`can_write`/`can_spawn` produce correct Access and resolved entries
2. **DynamicCtx operations**: read, try_read, write, try_write, spawn — with commit verification
3. **Debug assertions**: `#[should_panic]` for undeclared type, `#[cfg(debug_assertions)]` for read-vs-write mismatch
4. **Transact integration**: dynamic_call through Optimistic (success + conflict), Pessimistic (lock acquisition)
5. **Cross-reducer conflict**: dynamic Access conflicts with static Access on overlapping components
6. **Example**: demonstrate conditional access pattern — the motivating use case
