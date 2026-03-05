# Durable Reducers — Design Document

> QueryWriter handle + unified `call` dispatch. Buffered query iteration for WAL-compatible bulk updates.

**Date:** 2026-03-05
**Prereq:** Reducer system (PR #21), Dynamic reducers (PR #23)
**Exploration:** `docs/plans/durable-reducers.md`

---

## Motivation

Transactional reducers (`register_entity`, `register_spawner`, dynamic) go through `strategy.transact()` — wrapping with `Durable` gives WAL logging for free. Query reducers (`register_query`, `register_query_ref`) take `&mut World` and mutate directly — no ChangeSet, no WAL logging possible.

`QueryWriter` fills this gap: iterate like a query, buffer like a transaction. The SIMD path is lost, but the iteration pattern and filter support (`Changed<T>`) are preserved.

## Cost Matrix

| Reducer type | Mutation | SIMD | Persisted | Use case |
|---|---|---|---|---|
| `register_entity` | Buffered | No | Via Durable | Targeted entity operations |
| `register_query` | Direct | Yes | No | Hot loops, derived state |
| `register_query_writer` | Buffered | No | Via Durable | Bulk updates needing persistence |

The user chooses the tradeoff explicitly. The type system enforces it: different registration methods, different handles, different write paths.

## `WriterQuery` Trait

Separate trait in `reducer.rs`. NOT added to `WorldQuery` — persistence concerns stay out of the storage engine.

```rust
/// Maps WorldQuery items to their buffered writer equivalents.
/// &T passes through unchanged. &mut T becomes WritableRef<T>.
pub trait WriterQuery: WorldQuery {
    type WriterItem<'a>;

    fn fetch_writer<'a>(
        fetch: &Self::Fetch<'a>,
        row: usize,
        entity: Entity,
        changeset: &'a mut EnumChangeSet,
        resolved: &ResolvedComponents,
    ) -> Self::WriterItem<'a>;
}
```

### Item Mapping

| WorldQuery type | `WriterItem` |
|---|---|
| `&T` | `&T` (passthrough) |
| `&mut T` | `WritableRef<'a, T>` |
| `Option<&T>` | `Option<&T>` (passthrough) |
| `Option<&mut T>` | `Option<WritableRef<'a, T>>` |
| `Entity` | `Entity` (passthrough) |
| `Changed<T>` | N/A — filter only, no item transformation |
| Tuples 1-12 | Macro-generated, maps each element |

The trait bound on `register_query_writer` is `Q: WriterQuery + 'static`. Since `WriterQuery: WorldQuery`, all existing query matching, archetype scanning, and filter support works unchanged.

## `WritableRef<T>`

Per-component buffered write handle. Each `&mut T` in the query becomes a `WritableRef<T>` in the callback.

```rust
pub struct WritableRef<'a, T: Component> {
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: &'a mut EnumChangeSet,
}

impl<'a, T: Component> WritableRef<'a, T> {
    /// Read current value (zero-cost — pointer into BlobVec).
    pub fn get(&self) -> &T { self.current }

    /// Buffer a replacement value into the ChangeSet.
    pub fn set(&mut self, value: T) {
        self.changeset.insert_raw::<T>(self.entity, self.comp_id, value);
    }

    /// Clone-mutate-set in one call. Clone bound only on this method.
    pub fn modify(&mut self, f: impl FnOnce(&mut T)) where T: Clone {
        let mut value = self.current.clone();
        f(&mut value);
        self.set(value);
    }
}
```

### Design Points

- No `Deref`/`DerefMut` to `&mut T` — writes always go through `set`/`modify`, never bypass the ChangeSet
- `comp_id` is pre-resolved at registration time, not looked up per-row
- `Clone` bound on `modify` only, not on the struct — components that aren't `Clone` use `get`/`set`
- Uses existing `insert_raw` which takes `(entity, comp_id, value)` without `&mut World`

### Method Summary

| Method | When | Cost |
|---|---|---|
| `get(&self) -> &T` | Read current value | Zero (pointer into BlobVec) |
| `set(&mut self, T)` | Replace entire component | One arena write |
| `modify(&mut self, FnOnce(&mut T))` | Tweak fields in place | Clone + arena write |

## `QueryWriter<Q>` Handle

```rust
pub struct QueryWriter<'a, Q: WriterQuery> {
    world: &'a mut World,
    changeset: &'a mut EnumChangeSet,
    resolved: &'a ResolvedComponents,
    _marker: PhantomData<Q>,
}

impl<'a, Q: WriterQuery> QueryWriter<'a, Q> {
    pub fn for_each(&mut self, mut f: impl FnMut(Q::WriterItem<'_>)) {
        // Uses world.query::<Q>() — full path with Changed<T> filters,
        // tick management, query cache. Each row: read-only items pass
        // through, &mut T items become WritableRef pointing at current
        // value + changeset.
    }

    pub fn count(&mut self) -> usize { ... }
}
```

`QueryWriter` holds `&mut World` (not `&World`) because `world.query()` requires `&mut self` for cache mutation and tick tracking. This is safe inside `transact` — the closure already has exclusive access. Writes are buffered into `ChangeSet`, not applied directly.

### Ergonomics Comparison

```rust
// Direct query reducer — zero cost, SIMD, not persistent
reducers.register_query::<(&Position, &mut Velocity), _, _>(
    "gravity", &world, |mut query, dt: f32| {
        query.for_each(|(pos, vel)| {
            vel.dy -= 9.81 * dt;
        });
    }
);

// Query writer — buffered, persistent, same feel
reducers.register_query_writer::<(&Position, &mut Velocity), _, _>(
    "gravity_persistent", &world, |mut query, dt: f32| {
        query.for_each(|pos, mut vel| {
            vel.modify(|v| {
                v.dy -= 9.81 * dt;
            });
        });
    }
);
```

## Registration

```rust
impl ReducerRegistry {
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
        let resolved = ResolvedComponents(/* pre-resolve mutable component IDs */);
        let access = Access::of::<Q>(world);
        // Returns ReducerId — transactional path via ReducerKind::Transactional
    }
}
```

Returns `ReducerId` — same ID type as entity and spawner reducers. All three go through `strategy.transact()`, produce a `ChangeSet`, work with `Durable`.

## Unified `call` Dispatch

### Current State

`call_entity` takes `entity: Entity` as a separate parameter:

```rust
pub fn call_entity<S, Args>(
    &self, strategy: &S, world: &mut World,
    id: ReducerId, entity: Entity, args: Args,
) -> Result<(), Conflict>
```

### New State

Entity is just args. Single `call` method:

```rust
pub fn call<S: Transact, Args: Clone + 'static>(
    &self,
    strategy: &S,
    world: &mut World,
    id: ReducerId,
    args: Args,
) -> Result<(), Conflict>
```

### Adapter Refactor

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

Changes:
- `&World` → `&mut World` (QueryWriter needs `world.query()`)
- `Entity` parameter removed — entity reducers fold it into args
- `ReducerKind::EntityTransactional` → `ReducerKind::Transactional`

### Call Site Migration

```rust
// Entity reducer: entity is part of args
registry.call(&strategy, &mut world, heal_id, (hero, 25u32))?;

// Query writer: no entity
registry.call(&strategy, &mut world, gravity_id, 0.1f32)?;

// Spawner: no entity
registry.call(&strategy, &mut world, spawn_id, 75u32)?;
```

`call_entity` is removed (breaking change). `run()` stays for scheduled query reducers (`QueryReducerId`). `dynamic_call()` stays for dynamic reducers (`DynamicReducerId`).

## What Does NOT Ship

| Feature | Reason |
|---|---|
| `WritableRef::deref_mut` | Bypasses ChangeSet — the bug this design prevents |
| `QueryWriter::par_for_each` | Parallel iteration with shared `&mut ChangeSet` needs partitioning design |
| `WriterQuery` for `Changed<T>` | Filter-only — no item type to transform |
| Automatic SIMD for query writers | Buffered writes are inherently non-vectorizable |
| `register_query_writer_ref` | Read-only query writer is just `register_query_ref` |

## Files

| File | Change |
|---|---|
| `crates/minkowski/src/reducer.rs` | `WritableRef`, `WriterQuery` trait + impls, `QueryWriter` handle, `register_query_writer`, unified `call`/`TransactionalAdapter`, rename `EntityTransactional` → `Transactional` |
| `crates/minkowski/src/lib.rs` | Export `WritableRef`, `QueryWriter`, `WriterQuery` |
| `crates/minkowski/src/changeset.rs` | None — `insert_raw` already exists |
| `examples/examples/reducer.rs` | Update `call_entity` → `call`, add query writer demo |
| `CLAUDE.md` | Update cost matrix, reducer docs, pub exports |

No new files. No new dependencies.

## Testing

1. **`WritableRef`** — `get` returns current value, `set` buffers into ChangeSet, `modify` clones + mutates + sets, `set` works without `Clone` bound
2. **`WriterQuery` impls** — `&T` passthrough, `&mut T` → `WritableRef`, `Option` wrapping, tuple mapping
3. **`QueryWriter::for_each`** — reads correct values, buffers writes, writes applied after commit
4. **`QueryWriter` + `Changed<T>`** — filter respected (only changed entities visited)
5. **`register_query_writer` + `call`** — end-to-end dispatch through `Optimistic`
6. **Unified `call`** — entity reducer with `(entity, args)`, spawner with args, query writer with args
7. **Access conflict** — query writer vs entity reducer on overlapping components detected correctly
