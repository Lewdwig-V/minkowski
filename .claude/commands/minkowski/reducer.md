---
description: Help with Minkowski ECS reducer selection — EntityMut, QueryMut, QueryRef, QueryWriter, Spawner, DynamicCtx
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user choose and implement the right reducer type for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing reducer patterns:

- `ReducerRegistry` construction and usage
- `register_entity` / `register_entity_despawn` calls (EntityMut reducers)
- `register_query` / `register_query_ref` / `register_query_writer` calls
- `register_spawner` calls
- `registry.dynamic()` builder usage (DynamicCtx)
- `registry.call()` (transactional dispatch) vs `registry.run()` (scheduled dispatch)
- `registry.dynamic_call()` (dynamic dispatch)
- Access patterns: which components are read vs written by each system
- Whether reducers need to be durable (WAL-compatible)

## Step 2: Recommend

**Strong defaults — pick the right handle type:**

| Need | Handle | Registration | Dispatch |
|---|---|---|---|
| Single-entity read + write | `EntityMut<C>` | `register_entity` | `call()` (transactional) |
| Single-entity with despawn/remove | `EntityMut<C>` | `register_entity_despawn` | `call()` (transactional) |
| Entity creation | `Spawner<B>` | `register_spawner` | `call()` (transactional) |
| Bulk read-only iteration | `QueryRef<Q>` | `register_query_ref` | `run()` (scheduled) |
| Bulk read-write, direct | `QueryMut<Q>` | `register_query` | `run()` (scheduled) |
| Bulk read-write, buffered | `QueryWriter<Q>` | `register_query_writer` | `call()` (transactional) |
| Runtime-conditional access | `DynamicCtx` | `registry.dynamic()` builder | `dynamic_call()` |

**Key distinctions:**
- `QueryMut` vs `QueryWriter`: QueryMut has direct `&mut T` (faster, compile-time safety, scheduled via `run()`). QueryWriter buffers writes via `WritableRef<T>` (transactional, durable-compatible via `call()`). **Use QueryWriter when the reducer needs to be durable (WAL-logged).**
- `EntityMut::remove()` is bounded by `Contains<T, IDX>` — you can only remove components declared in the component set.
- `EntityMut::despawn()` requires `register_entity_despawn` (sets despawn flag on Access).
- `Spawner` uses `EntityAllocator::reserve(&self)` for lock-free ID allocation — works inside transactional closures where only `&World` is available.
- Dynamic reducers (`DynamicCtx`): declare upper-bound access at registration time. Accessing undeclared types panics in all builds (`assert!`, not `debug_assert!`).

**Ask if unclear:**
- "Does the access pattern depend on runtime state?" — Use Dynamic reducer. Declare the superset of possible accesses in the builder.
- "Does this need to be durable (WAL-logged)?" — Use `QueryWriter` over `QueryMut`.
- "Do you need to despawn or remove components?" — Use `register_entity_despawn` or dynamic with `can_despawn()`/`can_remove()`.
- "Is this a read-only observer?" — Use `QueryRef` (register_query_ref). Cheaper, no mutable access needed.

**Registration pattern:**
```
let mut registry = ReducerRegistry::new();

// Entity reducer
let heal_id = registry.register_entity::<(Health,), u32, _>(
    &mut world, "heal",
    |entity_mut: EntityMut<(Health,)>, amount: u32| { ... },
);

// Query reducer (scheduled)
let movement_id = registry.register_query::<(&mut Pos, &Vel), f32, _>(
    &mut world, "movement",
    |mut query: QueryMut<(&mut Pos, &Vel)>, dt: f32| { ... },
);

// Dispatch
registry.call(&strategy, &mut world, heal_id, (entity, 10u32));
registry.run(&mut world, movement_id, 0.016f32);
```

## Step 3: Implement

Help write reducer code. Point to relevant examples:

- **All handle types**: See `examples/examples/reducer.rs` — comprehensive reference with EntityMut, QueryMut, QueryRef, QueryWriter, Spawner, DynamicCtx
- **EntityMut with rayon**: See `examples/examples/battle.rs` — parallel dispatch of entity reducers
- **QueryWriter + Durable**: See `examples/examples/persist.rs` — buffered writes compatible with WAL
- **QueryMut for simulation**: See `examples/examples/boids.rs` and `examples/examples/nbody.rs` — query reducers for physics steps
- **Access conflict detection**: See `examples/examples/scheduler.rs` — `registry.query_reducer_access(id)` for scheduling
- **Dynamic reducer**: See `examples/examples/reducer.rs` — `DynamicReducerBuilder` with `can_read`/`can_write`/`can_spawn`

For architecture details, see CLAUDE.md § "Reducer System".
