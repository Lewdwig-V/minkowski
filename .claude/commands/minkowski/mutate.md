---
description: Help with Minkowski ECS mutation strategies — CommandBuffer, EnumChangeSet, reducer handles, direct World methods
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user choose and implement the right mutation strategy for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing mutation patterns:

- `world.insert()` / `world.remove()` calls, especially inside loops (archetype migration cost)
- `CommandBuffer` usage for deferred structural changes
- `EnumChangeSet` usage for reversible/serializable mutations
- `world.spawn()` / `world.despawn()` call patterns
- `tx.write()` / `tx.spawn()` / `tx.remove()` for transactional mutations
- Reducer handle usage: `EntityMut`, `QueryWriter`, `Spawner`
- Any mutation happening inside a query `for_each` closure (cannot do structural changes there)

## Step 2: Recommend

**Strong defaults:**
- **Direct World methods** (`world.spawn`, `world.insert`, `world.get_mut`): Only for setup and debugging. Not for game loop logic.
- **CommandBuffer**: For structural changes (spawn/despawn/insert/remove) during query iteration. Store commands, apply after iteration: `cmds.apply(&mut world)`.
- **EnumChangeSet**: For reversible/serializable mutations. `apply()` returns a reverse changeset — applying the reverse undoes the original. Used by persistence (WAL serialization) and transactions internally.
- **Reducer handles**: The idiomatic mutation path for game loop logic:
  - `EntityMut<C>` — read + buffered write + remove for a single entity
  - `QueryWriter<Q>` — bulk iteration with buffered writes via `WritableRef<T>` (durable-compatible)
  - `QueryMut<Q>` — bulk iteration with direct `&mut T` (scheduled, compile-time safety)
  - `Spawner<B>` — entity creation via `reserve()` (lock-free allocation)
- **Every mutation in the game loop should go through a reducer** — this gives you conflict detection, transaction support, and persistence compatibility for free.

**Ask if unclear:**
- "Do you need rollback/undo?" — `EnumChangeSet` with reverse changeset. See `life.rs` for undo/redo pattern.
- "Is this part of a transaction?" — Use reducer handles (`EntityMut`, `QueryWriter`). They buffer writes into `EnumChangeSet` automatically.
- "Are you spawning/despawning during iteration?" — `CommandBuffer` for deferred application after the loop.
- "Does this mutation need to be durable (crash-safe)?" — Use `QueryWriter` (buffers writes, compatible with WAL via `Durable`). `QueryMut` writes directly and cannot be WAL-logged.

## Step 3: Implement

Help write mutation code. Point to relevant examples:

- **CommandBuffer**: `let mut cmds = CommandBuffer::new(); cmds.spawn((Pos::default(),)); cmds.apply(&mut world);`
- **EnumChangeSet undo**: See `examples/examples/life.rs` — builds changeset per generation, pushes reverse onto undo stack
- **EntityMut reducer**: See `examples/examples/reducer.rs` — `EntityMut<(Health,)>` for single-entity mutation
- **QueryWriter reducer**: See `examples/examples/persist.rs` — `QueryWriter` with `WritableRef::modify()` for buffered writes
- **Spawner reducer**: See `examples/examples/reducer.rs` — `Spawner<(Pos, Vel)>` for entity creation inside transactions

**Pitfall alert:** `world.insert()` in a hot loop causes archetype migration (O(components) per entity). If you need to add a component to many entities, consider designing the archetype upfront (spawn with all components) or batching the structural changes.

For architecture details, see CLAUDE.md § "Deferred Mutation" and § "Archetype Migration".
