---
description: Help with Minkowski ECS query design — tuple composition, filters, iteration strategies, caching
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user design queries for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing query patterns:

- `world.query::<` calls to see current query compositions
- `for_each` / `for_each_chunk` / `par_for_each` usage patterns
- `Changed<T>` filter usage
- `Option<&T>` usage for optional components
- `query_table` / `query_table_mut` calls (Table-based queries)
- `query_raw` calls (transaction read path — flag if used outside transactions)
- Queries inside loops that could benefit from caching awareness

## Step 2: Recommend

**Strong defaults:**
- `&T` for reads, `&mut T` for writes, `Entity` to get the entity ID alongside component data.
- `Option<&T>` for components that may or may not be present — does not filter the archetype, so you get all entities that match the required components, with `None` for entities missing the optional one.
- `Changed<T>` to skip archetypes whose column has not been mutably accessed since the last time this query ran. Note: first call always matches everything. "Changed" means "since this query last observed this column", not "since last frame".
- `for_each_chunk` for numeric tight loops — yields typed `&[T]`/`&mut [T]` slices per archetype that LLVM can auto-vectorize.
- `par_for_each` for CPU-heavy per-entity work — uses rayon under the hood.
- `query_table`/`query_table_mut` when using `#[derive(Table)]` — skips archetype matching entirely, gives named field access via `Ref<'w>`/`Mut<'w>`.
- Query cache is automatic — `world.query()` maintains a `HashMap<TypeId, QueryCacheEntry>`. Repeat calls with no new archetypes skip the archetype scan entirely.

**Ask if unclear:**
- "How often does this data change?" — If rarely, `Changed<T>` can skip entire archetypes. But understand: it tracks mutable access, not actual value change (pessimistic marking).
- "Do you need the entity ID alongside component data?" — Include `Entity` in the query tuple.
- "Is this a hot numeric loop?" — Use `for_each_chunk` for SIMD. Ensure components are `#[repr(align(16))]` for best vectorization.
- "Is per-entity work expensive (>1us)?" — Consider `par_for_each` for rayon parallelism.

## Step 3: Implement

Help write query code. Point to relevant examples:

- **Basic iteration**: `world.query::<(&Pos, &Vel)>().for_each(|(pos, vel)| { ... })`
- **Mutable queries**: `world.query::<(&mut Pos, &Vel)>().for_each(|(pos, vel)| { pos.x += vel.dx; })`
- **SIMD-friendly chunks**: See `examples/examples/boids.rs` — `for_each_chunk` with aligned components
- **Changed filter**: `world.query::<(&mut Render, Changed<Pos>)>().for_each(|(render, pos)| { ... })`
- **Optional components**: `world.query::<(&Pos, Option<&DebugLabel>)>().for_each(|(pos, label)| { ... })`
- **Table queries**: See `examples/examples/life.rs` — `query_table`/`query_table_mut` for named field access
- **Parallel iteration**: `world.query::<(&Pos,)>().par_for_each(|(pos,)| { /* heavy work */ })`

For architecture details on query caching and column alignment, see CLAUDE.md § "Query Caching" and § "Column Alignment & Vectorization".
