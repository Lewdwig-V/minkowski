---
description: Help with Minkowski ECS data modeling — components, bundles, Table schemas, sparse storage
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user design their data model for a Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing data modeling patterns:

- Component struct definitions: `grep` for `struct` definitions near `spawn()` or `Bundle` usage
- Existing `#[derive(Table)]` usage
- `world.spawn()` calls to see current bundle compositions
- `register_sparse()` calls for sparse component usage
- Nested structs used as component fields (anti-pattern to flag)
- Large component types that could be split

## Step 2: Recommend

**Strong defaults:**
- One concept = one component. Do not nest structs as component fields — flatten into separate components.
- Components should be `'static + Send + Sync`. Prefer `Copy` types when possible (enables zero-copy query).
- Use `#[derive(Table)]` when you have a fixed schema that is always queried together (e.g., physics: Pos+Vel+Accel). Table skips archetype matching entirely via `query_table`/`query_table_mut`.
- Use sparse components (`register_sparse`) for rare optional data that only a few entities have (e.g., DebugLabel, PlayerControlled). Stored as `HashMap<Entity, T>`, not in archetypes.
- Bundle tuples for spawn: `world.spawn((Pos::default(), Vel::default(), Health(100)))`. Tuples 1-12 implement Bundle.
- Avoid mega-components (structs with 10+ fields). Split by access pattern — components queried together should be separate components in the same archetype, not fields of one big struct.

**Ask if unclear:**
- "Is this data queried independently or always together?" — If always together with a fixed schema, consider `#[derive(Table)]`. If queried in different combinations, keep as separate components.
- "How many entities will have this component?" — If most entities: archetype component. If <5% of entities: sparse component.
- "Will this component be added/removed at runtime?" — Frequent `insert()`/`remove()` causes archetype migrations (O(components) per entity). Design to avoid remove-less patterns if possible.

## Step 3: Implement

Help write component definitions and spawn patterns. Point to relevant examples:

- **Table derive + fixed schemas**: See `examples/examples/life.rs` — `Cell` table with `CellState` + `NeighborCount`
- **Physics components**: See `examples/examples/boids.rs` — `Position(Vec2)`, `Velocity(Vec2)`, `Acceleration(Vec2)` as separate components
- **Bundle spawning**: See `examples/examples/battle.rs` — `world.spawn((Health(100), Team(0), Damage(10), Healing(5)))`
- **Reducer-aware modeling**: Components determine Access bitsets. One-concept-per-component gives the finest granularity for conflict detection.

For architecture details on how archetypes and storage work, see CLAUDE.md § "Storage Model" and § "Archetype Migration".
