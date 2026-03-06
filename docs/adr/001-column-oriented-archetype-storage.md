# ADR-001: Column-Oriented Archetype Storage

**Status:** Accepted
**Date:** 2026-02-28

## Context

Minkowski needs a storage model that combines the runtime flexibility of an ECS (arbitrary component composition) with the performance characteristics of a columnar database (cache-friendly iteration, SIMD vectorization). The engine must support millions of entities with heterogeneous component sets while maintaining predictable iteration performance.

## Decision

Store components in type-erased `BlobVec` columns grouped into archetypes, where each unique set of component types gets its own archetype. Columns are allocated with 64-byte alignment (cache line). Entities are generational IDs (u32 index + u32 generation packed into u64) with O(1) lookup via `entity_locations: Vec<Option<EntityLocation>>`. Optional sparse components (`HashMap<Entity, T>`) are available via `register_sparse` for data present on fewer than 5% of entities.

**Key insight: archetypes are an optimization, not a data model — the column layout gives database performance while the archetype system provides runtime flexibility.**

## Alternatives Considered

- Row-oriented storage (entity as struct-of-all-components) — poor cache utilization during iteration over subsets of components
- Pure columnar without archetypes (one global array per component type) — loses entity identity grouping, requires indirection for multi-component queries
- Fixed schemas only (table-per-schema, no runtime composition) — loses the ability to add/remove components at runtime

## Consequences

- Cache-friendly sequential iteration over component columns enables LLVM auto-vectorization via `for_each_chunk`
- `FixedBitSet` on each archetype enables O(1) query matching via bitwise subset checks
- O(1) entity lookup through the location indirection table
- Structural changes (insert/remove component) require archetype migration — copying all columns from old archetype to new
- Generational entity IDs prevent use-after-despawn without a notification system
- 64-byte column alignment + `target-cpu=native` enables platform-specific SIMD instructions
