# ADR-011: B-Tree and Hash Column Indexes

**Status:** Accepted
**Date:** 2026-03-06

## Context

All queries are full-archetype scans. Database-style workloads (find entities with score > 100, look up entity by name) need secondary indexes on component values for sub-linear access.

## Decision

Two independent index types — `BTreeIndex<T: Ord>` for range queries and `HashIndex<T: Hash + Eq>` for exact lookups. Both are external to World (same composition pattern as `SpatialIndex`), wrap standard collections with a reverse map for incremental updates, and implement the `SpatialIndex` trait for lifecycle compatibility.

**Key insight: index on whole components, not fields — if you need to index a field, make it a component. This follows the ECS convention and keeps the index API simple.**

## Alternatives Considered

- Index on extracted fields via accessor function — more flexible but requires `dyn Fn`, deferred to future enhancement
- Per-archetype B-trees merged at query time — complex merge-join for minimal benefit
- World-registered indexes with auto-update hooks — violates external composition principle
- Shared `ColumnIndex` trait — query shapes too different (range vs exact)

## Consequences

- O(log n) range queries and O(1) exact lookups on component values
- Incremental updates via per-index `ChangeTick` and `World::query_changed_since` avoid full rescans (does not use the shared `Changed<T>` query cache)
- Reverse map costs one `HashMap` entry per indexed entity
- Stale entries from despawns and component removals handled lazily — `get_valid`/`range_valid` filter at query time, `rebuild` reclaims memory
- Components must implement `Ord` (B-tree) or `Hash + Eq` (hash) plus `Clone`
- Not persisted — rebuilt from world state after crash recovery
- Concurrency is user-managed — indexes are external data structures like `SpatialIndex`
