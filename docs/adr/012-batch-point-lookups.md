# ADR-012: Batch Point-Lookups Over Query Planner

**Status:** Accepted
**Date:** 2026-03-06

## Context

With BTreeIndex and HashIndex in the engine, the next question was how to
efficiently fetch components for entities returned by index lookups. The
roadmap listed "Query planning (Volcano model)" — composable operator trees
that would optimise query execution across indexes.

## Decision

Provide batch point-lookups (`World::get_batch`, `World::get_batch_mut`)
instead of a query planner. The user composes index queries with batch
fetches manually.

### Why not a query planner?

- **ECS queries don't join.** The Volcano model solves joining rows across
  tables with unknown cardinalities. In an ECS, an entity either has the
  components or it doesn't — the "join" is the archetype bitset match,
  already O(1) per archetype.
- **The user already knows the access pattern.** ECS systems are compiled
  functions. The developer chooses between `query` and `index.range` at
  write time. A runtime planner adds overhead without new information.
- **Archetype matching is already fast.** Bitset subset check is
  O(archetypes), not O(entities).

### What batch lookups provide

- **Amortised resolution.** ComponentId lookup, sparse check, and column
  index lookup resolve once per type/archetype, not once per entity.
- **Cache locality.** Grouping by archetype means column data access is
  sequential within each group — the prefetcher stays happy.
- **Composability.** Index narrowing followed by batch fetch is a two-step
  pipeline. No operator trees, no cost model, no planner overhead.

### Duplicate detection in get_batch_mut

Aliased `&mut T` is undefined behaviour. `get_batch_mut` detects duplicates
unconditionally via sort-based comparison: after grouping by archetype, each
group is sorted by row and adjacent pairs are checked. Same row in the same
archetype means the same entity (each row holds exactly one entity). Zero
extra allocation — the sort operates on the grouping vecs already built for
cache locality. The sparse path sorts alive entity indices instead.

## Alternatives

- **Volcano-model query planner** — rejected. Solves a problem ECS queries
  don't have (joining across tables with unknown cardinalities).
- **`query_entities` with pre-filtered entity set** — deferred. Would allow
  multi-component fetch in one call, but `get_batch` per component is
  sufficient until profiling shows otherwise.
- **No new API** — rejected. Per-entity `get()` has poor cache locality for
  large candidate sets from index lookups.

## Consequences

- Two new methods on World: `get_batch<T>(&self, &[Entity])` and
  `get_batch_mut<T>(&mut self, &[Entity])`.
- `get_batch_mut` panics on duplicate entities (unconditional assert).
- Indexes remain external — no coupling between query engine and index types.
- The user is the planner. The type system is the cost model.
