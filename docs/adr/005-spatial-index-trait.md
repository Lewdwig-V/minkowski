# ADR-005: Spatial Index Trait

**Status:** Accepted
**Date:** 2026-03-01

## Context

Spatial queries (nearest neighbor, range, ray cast) are essential for simulations but index data structures vary wildly — uniform grids, quadtrees, BVH, k-d trees each have different query shapes and update strategies. The engine needs a common lifecycle contract without dictating implementation details.

## Decision

`SpatialIndex` is a two-method trait: `rebuild` (required, full reconstruction) and `update` (optional, defaults to rebuild). Indexes are fully external to World — no generic query method, no component type parameters, no stored World reference, no World registration. Despawned entities are handled via generational validation at query time.

**Key insight: external composition over internal integration — indexes use existing query primitives and own their data independently.**

## Alternatives Considered

- Built-in spatial grid on World — too specific, cannot serve quadtrees or BVH
- Generic query method on the trait (`query(&self, shape) -> Iterator`) — cannot express the diversity of query shapes (cell, AABB, radius, ray)
- World-integrated indexes (`world.register_index()`) — grows World's API surface with every new index pattern

## Consequences

- Structurally different algorithms (uniform grid in boids, quadtree in nbody) implement the same trait without friction
- Stale entities after despawn are caught by `world.is_alive()` at query time, cleaned up on next `rebuild`
- Users define query methods per concrete type — full type safety without trait-level generics
- Indexes compose from `world.query()` and `Changed<T>` — no special hooks needed
- World's API stays focused on entities and components; spatial concerns are entirely external
