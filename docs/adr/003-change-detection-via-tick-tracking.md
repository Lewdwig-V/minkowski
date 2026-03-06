# ADR-003: Change Detection via Tick Tracking

**Status:** Accepted
**Date:** 2026-02-28

## Context

Queries that process only recently changed data (e.g., syncing positions to a spatial index) need a way to skip unchanged entities. Per-entity dirty flags would be too fine-grained for column-oriented storage. The mechanism must be automatic — requiring users to manually advance ticks or mark changes is error-prone.

## Decision

Each `BlobVec` column stores a `changed_tick: Tick` (monotonic u64) that auto-advances on every mutable access. `Changed<T>` is a `WorldQuery` filter that skips entire archetypes whose column tick is older than the query's `last_read_tick`. Tick advancement is fully automatic — no user-facing `World::tick()` method. Marking is pessimistic (on mutable access, not on actual value change) but zero-cost at the write site.

**Key insight: pessimistic marking at the column level — zero cost at the write site, archetype-granularity skip at the read site.**

## Alternatives Considered

- Per-entity dirty flags — too fine-grained, O(n) scan even for unchanged data
- Frame-based tick management (`world.tick()`) — requires user discipline, easy to forget
- Observer/callback pattern — allocation per change event, lifetime complexity

## Consequences

- Every mutable access path (spawn, get_mut, insert, remove, query `&mut T`, query_table_mut, changeset apply) must mark the column changed — enforced by convention and audit rule
- `Changed<T>` means "since the last time this query observed this column," not per-frame — no concept of simulation time
- Three distinct meanings of "component set" in WorldQuery: `required_ids` (archetype matching), `accessed_ids` (conflict detection), `mutable_ids` (change detection)
- `Tick` is `pub(crate)` — implementation detail not exposed to users
- False positives possible (mutable access without actual value change) but no false negatives
