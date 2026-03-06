# ADR-006: Query Conflict Detection

**Status:** Accepted
**Date:** 2026-03-01

## Context

Framework-level schedulers need to detect data races between systems at registration time to determine which systems can run in parallel. This requires component-level read/write metadata extractable from query types, but the scheduling policy itself (dependency graphs, topological sort, parallel execution) varies by framework.

## Decision

`Access` struct with per-component read/write `FixedBitSet`s plus a `despawns: bool` flag. `Access::of::<Q>(world)` extracts metadata from any `WorldQuery` type. `conflicts_with()` applies the read-write lock rule via two bitwise ANDs over the component bitsets, plus a despawn-vs-any-access blanket conflict. Minkowski provides the metadata; scheduling policy is the framework's responsibility.

**Key insight: Minkowski is a storage engine, not a framework — provide the conflict detection primitives, not the scheduler.**

## Alternatives Considered

- Built-in system scheduler — framework concern, different users need different scheduling strategies
- Runtime-only conflict detection (check at execution time) — too late, races already happened
- Per-entity access tracking — too fine-grained, O(n) overhead for systems touching many entities

## Consequences

- O(1) conflict detection via bitset intersection — two AND operations plus one bool check
- `Access` composes with the reducer system for type-level conflict proofs at registration time
- `Option<&T>` accesses a component without requiring it — `accessed_ids` differs from `required_ids`
- The `despawns` flag creates blanket conflicts with any other access, preventing concurrent despawn + read
- Framework authors get precise metadata; Minkowski avoids opinionated scheduling decisions
