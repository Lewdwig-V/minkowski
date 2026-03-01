# Access Metadata for Query Conflict Detection

**Date:** 2026-03-01
**Status:** Approved

## Context

Minkowski is a storage engine, not a framework. System scheduling — conflict detection, dependency graphs, parallel execution — belongs to framework authors building on top of minkowski. But those authors need one primitive minkowski is uniquely positioned to provide: component-level access metadata for queries.

The infrastructure already exists. `WorldQuery::required_ids()` returns a `FixedBitSet` of components a query reads; `WorldQuery::mutable_ids()` returns what it writes. These are `pub(crate)` today. The task is to wrap them in a clean public API.

## Design

### `Access` struct

A thin wrapper around two `FixedBitSet`s extracted from `WorldQuery` trait methods.

```rust
/// Component-level access metadata for a query type.
/// Used by external schedulers to detect conflicts between systems.
pub struct Access {
    reads: FixedBitSet,   // components read but not written
    writes: FixedBitSet,  // components written (mutably accessed)
}
```

**Construction:**

```rust
impl Access {
    /// Build access metadata for a query type.
    /// Registers any unregistered component types as a side effect
    /// (same as world.query::<Q>()).
    pub fn of<Q: WorldQuery + 'static>(world: &mut World) -> Self;
}
```

Takes `&mut World` because `WorldQuery::required_ids()` and `mutable_ids()` take `&ComponentRegistry`, and component types may need registration (consistent with `world.query()` which also takes `&mut self`).

**Derivation:**
- `writes` = `Q::mutable_ids(&registry)`
- `reads` = `Q::required_ids(&registry)` minus `writes` (components read but not written)

**Accessors:**

```rust
impl Access {
    pub fn reads(&self) -> &FixedBitSet;
    pub fn writes(&self) -> &FixedBitSet;
}
```

**Conflict detection:**

```rust
impl Access {
    /// True if these two accesses cannot safely run concurrently.
    /// Conflict rule: two accesses conflict iff either writes to a
    /// component the other reads or writes.
    pub fn conflicts_with(&self, other: &Access) -> bool;
}
```

Implementation: standard read-write lock rule per component.
- `self.writes` intersects `other.reads | other.writes`, OR
- `other.writes` intersects `self.reads | self.writes`

Two bitwise ANDs + two `any()` checks. O(N/64) where N = total component count. For typical ECS (< 256 components): 4 cache lines.

### File location

`crates/minkowski/src/access.rs`, exported as `pub use access::Access` from `lib.rs`.

### What it doesn't do

- No schedule, executor, or topological sort
- No structural mutation tracking (spawn/despawn/insert/remove are inherently exclusive — framework's problem)
- No `world.get::<T>()` tracking — only `WorldQuery`-level access
- No `AccessSet` or batch conflict analysis
- No split World or concurrent `&mut World`

## Example: `scheduler.rs`

A minimal conflict analysis example showing how a framework author would use `Access`. Not a real framework — just enough to demonstrate the API.

**Six systems in three batches of two, all parallelizable within their batch:**

| System | Reads | Writes | Purpose |
|---|---|---|---|
| `movement` | `Vel` | `Pos` | Write/read split |
| `gravity` | — | `Vel` | Write-only |
| `health_regen` | — | `Health` | Disjoint from movement |
| `apply_damage` | — | `Health` | Write/write conflict with health_regen |
| `log_positions` | `Pos` | — | Read/write conflict with movement |
| `log_health` | `Health` | — | Read-only, conflicts with health writers |

**Conflict cases covered:**
- **Write/write conflict:** `health_regen` vs `apply_damage` (both write `Health`)
- **Read/write conflict:** `log_positions` vs `movement` (read `Pos` vs write `Pos`)
- **Disjoint writes parallelize:** `movement` + `health_regen` (different component sets)
- **Read-only batching:** `log_positions` + `log_health` (no writes, compatible with non-overlapping writers)

The example prints a full conflict matrix, groups systems into parallelizable batches using greedy coloring, then runs each batch sequentially. Parallel execution within a batch is explicitly left as the framework's responsibility.

## Dependencies

No new dependencies. Uses `FixedBitSet` (already a dependency) and `WorldQuery` trait methods (already implemented).
