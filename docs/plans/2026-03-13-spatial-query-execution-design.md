# Spatial Query Execution Integration

**Goal:** Wire the existing `SpatialLookup` / `SpatialGather` plan nodes into
the live query execution engine so that spatial predicates actually invoke a
user-provided lookup function at execution time instead of falling back to a
full archetype scan + per-entity filter.

**Scope:** `add_spatial_index_with_lookup`, `SpatialDriver`, Phase 8 index-gather
closure, join collector integration, execution tests.

**Supersedes:** The original version of this design doc proposed adding a
`query()` method to the `SpatialIndex` trait. That approach was rejected because
it imposes query-shape policy on the index implementor, violating the
"mechanisms not policy" principle that governs `SpatialIndex` design. See
"Alternatives Considered" for the full analysis.

---

## Problem

PR #97/#98 added spatial predicates to the planner's IR, capability discovery
via `SpatialIndex::supports()`, and cost-based driver selection. The planner
correctly emits `SpatialLookup` / `SpatialGather` nodes in EXPLAIN output and
uses the index's reported cost for plan selection. However, the execution engine
does not invoke any spatial index at runtime — all spatial predicates fall
through to filter fusion on a full archetype scan.

This means the planner selects the right plan but executes the wrong one.
For a 100K-entity world with a grid index that can answer a `Within` query in
O(1) cell lookups, the engine still scans all 100K entities and applies a
per-entity distance check.

The same gap exists for `IndexLookup` / `IndexGather` (BTree/Hash), which
also fall back to filter fusion. This design addresses spatial indexes first
because they have the largest selectivity asymmetry (spatial queries
typically return <1% of entities), but the pattern applies to BTree/Hash
gather as well.

## Current State

### What exists

| Component | File | Status |
|---|---|---|
| `SpatialExpr` (dimension-agnostic `Vec<f64>`) | `index.rs` | Complete |
| `SpatialCost { estimated_rows, cpu }` | `index.rs` | Complete |
| `SpatialIndex::supports()` | `index.rs` | Complete |
| `SpatialPredicate` enum | `planner.rs` | Complete |
| `Predicate::within`, `::intersects` | `planner.rs` | Complete |
| `PlanNode::SpatialLookup` | `planner.rs` | IR only |
| `VecExecNode::SpatialGather` | `planner.rs` | IR only |
| `SpatialIndexDescriptor` | `planner.rs` | Cost discovery only |
| `QueryPlanner::add_spatial_index` | `planner.rs` | Cost discovery only |
| `find_spatial_index()` | `planner.rs` | Cost discovery only |
| Full-plan-cost driver selection | `planner.rs` | Complete |
| Filter fusion in Phase 8 | `planner.rs` | All predicates fused as post-scan filters |
| `CompiledForEach` / `CompiledForEachRaw` | `planner.rs` | Scan + filter only |
| `collect_matching_entities` for joins | `planner.rs` | Scan-based, no index path |

### What's missing

1. **User-provided lookup function** — a closure registered alongside the
   spatial index that bridges between the planner's `SpatialExpr` protocol
   and the index's concrete query API.

2. **Index-driven entity collection** — a code path in `ScanBuilder::build()`
   Phase 8 that, when the driving access is `SpatialLookup` and a lookup
   function is available, compiles a closure that calls the lookup instead
   of scanning archetypes.

3. **Spatial driver threading** — `SpatialExpr` + lookup function must flow
   from Phase 1 (predicate classification) through to Phase 8 (closure
   compilation).

4. **Validation + refinement post-index** — spatial lookup results may contain
   stale entities. The execution path must apply generational validation
   (`world.is_alive()`) and all filter closures as post-index refinement.

5. **Join integration** — when the left side of a join has a spatial driver
   with a lookup, the left-side collector should use index-driven gathering
   instead of archetype scanning.

## Design

### Principle: Mechanisms Not Policy

The `SpatialIndex` trait is deliberately minimal — `rebuild`, `update`,
`supports`. It does **not** have a `query()` method. The rationale (from
the trait's original design):

> "A grid needs `query_cell()`, a BVH needs `query_aabb()`, a k-d tree
> needs `nearest()`. Forcing one query shape onto all index types would
> either over-constrain simple structures or under-serve complex ones."

Adding `query(&SpatialExpr) -> Vec<Entity>` to the trait would impose
query-shape policy on every implementor. Instead, the user provides a
lookup closure at registration time that adapts between the planner's
`SpatialExpr` protocol and their index's concrete API.

**The index decides how to answer queries. The planner decides when to ask.
The lookup closure is the adapter between those two concerns.**

### Registration API

```rust
type SpatialLookupFn = Arc<dyn Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync>;
```

Existing method (unchanged — cost discovery only):

```rust
pub fn add_spatial_index<T: Component>(
    &mut self,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    world: &World,
)
```

New method (cost discovery + execution):

```rust
pub fn add_spatial_index_with_lookup<T: Component>(
    &mut self,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    world: &World,
    lookup: impl Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync + 'static,
)
```

`add_spatial_index` constructs a `SpatialIndexDescriptor` with `lookup_fn: None` directly (same pattern, independent implementation).

Updated descriptor:

```rust
struct SpatialIndexDescriptor {
    component_name: &'static str,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    lookup_fn: Option<SpatialLookupFn>,   // NEW
}
```

**Usage example:**

```rust
let grid = Arc::new(SpatialGrid::new());
grid.rebuild(&mut world);

let grid_for_lookup = Arc::clone(&grid);
planner.add_spatial_index_with_lookup::<Pos>(
    Arc::clone(&grid),
    &world,
    move |expr: &SpatialExpr| -> Vec<Entity> {
        match expr {
            SpatialExpr::Within { center, radius } => {
                grid_for_lookup.query_radius(center, *radius)
            }
            SpatialExpr::Intersects { min, max } => {
                grid_for_lookup.query_aabb(min, max)
            }
            _ => Vec::new(),
        }
    },
);
```

### Spatial Driver

A `SpatialDriver` threads the lookup function from Phase 1 to Phase 8:

```rust
struct SpatialDriver {
    expr: SpatialExpr,
    lookup_fn: SpatialLookupFn,
}
```

Added as `spatial_driver: Option<SpatialDriver>` on `ScanBuilder`. Set in
Phase 1 when the best-cost spatial predicate is chosen as the driving
access AND the corresponding `SpatialIndexDescriptor` has
`lookup_fn: Some(_)`. Only the best-cost spatial predicate becomes the
driver; remaining spatial predicates are demoted to filter functions. If
the descriptor has no lookup, the plan falls back to scan + filter
(existing behavior).

### Phase 8: Index-Gather Closure

Phase 8 currently always compiles an archetype-scan closure. With a spatial
driver, it compiles an index-gather closure instead:

```
Phase 8 decision:
  if spatial_driver is Some:
      compiled_for_each = index_gather(lookup_fn, expr, changed, filter_fns)
  else:
      compiled_for_each = archetype_scan(required, changed, filter_fns)
```

The index-gather closure:

```rust
move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
    let candidates = lookup_fn(&expr);
    for entity in candidates {
        if !world.is_alive(entity) {
            continue;  // stale entry — generational validation
        }
        // Changed<T> filtering: look up entity's archetype and check
        // column ticks, same as the scan path does per-archetype.
        if !changed.is_clear() {
            if let Some(loc) = world.entity_location(entity) {
                let arch = &world.archetypes.archetypes[loc.archetype_id];
                if !passes_change_filter(arch, &changed, tick) {
                    continue;
                }
            }
        }
        if filter_fns.iter().all(|f| f(world, entity)) {
            callback(entity);  // passes all refinement filters
        }
    }
}
```

**Key properties:**

1. `is_alive` gates everything else (O(1) generation check).
2. `Changed<T>` is handled explicitly via per-entity archetype lookup. The
   scan path checks `passes_change_filter` once per archetype. The
   index-gather path checks it per entity because candidates may span
   multiple archetypes. The `changed.is_clear()` fast-path skips this
   entirely when no `Changed<T>` filter is in the query.
3. All `filter_fns` run — both the spatial predicate's own closure (which
   typically calls `world.get::<T>(entity)`, implicitly validating
   component presence) and any other predicates demoted to filters. This is
   the refinement step that handles lossy indexes and also catches entities
   that have had the queried component removed since the index was rebuilt.
4. Same closure compiled for both `CompiledForEach` and `CompiledForEachRaw`.

**Bypass-path invariant audit** (per CLAUDE.md convention):

| Invariant | Scan path | Index-gather path | Status |
|---|---|---|---|
| Change detection ticks | `passes_change_filter` per archetype | `passes_change_filter` per entity via `entity_location` | Maintained |
| Query cache invalidation | N/A (planner doesn't use World cache) | Same | N/A |
| Access bitset accuracy | Component in read set | Same (`TypeId::of::<T>()` at registration) | Maintained |
| Entity lifecycle | Archetype `entities` vec (always live) | `is_alive` check (index may contain stale) | Maintained |
| Component presence | Archetype guarantees components present | `filter_fn` calls `world.get::<T>()` | Maintained (user closure) |

### Join Integration

When a join's left side has a spatial driver, the left-side entity collector
uses the index-gather pattern instead of `collect_matching_entities`:

```
Join left-side collection:
  if spatial_driver.lookup_fn is Some:
      for entity in lookup_fn(&expr):
          if !world.is_alive(entity): continue
          if !changed.is_clear(): check passes_change_filter via entity_location
          if filter_fns.all(ok):
              scratch.push(entity)
  else:
      collect_matching_entities(world, required, changed, tick, scratch)
```

Same validation pipeline as Phase 8: `is_alive` → `Changed<T>` via
archetype lookup → `filter_fns`. The right side of a join is always a scan
(separate query type). Spatial drivers only affect the left/driving side.
Join hash-build and probe logic downstream is unchanged.

### Data Flow

```
User code                          Planner                         Execution
─────────                          ───────                         ─────────

grid.rebuild(&mut world)
  └─ populates grid cells

planner.add_spatial_index_with_lookup::<Pos>(grid, &world, |expr| {
    grid.query_radius(expr.center, expr.radius)
})
  └─ stores index + lookup_fn in spatial_indexes[TypeId::of::<Pos>()]

planner.scan::<(&Pos,)>()
  .filter(Predicate::within::<Pos>([50, 50], 10, distance_check))
  .build()
  │
  ├─ Phase 1: classify predicate as Spatial
  │   └─ find_spatial_index() → Accelerated(cost)
  │   └─ descriptor has lookup_fn → set spatial_driver
  │
  ├─ Phase 3: emit PlanNode::SpatialLookup
  ├─ Phase 6: lower to VecExecNode::SpatialGather
  │
  └─ Phase 8: compile index-gather closure
      └─ compiled_for_each = |world, tick, callback| {
             for entity in lookup_fn(&expr) {
                 if world.is_alive(entity) && filter_fns.all(ok) {
                     callback(entity);
                 }
             }
         }

plan.for_each(&mut world, |entity| { ... })
  └─ calls compiled_for_each
      └─ calls lookup_fn(Within{[50,50], 10})
          └─ grid.query_radius → ~50 candidate entities
      └─ validates + filters → ~45 live matches
      └─ callbacks fired for each
```

## Alternatives Considered

### Alternative A: Add `query()` to `SpatialIndex` trait

```rust
pub trait SpatialIndex {
    fn query(&self, expr: &SpatialExpr) -> Vec<Entity> { Vec::new() }
}
```

**Rejected.** Imposes query-shape policy on the trait. The original design
rationale explicitly excluded query methods because different indexes need
different query shapes. Adding `query()` says "all spatial indexes must
answer queries through this one shape" — the same kind of assumption that
led to hardcoded 2D coordinates in `SpatialExpr`. The default empty return
is also a footgun: `supports()` returns `Some` but `query()` returns
nothing, silently producing wrong results.

### Alternative B: Callback-based execution on the trait

```rust
pub trait SpatialIndex {
    fn query_for_each(&self, expr: &SpatialExpr, out: &mut dyn FnMut(Entity)) {}
}
```

**Rejected.** Same policy problem as Alternative A — adds a query method
to the trait. Zero-alloc benefit is real but can be achieved within the
user's lookup closure (e.g., `RefCell<Vec<Entity>>` buffer reuse).

### Alternative C: Separate `SpatialQuery` trait

**Rejected.** Doubles the registration surface. Users implement two traits
on the same struct. Two Arc handles pointing at the same object. Supertrait
bound couples them anyway.

### Alternative D: `Box<dyn Iterator>` return type

```rust
type SpatialLookupFn = Arc<dyn Fn(&SpatialExpr) -> Box<dyn Iterator<Item = Entity>> + Send + Sync>;
```

**Rejected.** The iterator must borrow the index's internal state, creating
lifetime entanglement with the `Arc<dyn SpatialIndex>`. The lookup closure
is `Fn(&SpatialExpr) -> R` where R must be `'static`. An iterator borrowing
internal grid cells can't satisfy that. `Vec<Entity>` is the honest return
type for a freshly-computed result set. If profiling shows allocation
pressure, the user's lookup closure can capture a `RefCell<Vec<Entity>>`
and reuse its buffer across calls — the planner doesn't need to know.

## Semantic Review

### 1. Can this be called with the wrong World?

The `SpatialLookupFn` captures the user's index by `Arc` at registration
time. The index was rebuilt against a specific World. If the plan is
executed against a different World, the index returns entity IDs from the
wrong universe.

**Mitigation:** Same as existing `IndexDescriptor` functions — the
`QueryPlanner` is constructed from `&'w World`, and the plan borrows the
same world at execution time. No cross-world protection exists today for
BTree/Hash either. A `WorldId` check at registration time (matching the
planner's world) would close this gap for all index types.

### 2. Can Drop observe inconsistent state?

`SpatialLookupFn` is an `Arc<dyn Fn>` — its Drop is a ref-count decrement
with no world interaction. No engine state is mutated during Drop of any
planner or plan component.

**Verdict:** Safe.

### 3. Can two threads reach this through `&self`?

The `SpatialLookupFn` is `Fn` (not `FnMut`) behind `Arc` — concurrent
calls are safe. `QueryPlanResult::execute()` / `for_each()` take `&mut self`,
preventing concurrent execution of the same plan. The `Send + Sync` bound
on the lookup closure enforces thread safety at the type level.

**Verdict:** Sound.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

When a spatial predicate has a lookup and another predicate (BTree/Hash/spatial)
is demoted to a filter, the filter still runs. Filter fusion collects closures
from ALL predicates — no predicate is silently dropped.

**Verdict:** Predicate semantics preserved.

### 5. What happens if this is abandoned halfway through?

`lookup_fn` returns a `Vec<Entity>`. If `for_each` panics mid-iteration,
the `Vec` is dropped. `last_read_tick` is only advanced after successful
completion. Dropping `QueryPlanResult` releases the captured `Arc`s. The
spatial index remains valid (user-owned).

**Verdict:** Safe.

### 6. Can a type bound be violated by a legal generic instantiation?

`add_spatial_index_with_lookup::<T>` requires `T: Component`. The lookup
closure is `impl Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync + 'static` —
fully concrete. `SpatialExpr` is `Clone + Debug`, no generics.

**Verdict:** No violation possible.

### 7. Does the API surface permit operations not covered by Access?

Spatial predicates produce `Predicate` values with
`component_type: TypeId::of::<T>()`. The `Access` bitset includes this
component in its read set. The lookup closure reads index-internal data,
not World data — same as BTree/Hash today. Access bitset is accurate.

**Verdict:** Sound.

## Implementation Plan

### Step 1: Add `SpatialLookupFn` and `add_spatial_index_with_lookup`

**File:** `crates/minkowski/src/planner.rs`

- Define `type SpatialLookupFn`.
- Add `lookup_fn: Option<SpatialLookupFn>` to `SpatialIndexDescriptor`.
- Implement `add_spatial_index_with_lookup::<T>()`.
- Refactor `add_spatial_index` to delegate with `lookup_fn: None`.
- Re-export `SpatialLookupFn` from `lib.rs`.

### Step 2: Thread `SpatialDriver` through `ScanBuilder`

**File:** `crates/minkowski/src/planner.rs`

- Define `SpatialDriver { expr: SpatialExpr, lookup_fn: SpatialLookupFn }`.
- Add `spatial_driver: Option<SpatialDriver>` to `ScanBuilder`.
- In `build()` Phase 1, when a spatial predicate is chosen as driver and the
  descriptor has `lookup_fn: Some(_)`, populate the spatial driver.
- Pass through to Phase 8.

### Step 3: Compile index-gather closure in Phase 8

**File:** `crates/minkowski/src/planner.rs`

- In Phase 8, check `spatial_driver.is_some()`.
- If present, compile `CompiledForEach` and `CompiledForEachRaw` using the
  index-gather pattern: lookup → `is_alive` → `Changed<T>` via
  `entity_location` + `passes_change_filter` → `filter_fns`.
- The `changed` bitset must be captured into the index-gather closure for
  `Changed<T>` filtering. Use `changed.is_clear()` fast-path to skip the
  archetype lookup when no change filter is active.
- Existing archetype-scan path unchanged (else branch).

### Step 4: Wire spatial driver into join collectors

**File:** `crates/minkowski/src/planner.rs`

- When the left side of a join has a spatial driver, build the
  `left_collector` using the index-gather pattern instead of
  `collect_matching_entities`.

### Step 5: Add execution tests

**File:** `crates/minkowski/src/planner.rs` (test module)

- `spatial_index_for_each_uses_lookup` — verify that a plan with a
  registered lookup calls it (not just filter fusion). Use an
  `AtomicUsize` counter in the closure to confirm invocation.
- `spatial_index_execute_returns_correct_entities` — spawn entities at
  known positions, run `execute()` with a `Within` predicate, verify only
  nearby entities are returned.
- `spatial_index_stale_entities_filtered` — despawn an entity after index
  rebuild, verify it does not appear in results.
- `spatial_index_for_each_raw_works` — verify read-only execution path.
- `spatial_index_join_uses_lookup` — verify index-driven gathering in a
  join's left collector.
- `spatial_index_without_lookup_falls_back` — verify that
  `add_spatial_index` (no lookup) still works for cost discovery, and
  execution falls back to scan + filter.
- `spatial_index_with_changed_filter` — verify that `Changed<T>` filtering
  works correctly in the index-gather path (mutate some entities, verify
  only changed entities pass through).

### Step 6: Update planner example

**File:** `examples/examples/planner.rs`

- Add a section demonstrating spatial index registration with lookup,
  plan building, and execution with `for_each()`.
- Show EXPLAIN output with `SpatialGather` node.

### Step 7: Add `TablePlanner` delegation

**File:** `crates/minkowski/src/planner.rs`

- Add `add_spatial_index_with_lookup::<C>()` to `TablePlanner` that
  delegates to the inner `QueryPlanner`, with the same
  `HasBTreeIndex`-style compile-time enforcement pattern if appropriate
  (or unconstrained if spatial indexes are orthogonal to table schemas).

### Step 8: Update CLAUDE.md

**File:** `CLAUDE.md`

- Document `add_spatial_index_with_lookup` in the Query Planner section.
- Note the "mechanisms not policy" principle for spatial execution.
- Update the example command list if the planner example changes.

## Known Follow-Up Work

- **Cross-world `WorldId` validation.** No index registration path
  (BTree, Hash, or Spatial) validates that the index was rebuilt against
  the same World the planner was constructed from. A `WorldId` check at
  registration time would close this gap for all index types.

- **Buffer reuse for lookup results.** `SpatialLookupFn` returns
  `Vec<Entity>`, allocating per call. For high-frequency queries on large
  result sets, the user can mitigate by capturing a `RefCell<Vec<Entity>>`
  in the closure for buffer reuse. A future `query_into`-style API
  (`Arc<dyn Fn(&SpatialExpr, &mut Vec<Entity>)>`) would make this
  ergonomic but is not needed for v1.
