# Spatial Query Execution Integration

**Goal:** Wire the existing `SpatialLookup` / `SpatialGather` plan nodes into
the live query execution engine so that spatial predicates actually invoke the
registered spatial index at execution time instead of falling back to a full
archetype scan + per-entity filter.

**Scope:** `Predicate::within`, `Predicate::intersects`, capability discovery
via `SpatialIndex::supports`, and index-driven entity gathering during
`execute()` / `for_each()` / `for_each_raw()`.

---

## Problem

PR #97 added spatial predicates to the planner's IR and capability discovery
to the `SpatialIndex` trait. The planner correctly emits `SpatialLookup` /
`SpatialGather` nodes in EXPLAIN output and uses the index's reported cost for
plan selection. However, the execution engine does not invoke the spatial
index at runtime — all spatial predicates fall through to filter fusion on a
full archetype scan.

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

| Component | File | Lines | Status |
|---|---|---|---|
| `SpatialExpr`, `SpatialCost` | `index.rs` | 12–62 | Complete |
| `SpatialIndex::supports()` | `index.rs` | 118–127 | Complete |
| `SpatialPredicate` enum | `planner.rs` | 198–270 | Complete |
| `Predicate::within`, `::intersects` | `planner.rs` | 421–478 | Complete |
| `PlanNode::SpatialLookup` | `planner.rs` | 563–569 | IR only |
| `VecExecNode::SpatialGather` | `planner.rs` | 2315–2321 | IR only |
| `SpatialIndexDescriptor` | `planner.rs` | 1846–1850 | Stores `Arc<dyn SpatialIndex>` |
| `QueryPlanner::add_spatial_index` | `planner.rs` | 1994–2014 | Registration only |
| `find_spatial_index()` | `planner.rs` | 2164–2173 | Cost discovery only |
| `IndexDescriptor.{eq,range}_lookup_fn` | `planner.rs` | 280–294 | Reserved, not called |
| Filter fusion in Phase 8 | `planner.rs` | 1744–1792 | All predicates fused as post-scan filters |
| `CompiledForEach` / `CompiledForEachRaw` | `planner.rs` | 1276–1281 | Scan + filter only |
| `EntityCollector` for joins | `planner.rs` | 1298 | Scan-based, no index path |

### What's missing

1. **`SpatialIndex::query()`** — a trait method that actually returns entities
   for a given `SpatialExpr`. Currently the trait has no query method by
   design ("a grid needs `query_cell()`, a BVH needs `query_aabb()`"). We
   need a type-erased query path for the planner to call.

2. **Index-driven entity collection** — a code path in `ScanBuilder::build()`
   Phase 8 that, when the driving access is `SpatialLookup`, compiles a
   closure that calls the spatial index instead of scanning archetypes.

3. **Spatial lookup function on `SpatialIndexDescriptor`** — analogous to
   `IndexDescriptor.eq_lookup_fn`, a type-erased function captured at
   registration time that invokes the concrete index's query method.

4. **Validation filter post-index** — spatial index results may contain stale
   entities (despawned or component-removed). The execution path must apply
   generational validation (`world.is_alive()`) and the predicate's filter
   closure as a post-index refinement step.

5. **Join integration** — when the driving access of a join's left or right
   collector is spatial, the collector should use index-driven gathering
   instead of archetype scanning.

## Proposed Design

### API Surface

#### New method on `SpatialIndex` trait

```rust
pub trait SpatialIndex {
    fn rebuild(&mut self, world: &mut World);
    fn update(&mut self, world: &mut World) { self.rebuild(world); }
    fn supports(&self, expr: &SpatialExpr) -> Option<SpatialCost>;

    /// Execute a spatial query, returning matching entity IDs.
    ///
    /// Results may include stale entities (despawned or component-removed).
    /// The caller is responsible for generational validation.
    ///
    /// Default returns empty — override alongside `supports()`.
    fn query(&self, _expr: &SpatialExpr) -> Vec<Entity> {
        Vec::new()
    }
}
```

**Rationale:** The existing design avoids query methods on the trait because
different indexes need different query shapes. However, the planner needs a
single type-erased entry point. `query()` is the minimal addition: it takes
the same `SpatialExpr` that `supports()` already accepts, returning a `Vec`
of candidate entities. The `Vec` return (not `&[Entity]`) avoids lifetime
coupling between the index and the caller. Implementors who advertise
`supports() -> Some(_)` are expected to override `query()` — the default
empty return acts as a safety net, not a correct implementation.

#### New lookup function type

```rust
/// Type-erased spatial lookup: takes a SpatialExpr, returns candidate entities.
type SpatialLookupFn = Arc<dyn Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync>;
```

Captured at `add_spatial_index()` registration time when the concrete
`Arc<dyn SpatialIndex>` is available.

#### Updated `SpatialIndexDescriptor`

```rust
struct SpatialIndexDescriptor {
    component_name: &'static str,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    lookup_fn: SpatialLookupFn,   // NEW: execution-time query
}
```

The `lookup_fn` is created in `add_spatial_index()`:

```rust
let idx = Arc::clone(&index);
let lookup_fn: SpatialLookupFn = Arc::new(move |expr: &SpatialExpr| {
    idx.query(expr)
});
```

### Internal Architecture

#### Phase 8 bifurcation: scan-driven vs index-driven

Currently Phase 8 always compiles a scan closure. With spatial execution, the
builder checks whether the driving plan node is a `SpatialLookup`:

```
Phase 8 decision:
  if driving_node == SpatialLookup:
      compile_for_each = spatial_index_gather + filter_fns
  else:
      compile_for_each = archetype_scan + filter_fns   (existing path)
```

The spatial index gather closure:

```rust
let lookup = spatial_lookup_fn.clone();   // Arc clone
let expr = spatial_expr.clone();          // captured from predicate
let filter_fns = all_filter_fns.clone();  // remaining predicates

Box::new(move |world: &World, _tick: Tick, callback: &mut dyn FnMut(Entity)| {
    let candidates = lookup(&expr);
    for entity in candidates {
        if !world.is_alive(entity) {
            continue;  // stale entry
        }
        if filter_fns.iter().all(|f| f(world, entity)) {
            callback(entity);
        }
    }
})
```

#### Information flow through build()

The `SpatialExpr` and `SpatialLookupFn` must reach Phase 8 from Phase 1
(predicate classification). Three new fields on `ScanBuilder`:

```rust
pub struct ScanBuilder<'w> {
    // ... existing fields ...
    spatial_driver: Option<SpatialDriver>,  // NEW
}

struct SpatialDriver {
    expr: SpatialExpr,
    lookup_fn: SpatialLookupFn,
}
```

Set in `build()` Phase 1 when the best spatial index is chosen as driver.
Consumed in Phase 8 to compile the index-gather closure.

### Data Flow

```
User code                          Planner                         Execution
─────────                          ───────                         ─────────

grid.rebuild(&mut world)
  └─ populates grid cells

planner.add_spatial_index::<Pos>(grid, &world)
  └─ captures lookup_fn = Arc::new(move |expr| grid.query(expr))
  └─ stores in spatial_indexes[TypeId::of::<Pos>()]

planner.scan::<(&Pos,)>()
  .filter(Predicate::within::<Pos>(50, 50, 10, distance_check))
  .build()
  │
  ├─ Phase 1: classify predicate as Spatial
  │   └─ find_spatial_index() → Some(cost)
  │   └─ set spatial_driver = Some(SpatialDriver { expr, lookup_fn })
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
      └─ calls grid.query(Within{50,50,10})
          └─ returns ~50 candidate entities
      └─ validates + filters → ~45 live matches
      └─ callbacks fired for each
```

## Alternatives Considered

### Alternative A: Virtual query method returning `Arc<[Entity]>`

Instead of `Vec<Entity>`, return `Arc<[Entity]>` from `query()` to enable
zero-copy sharing with the scratch buffer.

**Pros:** Avoids allocation on every call if the index can cache results.
**Cons:** Forces indexes to allocate an `Arc` even when the caller
immediately iterates and discards. Most spatial indexes (grids, quadtrees)
build results on-the-fly from cell iteration — caching would require
interior mutability (`Mutex` or `RefCell`) inside the index, conflicting
with the trait's `&self` receiver. `Vec` is the honest return type for a
freshly-computed result set.

**Decision:** Use `Vec<Entity>`. If profiling shows allocation pressure, a
future `query_into(&self, expr, &mut Vec<Entity>)` method can amortize by
reusing caller-owned buffers.

### Alternative B: Callback-based query (`query_for_each`)

```rust
fn query_for_each(&self, expr: &SpatialExpr, callback: &mut dyn FnMut(Entity));
```

Avoids the intermediate `Vec` entirely — the index calls the callback
directly for each candidate.

**Pros:** True zero-alloc. Natural fit for the `CompiledForEach` closure
chain.
**Cons:** The callback is a `dyn FnMut` vtable call per entity — same cost
as the filter chain. More importantly, it prevents the caller from sorting
candidates by archetype for sequential access (the `IndexGather`
optimization). It also makes the trait harder to implement: every index
must accept a callback instead of returning a collection.

**Decision:** Start with `Vec<Entity>` for simplicity. If the hot path
needs zero-alloc iteration, add `query_for_each` as an optional trait
method later (default delegates to `query()` + iterate).

### Alternative C: Do not add `query()` to `SpatialIndex` — use a separate `SpatialQuery` trait

```rust
pub trait SpatialQuery {
    fn query(&self, expr: &SpatialExpr) -> Vec<Entity>;
}
```

Registered separately from `SpatialIndex`.

**Pros:** Keeps `SpatialIndex` minimal (rebuild/update only). Separates
lifecycle (rebuild) from query (lookup).
**Cons:** Doubles the registration surface (`add_spatial_index` +
`add_spatial_query`). Users must implement two traits on the same struct.
The planner already stores `Arc<dyn SpatialIndex>` — splitting into two
traits means two Arc handles pointing at the same object, or a supertrait
bound that couples them anyway.

**Decision:** Add `query()` to `SpatialIndex` with a default empty
implementation. One trait, one registration point, one Arc.

## Semantic Review

### 1. Can this be called with the wrong World?

The `SpatialLookupFn` captures an `Arc<dyn SpatialIndex>` at registration
time. The index was rebuilt against a specific World. If the plan is
executed against a different World, the index returns entity IDs from the
wrong universe.

**Mitigation:** Same as existing `IndexDescriptor` functions — the
`QueryPlanner` is constructed from `&'w World`, and the plan borrows the
same world at execution time. No cross-world protection exists today for
BTree/Hash either. A `WorldId` check at `add_spatial_index` time (matching
the planner's world) would close this gap for all index types.

### 2. Can Drop observe inconsistent state?

`SpatialLookupFn` is an `Arc<dyn Fn>` — its Drop is a ref-count decrement
with no world interaction. The `SpatialIndexDescriptor` stores an
`Arc<dyn SpatialIndex>` whose Drop is also a ref-count decrement. No
engine state is mutated during Drop of any planner or plan component.

**Verdict:** Safe. No inconsistent state observable.

### 3. Can two threads reach this through `&self`?

`SpatialIndex::query()` takes `&self`. If two threads share an
`Arc<dyn SpatialIndex>` and call `query()` concurrently, the index must be
thread-safe. The `Send + Sync` bound on `Arc<dyn SpatialIndex + Send + Sync>`
enforces this at the type level.

`QueryPlanResult::execute()` / `for_each()` take `&mut self`, preventing
concurrent execution of the same plan. The `SpatialLookupFn` inside the
plan is behind `Arc` (shared), but `Fn` (not `FnMut`) — concurrent calls
are safe.

**Verdict:** Sound. `&self` + `Send + Sync` on the index, `&mut self` on
plan execution.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

When both a spatial predicate and a BTree predicate target the same
component, the planner picks the cheaper driver and demotes the other to
a filter. The filter still runs — no predicate is silently dropped.

When multiple spatial predicates exist, the cheapest becomes the driver;
others become post-index filters. The filter functions are collected from
ALL predicates (index-driven and filter-only), so the full conjunction is
always enforced.

**Verdict:** Predicate semantics preserved. Filter fusion is additive
(all predicates applied), never subtractive.

### 5. What happens if this is abandoned halfway through?

`query()` returns a `Vec<Entity>`. If the plan's `for_each` closure panics
mid-iteration, the `Vec` is dropped (memory freed). No engine state is
modified — the `last_read_tick` is only advanced after the closure
completes successfully.

If the user drops the `QueryPlanResult` before calling `execute()`, the
captured `Arc`s are released. The spatial index remains valid (owned by
the user, not by the plan).

**Verdict:** Safe. Panic/drop leaves no dangling state.

### 6. Can a type bound be violated by a legal generic instantiation?

`add_spatial_index::<T>` requires `T: Component` (i.e., `'static + Send + Sync`).
The `SpatialExpr` is `Clone + Debug` — no generic parameters.
`SpatialLookupFn` is `Arc<dyn Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync>` —
fully concrete.

**Verdict:** No generic instantiation can violate bounds.

### 7. Does the API surface permit operations not covered by Access?

Spatial predicates produce `Predicate` values with
`component_type: TypeId::of::<T>()`. The planner's `Access` bitset
computation includes this component in its read set (spatial predicates
read position data). The `filter_fn` closure captures `&World` and
`Entity` — same as all other filter functions.

The `query()` method on the spatial index does not go through `World` at
all — it reads the index's internal data structures. This is correct:
the index was built from world data during `rebuild()`, and the planner
treats the index as a pre-computed acceleration structure, not a live
world accessor.

**Verdict:** Access bitset accurately reflects the plan's world access.
Index-internal reads are outside the Access model (same as BTree/Hash
today).

## Implementation Plan

### Step 1: Add `query()` to `SpatialIndex` trait

**File:** `crates/minkowski/src/index.rs`

- Add `fn query(&self, expr: &SpatialExpr) -> Vec<Entity>` with default
  empty implementation.
- Update existing test `GridIndex` to implement `query()`.
- Add test: index that returns `supports() -> Some` but default `query()`
  returns empty (safety net behavior).

### Step 2: Add `SpatialLookupFn` and capture in descriptor

**File:** `crates/minkowski/src/planner.rs`

- Define `type SpatialLookupFn = Arc<dyn Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync>`.
- Add `lookup_fn: SpatialLookupFn` field to `SpatialIndexDescriptor`.
- In `add_spatial_index()`, capture `lookup_fn` from the `Arc<dyn SpatialIndex>`.
- Update `SpatialIndexDescriptor` Debug impl.

### Step 3: Thread spatial driver through `ScanBuilder`

**File:** `crates/minkowski/src/planner.rs`

- Define `SpatialDriver { expr: SpatialExpr, lookup_fn: SpatialLookupFn }`.
- Add `spatial_driver: Option<SpatialDriver>` to `ScanBuilder`.
- In `build()` Phase 1, when a spatial predicate is chosen as driver,
  clone the `SpatialExpr` from the predicate and the `lookup_fn` from the
  descriptor into `spatial_driver`.
- Pass `spatial_driver` through to Phase 8.

### Step 4: Compile spatial index-gather closure in Phase 8

**File:** `crates/minkowski/src/planner.rs`

- In Phase 8, check `spatial_driver.is_some()`.
- If present, compile a `CompiledForEach` that calls `lookup_fn(&expr)`,
  validates with `world.is_alive()`, and applies `all_filter_fns`.
- Same for `CompiledForEachRaw`.
- Existing archetype-scan path unchanged (else branch).

### Step 5: Wire spatial driver into join collectors

**File:** `crates/minkowski/src/planner.rs`

- In Phase 7 (join execution state), when the left side has a spatial
  driver, build the `left_collector` using the index-gather pattern
  instead of `collect_matching_entities`.
- Apply filter functions to index results the same way the scan path does.

### Step 6: Add execution tests

**File:** `crates/minkowski/src/planner.rs` (test module)

- Test: `spatial_index_for_each_uses_index` — verify that a plan with a
  registered spatial index calls `query()` (not just filter fusion).
  Use a counting wrapper to confirm the index was actually queried.
- Test: `spatial_index_execute_returns_correct_entities` — spawn entities
  at known positions, run `execute()` with a `Within` predicate, verify
  only nearby entities are returned.
- Test: `spatial_index_stale_entities_filtered` — despawn an entity after
  index rebuild, verify it does not appear in results.
- Test: `spatial_index_for_each_raw_works` — verify read-only execution.
- Test: `spatial_index_join_uses_index` — verify index-driven gathering
  in a join's left collector.

### Step 7: Update planner example

**File:** `examples/examples/planner.rs`

- Add a section demonstrating spatial index registration, plan building,
  and execution with `for_each()`.
- Show EXPLAIN output with `SpatialGather` node.

### Step 8: Update CLAUDE.md

**File:** `CLAUDE.md`

- Document the new `SpatialIndex::query()` method in the Key Traits
  section.
- Add spatial predicates to the Query Planner section.
- Update the example command list if a new example is added.
