# Volcano-Model Query Planner Design

## Problem

Minkowski's query system is a flat scan: `world.query::<(&Pos, &Vel)>()` iterates every entity in every matching archetype. This is optimal when you need all matching entities, but wasteful when you only need a subset — e.g. "entities with Health > 50 whose position is in region [0,100]×[0,100]".

With BTreeIndex and HashIndex now in the engine, there's an opportunity to let queries *start* from an index lookup and fetch components only for the matching subset. Today this requires manual code:

```rust
let entities = pos_index.range(0.0..=100.0);
for &entity in entities {
    if world.is_alive(entity) {
        if let Some(vel) = world.get::<Vel>(entity) {
            // ...
        }
    }
}
```

A query planner replaces this manual wiring with composable operators that the engine can reason about and optimize.

**Why now?** The BTreeIndex/HashIndex PR landed, giving the engine its first secondary indexes. The Volcano model is the natural next step — it composes index scans with the existing iteration infrastructure.

## Current State

### Query execution (`crates/minkowski/src/world.rs:422-505`)
- `world.query::<Q>()` → cache lookup → incremental archetype scan → filter → mark columns → init_fetch → QueryIter
- Returns `QueryIter` with precomputed `Vec<(Fetch, len)>` pairs
- All iteration is archetype-sequential: row 0..len per archetype

### Query iterator (`crates/minkowski/src/query/iter.rs:1-85`)
- `QueryIter` supports three modes: `Iterator::next()`, `for_each_chunk`, `par_for_each`
- No composition — QueryIter is a flat list of (fetch, len) pairs

### WorldQuery trait (`crates/minkowski/src/query/fetch.rs:28-86`)
- `required_ids` → archetype filtering
- `init_fetch` → per-archetype column pointer capture
- `fetch(row)` → per-row item extraction
- `matches_filters` → archetype-level skip (Changed<T>)

### Column indexes (`crates/minkowski/src/index.rs`)
- `BTreeIndex<T>`: `get(value)`, `range(bounds)`, `get_valid`, `range_valid`
- `HashIndex<T>`: `get(value)`, `get_valid`
- Both return `&[Entity]` — unsorted, may contain stale entries
- External to World, updated via `SpatialIndex::update()`

### Entity-level fetch (`crates/minkowski/src/world.rs`)
- `world.get::<T>(entity)` → O(1) location lookup → archetype row fetch
- `world.get_mut::<T>(entity)` → same + change tick marking
- No batch `get_many` — each call does independent location lookup

### Access metadata (`crates/minkowski/src/access.rs`)
- `Access::of::<Q>(world)` → read/write bitsets from WorldQuery
- Used by ReducerRegistry for conflict detection

## Proposed Design

### Core Idea

Introduce composable **query operators** following the Volcano iterator model. Each operator implements a common trait (`QueryOp`) with `open`/`next`/`close` semantics. Operators compose into a plan tree evaluated lazily — data flows bottom-up through `next()` calls.

### Operator Types

| Operator | Input | Output | Description |
|---|---|---|---|
| `ArchScan<Q>` | — | `Q::Item` | Full archetype scan (existing behavior, wrapped) |
| `IndexScan<T>` | — | `Entity` | Entities from BTreeIndex range or HashIndex exact match |
| `Fetch<Q>` | `Entity` stream | `(Entity, Q::Item)` | Fetch components for a stream of entity IDs |
| `Filter<F>` | any stream | same stream | Predicate filter on items |
| `Intersect` | two `Entity` streams | `Entity` | Set intersection of two entity streams |

### API Surface

```rust
// Builder pattern — constructs a plan, then executes
let results: Vec<(Entity, &Pos, &Vel)> = world
    .plan()
    .index_range(&pos_index, 0.0..=100.0)   // IndexScan → Entity stream
    .fetch::<(Entity, &Pos, &Vel)>()          // Fetch components
    .filter(|(_, pos, vel)| vel.dx > 0.0)     // Predicate filter
    .collect();                               // Execute and collect

// Full scan (equivalent to world.query, but composable)
let results = world
    .plan()
    .scan::<(&mut Pos, &Vel)>()
    .filter(|(pos, vel)| pos.x < 100.0)
    .collect();

// Index intersection
let results = world
    .plan()
    .index_exact(&team_index, Team::Red)      // Entities on team Red
    .intersect_index(&health_index, 50..=100) // AND health 50-100
    .fetch::<(Entity, &Pos)>()
    .collect();
```

### Internal Architecture

```
crates/minkowski/src/query/
├── fetch.rs       # WorldQuery trait (existing)
├── iter.rs        # QueryIter (existing)
├── plan.rs        # NEW: QueryPlan builder, QueryOp trait, operators
└── mod.rs         # Module exports
```

**QueryOp trait:**

```rust
pub(crate) trait QueryOp {
    type Item<'w>;

    /// Prepare the operator for iteration.
    fn open(&mut self);

    /// Yield the next item, or None when exhausted.
    fn next(&mut self) -> Option<Self::Item<'_>>;

    /// Estimated number of rows (for cost-based decisions).
    fn estimate(&self) -> usize;
}
```

**QueryPlan builder:**

```rust
pub struct QueryPlan<'w> {
    world: &'w mut World,
}

impl<'w> QueryPlan<'w> {
    /// Start from a full archetype scan.
    pub fn scan<Q: WorldQuery>(self) -> ScanPlan<'w, Q>;

    /// Start from an index range lookup.
    pub fn index_range<T, R>(self, index: &BTreeIndex<T>, range: R) -> EntityPlan<'w>
    where
        T: Ord + Component,
        R: RangeBounds<T>;

    /// Start from an index exact match.
    pub fn index_exact<T>(self, index: &HashIndex<T>, value: &T) -> EntityPlan<'w>
    where
        T: Hash + Eq + Component;
}
```

**EntityPlan** (stream of entity IDs):

```rust
pub struct EntityPlan<'w> {
    world: &'w mut World,
    entities: Vec<Entity>,  // Materialized entity list from index
}

impl<'w> EntityPlan<'w> {
    /// Fetch components for each entity in the stream.
    pub fn fetch<Q: WorldQuery>(self) -> FetchPlan<'w, Q>;

    /// Intersect with another index lookup.
    pub fn intersect_index<T, R>(self, index: &BTreeIndex<T>, range: R) -> EntityPlan<'w>;

    /// Filter to alive entities only.
    pub fn alive(self) -> EntityPlan<'w>;
}
```

**FetchPlan** (stream of (Entity, components)):

```rust
pub struct FetchPlan<'w, Q: WorldQuery> {
    world: &'w mut World,
    entities: Vec<Entity>,
    _marker: PhantomData<Q>,
}

impl<'w, Q: WorldQuery> FetchPlan<'w, Q> {
    /// Apply a predicate filter.
    pub fn filter<F>(self, predicate: F) -> FilterPlan<'w, Q, F>;

    /// Collect results into a Vec.
    pub fn collect(self) -> Vec<(Entity, Q::Item<'w>)>;

    /// Iterate with a closure.
    pub fn for_each<F>(self, f: F) where F: FnMut((Entity, Q::Item<'_>));
}
```

### Data Flow

**Index-accelerated query:**

```
index_range(&pos_index, 0.0..100.0)
    → Vec<Entity> (from BTreeIndex::range)
    → filter is_alive (generational check)
    → for each entity: world.entity_locations[idx] → (arch_id, row)
    → group by archetype (sort by arch_id for cache locality)
    → per archetype: init_fetch once, fetch(row) for each entity
    → yield (Entity, Q::Item)
    → apply predicate filter
    → collect/for_each
```

**Key optimization — batch fetch by archetype:**

Instead of calling `world.get(entity)` N times (N location lookups, N archetype accesses), group entities by archetype first, then do a single `init_fetch` per archetype and fetch all rows in that archetype together. This is the same pattern as `QueryIter` but with a sparse row set instead of dense 0..len.

```rust
// Pseudocode for the batch fetch
let mut by_arch: HashMap<ArchetypeId, Vec<(Entity, usize)>> = HashMap::new();
for entity in entities {
    if let Some(loc) = world.entity_location(entity) {
        by_arch.entry(loc.archetype_id).or_default().push((entity, loc.row));
    }
}
for (arch_id, rows) in by_arch {
    let fetch = Q::init_fetch(archetype, registry);
    for (entity, row) in rows {
        let item = Q::fetch(&fetch, row);
        yield (entity, item);
    }
}
```

### Change Detection Integration

- `scan()` path: uses existing `world.query()` machinery — ticks, cache, `Changed<T>` filters all work
- `index_range/exact` + `fetch` path: does NOT go through the query cache. Change detection for mutable columns must be handled explicitly:
  - Advance tick before fetch
  - Mark mutable columns for each accessed archetype
  - This is the same pattern as `query_table_mut` — a bypass path that manually maintains tick invariants

### Access Integration

`QueryPlan` must produce `Access` metadata for reducer/scheduler integration:

```rust
impl<'w, Q: WorldQuery> FetchPlan<'w, Q> {
    pub fn access(world: &mut World) -> Access {
        Access::of::<Q>(world)
    }
}
```

Index reads don't touch World state, so they don't contribute to Access. Only the `fetch` step determines read/write access.

## Alternatives Considered

### A. Macro-based query DSL

```rust
mink_query! {
    FROM pos_index RANGE 0.0..100.0
    FETCH Pos, Vel
    WHERE vel.dx > 0.0
}
```

**Tradeoffs:** More SQL-like, potentially cleaner syntax. But proc macros are opaque to tooling, harder to debug, and the Volcano model's composability already gives good ergonomics via method chaining. The builder pattern is more Rustic and type-safe.

**Rejected** — proc macro complexity without sufficient benefit over builder pattern.

### B. Lazy virtual iterator (full Volcano pull model)

Each operator is a separate struct implementing a `QueryOp` trait with `next()`. Operators compose via wrapping:

```rust
let iter = Filter::new(
    Fetch::<(&Pos, &Vel)>::new(
        IndexScan::new(&pos_index, 0.0..100.0),
        &world,
    ),
    |item| item.1.dx > 0.0,
);
```

**Tradeoffs:** True lazy evaluation — no intermediate `Vec<Entity>`. But entity-by-entity pull through operators prevents the batch-by-archetype optimization (critical for cache locality). Also makes lifetime management painful — each operator borrows the previous one, creating nested borrow chains.

**Partially adopted** — the ScanPlan path uses lazy iteration (wrapping existing QueryIter), but the index path materializes entity lists to enable archetype-grouped batch fetch.

### C. No planner — keep manual wiring

Users continue writing manual loops:

```rust
for &entity in pos_index.range(0.0..100.0) {
    if world.is_alive(entity) { ... }
}
```

**Tradeoffs:** Zero new API surface, zero new complexity. But the archetype-grouped batch fetch optimization is non-trivial to implement correctly — users would need to understand entity locations, archetype IDs, and fetch initialization. The planner encapsulates this.

**Rejected** — the batch fetch optimization is too valuable to leave to users.

## Semantic Review

### 1. Can this be called with the wrong World?

`QueryPlan` holds `&'w mut World` — the plan is bound to the World that created it. Index references are separate: a `BTreeIndex` built from World A could be passed to a plan on World B. The entities would resolve to wrong locations. **Mitigation:** document that indexes must be built from the same World. No runtime check — same as `SpatialIndex` today.

### 2. Can Drop observe inconsistent state?

`QueryPlan` is a builder — no cleanup needed. `EntityPlan` and `FetchPlan` hold `Vec<Entity>` (owned) and `&mut World` (borrowed). Drop is trivial. No transaction lifecycle, no allocated engine resources.

### 3. Can two threads reach this through `&self`?

No. `QueryPlan` holds `&mut World` — exclusive access. Not `Send`/`Sync` by default (contains raw pointers via fetch state). Parallel execution would use `par_for_each` on the final result, not parallel plan construction.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

`intersect_index` computes set intersection of entity lists. Entity dedup is by identity (Entity is Copy, Eq). No ordering invariant to preserve — results are grouped by archetype for cache locality, not sorted by entity ID.

### 5. What happens if this is abandoned halfway through?

Builder pattern — abandoning a `QueryPlan` or `EntityPlan` drops the intermediate `Vec<Entity>` and releases the `&mut World` borrow. No side effects until `collect()` or `for_each()` is called. The `scan()` path wraps `QueryIter` which also has no cleanup.

Exception: if `fetch` marks mutable columns changed (tick advancement), abandoning iteration after `fetch` construction but before consuming results would leave columns marked changed without being read. This is pessimistic but safe — same behavior as constructing a `world.query::<&mut T>()` and dropping it.

### 6. Can a type bound be violated by a legal generic instantiation?

`Fetch<Q>` requires `Q: WorldQuery`. The builder methods constrain `T: Ord + Component` for BTreeIndex, `T: Hash + Eq + Component` for HashIndex. These match the index types' own bounds. No escape hatch.

### 7. Does the API surface of this handle permit any operation not covered by the Access bitset?

`FetchPlan<Q>` has the same access profile as `world.query::<Q>()` — the `Q` type determines reads and writes. Index scans don't touch component data (they return Entity IDs). `Access::of::<Q>(world)` accurately captures the plan's access. The `filter` closure receives items by reference — it can't expand access beyond Q.

## Implementation Plan

### Phase 1: Core planner (minimal)

1. **`crates/minkowski/src/query/plan.rs`** — New file:
   - `QueryPlan<'w>` builder struct
   - `EntityPlan<'w>` for entity ID streams (from indexes)
   - `FetchPlan<'w, Q>` for component fetch over entity streams
   - `ScanPlan<'w, Q>` wrapping existing `QueryIter` with `.filter()` support
   - Batch-by-archetype fetch logic (group entities by archetype, single init_fetch per archetype)

2. **`crates/minkowski/src/world.rs`** — Add `World::plan()` method:
   - Returns `QueryPlan<'w>` holding `&'w mut World`
   - Minimal: just a builder entry point

3. **`crates/minkowski/src/query/mod.rs`** — Export plan types

4. **`crates/minkowski/src/lib.rs`** — Re-export `QueryPlan` (and intermediate types if needed)

5. **Tests in `plan.rs`**:
   - `scan_equivalent_to_query` — scan().collect() == query().collect()
   - `index_range_fetch` — index_range + fetch returns correct components
   - `index_exact_fetch` — index_exact + fetch returns correct components
   - `filter_on_scan` — predicate filtering works
   - `filter_on_fetch` — predicate filtering after index fetch works
   - `intersect_two_indexes` — intersection returns entities in both
   - `batch_fetch_groups_by_archetype` — verify archetype grouping (internal)
   - `alive_filter` — stale entities filtered out
   - `empty_index_result` — no entities from index → empty result
   - `mutable_fetch_marks_changed` — change detection ticks work through plan

### Phase 2: Change detection + Access

6. **Tick management in FetchPlan** — mark mutable columns, advance ticks (same pattern as `query_table_mut`)

7. **Access integration** — `FetchPlan::access()` returns `Access::of::<Q>(world)` for scheduler compatibility

### Phase 3: Example + docs

8. **`examples/examples/planner.rs`** — Demonstrate:
   - Full scan with filter
   - Index-accelerated range query
   - Index intersection
   - Performance comparison: manual loop vs plan vs full scan

9. **`docs/adr/012-volcano-query-planner.md`** — ADR documenting the decision

10. **`CLAUDE.md`** — Add `QueryPlan` to pub API list, add example run command
