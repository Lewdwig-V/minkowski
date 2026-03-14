# BTree/Hash Index Execution Integration

**Goal:** Wire the existing `eq_lookup_fn` / `range_lookup_fn` on `IndexDescriptor`
into the query execution engine so that `IndexLookup` / `IndexGather` plan nodes
actually invoke the registered index at runtime instead of falling back to a full
archetype scan + per-entity filter.

**Supersedes:** The "What's missing" section in the spatial execution design doc
noted this gap: "The same gap exists for `IndexLookup` / `IndexGather` (BTree/Hash),
which also fall back to filter fusion."

---

## Problem

`add_btree_index` and `add_hash_index` capture type-erased lookup closures
(`eq_lookup_fn`, `range_lookup_fn`) at registration time. `Predicate::eq` and
`Predicate::range` store a `lookup_value: Option<Arc<dyn Any + Send + Sync>>`
with the concrete value to look up. The planner correctly emits
`PlanNode::IndexLookup` / `VecExecNode::IndexGather` in EXPLAIN output and uses
index costs for driver selection.

However, Phase 8 has no code path to invoke the lookup functions. All index
predicates fall through to filter fusion on a full archetype scan. The planner
selects the right plan but executes the wrong one.

## Current State

| Component | Status |
|---|---|
| `IndexDescriptor.eq_lookup_fn` / `range_lookup_fn` | Captured at registration, never called |
| `Predicate.lookup_value` | Stored, marked `#[expect(dead_code)]` |
| `PlanNode::IndexLookup` | IR only — emitted in explain, not executed |
| `VecExecNode::IndexGather` | IR only |
| Phase 8 `compiled_for_each` | Only handles `SpatialDriver`, else archetype scan |
| Phase 7 join `left_collector` | Only handles `SpatialDriver`, else archetype scan |

## Design

### Principle: Pre-bind at Phase 3, Execute Clean in Phase 7/8

The `PredicateLookupFn` takes `&dyn Any` — a type-erasure boundary where
generics meet the planner's homogeneous collections. Rather than propagating
`dyn Any` through the execution path, we bind the lookup function and value
together into a parameterless closure at Phase 3 (when we know both).

### IndexDriver

```rust
struct IndexDriver {
    lookup_fn: IndexLookupFn,  // Arc<dyn Fn() -> Arc<[Entity]> + Send + Sync>
}
```

Simpler than `SpatialDriver` — no expression parameter needed. The lookup
value is captured inside the closure.

### Phase 3: Driver Population

After the spatial driver check, if no spatial driver was created and the
best driving access is an `IndexLookup`:

```rust
let index_driver = if spatial_driver.is_none() {
    if let Some((first_pred, first_idx)) = index_preds.first() {
        let lookup_fn = match first_pred.kind {
            PredicateKind::Eq => first_idx.eq_lookup_fn.as_ref(),
            PredicateKind::Range => first_idx.range_lookup_fn.as_ref(),
            _ => None,
        };
        if let (Some(fn_ref), Some(value)) = (lookup_fn, &first_pred.lookup_value) {
            let bound_fn = Arc::clone(fn_ref);
            let bound_value = Arc::clone(value);
            Some(IndexDriver {
                lookup_fn: Arc::new(move || bound_fn(&*bound_value)),
            })
        } else {
            None
        }
    } else {
        None
    }
} else {
    None
};
```

### Phase 7/8: Execution

The priority chain in Phase 8 becomes:

```
if spatial_driver is Some:
    spatial index-gather closure
else if index_driver is Some:
    index-gather closure
else:
    archetype-scan closure
```

The index-gather closure:

```rust
move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
    let candidates = (driver.lookup_fn)();  // Arc<[Entity]>
    for &entity in candidates.iter() {
        if !world.is_alive(entity) {
            continue;
        }
        // Location lookup — skip unplaced entities.
        let idx = entity.index() as usize;
        let Some(loc) = (idx < world.entity_locations.len())
            .then(|| world.entity_locations[idx].as_ref())
            .flatten()
        else {
            continue;
        };
        let arch = &world.archetypes.archetypes[loc.archetype_id.0];
        if !required.is_subset(&arch.component_ids) {
            continue;
        }
        if !changed.is_clear() && !passes_change_filter(arch, &changed, tick) {
            continue;
        }
        if filter_fns.iter().all(|f| f(world, entity)) {
            callback(entity);
        }
    }
}
```

Same validation pipeline as spatial: `is_alive` → location → required →
`Changed<T>` → `filter_fns`. Same pattern for `compiled_for_each_raw` and
join `left_collector`.

### Driver Priority

Spatial driver takes precedence over index driver. If spatial is the
driving access, index predicates become post-filters (existing behavior).
Index driver only activates when spatial driver is `None`.

### Cleanup

Remove `#[expect(dead_code)]` from `Predicate::lookup_value` since it
is now read during `IndexDriver` construction.

## Semantic Review

### 1. Can this be called with the wrong World?

Same situation as spatial and existing index registration — no cross-world
`WorldId` validation exists. Pre-existing gap tracked in spatial design doc.

### 2. Can Drop observe inconsistent state?

`IndexLookupFn` is `Arc<dyn Fn>` — Drop is a ref-count decrement. Safe.

### 3. Can two threads reach this through `&self`?

`Fn` (not `FnMut`) behind `Arc` with `Send + Sync`. Plan execution takes
`&mut self`. Sound.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

Filter fusion collects from ALL predicates. The driving index predicate's
`filter_fn` is still applied as refinement. No predicate silently dropped.

### 5. What happens if this is abandoned halfway through?

`Arc<[Entity]>` from lookup is ref-counted — dropped cleanly on panic.
`last_read_tick` only advanced after successful completion.

### 6. Can a type bound be violated by a legal generic instantiation?

`IndexDriver` is fully concrete. No generics.

### 7. Does the API surface permit operations not covered by Access?

Index predicates include `component_type: TypeId::of::<T>()` in the Access
read set. Index-internal reads are outside the Access model (same as spatial
and existing BTree/Hash today).

## Bypass-Path Invariant Audit

| Invariant | Scan path | Index-gather path | Status |
|---|---|---|---|
| Change detection ticks | `passes_change_filter` per archetype | `passes_change_filter` per entity | Maintained |
| Query cache invalidation | N/A (planner doesn't use World cache) | Same | N/A |
| Access bitset accuracy | Component in read set | Same | Maintained |
| Entity lifecycle | Archetype `entities` vec (always live) | `is_alive` check | Maintained |
| Required components | `required.is_subset` per archetype | `required.is_subset` per entity | Maintained |
| Component presence | Archetype guarantees | `filter_fn` calls `world.get::<T>()` | Maintained |

## Implementation Plan

### Step 1: Add `IndexDriver` and populate in Phase 3

- Define `IndexDriver { lookup_fn: IndexLookupFn }` next to `SpatialDriver`.
- In `build()` Phase 3, after spatial driver creation, create index driver
  when spatial is absent and an index predicate with a lookup function is
  the driving access.
- Remove `#[expect(dead_code)]` from `Predicate::lookup_value`.

### Step 2: Wire index driver into Phase 8

- Extend the Phase 8 `compiled_for_each` / `compiled_for_each_raw` chain:
  spatial → index → archetype scan.
- Same validation pipeline as spatial index-gather.

### Step 3: Wire index driver into Phase 7 join collector

- Extend the join `left_collector` chain: spatial → index → archetype scan.

### Step 4: Add execution tests

- `index_for_each_uses_btree_lookup` — BTree eq lookup invoked via counter
- `index_for_each_uses_hash_lookup` — Hash eq lookup invoked
- `index_range_lookup_execution` — BTree range lookup
- `index_lookup_filters_stale_entities` — despawned entity filtered
- `index_lookup_filters_missing_required` — multi-component query
- `index_join_uses_lookup` — join left collector uses index driver

### Step 5: Update CLAUDE.md and example

- Document index execution in Query Planner section.
- Add index execution section to planner example.
