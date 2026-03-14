# Spatial Query Execution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire spatial index lookup functions into the query execution engine so that `SpatialLookup`/`SpatialGather` plan nodes actually invoke the registered spatial index at runtime instead of falling back to a full archetype scan.

**Architecture:** User provides a lookup closure via `add_spatial_index_with_lookup()` at registration time. The planner threads this through `ScanBuilder` as a `SpatialDriver`. Phase 8 compiles an index-gather closure that calls the lookup, validates entities (`is_alive`, `Changed<T>`), and applies refinement filters. Same pattern extends to join left-side collectors.

**Tech Stack:** Rust, minkowski ECS (planner.rs, index.rs, lib.rs)

**Spec:** `docs/plans/2026-03-13-spatial-query-execution-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/src/planner.rs` | Modify | `SpatialLookupFn` type, `SpatialIndexDescriptor.lookup_fn`, `SpatialDriver`, `add_spatial_index_with_lookup`, Phase 8 index-gather, join integration, tests |
| `crates/minkowski/src/lib.rs` | Modify | Re-export `SpatialLookupFn` |
| `examples/examples/planner.rs` | Modify | Spatial execution demo section |
| `CLAUDE.md` | Modify | Document new API |

---

## Task 1: Registration API — `SpatialLookupFn` and `add_spatial_index_with_lookup`

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1871-1875` (SpatialIndexDescriptor)
- Modify: `crates/minkowski/src/planner.rs:2057-2074` (add_spatial_index)
- Modify: `crates/minkowski/src/lib.rs:104` (re-exports)

- [ ] **Step 1: Add `SpatialLookupFn` type alias and update `SpatialIndexDescriptor`**

In `planner.rs`, after the existing `SpatialLookupResult` enum (around line 1870), add the type alias and update the descriptor:

```rust
/// Type-erased spatial lookup: takes a `SpatialExpr`, returns candidate entities.
/// Provided by the user at registration time to bridge between the planner's
/// expression protocol and the index's concrete query API.
pub type SpatialLookupFn = Arc<dyn Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync>;

struct SpatialIndexDescriptor {
    component_name: &'static str,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    lookup_fn: Option<SpatialLookupFn>,
}
```

- [ ] **Step 2: Implement `add_spatial_index_with_lookup`**

Add the new method to `QueryPlanner` right after the existing `add_spatial_index`:

```rust
/// Register a spatial index with both cost discovery and execution-time lookup.
///
/// The `lookup` closure bridges between the planner's [`SpatialExpr`]
/// protocol and the index's concrete query API. The planner makes no
/// assumptions about how the index answers queries — the closure is the
/// adapter.
///
/// # Example
///
/// ```ignore
/// let grid = Arc::new(SpatialGrid::new());
/// let g = Arc::clone(&grid);
/// planner.add_spatial_index_with_lookup::<Pos>(
///     Arc::clone(&grid), &world,
///     move |expr| match expr {
///         SpatialExpr::Within { center, radius } => g.query_radius(center, *radius),
///         _ => Vec::new(),
///     },
/// );
/// ```
pub fn add_spatial_index_with_lookup<T: Component>(
    &mut self,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    world: &World,
    lookup: impl Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync + 'static,
) {
    assert!(
        world.component_id::<T>().is_some(),
        "QueryPlanner::add_spatial_index_with_lookup: component `{}` not registered",
        std::any::type_name::<T>()
    );
    self.spatial_indexes.insert(
        TypeId::of::<T>(),
        SpatialIndexDescriptor {
            component_name: std::any::type_name::<T>(),
            index,
            lookup_fn: Some(Arc::new(lookup)),
        },
    );
}
```

- [ ] **Step 3: Refactor `add_spatial_index` to set `lookup_fn: None`**

Update the existing method to include the new field:

```rust
pub fn add_spatial_index<T: Component>(
    &mut self,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    world: &World,
) {
    assert!(
        world.component_id::<T>().is_some(),
        "QueryPlanner::add_spatial_index: component `{}` not registered in this World",
        std::any::type_name::<T>()
    );
    self.spatial_indexes.insert(
        TypeId::of::<T>(),
        SpatialIndexDescriptor {
            component_name: std::any::type_name::<T>(),
            index,
            lookup_fn: None,
        },
    );
}
```

- [ ] **Step 4: Re-export `SpatialLookupFn` from `lib.rs`**

Add `SpatialLookupFn` to the planner re-export line in `lib.rs:104`:

```rust
    QueryPlanResult, QueryPlanner, SpatialLookupFn, SpatialPredicate, SubscriptionBuilder, ...
```

- [ ] **Step 5: Run `cargo check` and `cargo test -p minkowski --lib`**

Run: `cargo check --workspace --quiet && cargo test -p minkowski --lib`
Expected: All existing tests pass (619+). No compilation errors.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs crates/minkowski/src/lib.rs
git commit -m "Add SpatialLookupFn and add_spatial_index_with_lookup registration"
```

---

## Task 2: Thread `SpatialDriver` through `ScanBuilder`

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1393-1409` (ScanBuilder struct)
- Modify: `crates/minkowski/src/planner.rs:1465-1590` (build() Phase 1)

- [ ] **Step 1: Define `SpatialDriver` and add to `ScanBuilder`**

Add the struct before `ScanBuilder` (around line 1390):

```rust
/// Carries the spatial lookup function and expression from Phase 1
/// (predicate classification) to Phase 8 (closure compilation).
struct SpatialDriver {
    expr: SpatialExpr,
    lookup_fn: SpatialLookupFn,
}
```

Add to `ScanBuilder`:

```rust
pub struct ScanBuilder<'w> {
    // ... existing fields ...
    /// Spatial index driver — set when a spatial predicate is chosen as the
    /// driving access and the index has a registered lookup function.
    spatial_driver: Option<SpatialDriver>,
}
```

- [ ] **Step 2: Initialize `spatial_driver: None` in `scan()` and `scan_with_estimate()`**

In both `QueryPlanner::scan()` (line ~2086) and `scan_with_estimate()` (line ~2151), add `spatial_driver: None` to the `ScanBuilder` initializer.

- [ ] **Step 3: Populate `spatial_driver` in Phase 1 of `build()`**

In `build()`, after Phase 1 classifies predicates (around line 1487-1510), when a spatial predicate is added to `spatial_preds`, also check if the descriptor has a `lookup_fn`. Then after Phase 2 sorting, when `use_spatial_driver` is true (line 1587), populate the driver:

Update `find_spatial_index` to return the lookup_fn when available. Change `SpatialLookupResult::Accelerated` to carry an `Option<SpatialLookupFn>`:

```rust
enum SpatialLookupResult {
    Accelerated(&'static str, SpatialCost, Option<SpatialLookupFn>),
    Declined(String),
    NoIndex,
}
```

Update `find_spatial_index`:

```rust
fn find_spatial_index(&self, pred: &Predicate) -> SpatialLookupResult {
    let PredicateKind::Spatial(sp) = &pred.kind else {
        return SpatialLookupResult::NoIndex;
    };
    let Some(desc) = self.spatial_indexes.get(&pred.component_type) else {
        return SpatialLookupResult::NoIndex;
    };
    let expr: SpatialExpr = sp.into();
    match desc.index.supports(&expr) {
        Some(cost) => SpatialLookupResult::Accelerated(
            desc.component_name,
            cost,
            desc.lookup_fn.as_ref().map(Arc::clone),
        ),
        None => SpatialLookupResult::Declined(sp.to_string()),
    }
}
```

Update Phase 1 to store lookup_fn alongside spatial_preds. Change the type from `Vec<(Predicate, SpatialCost)>` to `Vec<(Predicate, SpatialCost, Option<SpatialLookupFn>)>`:

```rust
let mut spatial_preds: Vec<(Predicate, SpatialCost, Option<SpatialLookupFn>)> = Vec::new();

// In the match arm:
SpatialLookupResult::Accelerated(_name, cost, lookup) => {
    spatial_preds.push((pred, cost, lookup));
}
```

Then after Phase 3's `use_spatial_driver` check (line 1587), set the driver:

```rust
// After: if use_spatial_driver && !spatial_preds.is_empty() {
let spatial_driver = if use_spatial_driver && !spatial_preds.is_empty() {
    let (first_pred, _, first_lookup) = &spatial_preds[0];
    if let Some(lookup_fn) = first_lookup {
        let PredicateKind::Spatial(sp) = &first_pred.kind else {
            unreachable!("spatial_preds only contains Spatial predicates");
        };
        Some(SpatialDriver {
            expr: sp.into(),
            lookup_fn: Arc::clone(lookup_fn),
        })
    } else {
        None
    }
} else {
    None
};
```

Note: the existing `spatial_preds[0]` destructuring in the SpatialLookup node construction must be updated for the new 3-tuple: `let (first_pred, first_cost, _) = &spatial_preds[0];`. Update all other `spatial_preds` iterations similarly (`.iter().skip(1)` patterns become `(pred, _, _)`).

- [ ] **Step 4: Pass `spatial_driver` through to Phase 8**

The `spatial_driver` local needs to be accessible in Phase 7 (join integration) and Phase 8 (closure compilation). Since `build()` consumes `self`, the driver is a local variable — just let it flow down to those phases.

- [ ] **Step 5: Run `cargo check` and `cargo test -p minkowski --lib`**

Run: `cargo check --workspace --quiet && cargo test -p minkowski --lib`
Expected: All tests pass. `spatial_driver` is always `None` in existing code paths since no tests call `add_spatial_index_with_lookup` yet.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Thread SpatialDriver through ScanBuilder from Phase 1 to Phase 8"
```

---

## Task 3: Compile index-gather closure in Phase 8

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1796-1840` (Phase 8)

- [ ] **Step 1: Write the failing test `spatial_index_for_each_uses_lookup`**

Add to the test module at the end of `planner.rs`:

```rust
#[test]
fn spatial_index_for_each_uses_lookup() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));
    let _e3 = world.spawn((Pos { x: 100.0, y: 100.0 },));

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = Arc::clone(&call_count);

    // Spatial index that supports Within and returns e1, e2 as candidates.
    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index_with_lookup::<Pos>(
        Arc::new(grid),
        &world,
        move |_expr: &SpatialExpr| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
            vec![e1, e2] // Only return nearby entities
        },
    );

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([5.0, 5.0], 10.0, |_, _| true))
        .build();

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| {
        results.push(entity);
    });

    // The lookup was called (not a full scan).
    assert!(
        call_count.load(Ordering::Relaxed) > 0,
        "lookup function was never called"
    );
    // Only the two nearby entities were returned.
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- spatial_index_for_each_uses_lookup`
Expected: FAIL — the lookup is never called because Phase 8 always compiles the archetype-scan closure.

- [ ] **Step 3: Implement index-gather closure in Phase 8**

In `build()`, Phase 8 (line ~1802), modify the `compiled_for_each` block. Before the existing `self.compile_for_each.map(|factory| { ... })`, check if `spatial_driver` is set:

```rust
let compiled_for_each = if self.joins.is_empty() {
    if let Some(driver) = &spatial_driver {
        // Index-gather path: call the lookup function instead of scanning archetypes.
        let lookup_fn = Arc::clone(&driver.lookup_fn);
        let expr = driver.expr.clone();
        let changed = self.planner.changed_for_spatial.clone().unwrap_or_default();
        Some(Box::new(
            move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                let candidates = lookup_fn(&expr);
                for entity in candidates {
                    if !world.is_alive(entity) {
                        continue;
                    }
                    if !changed.is_empty() {
                        let idx = entity.index() as usize;
                        if idx < world.entity_locations.len() {
                            if let Some(loc) = &world.entity_locations[idx] {
                                let arch = &world.archetypes.archetypes[loc.archetype_id];
                                if !passes_change_filter(arch, &changed, tick) {
                                    continue;
                                }
                            }
                        }
                    }
                    if all_filter_fns.iter().all(|f| f(world, entity)) {
                        callback(entity);
                    }
                }
            },
        ) as CompiledForEach)
    } else {
        self.compile_for_each.map(|factory| {
            // ... existing archetype-scan path unchanged ...
        })
    }
} else {
    None
};
```

**Important:** The `changed` bitset for the index-gather path must come from the query's `Changed<T>` components. This is the same `changed` bitset used in the scan path. It's computed in `scan::<Q>()` via `Q::changed_ids(self.planner.components)`. We need to thread it through.

Actually, looking at the existing code, the `changed` bitset is captured inside the `compile_for_each` factory closure (lines 2094, 2113). For the index-gather path, we need it at `build()` time. Add a `changed: Option<FixedBitSet>` field to `ScanBuilder`, set it in `scan()`:

```rust
pub struct ScanBuilder<'w> {
    // ... existing fields ...
    spatial_driver: Option<SpatialDriver>,
    /// Changed component bitset for spatial index-gather path.
    changed_for_spatial: Option<FixedBitSet>,
}
```

In `scan()` and `scan_with_estimate()`, add:
```rust
changed_for_spatial: Some(changed.clone()),
```

Then in Phase 8, the index-gather closure captures `self.changed_for_spatial.take().unwrap_or_default()`.

Do the same for `compiled_for_each_raw` — same closure logic, both paths use the index-gather pattern when a spatial driver is present.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- spatial_index_for_each_uses_lookup`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: All 620+ tests pass, clippy clean.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Compile index-gather closure in Phase 8 when spatial driver has lookup"
```

---

## Task 4: Join collector integration

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1730-1762` (Phase 7 left_collector)

- [ ] **Step 1: Write the failing test `spatial_index_join_uses_lookup`**

```rust
#[test]
fn spatial_index_join_uses_lookup() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 }, Score(10)));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 }, Score(20)));
    let _e3 = world.spawn((Pos { x: 100.0, y: 100.0 }, Score(30)));

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index_with_lookup::<Pos>(
        Arc::new(grid),
        &world,
        move |_expr| {
            cc.fetch_add(1, Ordering::Relaxed);
            vec![e1, e2]
        },
    );

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([5.0, 5.0], 10.0, |_, _| true))
        .join::<(&Score,)>(JoinKind::Inner)
        .build();

    let results = plan.execute(&mut world);
    assert!(call_count.load(Ordering::Relaxed) > 0, "lookup not called in join");
    // Only e1 and e2 have both Pos and Score, e3 is excluded by the lookup.
    assert_eq!(results.len(), 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- spatial_index_join_uses_lookup`
Expected: FAIL — join left collector always uses `collect_matching_entities`.

- [ ] **Step 3: Implement spatial-driven left collector in Phase 7**

In `build()` Phase 7 (line ~1737), before building `left_collector`, check if a spatial driver is available:

```rust
let left_collector: EntityCollector = if let Some(driver) = &spatial_driver {
    let lookup_fn = Arc::clone(&driver.lookup_fn);
    let expr = driver.expr.clone();
    let left_changed_for_index = left_changed.clone();
    Box::new(
        move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
            let candidates = lookup_fn(&expr);
            for entity in candidates {
                if !world.is_alive(entity) {
                    continue;
                }
                if !left_changed_for_index.is_empty() {
                    let idx = entity.index() as usize;
                    if idx < world.entity_locations.len() {
                        if let Some(loc) = &world.entity_locations[idx] {
                            let arch = &world.archetypes.archetypes[loc.archetype_id];
                            if !passes_change_filter(arch, &left_changed_for_index, tick) {
                                continue;
                            }
                        }
                    }
                }
                if left_filters.iter().all(|f| f(world, entity)) {
                    scratch.push(entity);
                }
            }
        },
    )
} else {
    // Existing archetype-scan path (unchanged)
    Box::new(
        move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
            // ... existing code ...
        },
    )
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski --lib -- spatial_index_join_uses_lookup`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: All tests pass, clippy clean.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Wire spatial driver into join left-side collector"
```

---

## Task 5: Execution tests

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (test module, end of file)

- [ ] **Step 1: Write `spatial_index_execute_returns_correct_entities`**

```rust
#[test]
fn spatial_index_execute_returns_correct_entities() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));
    let _far = world.spawn((Pos { x: 999.0, y: 999.0 },));

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index_with_lookup::<Pos>(
        Arc::new(grid),
        &world,
        move |_expr| vec![e1, e2],
    );

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true))
        .build();

    let results = plan.execute(&mut world);
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}
```

- [ ] **Step 2: Write `spatial_index_stale_entities_filtered`**

```rust
#[test]
fn spatial_index_stale_entities_filtered() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    // Lookup returns both, but e2 will be despawned before execution.
    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index_with_lookup::<Pos>(
        Arc::new(grid),
        &world,
        move |_expr| vec![e1, e2],
    );

    // Despawn e2 after index registration (index is stale).
    world.despawn(e2);

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true))
        .build();

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}
```

- [ ] **Step 3: Write `spatial_index_for_each_raw_works`**

```rust
#[test]
fn spatial_index_for_each_raw_works() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index_with_lookup::<Pos>(
        Arc::new(grid),
        &world,
        move |_expr| {
            cc.fetch_add(1, Ordering::Relaxed);
            vec![e1]
        },
    );

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true))
        .build();

    let mut results = Vec::new();
    plan.for_each_raw(&world, |entity| results.push(entity));

    assert!(call_count.load(Ordering::Relaxed) > 0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}
```

- [ ] **Step 4: Write `spatial_index_without_lookup_falls_back`**

```rust
#[test]
fn spatial_index_without_lookup_falls_back() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Pos { x: i as f32, y: i as f32 },));
    }

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    // Cost-only registration — no lookup closure.
    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index::<Pos>(Arc::new(grid), &world);

    let mut plan = planner
        .scan::<(&Pos,)>()
        .filter(Predicate::within::<Pos>([5.0, 5.0], 100.0, |_, _| true))
        .build();

    // Should fall back to scan + filter — returns all 10 entities
    // because the filter closure is |_, _| true.
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 10);
}
```

- [ ] **Step 5: Write `spatial_index_with_changed_filter`**

```rust
#[test]
fn spatial_index_with_changed_filter() {
    let mut world = World::new();
    let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
    let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

    let mut grid = TestGridIndex::new();
    grid.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_spatial_index_with_lookup::<Pos>(
        Arc::new(grid),
        &world,
        move |_expr| vec![e1, e2],
    );

    // First scan — both entities match (all are "changed" on first read).
    let mut plan = planner
        .scan::<(Changed<Pos>, &Pos)>()
        .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true))
        .build();

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));
    assert_eq!(results.len(), 2);

    // Mutate only e1's Pos.
    world.get_mut::<Pos>(e1).unwrap().x = 99.0;

    // Second scan — only e1 should pass Changed<Pos> filter.
    results.clear();
    plan.for_each(&mut world, |entity| results.push(entity));
    assert_eq!(results.len(), 1, "only mutated entity should pass Changed<Pos>");
    assert_eq!(results[0], e1);
}
```

- [ ] **Step 6: Run all new tests**

Run: `cargo test -p minkowski --lib -- spatial_index_for_each_uses spatial_index_execute spatial_index_stale spatial_index_for_each_raw spatial_index_without_lookup spatial_index_with_changed`
Expected: All 6 new tests PASS.

- [ ] **Step 7: Run full suite + clippy**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: 625+ tests pass, clippy clean.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Add spatial index execution tests (lookup, stale, raw, fallback, Changed<T>)"
```

---

## Task 6: Update planner example

**Files:**
- Modify: `examples/examples/planner.rs`

- [ ] **Step 1: Add spatial execution section to the planner example**

Add after the existing sections (near end of file, before `main()` returns). The section should demonstrate:

1. Building a simple spatial index (reuse the example's existing `Pos` type)
2. Registering with `add_spatial_index_with_lookup`
3. Building a plan with `Predicate::within`
4. EXPLAIN output showing `SpatialGather`
5. `for_each` execution with results

The spatial index can be a trivial one that just returns all entities within a distance — a Vec scan. The point is demonstrating the API, not building a high-performance index.

- [ ] **Step 2: Run the example**

Run: `cargo run -p minkowski-examples --example planner --release`
Expected: Output includes the spatial section with EXPLAIN showing `SpatialGather` and execution results.

- [ ] **Step 3: Commit**

```bash
git add examples/examples/planner.rs
git commit -m "Add spatial index execution demo to planner example"
```

---

## Task 7: `TablePlanner` delegation

**Files:**
- Modify: `crates/minkowski/src/planner.rs:2865-2949` (TablePlanner impl)

- [ ] **Step 1: Add `add_spatial_index_with_lookup` to `TablePlanner`**

Add after the existing `add_hash_index` method (line ~2924):

```rust
/// Register a spatial index with cost discovery and execution-time lookup.
///
/// Delegates to [`QueryPlanner::add_spatial_index_with_lookup`].
/// No compile-time index enforcement — spatial indexes are orthogonal to
/// table schemas.
pub fn add_spatial_index_with_lookup<C: Component>(
    &mut self,
    index: Arc<dyn SpatialIndex + Send + Sync>,
    lookup: impl Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync + 'static,
) {
    self.planner
        .add_spatial_index_with_lookup::<C>(index, self.world, lookup);
}

/// Register a spatial index for cost discovery only (no execution-time lookup).
///
/// Delegates to [`QueryPlanner::add_spatial_index`].
pub fn add_spatial_index<C: Component>(
    &mut self,
    index: Arc<dyn SpatialIndex + Send + Sync>,
) {
    self.planner.add_spatial_index::<C>(index, self.world);
}
```

- [ ] **Step 2: Run `cargo check` and `cargo test -p minkowski --lib`**

Run: `cargo check --workspace --quiet && cargo test -p minkowski --lib`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Add spatial index delegation methods to TablePlanner"
```

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Query Planner section**

In CLAUDE.md's "Query Planner" section, add documentation for the new spatial execution path. After the existing paragraph about `QueryPlanResult`:

```markdown
`QueryPlanner::add_spatial_index_with_lookup::<T>(index, world, lookup)` registers a spatial index with an execution-time lookup closure. The closure bridges between the planner's `SpatialExpr` protocol and the index's concrete query API — the planner makes no assumptions about how the index answers queries (mechanisms not policy). When a spatial predicate is chosen as the driving access, Phase 8 compiles an index-gather closure that calls the lookup instead of scanning archetypes. `add_spatial_index` (without lookup) remains the cost-only registration path — plans fall back to scan + filter at execution time.
```

- [ ] **Step 2: Run `cargo test -p minkowski --lib` (sanity check)**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (CLAUDE.md changes don't affect tests, but verify nothing broke).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document add_spatial_index_with_lookup in CLAUDE.md"
```

---

## Final Verification

- [ ] **Run full test suite:** `cargo test -p minkowski`
- [ ] **Run clippy:** `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] **Run planner example:** `cargo run -p minkowski-examples --example planner --release`
- [ ] **Verify test count:** Should be 625+ (619 base + 6 new spatial execution tests + regression test from PR #98)
