# BTree/Hash Index Execution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing BTree/Hash index lookup functions into the query execution engine so that `IndexLookup`/`IndexGather` plan nodes invoke the registered index at runtime instead of falling back to a full archetype scan.

**Architecture:** An `IndexDriver` pre-binds the index's type-erased lookup function with the predicate's lookup value into a parameterless closure at Phase 3. Phase 7/8 check for the index driver (after the spatial driver check) and compile an index-gather closure with the same validation pipeline: `is_alive` → location → required → `Changed<T>` → `filter_fns`.

**Tech Stack:** Rust, minkowski ECS (planner.rs)

**Spec:** `docs/plans/2026-03-14-index-execution-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/src/planner.rs` | Modify | `IndexDriver`, Phase 3 population, Phase 7/8 index-gather, tests |
| `examples/examples/planner.rs` | Modify | Index execution demo section |
| `CLAUDE.md` | Modify | Document index execution |

---

## Task 1: `IndexDriver` struct + Phase 3 population

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1418` (next to SpatialDriver)
- Modify: `crates/minkowski/src/planner.rs:1623-1645` (Phase 3 driver creation)
- Modify: `crates/minkowski/src/planner.rs:339` (remove dead_code from lookup_value)

- [ ] **Step 1: Define `IndexDriver` next to `SpatialDriver`**

After `SpatialDriver` (line ~1421), add:

```rust
/// Carries a pre-bound index lookup function from Phase 3 (driver selection)
/// to Phase 7 (join collectors) and Phase 8 (closure compilation).
///
/// The lookup function and predicate value are bound together at construction
/// time, so the execution path never sees `dyn Any`.
struct IndexDriver {
    lookup_fn: IndexLookupFn,
}
```

- [ ] **Step 2: Remove `#[expect(dead_code)]` from `Predicate::lookup_value`**

At line ~339, remove the `#[expect(dead_code)]` annotation and update the doc comment:

```rust
    /// Type-erased predicate value for index lookups. Eq stores `Arc<T>`,
    /// Range stores `Arc<(Bound<T>, Bound<T>)>`. Bound into the `IndexDriver`
    /// lookup closure at Phase 3 plan-build time.
    lookup_value: Option<Arc<dyn std::any::Any + Send + Sync>>,
```

- [ ] **Step 3: Populate `index_driver` in Phase 3 of `build()`**

After the spatial driver creation (line ~1645, after the closing of `let spatial_driver = ...;`), add:

```rust
// Compute index driver — pre-binds lookup fn + value for Phase 7/8.
// Only activates when no spatial driver is present and the best driving
// access is an index predicate with a registered lookup function.
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

- [ ] **Step 4: Run `cargo check` and `cargo test -p minkowski --lib`**

Run: `cargo check --workspace --quiet && cargo test -p minkowski --lib`
Expected: All 633 tests pass. `index_driver` is unused (compiler warning is OK — Task 2 will consume it).

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Add IndexDriver and populate in Phase 3 with pre-bound lookup"
```

---

## Task 2: Phase 8 index-gather closure

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1908-1964` (Phase 8 compiled_for_each)
- Modify: `crates/minkowski/src/planner.rs:1968-2027` (Phase 8 compiled_for_each_raw)

- [ ] **Step 1: Write failing test `index_for_each_uses_btree_lookup`**

Add to the test module at the end of `planner.rs`:

```rust
#[test]
fn index_for_each_uses_btree_lookup() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));
    let _e3 = world.spawn((Score(99),));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    // Wrap in Arc to track call count via the btree's get method.
    let btree = Arc::new(btree);
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);
    let btree_for_lookup = Arc::clone(&btree);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&btree, &world);

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    // Without index execution, for_each scans all archetypes.
    // With it, the btree lookup returns only e1 and e2.
    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| {
        results.push(entity);
    });

    // Both entities with Score(42) should be returned, not Score(99).
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}
```

Note: this test will pass even without index execution (because the filter closure catches Score(99)). To truly verify the index path is used, we need to check that fewer entities are examined. But the structural test is still valuable. A more precise test can check via `explain()` that the plan has `IndexLookup` as driver.

- [ ] **Step 2: Extend Phase 8 `compiled_for_each` with index driver branch**

Change the Phase 8 structure from:

```rust
if let Some(ref driver) = spatial_driver {
    // spatial index-gather
} else {
    // archetype scan
}
```

To:

```rust
if let Some(ref driver) = spatial_driver {
    // spatial index-gather (unchanged)
} else if let Some(ref driver) = index_driver {
    // index-gather — same validation pipeline, different candidate source
    let lookup_fn = Arc::clone(&driver.lookup_fn);
    let required = required_for_index;
    let changed = changed_for_index;
    Some(Box::new(
        move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
            let candidates = lookup_fn();
            for &entity in candidates.iter() {
                if !world.is_alive(entity) {
                    continue;
                }
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
                if !changed.is_clear()
                    && !passes_change_filter(arch, &changed, tick)
                {
                    continue;
                }
                if all_filter_fns.iter().all(|f| f(world, entity)) {
                    callback(entity);
                }
            }
        },
    ) as CompiledForEach)
} else {
    // archetype scan (unchanged)
}
```

**Important:** The `required_for_index` and `changed_for_index` variables are consumed by the spatial branch (moved into the closure). When the spatial branch is not taken but the index branch is, these need to be available. The current code does:

```rust
let required_for_index = self.required_for_spatial.take()...;
let changed_for_index = self.changed_for_spatial.take()...;
```

These are taken once. The spatial branch captures them by move. The index branch needs them too. Since only one branch executes (if/else if/else), the compiler should be OK — but Rust's move checker may not see this. If it complains, clone the values before the branch:

```rust
let required_for_index_clone = required_for_index.clone();
let changed_for_index_clone = changed_for_index.clone();
```

Then use the originals in the spatial branch and the clones in the index branch.

- [ ] **Step 3: Do the same for `compiled_for_each_raw`**

Same three-way branch: spatial → index → archetype scan. Use `required_for_index_raw`, `changed_for_index_raw`, `all_filter_fns_raw`.

- [ ] **Step 4: Run test and full suite**

Run: `cargo test -p minkowski --lib -- index_for_each_uses_btree_lookup && cargo test -p minkowski --lib`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Compile index-gather closure in Phase 8 when index driver is present"
```

---

## Task 3: Phase 7 join collector

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1790-1830` (Phase 7 left_collector)

- [ ] **Step 1: Write failing test `index_join_uses_lookup`**

```rust
#[test]
fn index_join_uses_lookup() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 }));
    let e2 = world.spawn((Score(42), Pos { x: 2.0, y: 2.0 }));
    let _e3 = world.spawn((Score(99), Pos { x: 3.0, y: 3.0 }));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world);

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .join::<(&Pos,)>(JoinKind::Inner)
        .build();

    let results = plan.execute(&mut world);
    // Only e1 and e2 match Score(42) and have Pos.
    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}
```

- [ ] **Step 2: Extend Phase 7 left_collector with index driver branch**

Change the left_collector from:

```rust
if let Some(ref driver) = spatial_driver {
    // spatial index-gather
} else {
    // archetype scan
}
```

To:

```rust
if let Some(ref driver) = spatial_driver {
    // spatial index-gather (unchanged)
} else if let Some(ref driver) = index_driver {
    // index-gather — same validation pipeline
    let lookup_fn = Arc::clone(&driver.lookup_fn);
    let left_changed_for_index = left_changed.clone();
    let left_required_for_index = left_required.clone();
    Box::new(
        move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
            let candidates = lookup_fn();
            for &entity in candidates.iter() {
                if !world.is_alive(entity) {
                    continue;
                }
                let idx = entity.index() as usize;
                let Some(loc) = (idx < world.entity_locations.len())
                    .then(|| world.entity_locations[idx].as_ref())
                    .flatten()
                else {
                    continue;
                };
                let arch = &world.archetypes.archetypes[loc.archetype_id.0];
                if !left_required_for_index.is_subset(&arch.component_ids) {
                    continue;
                }
                if !left_changed_for_index.is_clear()
                    && !passes_change_filter(arch, &left_changed_for_index, tick)
                {
                    continue;
                }
                if left_filters.iter().all(|f| f(world, entity)) {
                    scratch.push(entity);
                }
            }
        },
    )
} else {
    // archetype scan (unchanged)
}
```

Note: `left_required` and `left_changed` are consumed by the else-branch (archetype scan). Clone them for the index branch before the if/else chain if the compiler complains about moves.

- [ ] **Step 3: Run test and full suite**

Run: `cargo test -p minkowski --lib -- index_join_uses_lookup && cargo test -p minkowski --lib`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Wire index driver into join left-side collector"
```

---

## Task 4: Execution tests

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (test module)

- [ ] **Step 1: Write `index_for_each_uses_hash_lookup`**

```rust
#[test]
fn index_for_each_uses_hash_lookup() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));
    let _e3 = world.spawn((Score(99),));

    let mut hash = HashIndex::<Score>::new();
    hash.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_hash_index(&Arc::new(hash), &world);

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));

    assert_eq!(results.len(), 2);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
}
```

- [ ] **Step 2: Write `index_range_lookup_execution`**

```rust
#[test]
fn index_range_lookup_execution() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));
    let e3 = world.spawn((Score(30),));
    let _e4 = world.spawn((Score(100),));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world);

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(5)..Score(35)))
        .build();

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));

    assert_eq!(results.len(), 3);
    assert!(results.contains(&e1));
    assert!(results.contains(&e2));
    assert!(results.contains(&e3));
}
```

- [ ] **Step 3: Write `index_lookup_filters_stale_entities`**

```rust
#[test]
fn index_lookup_filters_stale_entities() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world);

    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    // Despawn e2 after plan is built — index is stale.
    world.despawn(e2);

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}
```

- [ ] **Step 4: Write `index_lookup_filters_missing_required`**

```rust
#[test]
fn index_lookup_filters_missing_required() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 })); // has both
    let _e2 = world.spawn((Score(42),)); // only Score

    let mut btree = BTreeIndex::<Score>::new();
    btree.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(btree), &world);

    // Query requires BOTH Score and Pos.
    let mut plan = planner
        .scan::<(&Score, &Pos)>()
        .filter(Predicate::eq::<Score>(Score(42)))
        .build();

    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));

    assert_eq!(results.len(), 1, "entity missing Pos should be filtered");
    assert_eq!(results[0], e1);
}
```

- [ ] **Step 5: Run all new + existing tests**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: 639+ tests pass (633 existing + 6 new), clippy clean.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Add BTree/Hash index execution tests (eq, hash, range, stale, required, join)"
```

---

## Task 5: CLAUDE.md + example update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `examples/examples/planner.rs`

- [ ] **Step 1: Update CLAUDE.md Query Planner section**

After the existing paragraph about `add_spatial_index_with_lookup`, add:

```markdown
BTree and Hash index lookup functions (`eq_lookup_fn`, `range_lookup_fn`) captured at `add_btree_index` / `add_hash_index` registration are invoked at execution time when the index is chosen as the driving access. The predicate's `lookup_value` is pre-bound into the lookup closure at Phase 3 plan-build time (`IndexDriver`), so the execution path never handles `dyn Any`. Same validation pipeline as spatial: `is_alive` → archetype location → required components → `Changed<T>` → filter refinement.
```

- [ ] **Step 2: Add index execution section to planner example**

Read the existing planner example to find the right insertion point. Add a section demonstrating:

1. BTree index driving an eq lookup via `execute()`
2. Printing how many entities the index returns vs full scan
3. EXPLAIN showing `IndexGather`

This should be brief — the spatial section already demonstrates the pattern.

- [ ] **Step 3: Run example and full suite**

Run: `cargo run -p minkowski-examples --example planner --release && cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: Example runs, all tests pass, clippy clean.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md examples/examples/planner.rs
git commit -m "Document index execution in CLAUDE.md and planner example"
```

---

## Final Verification

- [ ] **Run full test suite:** `cargo test -p minkowski`
- [ ] **Run clippy:** `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] **Run planner example:** `cargo run -p minkowski-examples --example planner --release`
- [ ] **Verify test count:** Should be 639+ (633 base + 6 new index execution tests)
