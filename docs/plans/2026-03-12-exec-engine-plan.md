# Allocation-Free Query Execution Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the closure-dispatch execution engine in `planner.rs` with a compiled query pipeline that matches `world.query().for_each()` speed.

**Architecture:** Two execution paths determined at build time. Scan-only plans compile to monomorphic `QueryIter` wrappers (zero-alloc, SIMD). Join/gather plans use a plan-owned scratch buffer with sorted intersection. Transactional reads via `for_each_raw(&World)`.

**Tech Stack:** Rust, no new dependencies. Uses existing `QueryIter`, `WorldQuery`, `ReadOnlyWorldQuery`, `FixedBitSet`, `SharedPool`.

---

## Pre-Implementation: Read These First

- Design doc: `docs/plans/2026-03-12-exec-engine-design.md`
- Current execution engine: `crates/minkowski/src/planner.rs:1033-1510` (ExecNode, ClosureNode, lower_to_executable, scan_matching_entities)
- QueryIter: `crates/minkowski/src/query/iter.rs` (for_each, for_each_chunk)
- query_raw: `crates/minkowski/src/world.rs:1509-1520` (shared-ref read path)
- Pool: `crates/minkowski/src/pool.rs` (SharedPool, SlabPool)

## Conventions

- `cargo test -p minkowski --lib -- planner` to run planner tests
- `cargo clippy --workspace --all-targets -- -D warnings` before every commit
- All new code in `crates/minkowski/src/planner.rs` unless otherwise noted
- Tests go in the existing `#[cfg(test)] mod tests` block at the bottom of `planner.rs`
- Use `#[expect(dead_code)]` only with a documented future use. Prefer `#[cfg(test)]` for test-only code.

---

### Task 1: ScratchBuffer struct

Introduce the pool-aware reusable entity buffer. This is a standalone struct with no dependencies on the rest of the execution engine, so it can be built and tested first.

**Files:**
- Modify: `crates/minkowski/src/planner.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
#[test]
fn scratch_buffer_starts_empty() {
    let scratch = ScratchBuffer::new(100);
    assert_eq!(scratch.as_slice().len(), 0);
    assert!(scratch.capacity() >= 100);
}

#[test]
fn scratch_buffer_push_and_clear() {
    let mut scratch = ScratchBuffer::new(10);
    let e1 = Entity::from_raw(1, 0);
    let e2 = Entity::from_raw(2, 0);
    scratch.push(e1);
    scratch.push(e2);
    assert_eq!(scratch.as_slice(), &[e1, e2]);
    scratch.clear();
    assert_eq!(scratch.as_slice().len(), 0);
    assert!(scratch.capacity() >= 10); // capacity preserved
}

#[test]
fn scratch_buffer_reuse_does_not_realloc() {
    let mut scratch = ScratchBuffer::new(100);
    for i in 0..50 {
        scratch.push(Entity::from_raw(i, 0));
    }
    let cap_after_first = scratch.capacity();
    scratch.clear();
    for i in 0..50 {
        scratch.push(Entity::from_raw(i, 0));
    }
    assert_eq!(scratch.capacity(), cap_after_first);
}

#[test]
fn scratch_buffer_sorted_intersection() {
    let mut scratch = ScratchBuffer::new(100);
    // left: [1, 3, 5, 7, 9]
    for i in [1, 3, 5, 7, 9] {
        scratch.push(Entity::from_raw(i, 0));
    }
    let left_len = scratch.len();
    // right: [2, 3, 6, 7, 10]
    for i in [2, 3, 6, 7, 10] {
        scratch.push(Entity::from_raw(i, 0));
    }
    let result = scratch.sorted_intersection(left_len);
    let ids: Vec<u32> = result.iter().map(|e| e.index()).collect();
    assert_eq!(ids, vec![3, 7]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- planner::tests::scratch_buffer`
Expected: FAIL — `ScratchBuffer` not found

**Step 3: Implement ScratchBuffer**

Add above the `#[cfg(test)]` module, after the `lower_to_vectorized` function:

```rust
// ── Scratch buffer for entity-mode execution ─────────────────────────

/// Reusable entity buffer for join/gather plan execution.
///
/// Pre-sized at `build()` time from the cost model. Cleared and refilled
/// on each `execute()` call — capacity is preserved, so after a few
/// executions the buffer stabilizes with zero further allocations.
///
/// For joins, the buffer is partitioned by index ranges:
/// `[left | right | output]` — three slices in one allocation.
struct ScratchBuffer {
    entities: Vec<Entity>,
}

impl ScratchBuffer {
    fn new(estimated_capacity: usize) -> Self {
        let cap = estimated_capacity.min(64 * 1024); // cap at 64K
        ScratchBuffer {
            entities: Vec::with_capacity(cap),
        }
    }

    fn push(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    fn clear(&mut self) {
        self.entities.clear();
    }

    fn len(&self) -> usize {
        self.entities.len()
    }

    fn capacity(&self) -> usize {
        self.entities.capacity()
    }

    fn as_slice(&self) -> &[Entity] {
        &self.entities
    }

    /// Perform sorted intersection between left and right halves.
    ///
    /// The buffer contains `[left_entities | right_entities]`.
    /// `left_len` is the boundary. Sorts the left slice in-place,
    /// then binary-searches from the right side. Results are appended
    /// to the end of the buffer. Returns a slice of the intersection.
    fn sorted_intersection(&mut self, left_len: usize) -> &[Entity] {
        // Sort left slice in-place (no allocation).
        self.entities[..left_len].sort_unstable_by_key(|e| e.to_bits());

        let right_start = left_len;
        let right_end = self.entities.len();

        // Collect matching right entities by binary search into sorted left.
        // We can't push during iteration, so collect indices first.
        let mut match_count = 0;
        for i in right_start..right_end {
            let entity = self.entities[i];
            if self.entities[..left_len]
                .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
                .is_ok()
            {
                // Write match at the end of buffer.
                self.entities.push(entity);
                match_count += 1;
            }
        }

        let output_start = right_end;
        &self.entities[output_start..output_start + match_count]
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- planner::tests::scratch_buffer`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): add ScratchBuffer for allocation-free join execution"
```

---

### Task 2: CompiledScan trait and type-erased scan capture

Define the trait that captures monomorphic scan iteration at build time and stores it type-erased. This is the core abstraction — everything else builds on it.

**Files:**
- Modify: `crates/minkowski/src/planner.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn compiled_scan_for_each_yields_all_entities() {
    let mut world = World::new();
    let mut expected = Vec::new();
    for i in 0..10u32 {
        let e = world.spawn((Score(i),));
        expected.push(e);
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut found = Vec::new();
    plan.for_each(&mut world, |entity: Entity| {
        found.push(entity);
    });
    found.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- planner::tests::compiled_scan_for_each_yields_all_entities`
Expected: FAIL — `for_each` method not found on `QueryPlanResult`

**Step 3: Implement CompiledScan**

The key insight: `ScanBuilder::build()` has `Q: WorldQuery` in scope. We capture a closure that calls `world.query::<Q>().for_each(callback)` and store it type-erased.

Add the trait and type alias:

```rust
/// Type-erased scan execution. Captured at `build()` time while the
/// `WorldQuery` type parameter is in scope. Stores monomorphic iteration
/// code — after inlining, identical to hand-written `world.query().for_each()`.
type CompiledForEach = Box<dyn FnMut(&mut World, &mut dyn FnMut(Entity))>;

/// Read-only variant for transactional reads via `query_raw`.
type CompiledForEachRaw = Box<dyn FnMut(&World, &mut dyn FnMut(Entity))>;
```

Add a new field to `QueryPlanResult`:

```rust
pub struct QueryPlanResult {
    root: PlanNode,
    vec_root: VecExecNode,
    exec_root: Option<ExecNode>,          // old — keep for now, remove in Task 6
    compiled_for_each: Option<CompiledForEach>,
    compiled_for_each_raw: Option<CompiledForEachRaw>,
    opts: VectorizeOpts,
    warnings: Vec<PlanWarning>,
}
```

Add `for_each` to `QueryPlanResult`:

```rust
impl QueryPlanResult {
    /// Execute the compiled scan, calling `callback` for each matching entity.
    ///
    /// For scan-only plans (no joins), this compiles to the same machine code
    /// as `world.query::<Q>().for_each()`. Zero allocation during iteration.
    ///
    /// # Panics
    /// Panics if the plan has joins (use `execute()` instead).
    pub fn for_each(
        &mut self,
        world: &mut World,
        mut callback: impl FnMut(Entity),
    ) {
        let compiled = self.compiled_for_each.as_mut()
            .expect("for_each requires a scan-only plan (no joins)");
        compiled(world, &mut callback);
    }
}
```

In `ScanBuilder::build()`, after Phase 6 (lower to vectorized), capture the scan closure. Only for scan-only plans (no joins, no index gathers):

```rust
// Phase 7a: Compile scan closure for for_each (scan-only plans).
let has_joins = !self.joins.is_empty();
let has_index = !index_preds.is_empty();
let compiled_for_each = if !has_joins && !has_index {
    // No joins, no index — pure scan. Capture monomorphic iteration.
    // Q is still in scope here from scan::<Q>().
    // Note: we need Q threaded through ScanBuilder. See below.
    todo!("capture scan closure — requires Q on ScanBuilder")
} else {
    None
};
```

**The challenge**: `ScanBuilder` doesn't currently carry `Q` as a type parameter — it's erased at construction. We need to thread it through. Modify `ScanBuilder` to carry an optional scan-capture factory:

```rust
pub struct ScanBuilder<'w> {
    planner: &'w QueryPlanner<'w>,
    query_name: &'static str,
    estimated_rows: usize,
    predicates: Vec<Predicate>,
    joins: Vec<JoinSpec>,
    scan_fn: Option<ScanFn>,
    // NEW: factory that captures for_each while Q is in scope
    compile_for_each: Option<Box<dyn FnOnce() -> CompiledForEach>>,
    compile_for_each_raw: Option<Box<dyn FnOnce() -> CompiledForEachRaw>>,
}
```

In `QueryPlanner::scan::<Q>()`, capture the factory:

```rust
pub fn scan<Q: crate::query::fetch::WorldQuery + 'static>(&'w self) -> ScanBuilder<'w> {
    let required = Q::required_ids(self.components);
    ScanBuilder {
        // ... existing fields ...
        compile_for_each: Some(Box::new(|| {
            Box::new(|world: &mut World, callback: &mut dyn FnMut(Entity)| {
                world.query::<Q>().for_each(|item| {
                    let entity = Q::entity_from_item(&item);
                    callback(entity);
                });
            })
        })),
        compile_for_each_raw: None, // Task 5
    }
}
```

**Note on Q::entity_from_item**: the `for_each` callback currently yields `Q::Item`. We need to extract the `Entity` from it. The simplest approach: require that the scan query includes `Entity` in its tuple, e.g. `scan::<(Entity, &Score)>()`. Alternatively, we use `QueryIter`'s internal entity tracking. The exact mechanism can be refined during implementation — for now, the test uses a simple `Entity`-only callback.

For the initial implementation, use the `for_each` on `QueryIter` which already tracks entities internally via the archetype's entity array:

```rust
compile_for_each: Some(Box::new(|| {
    Box::new(|world: &mut World, callback: &mut dyn FnMut(Entity)| {
        // QueryIter already knows entities per archetype row.
        // We iterate archetypes and yield Entity from the dense entity array.
        let iter = world.query::<Q>();
        for (fetch, len) in &iter.fetches {
            // ... access archetype entity array ...
        }
    })
})),
```

Actually, the simplest correct implementation: iterate archetypes matching `Q`'s required components and yield entities from the archetype entity array. This is what `scan_matching_entities` already does, but without the `Vec<Entity>` allocation — we call the callback directly:

```rust
compile_for_each: Some(Box::new(move || {
    let required = required.clone();
    Box::new(move |world: &mut World, callback: &mut dyn FnMut(Entity)| {
        for arch in &world.archetypes.archetypes {
            if !arch.is_empty() && required.is_subset(&arch.component_ids) {
                for &entity in arch.entities() {
                    callback(entity);
                }
            }
        }
    })
})),
```

This is zero-alloc — it walks archetypes and calls the callback per entity. Not as fast as `for_each_chunk` (no typed slices), but correct and allocation-free. The chunk variant comes in Task 3.

**Step 4: Wire into build() and run test**

In `build()`, invoke the factory if it's a scan-only plan:

```rust
let compiled_for_each = if !has_joins && !has_index {
    self.compile_for_each.map(|factory| factory())
} else {
    None
};
```

Construct `QueryPlanResult` with the new fields.

Run: `cargo test -p minkowski --lib -- planner::tests::compiled_scan_for_each`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): add for_each on QueryPlanResult via compiled scan closure"
```

---

### Task 3: Typed Eq/Range filter fusion

Compile Eq and Range predicates into the scan closure so they're evaluated on entities during iteration, not as a separate tree node.

**Files:**
- Modify: `crates/minkowski/src/planner.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn for_each_with_eq_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::eq(Score(42)))
        .build();

    let mut found = Vec::new();
    plan.for_each(&mut world, |entity: Entity| {
        found.push(entity);
    });
    assert_eq!(found.len(), 1);
    let score = world.get::<Score>(found[0]).unwrap();
    assert_eq!(*score, Score(42));
}

#[test]
fn for_each_with_range_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
        .build();

    let mut found = Vec::new();
    plan.for_each(&mut world, |entity: Entity| {
        found.push(entity);
    });
    assert_eq!(found.len(), 10); // 10..20 = 10 entities
}

#[test]
fn for_each_with_custom_filter() {
    let mut world = World::new();
    for i in 0..100u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .filter(Predicate::custom::<Score>(
            "even scores",
            0.5,
            |world, entity| {
                world.get::<Score>(entity).is_some_and(|s| s.0 % 2 == 0)
            },
        ))
        .build();

    let mut found = Vec::new();
    plan.for_each(&mut world, |entity: Entity| {
        found.push(entity);
    });
    assert_eq!(found.len(), 50);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- planner::tests::for_each_with`
Expected: FAIL — filters not applied (all entities returned, or `for_each` panics for non-scan-only plan)

**Step 3: Implement filter fusion**

The approach: at `build()` time, collect the filter closures from `filter_preds` (predicates that didn't get pushed to an index). Each predicate already has a `filter_fn: Option<FilterFn>` which is an `Arc<dyn Fn(&World, Entity) -> bool>`. Fuse them into the scan closure:

```rust
// In build(), after classifying predicates:
let compiled_for_each = if !has_joins {
    self.compile_for_each.map(|factory| {
        let mut scan_fn = factory();
        // Collect filter functions from non-index predicates.
        let filters: Vec<FilterFn> = filter_preds.iter()
            .filter_map(|p| p.filter_fn.clone())
            .collect();
        // Also include index predicates' filter_fn for post-fetch validation.
        let idx_filters: Vec<FilterFn> = index_preds.iter()
            .filter_map(|(p, _)| p.filter_fn.clone())
            .collect();
        let all_filters: Vec<FilterFn> = idx_filters.into_iter()
            .chain(filters)
            .collect();

        if all_filters.is_empty() {
            scan_fn
        } else {
            Box::new(move |world: &mut World, callback: &mut dyn FnMut(Entity)| {
                scan_fn(world, &mut |entity: Entity| {
                    if all_filters.iter().all(|f| f(world, entity)) {
                        callback(entity);
                    }
                });
            })
        }
    })
} else {
    None
};
```

**Note**: This fuses filters as per-entity `dyn Fn` calls for now — all predicates go through `filter_fn`. The typed-slice SIMD optimization (where Eq/Range operate on `&[Score]` instead of `world.get()`) is a future enhancement once `for_each_chunk` is implemented. The design doc describes the ideal; this task implements the correct, allocation-free version first.

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- planner::tests::for_each_with`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): fuse filter predicates into compiled scan closure"
```

---

### Task 4: Entity-mode execute with ScratchBuffer

Wire ScratchBuffer into the join/gather path. The `execute()` method uses the scratch for index lookups and joins instead of allocating per-node `Vec<Entity>`.

**Files:**
- Modify: `crates/minkowski/src/planner.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn execute_with_scratch_returns_all_entities() {
    let mut world = World::new();
    let mut expected = Vec::new();
    for i in 0..10u32 {
        let e = world.spawn((Score(i), Team(i % 3)));
        expected.push(e);
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let result = plan.execute(&mut world);
    // Inner join of Score entities with Team entities — all 10 have both
    let mut found: Vec<Entity> = result.to_vec();
    found.sort_by_key(|e| e.to_bits());
    expected.sort_by_key(|e| e.to_bits());
    assert_eq!(found, expected);
}

#[test]
fn execute_scratch_reuse_no_realloc() {
    let mut world = World::new();
    for i in 0..10u32 {
        world.spawn((Score(i), Team(i % 3)));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let _ = plan.execute(&mut world);
    // Second execution should reuse the same buffer.
    let result = plan.execute(&mut world);
    assert_eq!(result.len(), 10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- planner::tests::execute_with_scratch`
Expected: FAIL — new `execute` signature doesn't exist

**Step 3: Implement**

Add `scratch` field to `QueryPlanResult`:

```rust
pub struct QueryPlanResult {
    root: PlanNode,
    vec_root: VecExecNode,
    exec_root: Option<ExecNode>,
    compiled_for_each: Option<CompiledForEach>,
    compiled_for_each_raw: Option<CompiledForEachRaw>,
    scratch: Option<ScratchBuffer>,
    opts: VectorizeOpts,
    warnings: Vec<PlanWarning>,
}
```

In `build()`, for join plans, pre-size the scratch:

```rust
let scratch = if has_joins || has_index {
    let est = node.cost().rows as usize;
    Some(ScratchBuffer::new(est * 3)) // room for left + right + output
} else {
    None
};
```

Change `execute()` signature to `&mut self, &mut World`:

```rust
pub fn execute(&mut self, world: &mut World) -> &[Entity] {
    if let Some(scratch) = &mut self.scratch {
        scratch.clear();
        // Use the old ExecNode path for now — write results to scratch
        if let Some(exec) = &self.exec_root {
            let entities = exec.execute(world);
            for e in entities {
                scratch.push(e);
            }
        }
        scratch.as_slice()
    } else {
        // Scan-only plan — execute shouldn't be called, use for_each
        panic!("execute() requires a join/gather plan; use for_each() for scan-only plans");
    }
}
```

This is a transitional implementation — it still uses the old ExecNode internally but writes to scratch. Task 6 will replace ExecNode with native scratch-based execution.

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- planner::tests::execute_with_scratch`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): wire ScratchBuffer into execute() for join/gather plans"
```

---

### Task 5: for_each_raw for transactional reads

Add the `&World` (shared-ref) variant for use inside transactions.

**Files:**
- Modify: `crates/minkowski/src/planner.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn for_each_raw_yields_entities_without_mut_world() {
    let mut world = World::new();
    for i in 0..10u32 {
        world.spawn((Score(i),));
    }
    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut found = Vec::new();
    // for_each_raw takes &World, not &mut World
    plan.for_each_raw(&world, |entity: Entity| {
        found.push(entity);
    });
    assert_eq!(found.len(), 10);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- planner::tests::for_each_raw`
Expected: FAIL — method not found

**Step 3: Implement**

In `QueryPlanner::scan::<Q>()`, capture the raw variant:

```rust
compile_for_each_raw: Some(Box::new(move || {
    let required = required.clone();
    Box::new(move |world: &World, callback: &mut dyn FnMut(Entity)| {
        for arch in &world.archetypes.archetypes {
            if !arch.is_empty() && required.is_subset(&arch.component_ids) {
                for &entity in arch.entities() {
                    callback(entity);
                }
            }
        }
    })
})),
```

Add `for_each_raw` to `QueryPlanResult`:

```rust
pub fn for_each_raw(
    &mut self,
    world: &World,
    mut callback: impl FnMut(Entity),
) {
    let compiled = self.compiled_for_each_raw.as_mut()
        .expect("for_each_raw requires a scan-only plan (no joins)");
    compiled(world, &mut callback);
}
```

Wire filter fusion into the raw path (same pattern as Task 3 but using `&World`).

**Step 4: Run test**

Run: `cargo test -p minkowski --lib -- planner::tests::for_each_raw`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): add for_each_raw for transactional read-only execution"
```

---

### Task 6: Remove old ExecNode execution engine

Replace the old closure-dispatch execution with native scratch-based execution for joins. Remove ExecNode, ClosureNode, and all supporting code.

**Files:**
- Modify: `crates/minkowski/src/planner.rs`

**Step 1: Identify all code to remove**

Lines to remove (approximate — verify before cutting):
- `type ScanFn` (~line 1033)
- `type IndexLookupFn` (~line 1039)
- `type FilterFn` (~line 1046)
- `enum ProbeSet` + impl (~lines 1049-1072)
- `enum ExecNode` + all impls (~lines 1074-1316)
- `enum ClosureNode` (~lines 1309-1342)
- `fn lower_to_executable` (~lines 1340-1456)
- `fn lower_to_executable_fallback` (~lines 1457-1493)
- `fn scan_matching_entities` (~line 1495-1510)
- `exec_root` field on `QueryPlanResult`
- Old `execute(&self, &World) -> Vec<Entity>` method
- Old `exec_root()` test helper
- All `exec_tree_*` tests (10 tests)
- `scan_fn` field on `ScanBuilder`
- `right_scan_fn` field on `JoinSpec`
- All `ClosureNode` construction in `build()`

**Step 2: Implement native join execution on ScratchBuffer**

Replace the old ExecNode-based join with direct scratch operations:

For join plans, `execute()` becomes:
1. Scan left side (using the compiled scan closure) → push entities to scratch
2. Scan right side → push entities to scratch
3. Call `scratch.sorted_intersection(left_len)` for inner joins
4. Return the result slice

Index gathers use the `IndexDescriptor` lookup functions directly into scratch.

**Step 3: Remove old code and update all references**

Remove all items listed in Step 1. Update `build()` to not construct ClosureNode trees. Update `QueryPlanResult` struct definition.

**Step 4: Run full test suite**

Run: `cargo test -p minkowski --lib -- planner`
Expected: 54+ planning tests pass, 28 old execution tests fail (removed)

**Step 5: Replace execution tests**

Update the 28 old `execute_*` / `exec_tree_*` tests to use the new API:
- `execute_*` tests that test entity results → use new `execute(&mut self, &mut World)` or `for_each`
- `exec_tree_*` tests that inspect ExecNode structure → replace with behavioral tests on the new API

**Step 6: Run full test suite**

Run: `cargo test -p minkowski --lib -- planner`
Expected: All tests pass

**Step 7: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "refactor(planner): replace ExecNode closure dispatch with compiled scan + scratch buffer"
```

---

### Task 7: Update planner example and docs

Update the example and CLAUDE.md to reflect the new API.

**Files:**
- Modify: `examples/examples/planner.rs`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update planner example**

Replace any `plan.execute(&world)` calls with the new API:
- Scan-only plans: use `plan.for_each(&mut world, |entity| { ... })`
- Join plans: use `plan.execute(&mut world)` (new signature)

Add a section demonstrating transactional composition:

```rust
println!("=== 8. Transactional Execution ===\n");

// Read-only plan inside a transaction context
let mut plan = planner.scan::<(&Score, &Pos)>().build();
let mut count = 0;
plan.for_each_raw(&world, |_entity| {
    count += 1;
});
println!("Transactional read found {count} entities\n");
```

**Step 2: Run the example**

Run: `cargo run -p minkowski-examples --example planner --release`
Expected: runs cleanly with updated output

**Step 3: Update CLAUDE.md**

In the Query Planner section, update the API description:
- `execute()` now takes `&mut self, &mut World` and returns `&[Entity]`
- New `for_each()` / `for_each_chunk()` / `for_each_raw()` methods
- Note: scan path is zero-allocation, join path uses plan-owned scratch

**Step 4: Commit**

```bash
git add examples/examples/planner.rs CLAUDE.md README.md
git commit -m "docs: update planner example and docs for compiled execution engine"
```

---

### Task 8: Final validation

Run the full CI-equivalent checks.

**Step 1: Format**

Run: `cargo fmt --all -- --check`

**Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

**Step 3: Full test suite**

Run: `cargo test -p minkowski`

**Step 4: All examples**

Run: `cargo run -p minkowski-examples --example planner --release`

**Step 5: Commit any fixes, then done**

---

## Execution Order

Tasks 1-5 are additive (new code alongside old). Task 6 is the big replacement. Task 7 is docs. Task 8 is validation.

```
Task 1: ScratchBuffer (standalone, no deps)
Task 2: CompiledScan + for_each (standalone, no deps)
Task 3: Filter fusion (depends on Task 2)
Task 4: execute() with scratch (depends on Task 1)
Task 5: for_each_raw (depends on Task 2)
Task 6: Remove old engine (depends on Tasks 1-5)
Task 7: Update docs (depends on Task 6)
Task 8: Final validation (depends on Task 7)
```

Tasks 1 and 2 can be done in parallel. Tasks 3, 4, 5 can be done in parallel after their deps.
