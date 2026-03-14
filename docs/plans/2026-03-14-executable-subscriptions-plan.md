# Executable Subscriptions Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SubscriptionBuilder` produce executable `QueryPlanResult` plans by wrapping `ScanBuilder`, accepting `Predicate` objects with `Indexed<T>` witnesses, and removing the plan-only `SubscriptionPlan` type.

**Architecture:** `SubscriptionBuilder` becomes a validation wrapper around `ScanBuilder`. `where_eq`/`where_range` validate the `Indexed<T>` witness and delegate `Predicate` to the inner `ScanBuilder`. `build()` validates and delegates to `ScanBuilder::build()` which runs Phase 1-8 including `IndexDriver` creation. `SubscriptionPlan` and `IndexedPredicate` are removed.

**Tech Stack:** Rust, minkowski ECS (planner.rs, lib.rs)

**Spec:** `docs/plans/2026-03-14-executable-subscriptions-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/src/planner.rs` | Modify | Restructure `SubscriptionBuilder`, remove `SubscriptionPlan`/`IndexedPredicate`, update `subscribe()`, update tests |
| `crates/minkowski/src/lib.rs` | Modify | Remove `SubscriptionPlan` re-export |
| `examples/examples/planner.rs` | Modify | Update subscription example to use `Predicate` + execution |
| `CLAUDE.md` | Modify | Update Query Planner section |

---

## Task 1: Restructure `SubscriptionBuilder` to wrap `ScanBuilder`

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1160-1340` (SubscriptionBuilder, SubscriptionError, IndexedPredicate, SubscriptionPlan)
- Modify: `crates/minkowski/src/planner.rs:2644-2652` (subscribe())
- Modify: `crates/minkowski/src/lib.rs:105` (re-exports)

- [ ] **Step 1: Update `SubscriptionBuilder` struct**

Replace the current struct (lines 1168-1174):

```rust
pub struct SubscriptionBuilder<'w> {
    total_entities: usize,
    query_name: &'static str,
    indexed_predicates: Vec<IndexedPredicate>,
    errors: Vec<SubscriptionError>,
    _world: PhantomData<&'w World>,
}
```

With:

```rust
/// Builder for subscription queries that enforces every predicate has an index.
///
/// Unlike `ScanBuilder`, this uses the type system to guarantee that the
/// resulting plan can push updates without full table scans. Every call to
/// `where_eq` or `where_range` requires an `Indexed<T>` witness, which can
/// only be obtained from an actual index instance.
///
/// Produces a [`QueryPlanResult`] with full execution support (`execute`,
/// `for_each`, `for_each_raw`), backed by `IndexDriver` for index-gather
/// execution.
pub struct SubscriptionBuilder<'w> {
    scan: ScanBuilder<'w>,
    errors: Vec<SubscriptionError>,
    has_predicates: bool,
}
```

- [ ] **Step 2: Update `SubscriptionError` enum**

Replace `NanSelectivity` with `PredicateKindMismatch` and add `ComponentMismatch`:

```rust
pub enum SubscriptionError {
    /// `where_range` was called with a Hash index witness.
    HashIndexOnRange { component_name: &'static str },
    /// Predicate kind does not match the method (e.g., Range predicate
    /// passed to `where_eq`).
    PredicateKindMismatch {
        expected: &'static str,
        component_name: &'static str,
    },
    /// Predicate's component type does not match the `Indexed<T>` witness.
    ComponentMismatch {
        witness_type: &'static str,
        predicate_type: &'static str,
    },
    /// No predicates were added.
    NoPredicates,
}
```

Update the `Display` impl to handle the new/changed variants. Remove `NanSelectivity`.

- [ ] **Step 3: Remove `IndexedPredicate` struct**

Delete the `IndexedPredicate` struct (lines 1219-1224) entirely.

- [ ] **Step 4: Rewrite `where_eq` and `where_range`**

```rust
impl SubscriptionBuilder<'_> {
    /// Add an equality predicate backed by a proven index.
    ///
    /// The `Indexed<T>` witness guarantees at compile time that an index
    /// exists for component `T`. The `Predicate` carries the filter closure
    /// and lookup value for execution.
    pub fn where_eq<T: Component>(mut self, _witness: Indexed<T>, predicate: Predicate) -> Self {
        if predicate.component_type != TypeId::of::<T>() {
            self.errors.push(SubscriptionError::ComponentMismatch {
                witness_type: std::any::type_name::<T>(),
                predicate_type: predicate.component_name,
            });
            return self;
        }
        if !matches!(predicate.kind, PredicateKind::Eq) {
            self.errors.push(SubscriptionError::PredicateKindMismatch {
                expected: "Eq",
                component_name: std::any::type_name::<T>(),
            });
            return self;
        }
        self.has_predicates = true;
        self.scan = self.scan.filter(predicate);
        self
    }

    /// Add a range predicate backed by a proven BTree index.
    ///
    /// # Errors
    ///
    /// Returns [`SubscriptionError::HashIndexOnRange`] if `witness` was
    /// created from a `HashIndex`.
    pub fn where_range<T: Component + Ord + Clone>(
        mut self,
        witness: Indexed<T>,
        predicate: Predicate,
    ) -> Self {
        if witness.kind == IndexKind::Hash {
            self.errors.push(SubscriptionError::HashIndexOnRange {
                component_name: std::any::type_name::<T>(),
            });
            return self;
        }
        if predicate.component_type != TypeId::of::<T>() {
            self.errors.push(SubscriptionError::ComponentMismatch {
                witness_type: std::any::type_name::<T>(),
                predicate_type: predicate.component_name,
            });
            return self;
        }
        if !matches!(predicate.kind, PredicateKind::Range) {
            self.errors.push(SubscriptionError::PredicateKindMismatch {
                expected: "Range",
                component_name: std::any::type_name::<T>(),
            });
            return self;
        }
        self.has_predicates = true;
        self.scan = self.scan.filter(predicate);
        self
    }

    /// Compile the subscription into an executable plan.
    ///
    /// Every predicate is guaranteed to have an index (via `Indexed<T>`
    /// witnesses), so the plan uses `IndexDriver` for index-gather
    /// execution — never a full archetype scan for filtering.
    pub fn build(self) -> Result<QueryPlanResult, Vec<SubscriptionError>> {
        let mut errors = self.errors;
        if !self.has_predicates {
            errors.push(SubscriptionError::NoPredicates);
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(self.scan.build())
    }
}
```

Note: `Predicate.component_type` and `Predicate.kind` are `pub(crate)` fields — accessible within the crate. Verify this before implementing. If they're private, access them via methods or make them `pub(crate)`.

- [ ] **Step 5: Remove `SubscriptionPlan` and its impl blocks**

Delete everything from `pub struct SubscriptionPlan` (line 1343) through its `Debug` impl (line ~1385). This includes:
- `struct SubscriptionPlan`
- `impl SubscriptionPlan` (root, cost, explain)
- `impl fmt::Display for SubscriptionPlan`
- `impl fmt::Debug for SubscriptionPlan`

- [ ] **Step 6: Update `QueryPlanner::subscribe`**

Change (line 2644):

```rust
pub fn subscribe<Q: 'static>(&'w self) -> SubscriptionBuilder<'w> {
    SubscriptionBuilder {
        total_entities: self.total_entities,
        query_name: std::any::type_name::<Q>(),
        indexed_predicates: Vec::new(),
        errors: Vec::new(),
        _world: PhantomData,
    }
}
```

To:

```rust
pub fn subscribe<Q: crate::query::fetch::WorldQuery + 'static>(
    &'w self,
) -> SubscriptionBuilder<'w> {
    SubscriptionBuilder {
        scan: self.scan::<Q>(),
        errors: Vec::new(),
        has_predicates: false,
    }
}
```

- [ ] **Step 7: Remove `SubscriptionPlan` from lib.rs re-exports**

In `lib.rs:105`, remove `SubscriptionPlan` from the re-export list.

- [ ] **Step 8: Run `cargo check` and fix compilation errors**

Run: `cargo check --workspace --quiet`
Expected: Compilation errors in tests and examples that reference `SubscriptionPlan` or old `where_eq(witness, f64)` API. Fix these in the next tasks.

- [ ] **Step 9: Commit structural changes**

```bash
git add crates/minkowski/src/planner.rs crates/minkowski/src/lib.rs
git commit -m "Restructure SubscriptionBuilder to wrap ScanBuilder, remove SubscriptionPlan"
```

---

## Task 2: Update tests

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (test module, lines ~3987-4126)

- [ ] **Step 1: Rewrite `subscription_requires_indexed_witness`**

The test currently calls `.where_eq(witness, 0.001)` and checks `sub.root()` on a `SubscriptionPlan`. Update to use `Predicate::eq` and check the `QueryPlanResult`:

```rust
#[test]
fn subscription_requires_indexed_witness() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx.clone()), &world).unwrap();
    let witness = Indexed::btree(&idx);

    let plan = planner
        .subscribe::<(&Score,)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build()
        .unwrap();

    // Subscription plan has IndexLookup as driving access.
    match plan.root() {
        PlanNode::IndexLookup { index_kind, .. } => {
            assert_eq!(*index_kind, IndexKind::BTree);
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
    assert!(plan.cost().cpu() > 0.0);
}
```

Note: `Predicate::eq` now returns `Result` — use `Predicate::eq(Score(42))` (it was changed in PR #100). Check the exact current signature.

- [ ] **Step 2: Rewrite `subscription_multiple_predicates_ordered_by_selectivity`**

```rust
#[test]
fn subscription_multiple_predicates_ordered_by_selectivity() {
    let mut world = World::new();
    for i in 0..1000 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut score_idx = BTreeIndex::<Score>::new();
    score_idx.rebuild(&mut world);
    let mut team_idx = HashIndex::<Team>::new();
    team_idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(score_idx.clone()), &world).unwrap();
    planner.add_hash_index(&Arc::new(team_idx.clone()), &world).unwrap();
    let score_w = Indexed::btree(&score_idx);
    let team_w = Indexed::hash(&team_idx);

    let plan = planner
        .subscribe::<(&Score, &Team)>()
        .where_eq(team_w, Predicate::eq(Team(2)).with_selectivity(0.2))
        .where_eq(score_w, Predicate::eq(Score(42)).with_selectivity(0.001))
        .build()
        .unwrap();

    // Most selective predicate (Score) should be the driving index lookup.
    fn find_index_lookup(node: &PlanNode) -> Option<&str> {
        match node {
            PlanNode::IndexLookup { component_name, .. } => Some(component_name),
            PlanNode::Filter { child, .. } => find_index_lookup(child),
            _ => None,
        }
    }
    let driver = find_index_lookup(plan.root()).expect("no IndexLookup found");
    assert!(driver.contains("Score"), "expected Score as driver, got {driver}");
}
```

- [ ] **Step 3: Rewrite `subscription_where_range_accepts_btree_witness`**

```rust
#[test]
fn subscription_where_range_accepts_btree_witness() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx.clone()), &world).unwrap();
    let witness = Indexed::btree(&idx);

    let plan = planner
        .subscribe::<(&Score,)>()
        .where_range(witness, Predicate::range::<Score, _>(Score(10)..Score(50)).with_selectivity(0.4))
        .build()
        .unwrap();

    match plan.root() {
        PlanNode::IndexLookup { index_kind, .. } => {
            assert_eq!(*index_kind, IndexKind::BTree);
        }
        other => panic!("expected IndexLookup, got {:?}", other),
    }
}
```

- [ ] **Step 4: Update `subscription_where_range_rejects_hash_witness`**

```rust
#[test]
fn subscription_where_range_rejects_hash_witness() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i),));
    }
    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    let planner = QueryPlanner::new(&world);
    let witness = Indexed::hash(&idx);

    let result = planner
        .subscribe::<(&Score,)>()
        .where_range(witness, Predicate::range::<Score, _>(Score(10)..Score(50)).with_selectivity(0.4))
        .build();
    assert!(matches!(
        result,
        Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::HashIndexOnRange { .. }))
    ));
}
```

- [ ] **Step 5: Replace `subscription_nan_selectivity_returns_error` with new validation tests**

The NaN selectivity test is no longer relevant (selectivity comes from `Predicate` which sanitizes it). Replace with:

```rust
#[test]
fn subscription_no_predicates_returns_error() {
    let mut world = World::new();
    world.spawn((Score(1),));

    let planner = QueryPlanner::new(&world);
    let result = planner.subscribe::<(&Score,)>().build();
    assert!(matches!(
        result,
        Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::NoPredicates))
    ));
}

#[test]
fn subscription_component_mismatch_returns_error() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Score(i), Team(i % 5)));
    }
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let planner = QueryPlanner::new(&world);
    let score_witness = Indexed::btree(&idx);

    // Pass a Team predicate with a Score witness — component mismatch.
    let result = planner
        .subscribe::<(&Score, &Team)>()
        .where_eq(score_witness, Predicate::eq(Team(2)).with_selectivity(0.2))
        .build();
    assert!(matches!(
        result,
        Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::ComponentMismatch { .. }))
    ));
}
```

- [ ] **Step 6: Add executable subscription test**

```rust
#[test]
fn subscription_plan_is_executable() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let _e2 = world.spawn((Score(99),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let mut planner = QueryPlanner::new(&world);
    planner.add_btree_index(&Arc::new(idx.clone()), &world).unwrap();
    let witness = Indexed::btree(&idx);

    let mut plan = planner
        .subscribe::<(&Score,)>()
        .where_eq(witness, Predicate::eq(Score(42)))
        .build()
        .unwrap();

    // The plan is executable — for_each works.
    let mut results = Vec::new();
    plan.for_each(&mut world, |entity| results.push(entity));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], e1);
}
```

- [ ] **Step 7: Run tests**

Run: `cargo test -p minkowski --lib -- subscription && cargo test -p minkowski --lib`
Expected: All subscription tests pass, full suite passes.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "Update subscription tests for executable SubscriptionBuilder API"
```

---

## Task 3: Update planner example + docs

**Files:**
- Modify: `examples/examples/planner.rs:140-178`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update planner example subscription section**

Replace the subscription section (lines ~140-178) to use the new API with actual execution:

```rust
println!("=== 4. Subscription Queries (Compiler-Enforced Indexes) ===\n");

// Every predicate must provide an Indexed<T> witness — no full scans.
let score_witness = Indexed::btree(&score_btree);
let team_witness = Indexed::hash(&team_hash);

let mut sub = planner
    .subscribe::<(&Score, &Team)>()
    .where_eq(score_witness, Predicate::eq(Score(42)).with_selectivity(0.001))
    .where_eq(team_witness, Predicate::eq(Team(2)).with_selectivity(0.2))
    .build()
    .unwrap();

println!("Subscription plan (all predicates indexed):");
println!("{}", sub.explain());

// Subscription plans are executable — for_each uses IndexDriver.
let mut sub_count = 0;
sub.for_each(&mut world, |_| sub_count += 1);
println!("Subscription matched {sub_count} entities\n");
```

Update the constraint validation section too — it currently uses `full_scan.validate_constraints(...)` which is on `QueryPlanResult` and should still work. But the `sub` variable is now `QueryPlanResult` not `SubscriptionPlan`, so `sub.explain()` works directly.

Remove the `SubscriptionPlan` import if present. Add `Predicate` to the import list if not already there.

- [ ] **Step 2: Update CLAUDE.md**

In the Query Planner section, update the `SubscriptionBuilder` documentation. Find the existing mention and update to reflect the new API:

```markdown
`SubscriptionBuilder` wraps `ScanBuilder` with compile-time index enforcement via `Indexed<T>` witnesses. `where_eq(witness, predicate)` and `where_range(witness, predicate)` require an `Indexed<T>` proof that an index exists for the predicate's component. `build()` returns `QueryPlanResult` with full execution support — subscription plans use `IndexDriver` for index-gather execution, never a full archetype scan. The old plan-only `SubscriptionPlan` type has been removed.
```

Also remove `SubscriptionPlan` from the pub API list in CLAUDE.md if it's listed there.

- [ ] **Step 3: Run example + full suite + clippy**

Run: `cargo run -p minkowski-examples --example planner --release && cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: Example runs with subscription execution output, all tests pass, clippy clean.

- [ ] **Step 4: Commit**

```bash
git add examples/examples/planner.rs CLAUDE.md
git commit -m "Update planner example and CLAUDE.md for executable subscriptions"
```

---

## Final Verification

- [ ] **Run full test suite:** `cargo test -p minkowski`
- [ ] **Run clippy:** `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] **Run planner example:** `cargo run -p minkowski-examples --example planner --release`
- [ ] **Verify `SubscriptionPlan` is fully removed:** `grep -r 'SubscriptionPlan' crates/ examples/` returns nothing
- [ ] **Verify test count:** Should be 655+ (existing + new subscription tests)
