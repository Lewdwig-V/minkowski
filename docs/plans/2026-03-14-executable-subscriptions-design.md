# Executable Subscriptions (Phase 1)

**Goal:** Make `SubscriptionBuilder` produce executable `QueryPlanResult` plans
backed by compile-time index guarantees via `Indexed<T>` witnesses. Remove
the plan-only `SubscriptionPlan` type.

---

## Problem

`SubscriptionBuilder` builds plans where every predicate is proven index-backed
at compile time via `Indexed<T>` witnesses. However, it produces a
`SubscriptionPlan` that has `root()`, `cost()`, and `explain()` but no
execution methods. Meanwhile, `QueryPlanResult` (produced by `ScanBuilder`)
has full execution support via `IndexDriver` — `execute()`, `for_each()`,
`for_each_raw()`.

Two types represent the same concept (a compiled query plan) with different
capabilities. `SubscriptionPlan` is strictly less capable than
`QueryPlanResult`. The compile-time guarantee that every predicate has an
index is valuable, but it should produce an executable plan, not a
plan-only artifact.

## Design

### SubscriptionBuilder wraps ScanBuilder

`SubscriptionBuilder` becomes a compile-time validation layer around
`ScanBuilder`. It accepts real `Predicate` objects alongside `Indexed<T>`
witnesses, validates them, and delegates execution to the existing
Phase 1-8 machinery.

```rust
pub struct SubscriptionBuilder<'w> {
    scan: ScanBuilder<'w>,
    errors: Vec<SubscriptionError>,
    has_predicates: bool,
}
```

`subscribe::<Q>()` internally calls `self.scan::<Q>()` to capture the
query's required/changed bitsets and `compile_for_each` factory. The
generic `Q` is consumed at this point — `SubscriptionBuilder` does not
need to be generic over it.

### API

```rust
// subscribe() creates the builder with a ScanBuilder inside
let sub = planner.subscribe::<(&Score, &Team)>();

// where_eq/where_range take a witness + a Predicate
let plan = sub
    .where_eq(score_witness, Predicate::eq::<Score>(Score(42))?)
    .where_range(btree_witness, Predicate::range::<Score, _>(Score(10)..Score(50))?)
    .build()?;

// Returns QueryPlanResult — fully executable
plan.for_each(&mut world, |entity| { ... });
```

### where_eq / where_range

```rust
pub fn where_eq<T: Component>(
    mut self,
    witness: Indexed<T>,
    predicate: Predicate,
) -> Self {
    // Validate: predicate must be Eq kind
    if !matches!(predicate.kind, PredicateKind::Eq) {
        self.errors.push(SubscriptionError::PredicateKindMismatch {
            expected: "Eq",
            actual: format!("{:?}", predicate.kind),
            component_name: std::any::type_name::<T>(),
        });
        return self;
    }
    self.has_predicates = true;
    self.scan = self.scan.filter(predicate);
    self
}

pub fn where_range<T: Component + Ord + Clone>(
    mut self,
    witness: Indexed<T>,
    predicate: Predicate,
) -> Self {
    // Validate: witness must be BTree (Hash can't range)
    if witness.kind == IndexKind::Hash {
        self.errors.push(SubscriptionError::HashIndexOnRange {
            component_name: std::any::type_name::<T>(),
        });
        return self;
    }
    // Validate: predicate must be Range kind
    if !matches!(predicate.kind, PredicateKind::Range) {
        self.errors.push(SubscriptionError::PredicateKindMismatch {
            expected: "Range",
            actual: format!("{:?}", predicate.kind),
            component_name: std::any::type_name::<T>(),
        });
        return self;
    }
    self.has_predicates = true;
    self.scan = self.scan.filter(predicate);
    self
}
```

The `Indexed<T>` parameter's generic `T` provides compile-time enforcement
that the predicate's component matches the index's component. The witness
is consumed (its `kind` is checked for range validity) but its runtime
data isn't needed — the `ScanBuilder` finds the `IndexDescriptor` via
`TypeId` during Phase 1.

### build()

```rust
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
```

Delegates to `ScanBuilder::build()` which runs Phase 1-8. Since every
predicate is index-backed (witness proves this), Phase 1 will find
`IndexDescriptor` entries and Phase 3 will create an `IndexDriver`.
The resulting `QueryPlanResult` has `execute()`, `for_each()`, and
`for_each_raw()` with index-gather execution.

### Removals

- `SubscriptionPlan` struct and its `impl` block (root, cost, explain,
  Display, Debug)
- `IndexedPredicate` struct (no longer needed — real `Predicate` is used)
- `SubscriptionBuilder`'s internal `indexed_predicates: Vec<IndexedPredicate>`
- Old `where_eq(witness, selectivity)` / `where_range(witness, selectivity)`
  signatures

### What stays

- `Indexed<T>` — compile-time witness (unchanged)
- `SubscriptionError` — validation errors (gains `PredicateKindMismatch`)
- `CardinalityConstraint` — plan validation (works on `QueryPlanResult`
  via `root()`)
- `TablePlanner::indexed_btree` / `indexed_hash` — witness constructors

### SubscriptionError changes

Add one new variant:

```rust
PredicateKindMismatch {
    expected: &'static str,
    actual: String,
    component_name: &'static str,
},
```

Remove `NanSelectivity` (selectivity comes from the `Predicate`, which
already validates via `sanitize_selectivity`).

### CardinalityConstraint

Currently `CardinalityConstraint::validate` takes a `&SubscriptionPlan`.
Change to take `&QueryPlanResult` (both have `root() -> &PlanNode`).

### subscribe() in QueryPlanner

Currently:
```rust
pub fn subscribe<Q: 'static>(&'w self) -> SubscriptionBuilder<'w> {
    SubscriptionBuilder {
        total_entities: self.total_entities,
        query_name: std::any::type_name::<Q>(),
        indexed_predicates: Vec::new(),
        errors: Vec::new(),
    }
}
```

Changes to:
```rust
pub fn subscribe<Q: WorldQuery + 'static>(&'w self) -> SubscriptionBuilder<'w> {
    SubscriptionBuilder {
        scan: self.scan::<Q>(),
        errors: Vec::new(),
        has_predicates: false,
    }
}
```

Note: `Q` gains the `WorldQuery` bound (needed by `scan::<Q>()`). This
was implicitly required anyway — subscription queries must be valid
world queries.

### TablePlanner::subscribe

Delegates to inner `QueryPlanner::subscribe`, same as other methods.

## Semantic Review

### 1. Can this be called with the wrong World?

Same as `ScanBuilder` — the `QueryPlanner` borrows `&'w World`, and the
plan stores `world_id` for cross-world guards at execution time.

### 2. Does the witness actually prove the index exists?

`Indexed::btree(index)` requires a `&BTreeIndex<T>` reference. The index
must exist and be populated. The planner finds it via `TypeId` at Phase 1.
If the user creates a witness from one index but registers a different one,
Phase 1 still finds the registered index — the witness proves *an* index
of the right type exists, the planner uses *the registered* index.

### 3. Can a Predicate bypass the witness check?

No — `where_eq` and `where_range` are the only methods that add predicates
to the subscription builder. Both require an `Indexed<T>` parameter.
There is no `.filter()` method on `SubscriptionBuilder`.

### 4. What if the predicate's component doesn't match the witness?

The `Indexed<T>` generic parameter on `where_eq<T>` constrains the
component type. `Predicate::eq::<Score>()` returns a `Predicate` whose
`component_type` is `TypeId::of::<Score>()`. The witness `Indexed<Score>`
proves a `Score` index exists. If someone passes `Predicate::eq::<Health>()`
with `Indexed<Score>`, the predicate's `TypeId` won't match any index
in Phase 1, producing a `PlanWarning` and falling back to filter — but
the compilation still succeeds because `Predicate` is not generic. This
is a runtime mismatch, not a compile-time error.

Mitigation: `where_eq` could validate `pred.component_type == TypeId::of::<T>()`
and push a `SubscriptionError` on mismatch. This catches it at build time
rather than silently degrading to a scan.

## Implementation Steps

1. Restructure `SubscriptionBuilder` to wrap `ScanBuilder`
2. Update `where_eq` / `where_range` to accept `Predicate`
3. Update `build()` to return `QueryPlanResult`
4. Remove `SubscriptionPlan`, `IndexedPredicate`
5. Update `CardinalityConstraint::validate` signature
6. Add `PredicateKindMismatch` to `SubscriptionError`
7. Add component type validation in `where_eq`/`where_range`
8. Update `subscribe()` on `QueryPlanner` and `TablePlanner`
9. Update planner example
10. Update lib.rs re-exports (remove `SubscriptionPlan`)
11. Update CLAUDE.md

## Future Phases (Brief)

### Phase 2: Mutation Delta Tracking

Hook into every mutation path (spawn, insert, remove, despawn, changeset
apply) to capture deltas as `Vec<(Entity, ComponentId, DeltaOp)>` where
`DeltaOp` is `Insert | Update | Delete`. Emitted after each mutation
batch (e.g., after `CommandBuffer::apply`, after `Tx::commit`). The delta
is the raw material for incremental subscription evaluation.

### Phase 3: Incremental Subscription Evaluation

Given a delta from Phase 2, determine which active subscriptions' result
sets changed. For each affected subscription, compute the row-level delta
(entity entered/left the result set) by evaluating only the changed
entities against the subscription's predicates using the registered
indexes. Push deltas to registered callbacks. This is where the
`Indexed<T>` guarantee pays off — every predicate can be evaluated via
index lookup, never a full scan.

### Phase 4: Subscription Cache (minkowski-persist)

A `SubscriptionCache` in the `-persist` crate that maintains a local
mirror of subscribed rows, applies deltas from Phase 3, and provides
typed query access. Handles WAL-backed durability for offline scenarios
and replication convergence. The cache is the client-side half of the
SpacetimeDB subscription model.
