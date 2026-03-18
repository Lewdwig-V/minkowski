# Queries

Queries are tuple-typed — `world.query::<(&mut Pos, &Vel)>()` iterates all entities with both components, giving mutable access to `Pos` and shared access to `Vel`. Results are cached per query type, so repeat calls are nearly free.

`Changed<T>` skips entities whose component hasn't been modified since your last read — useful for incremental processing. `par_for_each` distributes iteration across threads via [rayon][rayon], and `for_each_chunk` yields typed slices for [SIMD][simd] auto-vectorization. Table queries (`world.query_table::<Transform>()`) skip archetype matching entirely when you know the exact schema.

```rust
// Dynamic query -- cached, skips archetype scan on repeat calls
for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
    pos.x += vel.dx;
}

// Change detection -- skip archetypes untouched since last read
for (pos, _) in world.query::<(&mut Pos, Changed<Vel>)>() {
    // only entities whose Vel column was mutably accessed
}

// Chunk iteration -- yields typed slices for auto-vectorization
world.query::<(&mut Pos, &Vel)>().for_each_chunk(|(positions, velocities)| {
    for i in 0..positions.len() {
        positions[i].x += velocities[i].dx;
    }
});
```

## Query Planner

`QueryPlanner` compiles queries into cost-optimized execution plans using a [compiled push-based][push-compiled] optimizer with automatic index selection, then executes them against the live world. Plans compile to reusable closures that process data in batches over 64-byte-aligned column slices, enabling SIMD auto-vectorization. Four execution modes: `execute_collect(&mut World) -> &[Entity]` and `execute_collect_raw(&World) -> &[Entity]` (collect into plan-owned scratch buffer, support joins), `execute_stream(&mut World, callback)` (streaming iteration, supports both scan-only and join plans), and `execute_stream_raw(&World, callback)` (transactional read-only path, supports both scan-only and join plans).

```rust
use minkowski::{QueryPlanner, Predicate, JoinKind};

let mut planner = QueryPlanner::new(&world);
planner.add_btree_index::<Score>(&score_index, &world).unwrap();
planner.add_hash_index::<Team>(&team_index, &world).unwrap();

let mut plan = planner
    .scan::<(&Pos, &Score)>()
    .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
    .build();

// Inspect the plan
println!("{}", plan.explain());  // shows vectorized execution plan

// Execute: collect matching entities into plan-owned scratch buffer
let entities = plan.execute_collect(&mut world);  // returns &[Entity]
for e in entities {
    let score = world.get::<Score>(*e).unwrap();
    // only entities with Score in [10..50) are returned
}

// Zero-alloc streaming iteration
plan.execute_stream(&mut world, |entity| {
    // process each matching entity directly — no intermediate buffer
});
```

`TablePlanner<T>` adds compile-time enforcement: if a table field is annotated with `#[index(btree)]` or `#[index(hash)]`, the planner requires the corresponding index at the type level. Missing indexes are type errors, not runtime warnings. See [Schema & Mutation](../README.md#schema--mutation) for the annotation syntax.

**Subscription queries** guarantee at compile time that every predicate is index-backed via `Indexed<T>` witnesses. Combined with `Changed<T>`, subscriptions skip archetypes whose indexed column has not been written since the last call — no delta tracking, caching, or event sourcing needed. (`Changed<T>` is archetype-granular: mutating one entity marks the entire column, so unchanged siblings in the same archetype may also pass.)

```rust
use minkowski::{Changed, HashDebounce, Indexed, Predicate, SubscriptionDebounce};

let witness = Indexed::btree(&score_index);
let mut sub = planner
    .subscribe::<(Changed<Score>, &Score)>()
    .where_eq(witness, Predicate::eq(Score(42)))
    .build()?;

// HashDebounce filters false positives from archetype-granular Changed<T>.
let mut debounce = HashDebounce::<Score>::new();

sub.execute_stream(&mut world, |entity| {
    let score = world.get::<Score>(entity).unwrap();
    if debounce.is_changed(entity, score) {
        // genuinely changed — react
    }
})?;
```

`HashDebounce<T>` is the default in-memory debounce filter. Implement `SubscriptionDebounce<T>` on your own type for external-backed deduplication.

## ER Joins (Entity-Reference Joins)

ER joins follow entity references like foreign keys in a relational database. Instead of intersecting two entity sets by identity (regular `join()`), an ER join reads a reference component from each left-side entity, extracts the target `Entity`, and checks whether that target is in the right-side set.

Implement the `AsEntityRef` trait on any component that contains a foreign-key reference to another entity:

```rust
use minkowski::{Entity, AsEntityRef};

#[derive(Clone, Copy)]
struct Parent(Entity);

impl AsEntityRef for Parent {
    fn entity_ref(&self) -> Entity {
        self.0
    }
}
```

Then use `er_join` in the query planner to filter by the referenced entity's components:

```rust
use minkowski::{QueryPlanner, JoinKind};

let planner = QueryPlanner::new(&world);

// "For each child with a Parent component, keep only those whose
//  parent entity has both Pos and Name."
let mut plan = planner
    .scan::<(&ChildTag, &Parent)>()
    .er_join::<Parent, (&Pos, &Name)>(JoinKind::Inner)
    .build();

let entities = plan.execute_collect(&mut world).unwrap();
// Only children whose parent has Pos and Name are returned.
```

**Join kinds**: `JoinKind::Inner` filters left entities to only those whose reference target is in the right set. `JoinKind::Left` keeps all left entities regardless (the right-side collection is skipped entirely for efficiency).

**Chaining**: Multiple ER joins can be chained. Each reads a different reference component from the left-side entity and filters independently:

```rust
let mut plan = planner
    .scan::<(&ChildTag, &Parent, &Owner)>()
    .er_join::<Parent, (&Name,)>(JoinKind::Inner)   // parent must have Name
    .er_join::<Owner, (&Score,)>(JoinKind::Inner)    // owner must have Score
    .build();
```

**Combining with regular joins**: Regular `join()` calls must come before any `er_join()` calls. The builder panics if you call `join()` after `er_join()`, because regular joins (sorted intersection) execute before ER joins (hash probing).

```rust
let mut plan = planner
    .scan::<(&ChildTag, &Parent, &Score)>()
    .join::<(&Team,)>(JoinKind::Inner)               // regular join first
    .er_join::<Parent, (&Name,)>(JoinKind::Inner)    // then ER join
    .build();
```

**Dead references**: If the referenced entity has been despawned, the reference extractor returns `None` and the entity is filtered out (inner join) or kept (left join). Generational entity IDs prevent false matches with reused indices.

**Cost hints**: The planner automatically estimates right-side cardinality by counting entities in matching archetypes. Use `with_right_estimate(n)` after a `join()` or `er_join()` to override this when you have better domain knowledge:

```rust
let mut plan = planner
    .scan::<(&ChildTag, &Parent)>()
    .er_join::<Parent, (&Name,)>(JoinKind::Inner)
    .with_right_estimate(100)  // "expect ~100 parents with Name"
    .unwrap()
    .build();
```

**Unregistered components**: If the reference component `R` is not yet registered in the world at plan-build time (e.g., the plan is built at startup before any entities with `R` exist), the planner defers resolution to execution time. A `PlanWarning::UnregisteredErComponent` is emitted. The plan will work correctly once entities with `R` are spawned.

## Materialized Views

`MaterializedView` wraps a `QueryPlanResult` and caches the matching entity list. On each `refresh()` call it re-executes the plan, but only if the debounce threshold has been met. Two layers of filtering: the plan's `Changed<T>` filter skips unchanged archetypes, and the configurable `DebouncePolicy` limits how often the plan runs at all.

```rust
use minkowski::{MaterializedView, DebouncePolicy};

let mut view = MaterializedView::new(plan)
    .with_debounce(DebouncePolicy::EveryNTicks(NonZeroU64::new(10).unwrap()));

// Per-frame: refresh and read
view.refresh(&mut world).unwrap();
for &entity in view.entities() {
    // cached, debounced result — stale by at most N frames
}
```

## Relationship to compiled execution, morsel scheduling, and reducers

The compiled query pipeline and a [typed reducer](../README.md#typed-reducers) are structurally similar — both are closed-over access declarations that the runtime can reason about externally. The divergence is in what that reasoning is *for*.

In the [morsel-driven model][morsel-driven], the unit of scheduling is a data fragment bound to a fixed pipeline. The query execution plan (QEP) is the consistency boundary — pipeline breakers enforce materialization points, and the dispatcher assigns morsels to worker threads. No transaction mechanism is needed because the plan topology *is* the isolation guarantee. Intra-query parallelism is structurally safe by construction.

Reducers solve a different problem: scheduling *multiple independent computations* that may conflict. The `Access` bitset is the consistency boundary — the registry and scheduler use it to prove disjointness or route through a transaction strategy (`Sequential`, `Optimistic`, `Pessimistic`). The type signature does the job that pipeline topology does in the morsel model.

Minkowski's query planner uses the [Neumann compilation model][push-compiled] (push-based compiled closures, batch granularity over 64-byte-aligned column slices) but lives in a world where the *inter-query* problem — "can these two things run concurrently?" — is solved by a completely different mechanism. The planner handles intra-query execution; reducers and transactions handle inter-query isolation. They compose but don't overlap.

<!-- Link definitions -->
[rayon]: https://github.com/rayon-rs/rayon
[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[push-compiled]: https://www.vldb.org/pvldb/vol4/p539-neumann.pdf
[morsel-driven]: https://db.in.tum.de/~leis/papers/morsels.pdf
