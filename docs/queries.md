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

`QueryPlanner` compiles queries into cost-optimized execution plans using a [morsel-driven][morsel-driven] push-based optimizer with automatic index selection, then executes them against the live world. Plans compile to reusable closures that process data in morsel-sized batches over 64-byte-aligned column slices, enabling SIMD auto-vectorization. Three execution modes: `execute(&mut World) -> &[Entity]` (collects into plan-owned scratch buffer, supports joins), `for_each(&mut World, callback)` (zero-allocation scan iteration), and `for_each_raw(&World, callback)` (transactional read-only path).

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
let entities = plan.execute(&mut world);  // returns &[Entity]
for e in entities {
    let score = world.get::<Score>(*e).unwrap();
    // only entities with Score in [10..50) are returned
}

// Zero-alloc iteration for scan-only plans
plan.for_each(&mut world, |entity| {
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

sub.for_each(&mut world, |entity| {
    let score = world.get::<Score>(entity).unwrap();
    if debounce.is_changed(entity, score) {
        // genuinely changed — react
    }
})?;
```

`HashDebounce<T>` is the default in-memory debounce filter. Implement `SubscriptionDebounce<T>` on your own type for external-backed deduplication.

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

<!-- Link definitions -->
[rayon]: https://github.com/rayon-rs/rayon
[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[morsel-driven]: https://db.in.tum.de/~leis/papers/morsels.pdf
