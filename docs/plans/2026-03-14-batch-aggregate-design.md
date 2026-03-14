# Batch Aggregate Execution Design

## Problem

`execute_aggregates` is ~13x slower than the equivalent manual `world.query()` loop.
The benchmark (`planner.rs:151`) shows `aggregate_count_sum_10k` vs `manual_count_sum_10k`
with 10K entities — the gap is entirely due to per-entity `world.get::<T>(entity)` lookups
inside a type-erased `Arc<dyn Fn(&World, Entity) -> Option<f64>>`.

Each `world.get::<T>` call (world.rs:665–682) performs:
1. `is_alive` — bounds check + generation comparison
2. `entity_locations[idx]` — location table lookup
3. `components.id::<T>()` — `HashMap<TypeId, ComponentId>` lookup
4. `is_sparse(comp_id)` — branch on storage kind
5. `archetype.column_index(comp_id)` — dense array lookup
6. `columns[col_idx].get_ptr(row)` — pointer arithmetic + cast

Steps 3–5 are *archetype-invariant* — they produce the same answer for every entity in the
same archetype. The current implementation re-derives them 10K times. A chunk-at-a-time
approach resolves them once per archetype and processes all rows as a typed slice.

## Current State

### Aggregate registration (`planner.rs:309–430`)

```rust
type ValueExtractor = Arc<dyn Fn(&World, Entity) -> Option<f64> + Send + Sync>;

fn make_extractor<T: Component>(
    extract: impl Fn(&T) -> f64 + Send + Sync + 'static,
) -> ValueExtractor {
    Arc::new(move |world: &World, entity: Entity| world.get::<T>(entity).map(&extract))
}
```

`AggregateExpr` stores `(op, label, Option<ValueExtractor>)`. The user-facing API
(`sum::<T>`, `min::<T>`, etc.) captures the component type inside `make_extractor`.

### Aggregate execution (`planner.rs:1440–1530`)

`execute_aggregates` clones the extractors (Arc bump), then calls
`compiled_for_each(world, tick, &mut |entity| { ... })` where the inner closure
calls each extractor per entity. The `compiled_for_each` callback emits entities
one at a time — there is no batch/chunk concept in this path.

### Existing chunk infrastructure (`query/iter.rs:70–81`)

`QueryIter::for_each_chunk` already yields typed slices per archetype:
```rust
pub fn for_each_chunk<F>(mut self, mut f: F) where F: FnMut(Q::Slice<'w>) {
    for (fetch, len) in &self.fetches {
        if *len > 0 {
            let slices = unsafe { Q::as_slice(fetch, *len) };
            f(slices);
        }
    }
}
```

This proves LLVM can auto-vectorize when given `&[T]` slices. The aggregate path
needs the same pattern routed through the planner's compiled scan.

### Archetype internals (`storage/archetype.rs:14–65`)

- `Archetype.columns: Vec<BlobVec>` — one BlobVec per component, in `sorted_ids` order
- `Archetype.component_index: Vec<Option<usize>>` — dense `ComponentId → column index` map
- `BlobVec::get_ptr(row) -> *mut u8` — pointer arithmetic: `base + row * stride`
- `BlobVec` columns are 64-byte aligned (cache line)

### Scan compilation (`planner.rs:2624–2775`)

Three compiled scan paths exist:
1. **Archetype scan** — iterates `world.archetypes.archetypes`, checks `required.is_subset`,
   change filter, emits per-entity
2. **Spatial index-driven** — `lookup_fn(&expr)` → `gather_index_candidates` → per-entity
3. **BTree/Hash index-driven** — `lookup_fn()` → `gather_index_candidates` → per-entity

All three emit entities individually via `callback(entity)`. The aggregate path then
calls `world.get` per entity. The batch path must change this to emit archetype chunks.

### Verify-before-design: APIs that exist vs need creation

| API | Status | Location |
|---|---|---|
| `BlobVec::get_ptr(row) -> *mut u8` | Exists | `storage/blob_vec.rs:120` |
| `Archetype::column_index(comp_id) -> Option<usize>` | Exists | `storage/archetype.rs:63` |
| `ComponentRegistry::id::<T>() -> Option<ComponentId>` | Exists | `component.rs:129` |
| `AggregateAccum::feed(f64)` | Exists | `planner.rs:459` |
| `AggregateAccum::feed_count()` | Exists | `planner.rs:481` |
| `AggregateExpr` public API (sum/min/max/avg/count) | Exists — unchanged | `planner.rs:340–403` |
| `gather_index_candidates` archetype cache | Exists | `planner.rs:2019` |
| `for_each_chunk` on `QueryIter` | Exists — reference pattern | `query/iter.rs:70` |
| `BatchExtractor` trait | **Needs creation** | `planner.rs` (new) |
| `TypedBatch<T, F>` struct | **Needs creation** | `planner.rs` (new) |
| `make_batch_extractor::<T>()` factory | **Needs creation** | `planner.rs` (new) |
| `CompiledAggScan` type alias | **Needs creation** | `planner.rs` (new) |
| Aggregate-aware archetype iteration | **Needs creation** | `planner.rs` (new) |
| Aggregate-aware index-gather | **Needs creation** | `planner.rs` (new) |

## Proposed Design

### API Surface

**No public API changes.** The user-facing `AggregateExpr::sum::<T>(name, |t| t.0 as f64)`
signature is unchanged. The batch path is a transparent internal optimization.

`AggregateExpr` grows one field:
```rust
pub struct AggregateExpr {
    op: AggregateOp,
    label: String,
    extractor: Option<ValueExtractor>,       // kept for join-plan fallback
    batch_extractor: Option<BatchFactory>,    // NEW: chunk-at-a-time path
}
```

`BatchFactory` is a boxed closure that produces a `Box<dyn BatchExtractor>`:
```rust
type BatchFactory = Box<dyn Fn() -> Box<dyn BatchExtractor> + Send + Sync>;
```

A factory (not a single instance) because `execute_aggregates` needs fresh accumulator
state per call, and the extractor holds mutable state (the column pointer binding).

### Internal Architecture

#### `BatchExtractor` trait

```rust
trait BatchExtractor: Send + Sync {
    /// Resolve column pointer for this archetype. Returns false if
    /// the component is absent (entity will be skipped).
    fn bind_archetype(&mut self, archetype: &Archetype) -> bool;

    /// Process all rows in the currently-bound archetype.
    /// Caller guarantees `bind_archetype` returned true.
    fn process_all(&mut self, count: usize, accum: &mut AggregateAccum);

    /// Process specific rows (for index-gather paths where entities
    /// are not necessarily contiguous within the archetype).
    fn process_rows(&mut self, rows: &[usize], accum: &mut AggregateAccum);
}
```

The trait has two processing methods:
- `process_all(count)` — for archetype scans where every row matches. Yields a
  contiguous `&[T]` slice to LLVM. This is the vectorization-friendly path.
- `process_rows(&[usize])` — for index-gather paths where specific rows are
  selected. The row indices are sorted (from archetype cache grouping), enabling
  sequential access but not necessarily contiguous.

#### `TypedBatch<T, F>` implementation

```rust
struct TypedBatch<T, F> {
    extract: F,
    comp_id: ComponentId,
    /// Set by bind_archetype — base pointer to the column's BlobVec data.
    col_ptr: *const T,
    _marker: PhantomData<T>,
}
```

`bind_archetype`:
```rust
fn bind_archetype(&mut self, archetype: &Archetype) -> bool {
    let Some(col_idx) = archetype.column_index(self.comp_id) else {
        return false;
    };
    // SAFETY: BlobVec stores T-layout data. column_index guarantees comp_id
    // matches this column. The archetype is borrowed for the duration of
    // the aggregate scan (we hold &World).
    self.col_ptr = unsafe { archetype.columns[col_idx].get_ptr(0) as *const T };
    true
}
```

`process_all` — the hot path:
```rust
fn process_all(&mut self, count: usize, accum: &mut AggregateAccum) {
    // SAFETY: bind_archetype set col_ptr; count == archetype.len().
    let slice = unsafe { std::slice::from_raw_parts(self.col_ptr, count) };
    for item in slice {
        accum.feed((self.extract)(item));
    }
}
```

This is the monomorphized inner loop. LLVM sees:
- Concrete `T` (not `dyn Any`)
- Concrete `F: Fn(&T) -> f64` (inlined)
- Contiguous `&[T]` slice (auto-vectorizable for Sum/Avg when `F` is a simple field access)

`process_rows`:
```rust
fn process_rows(&mut self, rows: &[usize], accum: &mut AggregateAccum) {
    for &row in rows {
        let item = unsafe { &*self.col_ptr.add(row) };
        accum.feed((self.extract)(item));
    }
}
```

#### `make_batch_extractor` factory

```rust
fn make_batch_extractor<T: Component>(
    comp_id: ComponentId,
    extract: impl Fn(&T) -> f64 + Send + Sync + Clone + 'static,
) -> BatchFactory {
    Box::new(move || -> Box<dyn BatchExtractor> {
        Box::new(TypedBatch {
            extract: extract.clone(),
            comp_id,
            col_ptr: std::ptr::null(),
            _marker: PhantomData,
        })
    })
}
```

The `Clone` bound on `extract` is needed because the factory must produce fresh
instances. The user's closure (e.g., `|s: &Score| s.0 as f64`) is trivially Clone.

The `ComponentId` is resolved once at `AggregateExpr` construction time
(via `ComponentRegistry::id::<T>()`), not per-entity. This requires that the
component is registered before the aggregate is created — which is guaranteed
because `scan::<Q>()` registers `Q`'s components, and the aggregate references
a component that must be in `Q`.

**Wait — `AggregateExpr` is constructed *before* `scan::<Q>()`** in user code:
```rust
let expr = AggregateExpr::sum::<Score>("Score", |s| s.0 as f64);
let plan = planner.scan::<(&Score,)>().aggregate(expr).build();
```

`AggregateExpr::sum::<Score>()` is a standalone constructor — it doesn't have
access to `&World` or `ComponentRegistry`. So `ComponentId` cannot be resolved
at `AggregateExpr` construction time.

**Resolution**: defer `ComponentId` resolution to `ScanBuilder::build()`, which
has access to the planner's `ComponentRegistry`. The `AggregateExpr` stores a
`TypeId` + the typed closure. `build()` resolves `TypeId → ComponentId` and
constructs the `BatchFactory`.

Updated `AggregateExpr`:
```rust
pub struct AggregateExpr {
    op: AggregateOp,
    label: String,
    extractor: Option<ValueExtractor>,
    /// TypeId of the component this aggregate reads. None for Count.
    component_type_id: Option<TypeId>,
    /// Factory for creating batch extractors. Constructed at build() time
    /// when ComponentId is available. Uses an inner factory pattern:
    /// the outer closure captures (comp_id, extract_fn) and produces
    /// fresh TypedBatch instances on each call.
    batch_factory: Option<BatchFactory>,
}
```

`make_extractor` is updated to also store the `TypeId`:
```rust
fn make_extractor_with_type_id<T: Component>(
    extract: impl Fn(&T) -> f64 + Send + Sync + Clone + 'static,
) -> (ValueExtractor, TypeId, BatchFactoryBuilder) {
    let type_id = TypeId::of::<T>();
    let extractor = Arc::new({
        let extract = extract.clone();
        move |world: &World, entity: Entity| world.get::<T>(entity).map(&extract)
    });
    let builder = Box::new(move |comp_id: ComponentId| -> BatchFactory {
        let extract = extract.clone();
        Box::new(move || -> Box<dyn BatchExtractor> {
            Box::new(TypedBatch {
                extract: extract.clone(),
                comp_id,
                col_ptr: std::ptr::null(),
                _marker: PhantomData::<T>,
            })
        })
    });
    (extractor, type_id, builder)
}
```

Where `BatchFactoryBuilder = Box<dyn FnOnce(ComponentId) -> BatchFactory + Send + Sync>`.

`ScanBuilder::build()` resolves the `ComponentId` and finalizes the factory:
```rust
for expr in &mut self.aggregates {
    if let Some(builder) = expr.batch_factory_builder.take() {
        if let Some(type_id) = expr.component_type_id {
            if let Some(comp_id) = planner.components.id_by_type_id(type_id) {
                expr.batch_factory = Some(builder(comp_id));
            }
            // If component not registered, batch_factory stays None,
            // falls back to per-entity ValueExtractor path.
        }
    }
}
```

### Data Flow

#### Archetype scan path (new `CompiledAggScan`)

```
for each archetype:
    skip if empty / required.is_subset fails / change filter fails
    for each batch_extractor:
        bind_archetype(archetype)           // resolve column pointer once
    count = archetype.len()
    for each (accum, batch_extractor):
        if op == Count:
            accum.count += count            // bulk count, no per-row work
        else:
            batch_extractor.process_all(count, accum)  // tight &[T] loop
```

Cost per archetype: 1 `is_subset` bitset check + N `column_index` lookups (one per
aggregate with a component) + M tight loops over `&[T]`.

Cost per entity: one `accum.feed()` call per aggregate — just the arithmetic.

#### Index-gather path (batched)

The existing `gather_index_candidates` groups entities by archetype via its 1-element
cache. The aggregate-aware variant extends this to collect row indices per archetype
run and flush them as a batch:

```
current_arch = None
row_buffer = Vec::new()

for each candidate entity:
    validate_entity → loc
    if loc.archetype_id != current_arch:
        flush(row_buffer, current_arch)  // process_rows on accumulated rows
        current_arch = loc.archetype_id
        for each batch_extractor:
            bind_archetype(archetype)
        row_buffer.clear()
    row_buffer.push(loc.row)

flush(row_buffer, current_arch)  // final batch
```

This reuses the archetype locality that spatial indexes naturally provide.

#### Join path — no change

Join plans collect entities into a scratch buffer first. These entities come from
multiple archetypes in unpredictable order. Batching here provides minimal benefit
(entities are already collected). The join path continues using the per-entity
`ValueExtractor` fallback.

### Count Optimization

`AggregateOp::Count` needs no component access. For archetype scans, the count is
simply `archetype.len()` — no per-row iteration at all. For index-gather, it's the
number of candidates that pass validation. This is already handled by `feed_count()`
but the archetype scan path can be further optimized to bulk-add `archetype.len()`
instead of calling `feed_count()` N times.

## Alternatives Considered

### Alternative 1: Match-arm dispatch on component type

Instead of a `dyn BatchExtractor`, enumerate supported types:
```rust
enum TypedColumn { F32(&[f32]), F64(&[f64]), I32(&[i32]), ... }
```

**Pros**: No vtable dispatch at all. Fully monomorphic.

**Tradeoffs**: Cannot handle user-defined component types (`struct Score(f32)` is not
`f32`). Would require users to implement a trait or use newtype unwrapping. Breaks the
ECS open-world assumption. Maintenance burden grows with each new supported type.

**Rejected because**: The vtable dispatch is per-archetype (~2ns), amortized over
thousands of rows. The monomorphized inner loop dominates — the vtable call is noise.

### Alternative 2: Monomorphize the entire aggregate pipeline

Template the scan closure over the aggregate types, producing a single monomorphic
closure that does archetype iteration + column resolution + accumulation:
```rust
fn compile_aggregate_scan<T: Component, F: Fn(&T) -> f64>(...)
    -> Box<dyn FnMut(&World) -> f64>
```

**Pros**: Zero dynamic dispatch. Maximum optimization potential.

**Tradeoffs**: Each combination of (scan_path × aggregate_count × component_types)
generates a distinct closure. Code size grows combinatorially. Cannot support
dynamically-composed aggregate sets (the number and types of aggregates must be
known at compile time). The current API allows `aggregate()` calls in a builder
chain — each call adds a runtime-determined expression.

**Rejected because**: The builder API is inherently dynamic. The vtable-per-archetype
approach preserves the builder pattern while delivering >90% of the monomorphic
performance. The inner loops are still fully monomorphized — only the archetype-level
dispatch goes through vtable.

### Alternative 3: Keep per-entity extraction but cache ComponentId + column index

A minimal optimization: resolve `ComponentId` and `column_index` once per archetype
(like the existing archetype cache), but still call the extractor per-entity with a
pre-resolved `(col_ptr, row)` instead of `world.get()`.

**Pros**: Minimal code change. No new traits. Eliminates steps 1–5 of `world.get`.

**Tradeoffs**: Still calls through `Arc<dyn Fn>` per entity (prevents inlining).
Still iterates entity-at-a-time (prevents SIMD). Gets maybe 3–5x improvement
instead of 10–13x. A halfway measure that doesn't reach the query baseline.

**Rejected because**: The gap to `world.query()` is primarily about LLVM seeing a
tight loop over a typed slice. Anything that keeps per-entity dynamic dispatch
leaves the vectorization opportunity on the table.

## Semantic Review

### 1. Can this be called with the wrong World?

No new risk. `execute_aggregates` already checks `self.world_id != world.world_id()`.
The `BatchExtractor` receives archetype references from the same World — the column
pointers are derived from `&World`-borrowed archetypes. No cross-world pointer
contamination is possible.

### 2. Can Drop observe inconsistent state?

`TypedBatch` holds a raw `*const T` that becomes dangling when the `&World` borrow
ends. But `TypedBatch` has no `Drop` impl — the pointer is not freed. The struct is
created fresh per `execute_aggregates` call (via `BatchFactory`) and dropped at the
end. The pointer is only valid during `process_all`/`process_rows`, which execute
within the `&World` borrow scope.

Risk: if `TypedBatch` were stored across calls and the archetype was reallocated
(entity migration changes BlobVec backing), the pointer would dangle. **Mitigation**:
`TypedBatch` is created fresh each call via the factory. `bind_archetype` is called
before every `process_*`. The pointer never outlives the `&World` borrow.

### 3. Can two threads reach this through `&self`?

No. `execute_aggregates` takes `&mut self`. `execute_aggregates_raw` takes `&mut self`
+ `&World`. The `&mut self` on `QueryPlanResult` prevents concurrent access to the
batch extractors. `TypedBatch` is `Send + Sync` (the `*const T` is derived from
an archetype column that is `Send + Sync` by component bounds), but it's never
shared — each call creates its own instances via the factory.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

No dedup or merge in this design. Each `AggregateExpr` produces its own independent
`BatchExtractor`. Multiple aggregates on the same component type each get their own
column pointer binding (redundant but safe — the `bind_archetype` cost is one
`column_index` lookup per aggregate per archetype, negligible vs row processing).

### 5. What happens if this is abandoned halfway through?

If `execute_aggregates` panics during `process_all` (e.g., the user's `extract`
closure panics), the `BatchExtractor` instances are dropped normally. No engine
state is modified — aggregates are read-only. The accumulators are local variables.
`last_read_tick` is only updated after successful completion. A panic leaves the
plan in a consistent state for retry.

### 6. Can a type bound be violated by a legal generic instantiation?

`TypedBatch<T, F>` requires `T: Component` (which implies `'static + Send + Sync`)
and `F: Fn(&T) -> f64 + Send + Sync + Clone + 'static`. These bounds are enforced
at `AggregateExpr::sum::<T>()` construction time — the same bounds as today.

The `Clone` requirement on `F` is new but trivially satisfied by all closures that
capture only `Copy`/`Clone` state (field accessors, arithmetic). If a user passes
a non-Clone closure, they get a compile error at `sum::<T>()` — clear and early.

**Wait**: the current API does NOT require `Clone`:
```rust
pub fn sum<T: Component>(
    name: &str,
    extract: impl Fn(&T) -> f64 + Send + Sync + 'static,  // no Clone
) -> Self
```

Adding `Clone` is a **breaking API change**. However, we can avoid it by wrapping
the closure in `Arc` before passing to the factory:
```rust
let extract = Arc::new(extract);
// BatchFactory clones the Arc (ref-count bump), not the closure
```

This preserves the existing API bounds while allowing the factory to produce
multiple instances.

### 7. Does the API surface permit any operation not covered by the Access bitset?

`BatchExtractor::bind_archetype` receives `&Archetype` and reads a column via
`get_ptr(0)` — a read-only operation. The aggregate scan does not call `get_ptr_mut`
or advance ticks on columns. This is consistent with the existing `execute_aggregates`
path which reads through `world.get::<T>()` (also read-only).

The bypass-path invariant is maintained: the batch path does not modify query cache,
does not advance column ticks, and does not perform structural mutations. It is a
pure read path, same as the current per-entity path.

## Implementation Plan

### Step 1: Add `BatchExtractor` trait and `TypedBatch` struct (`planner.rs`)

Add the trait definition, the concrete implementation, and the factory function.
All `pub(crate)` — internal to the planner. No public API changes.

```
crates/minkowski/src/planner.rs  (new items near line 310, after ValueExtractor)
```

### Step 2: Extend `AggregateExpr` with batch factory builder (`planner.rs`)

Add `component_type_id: Option<TypeId>` and
`batch_factory_builder: Option<BatchFactoryBuilder>` fields to `AggregateExpr`.
Update `make_extractor` to also produce the builder. The `ValueExtractor` is kept
as a fallback for join plans.

```
crates/minkowski/src/planner.rs  (modify AggregateExpr struct + constructors)
```

### Step 3: Resolve ComponentId in `ScanBuilder::build()` (`planner.rs`)

In the build method, after plan compilation, resolve each aggregate's `TypeId`
to `ComponentId` via the planner's `ComponentRegistry` and finalize the
`BatchFactory`. Store the factories on `QueryPlanResult`.

```
crates/minkowski/src/planner.rs  (modify ScanBuilder::build, ~line 2790)
```

### Step 4: Add compiled aggregate scan for archetype path (`planner.rs`)

New `CompiledAggScan` type:
```rust
type CompiledAggScan = Box<dyn FnMut(
    &World,
    Tick,
    &mut [Box<dyn BatchExtractor>],
    &mut [AggregateAccum],
)>;
```

Compile an archetype-scan closure that iterates matching archetypes, calls
`bind_archetype` per extractor, then `process_all(archetype.len())` per
accumulator. Bulk-count optimization for `Count` ops.

```
crates/minkowski/src/planner.rs  (new closure factory in ScanBuilder::build)
```

### Step 5: Add compiled aggregate scan for index-gather path (`planner.rs`)

Extend `gather_index_candidates` (or create a parallel `gather_index_batched`)
that collects row indices per archetype run and flushes via `process_rows`.
Reuses the existing 1-element archetype cache pattern.

```
crates/minkowski/src/planner.rs  (new function near gather_index_candidates)
```

### Step 6: Rewrite `execute_aggregates` to use batch path (`planner.rs`)

When `batch_factories` are available:
1. Create fresh `Box<dyn BatchExtractor>` instances via factories
2. Initialize `AggregateAccum` instances
3. Call the compiled aggregate scan
4. Fall through to per-entity path for join plans or when batch_factory is None

```
crates/minkowski/src/planner.rs  (modify execute_aggregates, ~line 1440)
```

### Step 7: Rewrite `execute_aggregates_raw` similarly (`planner.rs`)

Same batch optimization for the read-only transactional path. Uses the `_raw`
compiled scan variant.

```
crates/minkowski/src/planner.rs  (modify execute_aggregates_raw, ~line 1542)
```

### Step 8: Benchmark validation (`minkowski-bench/benches/planner.rs`)

Run existing benchmarks. Target: `aggregate_count_sum_10k` should approach
`manual_count_sum_10k` (currently ~13x gap → target <2x). No new benchmarks
needed — the existing pair directly measures the improvement.

```
cargo bench -p minkowski-bench -- aggregate
```

### Step 9: Add unit tests (`planner.rs`)

Add tests for:
- Batch path produces same results as per-entity path (all ops)
- Multi-archetype aggregates (entities spread across archetypes)
- Index-driven aggregate with batch path
- `Changed<T>` filter with batch aggregate
- Empty archetype handling
- Count-only aggregate (no component access)
- NaN propagation through batch path
- Fallback to per-entity path for join plans

Existing aggregate tests (30+ tests at line 8215+) serve as regression suite.

### Non-goals for this pass

- **Multi-column aggregates** (`SUM(price * quantity)`): deferred. The single-column
  batch path is the 80/20 — it eliminates the 13x overhead. Multi-column needs a
  `MultiBatchExtractor` that binds multiple columns, which is the same pattern but
  more surface area.
- **SIMD-explicit intrinsics**: we rely on LLVM auto-vectorization via `-C target-cpu=native`.
  The tight `&[T]` loop is the shape LLVM needs. Hand-written SIMD is a future
  optimization if auto-vectorization proves insufficient.
- **Parallel aggregates**: `par_for_each_chunk` with per-thread accumulators + merge.
  Orthogonal to the batch optimization — can be layered on top later.
