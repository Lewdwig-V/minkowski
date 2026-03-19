# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo test -p minkowski --lib          # Unit tests (644 tests, fast)
cargo test -p minkowski                # All tests including doc tests
cargo test -p minkowski -- entity      # Run tests matching a filter

cargo clippy --workspace --all-targets -- -D warnings   # Lint (strict, warnings are errors)
cargo fmt --all                                          # Format

cargo bench -p minkowski-bench                      # All standardized benchmarks (simple_insert, simple_iter, fragmented_iter, heavy_compute, add_remove, schedule, serialize, reducer, planner)
cargo bench -p minkowski-bench -- simple_iter       # Single scenario
cargo bench -p minkowski-bench -- simple_iter/par   # Sub-benchmark filter
cargo bench -p minkowski-persist       # Persistence benchmarks (snapshot save/load/zero-copy, WAL append)

cargo run -p minkowski-examples --example boids --release   # Boids flocking with query reducers + spatial grid (5K entities, 1K frames)
cargo run -p minkowski-examples --example life --release    # Game of Life with QueryMut reducer, Table (64x64 grid, 500 gens)
cargo run -p minkowski-examples --example nbody --release   # Barnes-Hut N-body with query reducers (2K entities, 1K frames)
cargo run -p minkowski-examples --example scheduler --release   # ReducerRegistry-based conflict detection + batch scheduling (6 systems, 10 frames)
cargo run -p minkowski-examples --example transaction --release   # Transaction strategies: raw Tx, TxScope, + reducer comparison (3 strategies, 100 entities)
cargo run -p minkowski-examples --example battle --release   # Multi-threaded EntityMut reducers with tunable conflict (500 entities, 100 frames)
cargo run -p minkowski-examples --example persist --release   # Durable QueryWriter reducer: WAL + rkyv snapshots + zero-copy load (100 entities, 3 archetypes, 10 frames)
cargo run -p minkowski-examples --example replicate --release   # Pull-based WAL replication: cursor + batch + apply to replica (20 source + 10 WAL, convergence check)
cargo run -p minkowski-examples --example reducer --release   # Typed reducer system: entity/query/spawner/query-writer/dynamic handles + structural mutations + conflict detection
cargo run -p minkowski-examples --example index --release   # B-tree range queries + hash exact lookups (200 entities)
cargo run -p minkowski-examples --example flatworm --release   # Flatworm (planarian) simulator: chemotaxis, fission, starvation, spatial grid (200 worms, 1K frames)
cargo run -p minkowski-examples --example circuit --release   # Analog circuit sim: 555 astable → LCR bandpass → 741 follower, symplectic Euler (200K steps, ASCII waveform)
cargo run -p minkowski-examples --example tactical --release   # Multi-operator tactical map: sparse components, par_for_each, Optimistic Conflict, entity bit packing, HashIndex stale validation, EnumChangeSet/MutationRef replication (100 units, 10 ticks, 2 threads)
cargo run -p minkowski-examples --example observe --release   # Observability: MetricsSnapshot capture, diff, entity churn (100 entities, 2 archetypes)
cargo run -p minkowski-examples --example blob --release   # Blob offloading: BlobRef component + BlobStore lifecycle trait, MemoryBlobStore cleanup (5 entities, orphan deletion)
cargo run -p minkowski-examples --example retention --release   # Retention: Expiry countdown + RetentionReducer, dispatch-count TTL, progressive despawn (5 entities, 8 frames)
cargo run -p minkowski-examples --example pool --release   # Memory pool: TigerBeetle-style WorldBuilder with 16 MB budget, try_spawn until exhaustion, pool stats (131K entities)
cargo run -p minkowski-examples --example profile_changeset --release   # Profiling harness: QueryWriter vs QueryMut flamegraph capture (10K entities, 1K iterations)
cargo run -p minkowski-examples --example planner --release   # Compiled push-based query planner: cost-based plans, index selection, joins, execute/for_each/for_each_raw (1K entities, 2 archetypes)
cargo run -p minkowski-examples --example materialized_view --release   # Materialized views: cached debounced subscription queries, change detection, invalidation (200 entities, 7 demos)

MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri nextest run -p minkowski --lib  # UB check (full suite ~860 tests, parallel via nextest)

RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p minkowski --lib --tests -Z build-std --target x86_64-unknown-linux-gnu -- --skip par_for_each  # TSan (data race detection)
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p minkowski --lib --tests -Z build-std --target x86_64-unknown-linux-gnu par_for_each  # TSan rayon tests

RUSTFLAGS="--cfg loom" cargo test -p minkowski --lib --features loom -- loom_tests  # loom: exhaustive concurrency verification

cargo +nightly fuzz run fuzz_world_ops -- -max_total_time=60     # fuzz: random World operations
cargo +nightly fuzz run fuzz_reducers -- -max_total_time=60      # fuzz: query iteration paths
cargo +nightly fuzz run fuzz_snapshot_load -- -max_total_time=60 -max_len=65536  # fuzz: snapshot deserialization
cargo +nightly fuzz run fuzz_wal_replay -- -max_total_time=60 -max_len=65536    # fuzz: WAL replay
```

Miri flags: `-Zmiri-tree-borrows` because crossbeam-epoch (rayon dep) violates Stacked Borrows. Runs the full test suite via nextest (parallel execution). Exclusions defined in `.config/nextest.toml` (`[profile.default-miri]`, auto-activated under Miri): `par_for_each` (rayon unsupported by Miri, covered by TSan), concurrent/contention pool tests (too slow, covered by TSan + Loom). To list selected tests: `cargo nextest list -p minkowski --lib --profile default-miri`.

Pre-commit hooks run `cargo fmt` and `cargo clippy -D warnings` on commit, `cargo test` on push.

## CI

### PR pipeline (`.github/workflows/ci.yml`)

Runs on every PR and push to main:

| Job | Toolchain | Duration | Command | `needs` |
|---|---|---|---|---|
| fmt | stable | ~30s | `cargo fmt --all -- --check` | — |
| clippy | stable | ~1min | `cargo clippy --workspace --all-targets -- -D warnings` | — |
| test | stable | ~2min | `cargo test -p minkowski` | fmt, clippy |
| tsan | nightly | ~2min | Two-step TSan run with `-Z build-std` (see Build & Test Commands) | test |
| loom | stable | ~1min | Exhaustive concurrency verification (see Build & Test Commands) | test |

Format and clippy run in parallel. Test runs after both pass. TSan and Loom run in parallel after test. A `ci-pass` aggregator job (runs with `if: always()`) is the single required status check for branch protection.

**Total PR pipeline wall-clock: ~4 minutes** (fmt+clippy parallel ~1min → test ~2min → tsan/loom parallel ~2min).

### Miri (`.github/workflows/miri.yml`)

Full Miri suite (~860 tests) via nextest parallel execution. Runs nightly (04:00 UTC) and on release tags (`v*`). Can also be triggered manually via `workflow_dispatch`.

**Flake guidance**: Miri and TSan use nightly toolchains that occasionally introduce regressions. If a nightly Miri run fails but the code hasn't changed, check the [Rust nightly changelog](https://releases.rs/) for recent Miri changes. Pin the nightly version in CI (`rust-toolchain.toml` or `dtolnay/rust-toolchain@<date>`) if a regression persists. Use `gh workflow run miri.yml` to re-run manually after a suspected nightly flake.

Run the full CI pipeline locally (PR checks + Miri):
```bash
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p minkowski && RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p minkowski --lib --tests -Z build-std --target x86_64-unknown-linux-gnu && RUSTFLAGS="--cfg loom" cargo test -p minkowski --lib --features loom -- loom_tests && MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri nextest run -p minkowski --lib
```

## Architecture

Minkowski is a **column-oriented archetype ECS**. Five crates: `minkowski` (core), `minkowski-derive` (`#[derive(Table)]` proc macro), `minkowski-persist` (WAL, snapshots, durable transactions), `minkowski-observe` (metrics capture and display), and `minkowski-examples` (examples as external API consumers).

```
                          ┌─────────────────────────────────────────────┐
                          │              User / Framework               │
                          └──────┬──────────────┬───────────────┬───────┘
                                 │              │               │
                    world.spawn()│  registry.call()    strategy.transact()
                    world.query()│  registry.run()             │
                                 │              │               │
          ┌──────────────────────▼──────────────▼───────────────▼──────────────┐
          │                            World                                   │
          │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
          │  │EntityAllocator│ │ComponentRegistry│ │  QueryCache (TypeId→  │   │
          │  │ gen[] + free  │ │  id→layout+drop │ │  matched archetypes)  │   │
          │  └──────┬───────┘ └───────┬──────┘  └──────────────────────────┘   │
          │         │                 │                                         │
          │  ┌──────▼─────────────────▼────────────────────────────────────┐   │
          │  │                    Archetypes                                │   │
          │  │  ┌─────────────────────────────────────────────────────┐    │   │
          │  │  │ Archetype { component_ids: FixedBitSet }            │    │   │
          │  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │    │   │
          │  │  │  │ BlobVec  │ │ BlobVec  │ │ BlobVec  │  ...cols   │    │   │
          │  │  │  │ (Pos)    │ │ (Vel)    │ │ (Health) │            │    │   │
          │  │  │  │ 64B-align│ │ 64B-align│ │ 64B-align│            │    │   │
          │  │  │  └──────────┘ └──────────┘ └──────────┘            │    │   │
          │  │  │  entities: [Entity] ── row↔entity mapping          │    │   │
          │  │  └─────────────────────────────────────────────────────┘    │   │
          │  │  entity_locations: [Option<(arch_id, row)>] ── O(1) lookup │   │
          │  └─────────────────────────────────────────────────────────────┘   │
          │  ┌───────────────────────┐  ┌──────────────────┐                   │
          │  │ PagedSparseSet (opt-in│  │  SlabPool (mmap) │                   │
          │  │  sparse components)   │  │  TCache (TLS)    │                   │
          │  └───────────────────────┘  └──────────────────┘                   │
          └────────────────────────────────────────────────────────────────────┘
                     │                            │
       ┌─────────────┘                            └──────────────┐
       │ External composition (not inside World)                  │
       │                                                          │
  ┌────▼─────────┐  ┌───────────────┐  ┌──────────────────┐  ┌───▼──────────┐
  │ReducerRegistry│ │ QueryPlanner  │  │ SpatialIndex     │  │ Transact     │
  │ call()/run() │  │ cost-based    │  │ BTreeIndex       │  │ Sequential   │
  │ Access bitsets│  │ plans, joins  │  │ HashIndex        │  │ Optimistic   │
  │              │  │ aggregates    │  │ grids, quadtrees │  │ Pessimistic  │
  └──────────────┘  └───────────────┘  └──────────────────┘  └──────────────┘
                                                                    │
                                                             ┌──────▼──────┐
                                                             │ Durable<S>  │
                                                             │ WAL+snapshot│
                                                             │ (persist)   │
                                                             └─────────────┘
```

### Storage Model

Each unique set of component types gets an **Archetype** — a struct containing parallel `BlobVec` columns (type-erased `Vec<T>` storing raw bytes via `Layout`) plus a `Vec<Entity>` for row-to-entity mapping. A `FixedBitSet` on each archetype tracks which `ComponentId`s it contains, enabling fast query matching via bitwise subset checks.

**Entity** = u64 bit-packed: low 32 bits = index, high 32 bits = generation. `EntityAllocator` maintains a generation array + free list. `entity_locations: Vec<Option<EntityLocation>>` maps entity index → (archetype_id, row) for O(1) lookup.

**Sparse components** (`PagedSparseSet` with 4096-entry pages mapped to a dense `BlobVec`) are opt-in via `insert_sparse`. Not stored in archetypes.

### Data Flow

`world.spawn((Pos, Vel))` → `Bundle::component_ids` registers types → `Archetypes::get_or_create` finds/creates archetype by sorted component ID set → `Bundle::put` writes each component into BlobVec columns via raw pointer copy → `EntityLocation` recorded.

`world.query::<(&mut Pos, &Vel)>()` → looks up `QueryCacheEntry` by `TypeId` → if archetype count unchanged, reuses cached `matched_ids`; otherwise incrementally scans only new archetypes via `required.is_subset(&arch.component_ids)` → `init_fetch` grabs raw column pointers per cached archetype → `QueryIter` yields items via pointer arithmetic (`ptr.add(row)`).

### Archetype Migration

`world.insert(entity, NewComponent)?` moves an entity from archetype A to A∪{new}: copies each column via `BlobVec::push` + `swap_remove_no_drop`, writes new component, updates all `EntityLocation`s (including the entity swapped into the vacated row). `get_pair_mut` uses `split_at_mut` for safe double-mutable archetype access.

### Key Traits

- **Component**: marker, blanket impl for `'static + Send + Sync`
- **Bundle** (unsafe): tuple impls 1-12 via `impl_bundle!` macro. `component_ids()` registers + sorts + deduplicates. `put()` yields component pointers via `ManuallyDrop`.
- **WorldQuery** (unsafe): tuple impls 1-12 via `impl_world_query_tuple!` macro. `Fetch` type holds `ThinSlicePtr<T>` (raw pointer wrapped for Send+Sync). Impls for `&T`, `&mut T`, `Entity`, `Option<&T>`, `Changed<T>`. Methods: `required_ids`, `init_fetch`, `fetch`, `as_slice`, `mutable_ids`, `matches_filters`.
- **Table** (unsafe): generated by `#[derive(Table)]`. Named struct = schema declaration + data container. Associated types `Ref<'w>` / `Mut<'w>` for typed row access. `TableDescriptor` caches archetype_id + field-to-column index mapping. `query_table`/`query_table_mut` skip archetype matching entirely. Field attributes `#[index(btree)]` / `#[index(hash)]` generate `HasBTreeIndex<C>` / `HasHashIndex<C>` marker trait impls for compile-time index enforcement via `TablePlanner`.
- **HasBTreeIndex<C>** (unsafe): marker trait generated by `#[derive(Table)]` for `#[index(btree)]` fields. Carries `FIELD_NAME` constant and `create_btree_index` default method.
- **HasHashIndex<C>** (unsafe): marker trait generated by `#[derive(Table)]` for `#[index(hash)]` fields. Carries `FIELD_NAME` constant and `create_hash_index` default method.
- **TableRow** (unsafe): constructs typed row references from raw column pointers. Generated for `FooRef<'w>` and `FooMut<'w>` by the derive macro.
- **Transact**: closure-based transaction API. Primary method `transact(&self, world, access, closure)` runs the closure in a retry loop. Building-block methods `begin()` and `try_commit()` enable advanced patterns. Impls: `Optimistic` (tick-based validation), `Pessimistic` (cooperative column locks). `Durable<S>` (in persist crate) wraps any `Transact` to add WAL logging.

### Query Caching

`World::query()` maintains a `HashMap<TypeId, QueryCacheEntry>` that caches matched archetype IDs per query type. On repeat calls with no new archetypes, the archetype scan is skipped entirely. When new archetypes are created (spawn/insert/remove can trigger this), only the new ones are scanned incrementally. Empty archetypes are filtered at iteration time, not cache time. `query()` takes `&mut self` for cache mutation.

### Column Alignment & Vectorization

BlobVec columns are allocated with 64-byte alignment (cache line). `QueryIter::for_each_chunk` yields typed `&[T]` / `&mut [T]` slices per archetype — LLVM can auto-vectorize loops over these slices. Reducer handles (`QueryMut::for_each`, `QueryRef::for_each`, `DynamicCtx::for_each`) delegate to `for_each_chunk` internally.

Component types that are 16-byte-aligned (e.g., `#[repr(align(16))]` or naturally `[f32; 4]`) vectorize better than odd-sized ones. The engine guarantees 64-byte column alignment; component layout determines whether LLVM can pack operations.

Build with `-C target-cpu=native` (configured in `.cargo/config.toml`) to enable platform-specific SIMD instructions.

### Change Detection

Each BlobVec column stores a `changed_tick: Tick` — the tick at which it was last mutably accessed. The tick is a monotonic u64 counter that auto-advances on every mutation and query — there is no user-facing `World::tick()`. Every mutable access path (spawn, get_mut, insert, query `&mut T`, query_table_mut, query_table_raw, changeset apply) advances the tick and marks affected columns. Marking is pessimistic (on mutable access, not actual write) but zero-cost at the write site.

`Changed<T>` is a `WorldQuery` filter that skips entire archetypes whose column tick is older than the query's `last_read_tick` (stored per query type in `QueryCacheEntry`). `Changed<T>` means "since the last time this query observed this column" — it has no concept of frames or simulation time. `Tick` is `pub(crate)` — not exposed to users.

### Deferred Mutation

`CommandBuffer` stores `Vec<Box<dyn FnOnce(&mut World) + Send>>`. Used during query iteration when structural changes (spawn/despawn/insert/remove) must be deferred. Applied via `cmds.apply(&mut world)`.

`EnumChangeSet` is the data-driven alternative: mutations are recorded as a `Vec<Mutation>` enum with component bytes in a contiguous `Arena`. `apply()` consumes the changeset and returns `Result<(), ApplyError>`. Useful for persistence (WAL serialization) and transactions. Typed safe helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) wrap the raw `record_*` methods — they auto-register component types and handle `ManuallyDrop` internally. The raw methods remain for power users who already have a `ComponentId`.

### Secondary Indexes

`SpatialIndex` is a lifecycle trait for user-owned spatial data structures (grids, quadtrees, BVH, k-d trees). Indexes are fully external to World — they compose from existing query primitives. The trait has two methods: `rebuild` (required, full reconstruction from scratch) and `update` (optional, defaults to rebuild — override for incremental updates via `Changed<T>`). `rebuild` is for initial population and periodic compaction. `update` is the per-frame call — cost is proportional to the number of changes, not total entity count. Call `update` once per frame before querying; after that, all lookups (direct `get`/`range`, planner execution, reducer queries) see consistent fresh data. Despawned entities are handled via generational validation: stale entries are skipped at query time when `world.is_alive()` returns false, and cleaned up on the next rebuild.

`BTreeIndex<T>` provides O(log n + k) range queries and exact-match via a `BTreeMap`. `HashIndex<T>` provides O(1) exact-match via `HashMap`. Both support validated queries (`get_valid`, `range_valid`) that filter despawned entities and removed components. Both implement `SpatialIndex` for lifecycle management. Both have incremental `update` implementations that scan only changed entities via `query_changed_since`.

### Compile-Time Index Enforcement

`#[derive(Table)]` supports `#[index(btree)]` and `#[index(hash)]` field attributes that generate marker trait impls:

```rust
#[derive(Table)]
struct Scores {
    #[index(btree)]
    score: Score,
    #[index(hash)]
    team: Team,
    name: Name,  // no index
}
```

This generates `unsafe impl HasBTreeIndex<Score> for Scores` and `unsafe impl HasHashIndex<Team> for Scores`. These marker traits carry a `FIELD_NAME` constant for diagnostics and a default `create_btree_index`/`create_hash_index` method that creates and populates the index from world state.

`TablePlanner<'w, T>` wraps `QueryPlanner` and uses these trait bounds to enforce index presence at compile time:
- `add_btree_index::<C>()` requires `T: HasBTreeIndex<C>` — calling it on an un-annotated field is a type error
- `add_hash_index::<C>()` requires `T: HasHashIndex<C>`
- `indexed_btree::<C>()` / `indexed_hash::<C>()` produce `Indexed<C>` witnesses for `SubscriptionBuilder`
- `scan::<Q>()` / `scan_with_estimate::<Q>()` delegate to the underlying `QueryPlanner`

The pattern: create indexes via the `HasBTreeIndex`/`HasHashIndex` trait methods (needs `&mut World`), then construct a `TablePlanner` (borrows `&World`) to register them and build plans. This two-phase split avoids borrow conflicts.

A field can carry both `#[index(btree)]` and `#[index(hash)]` simultaneously. Fields without an `#[index(...)]` attribute generate no marker traits — attempting to use them in `TablePlanner`'s index methods is a compile-time error.

### Query Planner

`QueryPlanner` implements a query planner adapted for in-memory ECS. All plans execute via chunked, slice-based iteration over 64-byte-aligned columns — LLVM auto-vectorizes loops over these contiguous slices. There is no separate "scalar" execution path; cost estimates bake in batch amortization, branchless filter eligibility, and cache-partitioned join factors directly.

**Plan nodes** (`PlanNode`): `Scan` (chunked archetype iteration with `avg_chunk_size`), `IndexLookup` (batch entity gather via BTree/Hash), `SpatialLookup` (spatial index gather), `Filter` (predicate on contiguous slices, with `branchless` flag), `HashJoin` (L2-partitioned with `partitions` count), `NestedLoopJoin` (batch iteration for small cardinalities), `ErJoin` (streaming hash join on entity references — follows foreign keys via `AsEntityRef` trait), `Aggregate` (single-pass stream aggregate).

**ER (Entity-Relationship) joins**: `ScanBuilder::er_join::<R, Q>(JoinKind)` adds a foreign-key join. `R: AsEntityRef` is a component on left-side entities that contains an `Entity` reference (foreign key). `Q` defines required components on the referenced entity. Execution: right side collected into `HashSet<Entity>`, left entities filtered by probing via `R::entity_ref()`. Inner joins keep only left entities whose reference target is in the right set; left joins keep all left entities (short-circuiting right-side collection). Regular `join()` calls must precede `er_join()` calls — the builder panics otherwise. ComponentId for `R` is pre-resolved at build time; if unregistered, deferred resolution via `OnceLock` handles long-lived plans built before components exist (emits `PlanWarning::UnregisteredErComponent`). Multiple ER joins can be chained — each filters independently on the left-side entity's respective reference component.

**Cost model** (`Cost`): `rows` (estimated output cardinality) and `cpu` (dimensionless relative units). Cost adjustments for batch amortization (0.9×), branchless filters (0.5× vs 0.85× branched), and cache partitioning (0.7×/0.9×) are applied directly during plan construction.

`QueryPlanResult` stores the `PlanNode` tree. `cost()` returns the estimated execution cost. `explain()` shows the plan tree. `root()` returns `&PlanNode` for introspection. Four execution methods: `execute_collect(&mut self, &mut World) -> &[Entity]` collects matching entities into plan-owned scratch (supports joins, advances tick), `execute_collect_raw(&mut self, &World) -> &[Entity]` is the transactional counterpart (supports joins, no tick advancement, takes `&World`), `execute_stream(&mut self, &mut World, callback)` streams entities through a callback (supports both scan-only and join plans), `execute_stream_raw(&mut self, &World, callback)` is the transactional read path (supports both scan-only and join plans, no tick advancement, no cache mutation, takes `&World`). For join plans, `execute_stream`/`execute_stream_raw` materialise into the plan's internal scratch buffer before streaming through the callback — the scratch is plan-local computation, invisible to the conflict model. Aggregate methods: `aggregate(&mut self, &mut World)` and `aggregate_raw(&mut self, &World)` compute `AggregateResult` in a single pass over matched entities. Both support scan-only and join plans. Aggregates are added via `ScanBuilder::aggregate(AggregateExpr)`. Five operations: `Count`, `Sum`, `Min`, `Max`, `Avg`. Value extraction uses type-erased `f64` closures. NaN propagates consistently through all operations.

`QueryPlanner::add_spatial_index_with_lookup::<T>(index, world, lookup)` registers a spatial index with an execution-time lookup closure. The closure bridges between the planner's `SpatialExpr` protocol and the index's concrete query API — the planner makes no assumptions about how the index answers queries (mechanisms not policy). When a spatial predicate is chosen as the driving access, Phase 8 compiles an index-gather closure that calls the lookup instead of scanning archetypes. `add_spatial_index` (without lookup) remains the cost-only registration path — plans fall back to scan + filter at execution time.

BTree and Hash index lookup functions (`eq_lookup_fn`, `range_lookup_fn`) captured at `add_btree_index` / `add_hash_index` registration are invoked at execution time when the index is chosen as the driving access. The predicate's `lookup_value` is pre-bound into the lookup closure at Phase 3 plan-build time (`IndexDriver`), so the execution path never handles `dyn Any`. Same validation pipeline as spatial: `is_alive` → archetype location → required components → `Changed<T>` → filter refinement.

`Indexed<T>` is a compile-time witness that an index exists for component `T`. Cannot be constructed directly — only via `Indexed::btree(&index)` or `Indexed::hash(&index)`. Used by `SubscriptionBuilder` to enforce that every predicate in a subscription query is backed by an index.

`SubscriptionBuilder` wraps `ScanBuilder` with compile-time index enforcement via `Indexed<T>` witnesses. `where_eq(witness, predicate)` and `where_range(witness, predicate)` require an `Indexed<T>` proof that an index exists for the predicate's component. `build()` returns `QueryPlanResult` with full execution support — subscription plans use `IndexDriver` for index-gather execution, never a full archetype scan. Combined with `Changed<T>` in the query type (e.g., `subscribe::<(Changed<Score>, &Score)>()`), subscriptions skip archetypes whose indexed column has not been written since the last call — no delta tracking, caching, or event sourcing infrastructure needed. `Changed<T>` is archetype-granular: mutating one entity marks the entire column as changed.

`TablePlanner<'w, T>` wraps `QueryPlanner` with compile-time index enforcement via `HasBTreeIndex`/`HasHashIndex` trait bounds. See "Compile-Time Index Enforcement" section.

### Materialized Views

`MaterializedView` wraps a `QueryPlanResult` (typically from `SubscriptionBuilder`) and caches the matching entity list. On each `refresh(&mut World)` call it re-executes the plan, but only if: (1) the plan's `Changed<T>` filter detects column mutations (archetype-granular), and (2) the configurable `DebouncePolicy` threshold has been met.

Two debounce modes: `DebouncePolicy::Immediate` (default — refresh every call, relying on `Changed<T>` for efficiency) and `DebouncePolicy::EveryNTicks(n)` (refresh at most once per `n` calls). `invalidate()` forces the next call to refresh regardless of policy. `set_policy()` switches policies at runtime.

The view is external to World (same composition pattern as `SpatialIndex`, `BTreeIndex`, `ReducerRegistry`). It owns its `QueryPlanResult` and manages tick advancement. `entities()` returns `&[Entity]` from the cached snapshot. `refresh()` returns `Result<RefreshOutcome, PlanExecError>` — `RefreshOutcome::Refreshed` if re-materialized, `RefreshOutcome::Suppressed` if debounce-suppressed. `RefreshOutcome` provides `was_refreshed()` and `was_suppressed()` convenience methods.

### System Scheduling Primitives

`Access` extracts component-level read/write metadata from any `WorldQuery` type. `Access::of::<(&mut Pos, &Vel)>(world)` returns a struct with two `FixedBitSet`s: reads (Vel) and writes (Pos), plus a `despawns: bool` flag. `conflicts_with()` detects whether two accesses violate the read-write lock rule — two bitwise ANDs over the component bitsets, plus despawn-vs-any-access blanket conflict. `has_any_access()` returns true if the access touches any component.

This is a building block for framework-level schedulers. Minkowski provides the access metadata; scheduling policy (dependency graphs, topological sort, parallel execution) is the framework's responsibility.

### Transaction Semantics

`Transact` is a closure-based trait. The primary entry point is `transact(&self, world, access, |tx, world| { ... })` which runs the closure in a retry loop, handling begin/commit/abort automatically. `transact_with(&self, world, access, |scope| { ... })` is an ergonomic variant that passes a `TxScope` bundling both `Tx` and `World` — methods like `scope.query()`, `scope.write()`, `scope.spawn()` don't require passing `world` explicitly. Building-block methods `begin()`, `try_commit()`, and `max_retries()` are also available for advanced patterns (e.g. parallel execute phases where multiple transactions read concurrently before committing sequentially).

`Tx` is the unified transaction handle. It does NOT hold `&mut World` — methods take `world` as a parameter. This split-phase design enables concurrent reads: `tx.query(&world)` uses `World::query_raw(&self)` (shared-ref, no ticks/cache, requires `ReadOnlyWorldQuery`), while writes go through `tx.write()`, `tx.remove()`, `tx.spawn()` which buffer into an internal `EnumChangeSet`. On successful commit, the changeset is applied atomically to World. `TxScope` wraps `Tx` + `World` for ergonomic use within `transact_with` — eliminates the need to pass `world` to every method call.

Three built-in strategies: `Sequential` (zero-cost passthrough — all ops delegate directly to World, commit always succeeds), `Optimistic` (live reads via `query_raw`, buffered writes into `EnumChangeSet`, tick-based validation at commit — `Err(Conflict)` if any accessed column was modified since begin, default 3 retries), `Pessimistic` (cooperative per-column locks acquired at begin, buffered writes, commit always succeeds — locks released on drop, default 64 retries with spin+yield backoff). `Optimistic` and `Pessimistic` are constructed with `::new(&world)` to capture a shared orphan queue handle.

Three tiers of mutation:

| Tier | API | Durability | Use case |
|---|---|---|---|
| Direct | `world.spawn()`, `world.insert()`, etc. | None | Single-threaded, no conflict detection |
| Transactional | `strategy.transact(world, access, \|tx, world\| { ... })` | In-memory | Concurrent access with conflict detection/retry |
| Durable | `Durable::new(strategy, wal, codecs).transact(...)` | WAL-backed | Crash-safe persistence via `minkowski-persist` crate |

`Durable<S>` (in `minkowski-persist`) wraps any `Transact` strategy. On successful commit, the forward changeset is written to the WAL before being applied to World. Failed attempts (retries) are not logged. WAL write failure panics — the durability invariant is non-negotiable.

Lock granularity is per-column `(ArchetypeId, ComponentId)`. `ColumnLockTable` is owned by `Pessimistic` strategy (not World — it's concurrency policy, not storage). Not MVCC — no version chains. Optimistic uses existing `changed_tick` infrastructure for validation. `World::query_raw(&self)` is the shared-ref read path — scans archetypes without touching cache or ticks.

**Entity ID lifecycle invariant**: entity IDs allocated during a transaction (`tx.spawn`) are tracked. On successful commit they become placed entities. On abort (drop without commit or conflict), the IDs are pushed to a shared `OrphanQueue` owned by World. World drains this queue automatically at the top of every `&mut self` method — bumping generations and recycling indices. No entity ID ever leaks, regardless of how the transaction ends. No manual drain step required.

### Choosing a Transaction Strategy

| | Sequential | Optimistic | Pessimistic |
|---|---|---|---|
| **Throughput** | Highest (zero overhead) | High (retries rare if low contention) | Medium (lock acquire/release cost) |
| **Contention handling** | N/A (single writer) | Retry on conflict (default 3×) | Block until lock acquired (default 64× spin+yield) |
| **Failure mode** | None — commit always succeeds | `Err(Conflict)` if column written by another thread since begin | Deadlock-free (total column ordering) but can starve under extreme contention |
| **Read path** | Direct `&mut World` | `query_raw(&World)` — shared ref, no cache/ticks | `query_raw(&World)` — shared ref, no cache/ticks |
| **Write path** | Direct mutation | Buffered in `EnumChangeSet`, applied at commit | Buffered in `EnumChangeSet`, applied at commit |
| **Best for** | Single-threaded game loops, batch setup, tooling | Multi-threaded with rare write overlap (e.g. spatial partitioned updates) | Multi-threaded with frequent write overlap (e.g. shared health pool, economy) |
| **Durable?** | Yes (wrap with `Durable<Sequential>`) | Yes (wrap with `Durable<Optimistic>`) | Yes (wrap with `Durable<Pessimistic>`) |

**Rules of thumb**:
- Start with `Sequential` unless you have a measured concurrency need.
- Use `Optimistic` when threads touch mostly disjoint component sets — retries are cheap but wasted work scales with conflict rate.
- Use `Pessimistic` when threads frequently write the same columns — locks prevent wasted work but add baseline overhead even without contention.
- All three compose with `Durable<S>` for WAL-backed crash safety.

### Reducer System

Typed reducers narrow what a closure *can* touch so that conflict freedom is provable from the type signature. Three execution models:

| Model | Handle types | Isolation | Conflict detection |
|---|---|---|---|
| Transactional | `EntityMut<C>`, `Spawner<B>`, `QueryWriter<Q>` | Buffered writes via EnumChangeSet | Runtime (optimistic ticks or pessimistic locks) |
| Scheduled | `QueryMut<Q>`, `QueryRef<Q>` | Direct `&mut World` (hidden) | Compile-time (Access bitsets) |
| Dynamic | `DynamicCtx` | Buffered writes via EnumChangeSet | Conservative (builder-declared upper bounds) |

**ComponentSet** declares a set of component types with pre-resolved IDs. **Contains<T, INDEX>** uses a const generic index to avoid coherence conflicts with generic tuple impls — the compiler infers INDEX at call sites. Both are macro-generated for tuples 1–12.

**Typed handles** hide World behind a facade exposing exactly the declared operations: `EntityRef<C>` (read-only), `EntityMut<C>` (read + buffered write + remove + optional despawn), `Spawner<B>` (entity creation via `reserve()`), `QueryRef<Q>` (read-only iteration), `QueryMut<Q>` (read-write iteration), `QueryWriter<Q>` (buffered query iteration via `WritableRef<T>`). EntityMut and Spawner hold `&mut EnumChangeSet` (not `&mut Tx`) for clean borrow splitting — Tx retains lifecycle ownership, handles borrow disjoint fields. `EntityMut::remove()` is bounded by `Contains<T, IDX>`. `EntityMut::despawn()` requires `register_entity_despawn` (sets despawn flag on Access).

**QueryWriter** iterates like a query but buffers writes through `ChangeSet` instead of mutating directly. `&T` items pass through unchanged; `&mut T` items become `WritableRef<T>` handles with `get`/`set`/`modify` methods. Uses manual archetype scanning (not `world.query()`) to avoid marking mutable columns as changed, which would cause self-conflict with optimistic validation. A separate `WriterQuery` trait (not on `WorldQuery`) defines the `&mut T` → `WritableRef<T>` mapping. Per-reducer `AtomicU64` tick state enables `Changed<T>` filter support. Compatible with `Durable` for WAL logging — the motivating use case.

**Dynamic reducers** trade compile-time precision for runtime flexibility. `DynamicReducerBuilder` (via `registry.dynamic(name, &mut world)`) declares upper-bound access with `can_read::<T>()`, `can_write::<T>()`, `can_spawn::<B>()`, `can_remove::<T>()`, `can_despawn()`. `DynamicCtx` provides `read`/`try_read`/`write`/`try_write`/`spawn`/`remove`/`try_remove`/`despawn`/`for_each` — accessing undeclared types, writing to read-only components, removing undeclared components, despawning without declaration, and iterating undeclared components all panic in all builds. `for_each::<Q>()` takes a `ReadOnlyWorldQuery` type parameter and yields typed slices per archetype — the iteration is fully typed, only the access validation is dynamic. Supports `Changed<T>` via per-reducer `Arc<AtomicU64>` tick state updated post-commit. Component IDs pre-resolved at registration; O(1) HashMap lookup by `TypeId` at runtime via `DynamicResolved`.

**ReducerRegistry** is external to World (same composition pattern as SpatialIndex). Registration type-erases closures with Access metadata and pre-resolved ComponentIds. Dispatch: `call()` for transactional reducers (entity, spawner, query writer — runs through `strategy.transact()`, entity is part of args), `run()` for scheduled query reducers (direct `&mut World`), `dynamic_call()` for dynamic reducers (routes through `strategy.transact()`). `id_by_name()` / `dynamic_id_by_name()` enable network dispatch. `access()` / `dynamic_access()` enable scheduler conflict analysis.

**EntityAllocator::reserve(&self)** provides lock-free entity ID allocation via `AtomicU32` — enables `Spawner` to work inside transactional closures where only `&World` is available.

## Key Conventions

- `pub` for user-facing API (`World`, `Entity`, `CommandBuffer`, `Bundle`, `WorldQuery`, `Table`, `EnumChangeSet`, `Changed`, `ChangeTick`, `ComponentId`, `SpatialIndex`, `Access`, `BTreeIndex`, `HashIndex`, `HasBTreeIndex`, `HasHashIndex`, `Transact`, `Tx`, `TxScope`, `Sequential`, `SequentialTx`, `Optimistic`, `Pessimistic`, `Conflict`, `TransactError`, `WorldMismatch`, `ReducerRegistry`, `ReducerId`, `QueryReducerId`, `DynamicReducerId`, `DynamicReducerBuilder`, `DynamicCtx`, `ComponentSet`, `Contains`, `EntityRef`, `EntityMut`, `QueryRef`, `QueryMut`, `QueryWriter`, `WritableRef`, `WriterQuery`, `Spawner`, `WorldStats`, `Expiry`, `WorldBuilder`, `PoolExhausted`, `HugePages`, `DeadEntity`, `InsertError`, `QueryPlanner`, `TablePlanner`, `Indexed`, `PlanNode`, `Predicate`, `Cost`, `IndexKind`, `JoinKind`, `AsEntityRef`, `QueryPlanResult`, `PlannerError`, `SubscriptionBuilder`, `SubscriptionError`, `CardinalityConstraint`, `PlanWarning`, `AggregateExpr`, `AggregateOp`, `AggregateResult`, `PlanExecError`, `MaterializedView`, `DebouncePolicy`). `pub(crate)` for internals (`BlobVec`, `Archetype`, `EntityAllocator`, `QueryCacheEntry`, `Tick`, `ColumnLockTable`, `OrphanQueue`, `TxCleanup`, `ResolvedComponents`, `DynamicResolved`). `ComponentRegistry` is `#[doc(hidden)] pub` — exposed only for derive macro codegen, not user code.
- `extern crate self as minkowski;` at crate root — allows `#[derive(Table)]` generated code (which references `::minkowski::*`) to resolve when used inside this crate's own tests.
- `#![allow(private_interfaces)]` at crate root — pub traits reference pub(crate) types in signatures. Intentional; fix when building public API facade.
- Every module has `#[cfg(test)] mod tests` with inline tests.
- `#[expect(dead_code)]` on fields/methods reserved for future phases.
- **Change detection invariant**: every path that hands out a mutable pointer to column data must either use `BlobVec::get_ptr_mut(row, tick)` (marks the column changed) or mark via the entry-point method (`World::query` for `&mut T`, `query_table_mut`, `query_table_raw`, `get_batch_mut`). `BlobVec::get_ptr` is the read path — writing through it silently bypasses `Changed<T>`. If you add a new mutable accessor, it must go through one of these two mechanisms.
- **Semantic review checklist**: every new primitive that touches concurrency, entity lifecycle, or cross-system state gets a "what can go wrong" review before implementation. The type system catches syntax; these catch design:
  1. Can this be called with the wrong World?
  2. Can Drop observe inconsistent state?
  3. Can two threads reach this through `&self`?
  4. Does dedup/merge/collapse preserve the strongest invariant?
  5. What happens if this is abandoned halfway through?
  6. Can a type bound be violated by a legal generic instantiation?
  7. Does the API surface of this handle permit any operation not covered by the Access bitset?
- **Transaction safety invariants**: any query path reachable from `&World` must be bounded by `ReadOnlyWorldQuery`. Any shared structure between World and a strategy uses `Arc` with a `WorldId` check at every entry point. Lock privilege in a `ColumnLockSet` can only escalate, never downgrade. Drop is the abort path — if a transaction can allocate engine resources (entity IDs, locks), it must be able to release them from Drop without `&mut World`, which means those resources route through interior-mutable shared handles (`OrphanQueue`, `Mutex<ColumnLockTable>`).
- **Error philosophy**: the decision to panic should lie with the user, not the library. If a condition is triggerable through public APIs (dead entity, pool exhaustion, invalid planner input, duplicate codec registration), return `Result` so the caller can `.unwrap()` if they want a panic, or handle the error otherwise. Panics are reserved for internal invariants that users cannot violate through legal API usage — forged IDs, unsafe trait contract violations, access boundary crossings. Key error types: `DeadEntity` (`insert`/`try_insert` on a despawned entity), `InsertError` (`DeadEntity | PoolExhausted` for `insert`/`try_insert`), `TransactError` (`Conflict | WorldMismatch` for transaction operations), `WorldMismatch` (cross-world strategy/plan usage), `PlannerError` (`InvalidPredicate | UnregisteredComponent | BuilderOrder` for planner builder methods), `PlanExecError` (`WorldMismatch | JoinNotSupported` for plan execution methods), `ApplyError` (changeset application failures including `AlreadyPlaced` for duplicate spawns), `CodecError` (including `DuplicateComponentName` and `DuplicateStableName` for codec registration).
- **Assert boundary rule**: if violating a check makes the scheduler's Access bitset disagree with reality, it's `assert!`. The scheduler runs in release builds; the checks that protect its invariants must also run in release builds. `debug_assert!` is for checks within an already-correct access boundary — like verifying that a read through `can_write` still works (over-declared but sound). Crossing the boundary from read declaration to write operation is unsound, full stop.
- **Verify-before-design rule**: before designing features that depend on existing APIs, `grep` for the actual current methods/types. Confirm which ones exist vs which need to be created. Past bugs: assuming `EntityAllocator::reserve()` had atomics (it didn't exist), proposing `&mut World` in reducer APIs (unsound). The type system doesn't catch "assumed this method exists" — verification does.
- **Derive macro visibility rule**: after implementing or modifying derive macros or code generation, always test the generated code from an external example/binary target (`minkowski-examples`) to verify `pub` vs `pub(crate)` visibility is correct. In-crate tests won't catch `pub(crate)` leaking into generated code.
- **Change detection audit rule**: when implementing change detection or mutation tracking, enumerate ALL mutation paths (spawn, get_mut, insert, remove, query `&mut T`, query_table_mut, query_table_raw, changeset apply, any new accessor) and verify each path triggers detection. Consider same-tick interleaving edge cases where two mutations in the same tick must both be visible.
- **Bypass-path invariant rule**: when adding a new code path that skips the normal access/mutation pipeline (e.g., `query_table_mut` bypassing `world.query()`, manual archetype scanning in QueryWriter, `query_raw` skipping cache/ticks), verify that ALL existing invariants are maintained through the bypass: change detection ticks, query cache invalidation, Access bitset accuracy, entity lifecycle tracking. The normal pipeline enforces these automatically; bypass paths must enforce them manually or explicitly document which invariants they intentionally skip and why.
- **Bug fix workflow rule**: when a specific bug is reported, address it directly with targeted investigation — don't route through brainstorming or design exploration workflows. Brainstorming is for new features and design decisions, not concrete bugs with reproduction steps.
- **Reducer determinism rule**: reducers must be pure functions of their handle state and args. No RNG, system time, HashMap iteration, I/O, or global mutable state. When writing example reducers, use deterministic alternatives (BTreeMap, args-provided seeds, pre-computed values). See `docs/reducer-correctness.md`.
- **Drop cleanup rule**: if Drop needs to clean up engine state, the cleanup path must be reachable from `&self`. This constrains where state can live. If it's on World, Drop can't reach it. If it's behind `Arc<Mutex<_>>` shared between World and the transaction, Drop can. Every future resource that transactions can allocate — entity IDs, archetype slots, reserved capacity — must follow this pattern or it will leak on abort.

# Unifying principle behind every ID type in the system:

| Type | Proof at construction | Residue after erasure |
|---|---|---|
| `ComponentId` | `T: Component`, layout, drop fn | `usize` |
| `ArchetypeId` | Sorted component set, column allocation | `usize` |
| `Entity` | Generational uniqueness, archetype placement | `u64` |
| `ReducerId` | Typed closure, access set, resolved components, args type | `usize` |
| `QueryReducerId` | `WorldQuery` bound, column access pattern | `usize` |
| `WorldId` | Unique allocator identity | `u64` |
| `TxId` | Transaction lifecycle, cleanup registration | `u64` |
| `ChangeTick` | Monotonic ordering, causal relationship | `u64` |

Every one of these is a small integer that stands in for a proof the type system already verified. The runtime never re-checks the proof — it trusts the integer because the only way to obtain it was through a typed construction path that enforced the invariants.

The corollary is that forging an ID — constructing one from a raw integer without going through the typed path — is the one thing that can violate the system's guarantees. Which is why all the constructors are `pub(crate)` or behind registration APIs. The user can hold IDs, copy them, store them, send them. They can't fabricate them. The type system did its work at the gate and the ID is the stamp that says "verified."

## Dependencies

### Core (`minkowski`)

| Crate | Version | Features | Why this crate / why this version |
|---|---|---|---|
| `fixedbitset` | 0.5 | — | Archetype component bitmasks for query matching. Compact, no-alloc subset checks via bitwise AND. |
| `parking_lot` | 0.12 | — | Mutex for `ColumnLockTable`, `OrphanQueue`, `Durable` WAL lock. Smaller, faster than `std::sync::Mutex`, no poisoning. |
| `rayon` | 1 | — | `par_for_each` parallel iteration. Work-stealing thread pool. |
| `atomic` | 0.6 | — | `Atomic<T>` for generic atomics in pool side table. Needed for types wider than platform word. |
| `libc` | 0.2 | — | `mmap`/`madvise` syscalls for pool slab allocation. |
| `memmap2` | 0.9 | — | Memory-mapped file I/O for zero-copy snapshot loading. |
| `loom` | 0.7 | optional | Exhaustive concurrency model checker. Behind `--features loom`. |
| `minkowski-derive` | path | — | `#[derive(Table)]` proc macro (syn/quote/proc-macro2). |

### Persistence (`minkowski-persist`)

| Crate | Version | Features | Why |
|---|---|---|---|
| `rkyv` | 0.8 | `alloc`, `bytecheck` | Zero-copy serialization for WAL records and snapshots. `bytecheck` for validation on untrusted input; `alloc` for `ArchivedVec`. |
| `memmap2` | 0.9 | — | Memory-mapped snapshot loading. |
| `parking_lot` | 0.12 | — | WAL segment lock. |
| `fixedbitset` | 0.5 | — | Archetype bitmask serialization in snapshots. |
| `thiserror` | 2 | — | Derive macros for `std::error::Error` impls. v2 uses `core::error::Error` (edition 2024). |
| `crc32fast` | 1 | — | CRC32 checksums for WAL frame integrity. |

### Dev / Bench / Examples

| Crate | Version | Purpose |
|---|---|---|
| `criterion` | 0.5 | Benchmark harness |
| `tempfile` | 3 | Temporary directories for persistence benchmarks |
| `fastrand` | — | Example RNG (deterministic seeds) |
| `cargo-nextest` | CI | Parallel test runner for Miri (installed via `taiki-e/install-action`) |
