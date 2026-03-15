---
description: Performance shakedown — data layout, vectorization, rkyv, cache analysis across hot paths
args:
  - name: target
    description: "'all' for full codebase, file path/module filter, or omit for diff-based (default on feature branches)"
    required: false
allowed-tools: Bash, Read, Glob, Grep, Agent
---

Run a performance shakedown on the Minkowski ECS codebase. This is a three-phase analysis: Phase 1 determines scope, Phase 2 dispatches 4 parallel subagents for deep analysis, Phase 3 synthesizes profiling recommendations.

## Phase 1 — Scope

Determine which files to analyze:

1. If `$ARGUMENTS` is `all`, or if you are on the `main` branch with no arguments: use the **full hot path list** below.
2. If `$ARGUMENTS` is a file path or module name: filter the hot path list to matching entries.
3. Otherwise (no arguments on a feature branch): run `git diff main...HEAD --name-only` and intersect with the hot path list below. Only files that appear in BOTH the diff and the hot path list are in scope.
4. If no changed files touch any hot path: report "No hot paths touched by this branch" and skip directly to Phase 3 discovery. Still check the diff for new hot path candidates.

### Static Hot Path List

**Storage (per-entity, per-archetype)** — `crates/minkowski/src/`
- `storage/blob_vec.rs` — `push`, `get_ptr`, `get_ptr_mut`, `swap_remove`, `swap_remove_no_drop`, `drop_in_place`, `copy_unchecked`, `set_len`
- `storage/archetype.rs` — `push`, `swap_remove`, entity/column iteration
- `storage/sparse.rs` — `PagedSparseSet::dense_index` (page lookup + generation check), `insert`, `get`, `remove`, `remove_internal`, `iter`; `SparseStorage::remove_all` (per-despawn)

**Query (per-entity iteration)** — `crates/minkowski/src/`
- `query/iter.rs` — `QueryIter::next`, `for_each`, `for_each_chunk`, `par_for_each`, `mark_iterated`
- `query/fetch.rs` — `init_fetch`, `fetch`, `as_slice`

**Mutation (spawn, migrate, changeset apply)** — `crates/minkowski/src/`
- `world.rs` — `spawn`, `insert`, `remove`, `despawn`, `despawn_batch` (group-sort-sweep), `get_mut`, `get_batch_mut`, `query`, `query_table_mut`, `has_changed` (archetype scan per call)
- `bundle.rs` — `Bundle::put`, `component_ids`
- `changeset.rs` — `EnumChangeSet::apply`, `changeset_insert_raw` (overwrite path), `changeset_remove_raw` (migration path), `record_insert` (arena alloc + Vec push), `Arena::alloc` (alignment + capacity check + copy)

**Reducer (per-entity iteration through handles)** — `crates/minkowski/src/`
- `reducer.rs` — `QueryWriter::for_each` (manual archetype scan), `WritableRef::modify` (clone + insert_raw), `WritableRef::set` (insert_raw), `QueryMut::for_each` (slice-based), `QueryRef::for_each` (slice-based), `DynamicCtx::for_each` (slice-based), `DynamicCtx::write` (HashMap lookup + insert_raw), `EntityMut::get`/`set`/`remove`, `Spawner::spawn`

**Persistence (I/O hot paths)** — `crates/minkowski-persist/src/`
- `wal.rs` — `append`, `replay_from`, `scan_last_seq`
- `snapshot.rs` — `save`, `save_to_bytes`, `load`, `load_from_bytes`
- `codec.rs` — `serialize`, `deserialize`, `raw_copy_size` usage
- `format.rs` — `serialize_record`, `deserialize_record`

**Entity allocation (concurrent spawner path)** — `crates/minkowski/src/`
- `entity.rs` — `EntityAllocator::reserve` (atomic), `alloc`, `materialize_reserved`

**Query planner & executor (plan execution)** — `crates/minkowski/src/`
- `planner.rs` — `CompiledForEach` / `CompiledForEachRaw` closures (per-entity callback dispatch), `passes_change_filter` (archetype-level `Changed<T>` tick comparison), `ScratchBuffer::sorted_intersection` (join entity merge), `EntityCollector` closures (join left/right collection), `IndexDriver` lookup closures (pre-bound `eq_lookup_fn` / `range_lookup_fn` for `IndexGather`), `make_extractor` (per-entity `world.get` in aggregates), `AggregateAccum::feed` / `feed_count` (accumulator hot loop), `lower_to_vectorized` (plan build, not execution — but called on every `ScanBuilder::build()`)

**Materialized views (per-refresh)** — `crates/minkowski/src/`
- `view.rs` — `MaterializedView::refresh` (world_id check + debounce gate + `plan.execute` + `extend_from_slice` cache copy)

**Transaction (commit path)** — `crates/minkowski/src/`
- `transaction.rs` — `try_commit`, `begin`, tick validation, changeset apply
- `lock_table.rs` — `acquire`, `release`

### Benchmark Baselines (as of v1.2.0)

Reference numbers from `cargo bench -p minkowski-bench` on the dev machine. Use these to contextualize findings — a "PERF-CRITICAL" on a path that takes 1.5µs total is less urgent than one on a 250µs path.

| Benchmark | Time | Per-entity | Notes |
|---|---|---|---|
| `simple_iter/for_each_chunk` | 1.5 µs | 0.15 ns | SIMD-friendly baseline |
| `simple_iter/for_each` | 14.5 µs | 1.45 ns | ~10x slower — per-element callback overhead |
| `simple_iter/par_for_each` | 289 µs | — | Rayon overhead dominates at 10K entities |
| `reducer/query_mut_10k` | 1.5 µs | 0.15 ns | Slice-based iteration |
| `reducer/query_writer_10k` | 93 µs | 9.3 ns | **~62x query_mut** — buffered write overhead |
| `reducer/dynamic_for_each_10k` | 132 µs | 13.2 ns | **~88x query_mut** — identity-hashed lookup + buffered writes |
| `changeset/record_10k_inserts` | 70 µs | 7.0 ns | Arena alloc + Vec push (recording only) |
| `changeset/apply_10k_overwrites` | 123 µs | 12.3 ns | Entity lookup + memcpy + tick mark (apply only) |
| `changeset/record_apply_10k` | 148 µs | 14.8 ns | Full round-trip (record + apply) |
| `changeset/new_drop_empty` | 10 ns | — | Per-transaction allocation cost |
| `simple_insert/batch` | 1.74 ms | 174 ns | Direct spawn |
| `simple_insert/changeset` | 2.58 ms | 258 ns | Changeset spawn ~1.5x direct |
| `add_remove/add_remove` | 1.30 ms | 130 ns | Archetype migration cost |
| `heavy_compute/sequential` | 9.3 µs | 9.3 ns | 4x4 matrix inversion |
| `heavy_compute/parallel` | 402 µs | — | Rayon overhead dominates at 1K entities |
| `schedule/5_systems_10k` | 30.5 µs | — | Access conflict detection + dispatch |
| `serialize/snapshot_save` | 495 µs | 495 ns | 1K entities |
| `serialize/snapshot_load` | 311 µs | 311 ns | 1K entities |
| `serialize/wal_append` | 1.22 µs | — | Single mutation |
| `serialize/wal_replay` | 1.72 ms | 1.72 µs | 1K mutations |

| `planner/scan_for_each_10k` | 9.5 µs | 0.95 ns | Planner scan — ~2.5x `query_for_each` (type-erased dispatch overhead) |
| `planner/query_for_each_10k` | 3.9 µs | 0.39 ns | Direct `world.query()` baseline for comparison |
| `planner/btree_range_10pct` | 11.4 µs | 11.4 ns/match | BTree `IndexGather` — 1K entities out of 10K |
| `planner/hash_eq_1` | 39 ns | — | Hash `IndexGather` — single entity lookup |
| `planner/custom_filter_50pct` | 63.4 µs | 12.7 ns | `Predicate::custom` — per-entity `Arc<dyn Fn>` + `world.get()` |
| `planner/changed_skip_10k` | 6.8 ns | — | `Changed<T>` skip — all archetypes filtered (tick comparison only) |
| `planner/aggregate_count_sum_10k` | 76.1 µs | 7.6 ns | COUNT + SUM via type-erased extractors |
| `planner/manual_count_sum_10k` | 5.8 µs | 0.58 ns | Same COUNT + SUM via direct query — **~13x faster** |
| `planner/execute_collect_10k` | 14.7 µs | 1.47 ns | `execute()` — entity push into scratch buffer |

**Missing benchmark coverage** (no bench exists for these hot paths):
- `MaterializedView::refresh` (debounce gate + plan execute + cache copy)

### Regression Guards

These optimizations were applied in the v1.1.0 profiling investigation (PR #90, #92).
If any of them are accidentally reverted, the corresponding benchmarks will regress.
Agents should verify these invariants are maintained when modifying the listed files.

1. **`#[inline]` on changeset hot paths** (`changeset.rs`, `reducer.rs`):
   `Arena::alloc`, `Arena::get`, `record_insert`, `insert_raw`, `WritableRef::set`,
   `WritableRef::modify` must remain `#[inline]`. Removing any of these causes ~30%
   QueryWriter regression — profiling showed 33% of self-time was function call
   prologue/epilogue on these small functions. Benchmark: `reducer/query_writer_10k`.

2. **QueryWriter pre-allocation** (`reducer.rs:QueryWriter::for_each`):
   The entity-count pre-scan + `mutations.reserve()` + `arena.reserve()` block must
   remain in `for_each`, capped at `MAX_PREALLOC_MUTATIONS` (64K). Removing it causes
   ~7% QueryWriter regression from Vec/Arena reallocation during iteration. The cap
   prevents overallocation for conditional-update reducers that match many entities but
   write to few. Benchmark: `reducer/query_writer_10k`.

3. **Identity hasher for `DynamicResolved`** (`reducer.rs:TypeIdHasher`):
   The `HashMap<TypeId, ComponentId, TypeIdBuildHasher>` in `DynamicResolved` must use
   the identity hasher, not the default `SipHash`. `TypeId` is a compiler-generated
   `u64` — SipHash was spending 29% of DynamicCtx time on unnecessary cryptographic
   hashing. Benchmark: `reducer/dynamic_for_each_10k`.

4. **`#[inline]` on DynamicCtx hot paths** (`reducer.rs`):
   `DynamicCtx::write` and `DynamicResolved::lookup` must remain `#[inline]`.
   Benchmark: `reducer/dynamic_for_each_10k`.

5. **Query planner `#[inline]` on accessors** (`planner.rs`):
   `Cost::rows`, `Cost::cpu`, `Cost::total`, `PlanNode::cost`, `PlanNode::estimated_rows`,
   `ProbeSet::contains` must remain `#[inline]`. These are called during plan build and
   cost comparison.
   Benchmark: none yet — exercise via `cargo run -p minkowski-examples --example planner --release`.

6. **Query planner `PredicateKind` is fieldless for `Eq`/`Range`** (`planner.rs`):
   `PredicateKind::Eq` and `PredicateKind::Range` must remain unit variants (no `String`
   payload). The Debug impl uses `component_name` from the parent `Predicate` struct.
   Adding payload strings back would re-introduce 2 heap allocations per predicate.

7. **Query planner `IndexDriver` uses pre-bound lookup closures** (`planner.rs`):
   `eq_lookup_fn` / `range_lookup_fn` / `all_entities_fn` are captured at plan-build
   time (Phase 3 `IndexDriver` construction), not resolved at execution time. This
   avoids per-execution `dyn Any` downcasting. The lookup value is pre-bound into the
   closure via `Predicate`'s `lookup_value` field.

8. **`ScratchBuffer::sorted_intersection` avoids reallocation** (`planner.rs`):
   Join entity merge uses in-place sort + dedup on the existing `Vec<Entity>`, avoiding
   temporary `HashSet` allocation. The sorted intersection result is appended to the
   same buffer via `copy_within` + `truncate`.

### Known Bottlenecks (profiled, residual after optimization)

These have been analyzed and are documented here to prevent re-discovery:

1. **QueryWriter ~9x overhead** — after inlining and pre-allocation (down from 16x).
   Remaining cost is inherent to buffered writes: `clone()` + arena copy per entity in
   `WritableRef::modify`, then per-entity `entity_locations` lookup + `column_index` +
   `copy_nonoverlapping` in `changeset_insert_raw` during apply. The apply phase (62%
   of profile) is dominated by per-mutation enum match (computed jump table) and entity
   location lookup. A batch overwrite fast path in `apply()` could reduce this further
   (~20% estimated) but adds significant complexity for moderate gain.

2. **DynamicCtx ~1.8x over QueryWriter** — after identity hasher (down from ~2.7x).
   Remaining cost is the collect-then-write `Vec` pattern (structural to dynamic
   reducers — can't inline writes during iteration because the type is resolved at
   runtime) plus the `assert!` write-permission check per `ctx.write()` call.

3. **`par_for_each` / parallel overhead** — rayon's thread pool spawn + work stealing amortizes poorly below ~50K entities. At 10K entities, sequential `for_each` (slice-based) is 430x faster. This is a rayon characteristic, not a Minkowski bug. Users should use `par_for_each` only for large entity counts or expensive per-entity work (like `heavy_compute`).

4. **Changeset spawn ~1.5x direct** — arena allocation + mutation log overhead on top of the same archetype push work. Inherent to the data-driven mutation model.

5. **Query planner `CompiledForEach` is `Box<dyn FnMut>`** — per-entity callback through `&mut dyn FnMut(Entity)` prevents SIMD vectorization. Inherent to type-erased plan composition — the plan is compiled once and re-executed with different callbacks. Custom predicates (`Predicate::custom`) also use `Arc<dyn Fn>` for per-entity filtering. Annotated with `// PERF:` at the type alias.

6. **`MaterializedView::refresh` copies the entity list** — `extend_from_slice` from the plan's scratch buffer into the view's owned `Vec<Entity>` on every re-materialization. Borrowing the scratch directly would tie `entities()` lifetime to `&mut self` from refresh, breaking the common pattern of `view.refresh(); for e in view.entities() { world.get(e) }`. The copy is O(N) where N is result size, typically small for subscription queries.

## Phase 2 — Analysis

Read the scoped files identified in Phase 1. Then dispatch **all 4 agents in a single message** (parallel execution) using the Agent tool. Pass each agent the list of scoped files and the specific functions to examine from the hot path list.

**Important: `// PERF:` annotations** — Code on hot paths may contain `// PERF:` comments explaining why a pattern that *looks* suboptimal is intentional or unavoidable. These were added by previous shakedown runs after analysis determined the pattern is non-actionable. When an agent encounters a `// PERF:` comment, it should report the pattern as `PERF-OK (annotated)` with the rationale from the comment, not re-flag it as an issue. Include this instruction in each agent prompt.

### Agent 1: data-layout

Use the Agent tool with this prompt:

> You are analyzing Minkowski ECS data structures for performance. Read these files: [SCOPED FILE LIST].
>
> For each struct definition on a hot path, check:
> 1. **Field ordering**: Are fields ordered largest-to-smallest to minimize padding? Use `std::mem::size_of` reasoning — a `u8` between two `u64`s wastes 7 bytes of padding.
> 2. **Pointer chasing**: Flag any `Box`, `Vec`, `String`, `HashMap`, or other heap-owning type inside a component that is iterated per-entity. Each heap indirection is a cache miss.
> 3. **Component size**: Flag components >64 bytes that appear in hot queries — candidates for splitting rarely-accessed fields into separate components.
> 4. **Internal indirection**: Check `BlobVec`, `Archetype`, and `QueryCacheEntry` internal fields for unnecessary `Box` or `Arc` wrapping of small data.
> 5. **`#[repr(C)]` for persistence**: Check persistent component types (those registered with `CodecRegistry`) for `#[repr(C)]` — without it, `raw_copy_size` returns `None` and the zero-copy snapshot path falls back to typed deserialization.
>
> Report findings as:
> - `PERF-CRITICAL`: heap-owning type in per-entity hot loop, >64 byte component in a hot query
> - `PERF-OPPORTUNITY`: suboptimal field ordering, missing `#[repr(C)]` on persistent component
> - `PERF-OK`: struct checked, no issues (brief note)
>
> Include `file:line` references for every finding.

### Agent 2: vectorization

Use the Agent tool with this prompt:

> You are analyzing Minkowski ECS iteration paths for auto-vectorization fitness. Read these files: [SCOPED FILE LIST].
>
> Check:
> 1. **Vectorization blockers in for_each bodies**: Inside `for_each` closures (which yield typed slices per archetype), look for: function calls that won't inline (non-generic, cross-crate, or `dyn`), per-element branching (`if`/`match`), scalar math on types that could be SIMD-width (`f32` operations that could be `[f32; 4]`), and index-based array access instead of slice iteration. Note: `QueryIter::for_each_chunk` on `world.query()` is the underlying slice method; reducer handles (`QueryMut`, `QueryRef`, `DynamicCtx`) expose `for_each` which is slice-based.
> 3. **Component alignment**: Check component types used in hot numeric loops for `#[repr(align(16))]` or naturally 16-byte-aligned types. BlobVec provides 64-byte column alignment, but component layout determines SIMD packing.
> 4. **Build config**: Verify `.cargo/config.toml` has `target-cpu=native` for platform-specific SIMD instructions.
> 5. **QueryWriter iteration**: `QueryWriter::for_each` does manual archetype scanning. Check if its inner loop has vectorization blockers — the `WritableRef` indirection may prevent LLVM from seeing contiguous memory.
>
> Report with `PERF-CRITICAL` / `PERF-OPPORTUNITY` / `PERF-OK` ratings and `file:line` references.

### Agent 3: rkyv-compat

Use the Agent tool with this prompt:

> You are analyzing Minkowski ECS persistence hot paths for rkyv serialization efficiency. Read these files: [SCOPED FILE LIST].
>
> Check:
> 1. **`raw_copy_size` coverage**: For each component type registered with `CodecRegistry`, determine if `raw_copy_size` would return `Some` (i.e., `size_of::<T>() == size_of::<T::Archived>()`). If not, explain why — missing `#[repr(C)]`, non-trivial archived layout (e.g., `String`, `Vec`), or platform-dependent sizes.
> 2. **WAL append/replay**: In `wal.rs` `append` and `replay_from`, look for unnecessary heap allocations, redundant copies, or intermediate `Vec<u8>` buffers that could be eliminated.
> 3. **Snapshot zero-copy path**: In `snapshot.rs` `load` / `load_from_bytes`, verify that components with `raw_copy_size == Some` actually take the direct-copy path and don't fall through to `codecs.deserialize()`. Check `save` for avoidable allocations.
> 4. **Codec closures**: In `codec.rs`, examine the `serialize_fn` and `deserialize_fn` closures for avoidable work — unnecessary clones, redundant validation, allocations that could be reused.
> 5. **Format layer**: In `format.rs`, check `serialize_record`/`deserialize_record` for intermediate buffers or copies between rkyv and the wire format.
>
> Report with `PERF-CRITICAL` / `PERF-OPPORTUNITY` / `PERF-OK` ratings and `file:line` references.

### Agent 4: cache-and-misc

Use the Agent tool with this prompt:

> You are analyzing Minkowski ECS hot paths for cache efficiency and general performance issues. Read these files: [SCOPED FILE LIST].
>
> Check:
> 1. **False sharing**: Are any `AtomicU64`/`AtomicU32` fields on the same cache line (64 bytes) as frequently-written non-atomic data? Check `Tick`, `EntityAllocator`, `OrphanQueue` atomics.
> 2. **Allocation in hot loops**: `Vec::new()`, `Box::new()`, `HashMap::new()`, or `.collect()` inside per-entity iteration. Each allocation hits the global allocator. Look inside `for_each`, `for_each_chunk` (on `QueryIter`), and `par_for_each` closures.
> 3. **Redundant lookups**: Repeated `HashMap::get` for the same key, repeated `entity_locations[idx]` lookups that could be hoisted, or repeated `ComponentId` resolution that could be cached.
> 4. **Branch patterns**: `Option::unwrap()` or `match` in inner loops where the variant is always the same — branch predictor will handle it, but `unwrap_unchecked` (with safety comment) or restructuring may help.
> 5. **HashMap vs Vec**: `HashMap` used on a hot path where a dense `Vec` indexed by `ComponentId` (which is `usize`) or `ArchetypeId` would give O(1) indexed access without hashing.
> 6. **No UB**: Flag any suggestion from above that could introduce undefined behavior. Safety is non-negotiable — only suggest optimizations that preserve soundness.
>
> Report with `PERF-CRITICAL` / `PERF-OPPORTUNITY` / `PERF-OK` ratings and `file:line` references.

## Phase 3 — Profiling Recommendations

After all 4 agents complete, aggregate their reports into the output format below. Then add:

### Benchmark Coverage

1. List all benchmark files: run `ls crates/minkowski-bench/benches/`
2. For each `PERF-CRITICAL` finding, check if a benchmark covers that hot path. If not, suggest a targeted benchmark.
3. Suggest specific `cargo bench -p minkowski-bench -- <filter>` commands for existing benchmarks that cover affected paths. Also check `cargo bench -p minkowski-persist` for persistence-specific benchmarks.

### New Hot Path Discovery

1. If diff-scoped: grep the changed files for new `for_each`/`for_each_chunk`/`par_for_each` call sites, new `unsafe` blocks with pointer arithmetic, or new loops over `entity_locations`/`archetypes` that are NOT in the static hot path list above.
2. Flag these as "consider adding to the hot path list in `/perf-shakedown`".
3. Suggest `cargo flamegraph` or `perf record` commands for examples that exercise the changed code. Reference examples:
   - `boids` — query reducers + spatial grid (5K entities)
   - `life` — QueryMut + Table (64x64 grid)
   - `nbody` — Barnes-Hut + query reducers (2K entities)
   - `persist` — Durable QueryWriter + WAL + rkyv snapshots (100 entities, 3 archetypes)
   - `battle` — multi-threaded EntityMut reducers (500 entities)
   - `transaction` — 3 transaction strategies (100 entities)
   - `scheduler` — conflict detection + batch scheduling (6 systems)
   - `reducer` — all reducer handle types + structural mutations
   - `index` — B-tree + hash index lookups + planned queries (200 entities)
   - `planner` — query planner: scan, filter, join, subscription, aggregate execution (1K entities)
   - `materialized_view` — MaterializedView: debounced subscription refresh (200 entities)

## Output Format

Present the final report as:

```
## Performance Shakedown Report

### Scope
[List files analyzed. Note whether this was diff-scoped or full-codebase.]

### Data Layout
[Agent 1 findings, verbatim with severity ratings and file:line refs]

### Vectorization
[Agent 2 findings]

### rkyv Compatibility
[Agent 3 findings]

### Cache & Miscellaneous
[Agent 4 findings]

### Profiling Recommendations
[Benchmark coverage gaps, suggested commands]

### Hot Path List Maintenance
[Any suggested additions or removals to the static list in this command]

### Non-Actionable Annotations
[List any findings deemed non-actionable with rationale, to be added as // PERF: comments]
```

## Phase 4 — Annotate Non-Actionable Findings

After presenting the report and discussing findings with the user, add `// PERF:` comments to hot path code for any findings confirmed as non-actionable. This prevents future shakedown runs from re-flagging the same patterns.

Format: `// PERF: <concise rationale for why this is intentional or unavoidable>`

Place the comment immediately above or on the line containing the pattern. Examples:
- `// PERF: Per-item iteration only — WritableRef indirection is inherent to buffered writes.`
- `// PERF: Per-row Vec::new() unavoidable — ColumnData::values owns Vec<Vec<u8>>.`
- `// PERF: Full WAL scan on open required for crash recovery — no index or footer.`

Only annotate after user confirmation that the finding is non-actionable. Do not pre-annotate speculatively.
