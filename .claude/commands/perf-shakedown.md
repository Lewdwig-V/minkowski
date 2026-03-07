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
- `query/iter.rs` — `QueryIter::next`, `for_each`, `for_each_chunk`, `par_for_each`
- `query/fetch.rs` — `init_fetch`, `fetch`, `as_slice`

**Mutation (spawn, migrate, changeset apply)** — `crates/minkowski/src/`
- `world.rs` — `spawn`, `insert`, `remove`, `despawn`, `despawn_batch` (group-sort-sweep), `get_mut`, `get_batch_mut`, `query`, `query_table_mut`
- `bundle.rs` — `Bundle::put`, `component_ids`
- `changeset.rs` — `EnumChangeSet::apply`, `record_insert`, arena allocation

**Reducer (per-entity iteration through handles)** — `crates/minkowski/src/`
- `reducer.rs` — `QueryWriter::for_each` (manual archetype scan), `QueryMut::for_each`/`for_each_chunk`, `QueryRef::for_each`, `DynamicCtx::for_each`, `EntityMut::get`/`set`/`remove`, `Spawner::spawn`

**Persistence (I/O hot paths)** — `crates/minkowski-persist/src/`
- `wal.rs` — `append`, `replay_from`, `scan_last_seq`
- `snapshot.rs` — `save`, `load`, `load_zero_copy`
- `codec.rs` — `serialize`, `deserialize`, `raw_copy_size` usage
- `format.rs` — `serialize_record`, `deserialize_record`

**Entity allocation (concurrent spawner path)** — `crates/minkowski/src/`
- `entity.rs` — `EntityAllocator::reserve` (atomic), `alloc`, `materialize_reserved`

**Transaction (commit path)** — `crates/minkowski/src/`
- `transaction.rs` — `try_commit`, `begin`, tick validation, changeset apply
- `lock_table.rs` — `acquire`, `release`

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
> 1. **`for_each` vs `for_each_chunk`**: Any `for_each` call on numeric/math data (positions, velocities, forces) where `for_each_chunk` would yield typed slices that LLVM can auto-vectorize. `for_each_chunk` yields `&[T]`/`&mut [T]` slices per archetype — far more vectorizable than per-element callbacks.
> 2. **Vectorization blockers in chunk bodies**: Inside `for_each_chunk` closures, look for: function calls that won't inline (non-generic, cross-crate, or `dyn`), per-element branching (`if`/`match`), scalar math on types that could be SIMD-width (`f32` operations that could be `[f32; 4]`), and index-based array access instead of slice iteration.
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
> 3. **Snapshot zero-copy path**: In `snapshot.rs` `load_zero_copy`, verify that components with `raw_copy_size == Some` actually take the direct-copy path and don't fall through to `codecs.deserialize()`. Check `save` for avoidable allocations.
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
> 2. **Allocation in hot loops**: `Vec::new()`, `Box::new()`, `HashMap::new()`, or `.collect()` inside per-entity iteration. Each allocation hits the global allocator. Look inside `for_each`, `for_each_chunk`, and `par_for_each` closures.
> 3. **Redundant lookups**: Repeated `HashMap::get` for the same key, repeated `entity_locations[idx]` lookups that could be hoisted, or repeated `ComponentId` resolution that could be cached.
> 4. **Branch patterns**: `Option::unwrap()` or `match` in inner loops where the variant is always the same — branch predictor will handle it, but `unwrap_unchecked` (with safety comment) or restructuring may help.
> 5. **HashMap vs Vec**: `HashMap` used on a hot path where a dense `Vec` indexed by `ComponentId` (which is `usize`) or `ArchetypeId` would give O(1) indexed access without hashing.
> 6. **No UB**: Flag any suggestion from above that could introduce undefined behavior. Safety is non-negotiable — only suggest optimizations that preserve soundness.
>
> Report with `PERF-CRITICAL` / `PERF-OPPORTUNITY` / `PERF-OK` ratings and `file:line` references.

## Phase 3 — Profiling Recommendations

After all 4 agents complete, aggregate their reports into the output format below. Then add:

### Benchmark Coverage

1. List all benchmark files: run `ls crates/minkowski/benches/`
2. For each `PERF-CRITICAL` finding, check if a benchmark covers that hot path. If not, suggest a targeted benchmark.
3. Suggest specific `cargo bench -p minkowski -- <filter>` commands for existing benchmarks that cover affected paths.

### New Hot Path Discovery

1. If diff-scoped: grep the changed files for new `for_each`/`for_each_chunk`/`par_for_each` call sites, new `unsafe` blocks with pointer arithmetic, or new loops over `entity_locations`/`archetypes` that are NOT in the static hot path list above.
2. Flag these as "consider adding to the hot path list in `/perf-shakedown`".
3. Suggest `cargo flamegraph` or `perf record` commands for examples that exercise the changed code. Reference examples:
   - `boids` — query reducers + spatial grid (5K entities)
   - `life` — QueryMut + Table + undo/redo (64x64 grid)
   - `nbody` — Barnes-Hut + query reducers (2K entities)
   - `persist` — Durable QueryWriter + WAL + rkyv snapshots (100 entities, 3 archetypes)
   - `battle` — multi-threaded EntityMut reducers (500 entities)
   - `transaction` — 3 transaction strategies (100 entities)
   - `scheduler` — conflict detection + batch scheduling (6 systems)
   - `reducer` — all reducer handle types + structural mutations
   - `index` — B-tree + hash index lookups (200 entities)

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
- `// PERF: No for_each_chunk — WritableRef indirection is inherent to buffered writes.`
- `// PERF: Per-row Vec::new() unavoidable — ColumnData::values owns Vec<Vec<u8>>.`
- `// PERF: Full WAL scan on open required for crash recovery — no index or footer.`

Only annotate after user confirmation that the finding is non-actionable. Do not pre-annotate speculatively.
