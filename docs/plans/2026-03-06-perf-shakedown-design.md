# Performance Shakedown Command Design

## Problem

Minkowski has no systematic way to audit performance across its hot paths. The existing `/minkowski:optimize` command is conversational guidance — it tells the user *what* to optimize but doesn't *analyze* the code. Performance regressions can slip in through new code that touches storage, iteration, persistence, or transaction paths without anyone noticing until benchmarks degrade.

We need a slash command that performs a structured, repeatable performance audit — identifying data layout issues, vectorization blockers, cache-unfriendly patterns, rkyv inefficiencies, and other optimization opportunities across all hot paths.

## Current State

- `.claude/commands/minkowski/optimize.md` — conversational optimization guidance (step-by-step advice, not automated analysis)
- `.claude/commands/self-audit.md` — mutation path / visibility / edge case audit (correctness, not performance)
- `.claude/commands/soundness-audit.md` — soundness audit (safety, not performance)
- `benches/` — criterion benchmarks exist but coverage of all hot paths is unknown
- `.cargo/config.toml` — `target-cpu=native` already configured
- BlobVec columns use 64-byte alignment (cache line)
- rkyv `raw_copy_size` detection exists in `CodecRegistry` for zero-copy fast path

No existing command performs automated performance analysis.

## Proposed Design

### Command Interface

```
/perf-shakedown [target]
```

- No argument on a feature branch: scopes to `git diff main...HEAD` intersected with hot path list
- `all` or no argument on `main`: full codebase sweep of all hot paths
- Any other argument: treated as a file path or module name filter

### Three-Phase Architecture

**Phase 1 — Scope (orchestrator, sequential)**

Determines which files and functions to analyze:

1. Resolve target to a file list (diff-based or full)
2. Intersect with the static hot path list (see below)
3. If no overlap: report "no hot paths touched", skip to Phase 3 discovery
4. Partition the scoped files into per-agent work packages

**Phase 2 — Analysis (4 parallel subagents)**

Each subagent receives its scoped file list, the specific functions to examine, and a focused checklist.

**Phase 3 — Profiling Recommendations (orchestrator, sequential)**

Synthesizes subagent findings into actionable profiling suggestions and hot path list maintenance.

### Static Hot Path List

Hardcoded in the command prompt. Updated manually when new hot paths are added.

**Storage (per-entity, per-archetype)**
- `storage/blob_vec.rs` — `push`, `get_ptr`, `get_ptr_mut`, `swap_remove_no_drop`, `realloc`
- `storage/archetype.rs` — `push`, `swap_remove`, entity/column iteration
- `storage/sparse.rs` — `insert`, `get`, `remove`, iteration

**Query (per-entity iteration)**
- `query/iter.rs` — `QueryIter::next`, `for_each`, `for_each_chunk`, `par_for_each`
- `query/fetch.rs` — `init_fetch`, `fetch`, `as_slice`

**Mutation (spawn, migrate, changeset apply)**
- `world.rs` — `spawn`, `insert`, `remove`, `get_mut`, `get_batch_mut`, `query`, `query_table_mut`
- `bundle.rs` — `Bundle::put`, `component_ids`
- `changeset.rs` — `EnumChangeSet::apply`, `record_insert`, arena allocation

**Reducer (per-entity iteration through handles)**
- `reducer.rs` — `QueryWriter::for_each` (manual archetype scan), `QueryMut::for_each`/`for_each_chunk`, `QueryRef::for_each`, `DynamicCtx::for_each`, `EntityMut::get`/`set`/`remove`, `Spawner::spawn`

**Persistence (I/O hot paths)**
- `wal.rs` — `append`, `replay_from`, `scan_last_seq`
- `snapshot.rs` — `save`, `load`, `load_zero_copy`
- `codec.rs` — `serialize`, `deserialize`, `raw_copy_size` usage
- `format.rs` — `serialize_record`, `deserialize_record`

**Transaction (commit path)**
- `transaction.rs` — `try_commit`, `begin`, tick validation, changeset apply
- `lock_table.rs` — `acquire`, `release`

### Subagent Specifications

#### data-layout

Analyzes struct definitions and internal data structures on hot paths.

Checklist:
- Field ordering: largest-to-smallest to minimize padding
- Heap-owning types (`Box`, `Vec`, `String`, `HashMap`) inside components iterated per-entity — pointer chasing
- Components >64 bytes in hot queries — candidates for splitting
- `BlobVec` and `Archetype` internal fields for unnecessary indirection
- `#[repr(C)]` on persistent components that should benefit from `raw_copy_size`

#### vectorization

Analyzes iteration paths for auto-vectorization fitness.

Checklist:
- `for_each` on numeric data where `for_each_chunk` would be better
- Inside `for_each_chunk` bodies: non-inlineable function calls, branching per element, scalar math on aligned types, index-based access instead of slice iteration
- Component alignment — `#[repr(align(16))]` or natural 16-byte types for SIMD
- `.cargo/config.toml` has `target-cpu=native`
- `QueryWriter::for_each` manual iteration for vectorization blockers

#### rkyv-compat

Analyzes persistence hot paths for rkyv efficiency.

Checklist:
- For each persistent component, check if `raw_copy_size` returns `Some` — if not, why?
- WAL append/replay: unnecessary allocations or copies
- Snapshot save/load/load_zero_copy: verify zero-copy path is taken where possible
- Codec `serialize_fn`/`deserialize_fn` closures for avoidable work
- `Vec<u8>` intermediaries that could be eliminated

#### cache-and-misc

Analyzes all hot paths for cache and general performance issues.

Checklist:
- False sharing: `AtomicU64`/`AtomicU32` on same cache line as frequently-written non-atomic data
- Allocation in hot loops: `Vec::new()`, `Box::new()`, `HashMap` operations inside per-entity iteration
- Redundant lookups: repeated `HashMap::get` or `entity_locations` lookups that could be hoisted
- Branch patterns: `Option` unwraps or `match` in inner loops that are always the same variant
- `HashMap` in hot paths where a dense `Vec` indexed by ID would work
- No UB: flag anything that could introduce undefined behavior if "optimized"

### Phase 3 — Profiling Recommendations

After aggregating subagent reports:

**Validate the hot path list:**
- For `PERF-CRITICAL` findings, suggest targeted criterion benchmarks if none exist
- Cross-reference `benches/` to identify hot paths lacking benchmark coverage
- Suggest `cargo bench` commands for existing benchmarks covering affected paths

**Discover new hot paths:**
- Flag new `for_each`/`for_each_chunk`/`par_for_each` call sites, new `unsafe` pointer arithmetic, or new loops over `entity_locations` not in the static list
- Suggest `cargo flamegraph` or `perf record` commands scoped to specific examples
- Flag unexpected bottleneck patterns in non-hot-path files as candidates for list addition

**Output:** concrete commands to run, each with a one-line rationale.

### Output Format

Single structured report:

```
## Performance Shakedown Report

### Scope
[files analyzed, diff or full]

### Data Layout
[findings with PERF-CRITICAL / PERF-OPPORTUNITY / PERF-OK ratings, file:line refs]

### Vectorization
[findings]

### rkyv Compatibility
[findings]

### Cache & Miscellaneous
[findings]

### Profiling Recommendations
[concrete commands + rationale]

### Hot Path List Maintenance
[suggestions for additions/removals]
```

### Severity Ratings

- `PERF-CRITICAL` — measurable regression or missed optimization on a proven hot path (allocation in inner loop, `for_each` instead of `for_each_chunk` on numeric data)
- `PERF-OPPORTUNITY` — likely improvement but needs benchmarking to confirm (field reordering, `#[repr(C)]` addition)
- `PERF-OK` — area was checked and looks good (brief note)

## Alternatives Considered

### Single sequential prompt (no subagents)
One long command that works through all 6 areas in order. Simpler but slower, shallower analysis per area, and likely to hit context limits on a full-codebase sweep. Rejected because the analysis areas are genuinely independent and benefit from focused attention.

### Dynamic hot path discovery
Phase 1 greps for loop bodies, unsafe blocks, and `#[inline]` annotations to discover hot paths at runtime. More general but slower, less reliable, and might miss structural hot paths or flag non-hot code. Rejected because the hot paths in an ECS storage engine are structural and stable. Phase 3 discovery handles the "did we miss one?" concern.

### Always full-codebase (no diff scoping)
Every run analyzes all hot paths. Thorough but noisy for routine pre-PR checks where only a few files changed. Rejected in favor of diff-scoped default with `all` override.

## Files to Create

- `.claude/commands/perf-shakedown.md` — orchestrator command with embedded hot path list and subagent dispatch logic
