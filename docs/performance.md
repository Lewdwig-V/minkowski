# Performance

How Minkowski achieves its performance characteristics — the optimizations we made, why they matter, and what the structural limits are. All numbers from `cargo bench -p minkowski-bench` on a single core.

## Iteration

Column-oriented storage means components of the same type are contiguous in memory. A query like `world.query::<(&mut Pos, &Vel)>()` walks two arrays in lockstep — the prefetcher sees a linear access pattern and stays ahead.

**`for_each_chunk` yields typed slices** that LLVM auto-vectorizes. This is 10x faster than per-element `for_each` (1.55 µs vs 14.6 µs for 10K entities) because the compiler can use SIMD instructions on the contiguous slice. Use `for_each_chunk` for numeric workloads; use `for_each` when you need `Entity` handles or branching logic.

**`par_for_each` distributes chunks across Rayon threads.** At 10K entities the thread-pool overhead dominates (~340 µs); at 50K+ entities it amortizes and scales linearly.

## Query Planner

The planner compiles queries into push-based execution plans at build time, then executes them repeatedly against live world data.

### Join Elimination (343x)

Inner joins in an ECS are often redundant — they're testing whether entities have certain components, which archetype matching already knows. The planner detects inner joins that are pure component-presence filters and merges them into the left-side scan at build time. No materialization, no sort, no intersection. `join/for_each_batched_10k` dropped from 103 µs to 300 ns.

### Direct Archetype Iteration (1,792x)

Scan-only plans (no joins, no custom predicates, no index/spatial drivers) bypass the ScratchBuffer entirely and walk archetypes with `init_fetch`/`fetch` inline. This eliminates the type-erased `CompiledForEach` dispatch layer. `scan_for_each_10k` dropped from 9.5 µs to 5.25 ns — essentially the cost of a single archetype metadata check.

### Column-Aware Custom Filters

`Predicate::custom` dispatches through `Arc<dyn Fn>` per entity with a `world.get::<T>()` inside the closure — 12.7 ns/entity overhead. `Predicate::custom_column` receives `&T` directly from contiguous column slices, resolving the column once per archetype instead of per entity. Uses a boolean mask that multiple column filters AND together.

### Aggregates (13x)

Aggregates (COUNT, SUM, MIN, MAX, AVG) use cached extractors with specialized inner loops that iterate archetype columns directly, bypassing per-entity `world.get()`. The planner's aggregate path is now faster than a hand-written `world.query()` loop (5.84 µs vs 6.45 µs for 10K entities) because it avoids iterator machinery.

### Cache-Aware Partitioned Joins

For large hash joins where the working set exceeds L2 cache, entities are bucketed by `Entity::to_bits() % partitions` so each partition fits in cache during intersection. Partition count is computed from `build_rows * avg_component_bytes / l2_cache_bytes`. Falls back to single-partition `sorted_intersection` for small joins.

## Memory Allocation

### Lock-Free Slab Pool (6x)

The pool allocator uses lock-free intrusive stacks via 128-bit tagged pointer CAS (`Atomic<u128>`). ABA prevention via 64-bit monotonic tag. A side table routes deallocation to the correct size class.

### Thread-Local Cache

A per-thread L1 cache with 32-slot bins per size class sits in front of the lock-free stack. 15 out of 16 allocations hit the L1 cache (~3 instructions) instead of the global stack (~7 CAS operations). Epoch-based lazy flush prevents Rayon worker threads from hoarding memory. `add_remove/pool` dropped from 8.03 ms to 1.35 ms — within 5% of jemalloc.

## Entity Spawning

### Batch Spawning (5.2x)

`World::spawn_batch()` resolves the target archetype once, reserves column capacity with a single `BlobVec::reserve(n)`, then pushes entities in a tight loop. Individual `spawn()` calls pay the archetype hash-lookup per entity. `simple_insert/spawn_batch`: 343 µs vs 1.78 ms for individual spawns.

## Transactional Writes

### QueryWriter Streaming Archetype Buffers (1.5x)

`QueryWriter` buffers writes into an `EnumChangeSet` for atomic commit. The naive approach — push a `Mutation::Insert` per entity, apply by looking up each entity's location — paid 37% of execution time on per-entity bookkeeping.

The optimization: during `for_each`, the writer opens a pre-resolved `ArchetypeBatch` per archetype with `ColumnBatch` entries that cache the column index, drop function, and layout. `WritableRef::set()` pushes directly to the current batch via a pre-resolved `column_slot` index. The apply phase drains batches with zero per-entity lookups — no `is_alive`, no `entity_locations`, no `column_index`, no `ComponentRegistry::info`.

`query_writer_10k` dropped from 93 µs to 64 µs. Sparse updates (10% of entities modified) run at 5.9 µs — near `query_mut` territory. The remaining gap to `query_mut` (1.6 µs) is structural: arena allocation + clone for buffered writes. This is the cost of atomic commit semantics.

## Persistence

### WAL Replay (1.7x)

WAL replay collects all records in a first pass, then decodes and applies them as a single `EnumChangeSet`. This eliminates per-record changeset allocation, per-record `apply()` overhead, and per-record tick advancement. Throughput improved from 581K to 943K mutations/second.

## Structural Limits

Some costs are inherent to the design and cannot be optimized away without changing the model:

- **QueryWriter vs query_mut (41x gap)** — the remaining 64 µs is the cost of buffered atomic commits: arena allocation, value cloning, and changeset management. `query_mut` writes directly to column memory with no buffer. Closing this gap requires abandoning the buffered-write model (direct write + undo log, or shadow columns), which breaks read stability during iteration.

- **DynamicCtx overhead (15x vs query_mut)** — runtime type resolution + the collect-then-write pattern. Cannot be eliminated without losing dynamic dispatch.

- **Changeset spawn overhead (1.5x vs direct spawn)** — the arena allocation + mutation log is the price of undo/redo and WAL compatibility.

- **`for_each` vs `for_each_chunk` (10x)** — per-element callbacks prevent SIMD auto-vectorization. This is a user API choice, not an engine limitation.

## Benchmark Reference (v1.3.0)

Run benchmarks with `cargo bench -p minkowski-bench`.

### Iteration

| Benchmark | Time | Per-entity |
|---|---|---|
| `simple_iter/for_each_chunk` | 1.55 µs | 0.16 ns |
| `simple_iter/for_each` | 14.6 µs | 1.46 ns |
| `reducer/query_mut_10k` | 1.56 µs | 0.16 ns |
| `reducer/query_writer_10k` | 64.5 µs | 6.45 ns |
| `reducer/query_writer_sparse_update_10k` | 5.87 µs | 0.59 ns |
| `reducer/dynamic_for_each_10k` | 115.8 µs | 11.6 ns |

### Query Planner

| Benchmark | Time |
|---|---|
| `planner/scan_for_each_10k` | 5.25 ns |
| `planner/query_for_each_10k` | 5.87 µs |
| `planner/aggregate_count_sum_10k` | 5.84 µs |
| `planner/custom_filter_50pct` | 64.6 µs |
| `planner/btree_range_10pct` | 9.57 µs |
| `planner/hash_eq_1` | 42 ns |
| `planner/changed_skip_10k` | 7.3 ns |

### Joins

| Benchmark | Time |
|---|---|
| `join/for_each_batched_10k` | 300 ns |
| `join/for_each_chunk_10k` | 3.18 µs |
| `join/for_each_get_10k` | 37.9 µs |
| `join/manual_query_10k` | 4.76 µs |

### Entity Management

| Benchmark | Time |
|---|---|
| `simple_insert/spawn_batch` | 343 µs |
| `simple_insert/batch` | 1.78 ms |
| `add_remove/pool` | 1.32 ms |
| `add_remove/add_remove` | 1.32 ms |

### Persistence

| Benchmark | Time |
|---|---|
| `serialize/wal_append` | 1.25 µs |
| `serialize/wal_replay` | 1.06 ms |

---

## LSM Benchmarking & Tuning (2026-06-21)

**Methodology note.** The wall-clock numbers in the v1.3.0 table above are **not**
a valid cross-machine baseline — machine variance dominates. A same-host re-run
measured every ECS hot path ~30–38% "slower" than the v1.3.0 doc numbers, but that
uniform scaling across independent paths is the *signature of a slower host*, not a
regression. Use **iai-callgrind instruction counts** for regression detection —
deterministic and machine-independent. (iai benches require valgrind; the repo's
`target-cpu=native` makes valgrind SIGILL, so run them with
`RUSTFLAGS="-C target-cpu=x86-64-v2"`.)

### Regression audit — ECS hot paths (instruction-count A/B, same host)

`crates/minkowski-bench/benches/ecs_icount.rs`, v1.3.0 vs HEAD:

| Bench | v1.3.0 | HEAD | Δ |
|---|---:|---:|---:|
| `scan_for_each_10k` | 2,633 | 2,761 | +4.9% |
| `join_for_each_batched_10k` | 18,759 | 18,533 | −1.2% (faster) |
| `query_mut_10k` | 125,287 | 125,513 | +0.18% (flat) |

**No meaningful regression** — changes are small and mixed (the join fast path even
improved). The LSM/heap-persistence work *cannot* regress these: the query/iter/
planner paths live in `minkowski` core and do not link LSM code. The `scan` +4.9%
is core-planner evolution since v1.3.0, unrelated to the LSM. These benches are now
the machine-independent regression gate. Run:
`RUSTFLAGS="-C target-cpu=x86-64-v2" cargo bench -p minkowski-bench --bench ecs_icount`.

### Page-codec instruction counts (256 rows)

`crates/minkowski-lsm/benches/page_codec.rs` (iai-callgrind):

| Op | instr | /row |
|---|---:|---:|
| `rawcopy_column_256` (POD memcpy) | 674 | ~3 |
| `page_frame_decode_256` (offset slicing, zero-copy — NOT rkyv) | 7,471 | ~29 |
| `page_frame_encode_256` (offset table + value concat — NOT rkyv) | 58,841 | ~230 |
| `rkyv_decode_256` (real per-row `deserialize_by_type`) | 224,917 | ~879 |
| `rkyv_encode_256` (real per-row `serialize_by_type`) | 345,510 | ~1,350 |

Heap (Serialized) recovery costs **~879 instr/row vs ~3 for a RawCopy memcpy (~330×)**
— this quantifies and justifies the hybrid storage-kind design: POD columns stay on
the memcpy fast path, rkyv is paid only for heap columns. Note the page *framing*
(`serialized_page::encode`/`decode`) is cheap and zero-copy on decode; the cost is
the rkyv codec, not the Arrow offset table.

**Optimization opportunity (deferred to the perf shakedown).** Heap recovery re-runs
full rkyv `bytecheck` per row even though the page is CRC-validated on read. The
RawCopy path has a `CrcProof` fast lane that skips bytecheck; Serialized columns have
no equivalent. Given a `CrcProof`, recovery could `from_bytes_unchecked` Serialized
rows — plausibly **~halving** the ~879 instr/row decode cost.

### Level-count / compaction sweep

`crates/minkowski-persist/benches/lsm_level_sweep.rs`, growing workload, K=4 trigger:

| N | write-amp | total runs | recover µs |
|---:|---:|---:|---:|
| 3 | 0.380 | 4 | 47,137 |
| 4 | 0.423 | 1 | 42,219 |
| 5 | 0.423 | 1 | 41,220 |
| 7 | 0.423 | 1 | 41,609 |

**N=4 confirmed.** N=4/5/7 are identical. N=3 caps at L2, leaving **4 uncompacted
bottom-level runs vs 1** — it trades ~10% lower write-amp (0.380 vs 0.423, it skips
the final L2→L3 merge) for **~12% slower recovery** (47.1 ms vs 42.2 ms, more runs
to merge). N=4 captures the recovery benefit; deeper N gives nothing until ~4^N-flush
extremes (far beyond the ≤100×-RAM regime). Notes: the sweep must *grow* the dataset
(a constant workload self-supersedes and never cascades) **and** clear the dirty-page
tracker after each flush (`flush_and_record` takes `&World` and doesn't clear it —
otherwise every run is a full snapshot and write-amp/recovery are meaningless).
(`recover µs` is wall-clock and host-relative.)

### Bytecheck-skip on heap recovery

**Measured instruction counts** (`rkyv_decode` iai-callgrind bench, 256 rows):

| Path | instr (256 rows) | instr/row |
|---|---:|---:|
| `rkyv_decode_256` (checked, `from_bytes`) | 230,816 | 901.6 |
| `rkyv_decode_unchecked_256` (unchecked, `access_unchecked`) | 182,624 | 713.4 |

**20.9% reduction per row** on Serialized (heap) column decode. The win applies only to heap columns — POD/RawCopy columns already use a direct memcpy and are unaffected. End-to-end recovery benefit scales with the heap-column fraction of the schema.

**Gate design.** CRC proves page integrity and data provenance (the bytes are exactly what the writer produced) but does NOT prove rkyv well-formedness (the archived layout is what the current binary expects). A separate per-run `decode_fingerprint` (FNV-1a over `RKYV_DECODE_EPOCH` + each Serialized component's name, native size, native alignment, and archived size) proves layout-provenance: it is computed identically by the flush writer, compaction, and recovery via a single `run_fingerprint` function (no independent implementations that can drift). Recovery decodes unchecked only when the stored footer fingerprint is non-zero AND equals the recovering binary's freshly recomputed value AND the per-page CRC validates; any mismatch falls back to the full checked `from_bytes` path. Compaction carries the fingerprint forward when all input runs agree on the same value; if they disagree (mixed binary versions), the output fingerprint is zeroed, forcing checked decode on the next recovery. Page framing validation (offset-table bounds, `offsets[0]==0`, monotonicity, exact-consumption, in-bounds row slices) is preserved on both decode paths; bytecheck-skip trusts only each row's internal rkyv structure, not the framing or page bounds.

**`RKYV_DECODE_EPOCH`** (constant in `crates/minkowski-lsm/src/fingerprint.rs`) MUST be bumped on any rkyv upgrade that changes archived layouts without changing native or archived sizes. Bumping invalidates all stored fingerprints, causing recovery to fall back to checked decode on all existing runs.

**Miri coverage.** The unchecked path (`codec::raw_copy_tests::unchecked_decode_matches_checked_for_heap`) is verified UB-free for valid archive bytes under Tree Borrows with `-Zmiri-ignore-leaks` (rkyv's `Pool` deserializer leaks are in rkyv internals, not this code). Command: `MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski-lsm --lib unchecked_decode_matches_checked_for_heap` — result: clean, 1 passed. The recovery-level Miri run (`recovery::tests::recover_heap_unchecked_path_matches_values`) was attempted but is not practical under Tree Borrows: the test pulls in `World::spawn` which exercises the pool's tagged-pointer (`integer-to-pointer` cast) fast path, which Tree Borrows explicitly does not support — Miri emits the warning "Tree Borrows does not support integer-to-pointer casts, so the program is likely to go wrong when this pointer gets used" and the run is non-terminating (>10 min at 99% CPU). The unsafe seam itself (`access_unchecked`) is covered by the codec-level test; the pool's provenance model is a separate concern predating this feature. Follow-up: extend the nightly Miri job to include `minkowski-lsm` recovery tests under a mode that tolerates the pool's provenance model (Stacked Borrows rather than Tree Borrows, or strict-provenance pool refactor).

### Running the benchmarks

```sh
# criterion (wall-clock, host-relative throughput)
cargo bench -p minkowski-persist --bench lsm_throughput
cargo bench -p minkowski-persist --bench lsm_compaction      # prints WRITEAMP lines
cargo bench -p minkowski-persist --bench lsm_level_sweep      # prints SWEEP lines
# iai-callgrind (instruction counts — needs valgrind + the RUSTFLAGS override)
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo bench -p minkowski-lsm  --features bench-support --bench page_codec
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo bench -p minkowski-bench --bench ecs_icount
# or capture a full baseline artifact:
scripts/capture-bench-baseline.sh
```
