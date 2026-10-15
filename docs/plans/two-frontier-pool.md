# Two-Frontier Slab Pool Design

## Problem

`SlabPool` used to split its budget into six size classes in fixed proportions
(`[1, 2, 4, 20, 40, 33]` %). That caused three problems:

1. **Stranded capacity.** A class runs out while other classes still hold free
   blocks. Overflow (borrowing a block one class up) helped only at the
   margin. Measured on a 64 MiB pool with the
   new `pool_fragmentation` example, multi-archetype workloads hit
   `PoolExhausted` at **34–37 %** utilization.
2. **A hard 1 MiB ceiling.** `size_class_for` returned `None` above 1 MiB, so a
   single column of 256 B components stopped at 4 096 rows (**1.6 %** of a
   64 MiB budget). This hurts large SIMD columns the most.
3. **Eager page touching.** Construction threaded a next-pointer through every
   block. For classes ≤ 4 KiB that writes to every page, so `World::new()`
   (a 256 MiB pool that is supposed to be demand-paged) faulted in
   **75.9 MiB RSS** before the first spawn.

The proposal discussed was "two opposing stacks": tiny descriptors growing up,
array payloads growing down. Taken literally, it doesn't fit. Column frees are
not LIFO (doubling growth frees interleaved across archetypes), so holes would
add external fragmentation, and compacting them would move columns that
`ThinSlicePtr` / `query_raw` readers hold raw pointers into. The small
descriptors (`EntityAllocator`, `entity_locations`, archetype `entities`) also
don't live in the pool at all. This design keeps the useful part: two
frontiers that meet only at the true budget. Each frontier hands out whole
chunks or runs, not individual objects.

## Current State (before this change)

- `crates/minkowski/src/pool.rs`: `SlabPoolInner::new` carves all classes at
  startup and pushes every block onto per-class lock-free stacks
  (`AtomicHead` = `Atomic<u128>` tagged pointer). A side table (one byte per
  64 B slot) routes deallocation by class. The TCache holds 32 blocks per class
  and refills 16 at a time.
- Pool callers (verified with grep for `.allocate(` / `.deallocate(`):
  `BlobVec::{new, reserve, grow, try_grow, alloc_scratch}`,
  `Arena::{grow, drop}` (changeset), sparse dense values (a `BlobVec`).
  Nothing else allocates from the pool.
- Growth was always allocate → copy → deallocate. The comment cited system
  `realloc` alignment, which doesn't apply to the pool.

## Proposed Design

### API Surface

The allocator API is internal (`pub(crate)`). One public change: the overflow counters are removed from `WorldStats` (breaking).

| Item | Change |
|---|---|
| `SlabPool::allocate` / `deallocate` | Same signatures and contract. Requests > 64 KiB now succeed up to the budget (no 1 MiB ceiling). Alignment > 1 MiB → `Err(PoolExhausted)`. |
| `SlabPool::grow(ptr, old, new, live)` (new, `unsafe`) | Realloc. Returns the same pointer if the block already fits, or widens a large run in place. Otherwise allocate + copy + free. On `Err` the original allocation is untouched. |
| `capacity()` / `used()` | Unchanged. |
| `overflow_active_counts()` / `overflow_total_counts()` | **Removed**, together with cross-class overflow itself (see "Overflow removed" below). |
| `WorldStats::pool_overflow_active` / `pool_overflow_total` | **Removed (breaking).** |

### Internal Architecture

```
 window start (1 MiB aligned)                                   window end
 │ slab chunks: 64 KiB, one class each →        ← large runs: 1 MiB units │
 ├──────────────────────────────────────┬───────┬─────────────────────────┤
 0                                      lo  free hi·1MiB          n_units·1MiB
```

- **Window.** mmap `budget + 1 MiB` and use the 1 MiB-aligned `budget`-byte
  window inside it. `capacity()` stays `budget`.
- **Bottom frontier (`lo`, bytes).** When a slab class (64 B / 256 B / 1 KiB /
  4 KiB / 64 KiB) has an empty free list, `carve` takes one 64 KiB chunk under
  the frontier lock, returns block 0 to the caller, and pushes blocks 1..n
  onto the class's lock-free stack with a single CAS. Carved chunks never go
  back to the frontier.
- **Top frontier (`hi`, unit index).** Requests > 64 KiB are runs of
  `ceil(max(size, align) / 1 MiB)` units, tracked in `Frontier::units`
  (`0` = free, `k` = run start, `u32::MAX` = interior). Placement picks the
  highest-address free run inside the top region first, otherwise extends `hi`
  downward. Freeing a run at the low edge retreats `hi` past every free unit,
  so the bottom frontier can reuse that space immediately.
- **Allocation order for slab class c:** pop c's free list → carve a chunk for
  c → `PoolExhausted`. A class never borrows another class's blocks, and a
  slab request never takes a large run.
- **TCache.** Bins exist for the 5 slab classes only. Refill takes one block
  through the full path (it may carve), then up to 15 extras popped from the
  same class's *already-carved* free list. Extras never carve, so one thread's
  cache cannot pull fresh frontier memory away from other classes. Large runs
  bypass the TCache on both allocate and deallocate.
- **Side table entry** = the bare class index (`0..6`), or `0xFF` when
  unallocated.
- **`grow`.** For a large run, widen into free units directly above it first.
  Take any shortfall from free units directly below (or from unclaimed units
  below `hi` that are still ≥ `lo`), then `ptr::copy` (memmove) the live bytes
  down. This removes the transient 2× footprint of allocate-copy-free that
  limited one doubling column to under half the budget.

### Data Flow

`BlobVec::try_grow` → `grow_allocation` → `SlabPool::grow` → (slab fits →
same ptr) | (`large_grow` → `Frontier::grow_run` under the lock → side table
moved under the lock → memmove after unlock) | fallback allocate/copy/free.
`Arena::grow` takes the same route.

### Invariant Matrix

Invariant **I1**: chunks, runs, and free-list blocks never overlap, and every
returned pointer is inside the window. Invariant **I2**: the side table entry
of every live allocation names its true class; every other slot is
`UNALLOCATED`.

| Consumer \ input | slab block | carved chunk remainder | large run | grown run (moved) |
|---|---|---|---|---|
| `allocate` (TCache path) | enforced: `slab_pool_concurrent_allocate_no_duplicates`, `tcache_concurrent_no_duplicates`, `tcache_refill_caches_only_its_own_class` | enforced: `loom_concurrent_carve_no_duplicates` | exempt: bypasses TCache (`slab_pool_large_and_slab_share_one_budget`) | n/a |
| `deallocate` (TCache path) | enforced: `slab_pool_concurrent_alloc_dealloc_interleaved` | enforced (same as slab block) | enforced: routes to `large_deallocate` (`slab_pool_multi_megabyte_allocation`) | enforced: `grow_large_run_shifts_down_and_moves_bytes` |
| `global_*` path (loom runs it; also called directly by unit tests) | enforced: `loom_concurrent_pop_no_duplicates`, `slab_pool_dealloc_routes_by_side_table_not_layout` | enforced: `loom_concurrent_carve_no_duplicates` | enforced: `loom_carve_and_large_run_disjoint` | enforced: `loom_grow_races_carve` |
| TCache flush/spill/drop | enforced: `tcache_epoch_flush_*`, `tcache_thread_exit_*` | enforced (same) | exempt: never cached (dealloc bypasses) | exempt: never cached |
| frontier collision | enforced: `slab_pool_concurrent_frontier_collision_no_overlap`, `slab_pool_one_class_can_exhaust_the_frontier` | enforced: `slab_pool_concurrent_frontier_collision_no_overlap` (tail reach: `slab_pool_window_is_unit_aligned_and_budget_sized`) | enforced: `slab_pool_large_run_reuses_interior_hole`, `loom_large_free_races_carve` | enforced: `grow_failure_leaves_original_allocation_intact`, `grow_doubling_column_reaches_whole_budget`, `loom_grow_races_take_run` |
| `grow` | enforced: `grow_within_same_slab_block_keeps_pointer` (fits), `grow_slab_to_larger_class_copies_live_bytes`, `grow_slab_block_into_large_run` | exempt: a chunk remainder is a slab block once handed out | enforced: `grow_large_run_in_place_upward`, `grow_large_run_falls_back_to_fresh_run`, `slab_pool_concurrent_grow_and_large_churn` | enforced: `grow_large_run_shifts_down_and_moves_bytes`, `loom_grow_races_carve`, `growth_past_one_mebibyte_preserves_values` (BlobVec), `recover_column_past_one_mebibyte_preserves_values` (LSM import) |

No-borrowing rule (a class never takes another class's blocks or a large
run): `slab_pool_exhausted_class_never_borrows_larger_class`,
`slab_pool_classes_never_share_blocks`,
`slab_pool_largest_slab_class_never_takes_large_run`.

Break harness: `fuzz_pool_columns` drives all of the above through `World` against a 3 MiB + 12 345 B budget (large runs, multi-unit grow, collision, archetype-wide frees) and checks every live value after each op against a shadow model.

### Convention Sheet

- `lo`: byte offset from the window start, **exclusive** end of the bottom
  region, always a multiple of `CHUNK_SIZE` (64 KiB). Starts at 0.
- `hi`: unit index, **inclusive** start of the top region. Starts at
  `n_units = budget / LARGE_UNIT` (empty top region).
- Unit `u` covers window bytes `[u·1 MiB, (u+1)·1 MiB)`.
- Bottom limit: `hi·1 MiB` if `hi < n_units`, else `budget` rounded down to
  64 KiB (the sub-unit tail is reachable only while the top is empty).
- Edge invariant: `hi == n_units || units[hi] != 0`. Pinned by
  `slab_pool_large_run_reuses_interior_hole` (retreat after an edge free) and
  `slab_pool_large_and_slab_share_one_budget` (bottom reuses the retreated
  space).

## Alternatives Considered

1. **Literal opposing object stacks** (bump up for small, bump down for
   large). Rejected: non-LIFO frees create holes. Reclaiming them needs
   compaction, which moves columns under live raw pointers (`ThinSlicePtr`,
   `query_raw` from `&World`). Exhaustion also becomes order-dependent, which
   breaks the fixed-budget predictability the pool exists for.
2. **Keep fixed proportions, tune them, widen overflow to all larger
   classes.** Cheap, but any fixed split is wrong for some workload, and it
   fixes neither the 1 MiB ceiling nor the RSS-at-construction cost.
3. **Single bottom frontier carving fixed 1 MiB chunks for every class.**
   Simpler, but a > 1 MiB run would need contiguous chunks interleaved with
   small-class chunks, so large runs fragment quickly. Two frontiers keep the
   long-lived small chunks and the movable-in-place large runs apart.
4. **Return fully-free slab chunks to the frontier.** Would also bound
   per-class stranding by live usage rather than peak usage, but detecting a
   fully-free chunk lock-free needs per-chunk occupancy counts on the hot
   path. Deferred (see Follow-ups).

## Semantic Review

1. **Wrong World?** The pool is shared via `Arc<SlabPool>`, and `deallocate`
   asserts the pointer is inside this pool's window (`lookup`). No new
   cross-world surface.
2. **Drop sees inconsistent state?** `SlabPoolInner::drop` frees only the side
   table box. `Frontier::units` is an owned `Box` dropped with the `Mutex`, and
   `SlabPoolInner::drop` never takes the frontier lock. `BlobVec::drop` /
   `Arena::drop` do take it, through `deallocate` of a large run. That is not
   reentrant (no pool call happens while the lock is held), so it is sound.
3. **Two threads through `&self`?** Yes, by design. Free lists are lock-free
   (unchanged protocol). Every frontier mutation and every large-run side
   table write happens under `Mutex<Frontier>`. Chunk remainders are published
   with the push CAS (AcqRel) after Relaxed link stores. The *logic* (who gets
   which block or unit) is covered by the 5 new loom models plus the existing
   3. Loom does not see the *memory ordering* of the next-pointer links:
   under `cfg(loom)` the list head is a `Mutex<u128>` shim and the link
   stores use `std` atomics. TSan covers that ordering through the std-thread
   `slab_pool_concurrent_*` and `tcache_concurrent_no_duplicates` tests.
   Exhaustion is deterministic only for single-threaded use of a pool:
   across threads, TCache holdings and the order in which threads take the
   frontier lock decide placement, and so whether a later `grow` fits in
   place. Committed state below the budget is unaffected, and addresses are
   never persisted.
4. **Dedup/merge preserves the strongest invariant?** `release_run` and the
   `hi` retreat only ever clear units that the run being freed owned (asserted
   run start). `grow_run` claims only free units.
5. **Abandoned halfway?** `carve` holds the lock only to bump `lo`. If the
   thread dies after that, the chunk is leaked to that class, not corrupted.
   `large_grow` commits units and the side table under the lock before the
   memmove; the memmove touches only owned units. `grow` on `Err` leaves the
   original untouched (tested).
6. **Type bound violable by a legal generic?** No new generics.
7. **Operations outside the Access bitset?** Not applicable: allocator
   internals, no handle surface.

## Implementation Plan

1. `pool.rs`: `Frontier` (`carve_chunk`, `take_run`, `grow_run`,
   `release_run`). `SlabPoolInner` split into `pop_free` / `push_chain` /
   `carve` / `slab_allocate` / `large_allocate` / `large_deallocate` /
   `large_grow` / `lookup`. TCache refill rule. `SlabPool::grow`.
2. `storage/blob_vec.rs`: `reserve` / `grow` / `try_grow` go through
   `grow_allocation`. `changeset.rs`: `Arena::grow` goes through
   `SlabPool::grow`.
3. Tests: replace the proportion-based overflow tests with frontier
   exhaustion and no-borrowing tests, plus the new frontier, grow, and loom
   tests in the matrix above.
4. `examples/pool_fragmentation.rs`: utilization-at-exhaustion and
   RSS-after-`World::new()` benchmark. `fuzz/fuzz_targets/fuzz_pool_columns.rs`:
   break harness.
5. Limits: `MAX_POOL_BUDGET` (mmap `Layout` size, and `n_units < u32::MAX`
   for the run encoding, which also bounds the O(n_units) frontier scans).
   Hugepage mappings are rounded to 2 MiB so `munmap` gets a valid length.

## Results (64 MiB pool, `pool_fragmentation`, release)

| Scenario | before (first fail / final) | after (first fail / final) |
|---|---|---|
| small columns (32 archetypes) | 33.9 % / 33.9 % | 93.8 % / 93.8 % |
| one big column (256 B) | 1.6 % / 1.6 % | 50.0 % / 50.0 % |
| mixed (4 heavy, 60 light) | 7.8 % / 37.0 % | 92.0 % / 96.5 % |
| uniform (64 archetypes) | 37.0 % / 37.0 % | 95.3 % / 95.3 % |
| RSS after `World::new()` (256 MiB pool) | 75.9 MiB | 0.0 MiB |

The single-column ceiling of 50 % is set by doubling growth itself: the next
step would need the entire budget. The utilization numbers are deterministic
(single-threaded, fixed budget, address-independent window). The RSS line is
indicative only: it depends on the kernel and THP settings.

### Criterion benchmarks (`minkowski-bench`, release, `main` vs this branch)

| Benchmark | `main` | branch | Note |
|---|---|---|---|
| `simple_insert/batch` | 44.3 ms | 2.19 ms | fresh `World` per iteration: `main` threads every free-list block of the 256 MiB pool at construction |
| `simple_insert/spawn_batch` | 43.1 ms | 1.22 ms | same cause |
| `simple_insert/changeset` | **aborts** | runs | `Arena` needs 1.83 MiB > the old 1 MiB ceiling → `handle_alloc_error` |
| `reducer/*` (QueryWriter) | **aborts** | runs | `Arena::grow` → `handle_alloc_error` on `main` |
| `add_remove/*` | 2.99–3.21 ms | 2.97–3.22 ms | 3 interleaved runs each; within noise |
| `simple_iter/*` | 29.9–30.0 / 4.50–4.65 / 48.8–55.7 µs | 29.6–30.9 / 4.48–4.64 / 48.5–50.1 µs | `for_each` / `for_each_chunk` / `par_for_each`, 2 interleaved runs; no allocator on this path |

### Overflow removed

The first version kept the old cross-class overflow (`c → c+1`, and
64 KiB → 1-unit run). With both paths disabled, all four scenarios gave
identical numbers. With demand-carved chunks a class only runs dry once the
frontiers have met. Borrowing a larger block at that point wastes 4–16× per
request for a handful of extra allocations, and it cost an overflow bit in
the side table, telemetry, a TCache refill special case, and an extra
interleaving in the loom models. Overflow existed to soften `PROPORTIONS`
starvation; without `PROPORTIONS` it is dead weight. It is gone, along with
`WorldStats::pool_overflow_*` (breaking). Memory stranded in one class is
addressed by the chunk-return follow-up instead.

## Follow-ups

- Return fully-free slab chunks to the bottom frontier (alternative 4): the
  real fix for capacity stranded in one class after a workload shift
  (`slab_pool_one_class_can_exhaust_the_frontier` pins today's behavior).
- With prefault on, the up-to-1 MiB of mapping slack around the window is
  also faulted in. Pre-fault only the window, or `munmap` the slack.
- Expose `lo` / `hi` in `WorldStats` so operators can see how close the
  frontiers are.
- Optionally `madvise(MADV_DONTNEED)` large runs freed in demand-paged pools.
