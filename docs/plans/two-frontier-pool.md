# Two-Frontier Slab Pool Design

## Problem

`SlabPool` used to split its budget into six size classes in fixed proportions
(`[1, 2, 4, 20, 40, 33]` %). That caused three problems:

1. **Stranded capacity.** A class runs out while other classes still hold free
   blocks. Overflow only went one class up. Measured on a 64 MiB pool with the
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

All of it is internal (`pub(crate)`); no public API changes.

| Item | Change |
|---|---|
| `SlabPool::allocate` / `deallocate` | Same signatures and contract. Requests > 64 KiB now succeed up to the budget (no 1 MiB ceiling). Alignment > 1 MiB → `Err(PoolExhausted)`. |
| `SlabPool::grow(ptr, old, new, live)` (new, `unsafe`) | Realloc. Returns the same pointer if the block already fits, or widens a large run in place. Otherwise allocate + copy + free. On `Err` the original allocation is untouched. |
| `capacity()` / `used()` / `overflow_*_counts()` | Unchanged shape (`[u64; 6]`). Index 5 now counts class-4 overflow into a 1-unit large run. |
| `WorldStats` | Unchanged. |

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
  c → pop (c+1)'s free list (overflow) → for c = 4 only, a 1-unit large run
  (overflow) → `PoolExhausted`.
- **TCache.** Bins exist for the 5 slab classes only. Refill takes one block
  through the full path (it may carve or overflow), then up to 15 extras popped
  from the same class's *already-carved* free list. Extras never carve, so one
  thread's cache cannot pull fresh frontier memory away from other classes. An
  overflow block is returned uncached. Large runs bypass the TCache on both
  allocate and deallocate.
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

| Consumer \ input | slab block | carved chunk remainder | large run | overflow block | grown run (moved) |
|---|---|---|---|---|---|
| `allocate` (TCache path) | enforced: `slab_pool_concurrent_allocate_no_duplicates`, `tcache_concurrent_no_duplicates` | enforced: `loom_concurrent_carve_no_duplicates` | exempt: bypasses TCache (`slab_pool_large_and_slab_share_one_budget`) | enforced: `tcache_overflow_refill_correct_bin` | n/a |
| `deallocate` (TCache path) | enforced: `slab_pool_concurrent_alloc_dealloc_interleaved` | enforced (same as slab block) | enforced: routes to `large_deallocate` (`slab_pool_multi_megabyte_allocation`) | enforced: `slab_pool_overflow_dealloc_returns_to_correct_class` | enforced: `grow_large_run_shifts_down_and_moves_bytes` |
| `global_*` (loom path) | enforced: `loom_concurrent_pop_no_duplicates` | enforced: `loom_concurrent_carve_no_duplicates` | enforced: `loom_carve_and_large_run_disjoint` | enforced: `slab_pool_largest_slab_class_overflows_into_large_run` | enforced: `loom_grow_races_carve` |
| TCache flush/spill/drop | enforced: `tcache_epoch_flush_*`, `tcache_thread_exit_*` | enforced (same) | exempt: never cached (refill returns overflow uncached; dealloc bypasses) | enforced: `tcache_thread_exit_with_overflow_blocks` | exempt: never cached |
| frontier collision | enforced: `slab_pool_concurrent_frontier_collision_no_overlap` | enforced: `slab_pool_window_is_unit_aligned_and_budget_sized` | enforced: `slab_pool_large_run_reuses_interior_hole`, `loom_large_free_races_carve` | enforced: `loom_large_free_races_carve` (overflow interleaving) | enforced: `grow_failure_leaves_original_allocation_intact`, `grow_doubling_column_reaches_whole_budget` |

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
   table box. `Frontier::units` is an owned `Box` dropped with the `Mutex`. No
   Drop path takes the frontier lock.
3. **Two threads through `&self`?** Yes, by design. Free lists are lock-free
   (unchanged protocol). Every frontier mutation and every large-run side
   table write happens under `Mutex<Frontier>`. Chunk remainders are published
   with the push CAS (AcqRel) after Relaxed link stores. Covered by the 4 new
   loom models plus the existing 3.
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
3. Tests: rewrite the proportion-based overflow tests around frontier
   exhaustion, plus the new frontier, grow, and loom tests in the matrix above.
4. `examples/pool_fragmentation.rs`: utilization-at-exhaustion and
   RSS-after-`World::new()` benchmark.

## Results (64 MiB pool, `pool_fragmentation`, release)

| Scenario | before (first fail / final) | after (first fail / final) |
|---|---|---|
| small columns (32 archetypes) | 33.9 % / 33.9 % | 93.8 % / 93.8 % |
| one big column (256 B) | 1.6 % / 1.6 % | 50.0 % / 50.0 % |
| mixed (4 heavy, 60 light) | 7.8 % / 37.0 % | 92.0 % / 96.5 % |
| uniform (64 archetypes) | 37.0 % / 37.0 % | 95.3 % / 95.3 % |
| RSS after `World::new()` (256 MiB pool) | 75.9 MiB | 0.0 MiB |

The single-column ceiling of 50 % is set by doubling growth itself: the next
step would need the entire budget.

## Follow-ups

- Return fully-free slab chunks to the bottom frontier (alternative 4).
- Expose `lo` / `hi` in `WorldStats` so operators can see how close the
  frontiers are.
- Optionally `madvise(MADV_DONTNEED)` large runs freed in demand-paged pools.
