# Thread Caches Hold Their Pool Weakly (#293)

## Problem

Each thread's slab cache (`TCACHE`, `pool.rs:1240`) holds `pool: Arc<SlabPoolInner>` (`pool.rs:1153`). Dropping a `World` drops its `Arc<SlabPool>`, but every thread that last allocated or freed through that pool keeps `SlabPoolInner` alive: the mmap'd window, the side table, and any pre-faulted or `mlock`ed pages. The pool is released only when that thread touches a *different* pool (the cache is replaced, `pool.rs:1332-1338`) or exits (`Drop for TCache`, `pool.rs:1223`).

Measured with `examples/memory_footprint.rs` (branch `claude/memory-footprint-measure`, `dropped_world_retention`) on a pre-faulted 1 GiB world:

```
RSS: before 1066 MiB, world live 1082 MiB, after drop 1082 MiB, after next pool use 57 MiB
```

Idle Rayon workers that served a `par_for_each` pin the pool the same way, and nothing short of their next pool operation releases it.

## Current State

Verified on `main` (`cb23efa`).

**Who holds a strong reference to a pool.** Every owner of a live pool allocation holds an `Arc<SlabPool>`:

| Owner | Field |
|---|---|
| `World` | `pool: Arc<SlabPool>` (`world.rs:297`) |
| `BlobVec` (columns, sparse dense values) | `pool` (`storage/blob_vec.rs:20`) |
| `PoolVec` (generations, free list, locations, entity lists, sparse dense entities) | `pool` (`storage/pool_vec.rs:25`) |
| `PagedSparseSet` / `SparseStorage` (sparse pages) | `pool` (`storage/sparse.rs:46, 312`) |
| `EnumChangeSet` arena | `pool: Option<Arc<SlabPool>>` (`changeset.rs:100`); the shared scratch pool is a process-lifetime static (`pool.rs:77`) |

So while any block of a pool is allocated, that pool has a strong owner. Once the last one drops, the only remaining references are thread caches, and the blocks they hold are **free** blocks: cached between a `deallocate` and the next `allocate` (`pool.rs:1384-1407`), or prefetched by `refill` (`pool.rs:1172-1189`).

**What a cache does with its blocks.** It only ever:
- hands one out from `allocate`, which runs on `&self` of a `SlabPool` and so holds a strong reference for the duration (`pool.rs:1326-1356`);
- returns them through `global_deallocate_to_class` (`pool.rs:1078`) from `flush_all`, `spill` or `Drop`. That writes the free-list next-pointer into the block and reads the side table, so it **dereferences the window**.

**Where a cache uses its own `pool` field.** `refill`, `flush_all`, `spill` and `Drop` call through `self.pool`. `allocate` / `deallocate` call `refill` / `spill` / `flush_all` only after `Arc::ptr_eq(&existing.pool, &self.inner)` has succeeded or a new cache has been created for `self.inner` (`pool.rs:1332-1340, 1387-1394`). So in those paths `self.pool` and the caller's `self.inner` are the same pool. `flush_caches` checks `ptr_eq` before `flush_all` (`pool.rs:1483-1490`).

## Proposed Design

`TCache::pool` becomes `Weak<SlabPoolInner>`. The rules:

1. **Fast path (`allocate` / `deallocate`).** Identify the cache's pool by address: `Weak::as_ptr(&cache.pool) == Arc::as_ptr(&self.inner)`. On a match, every cache operation uses the caller's `&self.inner`, which is strong for the duration. `refill`, `flush_all` and `spill` take the pool as a `&SlabPoolInner` parameter instead of reading a field. The fast path stays one pointer compare: no `upgrade`, so no extra atomic.
2. **Replacing a cache for another pool.** `upgrade()` the old `Weak`. If that succeeds, flush the old cache into its pool, then drop the temporary `Arc`. If it fails, the old pool is gone: **discard** the bins (set every `count` to 0) without touching the pointers.
3. **Thread exit (`Drop for TCache`).** The same rule: upgrade and flush, or discard.
4. **`flush_caches`.** Compares by address as in rule 1 and flushes through `&self.inner`.

Discarding is sound because a dead pool's window is unmapped: its blocks belong to no one, and nothing will read the cached pointers again. Before the change those blocks would have gone back to a free list that no one could ever pop from again.

**Address identity is ABA-free.** A `Weak` keeps the `ArcInner` allocation alive (its weak count is non-zero) even after `SlabPoolInner` is dropped. So while a cache holds a `Weak` to pool A, no new `Arc<SlabPoolInner>` can occupy A's address, and a pointer compare can never mistake a new pool for a dead one.

**Release point.** After this change, the last strong reference is the last allocation owner. `SlabPoolInner::drop` (munmap, side table free) therefore runs on the thread that drops the `World` (or the last outliving column), whether or not other threads' caches still name the pool. The one exception is benign: a thread that upgraded at exit or replacement holds the pool for that flush only, and may itself run the final drop afterwards.

**Reentrancy.** `SlabPoolInner::drop` may now run inside `TCACHE.with` (rule 2, when the temporary `Arc` is the last). It unmaps the region and frees the side table `Box` through the system allocator, and never touches `TCACHE`, so there is no reentrant borrow of the cache cell. The same holds in `Drop for TCache` at thread exit.

Option 2 from the issue (flush the dropping thread's cache on `World::drop`) is subsumed: the dropping thread's own cache also holds only a `Weak`, so its pool is released the same way.

### API Surface

None. `TCache` and `SlabPoolInner` are private. Behavior change: a dropped world's mapping (and locked or pre-faulted pages) is released when the world drops, instead of at each caching thread's next foreign-pool operation or exit.

### Invariant Matrix

| Invariant | Holds by | Tested by (new) |
|---|---|---|
| **W1** a pool's mapping outlives every live allocation in it | strong owners (table above); caches no longer count | existing pool and world suites |
| **W2** a pool is released once its last allocation owner drops, even while other live threads cache its blocks | caches hold `Weak` | `dropped_pool_released_while_other_thread_caches_blocks` (idle thread parked holding cached blocks; `Weak::upgrade` on the inner pool is `None` after the drop) |
| **W3** a cache never dereferences a block of a dead pool | discard on failed upgrade (exit and replacement) | `thread_exit_after_pool_drop_discards_cache`, `cache_replacement_after_pool_drop_discards_cache` (Miri: a write into the unmapped window would be UB) |
| **W4** a cache whose pool is alive still returns its blocks (accounting unchanged) | upgrade-and-flush at exit and replacement | existing `slab_pool_concurrent_alloc_dealloc_interleaved` (`used() == 0` after thread exit) plus `cache_replacement_flushes_to_live_pool` |
| **W5** a fast-path match never confuses a new pool with a dead one | `Weak` keeps `ArcInner` allocated, so addresses are not reused | covered by W3's tests, which create a new pool after the drop |

### Convention Sheet

- Pool identity in a cache is `Weak::as_ptr(&cache.pool) == Arc::as_ptr(inner)`, always the `SlabPoolInner` address.
- A cache operation dereferences blocks only through a strong reference: the caller's `&self.inner`, or a successful `upgrade()` held for the operation.

## Alternatives Considered

- **Owner-side flush on drop (issue option 2).** It covers only the dropping thread, and the real fix is no larger. Superseded.
- **`upgrade()` on every fast-path operation.** Correct, but it adds a CAS on the strong count to every allocate and deallocate. The caller already holds a strong reference, so it buys nothing.
- **Multi-pool cache keyed by pool id (issue option 3).** This would also stop cache ping-pong between a world's pool and the scratch pool (#285), but it is an independent performance change. It composes with this design (each entry holds a `Weak`), so it is left out of scope.
- **Global retired-pool epoch.** Unneeded: no cache dereferences a block without a strong reference, so there is no window in which a retired pool must stay mapped.

## Semantic Review

1. **Can a cached block be handed out from a dead pool?** No. `allocate` hands out blocks only after the address matches its own live `self.inner`, and (W5) a dead pool's address cannot match a live one.
2. **Can a cached block be freed into a dead pool?** No. Flushes go through a strong reference (rules 1–4), and a failed upgrade discards.
3. **Race between a drop and another thread's exit flush?** The exiting thread either upgrades first, so it holds the pool through the flush and the final drop happens when it releases it, or it fails and discards. `Weak::upgrade` is atomic with respect to the last strong drop.
4. **Accounting.** `used_bytes` of a dead pool is unobservable. For live pools nothing changes: exit and replacement still flush.
5. **Loom.** Under `cfg(loom)` the cache is bypassed (`pool.rs:1358-1361, 1413-1416`), so the loom models are unaffected.
6. **Cost.** `Arc::downgrade` instead of `Arc::clone` on cache creation, a pointer compare instead of `Arc::ptr_eq` on the fast path, and one `upgrade` per cache replacement or thread exit. Gate: `simple_insert/*` and `add_remove/*` within noise.

## Implementation Plan

1. `TCache::pool: Weak<SlabPoolInner>`; `refill` / `flush_all` / `spill` take `&SlabPoolInner`; `Drop` upgrades or discards; allocate / deallocate / flush_caches compare by address and use the caller's pool.
2. Tests W2–W4, run under Miri as well.
3. Docs: `memory-management.md` (a dropped world releases its pool immediately) and CHANGELOG; refresh the `flush_pool_caches` doc.
4. Gates: fmt, clippy (1.97, 1.99), workspace tests, Miri on `pool::tests`, TSan, loom, and benches against main. Re-run the `memory_footprint` retention measurement.
