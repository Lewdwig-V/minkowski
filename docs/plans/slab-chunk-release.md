# Returning Fully-Free Slab Chunks (#280)

**Status: implemented on branch `claude/slab-chunk-release`, shelved.** The design works at the allocator level:
- R1–R6 have tests, and each fails with its mechanism removed;
- the R2 and R3 loom models pass;
- `add_remove/*` and `simple_insert/*` are within noise.

But it shows **no measurable gain on `World` workloads**. Through the `World` API, pool memory is almost never freed:
- `despawn` and component removal keep column capacity, and columns never shrink;
- sparse pages and entity bookkeeping are never freed;
- the only frees are old buffers released by doubling growth, which are small and spread across shared chunks.

The `pool_fragmentation` scenarios, plus a fill / despawn-all / refill churn, give identical numbers on `main` and the branch:

| Scenario | Result (main = branch) |
|---|---|
| small columns | 93.5% |
| one big column | 56.2% |
| mixed | 88.4% / 97.2% |
| uniform | 95.3% |
| churn, 8 B → 256 B | phase 2 at 96.9% |

Revisit when a feature frees pool memory: column shrink-to-fit, archetype cleanup, or sparse page release.

## Problem

The bottom frontier carves 64 KiB chunks for one slab class at a time, and a carved chunk then belongs to that class forever (`Frontier::carve_chunk`, `pool.rs:473`; `SlabPoolInner::carve`, `pool.rs:837`). Freed blocks go back onto their class's free list (`slab_deallocate`, `pool.rs:1060`), and nothing ever hands a chunk back to the frontier.

So each class's capacity is set by its **peak** usage, not its live usage. A workload whose size mix shifts can hit `PoolExhausted` while most of the budget sits free under another class. For example: a burst of 256 B columns, a despawn, then growth in 4 KiB columns. The same holds for large runs: once the bottom frontier has advanced, the top region can never take that space back. `slab_pool_one_class_can_exhaust_the_frontier` (`pool.rs:1973`) pins this today: after class 1 fills a 512 KiB pool and frees everything, a class 0 allocation still fails.

## Current State

Verified on `main` (`a58ba2c`).

| Fact | Where |
|---|---|
| Bottom region `[0, lo)` is carved chunks; `lo` only grows | `Frontier` invariants, `pool.rs:440` |
| A chunk's class is fixed at carve; blocks are threaded onto that class's free list | `carve`, `pool.rs:837` |
| Free lists are lock-free tagged-pointer stacks (`AtomicHead`, a 128-bit CAS; a `Mutex` under loom) | `load_head` / `cas_head`, `pool.rs:300-330` |
| The frontier lock is taken once per carve and once per large alloc, free or grow, never on the free-list fast path | `pool.rs:620` |
| A free block's side-table entry is `SIDE_TABLE_UNALLOCATED`, written before the push CAS | `slab_deallocate` |
| A block in a thread cache is still **allocated** for accounting: its side-table entry is its class and `used_bytes` still counts it | `SlabPool::deallocate` TCache path, `pool.rs:1384-1407` |
| Allocation failure: `slab_allocate` = `pop_free` or `carve`, else `PoolExhausted`. `TCache::refill` calls it for the first block. `large_allocate` fails when `take_run` does | `pool.rs:882, 900, 1206` |
| Allocating from a world's pool needs `&mut World` (spawn, insert, apply), but other threads' caches can push blocks back at any time: spill on their next pool operation, or `Drop` at thread exit | #293 design |

## Proposed Design

**Reclaim on exhaustion.** When an allocation cannot be served, the pool takes every class's free list, releases each chunk whose blocks are all on its list back to the frontier, puts the rest back, and retries once. Nothing changes on the fast path. The work runs only when the pool is otherwise exhausted.

### Reclaim (`SlabPoolInner::reclaim_chunks`, under the frontier lock)

1. **Steal** each slab class's free list: CAS its head to empty (tag bumped). The reclaimer now exclusively owns that chain. Concurrent pushes go to the new list and are untouched. A concurrent pop sees an empty list and goes to the slow path (rule 3 below), so it cannot fail spuriously.
2. **Count:** walk each stolen chain and increment `counts[block_offset / CHUNK_SIZE]`. `counts` is a persistent `Box<[u16]>` of `budget / CHUNK_SIZE` entries, allocated at construction and used only under the frontier lock. A chunk of class `c` is fully free when its count reaches `CHUNK_SIZE / SIZE_CLASSES[c]` (≤ 1024, so `u16` suffices).
3. **Release** each fully-free chunk to the frontier (`Frontier::release_chunk`):
   - mark it in a free-chunk bitmap, `free_chunks: Box<[u64]>`, one bit per chunk;
   - then, while the chunk just below `lo` is marked, unmark it and lower `lo`.

   Lowering `lo` is what lets large runs reclaim the space, since the top region can extend down to `lo`.
4. **Re-push** the surviving blocks, those of chunks not released, as one chain per class (`push_chain`). Then zero `counts`.

`Frontier::carve_chunk` takes the lowest marked chunk from `free_chunks` before bumping `lo`. That chunk is below `lo`, so it doesn't affect the top region. Released chunks keep their side-table slots at `UNALLOCATED`, since every block in them was free, so carving them again is identical to a fresh carve.

### When it runs

1. `slab_allocate`: if `pop_free` and `carve` both fail, take the frontier lock, `pop_free` again, and only then reclaim and retry `carve` once.
2. `large_allocate` / `large_grow`: if `take_run` / `grow_run` fails, reclaim under the already-held lock and retry once.
3. **Slow-path recheck:** every failure path re-pops after acquiring the frontier lock. That lock serializes reclaims, so a pop that raced a steal waits for the reclaim, then sees the re-pushed blocks.
4. **Thread cache:** in `SlabPool::allocate`, when `refill` fails, the cache first flushes its own bins (`flush_all`) and then retries `global_allocate` once. Otherwise up to 32 cached blocks per class on the calling thread would pin their chunks. This runs inside `TCACHE.with`, and `flush_all` / `global_allocate` never touch `TCACHE` (existing rule).
5. **Repeat-failure guard:** reclaim is skipped when nothing was freed since the last reclaim that released nothing: `used_bytes >= used_at_last_futile_reclaim`, stored under the frontier lock. A program that sits at capacity and keeps retrying does not walk every free list on every failed allocation. `used_bytes` only falls through global frees, which are the only frees reclaim can use.

### What reclaim cannot release

- **A chunk with any block still allocated.** Fragmentation inside a chunk is unchanged. Release is per chunk, not compaction.
- **A chunk with a block in another thread's cache.** That block counts as allocated. Rule 4 flushes only the calling thread's cache, and other threads' blocks come back on their next pool operation or at thread exit, as the docs already say for `flush_pool_caches`.
- **Interior free chunks for large runs.** Large runs need 1 MiB-aligned units in the top region, so only chunks at the top of the bottom region (reached by lowering `lo`) become available to them. Interior chunks serve any slab class.

### API Surface

None public. Behavior change: `PoolExhausted` now means no class has a fully-free chunk to give up, not merely that the frontiers have met. `slab_pool_one_class_can_exhaust_the_frontier` is inverted. Docs: the `memory-management.md` paragraph on `PoolExhausted`, and the `two-frontier-pool.md` follow-up.

### Invariant Matrix

| Invariant | Holds by | Tested by (new) |
|---|---|---|
| **R1** a released chunk has no allocated or cached block | released only when all its blocks are on the stolen free lists; cached blocks keep their side-table class and are never on a free list | `reclaim_skips_chunk_with_live_block`, `reclaim_skips_chunk_with_cached_block` |
| **R2** no block is lost or duplicated by a reclaim | the steal is a single CAS; every stolen block is either in a released chunk or re-pushed; concurrent pushes go to the new head | `reclaim_preserves_free_blocks` (count equality), loom `loom_reclaim_races_push` / `loom_reclaim_races_pop` |
| **R3** an allocation that would succeed without reclaim still succeeds | slow-path recheck under the frontier lock (rule 3) | loom `loom_pop_during_reclaim_never_spuriously_fails` |
| **R4** frontier invariants hold: `lo` multiple of `CHUNK_SIZE`; `free_chunks` only below `lo`; `lo ≤ hi · LARGE_UNIT` | `release_chunk` lowers `lo` only over marked chunks; `carve_chunk` clears the bit it takes | unit tests on `Frontier` (`release_chunk_lowers_lo`, `carve_prefers_released_chunk`) |
| **R5** class shift: fill A, free all, fill B, B gets ~the whole budget | reclaim on B's first failure | inverted `slab_pool_one_class_can_exhaust_the_frontier`; `pool_fragmentation` churn scenario |
| **R6** slab, then large: fill slab, free all, allocate one large run of ~the budget | `lo` lowered by reclaim | `reclaim_lets_large_run_reuse_slab_space` |
| **R7** fast path unchanged | reclaim only on failure paths | `add_remove/*`, `simple_insert/*` benches within noise |

### Convention Sheet

- Chunk index = `window_offset / CHUNK_SIZE`; `free_chunks` bit `i` covers `[i · CHUNK_SIZE, (i + 1) · CHUNK_SIZE)`.
- "Fully free" = `counts[i] == CHUNK_SIZE / SIZE_CLASSES[class]`, counted over the stolen lists only.
- Reclaim runs only while holding the frontier lock; `counts` and `used_at_last_futile_reclaim` are only touched under it.

## Alternatives Considered

- **Per-chunk free lists (jemalloc runs, issue option a).** The cleanest design, and it would release chunks as they empty. But it replaces the single lock-free stack per class on the fast path. That is a large rewrite of the most performance-sensitive code, with loom work to match, for a case that only matters at exhaustion.
- **Explicit compaction at a `&mut World` point (option b).** For example, `flush_pool_caches` rebuilding the free lists. `&mut World` does not stop other threads' caches from pushing (thread exit), so it would need the same steal-and-re-push protocol anyway, and users would have to know to call it. Reclaim-on-exhaustion gives the benefit automatically. A public `World::compact_pool` can be added later on top of `reclaim_chunks` if wanted.
- **Lazy draining (option c).** Marking a chunk as draining and dropping its blocks as they are popped needs a per-pop chunk check on the fast path. Rejected.
- **Reclaim on every free.** Needs per-chunk occupancy counts updated on the fast path. Rejected for the same reason.

## Semantic Review

1. **Fast-path cost:** none. `allocate` / `deallocate` / TCache hits are untouched. The extra work is one lock and one recheck on paths that would otherwise return `PoolExhausted`.
2. **Reclaim cost:** O(free blocks + chunks) per reclaim, bounded by `budget / 64` blocks. It runs only at exhaustion, and the guard (rule 5) stops repeats while nothing is freed.
3. **Concurrent pushes during a reclaim** land on the new list and are not counted. Their chunks are simply not released this time (conservative).
4. **Determinism:** single-threaded exhaustion stays deterministic. Reclaim is a deterministic function of the free lists.
5. **Loom:** `AtomicHead` is a `Mutex` under loom, so the steal is a lock-and-swap. Thread caches are bypassed under loom, so the rule-4 flush is not modelled; it is single-threaded logic covered by unit tests.
6. **Memory:** `counts` costs 2 B per 64 KiB chunk (32 KiB per GiB of budget) and `free_chunks` 1 bit per chunk. Both are system heap, allocated at construction like the side table.

## Risks

- **Pathological re-push cost:** a free list of millions of blocks is walked and re-pushed on each reclaim that does release something. That is acceptable at exhaustion, and is measured in the churn example.
- **Memory stranded in other threads' caches** still blocks release until those threads touch the pool. This is unchanged and already documented.

## Implementation Plan

1. `Frontier`: `free_chunks` bitmap, `release_chunk`, `carve_chunk` preferring released chunks; R4 unit tests.
2. `SlabPoolInner::reclaim_chunks` (steal, count, release, re-push) plus the futile-reclaim guard; R1, R2 and R6 tests.
3. Wire into `slab_allocate`, `large_allocate` and `large_grow` (rules 1–3) and the TCache refill fallback (rule 4); invert the pinning test (R5).
4. Loom models (R2, R3); TSan on the pool tests; the Miri pool subset.
5. `pool_fragmentation`: churn scenario (fill A, free, fill B; slab, then large), utilization before and after.
6. Docs: `memory-management.md`, `two-frontier-pool.md` follow-ups, CHANGELOG.
7. Gates: fmt, clippy (1.97, 1.99), workspace tests, loom, TSan, Miri, `fuzz_pool_columns` 60 s, benches against main.
