# Thread-Local Cache (TLC) for SlabPool

**Date**: 2026-03-16
**Status**: Approved
**Scope**: `crates/minkowski/src/pool.rs` (internal to `SlabPool`)
**Depends on**: Lock-free slab pool (PR #113, merged)

## Problem

The lock-free `SlabPool` performs 7 operations per allocation (size-class
lookup, 128-bit CAS, side table write, `used_bytes` atomic increment, etc.)
compared to jemalloc's ~1 operation (thread-local pointer bump). Benchmarks
show a 4.3x gap on single-threaded spawn workloads. The CAS-per-allocation
model is correct but expensive — the overhead is structural, not a bug.

## Solution

Add a thread-local L1 cache inside `SlabPool` that amortizes the global
lock-free operations across batches of 16 allocations. 15 out of 16
allocations become a pointer pop from a thread-local array (~3 instructions).
The 16th triggers a batch refill from the global lock-free stack.

## Architecture

Two-tier memory hierarchy, invisible to consumers:

```
User (BlobVec) → pool.allocate(layout)
                       │
              ┌─── SlabPool::allocate ───┐
              │  1. Epoch check           │
              │  2. TCache bin pop        │  ← L1: ~3 instructions
              │     (hit? return)         │
              │  3. Global refill (miss)  │  ← L2: 16× CAS + side table
              │  4. Return 1, cache 15    │
              └───────────────────────────┘
```

- The `PoolAllocator` trait is unchanged — TCache is invisible to consumers
- `SystemAllocator` has no TCache (jemalloc provides its own)
- The TCache is a pure caching layer over the existing lock-free stack

## Design

### Struct Split: SlabPool → SlabPool + SlabPoolInner

`SlabPool` becomes a thin wrapper. The backing state moves to `SlabPoolInner`
behind an `Arc`, so thread-local caches can hold a reference that keeps the
mmap region alive.

```rust
pub(crate) struct SlabPool {
    inner: Arc<SlabPoolInner>,
}

struct SlabPoolInner {
    _region: MmapRegion,
    base: *mut u8,
    total: usize,
    heads: [AtomicHead; NUM_SIZE_CLASSES],
    side_table: *mut u8,
    side_table_len: usize,
    used_bytes: AtomicUsize,
    overflow_active: [AtomicUsize; NUM_SIZE_CLASSES],
    overflow_total: [AtomicUsize; NUM_SIZE_CLASSES],
    epoch: AtomicU64,
}
```

The existing `allocate`/`deallocate` logic moves to `SlabPoolInner` as
`global_allocate` / `global_deallocate`. `SlabPool::allocate` checks the
TCache first, then falls through to `inner.global_allocate()`.

**`SlabPoolInner` method contracts (SAFETY-CRITICAL):**

```rust
impl SlabPoolInner {
    /// Allocate one block from the global lock-free stack.
    /// Does CAS + side table write + used_bytes increment.
    /// Does NOT touch TCACHE thread-local.
    fn global_allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted>;

    /// Return one block to the global lock-free stack.
    /// Reads side table for class routing, does CAS + side table clear
    /// + used_bytes decrement + overflow counter update.
    /// Does NOT touch TCACHE thread-local.
    fn global_deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Return one block to a specific class's global free list.
    /// Used by TCache::drop and spill paths. Class is caller-provided
    /// (already read from side table by the TCache layer).
    /// Does NOT touch TCACHE thread-local.
    fn global_deallocate_to_class(&self, class: usize, ptr: *mut u8);
}
```

**Reentrancy invariant**: `global_allocate`, `global_deallocate`, and
`global_deallocate_to_class` must NEVER access the `TCACHE` thread-local.
All TCache interactions happen exclusively in `SlabPool::allocate` and
`SlabPool::deallocate`. The `TCache::drop` flush and the spill path both
call `global_deallocate_to_class` directly on `SlabPoolInner`, bypassing
`SlabPool::deallocate`. This prevents reentrancy on the
`UnsafeCell<Option<TCache>>` during destruction.

### TCache Data Structure

```rust
const TCACHE_REFILL: usize = 16;   // blocks grabbed per refill
const TCACHE_CAPACITY: usize = 32; // max blocks per bin before spill
const TCACHE_SPILL: usize = 16;    // blocks returned on overflow

/// Per-class block cache. 264 bytes (256 stack + 8 count).
///
/// `count` is placed AFTER `stack` so that `stack[31]` (the most recent
/// push in steady state) and `count` are physically adjacent — the CPU
/// can fetch both in a single 64-byte cache-line burst when the bin is
/// near-full.
///
/// The full TCache (6 bins) is ~1.6 KB — about 5% of a 32 KB L1d cache.
/// This is small enough to be a "quiet neighbor" for component data during
/// archetype iteration. Hardware prefetching handles the linear stack
/// access pattern effectively.
#[repr(C)]
struct TCacheBin {
    stack: [*mut u8; TCACHE_CAPACITY],
    count: usize,
}

struct TCache {
    bins: [TCacheBin; NUM_SIZE_CLASSES],
    local_epoch: u64,
    pool: Arc<SlabPoolInner>,
}
```

**Reentrancy safety**: `allocate(&self)` accesses the `UnsafeCell` via
`TCACHE.with(|cell| { ... })`, gets `&mut TCache`, performs the pop or
refill, and returns. There is no callback, closure, or yield point within
this critical section that could trigger a nested `allocate()` call. The
BlobVec growth path (`push → grow → pool.allocate`) is the sole caller,
and growth never occurs from within the allocator itself. `TCache::drop`
avoids reentrancy by calling `global_deallocate_to_class` on `SlabPoolInner`
directly, bypassing `SlabPool::deallocate`.
```

Thread-local storage uses `UnsafeCell<Option<TCache>>` rather than `RefCell`
to avoid runtime borrow checking on the ultra-hot allocation path. The
`Option` handles lazy initialization on first access per thread.

```rust
thread_local! {
    static TCACHE: UnsafeCell<Option<TCache>> = const { UnsafeCell::new(None) };
}
```

**Multi-pool guard**: A process may have at most one `SlabPool` (multiple
`World`s share the same pool, or use `SystemAllocator`). On TCache init,
`debug_assert!` verifies `Arc::ptr_eq(&cache.pool, &self.inner)`. In
practice, `WorldBuilder` creates one pool per process — multiple pools are
a configuration error, not a supported use case.

**Loom**: Under `cfg(loom)`, the TCache is disabled entirely. `SlabPool::allocate`
and `deallocate` call `inner.global_allocate()` / `inner.global_deallocate()`
directly, bypassing the thread-local. This matches the lock-free pool spec's
approach of shimming `AtomicHead` under loom — loom tests verify the global
pool's CAS logic and epoch atomics, not the TCache (which is thread-local
and contention-free by definition).

### Allocation Fast Path

```
fn allocate(&self, layout) -> Result<NonNull<u8>, PoolExhausted>:
    1. Zero-size fast path (unchanged)
    2. class = size_class_for(layout)
    3. Access TCACHE thread-local:
       a. Lazy-init if None (clone Arc<SlabPoolInner>)
       b. Epoch check: if local_epoch != global_epoch, flush all bins
       c. If bins[class].count > 0:
          - count -= 1
          - return stack[count]              ← HOT PATH: ~3 instructions
       d. Else: refill(class, layout)
```

**Refill** calls `inner.global_allocate()` up to TCACHE_REFILL (16) times.
Each call does the existing CAS + side table + used_bytes logic. Returns 1
block to the caller, pushes up to 15 into the local bin.

If the target class is exhausted globally and overflows to a larger class,
the overflow blocks go into the bin matching their *actual* class (read from
the side table after global_allocate returns). The user receives a block that
satisfies their layout (the overflow class is always >= the requested class).

### Deallocation Fast Path

```
fn deallocate(&self, ptr, layout):
    1. Zero-size fast path (unchanged)
    2. Read side table to find actual_class   ← 1 byte, L1-hot
    3. Access TCACHE thread-local:
       a. Lazy-init if None
       b. bins[actual_class].stack[count] = ptr
       c. count += 1
       d. If count >= TCACHE_CAPACITY:
          - spill(actual_class)              ← return 16 blocks to global
```

**Side-table-first routing**: The actual class is read from the side table
(not derived from `layout`) before pushing to the local bin. This ensures
bins are "pure" — each bin only contains blocks from its designated size
class. Benefits:

1. **No bin drift**: Overflow blocks go to the correct bin, not the requested
   class bin. Prevents accumulation of class-1 blocks in bins[0].
2. **Batch spill**: When a bin spills, all 16 blocks belong to the same class.
   No per-block side table re-scan needed during global_deallocate.
3. **Cross-thread dealloc**: Thread A allocates (side table written), thread B
   deallocates (side table read) — works correctly because the side table was
   populated during refill.

**Spill** calls `inner.global_deallocate()` for TCACHE_SPILL (16) blocks from
the bottom of the bin (oldest first for cache locality). After spill, the bin
retains TCACHE_CAPACITY - TCACHE_SPILL (16) blocks.

### Epoch-Based Lazy Flush

Rayon threads are long-lived and may hoard blocks indefinitely. The epoch
mechanism provides a manual pressure valve:

```rust
impl SlabPoolInner {
    fn bump_epoch(&self) {
        self.epoch.fetch_add(1, Ordering::Release);
    }
}
```

On every `allocate` and `deallocate`, the TCache compares its `local_epoch`
to the global `epoch` (a single `Acquire` load — near-zero cost on x86,
ensures visibility of the `Release` store in `bump_epoch` on ARM). If they
differ, all bins are flushed to the global pool before proceeding.

**Exposed via `PoolAllocator` trait:**

```rust
pub unsafe trait PoolAllocator: Send + Sync {
    // ... existing methods ...
    fn flush_caches(&self) {}  // default no-op
}
```

`SlabPool` overrides to call `inner.bump_epoch()`. `SystemAllocator` uses
the default no-op. `World` exposes this as `world.flush_pool_caches()`.

Use case: call at the end of a level load or batch spawn/despawn operation
to release hoarded blocks back to the global pool for other threads.

### Drop and Lifetime Management

**Ownership chain:**

```
World → SlabPool → Arc<SlabPoolInner>
                         ↑
thread_local TCache ─────┘ (also holds Arc<SlabPoolInner>)
```

**TCache::drop:**

```rust
impl Drop for TCache {
    fn drop(&mut self) {
        for class in 0..NUM_SIZE_CLASSES {
            let bin = &mut self.bins[class];
            if bin.count > 0 {
                // Return all cached blocks to the global pool.
                for i in 0..bin.count {
                    self.pool.global_deallocate_single(class, bin.stack[i]);
                }
                bin.count = 0;
            }
        }
        // Arc<SlabPoolInner> drops here.
    }
}
```

**Pool drop ordering:** `Arc` reference counting guarantees `SlabPoolInner`
(and its mmap region) outlives all TCache instances, regardless of
thread-local drop order. When `World` drops, the `SlabPool`'s Arc refcount
decrements. If threads still hold TCaches, `SlabPoolInner` stays alive until
the last TCache drops.

**Main thread at program exit:** `thread_local!` destructors may not run for
the main thread on some platforms. Cached blocks leak cosmetically — the OS
reclaims the mmap on process exit. `used_bytes` may not reach zero. This
matches jemalloc's behavior and is not a correctness issue.

### Side Table Interaction

- **Refill**: `global_allocate()` writes side table entries (unchanged from
  current implementation). Blocks enter the TCache with valid side table
  entries.
- **TCache pop (allocate)**: No side table access. The entry was written
  during refill.
- **TCache push (deallocate)**: Reads side table to determine actual class
  for correct bin placement (1 byte read, L1-hot).
- **Spill**: `global_deallocate()` reads side table to verify class, clears
  entry to `SIDE_TABLE_UNALLOCATED` (unchanged from current implementation).
- **TCache Drop (thread exit)**: Same as spill — calls `global_deallocate()`
  per block.

The side table remains the single source of truth for class routing. The
TCache never writes to the side table — it only reads on deallocate to
select the correct bin.

## API Changes

One new default method on `PoolAllocator`:

```rust
fn flush_caches(&self) {}
```

One new method on `World`:

```rust
pub fn flush_pool_caches(&mut self) {
    self.pool.flush_caches();
}
```

All other APIs unchanged. `BlobVec`, `Archetype`, `SparseStorage` are
unmodified.

## Testing Strategy

### Unit Tests

- **TCache hit path**: allocate N < TCACHE_REFILL blocks, verify no global
  CAS after the first refill batch.
- **Refill**: allocate TCACHE_REFILL + 1 blocks, verify exactly 2 global
  refill batches.
- **Spill**: deallocate TCACHE_CAPACITY blocks, verify global_deallocate
  called for TCACHE_SPILL blocks.
- **Epoch flush**: allocate blocks, bump epoch, next allocate triggers flush,
  verify bins are empty.
- **Cross-thread dealloc**: thread A allocates, thread B deallocates, verify
  correct side table routing and used_bytes accounting.
- **Overflow in refill**: exhaust class 0, verify refill returns class-1
  blocks placed in bins[1].
- **Drop cleanup**: spawn a thread, allocate blocks, join thread, verify
  used_bytes returns to zero.

### Concurrency Tests

- **Multi-thread alloc/dealloc**: 8 threads, 1000 ops each, verify no
  duplicates and used_bytes == 0 at end.
- **Epoch flush under contention**: bump epoch while threads are allocating,
  verify no lost blocks.

### Loom Tests

- **Epoch visibility**: one thread bumps epoch, another thread sees it on
  next allocate (Release/Acquire ordering on epoch).
- Note: TCache itself is thread-local (no contention), so loom tests focus
  on the epoch atomic and the global pool interactions.

### Benchmark Validation

- `simple_insert/pool`: target < 2.6 ms (within 1.5x of system allocator's
  1.74 ms). Currently 8.74 ms.
- `add_remove/pool`: target < 2.6 ms (within 2x of system allocator's
  1.30 ms). Currently 8.03 ms.

## Performance Model

| Path | Operations | Frequency |
|---|---|---|
| TCache hit (allocate) | decrement + array read | 15/16 allocations |
| TCache hit (deallocate) | side table byte read + array write + increment | 15/16 deallocations |
| Refill (allocate miss) | 16 × (CAS + side table write + atomic add) | 1/16 allocations |
| Spill (deallocate full) | 16 × (CAS + side table clear + atomic sub) | 1/16 deallocations (steady state) |
| Epoch check | 1 atomic load (Relaxed) | every allocate/deallocate |
| Epoch flush | N × global_deallocate (all cached blocks) | manual trigger only |

**Expected speedup**: 16× reduction in global CAS operations. The TCache hit
path is ~3 instructions vs the current ~20 instructions per allocation.
Combined with side table locality (contiguous byte array), the target of
1.5x system allocator should be achievable.

## Non-Goals

- **Batch CAS** (pop/push 16 blocks in one CAS by walking the intrusive
  list). Deferred — the 16-sequential-CAS approach during refill/spill is
  simpler and the TCache already reduces CAS frequency by 94%.
- **Per-class TCache sizing**. All classes use the same 32-block capacity.
  Adaptive sizing based on allocation patterns is a future optimization.
- **TCache for SystemAllocator**. jemalloc already has thread-local caching.
  Adding a second layer would add overhead, not remove it.

## Rollback

The TCache is entirely internal to `SlabPool`. Reverting means removing the
`SlabPoolInner` split and the thread-local machinery — the `PoolAllocator`
trait's `flush_caches()` default no-op can remain harmlessly. The lock-free
stack continues to work as the sole allocation path.
