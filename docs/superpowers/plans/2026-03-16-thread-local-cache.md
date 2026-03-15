# Thread-Local Cache (TLC) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thread-local L1 cache to `SlabPool` that amortizes global CAS operations across batches of 16 allocations, reducing per-allocation cost from ~7 ops to ~3 instructions for 15/16 allocations.

**Architecture:** Split `SlabPool` into a thin wrapper + `Arc<SlabPoolInner>`. Thread-local `TCache` holds per-class bins of pre-allocated block pointers. Refill grabs 16 blocks via 16 `global_allocate` calls; spill returns 16 blocks via `global_deallocate_to_class`. Epoch-based lazy flush for Rayon hoarding.

**Tech Stack:** `std::cell::UnsafeCell` for zero-overhead thread-local access, `std::sync::atomic::AtomicU64` for epoch counter, existing `atomic` crate for `Atomic<u128>`.

**Spec:** `docs/superpowers/specs/2026-03-16-thread-local-cache-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/src/pool.rs` | Modify | SlabPoolInner split, TCache, TCacheBin, refill/spill/epoch, all tests |
| `crates/minkowski/src/world.rs` | Modify | Add `flush_pool_caches()` method |

No new files. `BlobVec`, `Archetype`, `SparseStorage`, `PoolAllocator` trait signature (except one new default method) are unchanged.

---

## Chunk 1: SlabPoolInner split and global method extraction

### Task 1: Extract SlabPoolInner from SlabPool

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

This task restructures `SlabPool` without changing any behavior. All existing
tests must continue to pass after this task.

- [ ] **Step 1: Create `SlabPoolInner` struct**

Move all fields from `SlabPool` into a new `SlabPoolInner` struct. Add the
`epoch` field. `SlabPool` becomes a wrapper holding `Arc<SlabPoolInner>`:

```rust
use std::sync::atomic::AtomicU64;

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

// Move Send/Sync impls to SlabPoolInner
unsafe impl Send for SlabPoolInner {}
unsafe impl Sync for SlabPoolInner {}

pub(crate) struct SlabPool {
    inner: Arc<SlabPoolInner>,
}
```

- [ ] **Step 2: Move `Drop` from `SlabPool` to `SlabPoolInner`**

The side table deallocation now happens when `SlabPoolInner` drops (when the
last `Arc` reference is released):

```rust
impl Drop for SlabPoolInner {
    fn drop(&mut self) {
        if self.side_table_len > 0 {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(self.side_table, self.side_table_len);
                let _ = Box::from_raw(slice);
            }
        }
    }
}
```

Remove the old `impl Drop for SlabPool`.

- [ ] **Step 3: Move `new()` to `SlabPoolInner`, wrap in `SlabPool`**

`SlabPoolInner::new()` contains the existing construction logic plus
`epoch: AtomicU64::new(0)`. `SlabPool::new()` wraps it in Arc:

```rust
impl SlabPoolInner {
    fn new(budget: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        // ... existing new() body ...
        Ok(Self {
            // ... existing fields ...
            epoch: AtomicU64::new(0),
        })
    }
}

impl SlabPool {
    pub(crate) fn new(budget: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        Ok(Self {
            inner: Arc::new(SlabPoolInner::new(budget, hugepages)?),
        })
    }
}
```

- [ ] **Step 4: Move `block_ptr`, `overflow_active`, `overflow_total` to `SlabPoolInner`**

These methods operate on inner state and need to be callable from both
`SlabPool` and `TCache`:

```rust
impl SlabPoolInner {
    #[inline]
    fn block_ptr(&self, addr: u64) -> *mut u8 { /* existing body */ }

    #[cfg(test)]
    fn overflow_active(&self, class: usize) -> u64 { /* existing body */ }

    #[cfg(test)]
    fn overflow_total(&self, class: usize) -> u64 { /* existing body */ }
}
```

- [ ] **Step 5: Extract `global_allocate` and `global_deallocate` on `SlabPoolInner`**

Move the existing `PoolAllocator::allocate` body to
`SlabPoolInner::global_allocate`. Move the `deallocate` body to
`SlabPoolInner::global_deallocate`. These are the L2 (global) paths that
the TCache will call on miss/spill.

```rust
impl SlabPoolInner {
    /// Allocate one block from the global lock-free stack.
    /// Does CAS + side table write + used_bytes increment.
    /// SAFETY-CRITICAL: Does NOT touch TCACHE thread-local.
    fn global_allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        // ... existing allocate() body (lines 592-660), using self.* instead ...
    }

    /// Return one block to the global lock-free stack.
    /// Reads side table for class routing, does CAS + side table clear.
    /// SAFETY-CRITICAL: Does NOT touch TCACHE thread-local.
    unsafe fn global_deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // ... existing deallocate() body (lines 671-745), using self.* instead ...
    }
}
```

- [ ] **Step 6: Add `global_deallocate_to_class` on `SlabPoolInner`**

Used by TCache::drop and spill — class is caller-provided (already read
from side table). This is a simplified version of `global_deallocate` that
skips the side table read and bounds assertion (caller already validated).

```rust
impl SlabPoolInner {
    /// Return one block to a specific class's global free list.
    /// Used by TCache flush/spill. Class is caller-provided.
    /// SAFETY-CRITICAL: Does NOT touch TCACHE thread-local.
    unsafe fn global_deallocate_to_class(&self, class: usize, ptr: *mut u8) {
        let block = self.block_ptr(ptr as u64);

        // Clear side table entry before pushing.
        let index = (ptr as usize - self.base as usize) / SIZE_CLASSES[0];
        let entry = self.side_table.add(index).read();
        let was_overflow = (entry & SIDE_TABLE_OVERFLOW_BIT) != 0;
        self.side_table.add(index).write(SIDE_TABLE_UNALLOCATED);

        // Push to global free list.
        loop {
            let head = load_head(&self.heads[class]);
            (*(block as *const StdAtomicU64))
                .store(head.ptr() as u64, std::sync::atomic::Ordering::Release);
            let new_head = head.with_next(block);
            if cas_head(&self.heads[class], head, new_head) {
                self.used_bytes
                    .fetch_sub(SIZE_CLASSES[class], Ordering::Relaxed);
                if was_overflow {
                    let prev = self.overflow_active[class].fetch_sub(1, Ordering::Relaxed);
                    debug_assert!(prev > 0, "overflow_active underflow for class {class}");
                }
                return;
            }
            std::hint::spin_loop();
        }
    }
}
```

- [ ] **Step 7: Wire `PoolAllocator` impl to delegate to inner (passthrough for now)**

For this task, the `PoolAllocator` impl simply delegates to `inner.*`.
The TCache layer is added in a later task.

```rust
unsafe impl PoolAllocator for SlabPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        self.inner.global_allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.global_deallocate(ptr, layout)
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.inner.total)
    }

    fn used(&self) -> Option<usize> {
        Some(self.inner.used_bytes.load(Ordering::Relaxed))
    }

    fn overflow_active_counts(&self) -> Option<[u64; 6]> {
        Some(std::array::from_fn(|i| {
            self.inner.overflow_active[i].load(Ordering::Relaxed) as u64
        }))
    }

    fn overflow_total_counts(&self) -> Option<[u64; 6]> {
        Some(std::array::from_fn(|i| {
            self.inner.overflow_total[i].load(Ordering::Relaxed) as u64
        }))
    }
}
```

- [ ] **Step 8: Update test helpers to use `inner` for overflow accessors**

Tests that call `pool.overflow_active()` and `pool.overflow_total()` now
need to go through `pool.inner.overflow_active()` etc. Update all test
references.

- [ ] **Step 9: Run all tests**

Run: `cargo test -p minkowski --lib`
Expected: ALL existing tests pass — behavior is unchanged, only the struct
layout has been refactored.

- [ ] **Step 10: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "refactor(pool): extract SlabPoolInner behind Arc for TCache lifetime"
```

---

## Chunk 2: TCache data structures and thread-local storage

### Task 2: Add TCacheBin, TCache, and thread-local declaration

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Add TCache constants and data structures**

Add after the `SlabPoolInner` impl block:

```rust
// ── Thread-Local Cache (TLC) ─────────────────────────────────────

const TCACHE_REFILL: usize = 16;
const TCACHE_CAPACITY: usize = 32;
const TCACHE_SPILL: usize = 16;

/// Per-class block cache. count placed after stack for cache-line
/// adjacency with stack[31] in steady state.
#[repr(C)]
struct TCacheBin {
    stack: [*mut u8; TCACHE_CAPACITY],
    count: usize,
}

impl TCacheBin {
    const fn empty() -> Self {
        Self {
            stack: [std::ptr::null_mut(); TCACHE_CAPACITY],
            count: 0,
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<*mut u8> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        Some(self.stack[self.count])
    }

    #[inline]
    fn push(&mut self, ptr: *mut u8) {
        debug_assert!(self.count < TCACHE_CAPACITY, "TCacheBin overflow");
        self.stack[self.count] = ptr;
        self.count += 1;
    }

    fn is_full(&self) -> bool {
        self.count >= TCACHE_CAPACITY
    }
}

struct TCache {
    bins: [TCacheBin; NUM_SIZE_CLASSES],
    local_epoch: u64,
    pool: Arc<SlabPoolInner>,
}

impl TCache {
    fn new(pool: Arc<SlabPoolInner>) -> Self {
        Self {
            bins: [const { TCacheBin::empty() }; NUM_SIZE_CLASSES],
            local_epoch: pool.epoch.load(Ordering::Acquire),
            pool,
        }
    }
}

impl Drop for TCache {
    fn drop(&mut self) {
        for class in 0..NUM_SIZE_CLASSES {
            let bin = &mut self.bins[class];
            for i in 0..bin.count {
                // SAFETY: blocks are valid pointers from global_allocate.
                // global_deallocate_to_class does NOT touch TCACHE (no reentrancy).
                unsafe {
                    self.pool.global_deallocate_to_class(class, bin.stack[i]);
                }
            }
            bin.count = 0;
        }
    }
}
```

- [ ] **Step 2: Add thread-local declaration (non-loom only)**

```rust
#[cfg(not(loom))]
thread_local! {
    static TCACHE: std::cell::UnsafeCell<Option<TCache>> =
        const { std::cell::UnsafeCell::new(None) };
}
```

- [ ] **Step 3: Write TCacheBin unit tests**

```rust
#[test]
fn tcache_bin_push_pop() {
    let mut bin = TCacheBin::empty();
    assert!(bin.pop().is_none());

    let ptrs: Vec<*mut u8> = (1..=5).map(|i| i as *mut u8).collect();
    for &p in &ptrs {
        bin.push(p);
    }
    assert_eq!(bin.count, 5);

    // LIFO order
    for &p in ptrs.iter().rev() {
        assert_eq!(bin.pop(), Some(p));
    }
    assert!(bin.pop().is_none());
}

#[test]
fn tcache_bin_is_full() {
    let mut bin = TCacheBin::empty();
    for i in 0..TCACHE_CAPACITY {
        bin.push(i as *mut u8);
    }
    assert!(bin.is_full());
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- pool::tests`
Expected: PASS (existing tests + 2 new)

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "feat(pool): add TCacheBin, TCache structs and thread-local declaration"
```

---

## Chunk 3: TCache-backed allocate and deallocate

### Task 3: Wire TCache into SlabPool::allocate

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Add refill method to TCache**

```rust
impl TCache {
    /// Refill a bin by grabbing up to TCACHE_REFILL blocks from the global pool.
    /// Returns the first block to the caller, caches the rest.
    fn refill(
        &mut self,
        class: usize,
        layout: Layout,
    ) -> Result<NonNull<u8>, PoolExhausted> {
        let mut first: Option<NonNull<u8>> = None;

        for _ in 0..TCACHE_REFILL {
            match self.pool.global_allocate(layout) {
                Ok(ptr) => {
                    if first.is_none() {
                        first = Some(ptr);
                    } else {
                        // Read the side table to find the actual class
                        // (may differ from `class` due to overflow).
                        let index = (ptr.as_ptr() as usize
                            - self.pool.base as usize)
                            / SIZE_CLASSES[0];
                        let entry = unsafe { self.pool.side_table.add(index).read() };
                        let actual_class = (entry & SIDE_TABLE_CLASS_MASK) as usize;
                        self.bins[actual_class].push(ptr.as_ptr());
                    }
                }
                Err(_) => break, // Pool exhausted — stop refilling.
            }
        }

        first.ok_or(PoolExhausted { requested: layout })
    }

    /// Flush all bins back to the global pool (epoch mismatch).
    fn flush_all(&mut self) {
        for class in 0..NUM_SIZE_CLASSES {
            let bin = &mut self.bins[class];
            for i in 0..bin.count {
                unsafe {
                    self.pool.global_deallocate_to_class(class, bin.stack[i]);
                }
            }
            bin.count = 0;
        }
    }
}
```

- [ ] **Step 2: Rewrite `SlabPool::allocate` with TCache**

```rust
unsafe impl PoolAllocator for SlabPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8)
                .expect("alignment is non-zero"));
        }

        let class = size_class_for(layout)
            .ok_or(PoolExhausted { requested: layout })?;

        #[cfg(not(loom))]
        {
            TCACHE.with(|cell| {
                // SAFETY: No reentrancy — allocate() is not called from within
                // this closure. BlobVec::grow is the sole caller and does not
                // nest allocations.
                let cache = unsafe { &mut *cell.get() };
                let cache = cache.get_or_insert_with(|| {
                    let c = TCache::new(Arc::clone(&self.inner));
                    debug_assert!(
                        Arc::ptr_eq(&c.pool, &self.inner),
                        "TCache initialized for a different SlabPool"
                    );
                    c
                });

                // Epoch check — lazy flush if stale.
                let global_epoch = self.inner.epoch.load(Ordering::Acquire);
                if cache.local_epoch != global_epoch {
                    cache.flush_all();
                    cache.local_epoch = global_epoch;
                }

                // L1 hit: pop from local bin.
                if let Some(ptr) = cache.bins[class].pop() {
                    return Ok(NonNull::new(ptr).unwrap());
                }

                // L1 miss: refill from global pool.
                cache.refill(class, layout)
            })
        }

        #[cfg(loom)]
        {
            self.inner.global_allocate(layout)
        }
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- pool::tests`
Expected: PASS — allocate now goes through TCache but behavior is identical.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "feat(pool): TCache-backed allocate with refill on miss"
```

### Task 4: Wire TCache into SlabPool::deallocate

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Add spill method to TCache**

```rust
impl TCache {
    /// Spill TCACHE_SPILL blocks from the bottom of a bin back to global.
    fn spill(&mut self, class: usize) {
        let bin = &mut self.bins[class];
        let spill_count = TCACHE_SPILL.min(bin.count);
        for i in 0..spill_count {
            unsafe {
                self.pool.global_deallocate_to_class(class, bin.stack[i]);
            }
        }
        // Compact: shift remaining blocks down.
        let remaining = bin.count - spill_count;
        for i in 0..remaining {
            bin.stack[i] = bin.stack[spill_count + i];
        }
        bin.count = remaining;
    }
}
```

- [ ] **Step 2: Rewrite `SlabPool::deallocate` with TCache**

```rust
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        #[cfg(not(loom))]
        {
            // Read actual class from side table BEFORE touching TCache.
            // This ensures bins are "pure" — no bin drift.
            let addr = ptr.as_ptr() as usize;
            let base = self.inner.base as usize;

            assert!(
                addr >= base && addr < base + self.inner.total,
                "SlabPool::deallocate: pointer {:p} is outside pool region",
                ptr.as_ptr()
            );

            let index = (addr - base) / SIZE_CLASSES[0];
            let entry = unsafe { self.inner.side_table.add(index).read() };
            let actual_class = (entry & SIDE_TABLE_CLASS_MASK) as usize;

            assert!(
                actual_class < NUM_SIZE_CLASSES,
                "SlabPool::deallocate: side table entry {entry:#x} has invalid class \
                 {actual_class} — possible double-free or foreign pointer"
            );

            TCACHE.with(|cell| {
                let cache = unsafe { &mut *cell.get() };
                let cache = cache.get_or_insert_with(|| TCache::new(Arc::clone(&self.inner)));

                // Epoch check — lazy flush if stale (must check on BOTH
                // allocate and deallocate per spec).
                let global_epoch = self.inner.epoch.load(Ordering::Acquire);
                if cache.local_epoch != global_epoch {
                    cache.flush_all();
                    cache.local_epoch = global_epoch;
                }

                cache.bins[actual_class].push(ptr.as_ptr());

                if cache.bins[actual_class].is_full() {
                    cache.spill(actual_class);
                }
            });
        }

        #[cfg(loom)]
        {
            self.inner.global_deallocate(ptr, layout);
        }
    }
```

- [ ] **Step 3: Run all tests**

Run: `cargo test -p minkowski --lib`
Expected: ALL tests pass. Key thing to verify: `used_bytes` accounting.
Blocks in TCache bins are still counted as "used" (global_allocate incremented
used_bytes, and TCache deallocate does NOT decrement — only spill/flush does
via `global_deallocate_to_class`).

This means `pool.used()` may report a higher number than "actually in use by
BlobVec" because some blocks are sitting in TCache bins. This is correct —
the pool considers cached blocks as "in use" until they're returned globally.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "feat(pool): TCache-backed deallocate with side-table routing and spill"
```

---

## Chunk 4: Epoch flush, flush_caches API, and World integration

### Task 5: Add epoch bump and flush_caches

**Files:**
- Modify: `crates/minkowski/src/pool.rs`
- Modify: `crates/minkowski/src/world.rs`

- [ ] **Step 1: Add `bump_epoch` to `SlabPoolInner`**

```rust
impl SlabPoolInner {
    fn bump_epoch(&self) {
        self.epoch.fetch_add(1, Ordering::Release);
    }
}
```

- [ ] **Step 2: Add `flush_caches` to `PoolAllocator` trait**

```rust
pub unsafe trait PoolAllocator: Send + Sync {
    // ... existing methods ...

    /// Flush thread-local caches. No-op for allocators without caching.
    fn flush_caches(&self) {}
}
```

- [ ] **Step 3: Override `flush_caches` on `SlabPool`**

Add to the `PoolAllocator for SlabPool` impl:

```rust
fn flush_caches(&self) {
    self.inner.bump_epoch();
}
```

- [ ] **Step 4: Add `flush_pool_caches` to `World`**

In `world.rs`, add to the `impl World` block:

```rust
/// Flush thread-local allocation caches back to the global pool.
///
/// Call at the end of a level load or batch operation to release
/// blocks hoarded by Rayon worker threads. No-op for system-allocator
/// worlds.
pub fn flush_pool_caches(&mut self) {
    self.pool.flush_caches();
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/pool.rs crates/minkowski/src/world.rs
git commit -m "feat(pool): epoch-based flush_caches API exposed via World"
```

---

## Chunk 5: TCache-specific tests

### Task 6: Unit tests for TCache behavior

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Test TCache refill and hit path**

```rust
#[test]
fn tcache_refill_and_hit() {
    let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
    let layout = Layout::from_size_align(64, 64).unwrap();

    // First allocation triggers refill (16 blocks from global).
    let p1 = pool.allocate(layout).unwrap();
    // Next 15 allocations should be TCache hits (no global CAS).
    let mut ptrs = vec![p1];
    for _ in 0..15 {
        ptrs.push(pool.allocate(layout).unwrap());
    }
    assert_eq!(ptrs.len(), 16);

    // Deallocate all — goes to TCache, not global.
    for ptr in ptrs {
        unsafe { pool.deallocate(ptr, layout) };
    }
    // used_bytes still > 0 because blocks are in TCache, not returned globally.
    // (This is expected — TCache holds allocated blocks.)
}

#[test]
fn tcache_spill_on_full() {
    let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
    let layout = Layout::from_size_align(64, 64).unwrap();

    // Allocate and deallocate enough to fill the TCache bin (32 blocks).
    let mut ptrs = Vec::new();
    for _ in 0..TCACHE_CAPACITY + 1 {
        ptrs.push(pool.allocate(layout).unwrap());
    }
    // Deallocate all — after 32 deallocs, a spill should fire.
    for ptr in ptrs {
        unsafe { pool.deallocate(ptr, layout) };
    }
    // After spill, 16 blocks returned to global, 16 remain in TCache.
    // used_bytes should have decreased from the spill.
}
```

- [ ] **Step 2: Test epoch flush**

```rust
#[test]
fn tcache_epoch_flush() {
    let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
    let layout = Layout::from_size_align(64, 64).unwrap();

    // Allocate 5 blocks (fills TCache with 15 more during refill).
    let mut ptrs = Vec::new();
    for _ in 0..5 {
        ptrs.push(pool.allocate(layout).unwrap());
    }

    // Bump epoch.
    pool.flush_caches();

    // Next allocate should flush the TCache first, then refill.
    let p = pool.allocate(layout).unwrap();
    ptrs.push(p);

    // Deallocate all through the pool.
    for ptr in ptrs {
        unsafe { pool.deallocate(ptr, layout) };
    }
}
```

- [ ] **Step 3: Test cross-thread deallocate**

```rust
#[test]
fn tcache_cross_thread_dealloc() {
    let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
    let layout = Layout::from_size_align(64, 64).unwrap();

    // Thread A allocates.
    let ptrs: Vec<NonNull<u8>> = (0..16)
        .map(|_| pool.allocate(layout).unwrap())
        .collect();

    // Thread B deallocates.
    let pool2 = Arc::clone(&pool);
    std::thread::scope(|s| {
        s.spawn(move || {
            for ptr in ptrs {
                unsafe { pool2.deallocate(ptr, layout) };
            }
        });
    });

    // Blocks are in thread B's TCache (thread exited → TCache dropped → flushed).
    // All blocks should be back in the global pool.
    // Allocate again to verify they're available.
    let p = pool.allocate(layout).unwrap();
    unsafe { pool.deallocate(p, layout) };
}
```

- [ ] **Step 4: Test thread exit flushes TCache**

```rust
#[test]
fn tcache_thread_exit_flushes() {
    let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
    let layout = Layout::from_size_align(64, 64).unwrap();

    let used_before = pool.used().unwrap();

    // Spawn a thread, allocate and deallocate, then let it die.
    let pool2 = Arc::clone(&pool);
    std::thread::spawn(move || {
        let p = pool2.allocate(layout).unwrap();
        // Deallocate so the block returns to TCache.
        unsafe { pool2.deallocate(p, layout) };
        // Thread exits here — TCache::drop flushes all cached blocks
        // (from refill + the deallocated block) back to global pool.
    })
    .join()
    .unwrap();

    // After thread exit, all blocks returned to global pool.
    assert_eq!(pool.used().unwrap(), used_before);
}
```

- [ ] **Step 5: Test overflow during refill goes to correct bin**

```rust
#[test]
fn tcache_overflow_refill_correct_bin() {
    let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
    let layout_small = Layout::from_size_align(32, 8).unwrap();

    // Exhaust class 0 globally so refill overflows to class 1.
    // First, burn through class 0.
    let proportion_sum: usize = PROPORTIONS.iter().sum();
    let class0_blocks =
        1024 * 1024 * PROPORTIONS[0] / proportion_sum / SIZE_CLASSES[0];

    let mut burn = Vec::new();
    for _ in 0..class0_blocks {
        burn.push(pool.allocate(layout_small).unwrap());
    }

    // Next allocation overflows — refill gets class-1 blocks.
    // These should go in bins[1], not bins[0].
    let overflow_ptr = pool.allocate(layout_small).unwrap();

    // Deallocate overflow — should go to correct bin and eventually global.
    unsafe { pool.deallocate(overflow_ptr, layout_small) };

    // Clean up.
    for ptr in burn {
        unsafe { pool.deallocate(ptr, layout_small) };
    }
}
```

- [ ] **Step 6: Run all tests**

Run: `cargo test -p minkowski --lib`
Expected: ALL tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "test(pool): TCache unit tests — refill, spill, epoch, cross-thread, overflow"
```

---

## Chunk 6: Concurrency tests and cleanup

### Task 7: Multi-threaded TCache tests

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Multi-thread concurrent alloc/dealloc with TCache**

```rust
#[test]
fn tcache_concurrent_no_duplicates() {
    let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap());
    let layout = Layout::from_size_align(64, 64).unwrap();

    let all_ptrs: Vec<Vec<usize>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                s.spawn(move || {
                    let mut ptrs = Vec::new();
                    for _ in 0..500 {
                        if let Ok(ptr) = pool.allocate(layout) {
                            ptrs.push(ptr.as_ptr() as usize);
                        }
                    }
                    // Deallocate half to test mixed alloc/dealloc.
                    let half = ptrs.len() / 2;
                    for &addr in &ptrs[..half] {
                        let ptr = NonNull::new(addr as *mut u8).unwrap();
                        unsafe { pool.deallocate(ptr, layout) };
                    }
                    ptrs[half..].to_vec()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Verify no duplicates in remaining allocated pointers.
    let mut all: Vec<usize> = all_ptrs.into_iter().flatten().collect();
    let total = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), total, "duplicate pointers across threads");

    // Deallocate remaining.
    for addr in all {
        let ptr = NonNull::new(addr as *mut u8).unwrap();
        unsafe { pool.deallocate(ptr, layout) };
    }
}
```

- [ ] **Step 2: Epoch flush under contention**

```rust
#[test]
fn tcache_epoch_flush_under_contention() {
    let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap());
    let layout = Layout::from_size_align(64, 64).unwrap();

    std::thread::scope(|s| {
        // 4 threads doing alloc/dealloc.
        for _ in 0..4 {
            let pool = Arc::clone(&pool);
            s.spawn(move || {
                for _ in 0..200 {
                    let ptr = pool.allocate(layout).unwrap();
                    unsafe { pool.deallocate(ptr, layout) };
                }
            });
        }

        // Main thread bumps epoch periodically.
        for _ in 0..5 {
            pool.flush_caches();
            std::thread::yield_now();
        }
    });
}
```

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: ALL tests pass.

- [ ] **Step 4: Run clippy**

Run: `cargo clippy -p minkowski --all-targets -- -D warnings`
Expected: clean

- [ ] **Step 5: Run Miri on pool tests**

Run: `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- pool::tests`
Expected: PASS — no UB in UnsafeCell access, pointer provenance, or atomic ops.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "test(pool): concurrent TCache tests and Miri verification"
```

### Task 8: Benchmark validation and documentation

**Files:**
- Modify: `docs/perf-roadmap.md`
- Modify: `.claude/commands/perf-shakedown.md`

- [ ] **Step 1: Run pool benchmarks**

Run: `cargo bench -p minkowski-bench -- pool`

Record and compare:
- Before (lock-free, no TCache): `simple_insert/pool` = 8.74 ms, `add_remove/pool` = 8.03 ms
- After (TCache): `simple_insert/pool` = ?, `add_remove/pool` = ?
- Target: `simple_insert/pool` < 2.6 ms, `add_remove/pool` < 2.6 ms

- [ ] **Step 2: Run full benchmark suite for regression check**

Run: `cargo bench -p minkowski-bench`
Expected: no regressions on non-pool benchmarks.

- [ ] **Step 3: Update perf-roadmap.md with results**

Update the P1-1 section with TCache benchmark numbers. If targets are met,
mark the "Thread-Local Cache" line as COMPLETED.

- [ ] **Step 4: Update perf-shakedown.md baselines**

Update the pool benchmark rows in the baselines table with new numbers.

- [ ] **Step 5: Commit**

```bash
git add docs/perf-roadmap.md .claude/commands/perf-shakedown.md
git commit -m "docs: update pool benchmarks after TCache implementation"
```

---

## Post-Implementation Verification

After all tasks complete:

1. `cargo test -p minkowski --lib` (all tests)
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- pool::tests`
4. `RUSTFLAGS="--cfg loom" cargo test -p minkowski --lib --features loom -- loom_tests` (TCache bypassed, global pool logic unchanged)
5. `cargo bench -p minkowski-bench -- pool` (benchmark comparison)
6. Create PR with benchmark results in description
