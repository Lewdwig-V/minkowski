# Lock-Free Slab Pool Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mutex-guarded slab pool with a lock-free intrusive stack using tagged pointers, add a side table for deallocation routing, and implement single-step overflow.

**Architecture:** Each size class's `Mutex<Vec<*mut u8>>` becomes an `Atomic<u128>` tagged pointer head for a lock-free intrusive linked list. A byte-per-block side table tracks the actual size class for correct deallocation routing. Exhausted classes overflow to the next larger class (one step up).

**Tech Stack:** `atomic` crate for portable `Atomic<u128>`, `loom` for concurrency verification (Mutex shim under cfg(loom)), existing `memmap2` for the mmap region.

**Spec:** `docs/superpowers/specs/2026-03-14-lock-free-slab-pool-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/Cargo.toml` | Modify | Add `atomic` dependency |
| `crates/minkowski/src/pool.rs` | Modify | Replace SlabPool internals: TaggedPtr, intrusive stack, side table, overflow, telemetry |
| `crates/minkowski/src/world.rs` | Modify | Add `pool_overflow_active`/`pool_overflow_total` to `WorldStats` |
| `crates/minkowski/src/lib.rs` | Modify (if needed) | Verify `NUM_SIZE_CLASSES` const is accessible for WorldStats array |

No new files. No changes to `BlobVec`, `Archetype`, `PoolAllocator` trait, `SharedPool`, or any code outside `pool.rs` and `world.rs`.

---

## Chunk 1: Foundation — TaggedPtr, AtomicHead shim, dependency

### Task 1: Add `atomic` dependency

**Files:**
- Modify: `crates/minkowski/Cargo.toml`

- [ ] **Step 1: Add atomic crate**

In `crates/minkowski/Cargo.toml`, add to `[dependencies]`:

```toml
atomic = "0.6"
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p minkowski`
Expected: success (no code uses it yet)

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/Cargo.toml
git commit -m "deps: add atomic crate for portable Atomic<u128>"
```

### Task 2: TaggedPtr and AtomicHead abstraction

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Write tests for TaggedPtr**

Add to the `tests` module in `pool.rs`:

```rust
#[test]
fn tagged_ptr_round_trip() {
    let ptr = 0xDEAD_BEEF_u64;
    let tag = 42_u64;
    let packed = (ptr as u128) | ((tag as u128) << 64);
    let (got_ptr, got_tag) = (packed as u64, (packed >> 64) as u64);
    assert_eq!(got_ptr, ptr);
    assert_eq!(got_tag, tag);
}

#[test]
fn tagged_ptr_empty() {
    let tp = TaggedPtr::empty();
    assert!(tp.is_empty());
    assert_eq!(tp.ptr(), std::ptr::null_mut());
}

#[test]
fn tagged_ptr_with_real_pointer() {
    let mut val = 0u64;
    let ptr = &mut val as *mut u64 as *mut u8;
    let tp = TaggedPtr::new(ptr, 7);
    assert!(!tp.is_empty());
    assert_eq!(tp.ptr(), ptr);
    assert_eq!(tp.tag(), 7);
}
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cargo test -p minkowski --lib -- pool::tests::tagged_ptr`
Expected: FAIL (TaggedPtr not defined)

- [ ] **Step 3: Implement TaggedPtr**

Add above the `SlabPool` definition in `pool.rs`:

```rust
// ── Lock-free intrusive stack ────────────────────────────────────

/// Tagged pointer for ABA-safe lock-free stack operations.
///
/// Packed as a `u128`: low 64 bits = pointer, high 64 bits = monotonic tag.
/// The tag increments on every push/pop, preventing ABA — even if a pointer
/// is recycled to the same address, the tag will differ.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct TaggedPtr(u128);

impl TaggedPtr {
    /// Empty stack head (null pointer, tag 0).
    const fn empty() -> Self {
        Self(0)
    }

    /// Create a tagged pointer with the given raw pointer and tag.
    fn new(ptr: *mut u8, tag: u64) -> Self {
        Self((ptr as u64 as u128) | ((tag as u128) << 64))
    }

    /// The raw pointer (null if empty).
    fn ptr(self) -> *mut u8 {
        self.0 as u64 as *mut u8
    }

    /// The monotonic tag.
    fn tag(self) -> u64 {
        (self.0 >> 64) as u64
    }

    /// True if the pointer is null (stack is empty).
    fn is_empty(self) -> bool {
        (self.0 as u64) == 0
    }

    /// Return a new TaggedPtr with the same tag incremented by 1.
    fn with_next(self, ptr: *mut u8) -> Self {
        Self::new(ptr, self.tag() + 1)
    }
}

impl fmt::Debug for TaggedPtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaggedPtr({:p}, tag={})", self.ptr(), self.tag())
    }
}
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cargo test -p minkowski --lib -- pool::tests::tagged_ptr`
Expected: PASS (3 tests)

- [ ] **Step 5: Implement AtomicHead abstraction**

Add the cfg-split type alias and helper methods below `TaggedPtr`:

```rust
/// Atomic head for a lock-free intrusive stack.
///
/// Under `cfg(loom)`, falls back to a Mutex because loom cannot model
/// 128-bit atomic operations. The Mutex shim verifies logical correctness
/// (no lost blocks, no duplicates) but not CAS retry paths.
#[cfg(not(loom))]
type AtomicHead = atomic::Atomic<u128>;

#[cfg(loom)]
type AtomicHead = Mutex<u128>;

/// Load the current head of a free list.
#[inline]
fn load_head(head: &AtomicHead) -> TaggedPtr {
    #[cfg(not(loom))]
    {
        TaggedPtr(head.load(Ordering::Acquire))
    }
    #[cfg(loom)]
    {
        TaggedPtr(*head.lock())
    }
}

/// Compare-and-swap the head of a free list. Returns `true` on success.
#[inline]
fn cas_head(head: &AtomicHead, current: TaggedPtr, new: TaggedPtr) -> bool {
    #[cfg(not(loom))]
    {
        head.compare_exchange(current.0, new.0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
    #[cfg(loom)]
    {
        let mut guard = head.lock();
        if *guard == current.0 {
            *guard = new.0;
            true
        } else {
            false
        }
    }
}

/// Create a new AtomicHead initialized with the given TaggedPtr.
fn new_atomic_head(tp: TaggedPtr) -> AtomicHead {
    #[cfg(not(loom))]
    {
        atomic::Atomic::new(tp.0)
    }
    #[cfg(loom)]
    {
        Mutex::new(tp.0)
    }
}
```

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p minkowski`
Expected: success (AtomicHead defined but not yet used by SlabPool)

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "feat(pool): add TaggedPtr and AtomicHead abstraction for lock-free stack"
```

---

## Chunk 2: Side table and SlabPool struct migration

### Task 3: Side table constants and SlabPool struct change

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Write side table index tests**

Add to `tests` module:

```rust
#[test]
fn side_table_index_computation() {
    let base = 0x1000_usize as *mut u8;
    let min_block = SIZE_CLASSES[0]; // 64
    // Block at base+0 → index 0
    assert_eq!((0x1000 - 0x1000) / min_block, 0);
    // Block at base+64 → index 1
    assert_eq!((0x1040 - 0x1000) / min_block, 1);
    // Block at base+128 → index 2
    assert_eq!((0x1080 - 0x1000) / min_block, 2);
}

#[test]
fn side_table_overflow_flag() {
    let entry_normal: u8 = 3; // class 3, no overflow
    let entry_overflow: u8 = 3 | SIDE_TABLE_OVERFLOW_BIT;
    assert_eq!(entry_normal & SIDE_TABLE_CLASS_MASK, 3);
    assert_eq!(entry_overflow & SIDE_TABLE_CLASS_MASK, 3);
    assert_eq!(entry_normal & SIDE_TABLE_OVERFLOW_BIT, 0);
    assert_ne!(entry_overflow & SIDE_TABLE_OVERFLOW_BIT, 0);
}
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cargo test -p minkowski --lib -- pool::tests::side_table`
Expected: FAIL (constants not defined)

- [ ] **Step 3: Add side table constants**

Add below `PROPORTIONS`:

```rust
/// Sentinel value for unallocated side table entries.
const SIDE_TABLE_UNALLOCATED: u8 = 0xFF;

/// Mask for the class index in a side table entry (bits 0..3).
const SIDE_TABLE_CLASS_MASK: u8 = 0x0F;

/// Bit flag indicating this allocation overflowed from a smaller class.
const SIDE_TABLE_OVERFLOW_BIT: u8 = 0x80;
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cargo test -p minkowski --lib -- pool::tests::side_table`
Expected: PASS (2 tests)

- [ ] **Step 5: Rewrite SlabPool struct**

Replace the `SlabPool` struct definition (keeping MmapRegion, HugePages, etc. unchanged).

The side table is stored as a raw pointer + length because we need to write
through `&self` (after a successful CAS, the block is owned by the writing
thread — no concurrent access). Using `Vec<u8>` or `Box<[u8]>` would require
deriving a `*mut` from a shared reference, which is UB. A raw pointer owned
by the struct avoids this:

```rust
pub(crate) struct SlabPool {
    _region: MmapRegion,
    base: *mut u8,
    total: usize,
    heads: [AtomicHead; NUM_SIZE_CLASSES],
    /// Deallocation class routing table. One byte per MIN_BLOCK_SIZE slot.
    /// Owned raw pointer — written after successful CAS (single-owner),
    /// read before CAS on dealloc (single-owner). No concurrent access.
    side_table: *mut u8,
    side_table_len: usize,
    used_bytes: AtomicUsize,
    overflow_active: [AtomicUsize; NUM_SIZE_CLASSES],
    overflow_total: [AtomicUsize; NUM_SIZE_CLASSES],
}
```

Add `Drop` for the side table allocation:

```rust
impl Drop for SlabPool {
    fn drop(&mut self) {
        if self.side_table_len > 0 {
            // SAFETY: side_table was allocated via Vec::into_raw_parts in new().
            unsafe {
                let _ = Vec::from_raw_parts(self.side_table, self.side_table_len, self.side_table_len);
            }
        }
    }
}
```

Update the `Send`/`Sync` SAFETY comment:

```rust
// SAFETY: Pointers into the mmap region are only accessed through atomic CAS
// operations (or Mutex under loom). The mmap region is owned by SlabPool and
// outlives all allocations. The side table raw pointer is written only by
// the thread that owns the block (after successful CAS) and read before
// the CAS that returns it — single-owner access, no concurrent mutation.
unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}
```

- [ ] **Step 6: Verify compilation fails (new() and allocate/deallocate reference old fields)**

Run: `cargo check -p minkowski 2>&1 | head -5`
Expected: errors about `free_lists` not existing — confirms we need to update the methods next.

- [ ] **Step 7: Commit (WIP — struct changed, methods not yet updated)**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "wip(pool): migrate SlabPool struct to AtomicHead + side table"
```

---

## Chunk 3: Initialization — intrusive linked list construction

### Task 4: Rewrite SlabPool::new()

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Rewrite new() to chain blocks as intrusive linked list**

Replace the body of `SlabPool::new()`.

Key design decisions:
- Side table allocated as `Vec`, then converted to raw pointer via `into_raw_parts` (matched by `from_raw_parts` in `Drop`).
- Heads array built in a single pass via `from_fn` — no init-then-reassign. Each closure invocation computes the linked list for one class using a shared `&mut offset` (via `Cell` since `from_fn` takes `Fn`).
- Iterating blocks in reverse builds the list so the head points to the lowest address (cache-friendly early allocations).

```rust
pub(crate) fn new(budget: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
    let mut region = MmapRegion::new(budget, hugepages)?;
    let base = region.as_mut_ptr();
    let total = region.len();

    // Allocate side table as raw pointer (see struct doc for rationale).
    let side_table_len = total / SIZE_CLASSES[0];
    let mut st_vec = vec![SIDE_TABLE_UNALLOCATED; side_table_len];
    let side_table = st_vec.as_mut_ptr();
    std::mem::forget(st_vec); // ownership transferred to raw pointer

    let proportion_sum: usize = PROPORTIONS.iter().sum();

    // Build intrusive linked lists and heads in a single pass.
    // Use a shared offset counter to partition the mmap region sequentially.
    let mut offset: usize = 0;

    let mut head_values: [TaggedPtr; NUM_SIZE_CLASSES] = [TaggedPtr::empty(); NUM_SIZE_CLASSES];

    for class in 0..NUM_SIZE_CLASSES {
        let block_size = SIZE_CLASSES[class];

        // Align absolute address to block_size.
        let abs_addr = base as usize + offset;
        let aligned_addr = (abs_addr + block_size - 1) & !(block_size - 1);
        offset = aligned_addr - base as usize;

        let class_bytes = total * PROPORTIONS[class] / proportion_sum;
        let block_count = class_bytes / block_size;

        // Chain blocks as intrusive linked list (reverse iteration
        // so head points to lowest address for cache locality).
        let mut first_block: *mut u8 = std::ptr::null_mut();
        for i in (0..block_count).rev() {
            let block_offset = offset + i * block_size;
            if block_offset + block_size > total {
                continue;
            }
            // SAFETY: block_offset + block_size <= total, base is valid
            // for total bytes. Block is ≥64 bytes (MIN_BLOCK_SIZE invariant),
            // so writing 8 bytes of next-pointer is within bounds.
            let block_ptr = unsafe { base.add(block_offset) };
            unsafe {
                (block_ptr as *mut u64).write(first_block as u64);
            }
            first_block = block_ptr;
        }

        head_values[class] = TaggedPtr::new(first_block, 0);
        offset += block_count * block_size;
    }

    let heads: [AtomicHead; NUM_SIZE_CLASSES] =
        std::array::from_fn(|i| new_atomic_head(head_values[i]));

    Ok(Self {
        _region: region,
        base,
        total,
        heads,
        side_table,
        side_table_len,
        used_bytes: AtomicUsize::new(0),
        overflow_active: std::array::from_fn(|_| AtomicUsize::new(0)),
        overflow_total: std::array::from_fn(|_| AtomicUsize::new(0)),
    })
}
```

- [ ] **Step 2: Verify new() compiles (allocate/deallocate still broken)**

Run: `cargo check -p minkowski 2>&1 | grep "error"`
Expected: errors only in `allocate` and `deallocate` methods, not in `new()`

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "feat(pool): rewrite SlabPool::new() with intrusive linked list initialization"
```

---

## Chunk 4: Lock-free allocate and deallocate

### Task 5: Implement lock-free allocate with overflow

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Write allocate test for basic round-trip**

The existing `slab_pool_allocate_and_deallocate` test covers this. Keep it unchanged — it should pass once both methods are implemented.

- [ ] **Step 2: Write overflow test**

Add to `tests` module:

```rust
#[test]
fn slab_pool_overflow_to_next_class() {
    // Small pool so class 0 exhausts quickly.
    let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
    let layout_small = Layout::from_size_align(32, 8).unwrap();

    // Exhaust class 0 (64B).
    let mut ptrs = Vec::new();
    while let Ok(ptr) = pool.allocate(layout_small) {
        ptrs.push(ptr);
    }

    // This fails in the old code (no overflow). With overflow, it should
    // succeed by taking a block from class 1 (256B).
    // We need to deallocate one to make class 0 exhausted again after
    // the test setup, but actually class 0 IS exhausted. The loop above
    // allocated until failure, so class 0 is empty. Now overflow should
    // kick in on the next allocate.
    //
    // Wait — the loop consumed all class 0 blocks and then returned Err.
    // But overflow means the Err should NOT have been returned — overflow
    // tries class 1. So the loop itself will overflow. We need a
    // different approach: exhaust ALL classes to verify PoolExhausted.
    //
    // Let's instead test that we get MORE blocks than class 0 alone:
    let proportion_sum: usize = PROPORTIONS.iter().sum();
    let class0_bytes = 1024 * 1024 * PROPORTIONS[0] / proportion_sum;
    let class0_blocks = class0_bytes / SIZE_CLASSES[0];

    // With overflow, we should get MORE than class 0's block count.
    assert!(
        ptrs.len() > class0_blocks,
        "expected overflow: got {} blocks, class 0 has {} blocks",
        ptrs.len(),
        class0_blocks
    );

    // Clean up.
    let layout_for_dealloc = layout_small;
    for ptr in ptrs {
        unsafe { pool.deallocate(ptr, layout_for_dealloc) };
    }
    assert_eq!(pool.used(), Some(0));
}

#[test]
fn slab_pool_overflow_dealloc_returns_to_correct_class() {
    let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
    let layout_small = Layout::from_size_align(32, 8).unwrap();

    // Exhaust class 0, forcing overflow to class 1.
    let proportion_sum: usize = PROPORTIONS.iter().sum();
    let class0_blocks = 1024 * 1024 * PROPORTIONS[0] / proportion_sum / SIZE_CLASSES[0];

    let mut ptrs = Vec::new();
    for _ in 0..class0_blocks + 1 {
        ptrs.push(pool.allocate(layout_small).unwrap());
    }

    // The last allocation overflowed. Deallocate it — used_bytes should
    // decrease by the OVERFLOW class size (256), not the requested class (64).
    let used_before = pool.used().unwrap();
    unsafe { pool.deallocate(ptrs.pop().unwrap(), layout_small) };
    let used_after = pool.used().unwrap();
    let freed = used_before - used_after;
    // Overflow block came from class 1 (256B).
    assert_eq!(freed, SIZE_CLASSES[1], "overflow block should free class 1 size");

    for ptr in ptrs {
        unsafe { pool.deallocate(ptr, layout_small) };
    }
}
```

- [ ] **Step 3: Implement allocate with overflow**

Replace the `PoolAllocator::allocate` implementation:

```rust
fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
    if layout.size() == 0 {
        return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
    }

    let class = size_class_for(layout).ok_or(PoolExhausted { requested: layout })?;

    // Try the target class, then overflow to the next larger class.
    for try_class in class..NUM_SIZE_CLASSES.min(class + 2) {
        loop {
            let head = load_head(&self.heads[try_class]);
            if head.is_empty() {
                break; // This class is exhausted, try next.
            }

            // SAFETY: head.ptr() is a valid block pointer within the mmap
            // region. The block is ≥64 bytes and aligned to its class size,
            // so reading 8 bytes at offset 0 is safe and aligned.
            let next_raw = unsafe { std::ptr::read(head.ptr() as *const u64) };
            let next_ptr = if next_raw == 0 {
                std::ptr::null_mut()
            } else {
                next_raw as *mut u8
            };
            let new_head = head.with_next(next_ptr);

            if cas_head(&self.heads[try_class], head, new_head) {
                // CAS succeeded — we own this block now.
                let block_size = SIZE_CLASSES[try_class];
                self.used_bytes.fetch_add(block_size, Ordering::Relaxed);

                // Update side table: record actual class + overflow flag.
                // SAFETY: index is within bounds (ptr is in the mmap region),
                // and the block is owned by this thread (CAS succeeded).
                let index = (head.ptr() as usize - self.base as usize) / SIZE_CLASSES[0];
                let overflow = try_class != class;
                let entry = try_class as u8
                    | if overflow { SIDE_TABLE_OVERFLOW_BIT } else { 0 };
                unsafe { self.side_table.add(index).write(entry) };

                if overflow {
                    self.overflow_active[try_class].fetch_add(1, Ordering::Relaxed);
                    self.overflow_total[try_class].fetch_add(1, Ordering::Relaxed);
                }

                debug_assert!(
                    (head.ptr() as usize).is_multiple_of(layout.align()),
                    "SlabPool: block at {:p} is not aligned to {}",
                    head.ptr(), layout.align()
                );

                return Ok(NonNull::new(head.ptr()).expect("free list block is non-null"));
            }
            // CAS failed — another thread popped. Retry.
        }
    }

    Err(PoolExhausted { requested: layout })
}
```

**Important**: `self.side_table[index]` is a write to a `Vec<u8>` through `&self`. This requires interior mutability. Since the side table entry is only written when the block is owned (after CAS success, before returning to caller), we need `UnsafeCell` or use raw pointer writes:

```rust
// In the struct, change side_table to:
side_table: Box<[std::cell::UnsafeCell<u8>]>,

// And access via:
unsafe { *self.side_table[index].get() = entry; }
```

Or simpler: store as `*mut u8` raw pointer to a heap allocation:

```rust
side_table: *mut u8,
side_table_len: usize,
```

The raw pointer approach is cleaner for this use case — we already have `unsafe impl Sync`. Let's use `Box<[u8]>` and cast through a raw pointer for the write:

```rust
// Write side table entry (safe: block is owned, no concurrent access).
unsafe {
    let st_ptr = self.side_table.as_ptr() as *mut u8;
    st_ptr.add(index).write(entry);
}
```

This requires `side_table` to be allocated as a `Box<[u8]>` rather than `Vec<u8>`, and the const-pointer-to-mut cast is sound because no other thread accesses this index while the block is allocated.

- [ ] **Step 4: Implement deallocate**

Replace the `PoolAllocator::deallocate` implementation:

```rust
unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
    if layout.size() == 0 {
        return;
    }

    debug_assert!(
        ptr.as_ptr() >= self.base
            && (ptr.as_ptr() as usize) < self.base as usize + self.total,
        "SlabPool::deallocate: pointer {:p} is outside pool region",
        ptr.as_ptr()
    );

    // Read actual class from side table (authoritative — ignores layout).
    // SAFETY: index is within bounds, block is owned by this thread.
    let index = (ptr.as_ptr() as usize - self.base as usize) / SIZE_CLASSES[0];
    let entry = unsafe { self.side_table.add(index).read() };
    let actual_class = (entry & SIDE_TABLE_CLASS_MASK) as usize;
    let was_overflow = (entry & SIDE_TABLE_OVERFLOW_BIT) != 0;

    debug_assert!(
        actual_class < NUM_SIZE_CLASSES,
        "SlabPool::deallocate: side table entry {entry:#x} has invalid class {actual_class}"
    );

    // Push block back onto its actual class's free list.
    loop {
        let head = load_head(&self.heads[actual_class]);
        // Write current head pointer into block's first 8 bytes.
        unsafe {
            (ptr.as_ptr() as *mut u64).write(head.ptr() as u64);
        }
        let new_head = head.with_next(ptr.as_ptr());

        if cas_head(&self.heads[actual_class], head, new_head) {
            self.used_bytes
                .fetch_sub(SIZE_CLASSES[actual_class], Ordering::Relaxed);

            if was_overflow {
                self.overflow_active[actual_class].fetch_sub(1, Ordering::Relaxed);
            }

            // Mark side table entry as unallocated.
            // SAFETY: block is being returned, this thread still owns it.
            unsafe { self.side_table.add(index).write(SIDE_TABLE_UNALLOCATED) };

            return;
        }
        // CAS failed — another thread pushed. Retry.
    }
}
```

- [ ] **Step 5: Add overflow telemetry accessors**

Add to the `impl SlabPool` block:

```rust
/// Number of blocks in `class` currently serving overflow requests
/// from smaller classes.
#[expect(dead_code)]
pub(crate) fn overflow_active(&self, class: usize) -> u64 {
    self.overflow_active[class].load(Ordering::Relaxed) as u64
}

/// Cumulative count of overflow allocations served by `class`.
#[expect(dead_code)]
pub(crate) fn overflow_total(&self, class: usize) -> u64 {
    self.overflow_total[class].load(Ordering::Relaxed) as u64
}
```

- [ ] **Step 6: Update SAFETY comment on deallocate trait doc**

Update the `PoolAllocator` trait's `deallocate` doc:

```rust
/// Return a block to the pool.
///
/// # Safety
///
/// `ptr` must have been returned by a prior call to `allocate` on this
/// pool. The caller must not use `ptr` after this call. The `layout`
/// parameter is accepted for API compatibility but is not used for
/// class routing — the side table is authoritative.
```

- [ ] **Step 7: Run all existing pool tests**

Run: `cargo test -p minkowski --lib -- pool::tests`
Expected: PASS — existing tests work with new internals. The `slab_pool_no_cross_class_fallback` test will FAIL because overflow is now enabled. Update it:

```rust
#[test]
fn slab_pool_overflow_enabled() {
    // Verify that exhausting one size class DOES spill into the next
    // larger class (one step up). This replaced the old
    // no_cross_class_fallback test.
    let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
    let layout_small = Layout::from_size_align(32, 8).unwrap();

    // Exhaust class 0 (64B blocks) — overflow should kick in.
    let proportion_sum: usize = PROPORTIONS.iter().sum();
    let class0_blocks = 1024 * 1024 * PROPORTIONS[0] / proportion_sum / SIZE_CLASSES[0];

    let mut ptrs = Vec::new();
    // Allocate more than class 0 can hold.
    for _ in 0..class0_blocks + 5 {
        ptrs.push(pool.allocate(layout_small).unwrap());
    }

    assert!(ptrs.len() > class0_blocks, "overflow should provide extra blocks");

    for ptr in ptrs {
        unsafe { pool.deallocate(ptr, layout_small) };
    }
    assert_eq!(pool.used(), Some(0));
}
```

- [ ] **Step 8: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "feat(pool): lock-free allocate/deallocate with side table and overflow"
```

---

## Chunk 5: WorldStats integration and telemetry

### Task 6: Add overflow stats to WorldStats

**Files:**
- Modify: `crates/minkowski/src/world.rs`
- Modify: `crates/minkowski/src/pool.rs` (add trait method or downcast)

- [ ] **Step 1: Add fields to WorldStats**

In `world.rs`, add to `WorldStats`:

```rust
/// Per-class count of blocks currently serving overflow allocations.
/// `None` for system-allocator worlds. Array indices match size classes:
/// [64B, 256B, 1KB, 4KB, 64KB, 1MB].
pub pool_overflow_active: Option<[u64; 6]>,
/// Per-class cumulative count of overflow allocations.
pub pool_overflow_total: Option<[u64; 6]>,
```

- [ ] **Step 2: Add overflow methods to PoolAllocator trait**

In `pool.rs`, add default methods to the `PoolAllocator` trait:

```rust
/// Per-class active overflow count. `None` if not tracked.
fn overflow_active_counts(&self) -> Option<[u64; 6]> {
    None
}

/// Per-class cumulative overflow count. `None` if not tracked.
fn overflow_total_counts(&self) -> Option<[u64; 6]> {
    None
}
```

Implement for `SlabPool`:

```rust
fn overflow_active_counts(&self) -> Option<[u64; 6]> {
    Some(std::array::from_fn(|i| self.overflow_active[i].load(Ordering::Relaxed) as u64))
}

fn overflow_total_counts(&self) -> Option<[u64; 6]> {
    Some(std::array::from_fn(|i| self.overflow_total[i].load(Ordering::Relaxed) as u64))
}
```

- [ ] **Step 3: Wire into World::stats()**

Find the `stats()` method in `world.rs` and add:

```rust
pool_overflow_active: self.pool.overflow_active_counts(),
pool_overflow_total: self.pool.overflow_total_counts(),
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/pool.rs crates/minkowski/src/world.rs
git commit -m "feat(pool): expose overflow telemetry via WorldStats"
```

---

## Chunk 6: Concurrency and loom tests

### Task 7: Std thread concurrency tests

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Update existing concurrent test**

Replace `slab_pool_concurrent_allocate` with a more thorough version:

```rust
#[test]
fn slab_pool_concurrent_allocate_no_duplicates() {
    let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap());
    let layout = Layout::from_size_align(64, 64).unwrap();

    let all_ptrs: Vec<Vec<NonNull<u8>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                s.spawn(move || {
                    let mut ptrs = Vec::new();
                    for _ in 0..200 {
                        if let Ok(ptr) = pool.allocate(layout) {
                            ptrs.push(ptr);
                        }
                    }
                    ptrs
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Verify no duplicates across all threads.
    let mut all: Vec<usize> = all_ptrs
        .iter()
        .flat_map(|v| v.iter().map(|p| p.as_ptr() as usize))
        .collect();
    let total = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), total, "duplicate pointers returned across threads");

    // Deallocate all.
    for ptrs in all_ptrs {
        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout) };
        }
    }
    assert_eq!(pool.used(), Some(0));
}

#[test]
fn slab_pool_concurrent_alloc_dealloc_interleaved() {
    let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap());
    let layout = Layout::from_size_align(64, 64).unwrap();

    std::thread::scope(|s| {
        for _ in 0..4 {
            let pool = Arc::clone(&pool);
            s.spawn(move || {
                for _ in 0..1000 {
                    let ptr = pool.allocate(layout).unwrap();
                    // SAFETY: ptr was just allocated.
                    unsafe { pool.deallocate(ptr, layout) };
                }
            });
        }
    });

    assert_eq!(pool.used(), Some(0));
}
```

- [ ] **Step 2: Run concurrency tests**

Run: `cargo test -p minkowski --lib -- pool::tests::slab_pool_concurrent`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "test(pool): concurrent allocate/deallocate with no-duplicate verification"
```

### Task 8: Loom tests

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Update loom tests for lock-free stack**

Replace the existing `loom_tests` module:

```rust
#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::thread;
    use std::alloc::Layout;

    #[test]
    fn loom_concurrent_pop_no_duplicates() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || p1.allocate(layout).unwrap());

            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(layout).unwrap());

            let ptr1 = t1.join().unwrap();
            let ptr2 = t2.join().unwrap();
            assert_ne!(ptr1.as_ptr(), ptr2.as_ptr(), "two threads got the same block");
        });
    }

    #[test]
    fn loom_concurrent_push_no_lost_blocks() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            // Allocate two blocks.
            let ptr1 = pool.allocate(layout).unwrap();
            let ptr2 = pool.allocate(layout).unwrap();

            // Deallocate concurrently.
            let p1 = pool.clone();
            let t1 = thread::spawn(move || unsafe { p1.deallocate(ptr1, layout) });

            let p2 = pool.clone();
            let t2 = thread::spawn(move || unsafe { p2.deallocate(ptr2, layout) });

            t1.join().unwrap();
            t2.join().unwrap();

            assert_eq!(pool.used(), Some(0), "blocks were lost during concurrent push");
        });
    }

    #[test]
    fn loom_push_pop_concurrent() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            let ptr = pool.allocate(layout).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || unsafe { p1.deallocate(ptr, layout) });

            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(layout));

            t1.join().unwrap();
            let _ = t2.join().unwrap(); // may or may not succeed

            // Final state: either 0 or 1 block allocated.
            let used = pool.used().unwrap();
            assert!(used == 0 || used == SIZE_CLASSES[0]);
        });
    }
}
```

- [ ] **Step 2: Run loom tests**

Run: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --lib --features loom -- loom_tests`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "test(pool): loom tests for lock-free intrusive stack"
```

---

## Chunk 7: Cleanup, PERF annotation, and benchmark validation

### Task 9: Remove old PERF comment and add new one

**Files:**
- Modify: `crates/minkowski/src/pool.rs`

- [ ] **Step 1: Remove the old PERF comment**

Delete the comment above the old SlabPool struct:
```
// PERF: Mutex is a single atomic CAS on uncontended paths.
// If profiling shows contention, consider upgrading to a lock-free
// structure with ABA protection (tagged pointers or epoch-based reclamation).
```

Replace with:
```rust
// PERF: Lock-free intrusive stack via AtomicHead (Atomic<u128> tagged pointer).
// ABA prevention via 64-bit monotonic tag. Side table routes deallocation to
// the correct class regardless of the caller's Layout. Single-step overflow
// from exhausted class to the next larger class.
```

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p minkowski -- -D warnings`
Expected: clean

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: all tests pass

- [ ] **Step 4: Run Miri**

Run: `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- pool::tests`
Expected: PASS — no UB in pointer arithmetic or atomic operations

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/pool.rs
git commit -m "chore(pool): update PERF annotations for lock-free implementation"
```

### Task 10: Benchmark validation

**Files:** None (run only)

- [ ] **Step 1: Run pool benchmarks**

Run: `cargo bench -p minkowski-bench -- pool`
Expected: `simple_insert/pool` < 2.6 ms (target), `add_remove/pool` < 2.6 ms (target)

Record the actual numbers for comparison:
- Before (Mutex): `simple_insert/pool` = 7.54 ms, `add_remove/pool` = 8.24 ms
- After (lock-free): `simple_insert/pool` = ?, `add_remove/pool` = ?

- [ ] **Step 2: Run full benchmark suite to check for regressions**

Run: `cargo bench -p minkowski-bench`
Expected: no regressions on non-pool benchmarks

- [ ] **Step 3: Update perf-shakedown baselines if targets met**

Update the pool rows in `.claude/commands/perf-shakedown.md` with new numbers.

- [ ] **Step 4: Update docs/perf-roadmap.md**

Mark P1-1 (lock-free slab pool allocator) as completed with actual benchmark results.

- [ ] **Step 5: Commit**

```bash
git add .claude/commands/perf-shakedown.md docs/perf-roadmap.md
git commit -m "docs: update pool benchmarks after lock-free migration"
```

---

## Post-Implementation

After all tasks complete:

1. Run `cargo test -p minkowski` (all tests including doc tests)
2. Run `cargo clippy --workspace --all-targets -- -D warnings`
3. Run Miri: `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- pool`
4. Run loom: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --lib --features loom -- loom_tests`
5. Run benchmarks: `cargo bench -p minkowski-bench -- pool`
6. Create PR with benchmark comparison in the description
