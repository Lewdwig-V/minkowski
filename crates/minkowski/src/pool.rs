//! Memory pool allocator trait and implementations.
//!
//! The pool allocator is the single backing allocator for all internal
//! data structures (BlobVec, Arena, entity tables, sparse pages).
//! Two implementations: `SystemAllocator` (current behavior, default)
//! and `SlabPool` (TigerBeetle-style fixed budget with mmap).

use std::alloc::Layout;
use std::fmt;
use std::ptr::NonNull;

#[cfg(loom)]
use crate::sync::Mutex;
use crate::sync::{Arc, AtomicUsize, Ordering};

/// Error returned when the memory pool is exhausted.
#[derive(Debug, Clone)]
pub struct PoolExhausted {
    pub requested: Layout,
}

impl fmt::Display for PoolExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "memory pool exhausted: requested {} bytes (align {})",
            self.requested.size(),
            self.requested.align()
        )
    }
}

impl std::error::Error for PoolExhausted {}

/// Hugepage configuration for the memory pool.
#[derive(Clone, Copy, Debug, Default)]
pub enum HugePages {
    /// Attempt 2 MiB hugepages, fall back to 4 KiB pages silently.
    #[default]
    Try,
    /// Require hugepages — fail if unavailable.
    Require,
    /// Use regular 4 KiB pages only.
    Off,
}

/// Backing allocator for all internal ECS data structures.
///
/// # Safety
///
/// Implementations must return properly aligned, non-overlapping memory
/// regions. `deallocate` must only be called with pointers returned by
/// a prior call to `allocate` with the same `Layout`.
#[allow(dead_code)]
pub unsafe trait PoolAllocator: Send + Sync {
    /// Allocate a block satisfying `layout`.
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted>;

    /// Return a block to the pool.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to `allocate` with
    /// the same `Layout`. The caller must not use `ptr` after this call.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Total capacity in bytes, if bounded. `None` for unbounded allocators.
    fn capacity(&self) -> Option<usize> {
        None
    }

    /// Bytes currently allocated. `None` if not tracked.
    fn used(&self) -> Option<usize> {
        None
    }

    /// Per-class active overflow count. `None` if not tracked.
    fn overflow_active_counts(&self) -> Option<[u64; 6]> {
        None
    }

    /// Per-class cumulative overflow count. `None` if not tracked.
    fn overflow_total_counts(&self) -> Option<[u64; 6]> {
        None
    }
}

/// Default allocator -- delegates to `std::alloc`. Unbounded, panics on OOM.
/// This is the allocator used by `World::new()`.
pub(crate) struct SystemAllocator;

// SAFETY: SystemAllocator delegates to the global allocator which returns
// properly aligned, non-overlapping memory regions. Zero-size allocations
// return dangling aligned pointers (never passed to `std::alloc::dealloc`).
unsafe impl PoolAllocator for SystemAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            // Zero-size: return dangling aligned pointer.
            // `layout.align()` is guaranteed non-zero by Layout invariants.
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }
        // SAFETY: layout has non-zero size (checked above).
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).ok_or_else(|| {
            // Preserve current behavior: panic on OOM for system allocator.
            std::alloc::handle_alloc_error(layout)
        })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // SAFETY: caller guarantees `ptr` was returned by a prior `allocate`
        // with the same layout, and `layout.size() > 0`.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

/// Shared handle to a pool allocator. Cheaply cloneable.
///
/// Under `cfg(loom)`, uses `Arc<Box<dyn PoolAllocator>>` because loom's `Arc`
/// does not support direct trait-object coercion.
#[allow(dead_code)]
#[cfg(not(loom))]
pub(crate) type SharedPool = Arc<dyn PoolAllocator>;

#[allow(dead_code)]
#[cfg(loom)]
pub(crate) type SharedPool = Arc<Box<dyn PoolAllocator>>;

/// Create the default (system) allocator as a shared pool.
#[allow(dead_code)]
pub(crate) fn default_pool() -> SharedPool {
    into_shared(SystemAllocator)
}

/// Wrap a concrete allocator into a `SharedPool`.
///
/// This helper exists because `loom::sync::Arc` does not support automatic
/// trait-object coercion from `Arc<T>` to `Arc<dyn PoolAllocator>`.
/// On `std`, this is a simple `Arc::new(alloc)`.
#[cfg(not(loom))]
pub(crate) fn into_shared<A: PoolAllocator + 'static>(alloc: A) -> SharedPool {
    Arc::new(alloc)
}

#[cfg(loom)]
pub(crate) fn into_shared<A: PoolAllocator + 'static>(alloc: A) -> SharedPool {
    // loom::sync::Arc doesn't coerce to dyn. Box first, then Arc the Box.
    let boxed: Box<dyn PoolAllocator> = Box::new(alloc);
    Arc::from(boxed)
}

// ── mlockall ────────────────────────────────────────────────────────

/// Attempt `mlockall(MCL_CURRENT | MCL_FUTURE)` to lock all current and future
/// memory mappings into physical RAM. This is a process-global operation.
///
/// Returns `true` if the call succeeded, `false` if it failed (insufficient
/// privileges, unsupported platform, etc.).
#[cfg(all(unix, not(miri)))]
pub(crate) fn try_mlockall() -> bool {
    // SAFETY: mlockall is a process-global operation with no memory
    // unsafety. It may fail due to insufficient RLIMIT_MEMLOCK.
    unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) == 0 }
}

#[cfg(any(not(unix), miri))]
pub(crate) fn try_mlockall() -> bool {
    false
}

// ── MmapRegion ──────────────────────────────────────────────────────

/// Memory region backed by an anonymous mmap with optional hugepage support.
///
/// Uses `memmap2` for cross-platform mmap. Pages are demand-paged by the OS;
/// the first access to each page triggers a soft fault.
struct MmapRegion {
    mmap: memmap2::MmapMut,
    #[allow(dead_code)]
    huge: bool,
}

impl MmapRegion {
    /// Create the mmap region with full pre-fault chain.
    ///
    /// Miri only supports `mmap` with `MAP_PRIVATE|MAP_ANONYMOUS` — no
    /// `MAP_POPULATE`, `MAP_HUGETLB`, `mlock`, or `write_volatile` page
    /// touch. Under `cfg(miri)` we use plain `map_anon()` and skip
    /// pre-faulting entirely (there's no real VM subsystem to fault into).
    #[cfg(not(miri))]
    fn new(size: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        use memmap2::MmapOptions;
        let layout = Layout::from_size_align(size, 1).expect("valid layout");

        // Try hugepages first if requested.
        if matches!(hugepages, HugePages::Try | HugePages::Require) {
            let result = MmapOptions::new()
                .len(size)
                .populate() // MAP_POPULATE — pre-fault with hugepages
                .huge(Some(21)) // 2 MiB hugepages (2^21)
                .map_anon();

            match result {
                Ok(mmap) => return Ok(Self { mmap, huge: true }),
                Err(_) if matches!(hugepages, HugePages::Try) => {
                    // Fall through to regular pages.
                }
                Err(_) => return Err(PoolExhausted { requested: layout }),
            }
        }

        // Regular pages — pre-fault via fallback chain.
        // Pre-faulting is NOT optional: we must know at startup whether
        // the system can back the mapping with physical RAM. Three paths,
        // tried in order of preference:
        //
        // 1. MAP_POPULATE — kernel pre-faults pages at mmap time.
        //    Fastest on native Linux, but can hang/fail on WSL2 or
        //    older kernels.
        //
        // 2. mlock — forces pages into physical RAM after mmap.
        //    Works on most systems but requires sufficient RLIMIT_MEMLOCK.
        //
        // 3. Manual touch sweep — write one byte per page (4 KB stride).
        //    Always works, no privilege requirements. Slowest but guaranteed.

        // Try MAP_POPULATE first. Track whether it succeeded so we can
        // skip redundant mlock/touch when it did.
        let (mut mmap, populated) = MmapOptions::new()
            .len(size)
            .populate()
            .map_anon()
            .map(|m| (m, true))
            .or_else(|_| MmapOptions::new().len(size).map_anon().map(|m| (m, false)))
            .map_err(|_| PoolExhausted { requested: layout })?;

        // If MAP_POPULATE succeeded, pages are already faulted. If it
        // didn't (we fell through to the plain mmap), we need to force
        // pages into RAM via mlock or manual touch.
        if !populated && mmap.lock().is_err() {
            // mlock failed (insufficient RLIMIT_MEMLOCK or platform
            // doesn't support it) — manual touch sweep as final fallback.
            Self::prefault_manual(&mut mmap);
        }

        Ok(Self { mmap, huge: false })
    }

    /// Miri-compatible path: plain `MAP_PRIVATE|MAP_ANONYMOUS` only.
    #[cfg(miri)]
    fn new(size: usize, _hugepages: HugePages) -> Result<Self, PoolExhausted> {
        use memmap2::MmapOptions;
        let layout = Layout::from_size_align(size, 1).expect("valid layout");
        let mmap = MmapOptions::new()
            .len(size)
            .map_anon()
            .map_err(|_| PoolExhausted { requested: layout })?;
        Ok(Self { mmap, huge: false })
    }

    /// Write one byte per page to force the kernel to back the mapping
    /// with physical memory. This is the last-resort pre-fault path.
    #[cfg(not(miri))]
    fn prefault_manual(mmap: &mut memmap2::MmapMut) {
        const PAGE_SIZE: usize = 4096;
        let ptr = mmap.as_mut_ptr();
        let len = mmap.len();
        for offset in (0..len).step_by(PAGE_SIZE) {
            // Safety: offset < len, and len == mmap size.
            unsafe { ptr.add(offset).write_volatile(0) };
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.mmap.len()
    }
}

// ── Lock-free intrusive stack ────────────────────────────────────────

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

// ── SlabPool ────────────────────────────────────────────────────────

const NUM_SIZE_CLASSES: usize = 6;
const SIZE_CLASSES: [usize; NUM_SIZE_CLASSES] = [64, 256, 1024, 4096, 65_536, 1_048_576];

/// Proportional split tuned for ECS patterns (percentages of total budget):
/// - Class 0 (64 B):   1% — small components
/// - Class 1 (256 B):  2% — medium components
/// - Class 2 (1 KB):   4% — sparse pages, small columns
/// - Class 3 (4 KB):  20% — OS page, column growth
/// - Class 4 (64 KB): 40% — large column segments
/// - Class 5 (1 MB):  33% — bulk column pre-allocation
const PROPORTIONS: [usize; NUM_SIZE_CLASSES] = [1, 2, 4, 20, 40, 33];

/// Sentinel value for unallocated side table entries.
const SIDE_TABLE_UNALLOCATED: u8 = 0xFF;

/// Mask for the class index in a side table entry (bits 0..3).
const SIDE_TABLE_CLASS_MASK: u8 = 0x0F;

/// Bit flag indicating this allocation overflowed from a smaller class.
const SIDE_TABLE_OVERFLOW_BIT: u8 = 0x80;

/// Return the size-class index for a given layout, or `None` if too large.
fn size_class_for(layout: Layout) -> Option<usize> {
    // The effective allocation size must satisfy both size and alignment.
    let size = layout.size().max(layout.align());
    SIZE_CLASSES.iter().position(|&s| s >= size)
}

/// TigerBeetle-style slab allocator backed by a single mmap'd region.
///
/// Memory is partitioned into fixed-size blocks across six size classes.
/// Allocation and deallocation use a lock-free intrusive stack per class
/// via `AtomicHead` (128-bit tagged pointer CAS).
///
// PERF: Lock-free intrusive stack via AtomicHead (Atomic<u128> tagged pointer).
// ABA prevention via 64-bit monotonic tag. Side table routes deallocation to
// the correct class regardless of the caller's Layout. Single-step overflow
// from exhausted class to the next larger class.
pub(crate) struct SlabPool {
    /// Keeps the mmap alive. Never re-borrowed after construction —
    /// use `base` and `total` instead to avoid Tree Borrows invalidation.
    _region: MmapRegion,
    /// Cached from `region.as_mut_ptr()` at construction. Using
    /// `region.as_ptr()` later would create a shared reborrow that
    /// freezes the tag, making all writes through mutable pointers UB
    /// under Tree Borrows.
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

impl Drop for SlabPool {
    fn drop(&mut self) {
        if self.side_table_len > 0 {
            // SAFETY: side_table was allocated via Vec in new(), then
            // ownership was transferred to the raw pointer via mem::forget.
            unsafe {
                let _ =
                    Vec::from_raw_parts(self.side_table, self.side_table_len, self.side_table_len);
            }
        }
    }
}

// SAFETY: Pointers into the mmap region are only accessed through atomic CAS
// operations (or Mutex under loom). The mmap region is owned by SlabPool and
// outlives all allocations. The side table raw pointer is written only by
// the thread that owns the block (after successful CAS) and read before
// the CAS that returns it — single-owner access, no concurrent mutation.
unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    /// Create a new slab pool backed by an mmap region of `budget` bytes.
    pub(crate) fn new(budget: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        let mut region = MmapRegion::new(budget, hugepages)?;

        // Cache base pointer and length now. After this point we never
        // re-borrow the MmapMut — calling `as_ptr()` would create a
        // shared reborrow that freezes all outstanding mutable pointers
        // under Tree Borrows.
        let base = region.as_mut_ptr();
        let total = region.len();

        // Allocate side table as raw pointer (see struct doc for rationale).
        let side_table_len = total / SIZE_CLASSES[0];
        let mut st_vec = vec![SIDE_TABLE_UNALLOCATED; side_table_len];
        let side_table = st_vec.as_mut_ptr();
        std::mem::forget(st_vec); // ownership transferred to raw pointer

        let proportion_sum: usize = PROPORTIONS.iter().sum();

        // Build intrusive linked lists and heads in a single pass.
        let mut offset: usize = 0;
        let mut head_values: [TaggedPtr; NUM_SIZE_CLASSES] = [TaggedPtr::empty(); NUM_SIZE_CLASSES];

        for class in 0..NUM_SIZE_CLASSES {
            let block_size = SIZE_CLASSES[class];

            // Align the absolute address (base + offset) to block_size, not just offset.
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
                // for total bytes. Block is >= 64 bytes (MIN_BLOCK_SIZE invariant),
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
}

// SAFETY: SlabPool returns properly aligned, non-overlapping memory regions.
// Each block is a fixed-size slab from the mmap region. Blocks within each
// size class are aligned to their class size. Size classes are chosen such
// that `layout.size() <= block_size` and `layout.align() <= block_size`.
// Lock-free CAS on AtomicHead serializes access to each free list.
unsafe impl PoolAllocator for SlabPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }

        let class = size_class_for(layout).ok_or(PoolExhausted { requested: layout })?;

        // Try the target class, then overflow to the next larger class.
        // Indexing multiple arrays (heads, overflow_active, overflow_total,
        // SIZE_CLASSES) by try_class — iterator+enumerate would be less clear.
        #[allow(clippy::needless_range_loop)]
        for try_class in class..NUM_SIZE_CLASSES.min(class + 2) {
            loop {
                let head = load_head(&self.heads[try_class]);
                if head.is_empty() {
                    break; // This class is exhausted, try next.
                }

                // SAFETY: head.ptr() is a valid block pointer within the mmap
                // region. The block is >= 64 bytes and aligned to its class size,
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
                    let entry =
                        try_class as u8 | if overflow { SIDE_TABLE_OVERFLOW_BIT } else { 0 };
                    unsafe { self.side_table.add(index).write(entry) };

                    if overflow {
                        self.overflow_active[try_class].fetch_add(1, Ordering::Relaxed);
                        self.overflow_total[try_class].fetch_add(1, Ordering::Relaxed);
                    }

                    debug_assert!(
                        (head.ptr() as usize).is_multiple_of(layout.align()),
                        "SlabPool: block at {:p} is not aligned to {}",
                        head.ptr(),
                        layout.align()
                    );

                    return Ok(NonNull::new(head.ptr()).expect("free list block is non-null"));
                }
                // CAS failed — another thread popped. Retry.
            }
        }

        Err(PoolExhausted { requested: layout })
    }

    /// Return a block to the pool.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to `allocate` on this
    /// pool. The caller must not use `ptr` after this call. The `layout`
    /// parameter is accepted for API compatibility but is not used for
    /// class routing — the side table is authoritative.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        debug_assert!(
            ptr.as_ptr() >= self.base && (ptr.as_ptr() as usize) < self.base as usize + self.total,
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

    fn capacity(&self) -> Option<usize> {
        Some(self.total)
    }

    fn used(&self) -> Option<usize> {
        Some(self.used_bytes.load(Ordering::Relaxed))
    }

    fn overflow_active_counts(&self) -> Option<[u64; 6]> {
        Some(std::array::from_fn(|i| {
            self.overflow_active[i].load(Ordering::Relaxed) as u64
        }))
    }

    fn overflow_total_counts(&self) -> Option<[u64; 6]> {
        Some(std::array::from_fn(|i| {
            self.overflow_total[i].load(Ordering::Relaxed) as u64
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_allocator_allocate_and_deallocate() {
        let alloc = SystemAllocator;
        let layout = Layout::from_size_align(256, 64).unwrap();
        let ptr = alloc.allocate(layout).unwrap();
        // SAFETY: ptr points to 256 bytes of allocated memory.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0xAB, 256) };
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn system_allocator_zero_size() {
        let alloc = SystemAllocator;
        let layout = Layout::from_size_align(0, 1).unwrap();
        let ptr = alloc.allocate(layout).unwrap();
        // SAFETY: ptr was returned by `allocate` with this layout (zero-size).
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn system_allocator_capacity_is_none() {
        let alloc = SystemAllocator;
        assert!(alloc.capacity().is_none());
        assert!(alloc.used().is_none());
    }

    #[test]
    fn shared_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedPool>();
    }

    #[test]
    fn pool_exhausted_display() {
        let err = PoolExhausted {
            requested: Layout::from_size_align(1024, 64).unwrap(),
        };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("64"));
    }

    // ── Side table tests ─────────────────────────────────────────────

    #[test]
    fn side_table_index_computation() {
        let base = 0x1000_usize as *mut u8;
        let min_block = SIZE_CLASSES[0]; // 64
        // Block at base+0 -> index 0
        assert_eq!((0x1000 - base as usize) / min_block, 0);
        // Block at base+64 -> index 1
        assert_eq!((0x1040 - base as usize) / min_block, 1);
        // Block at base+128 -> index 2
        assert_eq!((0x1080 - base as usize) / min_block, 2);
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

    // ── TaggedPtr tests ──────────────────────────────────────────────

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

    // ── SlabPool tests ──────────────────────────────────────────────

    #[test]
    fn slab_pool_allocate_and_deallocate() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!(pool.used().unwrap() > 0);
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_returns_error_on_exhaustion() {
        let pool = SlabPool::new(64 * 1024, HugePages::Off).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.allocate(layout) {
            ptrs.push(ptr);
        }
        assert!(!ptrs.is_empty());

        // Deallocate one and re-allocate succeeds.
        // SAFETY: ptrs were returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptrs.pop().unwrap(), layout) };
        assert!(pool.allocate(layout).is_ok());
    }

    #[test]
    fn slab_pool_observability() {
        let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
        assert_eq!(pool.capacity(), Some(1024 * 1024));
        assert_eq!(pool.used(), Some(0));

        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!(pool.used().unwrap() > 0);
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_concurrent_allocate() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap());
        let layout = Layout::from_size_align(64, 64).unwrap();

        std::thread::scope(|s| {
            for _ in 0..4 {
                let pool = Arc::clone(&pool);
                s.spawn(move || {
                    let mut ptrs = Vec::new();
                    for _ in 0..100 {
                        if let Ok(ptr) = pool.allocate(layout) {
                            ptrs.push(ptr);
                        }
                    }
                    for ptr in ptrs {
                        // SAFETY: each ptr was returned by `allocate` and is
                        // only deallocated once by this thread.
                        unsafe { pool.deallocate(ptr, layout) };
                    }
                });
            }
        });
    }

    #[test]
    fn slab_pool_multiple_size_classes() {
        let pool = SlabPool::new(8 * 1024 * 1024, HugePages::Off).unwrap();

        let small = Layout::from_size_align(32, 8).unwrap();
        let medium = Layout::from_size_align(200, 8).unwrap();
        let large = Layout::from_size_align(4000, 8).unwrap();

        let p1 = pool.allocate(small).unwrap();
        let p2 = pool.allocate(medium).unwrap();
        let p3 = pool.allocate(large).unwrap();

        // All should be non-null and different.
        assert_ne!(p1.as_ptr(), p2.as_ptr());
        assert_ne!(p2.as_ptr(), p3.as_ptr());

        // SAFETY: each ptr was returned by `allocate` with the corresponding layout.
        unsafe {
            pool.deallocate(p1, small);
            pool.deallocate(p2, medium);
            pool.deallocate(p3, large);
        }
    }

    #[test]
    fn slab_pool_used_returns_to_zero_after_dealloc() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
        assert_eq!(pool.used(), Some(0));

        let layout = Layout::from_size_align(64, 64).unwrap();
        let p1 = pool.allocate(layout).unwrap();
        let p2 = pool.allocate(layout).unwrap();
        assert!(pool.used().unwrap() > 0);

        // SAFETY: ptrs were returned by `allocate` with this layout.
        unsafe {
            pool.deallocate(p1, layout);
            pool.deallocate(p2, layout);
        }
        assert_eq!(pool.used(), Some(0));
    }

    #[test]
    fn slab_pool_zero_size_allocation() {
        let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
        let layout = Layout::from_size_align(0, 1).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert_eq!(pool.used(), Some(0), "ZST should not consume pool bytes");
        // SAFETY: ptr was returned by `allocate` with this layout (zero-size).
        unsafe { pool.deallocate(ptr, layout) };
        assert_eq!(pool.used(), Some(0));
    }

    #[test]
    fn slab_pool_oversized_returns_error() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
        // 2 MB exceeds the largest size class (1 MB).
        let layout = Layout::from_size_align(2 * 1024 * 1024, 8).unwrap();
        assert!(pool.allocate(layout).is_err());
    }

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

        assert!(
            ptrs.len() > class0_blocks,
            "overflow should provide extra blocks"
        );

        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout_small) };
        }
        assert_eq!(pool.used(), Some(0));
    }

    #[test]
    fn slab_pool_overflow_to_next_class() {
        // Small pool so class 0 exhausts quickly.
        let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
        let layout_small = Layout::from_size_align(32, 8).unwrap();

        // Exhaust all classes by allocating until failure.
        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.allocate(layout_small) {
            ptrs.push(ptr);
        }

        // With overflow, we should get MORE blocks than class 0 alone:
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
        assert_eq!(
            freed, SIZE_CLASSES[1],
            "overflow block should free class 1 size"
        );

        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout_small) };
        }
    }
}

// ── Loom tests ─────────────────────────────────────────────────────

#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::thread;
    use std::alloc::Layout;

    /// Two threads concurrently allocate and deallocate from the same pool.
    /// Verifies: (1) no double-allocation (all pointers unique), (2) `used_bytes`
    /// returns to zero after all deallocations.
    #[test]
    fn loom_concurrent_allocate_deallocate() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || {
                let ptr = p1.allocate(layout).unwrap();
                // SAFETY: ptr from allocate, deallocated once.
                unsafe { p1.deallocate(ptr, layout) };
            });

            let p2 = pool.clone();
            let t2 = thread::spawn(move || {
                let ptr = p2.allocate(layout).unwrap();
                // SAFETY: ptr from allocate, deallocated once.
                unsafe { p2.deallocate(ptr, layout) };
            });

            t1.join().unwrap();
            t2.join().unwrap();

            assert_eq!(pool.used(), Some(0));
        });
    }

    /// Two threads allocate from the same size class. Verifies no duplicate
    /// pointers are returned (the Mutex serializes access correctly).
    #[test]
    fn loom_no_duplicate_allocations() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || p1.allocate(layout).unwrap());

            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(layout).unwrap());

            let ptr1 = t1.join().unwrap();
            let ptr2 = t2.join().unwrap();

            assert_ne!(
                ptr1.as_ptr(),
                ptr2.as_ptr(),
                "two threads got the same block"
            );
        });
    }
}
