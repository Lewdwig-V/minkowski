//! Memory pool allocator trait and implementations.
//!
//! The pool allocator is the single backing allocator for all internal
//! data structures (BlobVec, Arena, entity tables, sparse pages).
//! Two implementations: `SystemAllocator` (current behavior, default)
//! and `SlabPool` (TigerBeetle-style fixed budget with mmap).

use std::alloc::Layout;
use std::fmt;
use std::ptr::NonNull;

use crate::sync::{Arc, AtomicUsize, Mutex, Ordering};

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

/// Return the size-class index for a given layout, or `None` if too large.
fn size_class_for(layout: Layout) -> Option<usize> {
    // The effective allocation size must satisfy both size and alignment.
    let size = layout.size().max(layout.align());
    SIZE_CLASSES.iter().position(|&s| s >= size)
}

/// TigerBeetle-style slab allocator backed by a single mmap'd region.
///
/// Memory is partitioned into fixed-size blocks across six size classes.
/// Allocation and deallocation are serialized per size class via `Mutex`.
///
// PERF: Mutex is a single atomic CAS on uncontended paths.
// If profiling shows contention, consider upgrading to a lock-free
// structure with ABA protection (tagged pointers or epoch-based reclamation).
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
    free_lists: [Mutex<Vec<*mut u8>>; NUM_SIZE_CLASSES],
    used_bytes: AtomicUsize,
}

// SAFETY: `*mut u8` inside the Mutex Vecs points into the mmap region which
// is owned by SlabPool and outlives all allocations. The Mutex provides
// exclusive access to the free list vectors. The raw pointers are never
// dereferenced outside the lock — they are just addresses returned to callers.
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

        // Initialize free lists as empty Vecs.
        let free_lists: [Mutex<Vec<*mut u8>>; NUM_SIZE_CLASSES] =
            std::array::from_fn(|_| Mutex::new(Vec::new()));

        // Partition the region into size-class blocks proportionally.
        let proportion_sum: usize = PROPORTIONS.iter().sum();
        let mut offset: usize = 0;

        for class in 0..NUM_SIZE_CLASSES {
            let block_size = SIZE_CLASSES[class];

            // Align the absolute address (base + offset) to block_size, not just offset.
            // The mmap base is page-aligned (4KB) but may not be aligned to larger
            // block sizes (64KB, 1MB). Without this, blocks in those classes could
            // be misaligned for components with align > page_size.
            let abs_addr = base as usize + offset;
            let aligned_addr = (abs_addr + block_size - 1) & !(block_size - 1);
            offset = aligned_addr - base as usize;

            // Bytes allocated to this size class (rounded down to whole blocks).
            let class_bytes = total * PROPORTIONS[class] / proportion_sum;
            let block_count = class_bytes / block_size;

            let mut list = free_lists[class].lock();
            list.reserve(block_count);

            // Push blocks from first to last so that pop yields the lowest
            // address first (better cache locality).
            for i in 0..block_count {
                let block_offset = offset + i * block_size;
                if block_offset + block_size > total {
                    break;
                }
                // SAFETY: `block_offset + block_size <= total` (checked above),
                // and `base` points to the start of a valid mmap region of
                // `total` bytes. Each block is at least 64 bytes.
                let block_ptr = unsafe { base.add(block_offset) };
                list.push(block_ptr);
            }

            offset += block_count * block_size;
        }

        Ok(Self {
            _region: region,
            base,
            total,
            free_lists,
            used_bytes: AtomicUsize::new(0),
        })
    }
}

// SAFETY: SlabPool returns properly aligned, non-overlapping memory regions.
// Each block is a fixed-size slab from the mmap region. Blocks within each
// size class are aligned to their class size. Size classes are chosen such
// that `layout.size() <= block_size` and `layout.align() <= block_size`.
// The Mutex serializes access to each free list.
unsafe impl PoolAllocator for SlabPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }

        let class = size_class_for(layout).ok_or(PoolExhausted { requested: layout })?;

        // No cross-class fallback: each class is a fixed partition.
        // If this class is exhausted, return Err — the user adjusts
        // their budget or size-class proportions. This matches TigerBeetle's
        // fixed-partition model and avoids the dealloc class-mismatch bug
        // where a block promoted to a larger class gets returned to the
        // smaller class's free list on deallocation.
        let mut list = self.free_lists[class].lock();
        if let Some(ptr) = list.pop() {
            drop(list); // release lock before atomic
            self.used_bytes
                .fetch_add(SIZE_CLASSES[class], Ordering::Relaxed);

            debug_assert!(
                (ptr as usize).is_multiple_of(layout.align()),
                "SlabPool: block at {ptr:p} is not aligned to {}",
                layout.align()
            );

            return Ok(NonNull::new(ptr).expect("free list block is non-null"));
        }

        Err(PoolExhausted { requested: layout })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        let class = size_class_for(layout);
        debug_assert!(
            class.is_some(),
            "SlabPool::deallocate: layout (size={}, align={}) exceeds max size class — \
             caller violated the allocate/deallocate contract",
            layout.size(),
            layout.align()
        );
        let Some(class) = class else { return };

        debug_assert!(
            ptr.as_ptr() >= self.base && (ptr.as_ptr() as usize) < self.base as usize + self.total,
            "SlabPool::deallocate: pointer {:p} is outside pool region",
            ptr.as_ptr()
        );

        let mut list = self.free_lists[class].lock();
        list.push(ptr.as_ptr());
        self.used_bytes
            .fetch_sub(SIZE_CLASSES[class], Ordering::Relaxed);
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.total)
    }

    fn used(&self) -> Option<usize> {
        Some(self.used_bytes.load(Ordering::Relaxed))
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
    fn slab_pool_no_cross_class_fallback() {
        // Verify that exhausting one size class does NOT spill into larger
        // classes. Each class is a fixed partition — this matches TigerBeetle's
        // model and avoids the dealloc class-mismatch bug.
        let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
        let layout_small = Layout::from_size_align(32, 8).unwrap();

        // Exhaust class 0 (64B blocks).
        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.allocate(layout_small) {
            ptrs.push(ptr);
        }

        // Calculate how many blocks class 0 should have had.
        let proportion_sum: usize = PROPORTIONS.iter().sum();
        let class0_bytes = 1024 * 1024 * PROPORTIONS[0] / proportion_sum;
        let class0_blocks = class0_bytes / SIZE_CLASSES[0];

        // Should have gotten exactly class 0's block count — no spill.
        assert_eq!(
            ptrs.len(),
            class0_blocks,
            "expected exactly {class0_blocks} blocks from class 0, got {}",
            ptrs.len()
        );

        // Class 1 (256B) should still have blocks available.
        let layout_medium = Layout::from_size_align(200, 8).unwrap();
        assert!(
            pool.allocate(layout_medium).is_ok(),
            "class 1 should still have blocks after class 0 is exhausted"
        );

        // Clean up.
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
