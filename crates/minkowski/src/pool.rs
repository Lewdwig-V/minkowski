//! Memory pool allocator: `SlabPool`, the single backing allocator for all
//! internal data structures.
//!
//! The pool allocator is the single backing allocator for all internal
//! data structures (BlobVec, Arena, entity tables, sparse pages).
//! All allocations go through `SlabPool` — a TigerBeetle-style
//! mmap-backed allocator with fixed budget and lock-free slab classes.

use std::alloc::Layout;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::AtomicU64 as StdAtomicU64;
use std::sync::atomic::AtomicU64;

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
/// Default pool budget: 256 MiB. Virtual address space is cheap on 64-bit;
/// physical pages are demand-paged (no pre-faulting for the default pool).
pub(crate) const DEFAULT_POOL_BUDGET: usize = 256 * 1024 * 1024;

/// Try to create the default mmap-backed pool (256 MiB, demand-paged).
///
/// Returns `Err(PoolExhausted)` if the mmap allocation fails (e.g., restricted
/// container environment).
pub(crate) fn try_default_pool(hugepages: HugePages) -> Result<Arc<SlabPool>, PoolExhausted> {
    SlabPool::new(DEFAULT_POOL_BUDGET, hugepages, false).map(Arc::new)
}

/// Create the default mmap-backed pool (256 MiB, demand-paged, no hugepages).
///
/// Panics if the mmap allocation fails. Prefer [`try_default_pool`] when the
/// caller can handle the error (e.g., `WorldBuilder::build()`).
pub(crate) fn default_pool() -> Arc<SlabPool> {
    try_default_pool(HugePages::default())
        .expect("failed to allocate default 256 MiB memory pool via mmap")
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
}

impl MmapRegion {
    /// Create the mmap region.
    ///
    /// When `prefault` is true, pages are pre-faulted into physical RAM
    /// via a three-step fallback chain (MAP_POPULATE → mlock → manual
    /// touch). When false, pages are demand-paged by the OS — the first
    /// access to each page triggers a soft fault.
    ///
    /// Miri only supports `mmap` with `MAP_PRIVATE|MAP_ANONYMOUS` — no
    /// `MAP_POPULATE`, `MAP_HUGETLB`, `mlock`, or `write_volatile` page
    /// touch. Under `cfg(miri)` we use plain `map_anon()` and skip
    /// pre-faulting entirely (there's no real VM subsystem to fault into).
    #[cfg(not(miri))]
    fn new(size: usize, hugepages: HugePages, prefault: bool) -> Result<Self, PoolExhausted> {
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
                Ok(mmap) => return Ok(Self { mmap }),
                Err(_) if matches!(hugepages, HugePages::Try) => {
                    // Fall through to regular pages.
                }
                Err(_) => return Err(PoolExhausted { requested: layout }),
            }
        }

        if !prefault {
            // Lazy mode: demand-paged, no pre-faulting. The OS backs
            // pages with physical RAM on first access (soft fault).
            let mmap = MmapOptions::new()
                .len(size)
                .map_anon()
                .map_err(|_| PoolExhausted { requested: layout })?;
            return Ok(Self { mmap });
        }

        // Regular pages — pre-fault via fallback chain.
        // Pre-faulting is NOT optional for bounded pools: we must know
        // at startup whether the system can back the mapping with
        // physical RAM. Three paths, tried in order of preference:
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

        Ok(Self { mmap })
    }

    /// Miri-compatible path: plain `MAP_PRIVATE|MAP_ANONYMOUS` only.
    #[cfg(miri)]
    fn new(size: usize, _hugepages: HugePages, _prefault: bool) -> Result<Self, PoolExhausted> {
        use memmap2::MmapOptions;
        let layout = Layout::from_size_align(size, 1).expect("valid layout");
        let mmap = MmapOptions::new()
            .len(size)
            .map_anon()
            .map_err(|_| PoolExhausted { requested: layout })?;
        Ok(Self { mmap })
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

/// Block size per class. Classes 0..=4 are slab classes (lock-free free
/// lists, carved in `CHUNK_SIZE` chunks from the bottom frontier). Class 5
/// (`LARGE_CLASS`) is not a slab class: it is the large-run path, and its
/// entry here is the large-run unit size (`LARGE_UNIT`).
const SIZE_CLASSES: [usize; NUM_SIZE_CLASSES] = [64, 256, 1024, 4096, 65_536, LARGE_UNIT];

/// Number of slab classes (served from per-class free lists + TCache).
const NUM_SLAB_CLASSES: usize = 5;

/// Class index of the large-run path. Requests whose effective size
/// (`max(size, align)`) exceeds the largest slab class are served as
/// contiguous runs of `LARGE_UNIT`s from the top frontier.
const LARGE_CLASS: usize = NUM_SLAB_CLASSES;

/// Granularity of the bottom frontier. Each carve hands one chunk to one
/// slab class. 64 KiB = the largest slab block, so every chunk is aligned to
/// every slab block size it can be split into.
const CHUNK_SIZE: usize = 65_536;

/// Granularity (and alignment) of the top frontier. Large runs are whole
/// multiples of this.
const LARGE_UNIT: usize = 1_048_576;

/// Alignment of the pool window inside the mmap. The mmap is over-sized by
/// this much so a `LARGE_UNIT`-aligned window of exactly `budget` bytes
/// always fits, whatever address the OS returns.
const WINDOW_ALIGN: usize = LARGE_UNIT;

/// Largest accepted pool budget. Two limits, both enforced in
/// `SlabPoolInner::new` with `PoolExhausted`:
/// - the mmap is `budget + WINDOW_ALIGN` bytes, which must stay a valid
///   `Layout` size (`<= isize::MAX`);
/// - run lengths are stored as `u32` in `Frontier::units`, where
///   `UNIT_INTERIOR == u32::MAX` is a sentinel, so `n_units < u32::MAX`.
///
/// The second also bounds the frontier lock hold time: `take_run`,
/// `grow_run` and `release_run` scan at most `n_units = budget / LARGE_UNIT`
/// entries (256 for the default pool, 65 536 for a 64 GiB pool).
const MAX_POOL_BUDGET: usize = {
    let by_layout = isize::MAX as usize - WINDOW_ALIGN - HUGE_PAGE_SIZE;
    let by_units = (u32::MAX as usize - 1).saturating_mul(LARGE_UNIT);
    if by_layout < by_units {
        by_layout
    } else {
        by_units
    }
};

/// 2 MiB hugepage size. Hugepage mappings are sized to a multiple of this:
/// the kernel rounds a `MAP_HUGETLB` length up, but `munmap` with the
/// unrounded length (what `memmap2` passes on drop) can fail with EINVAL.
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// `Frontier::units` entry for a free unit.
const UNIT_FREE: u32 = 0;

/// `Frontier::units` entry for a unit inside (not at the start of) a run.
const UNIT_INTERIOR: u32 = u32::MAX;

/// Sentinel value for unallocated side table entries. Allocated entries
/// hold the class index (`0..NUM_SIZE_CLASSES`).
const SIDE_TABLE_UNALLOCATED: u8 = 0xFF;

/// Return the class index for a given layout, or `None` if it cannot be
/// served (alignment above `LARGE_UNIT`).
fn size_class_for(layout: Layout) -> Option<usize> {
    if layout.align() > LARGE_UNIT {
        return None;
    }
    // The effective allocation size must satisfy both size and alignment.
    let size = layout.size().max(layout.align());
    Some(
        SIZE_CLASSES[..NUM_SLAB_CLASSES]
            .iter()
            .position(|&s| s >= size)
            .unwrap_or(LARGE_CLASS),
    )
}

/// Number of `LARGE_UNIT`s a large request occupies.
fn large_units_for(layout: Layout) -> usize {
    layout.size().max(layout.align()).div_ceil(LARGE_UNIT)
}

/// The two opposing frontiers of the pool window.
///
/// ```text
///  window start                                              window end
///  │ slab chunks (64 KiB, one class each) →     ← large runs (1 MiB units) │
///  ├────────────────────────────────────┬───────┬─────────────────────────┤
///  0                                    lo  free hi·LARGE_UNIT       n_units·LARGE_UNIT
/// ```
///
/// Invariants (all hold whenever the lock is released):
/// - `lo` is a multiple of `CHUNK_SIZE`; `[0, lo)` is carved into slab
///   chunks and never returns to the frontier.
/// - `hi <= units.len()`; units `[hi, units.len())` form the top region.
///   `lo <= hi * LARGE_UNIT` whenever `hi < units.len()`.
/// - `hi == units.len()` or `units[hi] != UNIT_FREE`: the lowest unit of
///   the top region is always in use, so freed runs at the edge give their
///   space back to the bottom frontier immediately.
/// - A run of `k` units starting at `u` has `units[u] == k` and
///   `units[u+1..u+k] == UNIT_INTERIOR`; all other top-region units are
///   `UNIT_FREE`.
///
/// Cost: every method is O(`units.len()`) worst case, bounded by
/// `MAX_POOL_BUDGET / LARGE_UNIT`; `carve_chunk` is O(1). All run under the
/// frontier lock, which the slab free-list fast path never takes.
struct Frontier {
    lo: usize,
    hi: usize,
    units: Box<[u32]>,
}

impl Frontier {
    /// Upper bound (exclusive, window offset) for the bottom frontier.
    fn bottom_limit(&self, total: usize) -> usize {
        if self.hi == self.units.len() {
            total - total % CHUNK_SIZE
        } else {
            self.hi * LARGE_UNIT
        }
    }

    /// Carve one chunk off the bottom frontier. Returns its window offset.
    fn carve_chunk(&mut self, total: usize) -> Option<usize> {
        let start = self.lo;
        if start + CHUNK_SIZE > self.bottom_limit(total) {
            return None;
        }
        self.lo += CHUNK_SIZE;
        Some(start)
    }

    /// Reserve a run of `k` units. Prefers the highest-address free run
    /// inside the top region (keeps the top dense so the low edge can
    /// retreat); otherwise extends the top region downward. Returns the
    /// first unit index.
    fn take_run(&mut self, k: usize) -> Option<usize> {
        if k == 0 || k > self.units.len() {
            return None;
        }
        let k32 = u32::try_from(k).ok()?;
        let mut free = 0;
        let mut found = None;
        for u in (self.hi..self.units.len()).rev() {
            if self.units[u] == UNIT_FREE {
                free += 1;
                if free == k {
                    found = Some(u);
                    break;
                }
            } else {
                free = 0;
            }
        }
        let start = match found {
            Some(u) => u,
            None => {
                // Edge invariant: units[hi] is in use, so no free units
                // touch the frontier — the whole run must be new.
                let start = self.hi.checked_sub(k)?;
                if start * LARGE_UNIT < self.lo {
                    return None;
                }
                self.hi = start;
                start
            }
        };
        self.units[start] = k32;
        for unit in &mut self.units[start + 1..start + k] {
            *unit = UNIT_INTERIOR;
        }
        Some(start)
    }

    /// Widen the run starting at `start` to `k_new` units, in place where
    /// possible. Takes free units directly above the run first; any
    /// shortfall comes from free units directly below it (extending the top
    /// region downward if the run sits at the frontier edge). Returns the
    /// run's new first unit — `start` unless the run had to shift down, in
    /// which case the caller must move the live bytes. `None` (state
    /// unchanged) if no contiguous window around the run is large enough.
    fn grow_run(&mut self, start: usize, k_new: usize) -> Option<usize> {
        let k = self.units[start];
        assert!(
            start >= self.hi && k != UNIT_FREE && k != UNIT_INTERIOR,
            "SlabPool: grow of unit {start} which does not start a run"
        );
        let k = k as usize;
        if k_new <= k {
            return Some(start);
        }
        if k_new > self.units.len() {
            return None;
        }
        let k_new32 = u32::try_from(k_new).ok()?;
        let extra = k_new - k;

        let mut above = 0;
        while above < extra
            && start + k + above < self.units.len()
            && self.units[start + k + above] == UNIT_FREE
        {
            above += 1;
        }
        let shift = extra - above;
        let mut below = 0;
        while below < shift {
            let Some(u) = start.checked_sub(below + 1) else {
                break;
            };
            let free = if u >= self.hi {
                self.units[u] == UNIT_FREE
            } else {
                // Below the top region: claimable while above the bottom
                // frontier.
                u * LARGE_UNIT >= self.lo
            };
            if !free {
                break;
            }
            below += 1;
        }
        if below < shift {
            return None;
        }

        let new_start = start - shift;
        self.units[new_start] = k_new32;
        for unit in &mut self.units[new_start + 1..new_start + k_new] {
            *unit = UNIT_INTERIOR;
        }
        self.hi = self.hi.min(new_start);
        Some(new_start)
    }

    /// Release the run starting at unit `start`. Returns its length.
    fn release_run(&mut self, start: usize) -> usize {
        let k = self.units[start];
        // assert! not debug_assert!: releasing a non-run would mark live
        // memory free and let the bottom frontier carve over it.
        assert!(
            start >= self.hi && k != UNIT_FREE && k != UNIT_INTERIOR,
            "SlabPool: large free of unit {start} which does not start a run — \
             possible double-free or foreign pointer"
        );
        let k = k as usize;
        for unit in &mut self.units[start..start + k] {
            *unit = UNIT_FREE;
        }
        while self.hi < self.units.len() && self.units[self.hi] == UNIT_FREE {
            self.hi += 1;
        }
        k
    }
}

/// Inner state of the slab pool, shared via `Arc` across threads and TCache.
struct SlabPoolInner {
    /// Keeps the mmap alive. Never re-borrowed after construction —
    /// use `base` and `total` instead to avoid Tree Borrows invalidation.
    _region: MmapRegion,
    /// Start of the `WINDOW_ALIGN`-aligned pool window, derived from
    /// `region.as_mut_ptr()` at construction. Using `region.as_ptr()` later
    /// would create a shared reborrow that freezes the tag, making all
    /// writes through mutable pointers UB under Tree Borrows.
    base: *mut u8,
    /// Window size in bytes (== the requested budget).
    total: usize,
    /// Free-list heads for the slab classes. `heads[LARGE_CLASS]` is unused
    /// (large runs are tracked by `frontier`), kept so per-class arrays
    /// share one index space.
    heads: [AtomicHead; NUM_SIZE_CLASSES],
    /// Bottom/top frontier state. Taken once per chunk carve (every
    /// `CHUNK_SIZE` bytes of slab growth) and once per large alloc/free;
    /// never on the free-list fast path.
    frontier: Mutex<Frontier>,
    /// Deallocation class routing table. One byte per MIN_BLOCK_SIZE slot.
    /// Owned raw pointer — written by the block's owner: after a successful
    /// pop CAS or chunk carve (slab), or under the frontier lock (large
    /// runs); read by the owner before the push CAS / lock that releases the
    /// block. No concurrent access to any one entry.
    side_table: *mut u8,
    side_table_len: usize,
    used_bytes: AtomicUsize,
    epoch: AtomicU64,
}

impl Drop for SlabPoolInner {
    fn drop(&mut self) {
        if self.side_table_len > 0 {
            // SAFETY: side_table was allocated via Box::into_raw in new().
            // Reconstruct the Box to let it deallocate.
            unsafe {
                let slice = std::slice::from_raw_parts_mut(self.side_table, self.side_table_len);
                let _ = Box::from_raw(slice);
            }
        }
    }
}

// SAFETY: Pointers into the mmap region are only accessed through atomic CAS
// operations (or Mutex under loom) or under the frontier Mutex. The mmap
// region is owned by SlabPoolInner and outlives all allocations. The side
// table raw pointer is written only by the thread that owns the block (after
// successful CAS or while holding the frontier lock) and read before the
// CAS/lock that returns it — single-owner access, no concurrent mutation.
unsafe impl Send for SlabPoolInner {}
unsafe impl Sync for SlabPoolInner {}

impl SlabPoolInner {
    fn new(budget: usize, hugepages: HugePages, prefault: bool) -> Result<Self, PoolExhausted> {
        if budget > MAX_POOL_BUDGET {
            return Err(PoolExhausted {
                requested: Layout::from_size_align(MAX_POOL_BUDGET, 1)
                    .expect("MAX_POOL_BUDGET <= isize::MAX"),
            });
        }
        // Cannot overflow: budget <= MAX_POOL_BUDGET
        // <= isize::MAX - WINDOW_ALIGN - HUGE_PAGE_SIZE.
        let oversize = budget + WINDOW_ALIGN;
        let oversize = if matches!(hugepages, HugePages::Off) {
            oversize
        } else {
            oversize.next_multiple_of(HUGE_PAGE_SIZE)
        };
        let mut region = MmapRegion::new(oversize, hugepages, prefault)?;

        // Cache the window pointer now. After this point we never
        // re-borrow the MmapMut — calling `as_ptr()` would create a
        // shared reborrow that freezes all outstanding mutable pointers
        // under Tree Borrows.
        let raw = region.as_mut_ptr();
        let pad = (raw as usize).next_multiple_of(WINDOW_ALIGN) - raw as usize;
        // SAFETY: pad < WINDOW_ALIGN and the mapping is budget + WINDOW_ALIGN
        // bytes, so [raw + pad, raw + pad + budget) is inside the mapping.
        let base = unsafe { raw.add(pad) };
        let total = budget;

        // Allocate side table as raw pointer (see struct doc for rationale).
        // Box::into_raw transfers ownership; Drop reconstructs via Box::from_raw.
        let side_table_len = total / SIZE_CLASSES[0];
        let st_box = vec![SIDE_TABLE_UNALLOCATED; side_table_len].into_boxed_slice();
        let side_table = Box::into_raw(st_box) as *mut u8;

        // All free lists start empty: blocks are threaded lazily, one chunk
        // at a time, so a demand-paged pool touches no pages up front.
        let n_units = total / LARGE_UNIT;
        let frontier = Frontier {
            lo: 0,
            hi: n_units,
            units: vec![UNIT_FREE; n_units].into_boxed_slice(),
        };

        Ok(Self {
            _region: region,
            base,
            total,
            heads: std::array::from_fn(|_| new_atomic_head(TaggedPtr::empty())),
            frontier: Mutex::new(frontier),
            side_table,
            side_table_len,
            used_bytes: AtomicUsize::new(0),
            epoch: AtomicU64::new(0),
        })
    }

    /// Bytes carved by the bottom frontier and bytes spanned by the top
    /// region: `(lo, total_units_bytes - hi * LARGE_UNIT)`.
    #[cfg(test)]
    fn frontier_bytes(&self) -> (usize, usize) {
        let f = self.frontier.lock();
        (f.lo, (f.units.len() - f.hi) * LARGE_UNIT)
    }

    fn bump_epoch(&self) {
        self.epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Derive a valid pointer into the mmap region from a raw address.
    ///
    /// `TaggedPtr` stores addresses as `u64` inside a `u128` atomic — this
    /// round-trip strips pointer provenance. Under Tree Borrows, reading
    /// or writing through a provenance-less pointer is UB. This method
    /// restores provenance by computing the offset from `self.base` and
    /// re-deriving the pointer.
    #[inline]
    fn block_ptr(&self, addr: u64) -> *mut u8 {
        let addr_usize = addr as usize;
        debug_assert!(
            addr_usize >= self.base as usize && addr_usize < self.base as usize + self.total,
            "block_ptr: address {addr:#x} is outside pool region"
        );
        let offset = addr_usize - self.base as usize;
        // SAFETY: `addr` was originally derived from `self.base + offset`
        // during a chunk carve or a prior `deallocate`. The offset is within
        // the window (checked by debug_assert above).
        unsafe { self.base.add(offset) }
    }

    /// Side table slot for a block address.
    #[inline]
    fn side_index(&self, addr: usize) -> usize {
        (addr - self.base as usize) / SIZE_CLASSES[0]
    }

    /// Record a block as allocated: side table + accounting.
    ///
    /// # Safety
    /// The caller owns `block` (won the pop CAS, carved its chunk, or holds
    /// the frontier lock for a large run) and `block` is inside the window.
    /// The side table write is non-atomic: an unowned block could race with
    /// its real owner's write.
    unsafe fn record_allocation(&self, block: *mut u8, class: usize, bytes: usize) {
        self.used_bytes.fetch_add(bytes, Ordering::Relaxed);
        let entry = class as u8;
        // SAFETY: index is within bounds (block is in the window), and the
        // block is owned by this thread.
        unsafe {
            self.side_table
                .add(self.side_index(block as usize))
                .write(entry);
        };
    }

    /// Pop one block from a slab class's lock-free free list.
    fn pop_free(&self, class: usize) -> Option<*mut u8> {
        debug_assert!(class < NUM_SLAB_CLASSES);
        loop {
            let head = load_head(&self.heads[class]);
            if head.is_empty() {
                return None;
            }

            // Restore provenance from self.base to read the next-pointer.
            // SAFETY: head.ptr() is a valid block address within the window.
            // The block is >= 64 bytes and 8-byte aligned (min class is 64B),
            // so an AtomicU64 read at offset 0 is safe. Atomic load is
            // required because a concurrent deallocate may be writing a
            // next-pointer into this block (CAS retry loop in push).
            let block = self.block_ptr(head.ptr() as u64);
            let next_raw = unsafe {
                (*(block as *const StdAtomicU64)).load(std::sync::atomic::Ordering::Acquire)
            };
            let next_ptr = if next_raw == 0 {
                std::ptr::null_mut()
            } else {
                next_raw as *mut u8
            };

            if cas_head(&self.heads[class], head, head.with_next(next_ptr)) {
                return Some(block);
            }
            // CAS failed — another thread popped. Retry with spin hint.
            std::hint::spin_loop();
        }
    }

    /// Push the chain `first ..= last` (already linked through their
    /// next-pointers, `last`'s next to be filled in) onto a class free list.
    ///
    /// # Safety
    /// The caller owns every block in the chain and their side table
    /// entries are already `SIDE_TABLE_UNALLOCATED`.
    unsafe fn push_chain(&self, class: usize, first: *mut u8, last: *mut u8) {
        debug_assert!(class < NUM_SLAB_CLASSES);
        loop {
            let head = load_head(&self.heads[class]);
            // Atomic store is required because a concurrent pop may be
            // reading the next-pointer from this block (CAS retry loop).
            // SAFETY: the caller owns `last` (# Safety), a slab block of
            // >= 64 bytes aligned to its class size, so an 8-byte-aligned
            // AtomicU64 at offset 0 is in bounds.
            unsafe {
                (*(last as *const StdAtomicU64))
                    .store(head.ptr() as u64, std::sync::atomic::Ordering::Release);
            }
            if cas_head(&self.heads[class], head, head.with_next(first)) {
                return;
            }
            // CAS failed — another thread pushed. Retry with spin hint.
            std::hint::spin_loop();
        }
    }

    /// Carve a fresh chunk for `class` from the bottom frontier. Returns
    /// one block to the caller and pushes the rest onto the class free list.
    fn carve(&self, class: usize) -> Option<*mut u8> {
        debug_assert!(class < NUM_SLAB_CLASSES);
        let offset = self.frontier.lock().carve_chunk(self.total)?;
        let block_size = SIZE_CLASSES[class];
        let count = CHUNK_SIZE / block_size;
        // SAFETY: carve_chunk returned [offset, offset + CHUNK_SIZE) inside
        // the window, now exclusively owned by this thread.
        let chunk = unsafe { self.base.add(offset) };
        if count > 1 {
            // Thread blocks 1..count; block 0 goes to the caller. Atomic
            // stores keep init writes consistent with the atomic loads in
            // pop_free; publication is ordered by push_chain's CAS.
            for i in 1..count - 1 {
                // SAFETY: the chunk is exclusively owned (fresh carve), and
                // block i starts at i * block_size with block_size >= 64 and
                // (i + 1) * block_size <= CHUNK_SIZE, so the 8-byte, 8-aligned
                // AtomicU64 at its start is in bounds; the stored value is the
                // address of block i + 1, also inside the chunk.
                unsafe {
                    let block = chunk.add(i * block_size);
                    (*(block as *const StdAtomicU64)).store(
                        chunk.add((i + 1) * block_size) as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
            // SAFETY: blocks 1..count are owned (fresh chunk) and their side
            // table entries were never written (or cleared when a large run
            // previously covering them was released).
            unsafe {
                self.push_chain(
                    class,
                    chunk.add(block_size),
                    chunk.add((count - 1) * block_size),
                );
            }
        }
        Some(chunk)
    }

    /// Allocate from a slab class: its free list, then a fresh chunk from
    /// the bottom frontier. There is no cross-class borrowing: with
    /// demand-carved chunks a class only runs dry once the frontiers have
    /// met, and borrowing a larger block then would waste 4-16x per request
    /// for a handful of extra allocations (measured: no utilization change).
    fn slab_allocate(&self, class: usize, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        debug_assert!(class < NUM_SLAB_CLASSES);
        let block = self
            .pop_free(class)
            .or_else(|| self.carve(class))
            .ok_or(PoolExhausted { requested: layout })?;

        // SAFETY: block was popped (CAS won) or carved by this thread.
        unsafe { self.record_allocation(block, class, SIZE_CLASSES[class]) };
        debug_assert!(
            (block as usize).is_multiple_of(layout.align()),
            "SlabPool: block at {block:p} is not aligned to {}",
            layout.align()
        );
        Ok(NonNull::new(block).expect("free list block is non-null"))
    }

    /// Allocate a large run from the top frontier.
    fn large_allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        let k = large_units_for(layout);
        let mut frontier = self.frontier.lock();
        let start = frontier
            .take_run(k)
            .ok_or(PoolExhausted { requested: layout })?;
        // SAFETY: take_run returned units [start, start + k), inside the
        // window and exclusively owned while the run is recorded.
        let block = unsafe { self.base.add(start * LARGE_UNIT) };
        // Side table written under the lock: release_run clears it under the
        // same lock, so the entry is never observed half-updated.
        // SAFETY: the run was reserved by take_run under the held lock.
        unsafe { self.record_allocation(block, LARGE_CLASS, k * LARGE_UNIT) };
        drop(frontier);
        debug_assert!((block as usize).is_multiple_of(layout.align()));
        Ok(NonNull::new(block).expect("window pointer is non-null"))
    }

    /// Release a large run. Caller has already validated the side table
    /// entry says `LARGE_CLASS`.
    ///
    /// # Safety
    /// `block` must be the start of a live large run owned by the caller.
    unsafe fn large_deallocate(&self, block: *mut u8) {
        let offset = block as usize - self.base as usize;
        assert!(
            offset.is_multiple_of(LARGE_UNIT),
            "SlabPool::deallocate: large pointer {block:p} is not unit-aligned"
        );
        let mut frontier = self.frontier.lock();
        // Clear the side table BEFORE the units become reusable — once
        // release_run returns (and the lock drops) the bottom frontier may
        // carve over these units and write their slots.
        // SAFETY: index is within bounds; the run is owned by the caller.
        unsafe {
            self.side_table
                .add(self.side_index(block as usize))
                .write(SIDE_TABLE_UNALLOCATED);
        };
        let k = frontier.release_run(offset / LARGE_UNIT);
        drop(frontier);
        self.used_bytes.fetch_sub(k * LARGE_UNIT, Ordering::Relaxed);
    }

    /// Grow a large run to hold `new` in place (see `Frontier::grow_run`),
    /// moving `live` bytes down if the run shifts. Returns `None` with the
    /// run untouched if there is no room around it.
    ///
    /// # Safety
    /// `block` is the start of a live large run owned by the caller, and
    /// `live` does not exceed the run's length in bytes.
    unsafe fn large_grow(&self, block: *mut u8, new: Layout, live: usize) -> Option<NonNull<u8>> {
        let offset = block as usize - self.base as usize;
        assert!(
            offset.is_multiple_of(LARGE_UNIT),
            "SlabPool::grow: large pointer {block:p} is not unit-aligned"
        );
        let start = offset / LARGE_UNIT;
        let k_new = large_units_for(new);
        let mut frontier = self.frontier.lock();
        let k_old = frontier.units[start] as usize;
        let new_start = frontier.grow_run(start, k_new)?;
        let k_new = k_new.max(k_old);
        // SAFETY: new_start is a unit inside the window.
        let new_block = unsafe { self.base.add(new_start * LARGE_UNIT) };
        if new_start != start {
            // Move the routing entry with the run, under the lock (same
            // discipline as large_allocate / large_deallocate).
            // SAFETY: both indices are within bounds; the run is owned.
            unsafe {
                let old_slot = self.side_table.add(self.side_index(block as usize));
                let entry = old_slot.read();
                old_slot.write(SIDE_TABLE_UNALLOCATED);
                self.side_table
                    .add(self.side_index(new_block as usize))
                    .write(entry);
            }
        }
        drop(frontier);
        self.used_bytes
            .fetch_add((k_new - k_old) * LARGE_UNIT, Ordering::Relaxed);
        if new_start != start {
            // SAFETY: [new_block, new_block + k_new units) is owned by the
            // caller and contains the old run, so both ranges are valid;
            // `copy` handles the overlap.
            unsafe { std::ptr::copy(block, new_block, live) };
        }
        NonNull::new(new_block)
    }

    /// Allocate without touching the TCACHE thread-local.
    /// SAFETY-CRITICAL: Does NOT touch TCACHE thread-local.
    fn global_allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }
        match size_class_for(layout) {
            Some(LARGE_CLASS) => self.large_allocate(layout),
            Some(class) => self.slab_allocate(class, layout),
            None => Err(PoolExhausted { requested: layout }),
        }
    }

    /// Validate `ptr` against the window and side table; return its block
    /// pointer (provenance restored) and class.
    ///
    /// # Safety
    /// The caller owns the allocation at `ptr` (it is being deallocated or
    /// grown). Window bounds and the entry's class are asserted, but the
    /// side table read is non-atomic and only race-free for the owner.
    unsafe fn lookup(&self, ptr: NonNull<u8>) -> (*mut u8, usize) {
        let addr = ptr.as_ptr() as usize;

        // assert! not debug_assert!: a foreign pointer causes UB via
        // block_ptr() (wrapping ptr::add). Two comparisons prevent the
        // entire class of foreign-pointer bugs from becoming silent UB.
        assert!(
            addr >= self.base as usize && addr < self.base as usize + self.total,
            "SlabPool::deallocate: pointer {:p} is outside pool region [{:p}..+{:#x})",
            ptr.as_ptr(),
            self.base,
            self.total
        );

        // Read actual class from side table (authoritative — ignores layout).
        // SAFETY: index is within bounds, block is owned by this thread.
        let entry = unsafe { self.side_table.add(self.side_index(addr)).read() };
        let class = entry as usize;

        // assert! not debug_assert!: an OOB index on self.heads[] is instant
        // UB. This catches double-free and foreign-pointer contract violations.
        assert!(
            class < NUM_SIZE_CLASSES,
            "SlabPool::deallocate: side table entry {entry:#x} has invalid class {class} \
             — possible double-free or foreign pointer"
        );
        (self.block_ptr(addr as u64), class)
    }

    /// Return one allocation to the pool, routed by its side table entry.
    /// SAFETY-CRITICAL: Does NOT touch TCACHE thread-local.
    unsafe fn global_deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // SAFETY: forwarded caller contract (caller owns ptr).
        let (block, class) = unsafe { self.lookup(ptr) };
        if class == LARGE_CLASS {
            // SAFETY: forwarded caller contract.
            unsafe { self.large_deallocate(block) };
        } else {
            // SAFETY: forwarded caller contract.
            unsafe { self.slab_deallocate(class, block) };
        }
    }

    /// Return one slab block to its class's global free list.
    ///
    /// # Safety
    /// `block` is a live slab block of `class` owned by the caller.
    unsafe fn slab_deallocate(&self, class: usize, block: *mut u8) {
        // Mark side table entry as unallocated BEFORE pushing the block
        // back. Once the CAS succeeds, another thread can immediately pop
        // the block and write its own side table entry — writing after
        // the CAS would be a data race.
        // SAFETY: block is owned by this thread (not yet pushed).
        unsafe {
            self.side_table
                .add(self.side_index(block as usize))
                .write(SIDE_TABLE_UNALLOCATED);
        };
        // SAFETY: single owned block, side table cleared above.
        unsafe { self.push_chain(class, block, block) };
        self.used_bytes
            .fetch_sub(SIZE_CLASSES[class], Ordering::Relaxed);
    }

    /// Return one block to a specific class's global free list.
    /// Used by TCache flush/spill. Class is caller-provided.
    /// SAFETY-CRITICAL: Does NOT touch TCACHE thread-local.
    unsafe fn global_deallocate_to_class(&self, class: usize, ptr: *mut u8) {
        // TCache bins only ever hold slab blocks: large runs bypass it on
        // both allocate and deallocate, and refill never caches one.
        debug_assert!(
            class < NUM_SLAB_CLASSES,
            "class {class} is not a slab class"
        );
        let (block, actual) =
            // SAFETY: forwarded caller contract (TCache owns cached blocks).
            unsafe { self.lookup(NonNull::new(ptr).expect("cached block is non-null")) };

        // Verify caller-provided class matches side table (defense-in-depth).
        debug_assert_eq!(
            class, actual,
            "global_deallocate_to_class: bin class {class} != side table class {actual}"
        );
        // Route by the authoritative side table entry, never by the bin.
        if actual == LARGE_CLASS {
            // SAFETY: forwarded caller contract.
            unsafe { self.large_deallocate(block) };
        } else {
            // SAFETY: forwarded caller contract.
            unsafe { self.slab_deallocate(actual, block) };
        }
    }
}

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
        // assert! not debug_assert!: OOB write at stack[32] overwrites `count`
        // (adjacent in repr(C) layout), corrupting the allocator silently.
        assert!(self.count < TCACHE_CAPACITY, "TCacheBin overflow");
        self.stack[self.count] = ptr;
        self.count += 1;
    }

    fn is_full(&self) -> bool {
        self.count >= TCACHE_CAPACITY
    }
}

struct TCache {
    bins: [TCacheBin; NUM_SLAB_CLASSES],
    local_epoch: u64,
    pool: Arc<SlabPoolInner>,
}

impl TCache {
    fn new(pool: Arc<SlabPoolInner>) -> Self {
        Self {
            bins: [const { TCacheBin::empty() }; NUM_SLAB_CLASSES],
            local_epoch: pool.epoch.load(std::sync::atomic::Ordering::Acquire),
            pool,
        }
    }

    /// Refill a bin: one block for the caller from the global pool (which
    /// may carve a fresh chunk), plus up to TCACHE_REFILL - 1 extra blocks
    /// popped from `class`'s already-carved free list.
    ///
    /// Extras never carve: a thread cache must not pull fresh memory off the
    /// shared frontier that other classes (or large runs) could have used.
    fn refill(&mut self, class: usize, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        debug_assert!(class < NUM_SLAB_CLASSES);
        let first = self.pool.global_allocate(layout)?;
        for _ in 1..TCACHE_REFILL {
            let Some(block) = self.pool.pop_free(class) else {
                break;
            };
            // CAS in pop_free won — this thread owns the block.
            // SAFETY: pop_free's CAS won, so this thread owns `block`.
            unsafe {
                self.pool
                    .record_allocation(block, class, SIZE_CLASSES[class]);
            }
            if self.bins[class].is_full() {
                self.spill(class);
            }
            self.bins[class].push(block);
        }
        Ok(first)
    }

    /// Flush all bins back to the global pool (epoch mismatch).
    fn flush_all(&mut self) {
        for class in 0..NUM_SLAB_CLASSES {
            let bin = &mut self.bins[class];
            for i in 0..bin.count {
                unsafe {
                    self.pool.global_deallocate_to_class(class, bin.stack[i]);
                }
            }
            bin.count = 0;
        }
    }

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

impl Drop for TCache {
    fn drop(&mut self) {
        for class in 0..NUM_SLAB_CLASSES {
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

#[cfg(not(loom))]
thread_local! {
    static TCACHE: std::cell::UnsafeCell<Option<TCache>> =
        const { std::cell::UnsafeCell::new(None) };
}

// ── SlabPool (public wrapper) ────────────────────────────────────

/// TigerBeetle-style allocator backed by a single mmap'd region.
///
/// The region is one window with two opposing frontiers (see `Frontier`):
/// the bottom frontier grows upward, carving 64 KiB chunks on demand for
/// five slab classes (64 B – 64 KiB); the top frontier grows downward,
/// serving large requests as runs of 1 MiB units. No space is reserved per
/// class up front — the budget is exhausted only when the frontiers meet
/// (or every carved block is in use). Slab allocation uses a lock-free
/// intrusive stack per class via `AtomicHead` (128-bit tagged pointer CAS)
/// and a thread-local cache (TCache) that amortizes global CAS operations
/// across batches of 16 allocations.
///
// PERF: Lock-free intrusive stack via AtomicHead (Atomic<u128> tagged pointer).
// ABA prevention via 64-bit monotonic tag. Side table routes deallocation to
// the correct class regardless of the caller's Layout. The frontier Mutex is
// taken once per 64 KiB chunk carve and once per large alloc/free, never on
// the free-list fast path. TCache L1 provides ~3 instruction fast path for
// 15/16 allocations.
pub(crate) struct SlabPool {
    inner: Arc<SlabPoolInner>,
}

impl SlabPool {
    /// Create a new slab pool backed by an mmap region of `budget` bytes.
    ///
    /// When `prefault` is true, all pages are pre-faulted into physical
    /// RAM at creation time. When false, pages are demand-paged by the OS.
    pub(crate) fn new(
        budget: usize,
        hugepages: HugePages,
        prefault: bool,
    ) -> Result<Self, PoolExhausted> {
        Ok(Self {
            inner: Arc::new(SlabPoolInner::new(budget, hugepages, prefault)?),
        })
    }

    /// Flush the current thread's TCache back to the global pool.
    /// Test-only: needed for exact used_bytes accounting in tests.
    /// Under loom, TCache is bypassed — this is a no-op.
    #[cfg(test)]
    #[allow(clippy::unused_self)]
    fn flush_current_thread_cache(&self) {
        #[cfg(not(loom))]
        TCACHE.with(|cell| {
            // SAFETY: same no-reentrancy argument as allocate/deallocate.
            let cache = unsafe { &mut *cell.get() };
            if let Some(c) = cache.as_mut() {
                c.flush_all();
            }
        });
    }
}

// SAFETY: SlabPool returns properly aligned, non-overlapping memory regions.
// Slab blocks come from CHUNK_SIZE-aligned chunks, so each block is aligned
// to its class size; classes are chosen such that `layout.size() <=
// block_size` and `layout.align() <= block_size`. Large runs start on a
// LARGE_UNIT boundary and span `ceil(max(size, align) / LARGE_UNIT)` units;
// `size_class_for` rejects alignments above LARGE_UNIT. Chunks and runs are
// disjoint because the frontiers never cross (`Frontier` invariants).
// Lock-free CAS on AtomicHead serializes access to each free list.
impl SlabPool {
    pub(crate) fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }

        let class = size_class_for(layout).ok_or(PoolExhausted { requested: layout })?;

        // Large runs bypass the thread cache: they are rare, each one is at
        // least LARGE_UNIT bytes, and they must return to the shared top
        // frontier promptly so the bottom frontier can reclaim the space.
        if class == LARGE_CLASS {
            return self.inner.large_allocate(layout);
        }

        #[cfg(not(loom))]
        {
            TCACHE.with(|cell| {
                // SAFETY: No reentrancy — allocate() is not called from within
                // this closure. BlobVec::grow is the sole caller and does not
                // nest allocations.
                let slot = unsafe { &mut *cell.get() };

                // If the TCache belongs to a different pool, flush and replace.
                if let Some(existing) = slot.as_ref()
                    && !Arc::ptr_eq(&existing.pool, &self.inner)
                {
                    // Drop the old TCache (flushes blocks to old pool).
                    *slot = None;
                }

                let cache = slot.get_or_insert_with(|| TCache::new(Arc::clone(&self.inner)));

                // Epoch check — lazy flush if stale.
                let global_epoch = self.inner.epoch.load(std::sync::atomic::Ordering::Acquire);
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

    pub(crate) unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        #[cfg(not(loom))]
        {
            // Read actual class from side table BEFORE touching TCache.
            // This ensures bins are "pure" — no bin drift. `lookup` asserts
            // the pointer is inside the window and the entry is valid.
            // SAFETY: caller contract — ptr is a live allocation being freed.
            let (_, actual_class) = unsafe { self.inner.lookup(ptr) };

            if actual_class == LARGE_CLASS {
                // SAFETY: forwarded caller contract.
                unsafe { self.inner.global_deallocate(ptr, layout) };
                return;
            }

            TCACHE.with(|cell| {
                let slot = unsafe { &mut *cell.get() };

                // If the TCache belongs to a different pool, flush and replace.
                if let Some(existing) = slot.as_ref()
                    && !Arc::ptr_eq(&existing.pool, &self.inner)
                {
                    *slot = None;
                }

                let cache = slot.get_or_insert_with(|| TCache::new(Arc::clone(&self.inner)));

                // Epoch check — lazy flush if stale (must check on BOTH
                // allocate and deallocate per spec).
                let global_epoch = self.inner.epoch.load(std::sync::atomic::Ordering::Acquire);
                if cache.local_epoch != global_epoch {
                    cache.flush_all();
                    cache.local_epoch = global_epoch;
                }

                // Spill BEFORE pushing if full to avoid overflow.
                if cache.bins[actual_class].is_full() {
                    cache.spill(actual_class);
                }

                cache.bins[actual_class].push(ptr.as_ptr());
            });
        }

        #[cfg(loom)]
        {
            self.inner.global_deallocate(ptr, layout);
        }
    }

    /// Grow an allocation from `old` to `new`, preserving its first `live`
    /// bytes (realloc). Returns the same pointer when the current block
    /// already fits `new`, or when a large run can widen into free units
    /// around it — the latter avoids the transient 2x footprint of
    /// allocate-copy-free, which is what lets a single column grow past
    /// half the remaining budget. Otherwise falls back to allocate + copy +
    /// deallocate.
    ///
    /// On `Err` the original allocation is untouched and still valid.
    ///
    /// # Safety
    /// `ptr` was returned by `allocate(old)` on this pool and not yet
    /// deallocated; `live <= old.size()`. On `Ok`, `ptr` must no longer be
    /// used — the returned pointer (allocated with `new`) replaces it.
    pub(crate) unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        live: usize,
    ) -> Result<NonNull<u8>, PoolExhausted> {
        debug_assert!(new.size() >= old.size() && new.align() == old.align());
        debug_assert!(live <= old.size());
        if old.size() == 0 {
            return self.allocate(new);
        }
        let new_class = size_class_for(new).ok_or(PoolExhausted { requested: new })?;
        // SAFETY: caller contract — ptr is a live allocation being grown.
        let (block, class) = unsafe { self.inner.lookup(ptr) };
        if class < NUM_SLAB_CLASSES {
            // Blocks are aligned to their class size, so fitting the
            // effective size also satisfies the alignment.
            if new_class <= class {
                return Ok(ptr);
            }
        } else if let Some(grown) =
            // SAFETY: lookup returned LARGE_CLASS for ptr, so `block` is the
            // start of a live run owned by the caller (caller contract), and
            // live <= old.size() <= the run's length in bytes.
            unsafe { self.inner.large_grow(block, new, live) }
        {
            return Ok(grown);
        }

        let new_ptr = self.allocate(new)?;
        // SAFETY: distinct live allocations; live <= old.size() <= new.size().
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), live);
            self.deallocate(ptr, old);
        }
        Ok(new_ptr)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.inner.total
    }

    pub(crate) fn used(&self) -> usize {
        self.inner.used_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn flush_caches(&self) {
        // Eagerly flush the calling thread's TCache.
        #[cfg(not(loom))]
        TCACHE.with(|cell| {
            let slot = unsafe { &mut *cell.get() };
            if let Some(cache) = slot.as_mut()
                && Arc::ptr_eq(&cache.pool, &self.inner)
            {
                cache.flush_all();
            }
        });
        // Lazily flush other threads via epoch bump.
        self.inner.bump_epoch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pool_is_slab_pool() {
        let pool = default_pool();
        assert_eq!(pool.capacity(), DEFAULT_POOL_BUDGET);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn shared_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<SlabPool>>();
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
    fn side_table_entry_is_class_index() {
        // Allocated entries are the bare class index; the sentinel is out
        // of range so lookup's class assert rejects freed/foreign slots.
        assert!((SIDE_TABLE_UNALLOCATED as usize) >= NUM_SIZE_CLASSES);
        assert!(u8::try_from(NUM_SIZE_CLASSES).is_ok());
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
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!(pool.used() > 0);
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_returns_error_on_exhaustion() {
        let pool = SlabPool::new(64 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.allocate(layout) {
            ptrs.push(ptr);
        }
        assert_ne!(ptrs.len(), 0);

        // Deallocate one and re-allocate succeeds.
        // SAFETY: ptrs were returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptrs.pop().unwrap(), layout) };
        assert!(pool.allocate(layout).is_ok());
    }

    #[test]
    fn slab_pool_observability() {
        let pool = SlabPool::new(1024 * 1024, HugePages::Off, false).unwrap();
        assert_eq!(pool.capacity(), 1024 * 1024);
        assert_eq!(pool.used(), 0);

        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!(pool.used() > 0);
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_concurrent_allocate_no_duplicates() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off, false).unwrap());
        let layout = Layout::from_size_align(64, 64).unwrap();

        let all_ptrs: Vec<Vec<usize>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let pool = Arc::clone(&pool);
                    s.spawn(move || {
                        let mut ptrs = Vec::new();
                        for _ in 0..200 {
                            if let Ok(ptr) = pool.allocate(layout) {
                                ptrs.push(ptr.as_ptr() as usize);
                            }
                        }
                        ptrs
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Verify no duplicates across all threads.
        let mut all: Vec<usize> = all_ptrs.iter().flatten().copied().collect();
        let total = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(
            all.len(),
            total,
            "duplicate pointers returned across threads"
        );

        // Deallocate all.
        for addr in all_ptrs.iter().flatten() {
            let ptr = NonNull::new(*addr as *mut u8).unwrap();
            unsafe { pool.deallocate(ptr, layout) };
        }
        // Flush main thread's TCache so blocks return to global pool.
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_concurrent_alloc_dealloc_interleaved() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off, false).unwrap());
        let layout = Layout::from_size_align(64, 64).unwrap();

        // Use std::thread::spawn (not scope) so threads fully exit and
        // their thread-local TCache destructors run before we check.
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        let ptr = pool.allocate(layout).unwrap();
                        // SAFETY: ptr was just allocated.
                        unsafe { pool.deallocate(ptr, layout) };
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // After threads exit, their TCache destructors flush all blocks.
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_multiple_size_classes() {
        let pool = SlabPool::new(8 * 1024 * 1024, HugePages::Off, false).unwrap();

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
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        assert_eq!(pool.used(), 0);

        let layout = Layout::from_size_align(64, 64).unwrap();
        let p1 = pool.allocate(layout).unwrap();
        let p2 = pool.allocate(layout).unwrap();
        assert!(pool.used() > 0);

        // SAFETY: ptrs were returned by `allocate` with this layout.
        unsafe {
            pool.deallocate(p1, layout);
            pool.deallocate(p2, layout);
        }
        // Flush TCache so blocks return to global pool for accounting.
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_zero_size_allocation() {
        let pool = SlabPool::new(1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(0, 1).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert_eq!(pool.used(), 0, "ZST should not consume pool bytes");
        // SAFETY: ptr was returned by `allocate` with this layout (zero-size).
        unsafe { pool.deallocate(ptr, layout) };
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_oversized_returns_error() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        // Larger than the whole budget.
        let layout = Layout::from_size_align(5 * 1024 * 1024, 8).unwrap();
        assert!(pool.allocate(layout).is_err());
        // Alignment above LARGE_UNIT cannot be honored.
        let over_aligned = Layout::from_size_align(64, 2 * LARGE_UNIT).unwrap();
        assert!(pool.allocate(over_aligned).is_err());
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_rejects_budget_above_max() {
        // Must return PoolExhausted, not panic building the error layout.
        assert!(SlabPool::new(usize::MAX, HugePages::Off, false).is_err());
        assert!(SlabPool::new(MAX_POOL_BUDGET + 1, HugePages::Off, false).is_err());
        const { assert!(MAX_POOL_BUDGET / LARGE_UNIT < UNIT_INTERIOR as usize) };
    }

    #[test]
    fn slab_pool_multi_megabyte_allocation() {
        // The old 1 MiB ceiling is gone: large requests are unit runs.
        let pool = SlabPool::new(8 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(2 * 1024 * 1024 + 1, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!((ptr.as_ptr() as usize).is_multiple_of(LARGE_UNIT));
        assert_eq!(pool.used(), 3 * LARGE_UNIT, "rounded up to 3 units");
        // Whole run is writable.
        // SAFETY: ptr is valid for layout.size() bytes.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0xAB, layout.size()) };
        unsafe { pool.deallocate(ptr, layout) };
        assert_eq!(pool.used(), 0);
        assert_eq!(
            pool.inner.frontier_bytes().1,
            0,
            "top region fully retreated"
        );
    }

    /// Two chunks: small enough to exhaust quickly under Miri, and no room
    /// for a large unit, so frontier exhaustion is purely slab chunks.
    const SMALL_POOL: usize = 2 * CHUNK_SIZE;

    /// Fill the whole frontier with chunks of `layout`'s class via the
    /// global path. Returns every block handed out.
    fn exhaust_frontier(pool: &SlabPool, layout: Layout) -> Vec<NonNull<u8>> {
        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.inner.global_allocate(layout) {
            ptrs.push(ptr);
        }
        ptrs
    }

    #[test]
    fn slab_pool_exhausted_class_never_borrows_larger_class() {
        // Frontier full of class-1 chunks with free class-1 blocks: a
        // class-0 request fails instead of taking a 256 B block.
        let pool = SlabPool::new(SMALL_POOL, HugePages::Off, false).unwrap();
        let class1 = Layout::from_size_align(200, 8).unwrap();
        let class0 = Layout::from_size_align(32, 8).unwrap();

        let mut medium = exhaust_frontier(&pool, class1);
        assert_eq!(medium.len(), SMALL_POOL / SIZE_CLASSES[1]);
        let freed = medium.pop().unwrap();
        unsafe { pool.inner.global_deallocate(freed, class1) };

        assert!(pool.inner.global_allocate(class0).is_err());
        // The freed block is still there for its own class.
        let again = pool.inner.global_allocate(class1).unwrap();
        assert_eq!(again, freed);
        medium.push(again);

        for ptr in medium {
            unsafe { pool.inner.global_deallocate(ptr, class1) };
        }
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_classes_never_share_blocks() {
        // A free class-1 block is not handed to class 0; class 0 carves its
        // own chunk while frontier space remains.
        let pool = SlabPool::new(LARGE_UNIT, HugePages::Off, false).unwrap();
        let class1 = Layout::from_size_align(200, 8).unwrap();
        let class0 = Layout::from_size_align(32, 8).unwrap();

        let p1 = pool.inner.global_allocate(class1).unwrap();
        unsafe { pool.inner.global_deallocate(p1, class1) };
        let p0 = pool.inner.global_allocate(class0).unwrap();
        assert_ne!(p0, p1);
        assert_eq!(pool.inner.frontier_bytes().0, 2 * CHUNK_SIZE);
        assert_eq!(pool.used(), SIZE_CLASSES[0]);
        unsafe { pool.inner.global_deallocate(p0, class0) };
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_dealloc_routes_by_side_table_not_layout() {
        // The side table is authoritative: a wrong (but non-zero) layout on
        // deallocate still returns the block to its real class.
        let pool = SlabPool::new(LARGE_UNIT, HugePages::Off, false).unwrap();
        let class0 = Layout::from_size_align(32, 8).unwrap();
        let wrong = Layout::from_size_align(4000, 8).unwrap();

        let p = pool.inner.global_allocate(class0).unwrap();
        unsafe { pool.inner.global_deallocate(p, wrong) };
        assert_eq!(pool.used(), 0, "freed its real class size (64 B)");
        assert_eq!(pool.inner.global_allocate(class0).unwrap(), p);
        unsafe { pool.inner.global_deallocate(p, class0) };
    }

    #[test]
    fn slab_pool_one_class_can_exhaust_the_frontier() {
        // No per-class reservation: class 1 alone takes every chunk, and
        // class 0 then fails — even after class 1 frees everything, because
        // carved chunks stay with their class (see the design doc's
        // "return fully-free chunks" follow-up).
        let pool = SlabPool::new(512 * 1024, HugePages::Off, false).unwrap();
        let layout_small = Layout::from_size_align(32, 8).unwrap(); // class 0
        let layout_medium = Layout::from_size_align(200, 8).unwrap(); // class 1

        let mut ptrs1 = Vec::new();
        while let Ok(ptr) = pool.allocate(layout_medium) {
            ptrs1.push(ptr);
        }
        assert_eq!(ptrs1.len(), 512 * 1024 / SIZE_CLASSES[1]);
        assert!(pool.allocate(layout_small).is_err());

        for ptr in ptrs1 {
            unsafe { pool.deallocate(ptr, layout_medium) };
        }
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
        assert!(
            pool.allocate(layout_small).is_err(),
            "chunks stay with class 1"
        );
    }

    #[test]
    fn slab_pool_largest_slab_class_never_takes_large_run() {
        // 3 units. A at unit 2, B at unit 1 → hi = 1. Freeing A leaves a
        // hole at unit 2 that the bottom frontier cannot reach. Exhausting
        // the 64 KiB class must not consume that hole; a large request can.
        let pool = SlabPool::new(3 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let unit = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let class4 = Layout::from_size_align(60_000, 64).unwrap();

        let a = pool.inner.global_allocate(unit).unwrap();
        let b = pool.inner.global_allocate(unit).unwrap();
        unsafe { pool.inner.global_deallocate(a, unit) };

        let blocks = exhaust_frontier(&pool, class4);
        assert_eq!(
            blocks.len(),
            LARGE_UNIT / CHUNK_SIZE,
            "only the carved space below unit 1"
        );
        let reused = pool.inner.global_allocate(unit).unwrap();
        assert_eq!(reused, a, "the hole is still free for a large run");

        for ptr in blocks {
            unsafe { pool.inner.global_deallocate(ptr, class4) };
        }
        unsafe {
            pool.inner.global_deallocate(reused, unit);
            pool.inner.global_deallocate(b, unit);
        }
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_large_and_slab_share_one_budget() {
        let pool = SlabPool::new(8 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let layout_large = Layout::from_size_align(1_000_000, 8).unwrap();
        let layout_small = Layout::from_size_align(32, 8).unwrap();

        // Large runs can take the entire budget — nothing is reserved for
        // the slab classes up front.
        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.allocate(layout_large) {
            ptrs.push(ptr);
        }
        assert_eq!(ptrs.len(), 8);
        assert!(pool.allocate(layout_small).is_err(), "frontiers have met");

        // Freeing the lowest run retreats the top frontier; the bottom
        // frontier can immediately carve that space.
        let lowest = ptrs.pop().unwrap();
        unsafe { pool.deallocate(lowest, layout_large) };
        let small_ptr = pool.allocate(layout_small).unwrap();
        assert_eq!(small_ptr, lowest, "first chunk carved at the old top edge");
        unsafe { pool.deallocate(small_ptr, layout_small) };

        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout_large) };
        }
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_large_run_reuses_interior_hole() {
        let pool = SlabPool::new(4 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let unit = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let two = Layout::from_size_align(2 * LARGE_UNIT, 64).unwrap();

        let a = pool.allocate(unit).unwrap(); // unit 3
        let b = pool.allocate(unit).unwrap(); // unit 2
        let c = pool.allocate(unit).unwrap(); // unit 1
        assert!(a > b && b > c, "top frontier grows downward");

        unsafe { pool.deallocate(b, unit) };
        assert_eq!(
            pool.inner.frontier_bytes().1,
            3 * LARGE_UNIT,
            "hole is interior"
        );
        let d = pool.allocate(unit).unwrap();
        assert_eq!(d, b, "highest free run is reused before extending");

        // No 2-unit run fits (unit 0 + nothing contiguous above it is free).
        assert!(pool.allocate(two).is_err());

        unsafe { pool.deallocate(c, unit) };
        assert_eq!(
            pool.inner.frontier_bytes().1,
            2 * LARGE_UNIT,
            "freeing the edge run retreats the frontier"
        );
        let e = pool.allocate(two).unwrap();
        assert_eq!(e.as_ptr() as usize, c.as_ptr() as usize - LARGE_UNIT);

        unsafe {
            pool.deallocate(a, unit);
            pool.deallocate(d, unit);
            pool.deallocate(e, two);
        }
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.inner.frontier_bytes().1, 0);
    }

    /// Fill `len` bytes at `ptr` with a position-dependent pattern.
    unsafe fn fill(ptr: NonNull<u8>, len: usize) {
        for i in 0..len {
            unsafe { ptr.as_ptr().add(i).write((i % 251) as u8) };
        }
    }

    /// Check the pattern written by `fill`.
    unsafe fn check(ptr: NonNull<u8>, len: usize) {
        for i in 0..len {
            assert_eq!(
                unsafe { ptr.as_ptr().add(i).read() },
                (i % 251) as u8,
                "byte {i}"
            );
        }
    }

    #[test]
    fn grow_within_same_slab_block_keeps_pointer() {
        let pool = SlabPool::new(LARGE_UNIT, HugePages::Off, false).unwrap();
        let old = Layout::from_size_align(32, 8).unwrap();
        let new = Layout::from_size_align(64, 8).unwrap();
        let p = pool.allocate(old).unwrap();
        let used = pool.used();
        let q = unsafe { pool.grow(p, old, new, 32) }.unwrap();
        assert_eq!(p, q);
        assert_eq!(pool.used(), used);
        unsafe { pool.deallocate(q, new) };
    }

    #[test]
    fn grow_slab_to_larger_class_copies_live_bytes() {
        let pool = SlabPool::new(LARGE_UNIT, HugePages::Off, false).unwrap();
        let old = Layout::from_size_align(64, 64).unwrap();
        let new = Layout::from_size_align(256, 64).unwrap();
        let p = pool.allocate(old).unwrap();
        unsafe { fill(p, 48) };
        let q = unsafe { pool.grow(p, old, new, 48) }.unwrap();
        assert_ne!(p, q);
        unsafe { check(q, 48) };
        unsafe { pool.deallocate(q, new) };
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn grow_large_run_in_place_upward() {
        let pool = SlabPool::new(4 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let one = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let two = Layout::from_size_align(2 * LARGE_UNIT, 64).unwrap();
        let a = pool.allocate(one).unwrap(); // unit 3
        let b = pool.allocate(one).unwrap(); // unit 2
        unsafe { pool.deallocate(a, one) }; // hole directly above b
        unsafe { fill(b, 4096) };

        let grown = unsafe { pool.grow(b, one, two, 4096) }.unwrap();
        assert_eq!(grown, b, "widened into the free unit above, no move");
        unsafe { check(grown, 4096) };
        assert_eq!(pool.used(), 2 * LARGE_UNIT);
        unsafe { pool.deallocate(grown, two) };
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.inner.frontier_bytes().1, 0);
    }

    #[test]
    fn grow_large_run_shifts_down_and_moves_bytes() {
        let pool = SlabPool::new(4 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let one = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let three = Layout::from_size_align(3 * LARGE_UNIT, 64).unwrap();
        let a = pool.allocate(one).unwrap(); // unit 3: nothing above
        // Most of the unit natively; a few KiB under Miri, where a byte loop
        // over 1 MiB exceeds the per-test timeout.
        let live = if cfg!(miri) { 8192 + 7 } else { LARGE_UNIT - 7 };
        unsafe { fill(a, live) };

        let grown = unsafe { pool.grow(a, one, three, live) }.unwrap();
        assert_eq!(
            grown.as_ptr() as usize,
            a.as_ptr() as usize - 2 * LARGE_UNIT
        );
        unsafe { check(grown, live) };
        assert_eq!(pool.used(), 3 * LARGE_UNIT);
        assert_eq!(pool.inner.frontier_bytes().1, 3 * LARGE_UNIT);
        unsafe { pool.deallocate(grown, three) };
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.inner.frontier_bytes().1, 0);
    }

    #[test]
    fn grow_failure_leaves_original_allocation_intact() {
        let pool = SlabPool::new(2 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let one = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let two = Layout::from_size_align(2 * LARGE_UNIT, 64).unwrap();
        let slab = Layout::from_size_align(60_000, 64).unwrap();
        let a = pool.allocate(one).unwrap(); // unit 1
        let s = pool.allocate(slab).unwrap(); // chunk at 0 blocks unit 0
        unsafe { fill(a, 1000) };

        assert!(unsafe { pool.grow(a, one, two, 1000) }.is_err());
        unsafe { check(a, 1000) };
        assert_eq!(pool.used(), LARGE_UNIT + SIZE_CLASSES[4]);
        unsafe {
            pool.deallocate(a, one);
            pool.deallocate(s, slab);
        }
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn grow_large_run_falls_back_to_fresh_run() {
        // No room around the run (nothing above, an occupied unit below),
        // but a fresh 2-unit run fits elsewhere: allocate + copy + free.
        let pool = SlabPool::new(4 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let one = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let two = Layout::from_size_align(2 * LARGE_UNIT, 64).unwrap();
        let a = pool.allocate(one).unwrap(); // unit 3
        let b = pool.allocate(one).unwrap(); // unit 2
        unsafe { fill(a, 8192) };

        let grown = unsafe { pool.grow(a, one, two, 8192) }.unwrap();
        assert_eq!(
            grown.as_ptr() as usize,
            pool.inner.base as usize,
            "units 0..2"
        );
        unsafe { check(grown, 8192) };
        assert_eq!(pool.used(), 3 * LARGE_UNIT, "old run freed");
        unsafe {
            pool.deallocate(grown, two);
            pool.deallocate(b, one);
        }
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.inner.frontier_bytes().1, 0);
    }

    #[test]
    fn grow_slab_block_into_large_run() {
        // 64 KiB-class block grown past the largest slab class: allocate a
        // run, copy, free the slab block back to class 4.
        let pool = SlabPool::new(4 * LARGE_UNIT, HugePages::Off, false).unwrap();
        let class4 = Layout::from_size_align(60_000, 64).unwrap();
        let unit = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let p = pool.allocate(class4).unwrap();
        unsafe { fill(p, 60_000) };

        let grown = unsafe { pool.grow(p, class4, unit, 60_000) }.unwrap();
        assert!((grown.as_ptr() as usize).is_multiple_of(LARGE_UNIT));
        unsafe { check(grown, 60_000) };
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), LARGE_UNIT, "slab block returned");
        unsafe { pool.deallocate(grown, unit) };
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_concurrent_grow_and_large_churn() {
        // Growers widen/shift runs while churners allocate and free large
        // and slab blocks against the same frontiers. Each grower checks its
        // own bytes after every grow; a failed grow must leave them intact.
        let pool = Arc::new(SlabPool::new(32 * LARGE_UNIT, HugePages::Off, false).unwrap());
        // std::thread::spawn + join, not thread::scope: scope returns when the
        // closures finish, before the workers' thread-local TCache destructors
        // run, so cached blocks could still be outstanding at the used() == 0
        // check below. join waits for full thread exit.
        let mut handles = Vec::new();
        {
            for t in 0..4u8 {
                let pool = Arc::clone(&pool);
                handles.push(std::thread::spawn(move || {
                    for round in 0..20 {
                        let mut layout = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
                        let Ok(mut ptr) = pool.allocate(layout) else {
                            continue;
                        };
                        let tag = t.wrapping_mul(31).wrapping_add(round);
                        unsafe { std::ptr::write_bytes(ptr.as_ptr(), tag, 4096) };
                        for k in 2..=4 {
                            let next = Layout::from_size_align(k * LARGE_UNIT, 64).unwrap();
                            // Err leaves the original run intact (checked below).
                            if let Ok(p) = unsafe { pool.grow(ptr, layout, next, 4096) } {
                                ptr = p;
                                layout = next;
                            }
                            for i in 0..4096 {
                                assert_eq!(unsafe { ptr.as_ptr().add(i).read() }, tag);
                            }
                        }
                        unsafe { pool.deallocate(ptr, layout) };
                    }
                }));
            }
            for _ in 0..4 {
                let pool = Arc::clone(&pool);
                handles.push(std::thread::spawn(move || {
                    let large = Layout::from_size_align(LARGE_UNIT + 1, 64).unwrap();
                    let small = Layout::from_size_align(4000, 64).unwrap();
                    for _ in 0..50 {
                        let l = pool.allocate(large).ok();
                        let sm = pool.allocate(small).ok();
                        unsafe {
                            if let Some(p) = l {
                                pool.deallocate(p, large);
                            }
                            if let Some(p) = sm {
                                pool.deallocate(p, small);
                            }
                        }
                    }
                }));
            }
        }
        for h in handles {
            h.join().unwrap();
        }
        pool.flush_caches();
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.inner.frontier_bytes().1, 0, "all runs released");
    }

    #[test]
    fn grow_doubling_column_reaches_whole_budget() {
        // A single column doubling from 1 MiB: with allocate-copy-free it
        // would stall below half the budget; widening in place reaches all
        // of it.
        let units = 8;
        let pool = SlabPool::new(units * LARGE_UNIT, HugePages::Off, false).unwrap();
        let mut layout = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
        let mut ptr = pool.allocate(layout).unwrap();
        unsafe { fill(ptr, 4096) };
        while layout.size() < units * LARGE_UNIT {
            let next = Layout::from_size_align(layout.size() * 2, 64).unwrap();
            ptr = unsafe { pool.grow(ptr, layout, next, 4096) }.unwrap();
            layout = next;
            unsafe { check(ptr, 4096) };
        }
        assert_eq!(pool.used(), units * LARGE_UNIT);
        unsafe { pool.deallocate(ptr, layout) };
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_construction_touches_no_blocks() {
        // Lazy carving: no chunk is threaded until the first allocation.
        let pool = SlabPool::new(16 * LARGE_UNIT, HugePages::Off, false).unwrap();
        assert_eq!(pool.inner.frontier_bytes(), (0, 0));
        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert_eq!(pool.inner.frontier_bytes(), (CHUNK_SIZE, 0));
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_window_is_unit_aligned_and_budget_sized() {
        let pool = SlabPool::new(3 * LARGE_UNIT + 12_345, HugePages::Off, false).unwrap();
        assert!((pool.inner.base as usize).is_multiple_of(WINDOW_ALIGN));
        assert_eq!(pool.capacity(), 3 * LARGE_UNIT + 12_345);
        // Bottom can use the sub-unit tail only while the top is empty.
        let class4 = Layout::from_size_align(60_000, 64).unwrap();
        let blocks = exhaust_frontier(&pool, class4);
        assert_eq!(blocks.len(), (3 * LARGE_UNIT + 12_345) / CHUNK_SIZE);
        for ptr in blocks {
            unsafe { pool.inner.global_deallocate(ptr, class4) };
        }
    }

    #[test]
    fn slab_pool_concurrent_frontier_collision_no_overlap() {
        // Threads race small (carving) and large (top) allocations to
        // exhaustion; every returned range must be disjoint and in-window.
        let pool = Arc::new(SlabPool::new(32 * LARGE_UNIT, HugePages::Off, false).unwrap());
        let small = Layout::from_size_align(200, 8).unwrap();
        let large = Layout::from_size_align(LARGE_UNIT + LARGE_UNIT / 2, 64).unwrap();

        let ranges: Vec<(usize, usize, bool)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|t| {
                    let pool = Arc::clone(&pool);
                    s.spawn(move || {
                        let mut out = Vec::new();
                        let mut failures = 0;
                        while failures < 2 {
                            let is_large = t % 2 == 0 && out.len() % 8 == 0;
                            let layout = if is_large { large } else { small };
                            match pool.allocate(layout) {
                                Ok(p) => {
                                    let len = if is_large {
                                        2 * LARGE_UNIT
                                    } else {
                                        SIZE_CLASSES[1]
                                    };
                                    out.push((p.as_ptr() as usize, len, is_large));
                                }
                                Err(_) => failures += 1,
                            }
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });

        let base = pool.inner.base as usize;
        let mut sorted: Vec<(usize, usize)> = ranges.iter().map(|&(a, l, _)| (a, l)).collect();
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "overlap: {:x?} vs {:x?}",
                w[0],
                w[1]
            );
        }
        let (first, last) = (sorted[0], *sorted.last().unwrap());
        assert!(first.0 >= base && last.0 + last.1 <= base + pool.capacity());

        for (addr, _, is_large) in ranges {
            let layout = if is_large { large } else { small };
            unsafe { pool.deallocate(NonNull::new(addr as *mut u8).unwrap(), layout) };
        }
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_concurrent_multi_class() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off, false).unwrap());
        let layout_small = Layout::from_size_align(64, 64).unwrap();
        let layout_large = Layout::from_size_align(4000, 8).unwrap();

        // Use std::thread::spawn so thread-local destructors run on join.
        let mut handles = Vec::new();
        // 4 threads on class 0.
        for _ in 0..4 {
            let pool = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                for _ in 0..200 {
                    let ptr = pool.allocate(layout_small).unwrap();
                    unsafe { pool.deallocate(ptr, layout_small) };
                }
            }));
        }
        // 4 threads on class 3.
        for _ in 0..4 {
            let pool = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    let ptr = pool.allocate(layout_large).unwrap();
                    unsafe { pool.deallocate(ptr, layout_large) };
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_small_budget_empty_classes() {
        // 128KB budget — class 5 (1MB) gets zero blocks.
        let pool = SlabPool::new(128 * 1024, HugePages::Off, false).unwrap();
        let layout_huge = Layout::from_size_align(1_000_000, 8).unwrap();
        assert!(pool.allocate(layout_huge).is_err());

        // Small classes should still work.
        let layout_small = Layout::from_size_align(32, 8).unwrap();
        let ptr = pool.allocate(layout_small).unwrap();
        unsafe { pool.deallocate(ptr, layout_small) };
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn slab_pool_multi_class_used_bytes_tracking() {
        // Test exact used_bytes tracking via global paths (bypassing TCache).
        let pool = SlabPool::new(8 * 1024 * 1024, HugePages::Off, false).unwrap();
        assert_eq!(pool.used(), 0);

        let l0 = Layout::from_size_align(32, 8).unwrap(); // class 0 (64B)
        let l2 = Layout::from_size_align(900, 8).unwrap(); // class 2 (1KB)
        let l4 = Layout::from_size_align(60_000, 8).unwrap(); // class 4 (64KB)

        let p0 = pool.inner.global_allocate(l0).unwrap();
        assert_eq!(pool.used(), SIZE_CLASSES[0]);

        let p2 = pool.inner.global_allocate(l2).unwrap();
        assert_eq!(pool.used(), SIZE_CLASSES[0] + SIZE_CLASSES[2]);

        let p4 = pool.inner.global_allocate(l4).unwrap();
        assert_eq!(
            pool.used(),
            SIZE_CLASSES[0] + SIZE_CLASSES[2] + SIZE_CLASSES[4]
        );

        unsafe { pool.inner.global_deallocate(p4, l4) };
        assert_eq!(pool.used(), SIZE_CLASSES[0] + SIZE_CLASSES[2]);

        unsafe { pool.inner.global_deallocate(p2, l2) };
        assert_eq!(pool.used(), SIZE_CLASSES[0]);

        unsafe { pool.inner.global_deallocate(p0, l0) };
        assert_eq!(pool.used(), 0);
    }

    // ── TCacheBin unit tests ────────────────────────────────────────

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

    // ── TCache behavior tests ───────────────────────────────────────

    #[test]
    fn tcache_refill_and_hit() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        // First allocation triggers refill (16 blocks from global).
        let p1 = pool.allocate(layout).unwrap();
        // After refill: 16 blocks charged to used_bytes (1 returned, 15 cached).
        assert_eq!(pool.used(), TCACHE_REFILL * SIZE_CLASSES[0]);

        // Next 15 allocations are TCache hits (no additional global allocs).
        let mut ptrs = vec![p1];
        for _ in 0..15 {
            ptrs.push(pool.allocate(layout).unwrap());
        }
        assert_eq!(ptrs.len(), 16);
        // Still 16 blocks charged — all came from the same refill batch.
        assert_eq!(pool.used(), TCACHE_REFILL * SIZE_CLASSES[0]);

        // Deallocate all — goes to TCache, not global.
        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout) };
        }
        // used_bytes still 16*64 — blocks are in TCache, not returned globally.
        assert_eq!(pool.used(), TCACHE_REFILL * SIZE_CLASSES[0]);

        // Flush returns all to global.
        pool.flush_caches();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn tcache_spill_on_full() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        // Allocate enough blocks to fill the TCache bin on dealloc.
        let mut ptrs = Vec::new();
        for _ in 0..=TCACHE_CAPACITY {
            ptrs.push(pool.allocate(layout).unwrap());
        }
        let used_after_alloc = pool.used();

        // Deallocate all — after 32 deallocs, a spill fires (16 blocks
        // returned to global). The 33rd dealloc also goes to TCache.
        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout) };
        }

        // Spill returned TCACHE_SPILL blocks to global — used_bytes decreased.
        let used_after_dealloc = pool.used();
        assert!(
            used_after_dealloc < used_after_alloc,
            "spill should have returned blocks to global: before={used_after_alloc}, after={used_after_dealloc}"
        );

        // Flush TCache to get accurate count, then verify all returned.
        pool.flush_caches();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn tcache_epoch_flush() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        // Allocate 5 blocks (refill grabs 16 from global, returns 1, caches 15).
        // After 5 allocs: 5 in user hands, 11 in TCache, 16 charged to used_bytes.
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            ptrs.push(pool.allocate(layout).unwrap());
        }
        let used_before_flush = pool.used();

        // flush_caches eagerly flushes calling thread's TCache (11 blocks back
        // to global), then bumps epoch.
        pool.flush_caches();
        let used_after_flush = pool.used();
        assert!(
            used_after_flush < used_before_flush,
            "flush should return TCache blocks: before={used_before_flush}, after={used_after_flush}"
        );
        // Only the 5 user-held blocks should remain.
        assert_eq!(used_after_flush, 5 * SIZE_CLASSES[0]);

        // Next allocate triggers refill (TCache was flushed).
        let p = pool.allocate(layout).unwrap();
        ptrs.push(p);

        // Clean up.
        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout) };
        }
        pool.flush_caches();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn tcache_cross_thread_dealloc() {
        let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap());
        let layout = Layout::from_size_align(64, 64).unwrap();

        // Thread A allocates.
        let ptrs: Vec<usize> = (0..16)
            .map(|_| pool.allocate(layout).unwrap().as_ptr() as usize)
            .collect();

        // Thread B deallocates.
        let pool2 = Arc::clone(&pool);
        std::thread::scope(|s| {
            s.spawn(move || {
                for addr in ptrs {
                    let ptr = NonNull::new(addr as *mut u8).unwrap();
                    unsafe { pool2.deallocate(ptr, layout) };
                }
            });
        });

        // Blocks are in thread B's TCache (thread exited -> TCache dropped -> flushed).
        // All blocks should be back in the global pool.
        // Allocate again to verify they're available.
        let p = pool.allocate(layout).unwrap();
        unsafe { pool.deallocate(p, layout) };
    }

    #[test]
    fn tcache_thread_exit_flushes() {
        let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap());
        let layout = Layout::from_size_align(64, 64).unwrap();

        let used_before = pool.used();

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
        assert_eq!(pool.used(), used_before);
    }

    #[test]
    fn tcache_refill_caches_only_its_own_class() {
        let pool = SlabPool::new(SMALL_POOL, HugePages::Off, false).unwrap();
        let class1 = Layout::from_size_align(200, 8).unwrap();
        let class0 = Layout::from_size_align(32, 8).unwrap();

        // Frontier full of class-1 chunks; free a few through the global
        // path so they sit on the class-1 free list.
        let mut medium = exhaust_frontier(&pool, class1);
        for ptr in medium.drain(..4) {
            unsafe { pool.inner.global_deallocate(ptr, class1) };
        }

        // Class 0 cannot be served at all; nothing lands in any bin.
        assert!(pool.allocate(class0).is_err());
        #[cfg(not(loom))]
        TCACHE.with(|cell| {
            let cache = unsafe { &*cell.get() }.as_ref().unwrap();
            assert!(cache.bins.iter().all(|b| b.count == 0));
        });

        // Class 1 refills from its own free list: 1 returned + 3 cached.
        let p = pool.allocate(class1).unwrap();
        #[cfg(not(loom))]
        TCACHE.with(|cell| {
            let cache = unsafe { &*cell.get() }.as_ref().unwrap();
            assert_eq!(cache.bins[1].count, 3);
            assert_eq!(cache.bins[0].count, 0);
        });
        medium.push(p);

        for ptr in medium {
            unsafe { pool.deallocate(ptr, class1) };
        }
        pool.flush_current_thread_cache();
        assert_eq!(pool.used(), 0);
    }

    // ── Concurrent TCache tests ─────────────────────────────────────

    #[test]
    fn tcache_concurrent_no_duplicates() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off, false).unwrap());
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

    #[test]
    fn tcache_flush_caches_via_world() {
        let mut world = crate::World::builder()
            .memory_budget(4 * 1024 * 1024)
            .build()
            .unwrap();

        // Spawn some entities to trigger pool allocations.
        for _ in 0..100 {
            world.spawn((0u32,));
        }

        let used_before = world.stats().pool_used;
        assert!(used_before > 0);

        // flush_pool_caches should eagerly flush the calling thread's TCache.
        world.flush_pool_caches();

        // used_bytes should have decreased (TCache blocks returned to global).
        let used_after = world.stats().pool_used;
        assert!(
            used_after <= used_before,
            "flush should not increase used: before={used_before}, after={used_after}"
        );
    }

    #[test]
    fn tcache_rapid_fill_spill_cycles() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        // Repeated fill-spill cycles to stress the shift-down compaction.
        for _ in 0..10 {
            let mut ptrs = Vec::new();
            for _ in 0..=TCACHE_CAPACITY {
                ptrs.push(pool.allocate(layout).unwrap());
            }
            // Deallocate all — triggers spill when bin fills.
            for ptr in ptrs {
                unsafe { pool.deallocate(ptr, layout) };
            }
        }

        // All blocks should be recoverable after flush.
        pool.flush_caches();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn tcache_epoch_flush_dealloc_cached_blocks() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        // Allocate blocks then deallocate (blocks enter TCache via dealloc path).
        let mut ptrs = Vec::new();
        for _ in 0..10 {
            ptrs.push(pool.allocate(layout).unwrap());
        }
        for ptr in ptrs {
            unsafe { pool.deallocate(ptr, layout) };
        }

        let used_before = pool.used();

        // Bump epoch — next op should flush dealloc-cached blocks too.
        pool.flush_caches();

        let used_after = pool.used();
        assert_eq!(
            used_after, 0,
            "epoch flush should return all blocks: used={used_after}"
        );
        assert!(used_after <= used_before);
    }

    #[test]
    fn tcache_thread_exit_returns_refilled_blocks() {
        // A thread refills its bin from free blocks of an exhausted pool and
        // exits; TCache::drop must return every cached block.
        let pool = Arc::new(SlabPool::new(SMALL_POOL, HugePages::Off, false).unwrap());
        let class1 = Layout::from_size_align(200, 8).unwrap();

        let mut burn = exhaust_frontier(&pool, class1);
        for ptr in burn.drain(..4) {
            unsafe { pool.inner.global_deallocate(ptr, class1) };
        }
        let used_before = pool.used();

        let pool2 = Arc::clone(&pool);
        std::thread::spawn(move || {
            let p = pool2.allocate(class1).unwrap();
            unsafe { pool2.deallocate(p, class1) };
        })
        .join()
        .unwrap();
        assert_eq!(pool.used(), used_before, "thread exit returned its bin");

        for ptr in burn {
            unsafe { pool.deallocate(ptr, class1) };
        }
        pool.flush_caches();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn tcache_epoch_flush_under_contention() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off, false).unwrap());
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
}

// ── Loom tests ─────────────────────────────────────────────────────

#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::thread;
    use std::alloc::Layout;

    #[test]
    fn loom_concurrent_pop_no_duplicates() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap());
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

    #[test]
    fn loom_concurrent_push_no_lost_blocks() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap());
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

            assert_eq!(pool.used(), 0, "blocks were lost during concurrent push");
        });
    }

    #[test]
    fn loom_push_pop_concurrent() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * 1024 * 1024, HugePages::Off, false).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            let ptr = pool.allocate(layout).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || unsafe { p1.deallocate(ptr, layout) });

            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(layout));

            t1.join().unwrap();
            let _ = t2.join().unwrap(); // may or may not succeed

            // Final state: either 0 or 1 block allocated.
            let used = pool.used();
            assert!(used == 0 || used == SIZE_CLASSES[0]);
        });
    }

    #[test]
    fn loom_concurrent_carve_no_duplicates() {
        // Both threads find the class-0 free list empty and race to carve
        // (or one carves and the other pops the remainder).
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(4 * LARGE_UNIT, HugePages::Off, false).unwrap());
            let layout = Layout::from_size_align(64, 64).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || p1.allocate(layout).unwrap());
            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(layout).unwrap());

            let (a, b) = (t1.join().unwrap(), t2.join().unwrap());
            assert_ne!(a, b, "two threads got the same block");
            assert_eq!(pool.used(), 2 * SIZE_CLASSES[0]);
        });
    }

    #[test]
    fn loom_carve_and_large_run_disjoint() {
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(2 * LARGE_UNIT, HugePages::Off, false).unwrap());
            let slab = Layout::from_size_align(60_000, 64).unwrap();
            let large = Layout::from_size_align(LARGE_UNIT, 64).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || p1.allocate(slab).unwrap().as_ptr() as usize);
            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(large).unwrap().as_ptr() as usize);

            let (s, l) = (t1.join().unwrap(), t2.join().unwrap());
            assert!(
                s + SIZE_CLASSES[4] <= l || l + LARGE_UNIT <= s,
                "chunk overlaps run"
            );
            assert_eq!(pool.used(), SIZE_CLASSES[4] + LARGE_UNIT);
        });
    }

    #[test]
    fn loom_grow_races_carve() {
        // A run at unit 1 grows into unit 0 while another thread carves the
        // first bottom chunk: exactly one of them can have unit 0.
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(2 * LARGE_UNIT, HugePages::Off, false).unwrap());
            let one = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
            let two = Layout::from_size_align(2 * LARGE_UNIT, 64).unwrap();
            let slab = Layout::from_size_align(60_000, 64).unwrap();
            let run = pool.allocate(one).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || unsafe { p1.grow(run, one, two, 64) }.is_ok());
            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(slab).is_ok());

            let (grew, carved) = (t1.join().unwrap(), t2.join().unwrap());
            assert!(grew ^ carved, "grew={grew} carved={carved}");
            let expected = if grew {
                2 * LARGE_UNIT
            } else {
                LARGE_UNIT + SIZE_CLASSES[4]
            };
            assert_eq!(pool.used(), expected);
        });
    }

    #[test]
    fn loom_grow_races_take_run() {
        // One free unit (unit 0, below the run at unit 1): a grow wants it
        // as shift-down room, a fresh 1-unit allocation wants it as a run.
        // Exactly one wins; the loser fails cleanly.
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(2 * LARGE_UNIT, HugePages::Off, false).unwrap());
            let one = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
            let two = Layout::from_size_align(2 * LARGE_UNIT, 64).unwrap();
            let run = pool.allocate(one).unwrap();

            let p1 = pool.clone();
            let t1 = thread::spawn(move || unsafe { p1.grow(run, one, two, 64) }.is_ok());
            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(one).is_ok());

            let (grew, took) = (t1.join().unwrap(), t2.join().unwrap());
            assert!(grew ^ took, "grew={grew} took={took}");
            assert_eq!(pool.used(), 2 * LARGE_UNIT);
        });
    }

    #[test]
    fn loom_large_free_races_carve() {
        // Pool full of two runs. Freeing the lower run retreats the top
        // frontier while another thread tries to carve: the carve either
        // fails (ran first) or lands inside the freed unit.
        loom::model(|| {
            let pool = Arc::new(SlabPool::new(2 * LARGE_UNIT, HugePages::Off, false).unwrap());
            let slab = Layout::from_size_align(60_000, 64).unwrap();
            let large = Layout::from_size_align(LARGE_UNIT, 64).unwrap();
            let high = pool.allocate(large).unwrap();
            let low = pool.allocate(large).unwrap();
            let low_addr = low.as_ptr() as usize;

            let p1 = pool.clone();
            let t1 = thread::spawn(move || unsafe { p1.deallocate(low, large) });
            let p2 = pool.clone();
            let t2 = thread::spawn(move || p2.allocate(slab).ok().map(|p| p.as_ptr() as usize));

            t1.join().unwrap();
            match t2.join().unwrap() {
                Some(addr) => {
                    // Carved after the free: inside the freed unit.
                    assert!(addr >= low_addr && addr + SIZE_CLASSES[4] <= low_addr + LARGE_UNIT);
                    assert_eq!(pool.used(), LARGE_UNIT + SIZE_CLASSES[4]);
                }
                None => assert_eq!(pool.used(), LARGE_UNIT),
            }
            unsafe { pool.deallocate(high, large) };
        });
    }
}
