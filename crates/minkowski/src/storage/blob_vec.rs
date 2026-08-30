use std::alloc::Layout;
use std::ptr::NonNull;

use crate::sync::Arc;

use crate::pool::{PoolExhausted, SlabPool};
use crate::storage::dirty_pages::DirtyPageTracker;
use crate::tick::Tick;

/// Type-erased growable array. Stores raw bytes with a known `Layout`.
/// Used as the column storage inside archetypes.
pub(crate) struct BlobVec {
    pub(crate) item_layout: Layout,
    pub(crate) drop_fn: Option<unsafe fn(*mut u8)>,
    data: NonNull<u8>,
    len: usize,
    capacity: usize,
    pub(crate) changed_tick: Tick,
    pub(crate) dirty_pages: DirtyPageTracker,
    pool: Arc<SlabPool>,
}

// Safety: BlobVec stores Component data which requires Send + Sync.
unsafe impl Send for BlobVec {}
unsafe impl Sync for BlobVec {}

/// RAII scratch slot carved from the component pool. Lets callers reserve
/// overwrite capacity up front (multi-component overwrites preflight all
/// demand before mutating anything) and guarantees the slot returns to the
/// pool on every exit path, including a panicking component `Drop`.
pub(crate) struct PoolScratch {
    pool: Arc<SlabPool>,
    ptr: Option<NonNull<u8>>,
    layout: Layout,
}

impl PoolScratch {
    fn zst() -> Self {
        Self {
            pool: default_pool_shim(),
            ptr: None,
            layout: Layout::new::<()>(),
        }
    }

    fn as_ptr(&mut self) -> *mut u8 {
        match self.ptr {
            Some(p) => p.as_ptr(),
            None => std::ptr::null_mut(), // ZST: never dereferenced
        }
    }

    fn take_ptr(&mut self) -> Option<NonNull<u8>> {
        self.ptr.take()
    }
}

impl Drop for PoolScratch {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            unsafe { self.pool.deallocate(ptr, self.layout) };
        }
    }
}

// Placeholder so `PoolScratch::zst()` needs no pool handle; ZST scratches are
// never deallocated (ptr is None).
fn default_pool_shim() -> Arc<SlabPool> {
    crate::pool::default_pool()
}

/// A removed value evacuated to scratch, awaiting destruction. Produced by
/// [`BlobVec::begin_extract_swap`] and consumed by
/// [`BlobVec::commit_extracted_swap`] + [`ExtractedSlot::drop_value`].
pub(crate) struct ExtractedSlot {
    pool: Arc<SlabPool>,
    ptr: Option<NonNull<u8>>,
    layout: Layout,
    drop_fn: Option<unsafe fn(*mut u8)>,
}

impl ExtractedSlot {
    /// Run the stolen value's destructor. May panic — the caller must have
    /// already committed all structural state so the unwind sees a consistent
    /// world (leaked component internals are the accepted worst case).
    pub(crate) fn drop_value(&mut self) {
        if let (Some(drop_fn), Some(ptr)) = (self.drop_fn, self.ptr) {
            // SAFETY: ptr holds the only moved-out copy of the value.
            unsafe { drop_fn(ptr.as_ptr()) };
        }
    }
}

impl Drop for ExtractedSlot {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            unsafe { self.pool.deallocate(ptr, self.layout) };
        }
    }
}

impl BlobVec {
    /// Minimum allocation alignment for all BlobVec columns.
    /// 64 bytes = cache line on x86-64 and Apple Silicon.
    const MIN_COLUMN_ALIGN: usize = 64;

    /// Compute the allocation alignment for a BlobVec column.
    fn alloc_align(item: &Layout) -> usize {
        item.align().max(Self::MIN_COLUMN_ALIGN)
    }

    /// Mark this column as changed at the given tick.
    #[inline]
    pub(crate) fn mark_changed(&mut self, tick: Tick) {
        self.changed_tick = tick;
    }

    /// Mark the page containing `row` as dirty without advancing the column tick.
    ///
    /// Use this on write paths that call [`mark_changed`] once for a batch and
    /// then write individual rows via [`get_ptr`]. The tick is already set for
    /// the whole column; this records per-page granularity for incremental flush.
    #[inline]
    pub(crate) fn mark_row_dirty(&mut self, row: usize) {
        self.dirty_pages.mark_row(row);
    }

    /// Creates a new `BlobVec` for items with the given layout and optional drop function.
    pub fn new(
        item_layout: Layout,
        drop_fn: Option<unsafe fn(*mut u8)>,
        capacity: usize,
        pool: Arc<SlabPool>,
    ) -> Self {
        let (data, capacity) = if item_layout.size() == 0 {
            (NonNull::dangling(), usize::MAX)
        } else if capacity == 0 {
            (NonNull::dangling(), 0)
        } else {
            let layout = Layout::from_size_align(
                item_layout.size() * capacity,
                Self::alloc_align(&item_layout),
            )
            .expect("invalid layout");
            let data = pool
                .allocate(layout)
                .unwrap_or_else(|_| std::alloc::handle_alloc_error(layout));
            (data, capacity)
        };
        Self {
            item_layout,
            drop_fn,
            data,
            len: 0,
            capacity,
            changed_tick: Tick::default(),
            dirty_pages: DirtyPageTracker::new(),
            pool,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a raw pointer to the first byte of the allocation.
    ///
    /// For zero-sized components, returns a dangling pointer — callers must
    /// guard on `item_layout.size() == 0` before performing pointer arithmetic.
    #[inline]
    pub(crate) fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Ensures the column has capacity for at least `additional` more elements.
    /// If the column already has enough spare capacity, this is a no-op.
    pub(crate) fn reserve(&mut self, additional: usize) {
        let required = self.len + additional;
        if required <= self.capacity {
            return;
        }
        let size = self.item_layout.size();
        if size == 0 {
            return;
        }
        // Grow to at least the required capacity, doubling as needed.
        let mut new_capacity = if self.capacity == 0 { 4 } else { self.capacity };
        while new_capacity < required {
            new_capacity = new_capacity.checked_mul(2).expect("capacity overflow");
        }
        let new_layout = Layout::from_size_align(
            size.checked_mul(new_capacity).expect("capacity overflow"),
            Self::alloc_align(&self.item_layout),
        )
        .expect("invalid layout");

        let new_data = self
            .pool
            .allocate(new_layout)
            .unwrap_or_else(|_| std::alloc::handle_alloc_error(new_layout));
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    new_data.as_ptr(),
                    size * self.len,
                );
                let old_layout = Layout::from_size_align(
                    size * self.capacity,
                    Self::alloc_align(&self.item_layout),
                )
                .unwrap();
                self.pool.deallocate(self.data, old_layout);
            }
        }
        self.data = new_data;
        self.capacity = new_capacity;
    }

    /// Pushes a value by copying `item_layout.size()` bytes from `ptr`.
    ///
    /// # Safety
    /// `ptr` must point to a valid, initialized value matching this BlobVec's layout.
    /// Caller is responsible for not double-dropping the source value.
    pub unsafe fn push(&mut self, ptr: *mut u8) {
        if self.len == self.capacity {
            self.grow();
        }
        let row = self.len;
        let dst = self.ptr_at(row);
        let size = self.item_layout.size();
        if size > 0 {
            // SAFETY: caller guarantees ptr is valid for size bytes; dst is within allocated capacity
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst, size) };
        }
        self.len += 1;
        self.dirty_pages.mark_row(row);
    }

    /// Bulk-append `count` items by copying `bytes` (== `count * item_layout.size()`)
    /// to the end of the column in a single memcpy. For zero-sized types only the
    /// logical length advances; no bytes are copied.
    ///
    /// Unlike [`push`], this does NOT mark pages dirty or advance the changed
    /// tick: it is a recovery primitive that reconstructs already-committed state,
    /// which must not be re-flushed.
    ///
    /// # Safety
    /// - `bytes.len()` must equal `count * item_layout.size()`.
    /// - `bytes` must be a valid native (in-memory) representation of `count`
    ///   consecutive items of this column's type, and ownership of those items
    ///   is moved into this column (which holds the type's `drop_fn`). For POD
    ///   types this is trivial; for heap types the bytes carry reconstructed
    ///   values whose ownership transfers here, so each is dropped exactly once
    ///   when the column drops or the row is removed.
    pub(crate) unsafe fn append_bytes_unchecked(&mut self, bytes: &[u8], count: usize) {
        let size = self.item_layout.size();
        if size == 0 {
            self.len += count;
            return;
        }
        assert_eq!(
            bytes.len(),
            count * size,
            "append_bytes_unchecked: byte length must equal count * item_size"
        );
        self.reserve(count);
        let dst = self.ptr_at(self.len);
        // SAFETY: `reserve` guarantees capacity for `count` more items starting at
        // `dst`; `bytes` is valid for `bytes.len()` reads; src and dst are distinct
        // allocations so the copy is non-overlapping.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, count * size) };
        self.len += count;
    }

    /// Returns a raw pointer to the element at `row`.
    ///
    /// # Change detection invariant
    /// This returns `*mut u8` for internal mechanics (migration, reverse capture)
    /// but **does not mark the column changed**. Writing through this pointer
    /// bypasses change detection — `Changed<T>` queries will miss the mutation.
    ///
    /// For mutable access that respects change detection, use [`get_ptr_mut`]
    /// or ensure the caller marks the column via [`mark_changed`] or the
    /// entry-point methods (`query_table_mut`, `World::query` for `&mut T`).
    ///
    /// # Safety
    /// `row` must be in bounds (`row < len`).
    #[inline]
    pub unsafe fn get_ptr(&self, row: usize) -> *mut u8 {
        debug_assert!(row < self.len);
        self.ptr_at(row)
    }

    /// Returns a raw pointer to the element at `row` and marks the column
    /// changed at the given tick.
    ///
    /// This is the correct write-path accessor — use this (or ensure the
    /// caller marks via entry-point methods) for any mutation that should
    /// be visible to `Changed<T>` queries.
    ///
    /// # Safety
    /// `row` must be in bounds (`row < len`).
    #[inline]
    pub unsafe fn get_ptr_mut(&mut self, row: usize, tick: Tick) -> *mut u8 {
        debug_assert!(row < self.len);
        self.changed_tick = tick;
        self.dirty_pages.mark_row(row);
        self.ptr_at(row)
    }

    /// Removes the element at `row` by swapping it with the last element,
    /// then dropping the removed element.
    ///
    /// # Panic safety
    /// Structural mutation (byte-copy + length decrement) is committed by a
    /// drop guard REGARDLESS of whether the removed value's destructor panics.
    /// A panicking `Drop` therefore costs at most a leak of the partially
    /// dropped value — never a double-free, never a dangling slot readable
    /// through `0..len`. (Std `Vec` drain-guard pattern, type-erased.)
    ///
    /// # Safety
    /// `row` must be in bounds (`row < len`).
    /// Phase A of panic-safe removal: evacuate the value at `row` into a
    /// pool scratch slot WITHOUT mutating the column. The only fallible step;
    /// on `Err(PoolExhausted)` the column is untouched. Pair with
    /// [`commit_extracted_swap`](Self::commit_extracted_swap) (structural
    /// removal) and [`ExtractedSlot::drop_value`] (destruction).
    pub(crate) fn begin_extract_swap(
        &mut self,
        row: usize,
    ) -> Result<ExtractedSlot, PoolExhausted> {
        debug_assert!(row < self.len);
        let size = self.item_layout.size();
        if self.drop_fn.is_none() || size == 0 {
            // POD / ZST: nothing to evacuate; destructor-free path.
            return Ok(ExtractedSlot {
                pool: self.pool.clone(),
                ptr: None,
                layout: Layout::new::<()>(),
                drop_fn: None,
            });
        }
        let mut scratch = self.alloc_scratch()?;
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr_at(row), scratch.as_ptr(), size);
        }
        let ptr = scratch
            .take_ptr()
            .expect("non-ZST scratch always has a pointer");
        Ok(ExtractedSlot {
            pool: self.pool.clone(),
            ptr: Some(ptr),
            layout: Layout::from_size_align(size, self.item_layout.align())
                .expect("valid layout from component layout"),
            drop_fn: self.drop_fn,
        })
    }

    /// Phase B: structurally remove `row` (tail copy + shrink). Infallible.
    /// The removed value's bytes are already evacuated — see
    /// [`begin_extract_swap`](Self::begin_extract_swap).
    pub(crate) unsafe fn commit_extracted_swap(&mut self, row: usize) {
        debug_assert!(row < self.len);
        let last = self.len - 1;
        let size = self.item_layout.size();
        if row != last && size > 0 {
            let row_ptr = self.ptr_at(row);
            let last_ptr = self.ptr_at(last);
            // SAFETY: in-allocation, non-overlapping.
            unsafe { std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size) };
            self.dirty_pages.mark_row(row);
        }
        self.len -= 1;
    }

    pub unsafe fn swap_remove(&mut self, row: usize) {
        debug_assert!(row < self.len);
        let row_ptr = self.ptr_at(row);
        let drop_fn = self.drop_fn;
        // Committed on all exits (including unwind): overwrite row with the
        // tail element and shrink the logical length. Runs before this frame
        // unwinds past `self`, so BlobVec::drop and any later accessor see a
        // fully consistent column.
        struct RemoveGuard<'a> {
            bv: &'a mut BlobVec,
            row: usize,
        }
        impl Drop for RemoveGuard<'_> {
            fn drop(&mut self) {
                let last = self.bv.len - 1;
                let size = self.bv.item_layout.size();
                if self.row != last && size > 0 {
                    let row_ptr = self.bv.ptr_at(self.row);
                    let last_ptr = self.bv.ptr_at(last);
                    // SAFETY: row_ptr/last_ptr are in-allocation, non-overlapping.
                    unsafe { std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size) };
                    self.bv.dirty_pages.mark_row(self.row);
                }
                self.bv.len -= 1;
            }
        }
        let _guard = RemoveGuard { bv: self, row };
        if let Some(drop_fn) = drop_fn {
            // SAFETY: drop_fn matches this BlobVec's component type; row_ptr is
            // the only live copy of the removed value at this point.
            unsafe { drop_fn(row_ptr) };
        }
    }

    /// Removes the element at `row` by swapping with the last element.
    /// Neither drops the removed element nor copies it out.
    /// Used during archetype migration where data is moved via get_ptr + push.
    ///
    /// # Safety
    /// `row` must be in bounds. Caller must have already moved/copied the data.
    pub unsafe fn swap_remove_no_drop(&mut self, row: usize) {
        debug_assert!(row < self.len);
        let last = self.len - 1;
        let size = self.item_layout.size();
        if row != last && size > 0 {
            let row_ptr = self.ptr_at(row);
            let last_ptr = self.ptr_at(last);
            // SAFETY: last_ptr and row_ptr are non-overlapping valid pointers within allocation
            unsafe { std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size) };
            self.dirty_pages.mark_row(row);
        }
        self.len -= 1;
    }

    /// Overwrite the value at `row` with the bytes at `src`, dropping the old
    /// value. Panic-safe ordering for droppable types:
    ///
    /// 1. Steal the OLD value's bytes into a pool-allocated scratch slot,
    /// 2. write the NEW value into the live slot (structural commit),
    /// 3. run `drop_fn` on the stolen bytes and release the scratch.
    ///
    /// If the component's `Drop` panics, the live slot holds the new (valid)
    /// value and the leaked worst case is half-dropped scratch bytes — never
    /// freed bytes readable through `get`/query.
    ///
    /// Returns `Err(PoolExhausted)` if the scratch slot cannot be allocated;
    /// on error the column is UNCHANGED.
    ///
    /// # Safety
    /// `row` must be in bounds. `src` must point to a valid initialized value
    /// of this BlobVec's layout; caller must not double-drop the source value.
    pub(crate) unsafe fn replace_protected(
        &mut self,
        row: usize,
        src: *const u8,
    ) -> Result<(), PoolExhausted> {
        let scratch = self.alloc_scratch()?;
        // SAFETY: row < len per debug_assert; src valid per caller contract;
        // scratch came from this column's pool sized for this layout.
        unsafe { self.replace_protected_with(row, src, scratch) };
        Ok(())
    }

    /// Allocate a pool-backed scratch slot for [`replace_protected_with`].
    /// Used by callers that must reserve overwrite capacity for several
    /// components up front so a multi-component overwrite either fully
    /// commits or leaves state untouched.
    pub(crate) fn alloc_scratch(&self) -> Result<PoolScratch, PoolExhausted> {
        let size = self.item_layout.size();
        if size == 0 {
            return Ok(PoolScratch::zst());
        }
        let layout = Layout::from_size_align(size, self.item_layout.align())
            .expect("valid layout from component layout");
        let ptr = self.pool.allocate(layout)?;
        Ok(PoolScratch {
            pool: self.pool.clone(),
            ptr: Some(ptr),
            layout,
        })
    }

    /// Overwrite the value at `row` with the bytes at `src`, dropping the old
    /// value into `scratch`. Infallible — allocation happened earlier via
    /// [`alloc_scratch`], which lets multi-component overwrites preflight all
    /// scratch demand before mutating anything.
    ///
    /// Panic-safe ordering: steal old bytes → commit new bytes → drop stolen.
    /// If the component's `Drop` panics, the live slot holds the new (valid)
    /// value; worst case is half-dropped scratch bytes.
    ///
    /// # Safety
    /// `row` must be in bounds. `src` must point to a valid initialized value
    /// of this BlobVec's layout; caller must not double-drop the source value.
    /// The scratch must come from this column's pool (`alloc_scratch`) and be
    /// sized for this layout.
    pub(crate) unsafe fn replace_protected_with(
        &mut self,
        row: usize,
        src: *const u8,
        mut scratch: PoolScratch,
    ) {
        debug_assert!(row < self.len);
        let size = self.item_layout.size();
        let dst = self.ptr_at(row);
        if self.drop_fn.is_none() || size == 0 {
            unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(dst, scratch.as_ptr(), size);
            std::ptr::copy_nonoverlapping(src, dst, size);
            // SAFETY: scratch holds the only moved-out copy of the old value;
            // drop_fn matches this column's component type.
            (self.drop_fn.unwrap_unchecked())(scratch.as_ptr());
        }
        // Scratch deallocation happens in PoolScratch::drop.
    }

    /// Copy element from `src_row` to `dst_row` without dropping either.
    /// Bitwise copy — no drop on dst (must be uninitialized or already dropped),
    /// no drop on src (caller ensures it won't be accessed again).
    ///
    /// # Safety
    /// Both rows must be in bounds. `dst_row` must be uninitialized or already
    /// dropped. `src_row` data becomes logically moved.
    pub unsafe fn copy_unchecked(&mut self, src_row: usize, dst_row: usize) {
        debug_assert!(src_row < self.len);
        debug_assert!(dst_row < self.len);
        let size = self.item_layout.size();
        if size > 0 {
            let src = self.ptr_at(src_row);
            let dst = self.ptr_at(dst_row);
            // SAFETY: src and dst are valid pointers within allocation; caller guarantees non-overlap semantics
            unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
            self.dirty_pages.mark_row(dst_row);
        }
    }

    /// Set the length directly. Caller must ensure all elements in
    /// `new_len..old_len` have been dropped or moved out.
    ///
    /// # Safety
    /// `new_len` must be <= current len. Elements beyond new_len must be
    /// already dropped/moved.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.len);
        self.len = new_len;
    }

    #[inline]
    fn ptr_at(&self, index: usize) -> *mut u8 {
        if self.item_layout.size() == 0 {
            NonNull::dangling().as_ptr()
        } else {
            // <= because push() writes at index == len (within allocated capacity).
            // Read-path callers (get_ptr, get_ptr_mut) have their own index < len checks.
            debug_assert!(
                index <= self.len,
                "BlobVec::ptr_at out of bounds: index {index}, len {}",
                self.len
            );
            unsafe { self.data.as_ptr().add(index * self.item_layout.size()) }
        }
    }

    fn grow(&mut self) {
        let size = self.item_layout.size();
        if size == 0 {
            return;
        }
        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };
        let new_layout = Layout::from_size_align(
            size.checked_mul(new_capacity).expect("capacity overflow"),
            Self::alloc_align(&self.item_layout),
        )
        .expect("invalid layout");

        // Always use alloc + copy + dealloc instead of realloc.
        // realloc may not preserve alignment > max_align_t (typically 16 bytes),
        // and we require 64-byte alignment for cache line / SIMD guarantees.
        //
        // Check the allocation result BEFORE copying — alloc can return null
        // under memory pressure, and copy_nonoverlapping on a null dst is UB.
        let new_data = self
            .pool
            .allocate(new_layout)
            .unwrap_or_else(|_| std::alloc::handle_alloc_error(new_layout));
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    new_data.as_ptr(),
                    size * self.len,
                );
                let old_layout = Layout::from_size_align(
                    size * self.capacity,
                    Self::alloc_align(&self.item_layout),
                )
                .unwrap();
                self.pool.deallocate(self.data, old_layout);
            }
        }
        self.data = new_data;
        self.capacity = new_capacity;
    }

    /// Like [`grow`] but returns `Err(PoolExhausted)` instead of panicking
    /// when the pool cannot satisfy the allocation.
    pub(crate) fn try_grow(&mut self) -> Result<(), PoolExhausted> {
        let size = self.item_layout.size();
        if size == 0 {
            return Ok(());
        }
        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };
        let new_layout = Layout::from_size_align(
            size.checked_mul(new_capacity).expect("capacity overflow"),
            Self::alloc_align(&self.item_layout),
        )
        .expect("invalid layout");

        let new_data = self.pool.allocate(new_layout)?;
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    new_data.as_ptr(),
                    size * self.len,
                );
                let old_layout = Layout::from_size_align(
                    size * self.capacity,
                    Self::alloc_align(&self.item_layout),
                )
                .unwrap();
                self.pool.deallocate(self.data, old_layout);
            }
        }
        self.data = new_data;
        self.capacity = new_capacity;
        Ok(())
    }
}

impl Drop for BlobVec {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.drop_fn {
            for i in 0..self.len {
                unsafe {
                    drop_fn(self.ptr_at(i));
                }
            }
        }
        let size = self.item_layout.size();
        if size > 0 && self.capacity > 0 {
            let layout =
                Layout::from_size_align(size * self.capacity, Self::alloc_align(&self.item_layout))
                    .unwrap();
            unsafe {
                self.pool.deallocate(self.data, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::default_pool;
    use std::alloc::Layout;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ── helpers ──────────────────────────────────────────────

    /// Push a typed value into a BlobVec, forgetting the original.
    unsafe fn push_val<T>(bv: &mut BlobVec, mut val: T) {
        let ptr = &mut val as *mut T as *mut u8;
        unsafe { bv.push(ptr) };
        std::mem::forget(val);
    }

    /// Read a typed value from a BlobVec row.
    unsafe fn read_val<T: Copy>(bv: &BlobVec, row: usize) -> T {
        let ptr = unsafe { bv.get_ptr(row) } as *const T;
        unsafe { *ptr }
    }

    fn bv_for<T>() -> BlobVec {
        let drop_fn = if std::mem::needs_drop::<T>() {
            Some(crate::component::drop_ptr::<T> as unsafe fn(*mut u8))
        } else {
            None
        };
        BlobVec::new(Layout::new::<T>(), drop_fn, 0, default_pool())
    }

    // ── tests ───────────────────────────────────────────────

    #[test]
    fn new_is_empty() {
        let bv = BlobVec::new(Layout::new::<u32>(), None, 0, default_pool());
        assert_eq!(bv.len(), 0);
    }

    #[test]
    fn push_increments_len() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 42u32);
        }
        assert_eq!(bv.len(), 1);
    }

    #[test]
    fn push_and_read_back() {
        let mut bv = bv_for::<u64>();
        unsafe {
            push_val(&mut bv, 100u64);
            push_val(&mut bv, 200u64);
            push_val(&mut bv, 300u64);
            assert_eq!(read_val::<u64>(&bv, 0), 100);
            assert_eq!(read_val::<u64>(&bv, 1), 200);
            assert_eq!(read_val::<u64>(&bv, 2), 300);
        }
    }

    #[test]
    fn push_triggers_growth() {
        let mut bv = bv_for::<u32>();
        // Push enough to force multiple reallocations
        for i in 0u32..256 {
            unsafe {
                push_val(&mut bv, i);
            }
        }
        assert_eq!(bv.len(), 256);
        unsafe {
            for i in 0u32..256 {
                assert_eq!(read_val::<u32>(&bv, i as usize), i);
            }
        }
    }

    #[test]
    fn swap_remove_last_element() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            bv.swap_remove(0);
        }
        assert_eq!(bv.len(), 0);
    }

    #[test]
    fn swap_remove_swaps_with_last() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
            push_val(&mut bv, 30u32);
            // Remove row 0 — last element (30) moves to row 0
            bv.swap_remove(0);
        }
        assert_eq!(bv.len(), 2);
        unsafe {
            assert_eq!(read_val::<u32>(&bv, 0), 30);
            assert_eq!(read_val::<u32>(&bv, 1), 20);
        }
    }

    #[test]
    fn drop_calls_drop_fn_for_all_elements() {
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        #[expect(dead_code)]
        struct Tracked(u32);
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let mut bv = bv_for::<Tracked>();
            unsafe {
                push_val(&mut bv, Tracked(1));
                push_val(&mut bv, Tracked(2));
                push_val(&mut bv, Tracked(3));
            }
            // bv drops here
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn swap_remove_drops_removed_element() {
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        #[expect(dead_code)]
        struct Tracked(u32);
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let mut bv = bv_for::<Tracked>();
            unsafe {
                push_val(&mut bv, Tracked(1));
                push_val(&mut bv, Tracked(2));
                bv.swap_remove(0);
            }
            // 1 drop from swap_remove
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
            // bv drops here — 1 more (the remaining Tracked(2))
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn zst_push_and_len() {
        // Zero-sized types should track length but not allocate
        let mut bv = BlobVec::new(Layout::new::<()>(), None, 0, default_pool());
        unsafe {
            let mut unit = ();
            bv.push(&mut unit as *mut () as *mut u8);
            bv.push(&mut unit as *mut () as *mut u8);
        }
        assert_eq!(bv.len(), 2);
    }

    #[test]
    fn column_base_is_64_byte_aligned() {
        for &(size, align) in &[(4, 4), (8, 8), (1, 1), (12, 4), (32, 16)] {
            let layout = Layout::from_size_align(size, align).unwrap();
            let mut bv = BlobVec::new(layout, None, 8, default_pool());
            unsafe {
                let mut val = vec![0u8; size];
                bv.push(val.as_mut_ptr());
            }
            let base = unsafe { bv.get_ptr(0) } as usize;
            assert_eq!(
                base % 64,
                0,
                "BlobVec base not 64-byte aligned for size={size}, align={align}, base={base:#x}"
            );
        }
    }

    #[test]
    fn initial_capacity() {
        let mut bv = BlobVec::new(Layout::new::<u32>(), None, 16, default_pool());
        // Should not reallocate for the first 16 pushes
        for i in 0u32..16 {
            unsafe {
                push_val(&mut bv, i);
            }
        }
        assert_eq!(bv.len(), 16);
        unsafe {
            assert_eq!(read_val::<u32>(&bv, 0), 0);
            assert_eq!(read_val::<u32>(&bv, 15), 15);
        }
    }

    #[test]
    fn changed_tick_default_and_mark() {
        use crate::tick::Tick;
        let mut bv = bv_for::<u32>();
        assert_eq!(bv.changed_tick, Tick::default());
        bv.mark_changed(Tick::new(42));
        assert_eq!(bv.changed_tick, Tick::new(42));
    }

    #[test]
    fn copy_unchecked_moves_data() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
            push_val(&mut bv, 30u32);
            bv.copy_unchecked(2, 0); // copy row 2 into row 0
            assert_eq!(read_val::<u32>(&bv, 0), 30);
            assert_eq!(read_val::<u32>(&bv, 1), 20);
            assert_eq!(read_val::<u32>(&bv, 2), 30); // src still has data
        }
    }

    #[test]
    fn set_len_truncates() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
            push_val(&mut bv, 30u32);
            bv.set_len(1);
        }
        assert_eq!(bv.len(), 1);
        unsafe {
            assert_eq!(read_val::<u32>(&bv, 0), 10);
        }
    }

    // ── Dirty page tracking ──────────────────────────────────────

    #[test]
    fn push_marks_page_dirty() {
        let mut bv = bv_for::<u32>();
        assert!(!bv.dirty_pages.any_dirty());
        unsafe { push_val(&mut bv, 42u32) };
        assert!(bv.dirty_pages.is_dirty(0));
        assert_eq!(bv.dirty_pages.dirty_count(), 1);
    }

    #[test]
    fn get_ptr_mut_marks_page_dirty() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
        }
        bv.dirty_pages.clear();

        unsafe { bv.get_ptr_mut(1, Tick::new(1)) };
        assert!(bv.dirty_pages.is_dirty(0)); // row 1 → page 0
    }

    #[test]
    fn swap_remove_marks_destination_page_dirty() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
            push_val(&mut bv, 30u32);
        }
        bv.dirty_pages.clear();

        // Remove row 0 — row 2 swaps into row 0
        unsafe { bv.swap_remove(0) };
        assert!(bv.dirty_pages.is_dirty(0));
    }

    #[test]
    fn swap_remove_last_does_not_mark_dirty() {
        let mut bv = bv_for::<u32>();
        unsafe { push_val(&mut bv, 10u32) };
        bv.dirty_pages.clear();

        // Remove the only element — no swap, just truncate
        unsafe { bv.swap_remove(0) };
        assert!(!bv.dirty_pages.any_dirty());
    }

    #[test]
    fn swap_remove_no_drop_marks_dirty() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
        }
        bv.dirty_pages.clear();

        unsafe { bv.swap_remove_no_drop(0) };
        assert!(bv.dirty_pages.is_dirty(0));
    }

    #[test]
    fn copy_unchecked_marks_dst_dirty() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
        }
        bv.dirty_pages.clear();

        unsafe { bv.copy_unchecked(1, 0) };
        assert!(bv.dirty_pages.is_dirty(0));
    }

    #[test]
    fn mark_row_dirty_without_tick() {
        let mut bv = bv_for::<u32>();
        unsafe { push_val(&mut bv, 42u32) };
        bv.dirty_pages.clear();

        bv.mark_row_dirty(0);
        assert!(bv.dirty_pages.is_dirty(0));
        // Tick should not have changed
        assert_eq!(bv.changed_tick, Tick::default());
    }

    #[test]
    fn clear_dirty_pages_resets() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
        }
        assert!(bv.dirty_pages.any_dirty());
        bv.dirty_pages.clear();
        assert!(!bv.dirty_pages.any_dirty());
    }

    #[test]
    fn append_bytes_unchecked_bulk_copies() {
        let mut bv = bv_for::<u32>();
        let src = [10u32, 20, 30];
        let bytes = unsafe {
            std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(&src))
        };
        unsafe { bv.append_bytes_unchecked(bytes, 3) };
        assert_eq!(bv.len(), 3);
        unsafe {
            assert_eq!(read_val::<u32>(&bv, 0), 10);
            assert_eq!(read_val::<u32>(&bv, 1), 20);
            assert_eq!(read_val::<u32>(&bv, 2), 30);
        }
    }

    #[test]
    fn append_bytes_unchecked_appends_after_existing() {
        let mut bv = bv_for::<u32>();
        unsafe { push_val(&mut bv, 1u32) };
        let src = [2u32, 3];
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, 8) };
        unsafe { bv.append_bytes_unchecked(bytes, 2) };
        assert_eq!(bv.len(), 3);
        unsafe {
            assert_eq!(read_val::<u32>(&bv, 0), 1);
            assert_eq!(read_val::<u32>(&bv, 1), 2);
            assert_eq!(read_val::<u32>(&bv, 2), 3);
        }
    }

    #[test]
    fn append_bytes_unchecked_zst_advances_len() {
        let mut bv = BlobVec::new(Layout::new::<()>(), None, 0, default_pool());
        unsafe { bv.append_bytes_unchecked(&[], 5) };
        assert_eq!(bv.len(), 5);
    }

    #[test]
    fn append_bytes_unchecked_does_not_mark_dirty() {
        let mut bv = bv_for::<u32>();
        let src = [7u32];
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, 4) };
        unsafe { bv.append_bytes_unchecked(bytes, 1) };
        assert!(
            !bv.dirty_pages.any_dirty(),
            "recovery append must not dirty pages"
        );
        assert_eq!(
            bv.changed_tick,
            crate::tick::Tick::default(),
            "recovery append must not advance changed_tick"
        );
    }
}
