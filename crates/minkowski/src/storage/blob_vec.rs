use std::alloc::Layout;
use std::ptr::NonNull;

use crate::pool::{PoolExhausted, SharedPool};
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
    pool: SharedPool,
}

// Safety: BlobVec stores Component data which requires Send + Sync.
unsafe impl Send for BlobVec {}
unsafe impl Sync for BlobVec {}

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

    /// Creates a new `BlobVec` for items with the given layout and optional drop function.
    pub fn new(
        item_layout: Layout,
        drop_fn: Option<unsafe fn(*mut u8)>,
        capacity: usize,
        pool: SharedPool,
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
            pool,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg_attr(not(test), expect(dead_code))]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
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
        let dst = self.ptr_at(self.len);
        let size = self.item_layout.size();
        if size > 0 {
            // SAFETY: caller guarantees ptr is valid for size bytes; dst is within allocated capacity
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst, size) };
        }
        self.len += 1;
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
        self.ptr_at(row)
    }

    /// Removes the element at `row` by swapping it with the last element,
    /// then dropping the removed element.
    ///
    /// # Safety
    /// `row` must be in bounds (`row < len`).
    pub unsafe fn swap_remove(&mut self, row: usize) {
        debug_assert!(row < self.len);
        let last = self.len - 1;
        let size = self.item_layout.size();
        if row != last && size > 0 {
            let row_ptr = self.ptr_at(row);
            if let Some(drop_fn) = self.drop_fn {
                // SAFETY: row_ptr is valid; drop_fn was set at construction for this type
                unsafe { drop_fn(row_ptr) };
            }
            let last_ptr = self.ptr_at(last);
            // SAFETY: last_ptr and row_ptr are non-overlapping valid pointers within allocation
            unsafe { std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size) };
        } else if let Some(drop_fn) = self.drop_fn {
            // SAFETY: ptr_at returns valid pointer; drop_fn was set at construction for this type
            unsafe { drop_fn(self.ptr_at(row)) };
        }
        self.len -= 1;
    }

    /// Removes the element at `row` by swapping it with the last element.
    /// The removed element is written to `ptr` instead of being dropped.
    ///
    /// # Safety
    /// `row` must be in bounds. `ptr` must be valid for writes of `item_layout.size()` bytes.
    #[cfg_attr(not(test), expect(dead_code))]
    pub unsafe fn swap_remove_unchecked(&mut self, row: usize, ptr: *mut u8) {
        debug_assert!(row < self.len);
        let last = self.len - 1;
        let size = self.item_layout.size();
        let row_ptr = self.ptr_at(row);
        if size > 0 {
            // SAFETY: row_ptr is valid for size bytes; ptr is valid per caller guarantee
            unsafe { std::ptr::copy_nonoverlapping(row_ptr, ptr, size) };
            // Move last into the gap (if not same row)
            if row != last {
                let last_ptr = self.ptr_at(last);
                // SAFETY: last_ptr and row_ptr are non-overlapping valid pointers
                unsafe { std::ptr::copy_nonoverlapping(last_ptr, row_ptr, size) };
            }
        }
        self.len -= 1;
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
        }
        self.len -= 1;
    }

    /// Drop the element at `row` in place without moving anything.
    /// The slot becomes logically uninitialized — caller must not read it
    /// or must overwrite it before any future access.
    ///
    /// # Safety
    /// `row` must be in bounds (`row < len`). Caller must ensure the slot
    /// is not accessed again without being reinitialized.
    pub unsafe fn drop_in_place(&mut self, row: usize) {
        debug_assert!(row < self.len);
        if let Some(drop_fn) = self.drop_fn {
            // SAFETY: ptr_at returns valid pointer; drop_fn was set at construction for this type
            unsafe { drop_fn(self.ptr_at(row)) };
        }
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

    /// Like [`push`] but returns `Err(PoolExhausted)` instead of panicking
    /// when the pool cannot grow to accommodate the new element.
    ///
    /// # Safety
    /// `ptr` must point to a valid, initialized value matching this BlobVec's layout.
    /// Caller is responsible for not double-dropping the source value.
    #[expect(dead_code)]
    pub(crate) unsafe fn try_push(&mut self, ptr: *mut u8) -> Result<(), PoolExhausted> {
        if self.len == self.capacity {
            self.try_grow()?;
        }
        let dst = self.ptr_at(self.len);
        let size = self.item_layout.size();
        if size > 0 {
            // SAFETY: caller guarantees ptr is valid for size bytes; dst is within allocated capacity
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst, size) };
        }
        self.len += 1;
        Ok(())
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

    unsafe fn drop_ptr<T>(ptr: *mut u8) {
        unsafe { std::ptr::drop_in_place(ptr as *mut T) };
    }

    fn bv_for<T>() -> BlobVec {
        let drop_fn = if std::mem::needs_drop::<T>() {
            Some(drop_ptr::<T> as unsafe fn(*mut u8))
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
        assert!(bv.is_empty());
    }

    #[test]
    fn push_increments_len() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 42u32);
        }
        assert_eq!(bv.len(), 1);
        assert!(!bv.is_empty());
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
    fn swap_remove_unchecked_returns_removed() {
        let mut bv = bv_for::<u64>();
        unsafe {
            push_val(&mut bv, 111u64);
            push_val(&mut bv, 222u64);
            let mut out: u64 = 0;
            bv.swap_remove_unchecked(0, &mut out as *mut u64 as *mut u8);
            assert_eq!(out, 111);
            assert_eq!(bv.len(), 1);
            assert_eq!(read_val::<u64>(&bv, 0), 222);
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

    #[test]
    fn drop_in_place_calls_drop_fn() {
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
        let mut bv = bv_for::<Tracked>();
        unsafe {
            push_val(&mut bv, Tracked(1));
            push_val(&mut bv, Tracked(2));
            push_val(&mut bv, Tracked(3));
            bv.drop_in_place(1); // drop middle element only
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(bv.len(), 3); // len unchanged — caller manages it

        // Prevent BlobVec::drop from double-dropping the already-dropped slot.
        // In real usage the caller would copy_unchecked + set_len to skip it.
        // For this test: copy last into slot 1, then set_len to 2.
        unsafe {
            bv.copy_unchecked(2, 1);
            bv.set_len(2);
        }
        // BlobVec::drop will now drop 2 remaining Tracked values
    }
}
