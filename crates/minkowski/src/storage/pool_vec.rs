//! `PoolVec<T>`: a growable vector of `Copy` values stored in a `SlabPool`.
//!
//! Used for per-entity bookkeeping (entity generations, the free list,
//! `entity_locations`, archetype entity lists, sparse dense entity lists) so
//! it counts against the world's pool budget alongside component storage.
//! Dereferences to a slice, so reads look exactly like `Vec` reads.
//!
//! Growth comes in two flavours, matching the column failure-mode rule in
//! `docs/plans/pool-entity-bookkeeping.md`: `try_*` methods return
//! `PoolExhausted` and leave the vector unchanged; the plain methods abort via
//! `handle_alloc_error`, exactly like `BlobVec::push` on infallible paths.
//! Shrinking operations never allocate.

use std::alloc::Layout;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::pool::{PoolExhausted, SlabPool};
use crate::sync::Arc;

pub(crate) struct PoolVec<T: Copy> {
    data: NonNull<T>,
    len: usize,
    cap: usize,
    pool: Arc<SlabPool>,
}

// SAFETY: PoolVec owns its buffer exclusively (like Vec<T>); all mutation goes
// through &mut self. The pool handle is Send + Sync.
unsafe impl<T: Copy + Send> Send for PoolVec<T> {}
// SAFETY: &PoolVec only exposes &[T]; same reasoning as Vec<T>: Sync.
unsafe impl<T: Copy + Sync> Sync for PoolVec<T> {}

impl<T: Copy> PoolVec<T> {
    /// Empty vector bound to `pool`. Allocates nothing.
    pub(crate) fn new(pool: Arc<SlabPool>) -> Self {
        const { assert!(size_of::<T>() > 0, "PoolVec does not support ZSTs") };
        Self {
            data: NonNull::dangling(),
            len: 0,
            cap: 0,
            pool,
        }
    }

    /// Vector holding a copy of `items`. Aborts if the pool is exhausted.
    pub(crate) fn from_slice(pool: Arc<SlabPool>, items: &[T]) -> Self {
        let mut v = Self::new(pool);
        v.extend_from_slice(items);
        v
    }

    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    fn layout(cap: usize) -> Layout {
        Layout::array::<T>(cap).expect("PoolVec capacity overflow")
    }

    /// Ensure room for `additional` more elements. On `Err` nothing changed.
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), PoolExhausted> {
        let required = self
            .len
            .checked_add(additional)
            .expect("PoolVec capacity overflow");
        if required <= self.cap {
            return Ok(());
        }
        let new_cap = required.max(self.cap.saturating_mul(2)).max(4);
        let new_layout = Self::layout(new_cap);
        let new_data = if self.cap == 0 {
            self.pool.allocate(new_layout)?
        } else {
            // SAFETY: data was allocated from self.pool with layout(cap), and
            // len * size_of::<T>() <= the old allocation's size. On Ok the
            // old pointer is replaced below and never used again.
            unsafe {
                self.pool.grow(
                    self.data.cast(),
                    Self::layout(self.cap),
                    new_layout,
                    self.len * size_of::<T>(),
                )?
            }
        };
        self.data = new_data.cast();
        self.cap = new_cap;
        Ok(())
    }

    /// Like [`try_reserve`](Self::try_reserve), aborting on exhaustion.
    pub(crate) fn reserve(&mut self, additional: usize) {
        if let Err(e) = self.try_reserve(additional) {
            std::alloc::handle_alloc_error(e.requested);
        }
    }

    /// Append `value`. On `Err` nothing changed.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "fallible paths reserve, then push")
    )]
    pub(crate) fn try_push(&mut self, value: T) -> Result<(), PoolExhausted> {
        self.try_reserve(1)?;
        // SAFETY: len < cap after the reserve; the slot is in bounds.
        unsafe { self.data.as_ptr().add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }

    /// Append `value`, aborting on exhaustion.
    pub(crate) fn push(&mut self, value: T) {
        self.reserve(1);
        // SAFETY: as in try_push.
        unsafe { self.data.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    /// Resize to `new_len`, filling new slots with `value`. On `Err` nothing
    /// changed. Shrinking never allocates.
    pub(crate) fn try_resize(&mut self, new_len: usize, value: T) -> Result<(), PoolExhausted> {
        if new_len <= self.len {
            self.len = new_len;
            return Ok(());
        }
        self.try_reserve(new_len - self.len)?;
        for i in self.len..new_len {
            // SAFETY: i < new_len <= cap.
            unsafe { self.data.as_ptr().add(i).write(value) };
        }
        self.len = new_len;
        Ok(())
    }

    /// Like [`try_resize`](Self::try_resize), aborting on exhaustion.
    pub(crate) fn resize(&mut self, new_len: usize, value: T) {
        if let Err(e) = self.try_resize(new_len, value) {
            std::alloc::handle_alloc_error(e.requested);
        }
    }

    /// Append a copy of `items`, aborting on exhaustion.
    pub(crate) fn extend_from_slice(&mut self, items: &[T]) {
        self.reserve(items.len());
        // SAFETY: len + items.len() <= cap after the reserve; `items` cannot
        // alias our buffer (we hold &mut self).
        unsafe {
            std::ptr::copy_nonoverlapping(
                items.as_ptr(),
                self.data.as_ptr().add(self.len),
                items.len(),
            );
        }
        self.len += items.len();
    }

    /// Remove and return element `index`, replacing it with the last one.
    /// Panics if `index >= len`. Never allocates.
    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "swap_remove index {index} out of bounds (len {})",
            self.len
        );
        let value = self[index];
        let last = self[self.len - 1];
        self[index] = last;
        self.len -= 1;
        value
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        let value = *self.last()?;
        self.len -= 1;
        Some(value)
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Keep only elements for which `keep` returns true, in order. Never
    /// allocates.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut write = 0;
        for read in 0..self.len {
            let v = self[read];
            if keep(&v) {
                self[write] = v;
                write += 1;
            }
        }
        self.len = write;
    }
}

impl<T: Copy> Deref for PoolVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: data is valid for len initialized elements (or dangling
        // with len == 0).
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for PoolVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in deref; &mut self gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl<'a, T: Copy> IntoIterator for &'a PoolVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Copy> IntoIterator for &'a mut PoolVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Copy> Drop for PoolVec<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            // SAFETY: data was allocated from self.pool with layout(cap) and is
            // not used after this point. T: Copy, so no element drop glue.
            unsafe {
                self.pool
                    .deallocate(self.data.cast(), Self::layout(self.cap));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::HugePages;

    fn pool(bytes: usize) -> Arc<SlabPool> {
        Arc::new(SlabPool::new(bytes, HugePages::Off, false).unwrap())
    }

    #[test]
    fn push_index_and_slice_reads() {
        let mut v = PoolVec::new(pool(1 << 20));
        for i in 0u32..1000 {
            v.push(i * 3);
        }
        assert_eq!(v.len(), 1000);
        assert_eq!(v[999], 2997);
        assert_eq!(v.iter().copied().sum::<u32>(), 3 * (999 * 1000 / 2));
    }

    #[test]
    fn grows_past_one_mebibyte_through_pool_grow() {
        let p = pool(8 << 20);
        let mut v = PoolVec::new(Arc::clone(&p));
        let n = (2 << 20) / 8 + 1; // > 2 MiB of u64
        for i in 0..n as u64 {
            v.push(i);
        }
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u64));
        assert!(p.used() >= n * 8);
        drop(v);
        p.flush_caches();
        assert_eq!(p.used(), 0, "drop returns the buffer");
    }

    #[test]
    fn failed_growth_leaves_vector_unchanged() {
        // Two 64 KiB chunks of budget: the vector can't grow past it.
        let p = pool(2 * 65_536);
        let mut v = PoolVec::new(p);
        while v.try_push(7u64).is_ok() {}
        let (len, cap) = (v.len(), v.capacity());
        assert!(v.try_push(8).is_err());
        assert!(v.try_reserve(1).is_err());
        assert!(v.try_resize(len + 10, 0).is_err());
        assert_eq!((v.len(), v.capacity()), (len, cap));
        assert!(v.iter().all(|&x| x == 7));
        // Shrinking still works with the pool exhausted.
        assert!(v.try_resize(1, 0).is_ok());
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn shrinking_ops_never_allocate() {
        let mut v = PoolVec::from_slice(pool(1 << 20), &[1u32, 2, 3, 4, 5, 6]);
        let cap = v.capacity();
        assert_eq!(v.swap_remove(1), 2);
        assert_eq!(&v[..], &[1, 6, 3, 4, 5]);
        v.retain(|&x| x % 2 == 1);
        assert_eq!(&v[..], &[1, 3, 5]);
        assert_eq!(v.pop(), Some(5));
        v.truncate(1);
        assert_eq!(&v[..], &[1]);
        v.truncate(0);
        assert!(v.is_empty());
        assert_eq!(v.capacity(), cap);
    }

    #[test]
    fn resize_fills_and_extend_appends() {
        let mut v = PoolVec::new(pool(1 << 20));
        v.resize(5, Some(9u16));
        v.extend_from_slice(&[None, Some(1)]);
        assert_eq!(
            &v[..],
            &[Some(9), Some(9), Some(9), Some(9), Some(9), None, Some(1)]
        );
        v.resize(2, None);
        assert_eq!(&v[..], &[Some(9), Some(9)]);
    }
}
