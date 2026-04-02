/// Per-column bitset tracking which pages were mutated since last flush.
///
/// A "page" is a fixed-size group of rows within a [`BlobVec`](super::blob_vec::BlobVec)
/// column. The tracker maintains one bit per page — set means at least one row
/// in that page was written since the last [`clear`](DirtyPageTracker::clear).
///
/// Used by the persistence layer to flush only modified pages instead of
/// snapshotting entire columns.
pub struct DirtyPageTracker {
    /// Packed bitset: bit `N` of word `N / 64` tracks page `N`.
    bits: Vec<u64>,
}

/// Number of rows per page. Each page maps to one bit in the tracker.
///
/// 256 rows ≈ 1 KiB for `u32` components, 2 KiB for `f64` — reasonable I/O
/// units for incremental persistence without excessive bitset overhead.
pub const PAGE_SIZE: usize = 256;

/// Number of pages tracked per `u64` word.
const PAGES_PER_WORD: usize = 64;

impl DirtyPageTracker {
    /// Create a new tracker with no dirty pages.
    pub fn new() -> Self {
        Self { bits: Vec::new() }
    }

    /// Mark the page containing `row` as dirty.
    #[inline]
    pub fn mark_row(&mut self, row: usize) {
        let page = row / PAGE_SIZE;
        self.mark_page(page);
    }

    /// Mark a specific page as dirty.
    #[inline]
    pub fn mark_page(&mut self, page: usize) {
        let word = page / PAGES_PER_WORD;
        let bit = page % PAGES_PER_WORD;
        if word >= self.bits.len() {
            self.bits.resize(word + 1, 0);
        }
        self.bits[word] |= 1u64 << bit;
    }

    /// Mark all pages that overlap `[start_row, end_row)` as dirty.
    pub fn mark_row_range(&mut self, start_row: usize, end_row: usize) {
        if start_row >= end_row {
            return;
        }
        let first_page = start_row / PAGE_SIZE;
        // end_row is exclusive, so the last touched row is end_row - 1.
        let last_page = (end_row - 1) / PAGE_SIZE;
        for page in first_page..=last_page {
            self.mark_page(page);
        }
    }

    /// Returns true if `page` has been mutated since the last clear.
    #[inline]
    pub fn is_dirty(&self, page: usize) -> bool {
        let word = page / PAGES_PER_WORD;
        let bit = page % PAGES_PER_WORD;
        self.bits.get(word).is_some_and(|w| w & (1u64 << bit) != 0)
    }

    /// Returns true if any page is dirty.
    pub fn any_dirty(&self) -> bool {
        self.bits.iter().any(|&w| w != 0)
    }

    /// Total number of dirty pages.
    pub fn dirty_count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate over all dirty page indices.
    pub fn dirty_pages(&self) -> DirtyPageIter<'_> {
        DirtyPageIter {
            bits: &self.bits,
            word_idx: 0,
            current_word: self.bits.first().copied().unwrap_or(0),
        }
    }

    /// Clear all dirty bits. Called after a successful flush.
    pub fn clear(&mut self) {
        for w in &mut self.bits {
            *w = 0;
        }
    }

    /// Convert a page index to the row range it covers: `[start, end)`.
    /// The caller must clamp `end` to the actual column length.
    #[inline]
    pub fn page_row_range(page: usize) -> (usize, usize) {
        let start = page * PAGE_SIZE;
        (start, start + PAGE_SIZE)
    }

    /// Number of pages needed to cover `row_count` rows.
    #[inline]
    pub fn pages_for_rows(row_count: usize) -> usize {
        row_count.div_ceil(PAGE_SIZE)
    }
}

impl Default for DirtyPageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over dirty page indices, yielded in ascending order.
pub struct DirtyPageIter<'a> {
    bits: &'a [u64],
    word_idx: usize,
    current_word: u64,
}

impl Iterator for DirtyPageIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                // Clear the lowest set bit.
                self.current_word &= self.current_word - 1;
                return Some(self.word_idx * PAGES_PER_WORD + bit);
            }
            self.word_idx += 1;
            if self.word_idx >= self.bits.len() {
                return None;
            }
            self.current_word = self.bits[self.word_idx];
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining: u32 = self.current_word.count_ones()
            + self.bits[self.word_idx.saturating_add(1)..]
                .iter()
                .map(|w| w.count_ones())
                .sum::<u32>();
        let r = remaining as usize;
        (r, Some(r))
    }
}

impl ExactSizeIterator for DirtyPageIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_no_dirty_pages() {
        let t = DirtyPageTracker::new();
        assert!(!t.any_dirty());
        assert_eq!(t.dirty_count(), 0);
        assert_eq!(t.dirty_pages().count(), 0);
    }

    #[test]
    fn mark_row_sets_correct_page() {
        let mut t = DirtyPageTracker::new();
        // Row 0 → page 0
        t.mark_row(0);
        assert!(t.is_dirty(0));
        assert!(!t.is_dirty(1));

        // Row 255 → still page 0
        t.mark_row(PAGE_SIZE - 1);
        assert!(t.is_dirty(0));

        // Row 256 → page 1
        t.mark_row(PAGE_SIZE);
        assert!(t.is_dirty(1));
    }

    #[test]
    fn mark_page_directly() {
        let mut t = DirtyPageTracker::new();
        t.mark_page(5);
        assert!(t.is_dirty(5));
        assert!(!t.is_dirty(4));
        assert!(!t.is_dirty(6));
    }

    #[test]
    fn mark_row_range_single_page() {
        let mut t = DirtyPageTracker::new();
        t.mark_row_range(10, 20);
        assert!(t.is_dirty(0));
        assert_eq!(t.dirty_count(), 1);
    }

    #[test]
    fn mark_row_range_spans_pages() {
        let mut t = DirtyPageTracker::new();
        // Range crosses from page 0 into page 1
        t.mark_row_range(PAGE_SIZE - 5, PAGE_SIZE + 5);
        assert!(t.is_dirty(0));
        assert!(t.is_dirty(1));
        assert_eq!(t.dirty_count(), 2);
    }

    #[test]
    fn mark_row_range_empty_is_noop() {
        let mut t = DirtyPageTracker::new();
        t.mark_row_range(10, 10);
        assert!(!t.any_dirty());
        t.mark_row_range(10, 5);
        assert!(!t.any_dirty());
    }

    #[test]
    fn clear_resets_all() {
        let mut t = DirtyPageTracker::new();
        t.mark_page(0);
        t.mark_page(3);
        t.mark_page(100);
        assert_eq!(t.dirty_count(), 3);
        t.clear();
        assert!(!t.any_dirty());
        assert_eq!(t.dirty_count(), 0);
    }

    #[test]
    fn dirty_pages_iter_ascending() {
        let mut t = DirtyPageTracker::new();
        t.mark_page(7);
        t.mark_page(2);
        t.mark_page(65); // second word
        t.mark_page(0);
        let pages: Vec<usize> = t.dirty_pages().collect();
        assert_eq!(pages, vec![0, 2, 7, 65]);
    }

    #[test]
    fn dirty_pages_iter_exact_size() {
        let mut t = DirtyPageTracker::new();
        t.mark_page(1);
        t.mark_page(3);
        t.mark_page(130);
        let iter = t.dirty_pages();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn page_row_range_covers_correct_rows() {
        let (start, end) = DirtyPageTracker::page_row_range(0);
        assert_eq!(start, 0);
        assert_eq!(end, PAGE_SIZE);

        let (start, end) = DirtyPageTracker::page_row_range(3);
        assert_eq!(start, 3 * PAGE_SIZE);
        assert_eq!(end, 4 * PAGE_SIZE);
    }

    #[test]
    fn pages_for_rows_rounds_up() {
        assert_eq!(DirtyPageTracker::pages_for_rows(0), 0);
        assert_eq!(DirtyPageTracker::pages_for_rows(1), 1);
        assert_eq!(DirtyPageTracker::pages_for_rows(PAGE_SIZE), 1);
        assert_eq!(DirtyPageTracker::pages_for_rows(PAGE_SIZE + 1), 2);
    }

    #[test]
    fn high_page_index() {
        let mut t = DirtyPageTracker::new();
        // Page 1000 — forces multiple words
        t.mark_page(1000);
        assert!(t.is_dirty(1000));
        assert!(!t.is_dirty(999));
        assert_eq!(t.dirty_count(), 1);
        let pages: Vec<usize> = t.dirty_pages().collect();
        assert_eq!(pages, vec![1000]);
    }

    #[test]
    fn is_dirty_out_of_range_is_false() {
        let t = DirtyPageTracker::new();
        assert!(!t.is_dirty(9999));
    }

    #[test]
    fn mark_same_page_twice_is_idempotent() {
        let mut t = DirtyPageTracker::new();
        t.mark_page(5);
        t.mark_page(5);
        assert_eq!(t.dirty_count(), 1);
    }
}
