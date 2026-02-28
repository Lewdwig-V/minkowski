use std::alloc::Layout;

/// Contiguous byte arena for component data. Mutations store integer offsets
/// into this arena, avoiding per-mutation heap allocation.
#[allow(dead_code)]
pub(crate) struct Arena {
    data: Vec<u8>,
}

#[allow(dead_code)]
impl Arena {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Copy `layout.size()` bytes from `src` into the arena.
    /// Returns the byte offset where data was written.
    pub fn alloc(&mut self, src: *const u8, layout: Layout) -> usize {
        if layout.size() == 0 {
            return 0;
        }
        let align = layout.align();
        let offset = (self.data.len() + align - 1) & !(align - 1);
        self.data.resize(offset + layout.size(), 0);
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.data.as_mut_ptr().add(offset), layout.size());
        }
        offset
    }

    /// Get a raw pointer to data at the given offset.
    pub fn get(&self, offset: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(offset) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_read_back() {
        let mut arena = Arena::new();
        let value: u32 = 42;
        let layout = Layout::new::<u32>();
        let offset = arena.alloc(&value as *const u32 as *const u8, layout);
        let ptr = arena.get(offset) as *const u32;
        assert_eq!(unsafe { *ptr }, 42);
    }

    #[test]
    fn arena_alignment() {
        let mut arena = Arena::new();
        let byte: u8 = 0xFF;
        let _ = arena.alloc(&byte as *const u8, Layout::new::<u8>());

        let val: u64 = 123456789;
        let offset = arena.alloc(&val as *const u64 as *const u8, Layout::new::<u64>());
        assert_eq!(offset % 8, 0, "u64 offset must be 8-byte aligned");
    }

    #[test]
    fn arena_zst() {
        let mut arena = Arena::new();
        let layout = Layout::new::<()>();
        let offset = arena.alloc(std::ptr::null(), layout);
        assert_eq!(offset, 0);
    }
}
