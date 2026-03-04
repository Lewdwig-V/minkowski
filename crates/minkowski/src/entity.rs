/// A unique entity identifier: 32-bit index + 32-bit generation packed into u64.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Entity(u64);

impl Entity {
    pub const DANGLING: Entity = Entity(u64::MAX);

    #[inline]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self((generation as u64) << 32 | index as u64)
    }

    #[inline]
    pub fn index(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Convert to raw u64 for serialization (preserves generation + index).
    #[inline]
    pub fn to_bits(self) -> u64 {
        self.0
    }

    /// Reconstruct from raw u64. The caller must ensure the bits represent
    /// a valid entity (correct generation for the target world).
    #[inline]
    pub fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

/// Allocates and recycles entity IDs with generational tracking.
///
/// Supports two allocation modes:
/// - `alloc(&mut self)`: standard allocation, recycles from free list.
/// - `reserve(&self)`: lock-free atomic allocation of fresh indices.
///   Returns Entity with generation 0. Reserved indices are NOT in the
///   generations vec yet — call `materialize_reserved()` before using
///   `alloc()` or `is_alive()` on reserved indices.
pub(crate) struct EntityAllocator {
    pub(crate) generations: Vec<u32>,
    pub(crate) free_list: Vec<u32>,
    /// Atomic counter for lock-free entity index reservation.
    next_reserved: std::sync::atomic::AtomicU32,
}

impl EntityAllocator {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
            next_reserved: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Reserve a fresh entity index atomically (`&self`, no `&mut` needed).
    /// Returns Entity with generation 0. Reserved entities are NOT in the
    /// generations vec yet — call `materialize_reserved()` from `&mut self`
    /// before using `alloc()` or `is_alive()` on reserved indices.
    #[allow(dead_code)]
    pub fn reserve(&self) -> Entity {
        let index = self
            .next_reserved
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Entity::new(index, 0)
    }

    /// Sync the atomic counter to at least `generations.len()`.
    /// Called after snapshot restore to prevent `reserve()` from
    /// handing out already-used indices.
    pub fn sync_reserved(&mut self) {
        let len = self.generations.len() as u32;
        let current = *self.next_reserved.get_mut();
        if current < len {
            *self.next_reserved.get_mut() = len;
        }
    }

    /// Backfill the generations vec to cover all reserved indices.
    /// Called automatically by `alloc()`.
    pub fn materialize_reserved(&mut self) {
        let reserved = *self.next_reserved.get_mut();
        while self.generations.len() < reserved as usize {
            self.generations.push(0);
        }
    }

    pub fn alloc(&mut self) -> Entity {
        self.materialize_reserved();
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize];
            Entity::new(index, gen)
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.next_reserved
                .store(index + 1, std::sync::atomic::Ordering::Relaxed);
            Entity::new(index, 0)
        }
    }

    pub fn dealloc(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx < self.generations.len() && self.generations[idx] == entity.generation() {
            self.generations[idx] = self.generations[idx].wrapping_add(1);
            self.free_list.push(entity.index());
            true
        } else {
            false
        }
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.generations.len() && self.generations[idx] == entity.generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_bit_packing() {
        let e = Entity::new(42, 7);
        assert_eq!(e.index(), 42);
        assert_eq!(e.generation(), 7);
    }

    #[test]
    fn entity_max_values() {
        let e = Entity::new(u32::MAX, u32::MAX);
        assert_eq!(e.index(), u32::MAX);
        assert_eq!(e.generation(), u32::MAX);
    }

    #[test]
    fn entity_equality() {
        let a = Entity::new(1, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(1, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn allocator_basic() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.alloc();
        let e2 = alloc.alloc();
        assert_eq!(e1.index(), 0);
        assert_eq!(e1.generation(), 0);
        assert_eq!(e2.index(), 1);
        assert_eq!(e2.generation(), 0);
        assert!(alloc.is_alive(e1));
        assert!(alloc.is_alive(e2));
    }

    #[test]
    fn allocator_recycle() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.alloc();
        assert!(alloc.dealloc(e1));
        let e2 = alloc.alloc();
        assert_eq!(e2.index(), 0);
        assert_eq!(e2.generation(), 1);
        assert!(!alloc.is_alive(e1));
        assert!(alloc.is_alive(e2));
    }

    #[test]
    fn allocator_double_dealloc() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.alloc();
        assert!(alloc.dealloc(e));
        assert!(!alloc.dealloc(e));
    }

    #[test]
    fn entity_to_from_bits_round_trip() {
        let e = Entity::new(42, 7);
        let bits = e.to_bits();
        let e2 = Entity::from_bits(bits);
        assert_eq!(e, e2);
        assert_eq!(e2.index(), 42);
        assert_eq!(e2.generation(), 7);
    }

    // ── Reserve tests ────────────────────────────────────────────

    #[test]
    fn reserve_basic() {
        let alloc = EntityAllocator::new();
        let e1 = alloc.reserve();
        let e2 = alloc.reserve();
        assert_eq!(e1.index(), 0);
        assert_eq!(e1.generation(), 0);
        assert_eq!(e2.index(), 1);
        assert_eq!(e2.generation(), 0);
    }

    #[test]
    fn reserve_concurrent() {
        use std::sync::Arc;
        let alloc = Arc::new(EntityAllocator::new());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let alloc = alloc.clone();
                std::thread::spawn(move || (0..100).map(|_| alloc.reserve()).collect::<Vec<_>>())
            })
            .collect();
        let mut all_entities: Vec<Entity> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all_entities.sort_by_key(|e| e.index());
        all_entities.dedup_by_key(|e| e.index());
        assert_eq!(all_entities.len(), 400, "no duplicate indices");
    }

    #[test]
    fn reserve_then_alloc_no_overlap() {
        let mut alloc = EntityAllocator::new();
        let r1 = alloc.reserve();
        let r2 = alloc.reserve();
        alloc.materialize_reserved();
        let a1 = alloc.alloc();
        assert_eq!(r1.index(), 0);
        assert_eq!(r2.index(), 1);
        assert_eq!(a1.index(), 2);
    }

    #[test]
    fn alloc_after_reserve_recycles_free_list() {
        let mut alloc = EntityAllocator::new();
        // alloc two entities, dealloc one
        let e0 = alloc.alloc();
        let _e1 = alloc.alloc();
        alloc.dealloc(e0);

        // reserve should get a fresh index (not from free list)
        let r = alloc.reserve();
        assert_eq!(r.index(), 2);

        // alloc should still recycle from free list
        let a = alloc.alloc();
        assert_eq!(a.index(), 0);
        assert_eq!(a.generation(), 1);
    }
}
