use crate::sync::{AtomicU32, Ordering};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Ordered free slots with a lazy index for replay's arbitrary removals.
/// Tombstones avoid shifting the tail; snapshots expose only live entries.
#[derive(Default)]
pub(crate) struct FreeList {
    slots: Vec<u32>,
    positions: Option<HashMap<u32, usize>>,
    holes: usize,
    snapshot: OnceLock<Vec<u32>>,
}

impl From<Vec<u32>> for FreeList {
    fn from(slots: Vec<u32>) -> Self {
        Self {
            slots,
            ..Self::default()
        }
    }
}

impl FreeList {
    // The allocator never issues u32::MAX (Entity::DANGLING's index).
    const REMOVED: u32 = u32::MAX;

    pub(crate) fn len(&self) -> usize {
        self.slots.len() - self.holes
    }

    pub(crate) fn as_slice(&self) -> &[u32] {
        if self.holes == 0 {
            &self.slots
        } else {
            self.snapshot.get_or_init(|| {
                self.slots
                    .iter()
                    .copied()
                    .filter(|&i| i != Self::REMOVED)
                    .collect()
            })
        }
    }

    fn push(&mut self, index: u32) {
        self.snapshot.take();
        if let Some(positions) = &mut self.positions {
            positions.insert(index, self.slots.len());
        }
        self.slots.push(index);
    }

    fn extend(&mut self, indices: impl Iterator<Item = u32>) {
        for index in indices {
            self.push(index);
        }
    }

    fn pop(&mut self) -> Option<u32> {
        self.snapshot.take();
        while let Some(index) = self.slots.pop() {
            if index == Self::REMOVED {
                self.holes -= 1;
                continue;
            }
            if let Some(positions) = &mut self.positions {
                positions.remove(&index);
            }
            return Some(index);
        }
        self.positions = None;
        None
    }

    fn remove(&mut self, index: u32) -> bool {
        if self.slots.last() == Some(&index) {
            self.pop();
            return true;
        }
        let positions = self.positions.get_or_insert_with(|| {
            self.slots
                .iter()
                .enumerate()
                .map(|(pos, &i)| (i, pos))
                .collect()
        });
        let Some(pos) = positions.remove(&index) else {
            return false;
        };
        self.snapshot.take();
        self.slots[pos] = Self::REMOVED;
        self.holes += 1;
        // Rebuild only after enough removals to amortize the linear scan.
        if self.holes > self.slots.len() / 2 {
            self.slots.retain(|&i| i != Self::REMOVED);
            self.holes = 0;
            *positions = self
                .slots
                .iter()
                .enumerate()
                .map(|(pos, &i)| (i, pos))
                .collect();
        }
        true
    }
}

/// A unique entity identifier: 32-bit index + 32-bit generation packed into a u64.
///
/// The low 32 bits store the index (slot in the entity allocator), and the
/// high 32 bits store the generation (incremented each time the slot is
/// recycled). This makes stale handles detectable: after an entity is
/// despawned and its slot reused, the old handle's generation no longer
/// matches, so [`World::is_alive`](crate::World::is_alive) returns false.
///
/// [`Entity::DANGLING`] is a sentinel value (`u64::MAX`) used as a
/// placeholder before a real entity is assigned. Use [`to_bits`](Entity::to_bits)
/// / [`from_bits`](Entity::from_bits) for serialization.
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
///
/// # Pool threading status
///
/// The generations and free-slot storage use the standard system
/// allocator. Pool-backed allocation is **deferred to v2**:
///
/// - `generations: Vec<u32>` — one `u32` per entity index ever created.
///   Grows monotonically, never shrinks. Cold-path metadata accessed
///   only during alloc/dealloc/is_alive.
/// - `free_list: FreeList` — ordered recycled indices, with a lazy lookup
///   index for replay. Ordinary allocation uses its Vec tail directly.
///
/// Together these account for negligible memory relative to the BlobVec
/// columns that store actual component data (>95% of total). Converting
/// to pool-backed storage would require a pool-aware Vec type, which is
/// significant additional machinery for minimal memory accounting benefit.
pub(crate) struct EntityAllocator {
    pub(crate) generations: Vec<u32>,
    pub(crate) free_list: FreeList,
    // PERF: Padding isolates the atomic from Vec fields on separate cache
    // lines. Prevents false sharing when concurrent spawners (via reserve())
    // contend with sequential alloc()/materialize() which mutate the Vecs.
    _pad: [u8; 64],
    /// Atomic counter for lock-free entity index reservation.
    next_reserved: AtomicU32,
    /// Monotonic spawn counter for observability (single u64 increment, negligible overhead).
    pub(crate) total_spawns: u64,
    /// Monotonic despawn counter for observability (single u64 increment, negligible overhead).
    pub(crate) total_despawns: u64,
}

impl EntityAllocator {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: FreeList::default(),
            _pad: [0; 64],
            next_reserved: AtomicU32::new(0),
            total_spawns: 0,
            total_despawns: 0,
        }
    }

    /// Reserve a fresh entity index atomically (`&self`, no `&mut` needed).
    /// Returns Entity with generation 0. Reserved entities are NOT in the
    /// generations vec yet — call `materialize_reserved()` from `&mut self`
    /// before using `alloc()` or `is_alive()` on reserved indices.
    pub fn reserve(&self) -> Entity {
        let index = self
            .next_reserved
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                if current < u32::MAX {
                    Some(current + 1)
                } else {
                    None
                }
            })
            .expect("entity index space exhausted: reserve() cannot allocate past u32::MAX");
        Entity::new(index, 0)
    }

    /// Sync the atomic counter to at least `generations.len()`.
    /// Called after snapshot restore to prevent `reserve()` from
    /// handing out already-used indices.
    pub fn sync_reserved(&mut self) {
        let len = self.generations.len() as u32;
        let current = self.next_reserved.load(Ordering::Relaxed);
        if current < len {
            self.next_reserved.store(len, Ordering::Relaxed);
        }
    }

    /// Backfill the generations vec to cover all reserved indices.
    /// Called automatically by `alloc()`.
    pub fn materialize_reserved(&mut self) {
        let reserved = self.next_reserved.load(Ordering::Relaxed);
        let before = self.generations.len();
        while self.generations.len() < reserved as usize {
            self.generations.push(0);
        }
        // Count reserved entities that were just materialized as spawns.
        // reserve() is &self (lock-free), so it can't increment the counter;
        // this is the first &mut self point where we know the count.
        self.total_spawns += (self.generations.len() - before) as u64;
    }

    pub fn alloc(&mut self) -> Entity {
        self.materialize_reserved();
        // +1 for the entity alloc() itself returns;
        // materialize_reserved() already counted any prior reserve() calls.
        self.total_spawns += 1;
        if let Some(index) = self.free_list.pop() {
            let r#gen = self.generations[index as usize];
            Entity::new(index, r#gen)
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.next_reserved.store(index + 1, Ordering::Relaxed);
            Entity::new(index, 0)
        }
    }

    pub fn dealloc(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx < self.generations.len() && self.generations[idx] == entity.generation() {
            let next_gen = self.generations[idx].checked_add(1).expect(
                "entity generation overflow: slot has been recycled 2^32 times. \
                 This is a robustness limit — the slot can no longer be safely reused.",
            );
            self.generations[idx] = next_gen;
            self.free_list.push(entity.index());
            self.total_despawns += 1;
            true
        } else {
            false
        }
    }

    /// Claim an unplaced entity from a committed log in its original slot.
    /// The caller checks placement before calling this method.
    pub(crate) fn adopt(&mut self, entity: Entity) -> bool {
        // Keep the reservation counter representable after adopting the slot.
        if entity.index() == u32::MAX {
            return false;
        }
        self.materialize_reserved();
        let index = entity.index() as usize;
        if let Some(&generation) = self.generations.get(index) {
            if generation > entity.generation() {
                return false; // Never revive an older local handle.
            }
            if self.free_list.remove(entity.index()) {
                self.generations[index] = entity.generation();
                self.total_spawns += 1;
            } else if generation != entity.generation() {
                return false; // Another unplaced reservation owns this slot.
            }
        } else {
            let previous_len = self.generations.len();
            self.generations.resize(index + 1, 0);
            // A source can reserve indices that it never logs. Keep those
            // holes available without allocating or claiming unrelated IDs.
            self.free_list.extend(previous_len as u32..entity.index());
            self.generations[index] = entity.generation();
            self.sync_reserved();
            self.total_spawns += 1;
        }
        true
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
    fn free_list_matches_ordered_vec() {
        let mut expected: Vec<u32> = (0..64).collect();
        let mut free = FreeList::from(expected.clone());
        let mut next = 64;
        let mut seed = 42u64;
        for step in 0..2048 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            match (seed >> 32) % 4 {
                0 | 1 => {
                    free.push(next);
                    expected.push(next);
                    next += 1;
                }
                2 if !expected.is_empty() => {
                    let pos = seed as usize % expected.len();
                    assert!(free.remove(expected.remove(pos)));
                }
                _ => assert_eq!(free.pop(), expected.pop()),
            }
            assert!(!free.remove(next));
            assert_eq!(free.len(), expected.len());
            assert_eq!(free.as_slice(), expected);
            if step % 127 == 0 {
                free = expected.clone().into();
            }
        }
        while let Some(index) = expected.pop() {
            assert_eq!(free.pop(), Some(index));
        }
        assert_eq!(free.pop(), None);
        assert_eq!(free.len(), 0);
        assert!(free.as_slice().is_empty());
    }

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

    #[test]
    fn spawn_counter_counts_alloc() {
        let mut alloc = EntityAllocator::new();
        assert_eq!(alloc.total_spawns, 0);
        alloc.alloc();
        alloc.alloc();
        assert_eq!(alloc.total_spawns, 2);
    }

    #[test]
    fn spawn_counter_counts_reserve_on_materialize() {
        let mut alloc = EntityAllocator::new();
        alloc.reserve();
        alloc.reserve();
        alloc.reserve();
        // reserve() is &self — counter not yet incremented
        assert_eq!(alloc.total_spawns, 0);
        // materialize_reserved() is the first &mut self point
        alloc.materialize_reserved();
        assert_eq!(alloc.total_spawns, 3);
    }

    #[test]
    fn spawn_counter_no_double_count_alloc_after_reserve() {
        let mut alloc = EntityAllocator::new();
        alloc.reserve(); // index 0
        alloc.reserve(); // index 1
        // alloc calls materialize_reserved internally, then allocates one more
        let e = alloc.alloc();
        // 2 from materialize + 1 from alloc = 3
        assert_eq!(alloc.total_spawns, 3);
        assert_eq!(e.index(), 2); // fresh index, not reserved
    }

    #[test]
    fn despawn_counter_counts_dealloc() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.alloc();
        let _e2 = alloc.alloc();
        assert_eq!(alloc.total_despawns, 0);
        alloc.dealloc(e1);
        assert_eq!(alloc.total_despawns, 1);
        // Failed dealloc (double-free) must NOT increment counter
        alloc.dealloc(e1);
        assert_eq!(alloc.total_despawns, 1);
    }
}

#[cfg(loom)]
mod loom_tests {
    use super::EntityAllocator;
    use loom::sync::Arc;
    use loom::thread;

    /// Two threads call reserve() concurrently on the same EntityAllocator.
    /// Verifies all returned entity indices are unique — no duplicate IDs.
    /// Uses the real EntityAllocator::reserve() (AtomicU32 fetch_add).
    #[test]
    fn loom_reserve_no_duplicate_indices() {
        loom::model(|| {
            let alloc = Arc::new(EntityAllocator::new());

            let a1 = alloc.clone();
            let t1 = thread::spawn(move || {
                let e1 = a1.reserve();
                let e2 = a1.reserve();
                vec![e1.index(), e2.index()]
            });

            let a2 = alloc.clone();
            let t2 = thread::spawn(move || {
                let e1 = a2.reserve();
                let e2 = a2.reserve();
                vec![e1.index(), e2.index()]
            });

            let mut indices: Vec<u32> = Vec::new();
            indices.extend(t1.join().unwrap());
            indices.extend(t2.join().unwrap());

            indices.sort();
            assert_eq!(indices, vec![0, 1, 2, 3]);
        });
    }
}
