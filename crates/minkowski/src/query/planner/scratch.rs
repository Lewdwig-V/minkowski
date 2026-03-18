use crate::entity::Entity;
use crate::world::EntityLocation;

/// Pool-aware reusable entity buffer for allocation-free query execution.
///
/// Used by join/gather plans instead of per-node `Vec<Entity>` allocation.
/// Call [`clear`](ScratchBuffer::clear) between uses to reset length while
/// preserving the backing allocation.
pub(super) struct ScratchBuffer {
    pub(super) entities: Vec<Entity>,
}

/// Maximum pre-allocation cap: 64K entities.
pub(super) const SCRATCH_MAX_CAP: usize = 64 * 1024;

impl ScratchBuffer {
    /// Create a new buffer with the given estimated capacity, capped at 64K entities.
    pub(super) fn new(estimated_capacity: usize) -> Self {
        Self {
            entities: Vec::with_capacity(estimated_capacity.min(SCRATCH_MAX_CAP)),
        }
    }

    /// Append an entity to the buffer.
    pub(super) fn push(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Reset length to 0, preserving the backing allocation.
    pub(super) fn clear(&mut self) {
        self.entities.clear();
    }

    /// Number of entities currently in the buffer.
    pub(super) fn len(&self) -> usize {
        self.entities.len()
    }

    /// Current allocation capacity.
    #[cfg_attr(not(test), expect(dead_code))]
    pub(super) fn capacity(&self) -> usize {
        self.entities.capacity()
    }

    /// View the buffer contents as a slice.
    pub(super) fn as_slice(&self) -> &[Entity] {
        &self.entities
    }

    /// Compute the sorted intersection of two entity sets stored contiguously
    /// in this buffer as `[left_0..left_len | right_0..right_len]`.
    ///
    /// Sorts the left partition in-place by `to_bits()`, then for each entity
    /// in the right partition, binary-searches the sorted left. Matches are
    /// appended to the end of the buffer. Returns a slice over the matches.
    pub(super) fn sorted_intersection(&mut self, left_len: usize) -> &[Entity] {
        let total = self.entities.len();
        assert!(
            left_len <= total,
            "sorted_intersection: left_len ({left_len}) exceeds buffer length ({total})"
        );

        // Sort the left partition in place by raw bits (deterministic total order).
        self.entities[..left_len].sort_unstable_by_key(|e| e.to_bits());

        // Scan right partition, binary-search in sorted left, collect matches.
        let mut match_count = 0;
        for i in left_len..total {
            let entity = self.entities[i];
            let found = self.entities[..left_len]
                .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
                .is_ok();
            if found {
                self.entities.push(entity);
                match_count += 1;
            }
        }

        let final_len = self.entities.len();
        &self.entities[final_len - match_count..final_len]
    }

    /// Cache-aware partitioned intersection of two entity sets stored as
    /// `[left_0..left_len | right_0..right_len]`.
    ///
    /// Entities are bucketed by `to_bits() % partitions`. Each partition is
    /// intersected independently so the working set fits in L2 cache.
    /// Matches are appended to the end of the buffer. Returns a slice over
    /// the matches.
    ///
    /// `partitions` is clamped to the actual build-side cardinality (and
    /// capped at 4096) so that overestimated planner row counts cannot
    /// cause unbounded bucket allocation at runtime.
    pub(super) fn partitioned_intersection(
        &mut self,
        left_len: usize,
        partitions: usize,
    ) -> &[Entity] {
        assert!(
            partitions > 0,
            "partitioned_intersection: partitions must be > 0, got {partitions}"
        );
        let total = self.entities.len();
        assert!(
            left_len <= total,
            "partitioned_intersection: left_len ({left_len}) exceeds buffer length ({total})"
        );

        // Clamp partition count to runtime cardinality: more buckets than
        // entities wastes memory with no cache benefit. The hard cap prevents
        // a wildly overestimated row count from causing OOM.
        const MAX_PARTITIONS: usize = 4096;
        let partitions = partitions.min(left_len).min(MAX_PARTITIONS);
        if partitions <= 1 {
            return self.sorted_intersection(left_len);
        }

        // Bucket left and right entities by partition.
        let mut left_buckets: Vec<Vec<u64>> = vec![Vec::new(); partitions];
        for &e in &self.entities[..left_len] {
            let bits = e.to_bits();
            left_buckets[(bits % partitions as u64) as usize].push(bits);
        }
        // Sort each left bucket for binary search.
        for bucket in &mut left_buckets {
            bucket.sort_unstable();
        }

        // Probe right entities against their corresponding left bucket.
        let mut match_count = 0;
        for i in left_len..total {
            let entity = self.entities[i];
            let bits = entity.to_bits();
            let bucket = &left_buckets[(bits % partitions as u64) as usize];
            if bucket.binary_search(&bits).is_ok() {
                self.entities.push(entity);
                match_count += 1;
            }
        }

        let final_len = self.entities.len();
        &self.entities[final_len - match_count..final_len]
    }

    /// Sort entities by (ArchetypeId, Row) to restore cache locality after
    /// join materialisation. Entities from the same archetype become
    /// contiguous, and within each archetype group, rows are in physical
    /// memory order (ascending).
    ///
    /// # Panics
    /// Panics if any entity in the buffer has no location (dead entity).
    /// This should never happen: join collectors only iterate live archetypes.
    pub(super) fn sort_by_archetype(&mut self, entity_locations: &[Option<EntityLocation>]) {
        self.entities.sort_unstable_by_key(|e| {
            let loc = entity_locations[e.index() as usize]
                .expect("entity in scratch buffer has no location");
            ((loc.archetype_id.0 as u64) << 32) | (loc.row as u64)
        });
    }
}
