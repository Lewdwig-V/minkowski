use fixedbitset::FixedBitSet;

use crate::component::ComponentId;
use crate::query::fetch::WorldQuery;
use crate::world::World;

/// Component-level access metadata for a query type.
///
/// Used by external schedulers to detect conflicts between systems.
/// Two accesses conflict if either writes a component the other reads or writes
/// (standard read-write lock rule, applied per component), or if either declares
/// despawn capability while the other accesses any component.
///
/// # Example
///
/// ```
/// use minkowski::{Access, World};
///
/// #[derive(Clone, Copy)]
/// struct Pos(f32);
/// #[derive(Clone, Copy)]
/// struct Vel(f32);
/// #[derive(Clone, Copy)]
/// struct Health(u32);
///
/// let mut world = World::new();
///
/// let movement = Access::of::<(&mut Pos, &Vel)>(&mut world);
/// let regen = Access::of::<(&mut Health,)>(&mut world);
/// let log = Access::of::<(&Pos,)>(&mut world);
///
/// // Disjoint writes — no conflict
/// assert!(!movement.conflicts_with(&regen));
///
/// // Read Pos vs write Pos — conflict
/// assert!(movement.conflicts_with(&log));
/// ```
#[derive(Clone, Debug)]
pub struct Access {
    reads: FixedBitSet,
    writes: FixedBitSet,
    despawns: bool,
}

impl Access {
    /// Build access metadata for a query type.
    ///
    /// Registers any unregistered component types as a side effect
    /// (same as `world.query::<Q>()`).
    pub fn of<Q: WorldQuery + 'static>(world: &mut World) -> Self {
        Q::register(&mut world.components);
        let accessed = Q::accessed_ids(&world.components);
        let writes = Q::mutable_ids(&world.components);

        // reads = accessed - writes (components read but not written)
        let mut reads = accessed;
        reads.difference_with(&writes);

        // Normalize: replace zero-count bitsets with truly empty ones so that
        // `is_empty()` (which checks capacity, not popcount) works as expected.
        if reads.ones().next().is_none() {
            reads = FixedBitSet::new();
        }
        let writes = if writes.ones().next().is_none() {
            FixedBitSet::new()
        } else {
            writes
        };

        Self {
            reads,
            writes,
            despawns: false,
        }
    }

    /// Create an empty access set (no reads, no writes).
    pub fn empty() -> Self {
        Self {
            reads: FixedBitSet::new(),
            writes: FixedBitSet::new(),
            despawns: false,
        }
    }

    /// Add a read on the given component ID.
    pub fn add_read(&mut self, id: ComponentId) {
        self.reads.grow(id + 1);
        self.reads.insert(id);
    }

    /// Add a write on the given component ID.
    pub fn add_write(&mut self, id: ComponentId) {
        self.writes.grow(id + 1);
        self.writes.insert(id);
    }

    /// True if this access set includes the despawn capability.
    pub fn despawns(&self) -> bool {
        self.despawns
    }

    /// Mark this access set as including despawn capability.
    pub fn set_despawns(&mut self) {
        self.despawns = true;
    }

    /// True if this access set touches any component (reads or writes).
    pub fn has_any_access(&self) -> bool {
        self.reads.ones().next().is_some() || self.writes.ones().next().is_some()
    }

    /// Merge two access sets: union of reads, union of writes, OR of despawns.
    pub fn merge(&self, other: &Access) -> Access {
        let mut reads = self.reads.clone();
        let mut writes = self.writes.clone();
        reads.union_with(&other.reads);
        writes.union_with(&other.writes);
        Access {
            reads,
            writes,
            despawns: self.despawns || other.despawns,
        }
    }

    /// Components this query reads but does not write.
    pub fn reads(&self) -> &FixedBitSet {
        &self.reads
    }

    /// Components this query writes (mutably accesses).
    pub fn writes(&self) -> &FixedBitSet {
        &self.writes
    }

    /// True if these two accesses cannot safely run concurrently.
    ///
    /// Conflict rule: two accesses conflict if (1) either writes to a
    /// component the other reads or writes, or (2) either declares despawn
    /// capability while the other accesses any component.
    pub fn conflicts_with(&self, other: &Access) -> bool {
        // Does self write anything other reads or writes?
        if self.writes.intersection(&other.reads).next().is_some() {
            return true;
        }
        if self.writes.intersection(&other.writes).next().is_some() {
            return true;
        }
        // Does other write anything self reads?
        // (other writes ∩ self writes already covered above)
        if other.writes.intersection(&self.reads).next().is_some() {
            return true;
        }
        // Despawn conflicts with any component access on the other side
        if self.despawns && other.has_any_access() {
            return true;
        }
        if other.despawns && self.has_any_access() {
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entity;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Health(u32);

    #[test]
    fn reads_and_writes_for_immutable_ref() {
        let mut world = World::new();
        let a = Access::of::<(&Pos,)>(&mut world);
        assert!(!a.reads().is_empty());
        assert!(a.writes().is_empty());
    }

    #[test]
    fn reads_and_writes_for_mutable_ref() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos,)>(&mut world);
        assert!(a.reads().is_empty()); // reads = required - writes = empty
        assert!(!a.writes().is_empty());
    }

    #[test]
    fn mixed_read_write_query() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos, &Vel)>(&mut world);
        // Pos is written, Vel is read-only
        assert!(!a.writes().is_empty());
        assert!(!a.reads().is_empty());
    }

    #[test]
    fn no_conflict_disjoint_writes() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos,)>(&mut world);
        let b = Access::of::<(&mut Health,)>(&mut world);
        assert!(!a.conflicts_with(&b));
        assert!(!b.conflicts_with(&a));
    }

    #[test]
    fn conflict_read_write_same_component() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos,)>(&mut world);
        let b = Access::of::<(&Pos,)>(&mut world);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn conflict_write_write_same_component() {
        let mut world = World::new();
        let a = Access::of::<(&mut Health,)>(&mut world);
        let b = Access::of::<(&mut Health,)>(&mut world);
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn no_conflict_read_read_same_component() {
        let mut world = World::new();
        let a = Access::of::<(&Pos,)>(&mut world);
        let b = Access::of::<(&Pos,)>(&mut world);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn conflict_is_symmetric() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos, &Vel)>(&mut world);
        let b = Access::of::<(&mut Vel,)>(&mut world);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn no_conflict_empty_access() {
        let mut world = World::new();
        let a = Access::of::<(Entity,)>(&mut world);
        let b = Access::of::<(&mut Pos,)>(&mut world);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn conflict_optional_read_vs_write() {
        let mut world = World::new();
        // Option<&Pos> reads Pos when present — must conflict with &mut Pos
        let a = Access::of::<(Option<&Pos>,)>(&mut world);
        let b = Access::of::<(&mut Pos,)>(&mut world);
        assert!(!a.reads().is_empty(), "Option<&T> should report a read");
        assert!(
            a.conflicts_with(&b),
            "optional read must conflict with write"
        );
        assert!(
            b.conflicts_with(&a),
            "write must conflict with optional read"
        );
    }

    #[test]
    fn no_conflict_optional_read_vs_read() {
        let mut world = World::new();
        let a = Access::of::<(Option<&Pos>,)>(&mut world);
        let b = Access::of::<(&Pos,)>(&mut world);
        assert!(!a.conflicts_with(&b), "two readers never conflict");
    }

    // ── Builder API tests ─────────────────────────────────────────

    #[test]
    fn builder_empty() {
        let a = Access::empty();
        assert!(a.reads().is_empty());
        assert!(a.writes().is_empty());
    }

    #[test]
    fn builder_add_read() {
        let mut a = Access::empty();
        a.add_read(0);
        assert!(a.reads()[0]);
        assert!(a.writes().is_empty());
    }

    #[test]
    fn builder_add_write() {
        let mut a = Access::empty();
        a.add_write(0);
        assert!(a.reads().is_empty());
        assert!(a.writes()[0]);
    }

    #[test]
    fn builder_merge() {
        let mut a = Access::empty();
        a.add_read(0);
        let mut b = Access::empty();
        b.add_write(1);
        let merged = a.merge(&b);
        assert!(merged.reads()[0]);
        assert!(merged.writes()[1]);
    }

    #[test]
    fn builder_conflicts_with_of() {
        let mut world = World::new();
        let from_query = Access::of::<(&mut Pos,)>(&mut world);
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut from_builder = Access::empty();
        from_builder.add_read(pos_id);
        assert!(from_query.conflicts_with(&from_builder));
    }

    #[test]
    fn complex_disjoint_systems() {
        let mut world = World::new();
        // movement: reads Vel, writes Pos
        let movement = Access::of::<(&mut Pos, &Vel)>(&mut world);
        // health_regen: writes Health
        let regen = Access::of::<(&mut Health,)>(&mut world);
        // These touch completely different components
        assert!(!movement.conflicts_with(&regen));
    }

    // ── Despawn flag tests ──────────────────────────────────────

    #[test]
    fn despawn_flag_default_false() {
        let a = Access::empty();
        assert!(!a.despawns());
    }

    #[test]
    fn set_despawns() {
        let mut a = Access::empty();
        a.set_despawns();
        assert!(a.despawns());
    }

    #[test]
    fn has_any_access_empty() {
        let a = Access::empty();
        assert!(!a.has_any_access());
    }

    #[test]
    fn has_any_access_with_read() {
        let mut a = Access::empty();
        a.add_read(0);
        assert!(a.has_any_access());
    }

    #[test]
    fn has_any_access_with_write() {
        let mut a = Access::empty();
        a.add_write(0);
        assert!(a.has_any_access());
    }

    #[test]
    fn despawn_conflicts_with_reader() {
        let mut a = Access::empty();
        a.set_despawns();
        let mut b = Access::empty();
        b.add_read(0);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn despawn_conflicts_with_writer() {
        let mut a = Access::empty();
        a.set_despawns();
        let mut b = Access::empty();
        b.add_write(0);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn despawn_no_conflict_with_empty() {
        let mut a = Access::empty();
        a.set_despawns();
        let b = Access::empty();
        assert!(!a.conflicts_with(&b));
        assert!(!b.conflicts_with(&a));
    }

    #[test]
    fn two_despawners_with_reads_conflict() {
        let mut a = Access::empty();
        a.set_despawns();
        a.add_read(0);
        let mut b = Access::empty();
        b.set_despawns();
        b.add_read(1);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn despawn_merge_preserves_flag() {
        let mut a = Access::empty();
        a.set_despawns();
        let b = Access::empty();
        let merged = a.merge(&b);
        assert!(merged.despawns());
    }
}
