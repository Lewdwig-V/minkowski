use fixedbitset::FixedBitSet;

use crate::query::fetch::WorldQuery;
use crate::world::World;

/// Component-level access metadata for a query type.
///
/// Used by external schedulers to detect conflicts between systems.
/// Two accesses conflict if either writes a component the other reads or writes
/// (standard read-write lock rule, applied per component).
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
pub struct Access {
    reads: FixedBitSet,
    writes: FixedBitSet,
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

        Self { reads, writes }
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
    /// Conflict rule: two accesses conflict iff either writes to a
    /// component the other reads or writes.
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
}
