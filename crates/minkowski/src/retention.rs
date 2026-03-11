//! Expiry component and retention reducer for dispatch-count entity cleanup.
//!
//! `Expiry` marks entities for despawn after a set number of retention
//! dispatches. The retention reducer decrements on each run and despawns
//! entities that reach zero. The user controls dispatch frequency — the
//! engine never runs retention automatically.

/// Marks an entity for despawn after a number of retention dispatches.
///
/// Each call to `registry.run(&mut world, retention_id, ())` counts as
/// one dispatch. Entities with `Expiry::after(3)` survive 3 dispatches
/// and are despawned on the 3rd.
///
/// ```ignore
/// // This entity will be despawned on the 5th retention dispatch.
/// world.spawn((data, Expiry::after(5)));
/// ```
///
/// The countdown is in terms of retention dispatches — a concept the user
/// fully controls. It is not tied to ticks, frames, or wall-clock time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expiry(u32);

impl Expiry {
    /// Create an expiry that survives `dispatches` retention cycles.
    ///
    /// The entity is despawned on the dispatch that decrements this to zero.
    /// A value of 1 means "despawn on the next retention run."
    /// A value of 0 means "despawn immediately on the next retention run."
    pub fn after(dispatches: u32) -> Self {
        Self(dispatches)
    }

    /// Remaining dispatches before this entity expires.
    pub fn remaining(&self) -> u32 {
        self.0
    }

    /// Returns `true` if the entity should be despawned (countdown reached zero).
    pub fn is_expired(&self) -> bool {
        self.0 == 0
    }

    /// Decrement the countdown by one, saturating at zero.
    pub(crate) fn tick(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    #[test]
    fn expiry_is_a_component() {
        let mut world = World::new();
        let e = world.spawn((Expiry::after(5),));
        assert_eq!(world.get::<Expiry>(e).unwrap().remaining(), 5);
    }

    #[test]
    fn expiry_after_zero_is_immediately_expired() {
        let exp = Expiry::after(0);
        assert!(exp.is_expired());
        assert_eq!(exp.remaining(), 0);
    }

    #[test]
    fn expiry_tick_decrements() {
        let mut exp = Expiry::after(3);
        assert!(!exp.is_expired());
        exp.tick();
        assert_eq!(exp.remaining(), 2);
        exp.tick();
        assert_eq!(exp.remaining(), 1);
        exp.tick();
        assert_eq!(exp.remaining(), 0);
        assert!(exp.is_expired());
    }

    #[test]
    fn expiry_tick_saturates_at_zero() {
        let mut exp = Expiry::after(0);
        exp.tick();
        assert_eq!(exp.remaining(), 0);
    }

    #[test]
    fn expiry_coexists_with_other_components() {
        let mut world = World::new();
        let e = world.spawn((42u32, Expiry::after(10)));
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
        assert_eq!(world.get::<Expiry>(e).unwrap().remaining(), 10);
    }

    #[test]
    fn retention_despawns_expired_entities() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let e_immediate = world.spawn((Expiry::after(0), 1u32));
        let e_one = world.spawn((Expiry::after(1), 2u32));
        let e_many = world.spawn((Expiry::after(100), 3u32));
        let e_no_expiry = world.spawn((4u32,));

        registry.run(&mut world, retention_id, ()).unwrap();

        // after(0) → despawned on first dispatch
        assert!(!world.is_alive(e_immediate));
        // after(1) → decremented to 0, despawned on first dispatch
        assert!(!world.is_alive(e_one));
        // after(100) → decremented to 99, still alive
        assert!(world.is_alive(e_many));
        assert_eq!(world.get::<Expiry>(e_many).unwrap().remaining(), 99);
        // no expiry → untouched
        assert!(world.is_alive(e_no_expiry));
        assert_eq!(*world.get::<u32>(e_no_expiry).unwrap(), 4);
    }

    #[test]
    fn retention_progressive_expiry() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let e1 = world.spawn((Expiry::after(1),));
        let e2 = world.spawn((Expiry::after(2),));
        let e3 = world.spawn((Expiry::after(3),));

        // Dispatch 1: e1 despawned (1→0), e2 alive (2→1), e3 alive (3→2)
        registry.run(&mut world, retention_id, ()).unwrap();
        assert!(!world.is_alive(e1));
        assert!(world.is_alive(e2));
        assert!(world.is_alive(e3));

        // Dispatch 2: e2 despawned (1→0), e3 alive (2→1)
        registry.run(&mut world, retention_id, ()).unwrap();
        assert!(!world.is_alive(e2));
        assert!(world.is_alive(e3));

        // Dispatch 3: e3 despawned (1→0)
        registry.run(&mut world, retention_id, ()).unwrap();
        assert!(!world.is_alive(e3));
    }

    #[test]
    fn retention_is_idempotent_on_empty() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        world.spawn((Expiry::after(0),));

        registry.run(&mut world, retention_id, ()).unwrap();
        registry.run(&mut world, retention_id, ()).unwrap();

        let mut count = 0;
        world.query::<(&Expiry,)>().for_each(|_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn retention_noop_when_nothing_expired() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let e = world.spawn((Expiry::after(u32::MAX),));

        registry.run(&mut world, retention_id, ()).unwrap();

        assert!(world.is_alive(e));
    }

    #[test]
    fn retention_access_declares_despawns() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let access = registry.query_reducer_access(retention_id);
        assert!(access.despawns());
    }
}
