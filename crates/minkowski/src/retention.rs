//! Expiry component and retention reducer for tick-based entity cleanup.
//!
//! `Expiry` marks entities for despawn at a target tick. The retention
//! reducer scans for expired entities and batch-despawns them. The user
//! controls dispatch frequency — the engine never runs retention automatically.

use crate::tick::ChangeTick;

/// Marks an entity for despawn when the world tick reaches or exceeds this value.
///
/// Compute the deadline from the current tick via
/// [`World::change_tick`](crate::World::change_tick) and pass it at spawn time:
///
/// ```ignore
/// let deadline = Expiry::with_ttl(world.change_tick(), 1000);
/// world.spawn((data, deadline));
/// ```
///
/// The tick is a monotonic u64 from change detection — **not** wall-clock time.
/// For time-based TTL, convert duration to ticks based on your tick rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expiry(ChangeTick);

impl Expiry {
    /// Create an expiry at the given tick.
    pub fn at_tick(tick: ChangeTick) -> Self {
        Self(tick)
    }

    /// Create an expiry `ttl` ticks from `now`.
    pub fn with_ttl(now: ChangeTick, ttl: u64) -> Self {
        Self(ChangeTick::from_raw(now.to_raw().saturating_add(ttl)))
    }

    /// The deadline tick.
    pub fn deadline(&self) -> ChangeTick {
        self.0
    }

    /// Returns `true` if the deadline has been reached or passed.
    pub fn is_expired(&self, current: ChangeTick) -> bool {
        self.0.to_raw() <= current.to_raw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    #[test]
    fn expiry_is_a_component() {
        let mut world = World::new();
        let tick = world.change_tick();
        let e = world.spawn((Expiry::at_tick(tick),));
        assert_eq!(world.get::<Expiry>(e).unwrap().deadline(), tick);
    }

    #[test]
    fn expiry_round_trip() {
        let tick = ChangeTick::from_raw(42);
        let exp = Expiry::at_tick(tick);
        assert_eq!(exp.deadline().to_raw(), 42);
    }

    #[test]
    fn expiry_with_ttl() {
        let now = ChangeTick::from_raw(100);
        let exp = Expiry::with_ttl(now, 50);
        assert_eq!(exp.deadline().to_raw(), 150);
    }

    #[test]
    fn expiry_with_ttl_saturates() {
        let now = ChangeTick::from_raw(u64::MAX - 10);
        let exp = Expiry::with_ttl(now, 100);
        assert_eq!(exp.deadline().to_raw(), u64::MAX);
    }

    #[test]
    fn expiry_is_expired() {
        let exp = Expiry::at_tick(ChangeTick::from_raw(100));
        assert!(!exp.is_expired(ChangeTick::from_raw(99)));
        assert!(exp.is_expired(ChangeTick::from_raw(100))); // boundary: equal
        assert!(exp.is_expired(ChangeTick::from_raw(101)));
    }

    #[test]
    fn expiry_coexists_with_other_components() {
        let mut world = World::new();
        let tick = ChangeTick::from_raw(100);
        let e = world.spawn((42u32, Expiry::at_tick(tick)));
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
        assert_eq!(world.get::<Expiry>(e).unwrap().deadline().to_raw(), 100);
    }

    #[test]
    fn retention_despawns_expired_entities() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let tick = world.change_tick();
        let past = ChangeTick::from_raw(0); // already expired
        let future = ChangeTick::from_raw(tick.to_raw() + 1_000_000);

        let e_expired_1 = world.spawn((Expiry::at_tick(past), 1u32));
        let e_expired_2 = world.spawn((Expiry::at_tick(past), 2u32));
        let e_alive = world.spawn((Expiry::at_tick(future), 3u32));
        let e_no_expiry = world.spawn((4u32,));

        registry.run(&mut world, retention_id, ()).unwrap();

        assert!(!world.is_alive(e_expired_1));
        assert!(!world.is_alive(e_expired_2));
        assert!(world.is_alive(e_alive));
        assert!(world.is_alive(e_no_expiry));
        assert_eq!(*world.get::<u32>(e_alive).unwrap(), 3);
        assert_eq!(*world.get::<u32>(e_no_expiry).unwrap(), 4);
    }

    #[test]
    fn retention_is_idempotent() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let past = ChangeTick::from_raw(0);
        world.spawn((Expiry::at_tick(past),));

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

        let future = ChangeTick::from_raw(u64::MAX);
        let e = world.spawn((Expiry::at_tick(future),));

        registry.run(&mut world, retention_id, ()).unwrap();

        assert!(world.is_alive(e));
    }

    #[test]
    fn retention_despawns_at_boundary_tick() {
        // Regression: the retention adapter must compare deadlines against
        // the post-query tick, not a pre-query snapshot. world.query()
        // advances the tick, so a pre-query snapshot lets entities at
        // deadline == tick+1 survive one extra dispatch.
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        // Set deadline to exactly the tick the query will advance to.
        // The entity should be despawned on this run, not the next.
        let pre_tick = world.change_tick().to_raw();
        // world.query() inside retention will advance tick by at least 1,
        // so deadline = pre_tick + 1 should be caught.
        let deadline = ChangeTick::from_raw(pre_tick + 1);
        let e = world.spawn((Expiry::at_tick(deadline),));

        registry.run(&mut world, retention_id, ()).unwrap();
        assert!(
            !world.is_alive(e),
            "entity at deadline tick+1 should be despawned by the run that crosses its deadline"
        );
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
