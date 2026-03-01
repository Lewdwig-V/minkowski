use crate::world::World;

/// A secondary spatial index that can be rebuilt from world state.
///
/// Indexes are fully user-owned — the World has no awareness of them.
/// Implementations use standard query primitives (`world.query()`,
/// `Changed<T>`) internally. Query methods are defined per concrete
/// type, not on this trait.
///
/// # Design rationale
///
/// This trait deliberately excludes several things that were considered:
///
/// - **No generic query method.** A grid needs `query_cell()`, a BVH
///   needs `query_aabb()`, a k-d tree needs `nearest()`. Forcing one
///   query shape onto all index types would either over-constrain simple
///   structures or under-serve complex ones.
/// - **No component type parameters.** An index over `Position` and an
///   index over `(Position, Velocity)` would be different trait
///   instantiations, making it impossible to store mixed indexes in a
///   `Vec<Box<dyn SpatialIndex>>`.
/// - **No stored `&World` reference.** Indexes compose from the outside:
///   they receive the world transiently during `rebuild`/`update` and
///   own their data independently. This avoids lifetime coupling and
///   lets indexes outlive any particular borrow.
/// - **No registration on World.** Adding `world.register_index()` would
///   grow World's API with every index pattern someone invents. Keeping
///   indexes external means World stays focused on entities and
///   components.
///
/// The result is that structurally different algorithms (uniform grids,
/// quadtrees, BVH, k-d trees) all implement the same two-method trait
/// without friction — see the `boids` and `nbody` examples.
pub trait SpatialIndex {
    /// Reconstruct the index from scratch by scanning all matching entities.
    fn rebuild(&mut self, world: &mut World);

    /// Incrementally update the index. Defaults to full rebuild.
    ///
    /// Override this for indexes that can efficiently process only the
    /// entities whose indexed components changed since the last call.
    /// Despawned entities are handled lazily via generational validation
    /// at query time — stale entries are skipped when `world.is_alive()`
    /// returns false.
    fn update(&mut self, world: &mut World) {
        self.rebuild(world);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entity;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    /// Minimal index that collects entity IDs — validates the trait contract.
    struct EntityCollector {
        entities: Vec<Entity>,
    }

    impl EntityCollector {
        fn new() -> Self {
            Self {
                entities: Vec::new(),
            }
        }
    }

    impl SpatialIndex for EntityCollector {
        fn rebuild(&mut self, world: &mut World) {
            self.entities = world.query::<(Entity, &Pos)>().map(|(e, _)| e).collect();
        }
    }

    #[test]
    fn rebuild_collects_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));

        let mut idx = EntityCollector::new();
        idx.rebuild(&mut world);

        assert_eq!(idx.entities.len(), 2);
        assert!(idx.entities.contains(&e1));
        assert!(idx.entities.contains(&e2));
    }

    #[test]
    fn update_defaults_to_rebuild() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let mut idx = EntityCollector::new();
        idx.update(&mut world);

        assert_eq!(idx.entities.len(), 1);
    }

    #[test]
    fn stale_entries_detectable_via_is_alive() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));

        let mut idx = EntityCollector::new();
        idx.rebuild(&mut world);
        assert_eq!(idx.entities.len(), 2);

        // Despawn one entity — index is now stale
        world.despawn(e1);

        // Generational validation: filter at query time
        let live: Vec<_> = idx
            .entities
            .iter()
            .filter(|&&e| world.is_alive(e))
            .collect();
        assert_eq!(live.len(), 1);
        assert_eq!(*live[0], e2);
    }
}
