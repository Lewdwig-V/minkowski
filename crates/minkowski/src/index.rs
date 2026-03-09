use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::ops::RangeBounds;

use crate::component::Component;
use crate::entity::Entity;
use crate::tick::ChangeTick;
use crate::world::World;

/// A secondary spatial index that can be rebuilt from world state.
///
/// Indexes are fully user-owned — the World has no awareness of them.
/// Implementations use standard query primitives (`world.query()`,
/// `world.query_changed_since()`) internally. Query methods are defined
/// per concrete type, not on this trait.
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

/// A sorted index over a single component column, backed by a [`BTreeMap`].
///
/// Supports exact-match lookups via [`get`](Self::get) and ordered range
/// queries via [`range`](Self::range). The index is fully user-owned —
/// World has no awareness of it.
///
/// # Incremental updates
///
/// [`SpatialIndex::update`] uses per-index tick state to scan only entities
/// whose indexed component was mutated since the last call. Each index
/// instance tracks its own [`ChangeTick`], so multiple indexes on the same
/// component type can call `update` independently without interfering.
///
/// # Stale entries
///
/// Component removal (without despawn) leaves stale entries in the
/// index. These are filtered at query time by [`get_valid`](Self::get_valid)
/// and [`range_valid`](Self::range_valid), which check that the entity still
/// carries the indexed component. Memory from stale entries is reclaimed on
/// the next [`rebuild`](SpatialIndex::rebuild).
///
/// For most workloads, component removal is rare compared to value
/// mutation. If removal is frequent, call `rebuild` periodically
/// to compact the index.
///
/// | Method | Checks | Use case |
/// |---|---|---|
/// | [`get`](Self::get) | None | Caller knows entities are valid |
/// | [`get_valid`](Self::get_valid) | [`World::has`] | Filters despawns and removals |
pub struct BTreeIndex<T: Component + Ord + Clone> {
    tree: BTreeMap<T, Vec<Entity>>,
    reverse: HashMap<Entity, T>,
    last_sync: ChangeTick,
}

impl<T: Component + Ord + Clone> BTreeIndex<T> {
    /// Create an empty index. Call [`rebuild`](SpatialIndex::rebuild) to populate.
    pub fn new() -> Self {
        Self {
            tree: BTreeMap::new(),
            reverse: HashMap::new(),
            last_sync: ChangeTick::default(),
        }
    }

    /// Number of entities tracked by the index (including stale entries).
    pub fn len(&self) -> usize {
        self.reverse.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.reverse.is_empty()
    }

    /// Returns true if the entity is tracked by the index (may be stale).
    pub fn contains(&self, entity: Entity) -> bool {
        self.reverse.contains_key(&entity)
    }

    /// Return all entities with exactly the given component value.
    ///
    /// Returns an empty slice if no entities match.
    pub fn get(&self, value: &T) -> &[Entity] {
        self.tree.get(value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Iterate over `(value, entities)` pairs whose keys fall within `range`.
    pub fn range<R: RangeBounds<T>>(&self, range: R) -> impl Iterator<Item = (&T, &[Entity])> {
        self.tree.range(range).map(|(k, v)| (k, v.as_slice()))
    }

    /// Validated lookup — filters entities that were despawned or had `T` removed.
    pub fn get_valid<'a>(
        &'a self,
        value: &T,
        world: &'a World,
    ) -> impl Iterator<Item = Entity> + 'a {
        self.get(value)
            .iter()
            .copied()
            .filter(|&entity| world.has::<T>(entity))
    }

    /// Validated range query — filters entities that were despawned or had `T` removed.
    pub fn range_valid<'a, R: RangeBounds<T> + 'a>(
        &'a self,
        range: R,
        world: &'a World,
    ) -> impl Iterator<Item = (&'a T, Entity)> + 'a {
        self.range(range).flat_map(move |(k, entities)| {
            entities
                .iter()
                .copied()
                .filter(|&entity| world.has::<T>(entity))
                .map(move |entity| (k, entity))
        })
    }

    /// Borrow the internal state for serialization.
    pub fn as_raw_parts(&self) -> (&BTreeMap<T, Vec<Entity>>, &HashMap<Entity, T>, ChangeTick) {
        (&self.tree, &self.reverse, self.last_sync)
    }

    /// Reconstruct from deserialized parts.
    pub fn from_raw_parts(
        tree: BTreeMap<T, Vec<Entity>>,
        reverse: HashMap<Entity, T>,
        last_sync: ChangeTick,
    ) -> Self {
        debug_assert_eq!(
            tree.values().map(|v| v.len()).sum::<usize>(),
            reverse.len(),
            "BTreeIndex::from_raw_parts: forward/reverse map size mismatch"
        );
        Self {
            tree,
            reverse,
            last_sync,
        }
    }

    fn remove_entity(&mut self, entity: Entity) {
        if let Some(old_value) = self.reverse.remove(&entity) {
            if let Some(bucket) = self.tree.get_mut(&old_value) {
                bucket.retain(|&e| e != entity);
                if bucket.is_empty() {
                    self.tree.remove(&old_value);
                }
            }
        }
    }

    fn insert_entity(&mut self, entity: Entity, value: T) {
        self.reverse.insert(entity, value.clone());
        self.tree.entry(value).or_default().push(entity);
    }
}

impl<T: Component + Ord + Clone> Default for BTreeIndex<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Component + Ord + Clone> SpatialIndex for BTreeIndex<T> {
    fn rebuild(&mut self, world: &mut World) {
        self.tree.clear();
        self.reverse.clear();
        for (entity, value) in world.query::<(Entity, &T)>() {
            self.insert_entity(entity, value.clone());
        }
        self.last_sync = world.change_tick();
    }

    fn update(&mut self, world: &mut World) {
        let changed = world.query_changed_since::<T>(self.last_sync);
        for (entity, value) in changed {
            self.remove_entity(entity);
            self.insert_entity(entity, value);
        }
        self.last_sync = world.change_tick();
    }
}

/// A hash-based index over a single component column, backed by a [`HashMap`].
///
/// Supports exact-match lookups via [`get`](Self::get). For ordered queries,
/// use [`BTreeIndex`] instead.
///
/// # Incremental updates
///
/// [`SpatialIndex::update`] uses per-index tick state to scan only entities
/// whose indexed component was mutated since the last call. Each index
/// instance tracks its own [`ChangeTick`], so multiple indexes on the same
/// component type can call `update` independently without interfering.
///
/// # Stale entries
///
/// Component removal (without despawn) leaves stale entries in the
/// index. These are filtered at query time by [`get_valid`](Self::get_valid),
/// which checks that the entity still carries the indexed component.
/// Memory from stale entries is reclaimed on the next
/// [`rebuild`](SpatialIndex::rebuild).
///
/// | Method | Checks | Use case |
/// |---|---|---|
/// | [`get`](Self::get) | None | Caller knows entities are valid |
/// | [`get_valid`](Self::get_valid) | [`World::has`] | Filters despawns and removals |
pub struct HashIndex<T: Component + Hash + Eq + Clone> {
    map: HashMap<T, Vec<Entity>>,
    reverse: HashMap<Entity, T>,
    last_sync: ChangeTick,
}

impl<T: Component + Hash + Eq + Clone> HashIndex<T> {
    /// Create an empty index. Call [`rebuild`](SpatialIndex::rebuild) to populate.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            reverse: HashMap::new(),
            last_sync: ChangeTick::default(),
        }
    }

    /// Return all entities with exactly the given component value.
    ///
    /// Returns an empty slice if no entities match.
    pub fn get(&self, value: &T) -> &[Entity] {
        self.map.get(value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of entities tracked by the index (including stale entries).
    pub fn len(&self) -> usize {
        self.reverse.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.reverse.is_empty()
    }

    /// Returns true if the entity is tracked by the index (may be stale).
    pub fn contains(&self, entity: Entity) -> bool {
        self.reverse.contains_key(&entity)
    }

    /// Validated lookup — filters entities that were despawned or had `T` removed.
    pub fn get_valid<'a>(
        &'a self,
        value: &T,
        world: &'a World,
    ) -> impl Iterator<Item = Entity> + 'a {
        self.get(value)
            .iter()
            .copied()
            .filter(|&entity| world.has::<T>(entity))
    }

    /// Borrow the internal state for serialization.
    pub fn as_raw_parts(&self) -> (&HashMap<T, Vec<Entity>>, &HashMap<Entity, T>, ChangeTick) {
        (&self.map, &self.reverse, self.last_sync)
    }

    /// Reconstruct from deserialized parts.
    pub fn from_raw_parts(
        map: HashMap<T, Vec<Entity>>,
        reverse: HashMap<Entity, T>,
        last_sync: ChangeTick,
    ) -> Self {
        debug_assert_eq!(
            map.values().map(|v| v.len()).sum::<usize>(),
            reverse.len(),
            "HashIndex::from_raw_parts: forward/reverse map size mismatch"
        );
        Self {
            map,
            reverse,
            last_sync,
        }
    }

    fn remove_entity(&mut self, entity: Entity) {
        if let Some(old_value) = self.reverse.remove(&entity) {
            if let Some(bucket) = self.map.get_mut(&old_value) {
                bucket.retain(|&e| e != entity);
                if bucket.is_empty() {
                    self.map.remove(&old_value);
                }
            }
        }
    }

    fn insert_entity(&mut self, entity: Entity, value: T) {
        self.reverse.insert(entity, value.clone());
        self.map.entry(value).or_default().push(entity);
    }
}

impl<T: Component + Hash + Eq + Clone> Default for HashIndex<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Component + Hash + Eq + Clone> SpatialIndex for HashIndex<T> {
    fn rebuild(&mut self, world: &mut World) {
        self.map.clear();
        self.reverse.clear();
        for (entity, value) in world.query::<(Entity, &T)>() {
            self.insert_entity(entity, value.clone());
        }
        self.last_sync = world.change_tick();
    }

    fn update(&mut self, world: &mut World) {
        let changed = world.query_changed_since::<T>(self.last_sync);
        for (entity, value) in changed {
            self.remove_entity(entity);
            self.insert_entity(entity, value);
        }
        self.last_sync = world.change_tick();
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

    // --- BTreeIndex / HashIndex test component ---

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Score(u32);

    // --- BTreeIndex tests ---

    #[test]
    fn btree_rebuild_basic() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));
        let e3 = world.spawn((Score(10),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let tens = idx.get(&Score(10));
        assert_eq!(tens.len(), 2);
        assert!(tens.contains(&e1));
        assert!(tens.contains(&e3));

        let twenties = idx.get(&Score(20));
        assert_eq!(twenties.len(), 1);
        assert!(twenties.contains(&e2));

        // Missing value returns empty slice.
        assert!(idx.get(&Score(99)).is_empty());
    }

    #[test]
    fn btree_range() {
        let mut world = World::new();
        world.spawn((Score(5),));
        let e2 = world.spawn((Score(15),));
        let e3 = world.spawn((Score(25),));
        world.spawn((Score(35),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let in_range: Vec<_> = idx.range(Score(10)..Score(30)).collect();
        assert_eq!(in_range.len(), 2);

        let (val0, ents0) = in_range[0];
        assert_eq!(*val0, Score(15));
        assert!(ents0.contains(&e2));

        let (val1, ents1) = in_range[1];
        assert_eq!(*val1, Score(25));
        assert!(ents1.contains(&e3));
    }

    #[test]
    fn btree_empty() {
        let mut world = World::new();
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        assert!(idx.get(&Score(0)).is_empty());
        assert_eq!(idx.range(..).count(), 0);
    }

    #[test]
    fn btree_update_incremental() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        world.spawn((Score(20),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        assert_eq!(idx.get(&Score(10)).len(), 1);

        // Mutate e1's score from 10 to 30.
        *world.get_mut::<Score>(e1).unwrap() = Score(30);
        idx.update(&mut world);

        // Old bucket should be empty, new bucket should have the entity.
        assert!(idx.get(&Score(10)).is_empty());
        let thirties = idx.get(&Score(30));
        assert_eq!(thirties.len(), 1);
        assert!(thirties.contains(&e1));
    }

    #[test]
    fn btree_stale_after_despawn() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        assert_eq!(idx.get(&Score(10)).len(), 1);

        // Despawn e1 — index is now stale.
        world.despawn(e1);

        // Stale entry still present in index.
        let tens = idx.get(&Score(10));
        assert_eq!(tens.len(), 1);
        assert!(!world.is_alive(tens[0]));

        // Rebuild cleans up stale entries.
        idx.rebuild(&mut world);
        assert!(idx.get(&Score(10)).is_empty());

        let twenties = idx.get(&Score(20));
        assert_eq!(twenties.len(), 1);
        assert!(twenties.contains(&e2));
    }

    #[test]
    fn btree_multi_archetype() {
        let mut world = World::new();
        // Archetype (Score, Pos)
        let e1 = world.spawn((Score(10), Pos { x: 0.0, y: 0.0 }));
        // Archetype (Score,)
        let e2 = world.spawn((Score(10),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let tens = idx.get(&Score(10));
        assert_eq!(tens.len(), 2);
        assert!(tens.contains(&e1));
        assert!(tens.contains(&e2));
    }

    // --- HashIndex tests ---

    #[test]
    fn hash_rebuild_basic() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));
        let e3 = world.spawn((Score(10),));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);

        let tens = idx.get(&Score(10));
        assert_eq!(tens.len(), 2);
        assert!(tens.contains(&e1));
        assert!(tens.contains(&e3));

        let twenties = idx.get(&Score(20));
        assert_eq!(twenties.len(), 1);
        assert!(twenties.contains(&e2));

        assert!(idx.get(&Score(99)).is_empty());
    }

    #[test]
    fn hash_update_incremental() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        world.spawn((Score(20),));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);
        assert_eq!(idx.get(&Score(10)).len(), 1);

        // Mutate e1's score from 10 to 30.
        *world.get_mut::<Score>(e1).unwrap() = Score(30);
        idx.update(&mut world);

        assert!(idx.get(&Score(10)).is_empty());
        let thirties = idx.get(&Score(30));
        assert_eq!(thirties.len(), 1);
        assert!(thirties.contains(&e1));
    }

    #[test]
    fn hash_duplicate_values() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let e2 = world.spawn((Score(42),));
        let e3 = world.spawn((Score(42),));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);

        let fortytwos = idx.get(&Score(42));
        assert_eq!(fortytwos.len(), 3);
        assert!(fortytwos.contains(&e1));
        assert!(fortytwos.contains(&e2));
        assert!(fortytwos.contains(&e3));
    }

    #[test]
    fn spatial_index_trait_satisfaction() {
        let mut world = World::new();
        world.spawn((Score(1),));

        let mut indexes: Vec<Box<dyn SpatialIndex>> = vec![
            Box::new(BTreeIndex::<Score>::new()),
            Box::new(HashIndex::<Score>::new()),
        ];

        for idx in &mut indexes {
            idx.rebuild(&mut world);
        }

        for idx in &mut indexes {
            idx.update(&mut world);
        }
    }

    #[test]
    fn btree_get_valid_filters_removed_component() {
        let mut world = World::new();
        // Spawn with two components so remove::<Score> migrates, doesn't despawn
        let e1 = world.spawn((Score(10), Pos { x: 0.0, y: 0.0 }));
        let e2 = world.spawn((Score(10), Pos { x: 1.0, y: 1.0 }));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        assert_eq!(idx.get(&Score(10)).len(), 2);

        // Remove Score from e1 — entity stays alive, but loses the component
        world.remove::<Score>(e1);
        assert!(world.is_alive(e1));
        assert!(!world.has::<Score>(e1));

        // Raw get still returns stale entry
        assert_eq!(idx.get(&Score(10)).len(), 2);

        // get_valid filters it out
        let valid: Vec<_> = idx.get_valid(&Score(10), &world).collect();
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0], e2);

        // Rebuild cleans up
        idx.rebuild(&mut world);
        assert_eq!(idx.get(&Score(10)).len(), 1);
    }

    #[test]
    fn two_indexes_same_component_independent_update() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let _e2 = world.spawn((Score(20),));

        let mut btree = BTreeIndex::<Score>::new();
        let mut hash = HashIndex::<Score>::new();
        btree.rebuild(&mut world);
        hash.rebuild(&mut world);

        // Mutate e1
        *world.get_mut::<Score>(e1).unwrap() = Score(30);

        // Both indexes update sequentially — each has its own tick
        btree.update(&mut world);
        hash.update(&mut world);

        // Both should see the change
        assert!(btree.get(&Score(10)).is_empty());
        assert_eq!(btree.get(&Score(30)).len(), 1);

        assert!(hash.get(&Score(10)).is_empty());
        assert_eq!(hash.get(&Score(30)).len(), 1);
    }

    #[test]
    fn btree_range_valid_filters_removed_component() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10), Pos { x: 0.0, y: 0.0 }));
        let e2 = world.spawn((Score(15), Pos { x: 0.0, y: 0.0 }));
        let e3 = world.spawn((Score(20), Pos { x: 0.0, y: 0.0 }));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        // Remove Score from e2
        world.remove::<Score>(e2);

        // Raw range still returns all 3
        let raw: Vec<_> = idx.range(Score(5)..Score(25)).collect();
        assert_eq!(raw.len(), 3);

        // range_valid filters e2
        let valid: Vec<_> = idx.range_valid(Score(5)..Score(25), &world).collect();
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().any(|(_, e)| e == &e1));
        assert!(valid.iter().any(|(_, e)| e == &e3));
        assert!(!valid.iter().any(|(_, e)| e == &e2));
    }

    #[test]
    fn hash_get_valid_filters_removed_component() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42), Pos { x: 0.0, y: 0.0 }));
        let e2 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 }));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);
        assert_eq!(idx.get(&Score(42)).len(), 2);

        world.remove::<Score>(e1);

        // Raw get still returns stale entry
        assert_eq!(idx.get(&Score(42)).len(), 2);

        // get_valid filters it
        let valid: Vec<_> = idx.get_valid(&Score(42), &world).collect();
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0], e2);
    }

    #[test]
    fn btree_len_contains() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = BTreeIndex::<Score>::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);

        idx.rebuild(&mut world);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
        assert!(idx.contains(e1));
        assert!(idx.contains(e2));
        assert!(!idx.contains(Entity::DANGLING));
    }

    #[test]
    fn hash_len_contains() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));

        let mut idx = HashIndex::<Score>::new();
        assert!(idx.is_empty());

        idx.rebuild(&mut world);
        assert_eq!(idx.len(), 1);
        assert!(idx.contains(e1));
        assert!(!idx.contains(Entity::DANGLING));
    }

    #[test]
    fn btree_raw_parts_round_trip() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let (tree, reverse, last_sync) = idx.as_raw_parts();
        let restored =
            BTreeIndex::<Score>::from_raw_parts(tree.clone(), reverse.clone(), last_sync);

        assert_eq!(restored.get(&Score(10)).len(), 1);
        assert!(restored.get(&Score(10)).contains(&e1));
        assert_eq!(restored.get(&Score(20)).len(), 1);
        assert!(restored.get(&Score(20)).contains(&e2));
    }

    #[test]
    fn hash_raw_parts_round_trip() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);

        let (map, reverse, last_sync) = idx.as_raw_parts();
        let restored = HashIndex::<Score>::from_raw_parts(map.clone(), reverse.clone(), last_sync);

        assert_eq!(restored.get(&Score(10)).len(), 1);
        assert!(restored.get(&Score(10)).contains(&e1));
        assert_eq!(restored.get(&Score(20)).len(), 1);
        assert!(restored.get(&Score(20)).contains(&e2));
    }
}
