---
description: Help with Minkowski ECS indexing — SpatialIndex trait, BTreeIndex, HashIndex, grids, quadtrees, batch lookups
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user implement indexing for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing index patterns:

- `SpatialIndex` trait implementations (grids, quadtrees, BVH)
- `BTreeIndex` / `HashIndex` usage for column indexes
- `world.get_batch` / `world.get_batch_mut` for index-driven batch lookups
- `world.query_changed_since` / `ChangeTick` for incremental index updates
- `world.is_alive()` / `world.has::<T>()` checks on index query results
- `rebuild()` / `update()` call patterns
- Position component types and their spatial distribution

## Step 2: Recommend

### Choosing an index type

- **"I need all entities with component value in a range"** -> `BTreeIndex<T>` (T: Ord). Range queries, sorted iteration, O(log n) lookups.
- **"I need all entities with exact component value"** -> `HashIndex<T>` (T: Hash + Eq). O(1) exact lookups.
- **"I need spatial neighbor queries (within radius, nearest-N)"** -> Implement `SpatialIndex` trait with a grid or tree.
- **"I just need to iterate all entities with a component"** -> Use `world.query()` directly. No index needed.

### Column indexes (BTreeIndex, HashIndex)

**Strong defaults:**
- **Index on whole components** (not fields). Components are what ECS stores — `BTreeIndex<Score>`, not `BTreeIndex<u32>`.
- **External to World** — same composition pattern as SpatialIndex. No `world.register_index()`.
- **Each index tracks its own `ChangeTick`** for incremental updates via `world.query_changed_since()`. Two indexes on the same component type update independently.
- **Two-tier lookup**: `get()`/`range()` return raw results (may include stale entries). `get_valid()`/`range_valid()` filter via `world.has::<T>()` (slower but correct). Use raw for speed when you'll validate anyway; use valid when you need clean results.
- **Compose with `get_batch`** for efficient multi-entity component fetch after index narrowing:
  ```
  let candidates: Vec<Entity> = score_index
      .range_valid(Score(50)..Score(100), &world)
      .collect();
  let positions = world.get_batch::<Position>(&candidates);
  ```
- **Rebuild periodically** to reclaim memory from stale entries (despawns, component removals).

### Spatial indexes (SpatialIndex trait)

**Strong defaults:**
- **Only implement spatial indexing if you do spatial neighbor queries.** Not every simulation needs one.
- **Uniform density**: Grid. Cell size ~ interaction radius.
- **Clustered density**: Quadtree (2D), octree (3D), BVH.
- **Implement `SpatialIndex` trait**: `rebuild` is required. Override `update` for incremental updates using `world.query_changed_since()`.
- **Always check `world.is_alive(entity)` on query results.** Stale entries cleaned up on next `rebuild()`.
- **Rebuild between frames, not during iteration.**

### Batch lookups (get_batch / get_batch_mut)

- **Use `get_batch` when fetching a component for many entities from an index.** Groups by archetype internally — resolves ComponentId once, sequential memory access per archetype.
- **`get_batch_mut` panics on duplicate entities** (aliased `&mut T` is UB). Unconditional check.
- **For multiple components**, call `get_batch` once per component type. Each call groups by archetype independently.

**Ask if unclear:**
- "How often do entities move/change?" — Every frame: `rebuild()`. Rarely: `update()` with per-index ChangeTick.
- "What's the query pattern?" — Exact match: HashIndex. Range/sorted: BTreeIndex. Spatial: grid/tree.
- "How many candidates from the index?" — Under ~10: per-entity `get()` is fine. Over ~10: `get_batch` for cache locality.

## Step 3: Implement

Help write index code. Point to relevant examples:

- **BTreeIndex + HashIndex**: See `examples/examples/index.rs` — range queries, exact lookups, incremental update, stale detection, batch fetch composition
- **Uniform grid**: See `examples/examples/boids.rs` — `SpatialGrid` with cell-based neighbor lookup for 5K boids
- **Barnes-Hut quadtree**: See `examples/examples/nbody.rs` — `BarnesHutTree` with center-of-mass aggregation for 2K bodies

**Column index pattern:**
```
let mut index = BTreeIndex::<Score>::new();
index.rebuild(&mut world);

// Incremental update (each index tracks its own tick)
index.update(&mut world);

// Range query → batch fetch
let high: Vec<Entity> = index.range(Score(90)..).flat_map(|(_, e)| e).copied().collect();
let names = world.get_batch::<Name>(&high);
```

**SpatialIndex trait pattern:**
```
struct MyGrid { /* ... */ }
impl SpatialIndex for MyGrid {
    fn rebuild(&mut self, world: &mut World) {
        self.clear();
        world.query::<(Entity, &Position)>().for_each(|(entity, pos)| {
            self.insert(entity, pos);
        });
    }
    fn update(&mut self, world: &mut World) {
        let changed = world.query_changed_since::<Position>(self.last_sync);
        for (entity, pos) in changed {
            self.reindex(entity, pos);
        }
        self.last_sync = world.change_tick();
    }
}
```

For architecture details, see CLAUDE.md § "Secondary Indexes".
