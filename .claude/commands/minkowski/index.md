---
description: Help with Minkowski ECS spatial indexing — SpatialIndex trait, grids, quadtrees, neighbor queries
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user implement spatial indexing for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing spatial patterns:

- `SpatialIndex` trait implementations
- Grid, quadtree, BVH, or k-d tree structs
- Neighbor query patterns (distance checks, radius searches, nearest-N)
- `world.is_alive()` checks on entity references from spatial queries
- `rebuild()` / `update()` call patterns
- `Changed<T>` usage for incremental spatial updates
- Position component types and their spatial distribution

## Step 2: Recommend

**Strong defaults:**
- **Only implement spatial indexing if you do spatial neighbor queries.** Not every simulation needs one.
- **Uniform density**: Use a grid. Simple, cache-friendly, O(1) cell lookup. Cell size should be roughly the interaction radius.
- **Clustered or hierarchical density**: Use a quadtree (2D) or octree (3D). Adapts to non-uniform distributions.
- **Implement `SpatialIndex` trait**: `rebuild(&mut self, world: &mut World)` is required (full reconstruction). `update(&mut self, world: &mut World)` is optional — override for incremental updates using `Changed<T>` to only re-index moved entities.
- **Always check `world.is_alive(entity)` on query results.** Despawned entities leave stale entries in the index. Generation mismatch catches them at query time. Full `rebuild()` cleans them up.
- **Indexes are external to World** — they compose from existing query primitives (`world.query()`, `world.is_alive()`). No `world.register_index()`.
- **Rebuild between frames, not during iteration.** Call `index.rebuild(&mut world)` before the frame's query reducers run.

**Ask if unclear:**
- "How often do entities move?" — If every frame: full `rebuild()` is simplest. If rarely: implement `update()` for incremental re-indexing using `Changed<Position>`.
- "What's the spatial distribution?" — Uniform: grid (tune cell size to interaction radius). Clustered: tree (quadtree/BVH). Mixed: consider a two-level structure.
- "What's the query pattern?" — Range queries (all within radius): grid or tree both work. Nearest-N: tree with priority queue. Ray casting: BVH.
- "How many entities?" — Under 1K: brute force may be fine. 1K-100K: grid or quadtree. 100K+: consider parallel rebuild.

## Step 3: Implement

Help write spatial index code. Point to relevant examples:

- **Uniform grid**: See `examples/examples/boids.rs` — `SpatialGrid` with cell-based neighbor lookup for 5K boids
- **Barnes-Hut quadtree**: See `examples/examples/nbody.rs` — `BarnesHutTree` with center-of-mass aggregation for 2K bodies
- **SpatialIndex trait pattern**:
  ```
  struct MyGrid { /* ... */ }
  impl SpatialIndex for MyGrid {
      fn rebuild(&mut self, world: &mut World) {
          self.clear();
          world.query::<(Entity, &Position)>().for_each(|(entity, pos)| {
              self.insert(entity, pos);
          });
      }
      // Optional: incremental update
      fn update(&mut self, world: &mut World) {
          world.query::<(Entity, &Position, Changed<Position>)>().for_each(|(entity, pos, _)| {
              self.reindex(entity, pos);
          });
      }
  }
  ```
- **Stale entity handling**: Always validate: `if world.is_alive(entity) { /* use it */ }`

For architecture details, see CLAUDE.md § "Secondary Indexes".
