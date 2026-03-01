# Secondary Index Hooks Design

## Goal

Add a `SpatialIndex` trait that provides a shared lifecycle vocabulary for user-owned secondary indexes (spatial grids, quadtrees, BVH, k-d trees, etc.), plus a Barnes-Hut N-body simulation example using a quadtree. Refactor the existing boids grid through the same trait as a validation.

## Philosophy

Indexes are **fully external** to World. World has no awareness of them. The trait composes from existing public primitives (`world.query()`, `Changed<T>`, `world.is_alive()`). World's API surface does not grow.

## The `SpatialIndex` Trait

```rust
pub trait SpatialIndex {
    /// Reconstruct the index from scratch by scanning all matching entities.
    fn rebuild(&mut self, world: &mut World);

    /// Incrementally update the index. Defaults to full rebuild.
    fn update(&mut self, world: &mut World) {
        self.rebuild(world);
    }
}
```

- **`rebuild`** (required): full reconstruction from world state. Queries all matching entities and builds the index from scratch.
- **`update`** (optional, defaults to rebuild): incremental path. Implementations can override to use `Changed<T>` queries internally for O(k) updates where k << n.
- Takes `&mut World` because queries require `&mut self` for cache mutation.
- No query interface on the trait — each index type defines its own query methods natively (grid has `query_cell()`, quadtree has `compute_force()`, etc.).

## Despawn Handling

Despawned entities are handled via **generational validation**: stale index entries are skipped at query time when `world.is_alive(entity)` returns false (generation mismatch). No despawn log, no World changes, no new infrastructure. Stale entries are cleaned up on the next `rebuild()`.

## Barnes-Hut Quadtree (Example)

**File**: `crates/minkowski/examples/nbody.rs`

### Data Structure

```
BarnesHutTree {
    nodes: Vec<QuadNode>,   // arena-allocated, index 0 = root
    bounds: Rect,
    theta: f32,             // opening angle (accuracy vs speed)
}

QuadNode {
    bounds: Rect,
    center_of_mass: Vec2,
    total_mass: f32,
    entity: Option<Entity>,
    children: Option<[usize; 4]>,
}
```

- Standard recursive subdivision on insert; splits leaf when it gains a second entity.
- Bottom-up mass aggregation pass after all inserts.
- Force query: recursive tree walk using Barnes-Hut opening criterion (`node_size / distance < theta`). Leaf nodes with stale entities (generation mismatch) are skipped.

### N-Body Simulation Loop

**Components**: `Position(Vec2)`, `Velocity(Vec2)`, `Mass(f32)`
**Constants**: ~2,000 entities, 1,000 frames, theta = 0.5

Per frame:
1. `tree.rebuild(&mut world)` — query all `(Entity, &Position, &Mass)`, build quadtree, aggregate masses
2. Compute forces via Barnes-Hut walk (parallel over entity snapshot)
3. Symplectic Euler integration via `for_each_chunk` (vectorizable)
4. Periodic spawn/despawn churn (exercises stale-entry validation)
5. Print stats: frame time, entity count, total energy (conservation diagnostic)

## Boids Refactor

Extract the inline `Vec<Vec<usize>>` spatial grid from `boids.rs` into a `SpatialGrid` struct implementing `SpatialIndex`. Same behavior, structured through the trait. Validates that the trait works for simple grid structures alongside the quadtree.

## Deliverables

| File | Change |
|---|---|
| `crates/minkowski/src/index.rs` | New: `SpatialIndex` trait (~15 lines) |
| `crates/minkowski/src/lib.rs` | Add `pub mod index;` + re-export |
| `crates/minkowski/examples/nbody.rs` | New: quadtree + Barnes-Hut N-body (~300-400 lines) |
| `crates/minkowski/examples/boids.rs` | Refactor: extract grid into `SpatialGrid` impl |
| `CLAUDE.md` | Document trait and philosophy |
| `README.md` | Mention secondary indexes, add N-body example |

## What Doesn't Change

World, query system, change detection, EntityAllocator, archetypes, BlobVec — nothing. The trait composes from existing public API only.
