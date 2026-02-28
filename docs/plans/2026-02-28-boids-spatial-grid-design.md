# Boids Spatial Grid

## Problem

The N² force loop iterates all 5,000 boids for each boid — 25M distance checks per frame. The maximum interaction radius is 50.0 in a 500×500 world. Each boid has ~50 actual neighbors; the other 4,950 inner iterations are wasted distance checks that thrash the cache. The snapshot is ~80KB, and every boid touches every cache line in it.

## Design

A uniform grid local to the boids example. Not an engine feature — the grid is a `Vec<Vec<usize>>` rebuilt each frame from the snapshot. Cell size = `cohesion_radius` (50.0, the largest interaction radius). Grid dimensions = `ceil(500/50)` = 10×10 = 100 cells.

### Grid structure

```rust
let cell_size = params.cohesion_radius;
let grid_w = (params.world_size / cell_size).ceil() as usize;
let mut grid: Vec<Vec<usize>> = vec![vec![]; grid_w * grid_w];
```

Each cell stores snapshot indices. Insert is a single pass:

```rust
for (i, &(_, pos, _)) in snapshot.iter().enumerate() {
    let cx = ((pos.x / cell_size) as usize).min(grid_w - 1);
    let cy = ((pos.y / cell_size) as usize).min(grid_w - 1);
    grid[cy * grid_w + cx].push(i);
}
```

### Neighbor iteration

For each boid, iterate the 3×3 cell neighborhood (with toroidal wrapping):

```rust
let cx = (pos.x / cell_size) as usize;
let cy = (pos.y / cell_size) as usize;

for dy in [-1i32, 0, 1] {
    for dx in [-1i32, 0, 1] {
        let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
        let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
        for &j in &grid[ny * grid_w + nx] {
            let (_, other_pos, other_vel) = snapshot[j];
            // ... distance check + force accumulation
        }
    }
}
```

### Distance with wrapping

When checking neighbors across the world boundary, the naive `other_pos - pos` gives the wrong distance. Use the minimum image convention:

```rust
fn wrapped_diff(a: f32, b: f32, world_size: f32) -> f32 {
    let d = a - b;
    if d > world_size * 0.5 { d - world_size }
    else if d < -world_size * 0.5 { d + world_size }
    else { d }
}
```

### Expected improvement

- Inner loop: ~450 checks (9 cells × ~50) vs 5,000 → **~11× fewer iterations**
- Cache: touches ~450 snapshot entries instead of 5,000 per boid
- Grid build: O(N) single pass, negligible compared to force loop
- Parallelism: grid is built sequentially, force loop stays `par_iter` (grid read-only)

### Frame flow

```
1. Zero accelerations          (for_each_chunk — vectorized)
2. Snapshot positions/velocities
3. Build grid from snapshot     (sequential, O(N))
4. Force accumulation           (par_iter over boids, 3×3 grid neighbor lookup)
5. Apply forces
6. Integration                  (for_each_chunk — vectorized)
7. Churn + stats
```

### What changes

- **Added**: `grid` data structure, `build_grid` function, `wrapped_diff` helper
- **Changed**: Step 3 inner loop uses grid neighbors instead of full snapshot
- **Unchanged**: Everything else — the vectorized integration loops, the churn, the stats

### Future: engine-level secondary index hooks

The general pattern is: register a user-defined data structure that observes component changes (spawn, despawn, insert, or in-place mutation once change detection exists). This PR demonstrates the pattern with a manually-rebuilt grid. The engine hook (observer/notification API) is a separate design.

### Files

- Modify: `crates/minkowski/examples/boids.rs` — add grid, rewrite force loop
