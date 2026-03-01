# Secondary Index Hooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `SpatialIndex` lifecycle trait, a Barnes-Hut N-body example using a quadtree, and refactor the boids example to use the same trait.

**Architecture:** Minimal `SpatialIndex` trait (rebuild + optional incremental update) lives in the core crate. Indexes are fully external to World — they compose from existing query primitives. Despawned entities are handled via generational validation at query time. Two examples validate the trait against fundamentally different spatial structures (uniform grid vs. quadtree).

**Tech Stack:** Rust, minkowski ECS (existing query/entity API), no new dependencies

---

### Task 1: SpatialIndex trait + tests

**Files:**
- Create: `crates/minkowski/src/index.rs`
- Modify: `crates/minkowski/src/lib.rs:9-26`

**Step 1: Write the trait module with a test**

Create `crates/minkowski/src/index.rs`:

```rust
use crate::world::World;

/// A secondary spatial index that can be rebuilt from world state.
///
/// Indexes are fully user-owned — the World has no awareness of them.
/// Implementations use standard query primitives (`world.query()`,
/// `Changed<T>`) internally. Query methods are defined per concrete
/// type, not on this trait.
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
        let live: Vec<_> = idx.entities.iter().filter(|&&e| world.is_alive(e)).collect();
        assert_eq!(live.len(), 1);
        assert_eq!(*live[0], e2);
    }
}
```

**Step 2: Wire up in lib.rs**

Add `pub mod index;` and `pub use index::SpatialIndex;` to `crates/minkowski/src/lib.rs`.

Add after `pub mod table;` (line 16):
```rust
pub mod index;
```

Add after the `pub use world::World;` line (line 26):
```rust
pub use index::SpatialIndex;
```

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- index`
Expected: 3 tests pass

**Step 4: Run full test suite + clippy**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: All tests pass (should be 137 total), clippy clean

**Step 5: Commit**

```bash
git add crates/minkowski/src/index.rs crates/minkowski/src/lib.rs
git commit -m "feat: add SpatialIndex lifecycle trait"
```

---

### Task 2: Refactor boids spatial grid through SpatialIndex

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Goal:** Extract the inline `Vec<Vec<usize>>` grid into a `SpatialGrid` struct implementing `SpatialIndex`. The grid stores `(Entity, Vec2, Vec2)` tuples (entity, position, velocity) for neighbor queries. Same behavior, structured through the trait.

**Step 1: Add SpatialGrid struct and SpatialIndex impl**

Add after the `BoidParams` struct (around line 172), before constants:

```rust
use minkowski::SpatialIndex;

// ── Spatial Grid ───────────────────────────────────────────────────

struct SpatialGrid {
    cell_size: f32,
    grid_w: usize,
    world_size: f32,
    cells: Vec<Vec<usize>>,
    snapshot: Vec<(Entity, Vec2, Vec2)>,
}

impl SpatialGrid {
    fn new(cell_size: f32, world_size: f32) -> Self {
        let grid_w = (world_size / cell_size).ceil() as usize;
        Self {
            cell_size,
            grid_w,
            world_size,
            cells: Vec::new(),
            snapshot: Vec::new(),
        }
    }

    fn neighbors(&self, pos: Vec2) -> impl Iterator<Item = &(Entity, Vec2, Vec2)> {
        let cx = ((pos.x / self.cell_size) as usize).min(self.grid_w - 1);
        let cy = ((pos.y / self.cell_size) as usize).min(self.grid_w - 1);
        let grid_w = self.grid_w;
        (-1i32..=1).flat_map(move |dy| {
            (-1i32..=1).flat_map(move |dx| {
                let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                self.cells[ny * grid_w + nx]
                    .iter()
                    .map(move |&i| &self.snapshot[i])
            })
        })
    }
}

impl SpatialIndex for SpatialGrid {
    fn rebuild(&mut self, world: &mut World) {
        self.snapshot = world
            .query::<(Entity, &Position, &Velocity)>()
            .map(|(e, p, v)| (e, p.0, v.0))
            .collect();

        self.cells = vec![vec![]; self.grid_w * self.grid_w];
        for (i, &(_, pos, _)) in self.snapshot.iter().enumerate() {
            let cx = ((pos.x / self.cell_size) as usize).min(self.grid_w - 1);
            let cy = ((pos.y / self.cell_size) as usize).min(self.grid_w - 1);
            self.cells[cy * self.grid_w + cx].push(i);
        }
    }
}
```

**Step 2: Refactor main loop to use SpatialGrid**

Replace Steps 2-3 (the snapshot + grid build + force accumulation, roughly lines 230-315) with:

```rust
        // Step 2: Rebuild spatial index
        grid.rebuild(&mut world);

        // Step 3: Force accumulation — parallel, using grid neighbor queries
        let forces: Vec<(Entity, Vec2)> = {
            use rayon::prelude::*;
            grid.snapshot
                .par_iter()
                .map(|&(entity, pos, vel)| {
                    // ... same force accumulation logic, but using grid.neighbors(pos)
                    // instead of manual cell iteration
                })
                .collect()
        };
```

The force accumulation inner loop replaces the manual `grid[ny * grid_w + nx]` iteration with `grid.neighbors(pos)`.

Initialize the grid before the frame loop:
```rust
let mut grid = SpatialGrid::new(params.cohesion_radius, params.world_size);
```

**Step 3: Run the example to verify identical behavior**

Run: `cargo run -p minkowski --example boids --release`
Expected: Same output pattern as before — ~5000 entities, frame times in the 2-6ms range

**Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean

**Step 5: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "refactor: extract boids spatial grid into SpatialIndex impl"
```

---

### Task 3: Barnes-Hut quadtree — data structure

**Files:**
- Create: `crates/minkowski/examples/nbody.rs`

This task creates the quadtree data structure and its SpatialIndex impl. The simulation loop comes in Task 4.

**Step 1: Write the quadtree core**

Create `crates/minkowski/examples/nbody.rs` with:

1. **Vec2** — reuse the same pattern from boids (basic 2D vector math)
2. **Rect** — axis-aligned bounding box with `contains(point)` and `quadrant(point)` methods
3. **QuadNode** — `bounds: Rect`, `center_of_mass: Vec2`, `total_mass: f32`, `entity: Option<Entity>`, `children: Option<[usize; 4]>`
4. **BarnesHutTree** — `nodes: Vec<QuadNode>`, `bounds: Rect`, `theta: f32`

Methods on `BarnesHutTree`:
- `new(bounds, theta)` — creates root node
- `clear()` — resets to empty root, reuses allocation
- `insert(entity, pos, mass)` — recursive subdivision, splits leaf on collision, max depth 20
- `aggregate()` — bottom-up pass computing center_of_mass and total_mass for internal nodes
- `compute_force(pos, mass, world) -> Vec2` — recursive Barnes-Hut walk with `node_size / distance < theta` criterion. Calls `world.is_alive(entity)` at leaf nodes for generational validation.

**Step 2: Implement SpatialIndex**

```rust
impl SpatialIndex for BarnesHutTree {
    fn rebuild(&mut self, world: &mut World) {
        // Compute bounding box from all positions
        // Clear and rebuild tree
        // Insert all (Entity, &Position, &Mass) entities
        // Run aggregate() pass
    }
}
```

**Step 3: Add compile check**

Add a minimal `fn main()` placeholder so the example compiles:
```rust
fn main() {
    println!("N-body example (TODO: simulation loop)");
}
```

Run: `cargo build -p minkowski --example nbody`
Expected: Compiles

**Step 4: Commit**

```bash
git add crates/minkowski/examples/nbody.rs
git commit -m "feat: Barnes-Hut quadtree data structure"
```

---

### Task 4: N-body simulation loop

**Files:**
- Modify: `crates/minkowski/examples/nbody.rs`

**Step 1: Add components and constants**

```rust
#[derive(Clone, Copy)]
struct Position(Vec2);

#[derive(Clone, Copy)]
struct Velocity(Vec2);

#[derive(Clone, Copy)]
struct Mass(f32);

const ENTITY_COUNT: usize = 2_000;
const FRAME_COUNT: usize = 1_000;
const CHURN_INTERVAL: usize = 200;
const CHURN_COUNT: usize = 20;
const DT: f32 = 0.001;
const G: f32 = 6.674e-2;  // scaled for visual effect
const SOFTENING: f32 = 1.0;  // prevents singularities
const THETA: f32 = 0.5;
const WORLD_SIZE: f32 = 500.0;
```

**Step 2: Implement the simulation loop in main()**

Per-frame structure (same patterns as boids):

1. `tree.rebuild(&mut world)` — builds quadtree from all entities
2. Snapshot positions/masses for parallel force computation
3. Parallel force computation: for each entity, `tree.compute_force(pos, mass, &world)` returns gravitational acceleration. Uses `rayon::par_iter`.
4. Apply forces via `world.get_mut::<Velocity>(entity)`
5. Symplectic Euler integration via `for_each_chunk` (update velocity from force, update position from velocity)
6. Periodic spawn/despawn churn (every `CHURN_INTERVAL` frames) — despawn random entities, spawn replacements. This exercises the generational validation path.
7. Stats every `CHURN_INTERVAL` frames: entity count, frame time, total energy (kinetic + potential) as conservation diagnostic

**Step 3: Run the example**

Run: `cargo run -p minkowski --example nbody --release`
Expected: Runs 1000 frames, prints periodic stats, completes without panic

**Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean

**Step 5: Commit**

```bash
git add crates/minkowski/examples/nbody.rs
git commit -m "feat: Barnes-Hut N-body simulation example"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add nbody example to Build & Test Commands**

After the `life` example line (line 19), add:
```
cargo run -p minkowski --example nbody --release    # Barnes-Hut N-body (2K entities, 1K frames)
```

**Step 2: Add Secondary Indexes section to Architecture**

After the Deferred Mutation section (after line 94), add:

```markdown
### Secondary Indexes

`SpatialIndex` is a lifecycle trait for user-owned spatial data structures (grids, quadtrees, BVH, k-d trees). Indexes are fully external to World — they compose from existing query primitives. The trait has two methods: `rebuild` (required, full reconstruction) and `update` (optional, defaults to rebuild — override for incremental updates via `Changed<T>`). Despawned entities are handled via generational validation: stale entries are skipped at query time when `world.is_alive()` returns false, and cleaned up on the next rebuild.
```

**Step 3: Update Key Conventions**

Add `SpatialIndex` to the pub API list on line 98:
```
- `pub` for user-facing API (`World`, `Entity`, `CommandBuffer`, `Bundle`, `WorldQuery`, `Table`, `EnumChangeSet`, `Changed`, `ComponentId`, `SpatialIndex`).
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add SpatialIndex and nbody to CLAUDE.md"
```

---

### Task 6: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update Phase 4 description**

After the Phase 3 line (line 13), add:
```markdown
**Phase 4** — `SpatialIndex` lifecycle trait for user-owned secondary indexes (spatial grids, quadtrees, BVH, k-d trees). Indexes are external to World, compose from query primitives, and handle despawns via generational validation.
```

**Step 2: Add N-body example section**

After the Game of Life example section (around line 135), add an N-body example section with sample output, similar in style to the boids and life sections.

**Step 3: Update the roadmap table**

Mark Phase 4 secondary index hooks as done. Update the "What's next" section (line 149) to reflect Phase 4 completion.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add SpatialIndex and N-body example to README"
```

---

### Task 7: Push and create PR

**Step 1: Create branch and push**

```bash
git checkout -b feat/spatial-index
git push -u origin feat/spatial-index
```

**Step 2: Create PR**

```bash
gh pr create --title "feat: SpatialIndex trait + Barnes-Hut N-body example" --body "..."
```

PR body should cover:
- SpatialIndex trait (lifecycle only, no query abstraction)
- Boids refactored through the trait (validation)
- Barnes-Hut quadtree + N-body simulation example
- Generational validation for despawn handling
- CLAUDE.md + README updates
