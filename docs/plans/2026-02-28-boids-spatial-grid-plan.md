# Boids Spatial Grid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the O(N²) brute-force neighbor search in the boids example with a uniform spatial grid, reducing inner loop iterations from ~5,000 to ~450 per boid and improving cache locality.

**Architecture:** A `Vec<Vec<usize>>` grid local to the example, rebuilt each frame from the position snapshot. Cell size = max interaction radius (50.0). 3×3 neighbor lookup with toroidal wrapping. The grid is built sequentially; the force loop stays parallel via rayon.

**Tech Stack:** Rust, rayon (existing dependency)

---

### Task 1: Add helpers and grid data structure

Add the `wrapped_diff` helper for toroidal distance, and the grid build + lookup logic. This is all in `crates/minkowski/examples/boids.rs`.

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Add `wrapped_diff` helper**

Add after the `spawn_boid` function (after line 190), before `// ── Main`:

```rust
/// Minimum-image distance on a toroidal world.
/// Returns the shortest signed difference between `a` and `b` wrapping at `world_size`.
fn wrapped_diff(a: f32, b: f32, world_size: f32) -> f32 {
    let d = a - b;
    if d > world_size * 0.5 {
        d - world_size
    } else if d < -world_size * 0.5 {
        d + world_size
    } else {
        d
    }
}
```

**Step 2: Run to verify compilation**

Run: `cargo build -p minkowski --example boids --release`
Expected: Builds successfully (function is unused for now — that's fine).

**Step 3: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "feat: add wrapped_diff helper for toroidal distance

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Replace N² force loop with grid-accelerated neighbor search

Rewrite the force accumulation step to build a grid and iterate only nearby cells.

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Replace the force loop**

Replace the entire "Step 3: Force accumulation" block (lines 219-272) with:

```rust
        // Step 3: Build spatial grid + force accumulation (parallel)
        let cell_size = params.cohesion_radius; // largest interaction radius
        let grid_w = (params.world_size / cell_size).ceil() as usize;

        // Build grid: each cell contains snapshot indices of boids in that cell
        let mut grid: Vec<Vec<usize>> = vec![vec![]; grid_w * grid_w];
        for (i, &(_, pos, _)) in snapshot.iter().enumerate() {
            let cx = ((pos.x / cell_size) as usize).min(grid_w - 1);
            let cy = ((pos.y / cell_size) as usize).min(grid_w - 1);
            grid[cy * grid_w + cx].push(i);
        }

        // Force accumulation — iterate only 3x3 neighbor cells per boid
        let forces: Vec<(Entity, Vec2)> = {
            use rayon::prelude::*;
            snapshot
                .par_iter()
                .map(|&(entity, pos, vel)| {
                    let mut sep = Vec2::ZERO;
                    let mut ali = Vec2::ZERO;
                    let mut coh = Vec2::ZERO;
                    let mut sep_count = 0u32;
                    let mut ali_count = 0u32;
                    let mut coh_count = 0u32;

                    let cx = ((pos.x / cell_size) as usize).min(grid_w - 1);
                    let cy = ((pos.y / cell_size) as usize).min(grid_w - 1);

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                            let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                            for &j in &grid[ny * grid_w + nx] {
                                let (_, other_pos, other_vel) = snapshot[j];
                                let diff = Vec2::new(
                                    wrapped_diff(other_pos.x, pos.x, params.world_size),
                                    wrapped_diff(other_pos.y, pos.y, params.world_size),
                                );
                                let dist_sq = diff.length_sq();
                                if dist_sq < 1e-6 {
                                    continue;
                                }

                                let dist = dist_sq.sqrt();

                                if dist < params.separation_radius {
                                    sep = sep - diff.normalized() * (1.0 / dist);
                                    sep_count += 1;
                                }
                                if dist < params.alignment_radius {
                                    ali += other_vel;
                                    ali_count += 1;
                                }
                                if dist < params.cohesion_radius {
                                    coh += other_pos;
                                    coh_count += 1;
                                }
                            }
                        }
                    }

                    let mut force = Vec2::ZERO;
                    if sep_count > 0 {
                        force += sep / sep_count as f32 * params.separation_weight;
                    }
                    if ali_count > 0 {
                        let desired = ali / ali_count as f32 - vel;
                        force += desired * params.alignment_weight;
                    }
                    if coh_count > 0 {
                        let center = coh / coh_count as f32;
                        let desired = Vec2::new(
                            wrapped_diff(center.x, pos.x, params.world_size),
                            wrapped_diff(center.y, pos.y, params.world_size),
                        );
                        force += desired * params.cohesion_weight;
                    }

                    (entity, force.clamped(params.max_force))
                })
                .collect()
        };
```

Key changes from the N² version:
- Grid is built from snapshot (sequential, O(N))
- Inner loop iterates 3×3 cell neighborhood instead of all boids
- `other_pos - pos` replaced with `wrapped_diff` for correct toroidal distance
- Cohesion `center - pos` also uses `wrapped_diff` for correct averaging across boundaries
- Integer `rem_euclid` for cell wrapping (no float, no libm)

**Step 2: Run the example**

Run: `cargo run -p minkowski --example boids --release 2>&1`
Expected: Completes 1000 frames. Frame times should be significantly lower (~1-3ms vs ~5-7ms before). Entity count stable at 5000. `avg_vel` values may differ slightly from the N² version because the toroidal distance calculation changes which neighbors are considered near world edges — this is correct behavior.

**Step 3: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 4: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "perf: replace N² force loop with spatial grid neighbor search

Uniform grid with cell_size = cohesion_radius (50.0). Each boid
checks only its 3x3 cell neighborhood (~450 candidates) instead
of all 5,000. Uses wrapped_diff for correct toroidal distance.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update doc comment and verify

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Update the module doc comment**

The existing doc says "exercises: ... parallel iteration". Add a note about the spatial grid. After the existing "Exercises:" line (line 6), add:

```
//! The N² neighbor search is replaced by a uniform spatial grid — each boid
//! checks only a 3×3 cell neighborhood, reducing inner loop iterations from
//! ~5,000 to ~450 and improving cache locality.
```

**Step 2: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (the example has no unit tests, but engine tests should still pass).

**Step 3: Run the example one more time to confirm**

Run: `cargo run -p minkowski --example boids --release 2>&1 | head -5`
Expected: Frame times ~1-3ms, entities 5000.

**Step 4: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "docs: note spatial grid in boids module doc

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
