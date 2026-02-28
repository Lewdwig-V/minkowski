# Game of Life with Undo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Game of Life example that exercises `Changed<T>`, per-entity `get_mut`, and undo/replay — features the boids example doesn't cover.

**Architecture:** 64×64 grid of cell entities with `CellState(bool)` and `NeighborCount(u8)` components. Each generation detects dirty cells via `Changed<CellState>`, recounts neighbors, applies rules, and stores old states on an undo stack. After 500 generations, rewinds 50 via the undo stack, replays 50 forward, and verifies deterministic match.

**Tech Stack:** Rust, `fastrand` (existing dev dep)

---

### Task 1: Create the life example

**Files:**
- Create: `crates/minkowski/examples/life.rs`

**Requirements:**

The example must:

1. **Components**: `CellState(bool)` and `NeighborCount(u8)` — both `Clone + Copy`.

2. **Grid**: 64×64 = 4,096 entities. Spawn in row-major order. Store `Vec<Entity>` as grid index. Toroidal wrapping for neighbor lookup. Initial state: `fastrand::f32() < 0.45` for alive.

3. **Initial neighbor count**: After spawning all cells, compute neighbor counts for every cell via `get_mut::<NeighborCount>`.

4. **Generation loop** (500 generations):
   - Recount neighbors for all cells (read `CellState` via `world.get`, write `NeighborCount` via `world.get_mut`)
   - Apply Conway rules: alive + (<2 or >3 neighbors) → death; dead + 3 neighbors → birth
   - Collect changes as `Vec<(Entity, bool)>` (entity, old_state)
   - Apply new states via `world.get_mut::<CellState>`
   - Push the old-state vec onto `undo_stack: Vec<Vec<(Entity, bool)>>`
   - Use `world.query::<(Changed<CellState>,)>().count()` to verify change detection fires when cells mutate (assert it's > 0 when changes occurred)
   - Print stats every 50 generations: gen number, alive count, change count, frame time

5. **Rewind**: Pop 50 entries from undo stack, restore old states via `get_mut::<CellState>`, print alive count every 10 generations.

6. **Replay**: Re-simulate 50 generations forward (same rules, no undo recording), print alive count every 10 generations.

7. **Verify**: Assert final alive count matches the count at gen 499 (before rewind). Print match/mismatch.

**Output format:**
```
Game of Life 64x64 — 500 generations

gen 000 | alive: 1847 | changes: 1204 | dt: 1.2ms
gen 050 | alive: 1203 | changes:   82 | dt: 0.3ms
...
gen 499 | alive:  892 | changes:   14 | dt: 0.2ms

── rewinding 50 generations ──
gen 499 | alive:  892
gen 489 | alive:  901
...
gen 450 | alive:  923

── replaying 50 generations ──
gen 450 | alive:  923
gen 460 | alive:  912
...
gen 499 | alive:  892 ✓ (matches original)
Done.
```

**API note:** `EnumChangeSet` is exported but `ComponentId` is `pub(crate)`, so users can't construct changesets from outside the crate. The example uses a manual `Vec<(Entity, bool)>` undo stack instead. This exercises the same undo pattern without requiring internal API access.

**Doc comment at top of file:**
```rust
//! Game of Life with undo — exercises Changed<T>, get_mut, and undo/replay.
//!
//! Run: cargo run -p minkowski --example life --release
//!
//! Features exercised:
//! - `Changed<CellState>` for detecting which cells mutated each generation
//! - `get_mut` for per-entity state updates via grid index
//! - Undo stack for time-travel (rewind + deterministic replay)
//!
//! Features NOT exercised here (covered by boids):
//! - `for_each_chunk` / SIMD vectorization
//! - `par_for_each` / rayon parallelism
//! - `CommandBuffer` deferred mutation
//! - Spatial grid / archetype churn
```

**Step 1: Write the example**

Write `crates/minkowski/examples/life.rs` following the requirements above.

**Step 2: Build and run**

Run: `cargo run -p minkowski --example life --release 2>&1`
Expected: Completes with deterministic match (alive count at gen 499 matches after replay).

**Step 3: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 4: Commit**

```bash
git add crates/minkowski/examples/life.rs
git commit -m "feat: add Game of Life example with undo and Changed<T>

64x64 grid, 500 generations with undo/replay. Exercises Changed<T>,
get_mut, and per-entity undo stack — features boids doesn't cover.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Final verification

**Step 1: Full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass.

**Step 2: Run both examples**

Run: `cargo run -p minkowski --example life --release 2>&1 | tail -5`
Run: `cargo run -p minkowski --example boids --release 2>&1 | tail -5`
Expected: Both complete successfully.
