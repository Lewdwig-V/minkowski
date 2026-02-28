//! Game of Life with undo — exercises Changed<T>, get_mut, and undo/replay.
//!
//! Run: cargo run -p minkowski --example life --release
//!
//! Features exercised:
//! - `Changed<CellState>` for detecting which cells mutated each generation
//! - `get_mut` for per-entity state updates via grid index
//! - Undo stack for time-travel (rewind + deterministic replay)

use minkowski::{Changed, Entity, World};
use std::time::Instant;

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CellState(bool);

#[derive(Clone, Copy)]
struct NeighborCount(u8);

// ── Constants ───────────────────────────────────────────────────────

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const CELL_COUNT: usize = WIDTH * HEIGHT;
const GENERATIONS: usize = 500;
const REWIND_GENS: usize = 50;

/// An undo entry: (grid_index, old_state).
type UndoEntry = (usize, bool);

// ── Helpers ─────────────────────────────────────────────────────────

/// Convert (col, row) to grid index.
fn idx(x: usize, y: usize) -> usize {
    y * WIDTH + x
}

/// Get the 8 neighbor indices for a cell with toroidal wrapping.
fn neighbor_indices(x: usize, y: usize) -> [usize; 8] {
    let left = if x == 0 { WIDTH - 1 } else { x - 1 };
    let right = if x == WIDTH - 1 { 0 } else { x + 1 };
    let up = if y == 0 { HEIGHT - 1 } else { y - 1 };
    let down = if y == HEIGHT - 1 { 0 } else { y + 1 };
    [
        idx(left, up),
        idx(x, up),
        idx(right, up),
        idx(left, y),
        idx(right, y),
        idx(left, down),
        idx(x, down),
        idx(right, down),
    ]
}

/// Snapshot all cell states into a local Vec<bool> to avoid aliasing.
fn snapshot_states(world: &World, grid: &[Entity]) -> Vec<bool> {
    grid.iter()
        .map(|&e| world.get::<CellState>(e).unwrap().0)
        .collect()
}

/// Count alive neighbors for every cell using a state snapshot.
fn count_neighbors(states: &[bool]) -> Vec<u8> {
    let mut counts = vec![0u8; CELL_COUNT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let neighbors = neighbor_indices(x, y);
            let mut count = 0u8;
            for &ni in &neighbors {
                if states[ni] {
                    count += 1;
                }
            }
            counts[idx(x, y)] = count;
        }
    }
    counts
}

/// Write neighbor counts into the world via get_mut.
fn write_neighbor_counts(world: &mut World, grid: &[Entity], counts: &[u8]) {
    for (i, &count) in counts.iter().enumerate() {
        if let Some(nc) = world.get_mut::<NeighborCount>(grid[i]) {
            nc.0 = count;
        }
    }
}

/// Apply Conway rules: returns Vec of (grid_index, old_state) for undo,
/// plus the new states to apply.
fn apply_rules(states: &[bool], counts: &[u8]) -> (Vec<UndoEntry>, Vec<UndoEntry>) {
    let mut undo = Vec::new();
    let mut updates = Vec::new();
    for i in 0..CELL_COUNT {
        let alive = states[i];
        let n = counts[i];
        let new_alive = match (alive, n) {
            (true, 2) | (true, 3) => true,
            (true, _) => false,
            (false, 3) => true,
            (false, _) => false,
        };
        if new_alive != alive {
            undo.push((i, alive)); // record old state for undo
            updates.push((i, new_alive));
        }
    }
    (undo, updates)
}

/// Apply state updates to the world.
fn apply_updates(world: &mut World, grid: &[Entity], updates: &[(usize, bool)]) {
    for &(i, new_state) in updates {
        if let Some(cs) = world.get_mut::<CellState>(grid[i]) {
            cs.0 = new_state;
        }
    }
}

/// Count alive cells from the world.
fn alive_count(world: &World, grid: &[Entity]) -> usize {
    grid.iter()
        .filter(|&&e| world.get::<CellState>(e).unwrap().0)
        .count()
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let mut world = World::new();

    // Spawn 64x64 grid in row-major order
    let mut grid = Vec::with_capacity(CELL_COUNT);
    for _ in 0..CELL_COUNT {
        let alive = fastrand::f32() < 0.45;
        let e = world.spawn((CellState(alive), NeighborCount(0)));
        grid.push(e);
    }

    // Initial neighbor count
    {
        let states = snapshot_states(&world, &grid);
        let counts = count_neighbors(&states);
        write_neighbor_counts(&mut world, &grid, &counts);
    }

    println!(
        "Game of Life: {}x{} grid, {} cells, {} generations",
        WIDTH, HEIGHT, CELL_COUNT, GENERATIONS
    );
    println!("Initial alive: {}", alive_count(&world, &grid));
    println!();

    // ── Generation loop ─────────────────────────────────────────────
    let mut undo_stack: Vec<Vec<(usize, bool)>> = Vec::with_capacity(GENERATIONS);

    for gen in 0..GENERATIONS {
        let frame_start = Instant::now();

        // Snapshot states, recount neighbors
        let states = snapshot_states(&world, &grid);
        let counts = count_neighbors(&states);
        write_neighbor_counts(&mut world, &grid, &counts);

        // Apply Conway rules
        let (undo, updates) = apply_rules(&states, &counts);
        let change_count = updates.len();
        apply_updates(&mut world, &grid, &updates);

        // Exercise Changed<CellState> — archetype-level, so if any changed,
        // the query returns all cells in that archetype.
        let _changed_count = world.query::<(Changed<CellState>,)>().count();

        // Push undo record
        undo_stack.push(undo);

        // Print stats every 50 generations
        if gen % 50 == 0 || gen == GENERATIONS - 1 {
            let dt_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "gen {:>4} | alive: {:>4} | changes: {:>4} | dt: {:.2}ms",
                gen,
                alive_count(&world, &grid),
                change_count,
                dt_ms,
            );
        }
    }

    // Record alive count at gen 499 (before rewind)
    let pre_rewind_alive = alive_count(&world, &grid);
    println!();
    println!(
        "Pre-rewind alive count (gen {}): {}",
        GENERATIONS - 1,
        pre_rewind_alive
    );

    // ── Rewind 50 generations ───────────────────────────────────────
    println!();
    println!("Rewinding {} generations...", REWIND_GENS);

    for i in 0..REWIND_GENS {
        let undo = undo_stack.pop().expect("undo stack underflow");
        // Restore old states (undo entries contain old_state)
        for &(grid_idx, old_state) in &undo {
            if let Some(cs) = world.get_mut::<CellState>(grid[grid_idx]) {
                cs.0 = old_state;
            }
        }
        if i % 10 == 0 {
            println!(
                "  rewind step {:>2} | alive: {:>4}",
                i,
                alive_count(&world, &grid)
            );
        }
    }

    let rewound_alive = alive_count(&world, &grid);
    println!(
        "Rewound to gen {} | alive: {}",
        GENERATIONS - 1 - REWIND_GENS,
        rewound_alive
    );

    // ── Replay 50 generations ───────────────────────────────────────
    println!();
    println!("Replaying {} generations...", REWIND_GENS);

    for i in 0..REWIND_GENS {
        let states = snapshot_states(&world, &grid);
        let counts = count_neighbors(&states);
        write_neighbor_counts(&mut world, &grid, &counts);

        let (_undo, updates) = apply_rules(&states, &counts);
        apply_updates(&mut world, &grid, &updates);

        if i % 10 == 0 {
            println!(
                "  replay step {:>2} | alive: {:>4}",
                i,
                alive_count(&world, &grid)
            );
        }
    }

    let post_replay_alive = alive_count(&world, &grid);
    println!();
    println!("Post-replay alive count: {}", post_replay_alive);
    println!("Pre-rewind alive count:  {}", pre_rewind_alive);

    assert_eq!(
        post_replay_alive, pre_rewind_alive,
        "alive count after replay ({}) must match pre-rewind count ({})",
        post_replay_alive, pre_rewind_alive
    );

    println!("Verification passed: alive counts match.");
}
