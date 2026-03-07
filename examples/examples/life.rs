//! Game of Life with undo — exercises `#[derive(Table)]`, reducers, Changed<T>, EnumChangeSet,
//! and undo/replay.
//!
//! Run: cargo run -p minkowski-examples --example life --release
//!
//! Features exercised:
//! - `#[derive(Table)]` for typed row access that skips archetype matching
//! - `QueryMut` reducer for writing neighbor counts (Table + reducer integration)
//! - `query_table` for bulk reads over the cell grid
//! - `Changed<CellState>` for detecting which cells mutated each generation
//! - `EnumChangeSet::insert` for recording typed mutations with automatic undo
//! - Reversible changesets for time-travel (rewind + deterministic replay)

use minkowski::{
    Changed, Entity, EnumChangeSet, QueryMut, QueryReducerId, ReducerRegistry, Table, World,
};
use std::time::Instant;

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CellState(bool);

#[derive(Clone, Copy)]
struct NeighborCount(u8);

#[derive(Clone, Copy, Table)]
struct Cell {
    state: CellState,
    neighbors: NeighborCount,
}

// ── Constants ───────────────────────────────────────────────────────

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const CELL_COUNT: usize = WIDTH * HEIGHT;
const GENERATIONS: usize = 500;
const REWIND_GENS: usize = 50;

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
/// Uses `query_table` for typed row access without archetype matching.
fn snapshot_states(world: &mut World) -> Vec<bool> {
    world.query_table::<Cell>().map(|row| row.state.0).collect()
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

/// Apply Conway rules: returns (grid_index, new_state) for each cell that changed.
fn apply_rules(states: &[bool], counts: &[u8]) -> Vec<(usize, bool)> {
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
            updates.push((i, new_alive));
        }
    }
    updates
}

/// Build an EnumChangeSet from cell state updates and apply it.
/// Returns the reverse changeset for undo.
///
/// Uses EnumChangeSet directly (not a reducer) to capture reverse changesets
/// for undo/redo — this is the core of the time-travel demo.
fn apply_updates(world: &mut World, grid: &[Entity], updates: &[(usize, bool)]) -> EnumChangeSet {
    let mut cs = EnumChangeSet::new();
    for &(i, new_state) in updates {
        cs.insert::<CellState>(world, grid[i], CellState(new_state));
    }
    cs.apply(world)
}

/// Count alive cells from the world via `query_table`.
fn alive_count(world: &mut World) -> usize {
    world
        .query_table::<Cell>()
        .filter(|row| row.state.0)
        .count()
}

/// Write neighbor counts directly via `query_table_mut` — demonstrates
/// mixed read/write named field access on a single row (read `row.state`,
/// write `row.neighbors`) without tuple destructuring or archetype matching.
fn write_neighbors_via_table(world: &mut World, counts: &[u8]) {
    for (i, row) in world.query_table_mut::<Cell>().enumerate() {
        *row.neighbors = NeighborCount(counts[i]);
    }
}

/// Register a QueryMut reducer that writes pre-computed neighbor counts
/// into the NeighborCount column. Demonstrates Table + reducer integration.
fn register_write_neighbors(registry: &mut ReducerRegistry, world: &mut World) -> QueryReducerId {
    registry.register_query::<(&mut NeighborCount,), Vec<u8>, _>(
        world,
        "write_neighbor_counts",
        |mut query: QueryMut<'_, (&mut NeighborCount,)>, counts: Vec<u8>| {
            let mut i = 0;
            query.for_each(|(nc,)| {
                nc.0 = counts[i];
                i += 1;
            });
        },
    )
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let mut world = World::new();

    // Set up reducer registry with a QueryMut reducer for neighbor counting
    let mut registry = ReducerRegistry::new();
    let write_neighbors_id = register_write_neighbors(&mut registry, &mut world);

    // Spawn 64x64 grid in row-major order
    let mut grid = Vec::with_capacity(CELL_COUNT);
    for _ in 0..CELL_COUNT {
        let alive = fastrand::f32() < 0.45;
        let e = world.spawn(Cell {
            state: CellState(alive),
            neighbors: NeighborCount(0),
        });
        grid.push(e);
    }

    // Initial neighbor count via reducer
    {
        let states = snapshot_states(&mut world);
        let counts = count_neighbors(&states);
        registry.run(&mut world, write_neighbors_id, counts);
    }

    println!(
        "Game of Life: {}x{} grid, {} cells, {} generations",
        WIDTH, HEIGHT, CELL_COUNT, GENERATIONS
    );
    println!("Initial alive: {}", alive_count(&mut world));
    println!();

    // ── Generation loop ─────────────────────────────────────────────
    let mut undo_stack: Vec<EnumChangeSet> = Vec::with_capacity(GENERATIONS);

    for gen in 0..GENERATIONS {
        let frame_start = Instant::now();

        // Snapshot states, recount neighbors
        let states = snapshot_states(&mut world);
        let counts = count_neighbors(&states);
        // Alternate between reducer and query_table_mut to exercise both paths
        if gen % 2 == 0 {
            registry.run(&mut world, write_neighbors_id, counts.clone());
        } else {
            write_neighbors_via_table(&mut world, &counts);
        }

        // Apply Conway rules via EnumChangeSet — automatic undo capture.
        // Uses EnumChangeSet directly (not a reducer) to capture reverse
        // changesets for undo/redo.
        let updates = apply_rules(&states, &counts);
        let change_count = updates.len();
        let reverse = apply_updates(&mut world, &grid, &updates);

        // Exercise Changed<CellState> — archetype-level, so if any changed,
        // the query returns all cells in that archetype.
        let _changed_count = world.query::<(Changed<CellState>,)>().count();

        // Push reverse changeset for undo
        undo_stack.push(reverse);

        // Print stats every 50 generations
        if gen % 50 == 0 || gen == GENERATIONS - 1 {
            let dt_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "gen {:>4} | alive: {:>4} | changes: {:>4} | dt: {:.2}ms",
                gen,
                alive_count(&mut world),
                change_count,
                dt_ms,
            );
        }
    }

    // Record alive count at gen 499 (before rewind)
    let pre_rewind_alive = alive_count(&mut world);
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
        let reverse = undo_stack.pop().expect("undo stack underflow");
        // Apply the reverse changeset — restores previous cell states automatically
        let _ = reverse.apply(&mut world);
        if i % 10 == 0 {
            println!(
                "  rewind step {:>2} | alive: {:>4}",
                i,
                alive_count(&mut world)
            );
        }
    }

    let rewound_alive = alive_count(&mut world);
    println!(
        "Rewound to gen {} | alive: {}",
        GENERATIONS - 1 - REWIND_GENS,
        rewound_alive
    );

    // ── Replay 50 generations ───────────────────────────────────────
    println!();
    println!("Replaying {} generations...", REWIND_GENS);

    for i in 0..REWIND_GENS {
        let states = snapshot_states(&mut world);
        let counts = count_neighbors(&states);
        registry.run(&mut world, write_neighbors_id, counts.clone());

        let updates = apply_rules(&states, &counts);
        let _ = apply_updates(&mut world, &grid, &updates);

        if i % 10 == 0 {
            println!(
                "  replay step {:>2} | alive: {:>4}",
                i,
                alive_count(&mut world)
            );
        }
    }

    let post_replay_alive = alive_count(&mut world);
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
