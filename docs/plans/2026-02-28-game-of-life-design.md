# Game of Life with Undo

## Purpose

A second example that exercises features the boids example doesn't: `Changed<T>`, `EnumChangeSet` with reversible apply/rollback, `#[derive(Table)]`, `query_table`/`query_table_mut`, and per-entity `get_mut`.

## Design

### Schema

```rust
#[derive(Clone, Copy)]
struct CellState(bool); // alive or dead

#[derive(Table, Clone, Copy)]
struct Cell {
    state: CellState,
    neighbor_count: NeighborCount,
}

#[derive(Clone, Copy)]
struct NeighborCount(u8);
```

### Grid

64×64 = 4,096 entities. Position is implicit from spawn order (row-major). A `Vec<Entity>` grid index maps `(x, y) → Entity` for neighbor lookup. Toroidal wrapping.

Initial state: random ~45% alive (using fastrand).

### Generation loop

Each generation:

1. **Identify dirty cells**: Query `Changed<CellState>` to find which cells changed last generation. For each changed cell, mark its 8 neighbors as needing recount.

2. **Recount neighbors**: For each cell needing recount, look up its 8 neighbors via grid index, count alive ones, write `NeighborCount` via `get_mut`.

3. **Apply rules**: For each cell, check `NeighborCount`:
   - Alive + (count < 2 or count > 3) → die
   - Dead + count == 3 → birth
   - Otherwise → no change

   Record state changes (births and deaths) in an `EnumChangeSet`.

4. **Apply changeset**: `changeset.apply(&mut world)` → push returned reverse onto undo stack.

5. **Print stats**: Every 50 generations, print generation number + alive count + frame time.

### Undo demonstration

After 500 forward generations:
- Rewind 50 generations by popping and applying reverse changesets from the undo stack
- Replay 50 generations forward (re-simulate, not replay stored changesets)
- Verify the alive count at gen 500 matches the original (deterministic simulation)

### Output

```
gen 000 | alive: 1847 | dt: 0.3ms
gen 050 | alive: 1203 | dt: 0.1ms
...
gen 500 | alive:  892 | dt: 0.1ms
── rewinding 50 generations ──
gen 499 | alive:  895
gen 498 | alive:  897
...
gen 450 | alive:  901
── replaying 50 generations ──
gen 500 | alive:  892 ✓ (matches original)
Done.
```

### Features exercised

| Feature | How |
|---------|-----|
| `#[derive(Table)]` | Cell schema declaration |
| `Changed<CellState>` | Dirty-flag for neighbor recounting |
| `EnumChangeSet` | Record births/deaths each generation |
| Reversible apply | Undo stack for rewind |
| `query_table` | Read cell data via typed table access |
| `get_mut` | Per-entity neighbor count + state updates |

### Files

- Create: `crates/minkowski/examples/life.rs`
