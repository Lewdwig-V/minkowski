# Python SpatialGrid + Jupyter Notebooks Design

**Date**: 2026-03-07

## SpatialGrid

### API

```python
grid = mk.SpatialGrid(world_size=500.0, cell_size=25.0)
grid.rebuild(world)                    # scan all Position entities
neighbors = grid.query_radius(x, y, r) # → list[int] (entity IDs)
grid.entity_count                       # number of indexed entities
```

### Implementation

- `#[pyclass]` in `crates/minkowski-py/src/spatial.rs`
- Rust `HashMap<(i32, i32), Vec<(Entity, f32, f32)>>` — cell → entities with positions
- `rebuild()` queries `world.query::<(Entity, &Position)>()`
- `query_radius()` scans relevant cells, toroidal wrapping via `rem_euclid`, minimum-image distance
- Always toroidal (constructor takes `world_size` + `cell_size`)
- Returns entity IDs only — component data fetched via `world.query()`

### Touch points

1. New file `crates/minkowski-py/src/spatial.rs`
2. `lib.rs` — register in pymodule
3. No changes to core `minkowski` crate

## Notebooks

Four notebooks in `crates/minkowski-py/notebooks/`:

### 1. quickstart.ipynb
- Spawn entities, query as Polars DataFrame, write back, despawn
- No simulation loop — just API walkthrough

### 2. boids.ipynb
- Spawn 1K boids (Position, Velocity, Acceleration)
- Per-frame: `boids_forces` → `boids_integrate`
- Scatter plot animation colored by speed
- Demonstrate SpatialGrid for custom neighbor analysis

### 3. nbody.ipynb
- Spawn 200 bodies (Position, Velocity, Mass)
- Per-frame: `gravity` reducer
- Scatter plot animation, point size ∝ mass
- Keep N small — O(N²) in Python bridge reducer

### 4. life.ipynb
- Spawn 64×64 grid of CellState entities
- Per-frame: `life_step` reducer
- Heatmap animation (imshow, binary colormap)

### Visualization

All notebooks use matplotlib (no extra deps beyond polars + matplotlib).
Animation via `IPython.display.clear_output` + loop, not FuncAnimation (simpler, works in all Jupyter environments).

### Environment Setup

```bash
cd crates/minkowski-py
uv sync --all-extras && source .venv/bin/activate
maturin develop --release
jupyter lab notebooks/quickstart.ipynb
```

`uv sync --all-extras` creates a venv and installs all dependencies (polars, pyarrow,
matplotlib, numpy, jupyter, maturin). `maturin develop --release` builds the native
module into the venv.

## What's NOT in scope

- No incremental `update()` on grid — rebuild per frame is fine for notebooks
- No quadtree/BVH — users pull positions into numpy for scipy KDTree
- No new Rust components or reducers needed — all examples work with current bridge
