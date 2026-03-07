# Python ECS Bridge — Design Document

## Goal

Expose Minkowski's ECS directly to Python as an orchestration and analysis layer. Python users spawn entities, query state as Polars DataFrames, write modified data back, and call pre-compiled Rust reducers for hot loops. The ECS is the API, not a thing hidden behind opaque simulation wrappers.

## Key Decisions

- **Rust-defined components, Python-accessed by name** as Arrow columns. Adding a new component type requires Rust code + `maturin develop`.
- **Rust reducers as the "drop to Rust" path**, callable from Python by name with keyword parameters.
- **One-copy read path**: BlobVec `memcpy` → `arrow::RecordBatch`, then zero-copy via `pyo3-arrow` C Data Interface → PyArrow → Polars.
- **One-copy write path**: Arrow arrays received via `pyo3-arrow` (zero-copy in), then per-entity copy into BlobVec via `get_ptr_mut(row, tick)` to preserve change detection.
- **No dynamic component system**. No runtime schema definition. This is an orchestration layer, not a new engine.
- **Notebooks are the tests**. The ECS logic is tested by `cargo test -p minkowski` (344 tests). If notebooks run end-to-end, the bridge works.

## Architecture

```
┌─────────────────────────────────────┐
│  Python API (PyO3 #[pyclass])       │
│  World, ReducerRegistry, Entity IDs │
├─────────────────────────────────────┤
│  Arrow Bridge                       │
│  BlobVec → arrow::RecordBatch       │  one-copy (memcpy)
│  pyo3-arrow → PyArrow               │  zero-copy (C Data Interface)
│  PyArrow → Polars                   │  zero-copy
├─────────────────────────────────────┤
│  Minkowski ECS (existing crate)     │
│  World, Query, Reducers, BlobVec    │
└─────────────────────────────────────┘
```

### Component Registration

A Rust `ComponentSchema` maps each component struct to its Arrow representation — field names, Arrow data types, and byte offsets. Registered at startup in `#[pymodule]` init.

Example: `Position { x: f32, y: f32 }` maps to Arrow fields `[("pos_x", Float32), ("pos_y", Float32)]` with offsets `[0, 4]`.

### Query Path (read)

1. Python calls `world.query("Position", "Velocity")`
2. Rust resolves component names → `ComponentId`s
3. Uses pre-compiled query functions (boxed closures registered per component combination) to iterate results
4. `memcpy`s each column into an `arrow::array::Float32Array` buffer
5. Wraps columns in a `RecordBatch`, returns via `pyo3-arrow` (zero-copy FFI)
6. Python: `pl.DataFrame(arrow_table)` (zero-copy)

### Write-back Path

1. Python calls `world.write_column("Position", entity_ids, pos_x=array, pos_y=array)`
2. Rust receives Arrow arrays via `pyo3-arrow` (zero-copy FFI in)
3. For each entity, copies field values from Arrow arrays into BlobVec columns via `get_ptr_mut(row, tick)` (ensures change detection)

### Pre-compiled Query Functions

Rust queries are statically typed (`query::<(&Position, &Velocity)>()`), but Python requests components by name. Solved with a registration table of boxed closures — one per supported component combination. Users who add new components register new combos in Rust.

## Python API

```python
import minkowski_py as mk

# ── World lifecycle ──
world = mk.World()
registry = mk.ReducerRegistry(world)

# ── Spawning ──
entities = world.spawn_batch("Position,Velocity,Mass", {
    "pos_x": [0.0, 1.0, 2.0],
    "pos_y": [0.0, 1.0, 2.0],
    "vel_x": [1.0, 0.0, -1.0],
    "vel_y": [0.0, 1.0, 0.0],
    "mass":  [1.0, 1.0, 1.0],
})

e = world.spawn("Position,Velocity", pos_x=0.0, pos_y=0.0, vel_x=1.0, vel_y=0.0)

# ── Queries (read) ──
df = world.query("Position", "Velocity")       # Polars DataFrame
table = world.query_arrow("Position", "Velocity")  # PyArrow table

# ── Write-back ──
world.write_column("Position", entity_ids, pos_x=new_x, pos_y=new_y)

# ── Rust reducers ──
registry.run("movement", dt=0.016)
registry.run("gravity", g=0.06674, softening=1.0)

# ── Entity lifecycle ──
world.despawn(entity_id)
alive = world.is_alive(entity_id)

# ── Introspection ──
world.entity_count()
world.archetype_count()
world.component_names()
```

## Component Catalog

| Component | Fields | Arrow columns | Used by |
|-----------|--------|---------------|---------|
| `Position` | `x: f32, y: f32` | `pos_x, pos_y` | boids, nbody, flatworm, tactical |
| `Velocity` | `x: f32, y: f32` | `vel_x, vel_y` | boids, nbody |
| `Acceleration` | `x: f32, y: f32` | `acc_x, acc_y` | boids |
| `Mass` | `f32` | `mass` | nbody |
| `CellState` | `bool` | `alive` | life |
| `Heading` | `f32` | `heading` | flatworm |
| `Energy` | `f32` | `energy` | flatworm |
| `Health` | `u32` | `health` | tactical |
| `Faction` | `u8` | `faction` | tactical |

## Reducer Catalog

| Reducer | Type | Parameters | Description |
|---------|------|------------|-------------|
| `boids_forces` | QueryMut | `world_size, sep_r, ali_r, coh_r, weights, max_force` | Flocking force computation |
| `boids_integrate` | QueryMut | `max_speed, dt, world_size` | Velocity/position integration with wrapping |
| `gravity` | QueryMut | `g, softening, dt, world_size` | N-body gravitational forces + integration |
| `life_step` | Custom | `width, height` | One Game of Life generation |
| `movement` | QueryMut | `dt, world_size` | Simple heading-based movement with wrapping |

## Notebooks

| Notebook | Demonstrates |
|----------|-------------|
| `01_quickstart.ipynb` | Spawn, query as DataFrame, plot, write-back loop — core API in 5 minutes |
| `02_boids.ipynb` | Boids via Rust reducer, parameter sweeps, trajectory analysis |
| `03_nbody.ipynb` | Gravity reducer, energy analysis, constant sweep |
| `04_life.ipynb` | Life reducer, population curves, activity heatmaps |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `pyo3` | Python ↔ Rust FFI |
| `arrow` | Arrow array types + RecordBatch |
| `pyo3-arrow` | Zero-copy Arrow C Data Interface export |
| `minkowski` | ECS engine |
| `fastrand` | RNG for spawning helpers |

## What This Is Not

- Not a dynamic component system — no runtime schema definition
- Not a framework — Python orchestrates, Rust computes
- Not an attempt to make Python as fast as Rust — the copy is the cost of crossing the boundary, and it's one fast memcpy
