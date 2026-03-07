---
description: Help with Minkowski ECS Python bindings — adding components and reducers to the bridge, setting up maturin/uv build tooling, calling the ECS from Python/Jupyter. Use when "Python binding", "add component to Python", "register reducer for Python", "maturin", "Jupyter notebook", "query from Python", "write_column", "spawn from Python", "PyO3 bridge", "pyo3-arrow".
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

# Minkowski Python Bridge

## Overview

The Python bridge exposes Minkowski's ECS directly via PyO3. Rust owns storage and computation; Python owns orchestration and analysis. Data crosses the boundary as Arrow RecordBatches (one-copy BlobVec→Arrow, zero-copy Arrow→Polars via `pyo3-arrow`).

## Architecture

```
Python (orchestration)  →  PyO3 bridge  →  Minkowski ECS (storage)
  world.spawn(...)           typed dispatch     world.spawn((Pos, Vel))
  world.query(...)           Arrow bridge       archetype_column_ptr → RecordBatch
  world.write_column(...)    typed dispatch     world.get_mut::<T>()
  registry.run(...)          ReducerRegistry    register_query / run
```

## Quick Reference: Files to Touch

| Task | Files |
|------|-------|
| Add a component | `components.rs`, `pyworld.rs` (spawn + write dispatch) |
| Add a reducer | `reducers.rs` (impl + register + dispatch) |
| Add a bundle combo | `pyworld.rs` (`spawn_typed` match) |
| Build bindings | `cd crates/minkowski-py && uv venv && uv pip install -e ".[dev]" && maturin develop --release` |

## Adding a Component

Four touch points. Example: adding `Armor(u32)`.

### 1. Define the struct (`components.rs`)

```rust
#[derive(Clone, Copy)]
#[repr(transparent)]  // newtypes use transparent; multi-field use #[repr(C)]
pub struct Armor(pub u32);
```

### 2. Register the schema (`components.rs` → `register_all`)

```rust
register_schema!(registry, world, "Armor", Armor, [
    ("armor", DataType::UInt32, 0),
]);
```

For multi-field structs, use `offset_of!`:
```rust
register_schema!(registry, world, "MyComp", MyComp, [
    ("field_a", DataType::Float32, offset_of!(MyComp, a)),
    ("field_b", DataType::Float32, offset_of!(MyComp, b)),
]);
```

Supported Arrow types: `Float32`, `UInt32`, `UInt8`, `UInt64`, `Boolean`.

### 3. Add spawn dispatch (`pyworld.rs`)

Add a builder function:
```rust
fn build_armor(kwargs: &Bound<'_, PyDict>) -> PyResult<Armor> {
    Ok(Armor(kwarg!(kwargs, "armor", u32)?))
}
```

Add single-component arm to `spawn_typed`:
```rust
"Armor" => Ok(world.spawn((build_armor(kwargs)?,))),
```

Add any multi-component bundles you need (keys must be **alphabetically sorted**):
```rust
"Armor,Health,Position" => Ok(world.spawn((
    build_armor(kwargs)?, build_health(kwargs)?, build_position(kwargs)?,
))),
```

### 4. Add write dispatch (`pyworld.rs`)

Add to `write_field_to_entity`:
```rust
("Armor", "armor") => write_arm!(world, entity, value, Armor),
```

For multi-field structs:
```rust
("MyComp", "field_a") => write_arm!(world, entity, value, MyComp, a),
("MyComp", "field_b") => write_arm!(world, entity, value, MyComp, b),
```

### Rebuild and test

```bash
cd crates/minkowski-py
maturin develop --release
python3 -c "
import minkowski_py as mk
world = mk.World()
world.spawn('Armor', armor=50)
print(world.query('Armor'))
"
```

## Adding a Reducer

Three touch points in `reducers.rs`.

### 1. Define parameter struct

```rust
#[derive(Clone)]
pub struct MyReducerParams {
    pub speed: f32,
    pub world_size: f32,
}
```

### 2. Register in `register_all`

```rust
let id = registry.register_query::<(&mut Position, &Velocity), MyReducerParams, _>(
    world,
    "my_reducer",
    |mut query: QueryMut<'_, (&mut Position, &Velocity)>, params: MyReducerParams| {
        let ws = params.world_size;
        let mut i = 0;
        query.for_each(|(pos, vel)| {
            pos.x = (pos.x + vel.x * params.speed).rem_euclid(ws);
            pos.y = (pos.y + vel.y * params.speed).rem_euclid(ws);
            i += 1;
        });
    },
);
map.insert("my_reducer".to_string(), id);
```

### 3. Add dispatch arm

In `dispatch()`, add a match arm that extracts kwargs and validates:
```rust
"my_reducer" => {
    let params = MyReducerParams {
        speed: kwarg_f32(kwargs, "speed", 1.0)?,
        world_size: kwarg_f32(kwargs, "world_size", 500.0)?,
    };
    validate_positive("world_size", params.world_size)?;
    registry.run(world, id, params);
}
```

## Tooling Setup

```bash
cd crates/minkowski-py

# Create venv and install deps (including maturin as dev dep)
uv venv
uv pip install -e ".[dev]"

# Build the native module
maturin develop --release

# Run Python
python3 -c "import minkowski_py as mk; print(mk.World().component_names())"
```

Always use `uv`, never `pip` directly.

## Python API

```python
import minkowski_py as mk

world = mk.World()
registry = mk.ReducerRegistry(world)

# Spawn
e = world.spawn("Position,Velocity", pos_x=1.0, pos_y=2.0, vel_x=0.5, vel_y=0.0)
ids = world.spawn_batch("Position,Velocity", {
    "pos_x": [0.0, 1.0], "pos_y": [0.0, 1.0],
    "vel_x": [1.0, 0.0], "vel_y": [0.0, 1.0],
})

# Query → Polars DataFrame (one-copy + zero-copy)
df = world.query("Position", "Velocity")

# Query → PyArrow RecordBatch (one-copy + zero-copy)
table = world.query_arrow("Position", "Velocity")

# Write back
world.write_column("Position", entity_ids, pos_x=[10.0, 20.0], pos_y=[30.0, 40.0])

# Reducers
registry.run("boids_forces", world, world_size=500.0, sep_r=25.0)
registry.run("gravity", world, g=0.06674, softening=1.0, dt=0.001)

# Lifecycle
world.despawn(entity_id)
world.is_alive(entity_id)
world.entity_count()
world.component_names()
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Bundle key not sorted alphabetically | `spawn()` sorts names — match keys must be sorted too |
| Missing `#[repr(C)]` on multi-field struct | Required for correct `offset_of!` behavior in Arrow bridge |
| Missing `#[repr(transparent)]` on newtype | Required for correct byte-level reads |
| Using `wrap(v - ws)` instead of `rem_euclid` | Single-step wrap fails for large velocities |
| Forgot to add write dispatch arm | `write_column` will error with "unknown field" |
| Forgot to validate params in `dispatch()` | `world_size=0` causes division by zero |
