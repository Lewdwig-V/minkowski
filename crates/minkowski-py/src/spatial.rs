//! Python-facing spatial grid for radius queries over Position entities.

use crate::components::Position;
use crate::pyworld::PyWorld;
use minkowski::Entity;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

type CellMap = HashMap<(i32, i32), Vec<(Entity, f32, f32)>>;

#[pyclass(name = "SpatialGrid")]
pub struct PySpatialGrid {
    world_size: f32,
    cell_size: f32,
    inv_cell: f32,
    cells: CellMap,
}

#[pymethods]
impl PySpatialGrid {
    #[new]
    fn new(world_size: f32, cell_size: f32) -> PyResult<Self> {
        if !world_size.is_finite() || world_size <= 0.0 {
            return Err(PyValueError::new_err(
                "world_size must be finite and positive",
            ));
        }
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return Err(PyValueError::new_err(
                "cell_size must be finite and positive",
            ));
        }
        Ok(PySpatialGrid {
            world_size,
            cell_size,
            inv_cell: 1.0 / cell_size,
            cells: HashMap::new(),
        })
    }

    /// Rebuild the grid from all entities with a Position component.
    fn rebuild(&mut self, world: &mut PyWorld) {
        self.cells.clear();
        let ws = self.world_size;
        let inv = self.inv_cell;

        world
            .world
            .query::<(Entity, &Position)>()
            .for_each(|(entity, pos)| {
                let x = pos.x.rem_euclid(ws);
                let y = pos.y.rem_euclid(ws);
                let cx = (x * inv) as i32;
                let cy = (y * inv) as i32;
                self.cells.entry((cx, cy)).or_default().push((entity, x, y));
            });
    }

    /// Return entity IDs within `radius` of `(x, y)`, using toroidal distance.
    ///
    /// Results reflect the last `rebuild()` call. If entities have been spawned,
    /// despawned, or moved since then, call `rebuild()` first. Returned IDs
    /// should be validated with `world.is_alive()` if mutations occurred since
    /// the last rebuild.
    fn query_radius(&self, x: f32, y: f32, radius: f32) -> Vec<u64> {
        let ws = self.world_size;
        let half = ws * 0.5;
        let r2 = radius * radius;
        let inv = self.inv_cell;

        let xw = x.rem_euclid(ws);
        let yw = y.rem_euclid(ws);

        let cx = (xw * inv) as i32;
        let cy = (yw * inv) as i32;
        let grid_cells = (ws * inv).ceil() as i32;
        // Clamp scan radius so we never wrap around and visit the same cell twice.
        let cells_r = (radius * inv).ceil() as i32;
        let cells_r = cells_r.min(grid_cells / 2);

        let mut result = Vec::new();
        for dx in -cells_r..=cells_r {
            for dy in -cells_r..=cells_r {
                let gx = (cx + dx).rem_euclid(grid_cells);
                let gy = (cy + dy).rem_euclid(grid_cells);
                if let Some(bucket) = self.cells.get(&(gx, gy)) {
                    for &(entity, ex, ey) in bucket {
                        let mut ddx = (ex - xw).abs();
                        if ddx > half {
                            ddx = ws - ddx;
                        }
                        let mut ddy = (ey - yw).abs();
                        if ddy > half {
                            ddy = ws - ddy;
                        }
                        if ddx * ddx + ddy * ddy <= r2 {
                            result.push(entity.to_bits());
                        }
                    }
                }
            }
        }
        result
    }

    /// Number of entities in the grid (as of last `rebuild()`).
    #[getter]
    fn entity_count(&self) -> usize {
        self.cells.values().map(Vec::len).sum()
    }

    /// Cell size used for grid bucketing.
    #[getter]
    fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// World size (toroidal domain extent).
    #[getter]
    fn world_size(&self) -> f32 {
        self.world_size
    }
}
