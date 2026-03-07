//! Python bindings for the Minkowski ECS engine.
//!
//! Exposes simulation classes (`BoidsSim`, `NBodySim`, `LifeSim`) that each
//! manage a [`World`] internally and export entity state as Apache Arrow
//! columnar tables for efficient handoff to Polars / pandas / NumPy in Python.
//!
//! Build with maturin: `cd crates/minkowski-py && maturin develop --release`

// world.query() requires &mut World for cache mutation, so to_arrow/to_polars
// must take &mut self despite being logically read-only.
#![allow(clippy::wrong_self_convention)]
// The 5-tuple history type is clear with its inline comment.
#![allow(clippy::type_complexity)]

use minkowski::{Entity, World};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
struct Acceleration {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
struct Mass(f32);

#[derive(Clone, Copy)]
struct CellState(bool);

// ── Vec2 helpers ────────────────────────────────────────────────────

fn vec2_length_sq(x: f32, y: f32) -> f32 {
    x * x + y * y
}

fn vec2_length(x: f32, y: f32) -> f32 {
    vec2_length_sq(x, y).sqrt()
}

fn vec2_normalized(x: f32, y: f32) -> (f32, f32) {
    let len = vec2_length(x, y);
    if len < 1e-8 {
        (0.0, 0.0)
    } else {
        (x / len, y / len)
    }
}

fn vec2_clamped(x: f32, y: f32, max_len: f32) -> (f32, f32) {
    let len_sq = vec2_length_sq(x, y);
    if len_sq > max_len * max_len {
        let (nx, ny) = vec2_normalized(x, y);
        (nx * max_len, ny * max_len)
    } else {
        (x, y)
    }
}

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

fn wrap_position(pos: &mut Position, world_size: f32) {
    pos.x = pos.x.rem_euclid(world_size);
    pos.y = pos.y.rem_euclid(world_size);
}

// ── Arrow / Polars helpers ─────────────────────────────────────────

/// Build a PyArrow table from parallel column vectors.
/// `columns` is a list of (name, pyarrow_array) pairs.
fn build_arrow_table<'py>(
    py: Python<'py>,
    names: &[&str],
    arrays: Vec<Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let pyarrow = py.import("pyarrow")?;
    let name_list: Vec<&str> = names.to_vec();
    let table = pyarrow.call_method1("table", (arrays, name_list))?;
    Ok(table.into())
}

/// Convert a PyArrow table to a Polars DataFrame.
fn arrow_to_polars(py: Python<'_>, arrow_table: &PyObject) -> PyResult<PyObject> {
    let polars = py.import("polars")?;
    let df_class = polars.getattr("DataFrame")?;
    let df = df_class.call1((arrow_table,))?;
    Ok(df.into())
}

/// Create a pyarrow array from a Vec.
fn pa_array<'py, T: IntoPyObject<'py>>(py: Python<'py>, data: T) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;
    pyarrow.call_method1("array", (data,))
}

// ── Input validation ───────────────────────────────────────────────

fn validate_positive(name: &str, value: f32) -> PyResult<()> {
    if value <= 0.0 || !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a positive finite number (got {value})"
        )));
    }
    Ok(())
}

fn validate_non_negative(name: &str, value: f32) -> PyResult<()> {
    if value < 0.0 || !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a non-negative finite number (got {value})"
        )));
    }
    Ok(())
}

// ── Boids Simulation ────────────────────────────────────────────────

#[pyclass]
struct BoidsSim {
    world: World,
    params: BoidsParams,
    frame: usize,
    history: Vec<Vec<(f32, f32, f32, f32)>>, // (x, y, vx, vy) per frame
}

struct BoidsParams {
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
    max_speed: f32,
    max_force: f32,
    world_size: f32,
    dt: f32,
}

impl Default for BoidsParams {
    fn default() -> Self {
        Self {
            separation_radius: 25.0,
            alignment_radius: 50.0,
            cohesion_radius: 50.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            max_speed: 4.0,
            max_force: 0.1,
            world_size: 500.0,
            dt: 0.016,
        }
    }
}

#[pymethods]
impl BoidsSim {
    #[new]
    #[pyo3(signature = (
        n = 1000,
        world_size = 500.0,
        separation_radius = 25.0,
        alignment_radius = 50.0,
        cohesion_radius = 50.0,
        separation_weight = 1.5,
        alignment_weight = 1.0,
        cohesion_weight = 1.0,
        max_speed = 4.0,
        max_force = 0.1,
        dt = 0.016,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        world_size: f32,
        separation_radius: f32,
        alignment_radius: f32,
        cohesion_radius: f32,
        separation_weight: f32,
        alignment_weight: f32,
        cohesion_weight: f32,
        max_speed: f32,
        max_force: f32,
        dt: f32,
    ) -> PyResult<Self> {
        validate_positive("world_size", world_size)?;
        validate_positive("separation_radius", separation_radius)?;
        validate_positive("alignment_radius", alignment_radius)?;
        validate_positive("cohesion_radius", cohesion_radius)?;
        validate_non_negative("separation_weight", separation_weight)?;
        validate_non_negative("alignment_weight", alignment_weight)?;
        validate_non_negative("cohesion_weight", cohesion_weight)?;
        validate_positive("max_speed", max_speed)?;
        validate_positive("max_force", max_force)?;
        validate_positive("dt", dt)?;

        let params = BoidsParams {
            separation_radius,
            alignment_radius,
            cohesion_radius,
            separation_weight,
            alignment_weight,
            cohesion_weight,
            max_speed,
            max_force,
            world_size,
            dt,
        };

        let mut world = World::new();
        for _ in 0..n {
            let x = fastrand::f32() * world_size;
            let y = fastrand::f32() * world_size;
            let angle = fastrand::f32() * std::f32::consts::TAU;
            let speed = fastrand::f32() * max_speed;
            world.spawn((
                Position { x, y },
                Velocity {
                    x: angle.cos() * speed,
                    y: angle.sin() * speed,
                },
                Acceleration { x: 0.0, y: 0.0 },
            ));
        }

        Ok(BoidsSim {
            world,
            params,
            frame: 0,
            history: Vec::new(),
        })
    }

    /// Advance the simulation by `steps` frames, optionally recording history.
    #[pyo3(signature = (steps = 1, record = false))]
    fn step(&mut self, steps: usize, record: bool) {
        for _ in 0..steps {
            self.step_one();
            if record {
                self.record_snapshot();
            }
            self.frame += 1;
        }
    }

    /// Return the current frame number.
    fn current_frame(&self) -> usize {
        self.frame
    }

    /// Return the current entity count.
    fn entity_count(&mut self) -> usize {
        self.world.query::<&Position>().count()
    }

    /// Export current state as a PyArrow table with columns:
    /// entity_id, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y
    fn to_arrow(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let mut ids = Vec::new();
        let mut px = Vec::new();
        let mut py_vec = Vec::new();
        let mut vx = Vec::new();
        let mut vy = Vec::new();
        let mut ax = Vec::new();
        let mut ay = Vec::new();

        for (e, p, v, a) in self
            .world
            .query::<(Entity, &Position, &Velocity, &Acceleration)>()
        {
            ids.push(e.to_bits());
            px.push(p.x);
            py_vec.push(p.y);
            vx.push(v.x);
            vy.push(v.y);
            ax.push(a.x);
            ay.push(a.y);
        }

        build_arrow_table(
            py,
            &[
                "entity_id",
                "pos_x",
                "pos_y",
                "vel_x",
                "vel_y",
                "acc_x",
                "acc_y",
            ],
            vec![
                pa_array(py, ids)?,
                pa_array(py, px)?,
                pa_array(py, py_vec)?,
                pa_array(py, vx)?,
                pa_array(py, vy)?,
                pa_array(py, ax)?,
                pa_array(py, ay)?,
            ],
        )
    }

    /// Export current state as a Polars DataFrame.
    fn to_polars(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let arrow_table = self.to_arrow(py)?;
        arrow_to_polars(py, &arrow_table)
    }

    /// Return recorded history as a Polars DataFrame with columns:
    /// frame, pos_x, pos_y, vel_x, vel_y
    fn history_to_polars(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut frames = Vec::new();
        let mut px = Vec::new();
        let mut py_vec = Vec::new();
        let mut vx = Vec::new();
        let mut vy = Vec::new();

        for (frame_idx, snapshot) in self.history.iter().enumerate() {
            let frame = frame_idx as u64;
            for &(x, y, velx, vely) in snapshot {
                frames.push(frame);
                px.push(x);
                py_vec.push(y);
                vx.push(velx);
                vy.push(vely);
            }
        }

        let arrow_table = build_arrow_table(
            py,
            &["frame", "pos_x", "pos_y", "vel_x", "vel_y"],
            vec![
                pa_array(py, frames)?,
                pa_array(py, px)?,
                pa_array(py, py_vec)?,
                pa_array(py, vx)?,
                pa_array(py, vy)?,
            ],
        )?;
        arrow_to_polars(py, &arrow_table)
    }

    /// Clear recorded history to free memory.
    fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl BoidsSim {
    fn step_one(&mut self) {
        let params = &self.params;

        // Zero accelerations
        for acc in self.world.query::<&mut Acceleration>() {
            acc.x = 0.0;
            acc.y = 0.0;
        }

        // Build spatial snapshot
        let snapshot: Vec<(f32, f32, f32, f32)> = self
            .world
            .query::<(&Position, &Velocity)>()
            .map(|(p, v)| (p.x, p.y, v.x, v.y))
            .collect();

        // Build grid
        let cell_size = params.cohesion_radius;
        let grid_w = (params.world_size / cell_size).ceil() as usize;
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); grid_w * grid_w];
        for (i, &(x, y, _, _)) in snapshot.iter().enumerate() {
            let cx = ((x / cell_size) as usize).min(grid_w - 1);
            let cy = ((y / cell_size) as usize).min(grid_w - 1);
            cells[cy * grid_w + cx].push(i);
        }

        // Compute forces
        let mut forces: Vec<(f32, f32)> = Vec::with_capacity(snapshot.len());
        for idx in 0..snapshot.len() {
            let (px, py, vx, vy) = snapshot[idx];

            let mut sep_x = 0.0f32;
            let mut sep_y = 0.0f32;
            let mut ali_x = 0.0f32;
            let mut ali_y = 0.0f32;
            let mut coh_x = 0.0f32;
            let mut coh_y = 0.0f32;
            let mut sep_count = 0u32;
            let mut ali_count = 0u32;
            let mut coh_count = 0u32;

            let cx = ((px / cell_size) as usize).min(grid_w - 1);
            let cy = ((py / cell_size) as usize).min(grid_w - 1);

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                    let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                    for &j in &cells[ny * grid_w + nx] {
                        let (ox, oy, ovx, ovy) = snapshot[j];
                        let diff_x = wrapped_diff(ox, px, params.world_size);
                        let diff_y = wrapped_diff(oy, py, params.world_size);
                        let dist_sq = diff_x * diff_x + diff_y * diff_y;
                        if dist_sq < 1e-6 {
                            continue;
                        }
                        let dist = dist_sq.sqrt();

                        if dist < params.separation_radius {
                            let (nx, ny) = vec2_normalized(diff_x, diff_y);
                            sep_x -= nx / dist;
                            sep_y -= ny / dist;
                            sep_count += 1;
                        }
                        if dist < params.alignment_radius {
                            ali_x += ovx;
                            ali_y += ovy;
                            ali_count += 1;
                        }
                        if dist < params.cohesion_radius {
                            coh_x += diff_x;
                            coh_y += diff_y;
                            coh_count += 1;
                        }
                    }
                }
            }

            let mut fx = 0.0f32;
            let mut fy = 0.0f32;
            if sep_count > 0 {
                fx += sep_x / sep_count as f32 * params.separation_weight;
                fy += sep_y / sep_count as f32 * params.separation_weight;
            }
            if ali_count > 0 {
                let desired_x = ali_x / ali_count as f32 - vx;
                let desired_y = ali_y / ali_count as f32 - vy;
                fx += desired_x * params.alignment_weight;
                fy += desired_y * params.alignment_weight;
            }
            if coh_count > 0 {
                fx += (coh_x / coh_count as f32) * params.cohesion_weight;
                fy += (coh_y / coh_count as f32) * params.cohesion_weight;
            }

            let (fx, fy) = vec2_clamped(fx, fy, params.max_force);
            forces.push((fx, fy));
        }

        // Apply forces — zip with forces to avoid index-out-of-bounds if
        // snapshot and query iteration ever diverge.
        let mut force_iter = forces.into_iter();
        for (acc, vel, pos) in self
            .world
            .query::<(&mut Acceleration, &mut Velocity, &mut Position)>()
        {
            let (fx, fy) = force_iter
                .next()
                .expect("force count must match entity count");
            acc.x += fx;
            acc.y += fy;

            vel.x += acc.x * params.dt;
            vel.y += acc.y * params.dt;
            let (vx, vy) = vec2_clamped(vel.x, vel.y, params.max_speed);
            vel.x = vx;
            vel.y = vy;

            pos.x += vel.x * params.dt;
            pos.y += vel.y * params.dt;
            wrap_position(pos, params.world_size);
        }
    }

    fn record_snapshot(&mut self) {
        let snap: Vec<(f32, f32, f32, f32)> = self
            .world
            .query::<(&Position, &Velocity)>()
            .map(|(p, v)| (p.x, p.y, v.x, v.y))
            .collect();
        self.history.push(snap);
    }
}

// ── N-Body Simulation ───────────────────────────────────────────────

#[pyclass]
struct NBodySim {
    world: World,
    world_size: f32,
    g: f32,
    softening: f32,
    dt: f32,
    frame: usize,
    history: Vec<Vec<(f32, f32, f32, f32, f32)>>, // (x, y, vx, vy, mass) per frame
}

#[pymethods]
impl NBodySim {
    #[new]
    #[pyo3(signature = (n = 500, world_size = 500.0, g = 0.06674, softening = 1.0, dt = 0.001))]
    fn new(n: usize, world_size: f32, g: f32, softening: f32, dt: f32) -> PyResult<Self> {
        validate_positive("world_size", world_size)?;
        validate_positive("softening", softening)?;
        validate_positive("dt", dt)?;
        validate_non_negative("g", g)?;

        let mut world = World::new();
        for _ in 0..n {
            let x = fastrand::f32() * world_size;
            let y = fastrand::f32() * world_size;
            let vx = (fastrand::f32() - 0.5) * 10.0;
            let vy = (fastrand::f32() - 0.5) * 10.0;
            world.spawn((Position { x, y }, Velocity { x: vx, y: vy }, Mass(1.0)));
        }

        Ok(NBodySim {
            world,
            world_size,
            g,
            softening,
            dt,
            frame: 0,
            history: Vec::new(),
        })
    }

    /// Advance the simulation by `steps` frames, optionally recording history.
    #[pyo3(signature = (steps = 1, record = false))]
    fn step(&mut self, steps: usize, record: bool) {
        for _ in 0..steps {
            self.step_one();
            if record {
                self.record_snapshot();
            }
            self.frame += 1;
        }
    }

    /// Return the current frame number.
    fn current_frame(&self) -> usize {
        self.frame
    }

    /// Return the current entity count.
    fn entity_count(&mut self) -> usize {
        self.world.query::<&Position>().count()
    }

    /// Export current state as a PyArrow table with columns:
    /// entity_id, pos_x, pos_y, vel_x, vel_y, mass
    fn to_arrow(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let mut ids = Vec::new();
        let mut px = Vec::new();
        let mut py_vec = Vec::new();
        let mut vx = Vec::new();
        let mut vy = Vec::new();
        let mut mass = Vec::new();

        for (e, p, v, m) in self.world.query::<(Entity, &Position, &Velocity, &Mass)>() {
            ids.push(e.to_bits());
            px.push(p.x);
            py_vec.push(p.y);
            vx.push(v.x);
            vy.push(v.y);
            mass.push(m.0);
        }

        build_arrow_table(
            py,
            &["entity_id", "pos_x", "pos_y", "vel_x", "vel_y", "mass"],
            vec![
                pa_array(py, ids)?,
                pa_array(py, px)?,
                pa_array(py, py_vec)?,
                pa_array(py, vx)?,
                pa_array(py, vy)?,
                pa_array(py, mass)?,
            ],
        )
    }

    /// Export current state as a Polars DataFrame.
    fn to_polars(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let arrow_table = self.to_arrow(py)?;
        arrow_to_polars(py, &arrow_table)
    }

    /// Return recorded history as a Polars DataFrame with columns:
    /// frame, pos_x, pos_y, vel_x, vel_y, mass
    fn history_to_polars(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut frames = Vec::new();
        let mut px = Vec::new();
        let mut py_vec = Vec::new();
        let mut vx = Vec::new();
        let mut vy = Vec::new();
        let mut mass_vec = Vec::new();

        for (frame_idx, snapshot) in self.history.iter().enumerate() {
            let frame = frame_idx as u64;
            for &(x, y, velx, vely, m) in snapshot {
                frames.push(frame);
                px.push(x);
                py_vec.push(y);
                vx.push(velx);
                vy.push(vely);
                mass_vec.push(m);
            }
        }

        let arrow_table = build_arrow_table(
            py,
            &["frame", "pos_x", "pos_y", "vel_x", "vel_y", "mass"],
            vec![
                pa_array(py, frames)?,
                pa_array(py, px)?,
                pa_array(py, py_vec)?,
                pa_array(py, vx)?,
                pa_array(py, vy)?,
                pa_array(py, mass_vec)?,
            ],
        )?;
        arrow_to_polars(py, &arrow_table)
    }

    /// Clear recorded history to free memory.
    fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl NBodySim {
    fn step_one(&mut self) {
        // Snapshot all bodies
        let snapshot: Vec<(Entity, f32, f32, f32)> = self
            .world
            .query::<(Entity, &Position, &Mass)>()
            .map(|(e, p, m)| (e, p.x, p.y, m.0))
            .collect();

        // Compute forces (O(N^2) — fine for Python-scale entity counts)
        let ws = self.world_size;
        let g = self.g;
        let softening = self.softening;
        let dt = self.dt;

        let forces: Vec<(Entity, f32, f32)> = snapshot
            .iter()
            .map(|&(entity, px, py, mass)| {
                let mut fx = 0.0f32;
                let mut fy = 0.0f32;
                for &(_, ox, oy, om) in &snapshot {
                    let dx = wrapped_diff(ox, px, ws);
                    let dy = wrapped_diff(oy, py, ws);
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < 1e-6 {
                        continue;
                    }
                    let force_mag = g * mass * om / (dist_sq + softening * softening);
                    let dist = dist_sq.sqrt();
                    fx += dx / dist * force_mag;
                    fy += dy / dist * force_mag;
                }
                (entity, fx / mass, fy / mass)
            })
            .collect();

        // Apply forces
        for &(entity, ax, ay) in &forces {
            let vel = self
                .world
                .get_mut::<Velocity>(entity)
                .expect("entity from snapshot must still exist");
            vel.x += ax * dt;
            vel.y += ay * dt;
        }

        // Integrate positions
        for (pos, vel) in self.world.query::<(&mut Position, &Velocity)>() {
            pos.x += vel.x * dt;
            pos.y += vel.y * dt;
            wrap_position(pos, ws);
        }
    }

    fn record_snapshot(&mut self) {
        let snap: Vec<(f32, f32, f32, f32, f32)> = self
            .world
            .query::<(&Position, &Velocity, &Mass)>()
            .map(|(p, v, m)| (p.x, p.y, v.x, v.y, m.0))
            .collect();
        self.history.push(snap);
    }
}

// ── Game of Life Simulation ─────────────────────────────────────────

#[pyclass]
struct LifeSim {
    world: World,
    grid: Vec<Entity>,
    width: usize,
    height: usize,
    generation: usize,
    history: Vec<Vec<bool>>, // grid state per recorded frame
}

#[pymethods]
impl LifeSim {
    #[new]
    #[pyo3(signature = (width = 64, height = 64, density = 0.45))]
    fn new(width: usize, height: usize, density: f32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err(
                "width and height must be positive (got {width}x{height})",
            ));
        }
        if !(0.0..=1.0).contains(&density) {
            return Err(PyValueError::new_err(format!(
                "density must be in [0.0, 1.0] (got {density})"
            )));
        }

        let mut world = World::new();
        let cell_count = width * height;
        let mut grid = Vec::with_capacity(cell_count);

        for _ in 0..cell_count {
            let alive = fastrand::f32() < density;
            let e = world.spawn((CellState(alive),));
            grid.push(e);
        }

        Ok(LifeSim {
            world,
            grid,
            width,
            height,
            generation: 0,
            history: Vec::new(),
        })
    }

    /// Advance by `steps` generations, optionally recording history.
    #[pyo3(signature = (steps = 1, record = false))]
    fn step(&mut self, steps: usize, record: bool) {
        for _ in 0..steps {
            self.step_one();
            if record {
                self.record_snapshot();
            }
            self.generation += 1;
        }
    }

    /// Return the current generation number.
    fn current_generation(&self) -> usize {
        self.generation
    }

    /// Return the number of alive cells.
    fn alive_count(&mut self) -> usize {
        self.world.query::<&CellState>().filter(|c| c.0).count()
    }

    /// Return the current grid as a flat list of booleans (row-major).
    fn grid_state(&self) -> Vec<bool> {
        self.grid
            .iter()
            .map(|&e| {
                self.world
                    .get::<CellState>(e)
                    .expect("grid entity must be alive")
                    .0
            })
            .collect()
    }

    /// Export current grid as a PyArrow table with columns: row, col, alive
    fn to_arrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut alive = Vec::new();

        for y in 0..self.height {
            for x in 0..self.width {
                let e = self.grid[y * self.width + x];
                let is_alive = self
                    .world
                    .get::<CellState>(e)
                    .expect("grid entity must be alive")
                    .0;
                rows.push(y as u32);
                cols.push(x as u32);
                alive.push(is_alive);
            }
        }

        build_arrow_table(
            py,
            &["row", "col", "alive"],
            vec![
                pa_array(py, rows)?,
                pa_array(py, cols)?,
                pa_array(py, alive)?,
            ],
        )
    }

    /// Export current grid as a Polars DataFrame with columns: row, col, alive
    fn to_polars(&self, py: Python<'_>) -> PyResult<PyObject> {
        let arrow_table = self.to_arrow(py)?;
        arrow_to_polars(py, &arrow_table)
    }

    /// Return history as a Polars DataFrame with columns: generation, row, col, alive
    fn history_to_polars(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut gens = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut alive = Vec::new();

        for (gen_idx, snapshot) in self.history.iter().enumerate() {
            let gen = gen_idx as u64;
            for y in 0..self.height {
                for x in 0..self.width {
                    gens.push(gen);
                    rows.push(y as u32);
                    cols.push(x as u32);
                    alive.push(snapshot[y * self.width + x]);
                }
            }
        }

        let arrow_table = build_arrow_table(
            py,
            &["generation", "row", "col", "alive"],
            vec![
                pa_array(py, gens)?,
                pa_array(py, rows)?,
                pa_array(py, cols)?,
                pa_array(py, alive)?,
            ],
        )?;
        arrow_to_polars(py, &arrow_table)
    }

    /// Clear recorded history to free memory.
    fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Return (width, height).
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

impl LifeSim {
    fn step_one(&mut self) {
        let w = self.width;
        let h = self.height;
        let cell_count = w * h;

        // Snapshot states
        let states: Vec<bool> = self
            .grid
            .iter()
            .map(|&e| {
                self.world
                    .get::<CellState>(e)
                    .expect("grid entity must be alive")
                    .0
            })
            .collect();

        // Count neighbors
        let mut counts = vec![0u8; cell_count];
        for y in 0..h {
            for x in 0..w {
                let left = if x == 0 { w - 1 } else { x - 1 };
                let right = if x == w - 1 { 0 } else { x + 1 };
                let up = if y == 0 { h - 1 } else { y - 1 };
                let down = if y == h - 1 { 0 } else { y + 1 };
                let neighbors = [
                    up * w + left,
                    up * w + x,
                    up * w + right,
                    y * w + left,
                    y * w + right,
                    down * w + left,
                    down * w + x,
                    down * w + right,
                ];
                let mut count = 0u8;
                for &ni in &neighbors {
                    if states[ni] {
                        count += 1;
                    }
                }
                counts[y * w + x] = count;
            }
        }

        // Apply Conway rules
        for i in 0..cell_count {
            let alive = states[i];
            let n = counts[i];
            let new_alive = matches!((alive, n), (true, 2) | (true, 3) | (false, 3));
            if new_alive != alive {
                let cell = self
                    .world
                    .get_mut::<CellState>(self.grid[i])
                    .expect("grid entity must be alive");
                cell.0 = new_alive;
            }
        }
    }

    fn record_snapshot(&mut self) {
        let states: Vec<bool> = self
            .grid
            .iter()
            .map(|&e| {
                self.world
                    .get::<CellState>(e)
                    .expect("grid entity must be alive")
                    .0
            })
            .collect();
        self.history.push(states);
    }
}

// ── Python Module ───────────────────────────────────────────────────

#[pymodule]
fn _minkowski(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoidsSim>()?;
    m.add_class::<NBodySim>()?;
    m.add_class::<LifeSim>()?;
    Ok(())
}
