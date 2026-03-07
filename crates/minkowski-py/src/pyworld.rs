//! PyO3 wrapper around minkowski::World.

use crate::bridge::query_to_record_batch;
use crate::components::{
    Acceleration, CellState, Energy, Faction, Heading, Health, Mass, Nutrition, Position, Velocity,
    WormSize,
};
use crate::schema::SchemaRegistry;
use minkowski::{Entity, World};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_arrow::PyRecordBatch;

// ── Typed spawn dispatch ──────────────────────────────────────────

macro_rules! kwarg {
    ($kwargs:expr, $key:expr, $T:ty) => {{
        let kwargs: &Bound<'_, PyDict> = $kwargs;
        match kwargs.get_item($key)? {
            Some(v) => v.extract::<$T>(),
            None => Ok(<$T>::default()),
        }
    }};
}

/// Build a Position from kwargs.
fn build_position(kwargs: &Bound<'_, PyDict>) -> PyResult<Position> {
    Ok(Position {
        x: kwarg!(kwargs, "pos_x", f32)?,
        y: kwarg!(kwargs, "pos_y", f32)?,
    })
}

/// Build a Velocity from kwargs.
fn build_velocity(kwargs: &Bound<'_, PyDict>) -> PyResult<Velocity> {
    Ok(Velocity {
        x: kwarg!(kwargs, "vel_x", f32)?,
        y: kwarg!(kwargs, "vel_y", f32)?,
    })
}

/// Build an Acceleration from kwargs.
fn build_acceleration(kwargs: &Bound<'_, PyDict>) -> PyResult<Acceleration> {
    Ok(Acceleration {
        x: kwarg!(kwargs, "acc_x", f32)?,
        y: kwarg!(kwargs, "acc_y", f32)?,
    })
}

/// Build a Mass from kwargs.
fn build_mass(kwargs: &Bound<'_, PyDict>) -> PyResult<Mass> {
    Ok(Mass(kwarg!(kwargs, "mass", f32)?))
}

/// Build a CellState from kwargs.
fn build_cell_state(kwargs: &Bound<'_, PyDict>) -> PyResult<CellState> {
    Ok(CellState(kwarg!(kwargs, "alive", bool)?))
}

/// Build a Heading from kwargs.
fn build_heading(kwargs: &Bound<'_, PyDict>) -> PyResult<Heading> {
    Ok(Heading(kwarg!(kwargs, "heading", f32)?))
}

/// Build an Energy from kwargs.
fn build_energy(kwargs: &Bound<'_, PyDict>) -> PyResult<Energy> {
    Ok(Energy(kwarg!(kwargs, "energy", f32)?))
}

/// Build a Health from kwargs.
fn build_health(kwargs: &Bound<'_, PyDict>) -> PyResult<Health> {
    Ok(Health(kwarg!(kwargs, "health", u32)?))
}

/// Build a Faction from kwargs.
fn build_faction(kwargs: &Bound<'_, PyDict>) -> PyResult<Faction> {
    Ok(Faction(kwarg!(kwargs, "faction", u8)?))
}

/// Build a WormSize from kwargs.
fn build_worm_size(kwargs: &Bound<'_, PyDict>) -> PyResult<WormSize> {
    Ok(WormSize(kwarg!(kwargs, "worm_size", f32)?))
}

/// Build a Nutrition from kwargs.
fn build_nutrition(kwargs: &Bound<'_, PyDict>) -> PyResult<Nutrition> {
    Ok(Nutrition(kwarg!(kwargs, "nutrition", f32)?))
}

/// Typed spawn dispatch. Matches a sorted component-name key to a typed
/// `world.spawn(...)` call. Returns the spawned Entity.
fn spawn_typed(
    world: &mut World,
    sorted_key: &str,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<Entity> {
    match sorted_key {
        // ── Single-component bundles ──
        "Acceleration" => Ok(world.spawn((build_acceleration(kwargs)?,))),
        "CellState" => Ok(world.spawn((build_cell_state(kwargs)?,))),
        "Energy" => Ok(world.spawn((build_energy(kwargs)?,))),
        "Faction" => Ok(world.spawn((build_faction(kwargs)?,))),
        "Heading" => Ok(world.spawn((build_heading(kwargs)?,))),
        "Health" => Ok(world.spawn((build_health(kwargs)?,))),
        "Mass" => Ok(world.spawn((build_mass(kwargs)?,))),
        "Position" => Ok(world.spawn((build_position(kwargs)?,))),
        "Velocity" => Ok(world.spawn((build_velocity(kwargs)?,))),
        "WormSize" => Ok(world.spawn((build_worm_size(kwargs)?,))),
        "Nutrition" => Ok(world.spawn((build_nutrition(kwargs)?,))),

        // ── Two-component bundles ──
        "Position,Velocity" => Ok(world.spawn((build_position(kwargs)?, build_velocity(kwargs)?))),
        "Mass,Position" => Ok(world.spawn((build_mass(kwargs)?, build_position(kwargs)?))),
        "Heading,Position" => Ok(world.spawn((build_heading(kwargs)?, build_position(kwargs)?))),
        "Nutrition,Position" => {
            Ok(world.spawn((build_nutrition(kwargs)?, build_position(kwargs)?)))
        }

        // ── Three-component bundles ──
        "Acceleration,Position,Velocity" => Ok(world.spawn((
            build_acceleration(kwargs)?,
            build_position(kwargs)?,
            build_velocity(kwargs)?,
        ))),
        "Mass,Position,Velocity" => Ok(world.spawn((
            build_mass(kwargs)?,
            build_position(kwargs)?,
            build_velocity(kwargs)?,
        ))),
        "Energy,Heading,Position" => Ok(world.spawn((
            build_energy(kwargs)?,
            build_heading(kwargs)?,
            build_position(kwargs)?,
        ))),
        "Faction,Health,Position" => Ok(world.spawn((
            build_faction(kwargs)?,
            build_health(kwargs)?,
            build_position(kwargs)?,
        ))),

        // ── Four-component bundles (worm) ──
        "Energy,Heading,Position,WormSize" => Ok(world.spawn((
            build_energy(kwargs)?,
            build_heading(kwargs)?,
            build_position(kwargs)?,
            build_worm_size(kwargs)?,
        ))),

        // ── Four-component bundles ──
        "Acceleration,Mass,Position,Velocity" => Ok(world.spawn((
            build_acceleration(kwargs)?,
            build_mass(kwargs)?,
            build_position(kwargs)?,
            build_velocity(kwargs)?,
        ))),
        "Energy,Heading,Position,Velocity" => Ok(world.spawn((
            build_energy(kwargs)?,
            build_heading(kwargs)?,
            build_position(kwargs)?,
            build_velocity(kwargs)?,
        ))),
        "Faction,Health,Mass,Position" => Ok(world.spawn((
            build_faction(kwargs)?,
            build_health(kwargs)?,
            build_mass(kwargs)?,
            build_position(kwargs)?,
        ))),

        _ => Err(PyValueError::new_err(format!(
            "unsupported component bundle: \"{sorted_key}\". \
             Register this combination in pyworld.rs spawn_typed()."
        ))),
    }
}

// ── Typed write dispatch ──────────────────────────────────────────

macro_rules! write_arm {
    ($world:expr, $entity:expr, $value:expr, $Comp:ty, $field:ident) => {{
        let v = $value.extract()?;
        $world
            .get_mut::<$Comp>($entity)
            .ok_or_else(|| PyValueError::new_err(concat!("entity missing ", stringify!($Comp))))?
            .$field = v;
        Ok(())
    }};
    ($world:expr, $entity:expr, $value:expr, $Comp:ty) => {{
        let v = $value.extract()?;
        $world
            .get_mut::<$Comp>($entity)
            .ok_or_else(|| PyValueError::new_err(concat!("entity missing ", stringify!($Comp))))?
            .0 = v;
        Ok(())
    }};
}

fn write_field_to_entity(
    world: &mut World,
    entity: Entity,
    component_name: &str,
    field_name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (component_name, field_name) {
        ("Position", "pos_x") => write_arm!(world, entity, value, Position, x),
        ("Position", "pos_y") => write_arm!(world, entity, value, Position, y),
        ("Velocity", "vel_x") => write_arm!(world, entity, value, Velocity, x),
        ("Velocity", "vel_y") => write_arm!(world, entity, value, Velocity, y),
        ("Acceleration", "acc_x") => write_arm!(world, entity, value, Acceleration, x),
        ("Acceleration", "acc_y") => write_arm!(world, entity, value, Acceleration, y),
        ("Mass", "mass") => write_arm!(world, entity, value, Mass),
        ("CellState", "alive") => write_arm!(world, entity, value, CellState),
        ("Heading", "heading") => write_arm!(world, entity, value, Heading),
        ("Energy", "energy") => write_arm!(world, entity, value, Energy),
        ("Health", "health") => write_arm!(world, entity, value, Health),
        ("Faction", "faction") => write_arm!(world, entity, value, Faction),
        ("WormSize", "worm_size") => write_arm!(world, entity, value, WormSize),
        ("Nutrition", "nutrition") => write_arm!(world, entity, value, Nutrition),
        _ => Err(PyValueError::new_err(format!(
            "unknown field: {component_name}.{field_name}"
        ))),
    }
}

// ── PyWorld ───────────────────────────────────────────────────────

#[pyclass(name = "World")]
pub struct PyWorld {
    pub(crate) world: World,
    pub(crate) schema_registry: SchemaRegistry,
}

#[pymethods]
impl PyWorld {
    #[new]
    fn new() -> Self {
        let mut world = World::new();
        let mut schema_registry = SchemaRegistry::new();
        crate::components::register_all(&mut schema_registry, &mut world);
        PyWorld {
            world,
            schema_registry,
        }
    }

    /// Spawn a single entity.
    ///
    /// `components` is a comma-separated string of component names (e.g.
    /// "Position,Velocity"). Remaining kwargs are field values.
    ///
    /// Returns the entity's bit-packed u64 ID.
    #[pyo3(signature = (components, **kwargs))]
    fn spawn(&mut self, components: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<u64> {
        let kwargs = kwargs.ok_or_else(|| {
            PyValueError::new_err("spawn requires keyword arguments for field values")
        })?;

        // Sort component names to produce a canonical key for dispatch
        let mut names: Vec<&str> = components.split(',').map(|s| s.trim()).collect();
        names.sort();
        let key = names.join(",");

        let entity = spawn_typed(&mut self.world, &key, kwargs)?;
        Ok(entity.to_bits())
    }

    /// Spawn multiple entities from columnar data.
    ///
    /// `components` is a comma-separated string of component names.
    /// `data` is a dict mapping column names (e.g. "pos_x") to lists of values.
    /// All lists must have the same length.
    ///
    /// Returns a list of bit-packed u64 entity IDs.
    fn spawn_batch(&mut self, components: &str, data: &Bound<'_, PyDict>) -> PyResult<Vec<u64>> {
        // Determine batch size from the first list
        let first_key = data
            .keys()
            .get_item(0)
            .map_err(|_| PyValueError::new_err("spawn_batch data must be non-empty"))?;
        let first_list = data
            .get_item(&first_key)?
            .ok_or_else(|| PyValueError::new_err("spawn_batch data must be non-empty"))?;
        let n: usize = first_list.len()?;

        // Validate all columns have the same length
        for kv in data.iter() {
            let (k, v) = kv;
            let col_len: usize = v.len()?;
            if col_len != n {
                return Err(PyValueError::new_err(format!(
                    "column '{}' has length {col_len}, expected {n}",
                    k
                )));
            }
        }

        // Build a row-dict for each entity and spawn one at a time
        let mut entities = Vec::with_capacity(n);

        // Sort component names for canonical key
        let mut comp_names: Vec<&str> = components.split(',').map(|s| s.trim()).collect();
        comp_names.sort();
        let key = comp_names.join(",");

        for i in 0..n {
            // Build kwargs dict for this row
            let py = data.py();
            let row_dict = PyDict::new(py);
            for kv in data.iter() {
                let (k, v) = kv;
                let item = v.get_item(i)?;
                row_dict.set_item(k, item)?;
            }
            let entity = spawn_typed(&mut self.world, &key, &row_dict.as_borrowed())?;
            entities.push(entity.to_bits());
        }
        Ok(entities)
    }

    /// Query components and return a PyArrow RecordBatch.
    ///
    /// Pass component names as positional args:
    ///   `world.query_arrow("Position", "Velocity")`
    #[pyo3(signature = (*component_names))]
    fn query_arrow(&mut self, component_names: Vec<String>) -> PyResult<PyRecordBatch> {
        let names: Vec<&str> = component_names.iter().map(|s| s.as_str()).collect();
        let batch = query_to_record_batch(&mut self.world, &self.schema_registry, &names)
            .map_err(PyValueError::new_err)?;
        Ok(PyRecordBatch::new(batch))
    }

    /// Query components and return a Polars DataFrame.
    ///
    /// Pass component names as positional args:
    ///   `world.query("Position", "Velocity")`
    #[pyo3(signature = (*component_names))]
    fn query<'py>(
        &mut self,
        py: Python<'py>,
        component_names: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let names: Vec<&str> = component_names.iter().map(|s| s.as_str()).collect();
        let batch = query_to_record_batch(&mut self.world, &self.schema_registry, &names)
            .map_err(PyValueError::new_err)?;

        let py_batch = PyRecordBatch::new(batch);
        // Convert to pyarrow RecordBatch first, then to Polars DataFrame
        let pa_batch = py_batch.into_pyarrow(py)?;
        let polars = py.import("polars")?;
        polars.call_method1("from_arrow", (&pa_batch,))
    }

    /// Write column data back into the ECS.
    ///
    /// `component`: component name (e.g. "Position")
    /// `entity_ids`: list of u64 entity bit-packed IDs
    /// `kwargs`: column_name -> list of values (e.g. pos_x=[1.0, 2.0])
    #[pyo3(signature = (component, entity_ids, **kwargs))]
    fn write_column(
        &mut self,
        component: &str,
        entity_ids: Vec<u64>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let kwargs = kwargs
            .ok_or_else(|| PyValueError::new_err("write_column requires keyword arguments"))?;
        let schema = self
            .schema_registry
            .get(component)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component: {component}")))?;

        // Collect field names we need to write (only those present in kwargs)
        let field_names: Vec<&'static str> = schema
            .fields
            .iter()
            .filter_map(|fm| {
                kwargs.contains(fm.column_name).ok().and_then(|has| {
                    if has {
                        Some(fm.column_name)
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Validate column lengths match entity_ids
        for kv in kwargs.iter() {
            let (k, v) = kv;
            let col_len: usize = v.len()?;
            if col_len != entity_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "column '{}' has length {col_len}, expected {} (entity_ids length)",
                    k,
                    entity_ids.len()
                )));
            }
        }

        // Extract column lists once before entity loop
        let columns: Vec<(&str, Bound<'_, PyAny>)> = field_names
            .iter()
            .filter_map(|&name| kwargs.get_item(name).ok().flatten().map(|v| (name, v)))
            .collect();

        for (i, &bits) in entity_ids.iter().enumerate() {
            let entity = Entity::from_bits(bits);
            if !self.world.is_alive(entity) {
                return Err(PyValueError::new_err(format!("entity {bits} is not alive")));
            }
            for (field_name, ref py_list) in &columns {
                let value = py_list.get_item(i)?;
                write_field_to_entity(&mut self.world, entity, component, field_name, &value)?;
            }
        }
        Ok(())
    }

    /// Despawn an entity by its bit-packed u64 ID.
    /// Returns True if the entity existed and was despawned.
    fn despawn(&mut self, entity_bits: u64) -> bool {
        self.world.despawn(Entity::from_bits(entity_bits))
    }

    /// Check if an entity is alive.
    fn is_alive(&self, entity_bits: u64) -> bool {
        self.world.is_alive(Entity::from_bits(entity_bits))
    }

    /// Number of live entities across all archetypes.
    fn entity_count(&self) -> usize {
        (0..self.world.archetype_count())
            .map(|i| self.world.archetype_len(i))
            .sum()
    }

    /// Number of archetypes.
    fn archetype_count(&self) -> usize {
        self.world.archetype_count()
    }

    /// List registered component names (sorted).
    fn component_names(&self) -> Vec<&'static str> {
        self.schema_registry.names()
    }
}
