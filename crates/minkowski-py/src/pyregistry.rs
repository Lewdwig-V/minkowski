//! PyO3 wrapper around ReducerRegistry.
//!
//! Owns the `ReducerRegistry` and the name-to-ID map. The `run` method
//! takes `&mut PyWorld` as a parameter (not borrowed from self) to avoid
//! PyO3 borrow conflicts.

use crate::pyworld::PyWorld;
use minkowski::{QueryReducerId, ReducerRegistry, World};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass(name = "ReducerRegistry")]
pub struct PyReducerRegistry {
    registry: ReducerRegistry,
    name_to_id: HashMap<String, QueryReducerId>,
    /// Pointer to the World this registry was created from.
    /// Used to detect cross-world misuse (PyWorld is heap-allocated, stable address).
    world_ptr: *const World,
}

// Safety: world_ptr is never dereferenced — only compared for identity.
// PyO3 requires Send + Sync for #[pyclass]. The raw pointer is the only
// non-auto-trait field; it's safe because we only use it for address comparison.
unsafe impl Send for PyReducerRegistry {}
unsafe impl Sync for PyReducerRegistry {}

#[pymethods]
impl PyReducerRegistry {
    #[new]
    fn new(world: &mut PyWorld) -> Self {
        let mut registry = ReducerRegistry::new();
        let name_to_id = crate::reducers::register_all(&mut registry, &mut world.world);
        PyReducerRegistry {
            registry,
            name_to_id,
            world_ptr: &world.world as *const World,
        }
    }

    /// Run a reducer by name.
    ///
    /// Parameters are passed as keyword arguments:
    ///   `registry.run("boids_forces", world, world_size=500.0, sep_r=25.0, ...)`
    #[pyo3(signature = (name, world, **kwargs))]
    fn run(
        &self,
        name: &str,
        world: &mut PyWorld,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let actual_ptr = &world.world as *const World;
        if actual_ptr != self.world_ptr {
            return Err(PyValueError::new_err(
                "ReducerRegistry used with a different World than it was created from",
            ));
        }

        let id = self
            .name_to_id
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown reducer: {name}")))?;

        crate::reducers::dispatch(&self.registry, &mut world.world, name, *id, kwargs)
    }

    /// List registered reducer names (sorted).
    fn reducer_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.name_to_id.keys().cloned().collect();
        names.sort();
        names
    }
}
