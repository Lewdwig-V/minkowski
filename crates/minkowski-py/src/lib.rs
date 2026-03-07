//! Python bindings for the Minkowski ECS engine.
//!
//! Exposes World, Entity, and ReducerRegistry directly. Queries return
//! Arrow RecordBatches (one-copy from BlobVec, zero-copy to Python via
//! pyo3-arrow C Data Interface).

use pyo3::prelude::*;

mod bridge;
mod components;
mod pyregistry;
mod pyworld;
mod reducers;
mod schema;

#[pymodule]
fn _minkowski(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyworld::PyWorld>()?;
    m.add_class::<pyregistry::PyReducerRegistry>()?;
    Ok(())
}
