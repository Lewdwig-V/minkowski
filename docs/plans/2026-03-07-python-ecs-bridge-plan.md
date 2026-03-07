# Python ECS Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the opaque simulation wrappers with a thin Python API that exposes Minkowski's World, Entity, and ReducerRegistry directly, using Arrow for efficient data transfer.

**Architecture:** PyO3 `#[pyclass]` wrappers around `minkowski::World` and `minkowski::ReducerRegistry`. A `ComponentSchema` registry maps Rust component types to Arrow field names/types/offsets. Queries iterate archetypes via `archetype_column_ptr` and `memcpy` column data into `arrow::array` buffers, returned via `pyo3-arrow` zero-copy FFI. Write-back receives Arrow arrays and copies values into BlobVec columns via `get_mut`.

**Tech Stack:** `pyo3 0.25`, `arrow` (Rust), `pyo3-arrow 0.17`, `minkowski`, `fastrand`, `maturin`

---

### Task 1: Update Cargo.toml and delete old code

**Files:**
- Modify: `crates/minkowski-py/Cargo.toml`
- Rewrite: `crates/minkowski-py/src/lib.rs`
- Modify: `crates/minkowski-py/pyproject.toml`

**Step 1: Update Cargo.toml with new dependencies**

```toml
[package]
name = "minkowski-py"
version = "0.2.0"
edition = "2021"
publish = false

[lib]
name = "_minkowski"
crate-type = ["cdylib"]

[dependencies]
minkowski = { path = "../minkowski" }
pyo3 = { version = "0.25", features = ["extension-module"] }
pyo3-arrow = "0.17"
arrow = { version = "55", default-features = false }
fastrand = "2"
```

**Step 2: Replace lib.rs with a minimal skeleton**

```rust
//! Python bindings for the Minkowski ECS engine.
//!
//! Exposes World, Entity, and ReducerRegistry directly. Queries return
//! Arrow RecordBatches (one-copy from BlobVec, zero-copy to Python via
//! pyo3-arrow C Data Interface).

use pyo3::prelude::*;

#[pymodule]
fn _minkowski(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

**Step 3: Verify it compiles**

Run: `cd crates/minkowski-py && cargo check`
Expected: compiles with no errors (empty module)

**Step 4: Commit**

```bash
git add crates/minkowski-py/
git commit -m "Strip old simulation wrappers, add arrow/pyo3-arrow deps"
```

---

### Task 2: Component schema registry

**Files:**
- Create: `crates/minkowski-py/src/schema.rs`
- Modify: `crates/minkowski-py/src/lib.rs`

**Step 1: Implement ComponentSchema and SchemaRegistry**

`crates/minkowski-py/src/schema.rs`:

```rust
//! Maps Rust component types to Arrow field names, types, and byte offsets.

use arrow::datatypes::{DataType, Field};
use minkowski::ComponentId;
use std::collections::HashMap;

/// Describes how a single Rust component maps to Arrow columns.
pub struct ComponentSchema {
    /// Name used in Python (e.g., "Position")
    pub name: &'static str,
    /// Minkowski ComponentId (resolved at registration)
    pub component_id: ComponentId,
    /// Size of the Rust struct in bytes
    pub size: usize,
    /// Arrow fields with byte offsets into the struct
    pub fields: Vec<FieldMapping>,
}

pub struct FieldMapping {
    /// Arrow column name (e.g., "pos_x")
    pub column_name: &'static str,
    /// Arrow data type
    pub data_type: DataType,
    /// Byte offset within the component struct
    pub offset: usize,
}

/// Registry of all Python-accessible components.
pub struct SchemaRegistry {
    /// name -> schema
    schemas: HashMap<&'static str, ComponentSchema>,
    /// ComponentId -> name (reverse lookup)
    id_to_name: HashMap<ComponentId, &'static str>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            id_to_name: HashMap::new(),
        }
    }

    pub fn register(&mut self, schema: ComponentSchema) {
        let name = schema.name;
        let id = schema.component_id;
        self.id_to_name.insert(id, name);
        self.schemas.insert(name, schema);
    }

    pub fn get(&self, name: &str) -> Option<&ComponentSchema> {
        self.schemas.get(name)
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<_> = self.schemas.keys().copied().collect();
        names.sort();
        names
    }

    pub fn name_for_id(&self, id: ComponentId) -> Option<&'static str> {
        self.id_to_name.get(&id).copied()
    }
}
```

**Step 2: Add module declaration to lib.rs**

Add `mod schema;` to `crates/minkowski-py/src/lib.rs`.

**Step 3: Verify it compiles**

Run: `cd crates/minkowski-py && cargo check`

**Step 4: Commit**

```bash
git add crates/minkowski-py/src/schema.rs crates/minkowski-py/src/lib.rs
git commit -m "Add ComponentSchema registry for Arrow field mapping"
```

---

### Task 3: Component definitions and registration

**Files:**
- Create: `crates/minkowski-py/src/components.rs`
- Modify: `crates/minkowski-py/src/lib.rs`

**Step 1: Define all component structs and their schemas**

`crates/minkowski-py/src/components.rs`:

```rust
//! Rust component types and their Arrow schema mappings.

use crate::schema::{ComponentSchema, FieldMapping, SchemaRegistry};
use arrow::datatypes::DataType;
use minkowski::World;

// ── Component structs ──

#[derive(Clone, Copy)]
pub struct Position { pub x: f32, pub y: f32 }

#[derive(Clone, Copy)]
pub struct Velocity { pub x: f32, pub y: f32 }

#[derive(Clone, Copy)]
pub struct Acceleration { pub x: f32, pub y: f32 }

#[derive(Clone, Copy)]
pub struct Mass(pub f32);

#[derive(Clone, Copy)]
pub struct CellState(pub bool);

#[derive(Clone, Copy)]
pub struct Heading(pub f32);

#[derive(Clone, Copy)]
pub struct Energy(pub f32);

#[derive(Clone, Copy)]
pub struct Health(pub u32);

#[derive(Clone, Copy)]
pub struct Faction(pub u8);

// ── Helper macro for schema registration ──

macro_rules! register_schema {
    ($registry:expr, $world:expr, $name:literal, $type:ty,
     [$(($col:literal, $dtype:expr, $offset:expr)),+ $(,)?]) => {
        let comp_id = $world.register_component::<$type>();
        $registry.register(ComponentSchema {
            name: $name,
            component_id: comp_id,
            size: std::mem::size_of::<$type>(),
            fields: vec![
                $(FieldMapping {
                    column_name: $col,
                    data_type: $dtype,
                    offset: $offset,
                }),+
            ],
        });
    };
}

/// Register all component schemas. Call once at startup.
pub fn register_all(registry: &mut SchemaRegistry, world: &mut World) {
    use std::mem::offset_of;

    register_schema!(registry, world, "Position", Position, [
        ("pos_x", DataType::Float32, offset_of!(Position, x)),
        ("pos_y", DataType::Float32, offset_of!(Position, y)),
    ]);
    register_schema!(registry, world, "Velocity", Velocity, [
        ("vel_x", DataType::Float32, offset_of!(Velocity, x)),
        ("vel_y", DataType::Float32, offset_of!(Velocity, y)),
    ]);
    register_schema!(registry, world, "Acceleration", Acceleration, [
        ("acc_x", DataType::Float32, offset_of!(Acceleration, x)),
        ("acc_y", DataType::Float32, offset_of!(Acceleration, y)),
    ]);
    register_schema!(registry, world, "Mass", Mass, [
        ("mass", DataType::Float32, 0),
    ]);
    register_schema!(registry, world, "CellState", CellState, [
        ("alive", DataType::Boolean, 0),
    ]);
    register_schema!(registry, world, "Heading", Heading, [
        ("heading", DataType::Float32, 0),
    ]);
    register_schema!(registry, world, "Energy", Energy, [
        ("energy", DataType::Float32, 0),
    ]);
    register_schema!(registry, world, "Health", Health, [
        ("health", DataType::UInt32, 0),
    ]);
    register_schema!(registry, world, "Faction", Faction, [
        ("faction", DataType::UInt8, 0),
    ]);
}
```

**Step 2: Add `mod components;` to lib.rs**

**Step 3: Verify it compiles**

Run: `cd crates/minkowski-py && cargo check`

Note: `offset_of!` is stable since Rust 1.77. Verify the toolchain supports it.

**Step 4: Commit**

```bash
git add crates/minkowski-py/src/components.rs crates/minkowski-py/src/lib.rs
git commit -m "Define 9 component types with Arrow schema mappings"
```

---

### Task 4: Arrow bridge — read path (query_arrow)

**Files:**
- Create: `crates/minkowski-py/src/bridge.rs`
- Modify: `crates/minkowski-py/src/lib.rs`

**Step 1: Implement the BlobVec → Arrow RecordBatch conversion**

`crates/minkowski-py/src/bridge.rs`:

```rust
//! Arrow bridge: BlobVec column data → Arrow RecordBatch.

use crate::schema::{ComponentSchema, SchemaRegistry};
use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, UInt8Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use minkowski::{ComponentId, Entity, World};
use std::sync::Arc;

/// Build a RecordBatch from a query over the given component names.
/// The first column is always "entity_id" (UInt64).
pub fn query_to_record_batch(
    world: &mut World,
    schema_registry: &SchemaRegistry,
    component_names: &[&str],
) -> Result<RecordBatch, String> {
    // Resolve schemas
    let schemas: Vec<&ComponentSchema> = component_names
        .iter()
        .map(|name| {
            schema_registry
                .get(name)
                .ok_or_else(|| format!("unknown component: {name}"))
        })
        .collect::<Result<_, _>>()?;

    let comp_ids: Vec<ComponentId> = schemas.iter().map(|s| s.component_id).collect();

    // Collect matching archetype indices
    let arch_count = world.archetype_count();
    let mut matching_archs = Vec::new();
    for arch_idx in 0..arch_count {
        let arch_ids = world.archetype_component_ids(arch_idx);
        if comp_ids.iter().all(|id| arch_ids.contains(id)) && world.archetype_len(arch_idx) > 0 {
            matching_archs.push(arch_idx);
        }
    }

    // Count total entities
    let total: usize = matching_archs.iter().map(|&a| world.archetype_len(a)).sum();

    // Build Arrow fields
    let mut arrow_fields = vec![Field::new("entity_id", DataType::UInt64, false)];
    for schema in &schemas {
        for fm in &schema.fields {
            arrow_fields.push(Field::new(fm.column_name, fm.data_type.clone(), false));
        }
    }
    let arrow_schema = Arc::new(Schema::new(arrow_fields));

    // Prepare column builders
    let mut entity_ids = Vec::with_capacity(total);

    // For each field, prepare a typed Vec
    // We'll collect raw bytes and convert per data type
    struct ColumnCollector {
        data_type: DataType,
        offset: usize,
        comp_size: usize,
        comp_id: ComponentId,
        bytes: Vec<u8>,
    }

    let mut collectors: Vec<ColumnCollector> = Vec::new();
    for schema in &schemas {
        for fm in &schema.fields {
            collectors.push(ColumnCollector {
                data_type: fm.data_type.clone(),
                offset: fm.offset,
                comp_size: schema.size,
                comp_id: schema.component_id,
                bytes: Vec::new(),
            });
        }
    }

    // Iterate matching archetypes and copy data
    for &arch_idx in &matching_archs {
        let entities = world.archetype_entities(arch_idx);
        let len = entities.len();
        entity_ids.extend(entities.iter().map(|e| e.to_bits()));

        for collector in &mut collectors {
            // Get raw pointer to the start of this column in this archetype
            if len > 0 {
                let field_size = match &collector.data_type {
                    DataType::Float32 => 4,
                    DataType::UInt32 => 4,
                    DataType::UInt8 => 1,
                    DataType::UInt64 => 8,
                    DataType::Boolean => 1,
                    _ => return Err(format!("unsupported data type: {:?}", collector.data_type)),
                };

                for row in 0..len {
                    // SAFETY: arch_idx and comp_id are valid (we checked contains above),
                    // row < len. We only read, never write.
                    let ptr = unsafe {
                        world.archetype_column_ptr(arch_idx, collector.comp_id, row)
                    };
                    let field_ptr = unsafe { ptr.add(collector.offset) };
                    let slice = unsafe { std::slice::from_raw_parts(field_ptr, field_size) };
                    collector.bytes.extend_from_slice(slice);
                }
            }
        }
    }

    // Convert collectors to Arrow arrays
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + collectors.len());
    columns.push(Arc::new(UInt64Array::from(entity_ids)));

    for collector in &collectors {
        let array: ArrayRef = match &collector.data_type {
            DataType::Float32 => {
                let values: Vec<f32> = collector
                    .bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Arc::new(Float32Array::from(values))
            }
            DataType::UInt32 => {
                let values: Vec<u32> = collector
                    .bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Arc::new(UInt32Array::from(values))
            }
            DataType::UInt8 => {
                Arc::new(UInt8Array::from(collector.bytes.clone()))
            }
            DataType::Boolean => {
                let values: Vec<bool> = collector.bytes.iter().map(|&b| b != 0).collect();
                Arc::new(BooleanArray::from(values))
            }
            _ => return Err(format!("unsupported data type: {:?}", collector.data_type)),
        };
        columns.push(array);
    }

    RecordBatch::try_new(arrow_schema, columns)
        .map_err(|e| format!("arrow error: {e}"))
}
```

**Step 2: Add `mod bridge;` to lib.rs**

**Step 3: Verify it compiles**

Run: `cd crates/minkowski-py && cargo check`

**Step 4: Commit**

```bash
git add crates/minkowski-py/src/bridge.rs crates/minkowski-py/src/lib.rs
git commit -m "Arrow bridge: BlobVec → RecordBatch via archetype_column_ptr"
```

---

### Task 5: PyWorld — spawn, despawn, query, write_column, introspection

**Files:**
- Create: `crates/minkowski-py/src/pyworld.rs`
- Modify: `crates/minkowski-py/src/lib.rs`

**Step 1: Implement PyWorld**

This is the main `#[pyclass]` exposing World to Python. The key methods are `spawn`, `spawn_batch`, `query_arrow`, `query` (returns Polars), `write_column`, `despawn`, `is_alive`, and introspection helpers.

`crates/minkowski-py/src/pyworld.rs`:

```rust
//! PyO3 wrapper around minkowski::World.

use crate::bridge::query_to_record_batch;
use crate::components;
use crate::schema::SchemaRegistry;
use minkowski::{Entity, World};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_arrow::PyRecordBatch;

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
        components::register_all(&mut schema_registry, &mut world);
        PyWorld {
            world,
            schema_registry,
        }
    }

    /// Spawn a single entity. `components` is a comma-separated string of
    /// component names. Remaining kwargs are field values.
    ///
    /// Example: world.spawn("Position,Velocity", pos_x=0.0, pos_y=0.0, vel_x=1.0, vel_y=0.0)
    #[pyo3(signature = (components, **kwargs))]
    fn spawn(
        &mut self,
        components: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<u64> {
        let kwargs = kwargs.ok_or_else(|| {
            PyValueError::new_err("spawn requires keyword arguments for field values")
        })?;

        let comp_names: Vec<&str> = components.split(',').map(|s| s.trim()).collect();

        // Build raw bytes for each component from kwargs
        let mut comp_bytes: Vec<(minkowski::ComponentId, Vec<u8>)> = Vec::new();

        for name in &comp_names {
            let schema = self
                .schema_registry
                .get(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown component: {name}")))?;

            let mut bytes = vec![0u8; schema.size];
            for fm in &schema.fields {
                let value = kwargs
                    .get_item(fm.column_name)?
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "missing field '{}' for component '{}'",
                            fm.column_name, name
                        ))
                    })?;
                write_field_value(&mut bytes, fm.offset, &fm.data_type, &value)?;
            }
            comp_bytes.push((schema.component_id, bytes));
        }

        // Spawn via raw World API: alloc entity, then insert each component
        let entity = self.world.alloc_entity();
        for (comp_id, bytes) in &comp_bytes {
            // Use insert_raw or build component and insert
            // For now, spawn with the first component bundle and insert rest
            // Actually we need a raw spawn path. Let's use a simpler approach:
            // spawn an empty entity and insert components one by one.
            let layout = self.world.component_layout(*comp_id).unwrap();
            // SAFETY: bytes matches the component layout (we built it from the schema)
            unsafe {
                self.world.insert_raw(entity, *comp_id, bytes.as_ptr(), layout);
            }
        }

        Ok(entity.to_bits())
    }

    /// Query components and return a PyArrow RecordBatch.
    #[pyo3(signature = (*component_names))]
    fn query_arrow(
        &mut self,
        component_names: Vec<String>,
    ) -> PyResult<PyRecordBatch> {
        let names: Vec<&str> = component_names.iter().map(|s| s.as_str()).collect();
        let batch = query_to_record_batch(&mut self.world, &self.schema_registry, &names)
            .map_err(PyValueError::new_err)?;
        Ok(PyRecordBatch::new(batch))
    }

    /// Query components and return a Polars DataFrame.
    #[pyo3(signature = (*component_names))]
    fn query(&mut self, py: Python<'_>, component_names: Vec<String>) -> PyResult<PyObject> {
        let batch = self.query_arrow(component_names)?;
        // Convert RecordBatch → PyArrow table → Polars DataFrame
        let pyarrow = py.import("pyarrow")?;
        let pa_batch = batch.into_pyobject(py)?;
        let table = pyarrow.call_method1(
            "Table",
            (),
        )?;
        let table = table.call_method1("from_batches", (vec![pa_batch],))?;
        let polars = py.import("polars")?;
        let df_class = polars.getattr("DataFrame")?;
        let df = df_class.call1((table,))?;
        Ok(df.into())
    }

    /// Write column data back into the ECS.
    /// entity_ids: list of u64 entity bit-packed IDs
    /// kwargs: column_name → list of values
    #[pyo3(signature = (component, entity_ids, **kwargs))]
    fn write_column(
        &mut self,
        component: &str,
        entity_ids: Vec<u64>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let kwargs = kwargs.ok_or_else(|| {
            PyValueError::new_err("write_column requires keyword arguments")
        })?;
        let schema = self
            .schema_registry
            .get(component)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component: {component}")))?;

        for (i, &bits) in entity_ids.iter().enumerate() {
            let entity = Entity::from_bits(bits);
            // Read current component bytes, modify fields, write back
            // This goes through get_mut to trigger change detection
            for fm in &schema.fields {
                if let Some(py_list) = kwargs.get_item(fm.column_name)? {
                    let value = py_list.get_item(i)?;
                    // Write via get_mut for change detection
                    write_entity_field(
                        &mut self.world,
                        entity,
                        schema.component_id,
                        fm.offset,
                        &fm.data_type,
                        &value,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Despawn an entity by its bit-packed ID.
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

    /// List registered component names.
    fn component_names(&self) -> Vec<&'static str> {
        self.schema_registry.names()
    }
}

// ── Helpers ──

use arrow::datatypes::DataType;

fn write_field_value(
    bytes: &mut [u8],
    offset: usize,
    data_type: &DataType,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match data_type {
        DataType::Float32 => {
            let v: f32 = value.extract()?;
            bytes[offset..offset + 4].copy_from_slice(&v.to_ne_bytes());
        }
        DataType::UInt32 => {
            let v: u32 = value.extract()?;
            bytes[offset..offset + 4].copy_from_slice(&v.to_ne_bytes());
        }
        DataType::UInt8 => {
            let v: u8 = value.extract()?;
            bytes[offset] = v;
        }
        DataType::Boolean => {
            let v: bool = value.extract()?;
            bytes[offset] = v as u8;
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported data type: {data_type:?}"
            )));
        }
    }
    Ok(())
}

fn write_entity_field(
    world: &mut World,
    entity: Entity,
    comp_id: minkowski::ComponentId,
    offset: usize,
    data_type: &DataType,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    // Get mutable pointer to component data (triggers change detection)
    // We need the raw mutable path. Use archetype_column_ptr equivalent for mut.
    // For now, go through typed get_mut per component type — we'll need a raw path.
    // TODO: This needs World::get_mut_raw or similar. For the initial impl,
    // we route through the typed API using a match on component name.
    // This is a limitation we document and can improve later.
    let _ = (world, entity, comp_id, offset, data_type, value);
    Err(PyValueError::new_err(
        "write_column not yet implemented — requires raw mutable column access",
    ))
}
```

Note: `spawn` and `write_column` both need raw byte-level access to component storage. The current Minkowski API has `archetype_column_ptr` (read-only) but no `insert_raw` or mutable raw column access. **Task 5b** addresses this by adding minimal raw APIs to the engine, or by routing through typed APIs with a component name dispatch table.

**Step 2: Wire PyWorld into the module**

In `lib.rs`:
```rust
mod bridge;
mod components;
mod pyworld;
mod schema;

use pyo3::prelude::*;

#[pymodule]
fn _minkowski(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyworld::PyWorld>()?;
    Ok(())
}
```

**Step 3: Verify it compiles**

Run: `cd crates/minkowski-py && cargo check`

Note: The `insert_raw`, `write_entity_field`, and `PyRecordBatch::into_pyobject` calls may need adjustment based on the actual `pyo3-arrow` and `minkowski` APIs. Fix compile errors iteratively.

**Step 4: Commit**

```bash
git add crates/minkowski-py/src/
git commit -m "PyWorld: spawn, query_arrow, query, despawn, introspection"
```

---

### Task 5b: Raw spawn/write support in the engine (if needed)

**Files:**
- Modify: `crates/minkowski/src/world.rs`

If `World` doesn't have `insert_raw(entity, comp_id, *const u8, Layout)`, we need to add it. Check existing API first:

- `world.spawn(bundle)` requires a typed Bundle
- `world.insert(entity, component)` requires a typed component
- `EnumChangeSet::record_insert_raw(entity, comp_id, bytes)` exists for raw inserts

**Option A**: Use `EnumChangeSet` to record raw inserts, then `apply()`. This is the existing raw path.

**Option B**: Add `World::insert_raw(entity: Entity, comp_id: ComponentId, ptr: *const u8, layout: Layout)` — a low-level method that copies bytes into the archetype column.

Recommendation: **Option A** — it already exists, handles archetype migration, and is the intended raw mutation path.

For `write_column` (mutation of existing components), use `world.get_mut::<T>()` routed through a typed dispatch table keyed by component name. This is less elegant but safe and correct:

```rust
// In pyworld.rs — typed dispatch for write-back
fn write_field_to_entity(
    world: &mut World,
    entity: Entity,
    component_name: &str,
    field_name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (component_name, field_name) {
        ("Position", "pos_x") => {
            let v: f32 = value.extract()?;
            world.get_mut::<components::Position>(entity)
                .ok_or_else(|| PyValueError::new_err("entity missing Position"))?
                .x = v;
        }
        ("Position", "pos_y") => {
            let v: f32 = value.extract()?;
            world.get_mut::<components::Position>(entity)
                .ok_or_else(|| PyValueError::new_err("entity missing Position"))?
                .y = v;
        }
        // ... all other component/field combos ...
        _ => return Err(PyValueError::new_err(
            format!("unknown field: {component_name}.{field_name}")
        )),
    }
    Ok(())
}
```

This can be macro-generated to avoid boilerplate. The dispatch table is static and complete for the registered component set.

**Step 1: Implement the typed dispatch table (macro-generated)**

**Step 2: Implement spawn via EnumChangeSet**

**Step 3: Verify compile + basic smoke test**

**Step 4: Commit**

```bash
git commit -m "Raw spawn via EnumChangeSet, typed write-back dispatch"
```

---

### Task 6: Reducer bridge — register and call from Python

**Files:**
- Create: `crates/minkowski-py/src/reducers.rs`
- Create: `crates/minkowski-py/src/pyregistry.rs`
- Modify: `crates/minkowski-py/src/lib.rs`

**Step 1: Implement Rust reducers for existing examples**

`crates/minkowski-py/src/reducers.rs` — contains the actual reducer closures (boids_forces, boids_integrate, gravity, life_step, movement) using the component types from `components.rs`. These are the hot loops that Python calls by name.

Each reducer is registered via `ReducerRegistry::register_query` with a parameter struct extracted from Python kwargs.

**Step 2: Implement PyReducerRegistry**

`crates/minkowski-py/src/pyregistry.rs`:

```rust
use crate::pyworld::PyWorld;
use minkowski::{QueryReducerId, ReducerRegistry};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass(name = "ReducerRegistry")]
pub struct PyReducerRegistry {
    registry: ReducerRegistry,
    name_to_id: HashMap<String, QueryReducerId>,
}

#[pymethods]
impl PyReducerRegistry {
    #[new]
    fn new(world: &mut PyWorld) -> Self {
        let mut registry = ReducerRegistry::new();
        let name_to_id = crate::reducers::register_all(
            &mut registry,
            &mut world.world,
        );
        PyReducerRegistry { registry, name_to_id }
    }

    /// Run a reducer by name with keyword parameters.
    #[pyo3(signature = (name, world, **kwargs))]
    fn run(
        &self,
        name: &str,
        world: &mut PyWorld,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let id = self.name_to_id.get(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown reducer: {name}")))?;

        // Extract parameters and dispatch
        // Each reducer has its own parameter extraction logic
        crate::reducers::dispatch(
            &self.registry,
            &mut world.world,
            name,
            *id,
            kwargs,
        )
    }

    /// List registered reducer names.
    fn reducer_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.name_to_id.keys().cloned().collect();
        names.sort();
        names
    }
}
```

**Step 3: Wire into module**

```rust
m.add_class::<pyregistry::PyReducerRegistry>()?;
```

**Step 4: Verify compile**

Run: `cd crates/minkowski-py && cargo check`

**Step 5: Commit**

```bash
git add crates/minkowski-py/src/reducers.rs crates/minkowski-py/src/pyregistry.rs crates/minkowski-py/src/lib.rs
git commit -m "Reducer bridge: register Rust systems, call from Python by name"
```

---

### Task 7: Update __init__.py and pyproject.toml

**Files:**
- Rewrite: `crates/minkowski-py/python/minkowski_py/__init__.py`
- Modify: `crates/minkowski-py/pyproject.toml`

**Step 1: Update __init__.py**

```python
"""Minkowski ECS — Python bindings for the Minkowski entity-component system.

Quick start::

    import minkowski_py as mk

    world = mk.World()
    registry = mk.ReducerRegistry(world)

    # Spawn entities
    world.spawn("Position,Velocity", pos_x=0.0, pos_y=0.0, vel_x=1.0, vel_y=0.0)

    # Query as Polars DataFrame
    df = world.query("Position", "Velocity")

    # Run a Rust reducer
    registry.run("movement", world, dt=0.016)
"""

try:
    from minkowski_py._minkowski import World, ReducerRegistry
except ImportError as e:
    raise ImportError(
        "Failed to import Minkowski native module. "
        "Build with: cd crates/minkowski-py && maturin develop --release"
    ) from e

__all__ = ["World", "ReducerRegistry"]
__version__ = "0.2.0"
```

**Step 2: Update pyproject.toml**

Bump version to 0.2.0. Keep dependencies (pyarrow, polars, matplotlib, numpy).

**Step 3: Commit**

```bash
git add crates/minkowski-py/python/ crates/minkowski-py/pyproject.toml
git commit -m "Update Python package for ECS bridge API"
```

---

### Task 8: Build and smoke test

**Step 1: Build with maturin**

Run: `cd crates/minkowski-py && maturin develop --release`

**Step 2: Smoke test in Python**

```bash
python3 -c "
import minkowski_py as mk
world = mk.World()
e = world.spawn('Position,Velocity', pos_x=1.0, pos_y=2.0, vel_x=0.5, vel_y=-0.5)
print(f'Spawned entity: {e}')
print(f'Alive: {world.is_alive(e)}')
print(f'Entity count: {world.entity_count()}')
print(f'Components: {world.component_names()}')
table = world.query_arrow('Position', 'Velocity')
print(f'Arrow table: {table}')
df = world.query('Position', 'Velocity')
print(df)
"
```

Expected: Prints entity ID, alive=True, count=1, component list, Arrow table schema, and a Polars DataFrame with one row.

**Step 3: Fix any issues**

**Step 4: Commit**

```bash
git commit -m "Smoke test passing: spawn, query_arrow, query all work"
```

---

### Task 9: Notebooks

**Files:**
- Rewrite: `notebooks/01_quickstart.ipynb` (formerly 01_boids.ipynb)
- Rewrite: `notebooks/02_boids.ipynb`
- Rewrite: `notebooks/03_nbody.ipynb`
- Rewrite: `notebooks/04_life.ipynb`

Each notebook should demonstrate the API from the design doc. The quickstart shows the full spawn→query→modify→write-back loop. The simulation notebooks show Rust reducer usage with Polars analysis and matplotlib visualization.

**Step 1: Write 01_quickstart.ipynb**

Key cells: import, create world, spawn entities, query as DataFrame, plot, modify in Python, write back, query again to verify.

**Step 2: Write 02_boids.ipynb**

Key cells: spawn boids, run boids_forces + boids_integrate reducers in a loop, query + record history, plot trajectories and parameter sweeps.

**Step 3: Write 03_nbody.ipynb and 04_life.ipynb**

Similar pattern with gravity and life_step reducers.

**Step 4: Run all notebooks end-to-end**

**Step 5: Commit**

```bash
git add notebooks/
git commit -m "Rewrite notebooks for ECS bridge API"
```

---

### Task 10: Clean up and push

**Step 1: Run clippy on the py crate**

Run: `cd crates/minkowski-py && cargo clippy -- -D warnings`

**Step 2: Run workspace tests**

Run: `cargo test -p minkowski --lib`

**Step 3: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

**Step 4: Push**

```bash
git push
```

---

## Implementation Notes

### Known challenges

1. **`insert_raw` / raw spawn**: Minkowski doesn't expose raw byte-level spawn. Use `EnumChangeSet::record_insert_raw` + `apply()` as the existing raw path. If that doesn't handle initial placement (only archetype migration), we may need to add a `World::spawn_raw` method.

2. **Write-back change detection**: `write_column` needs to go through `get_mut` (which marks columns changed). A typed dispatch table (macro-generated) routes `(component_name, field_name)` → `world.get_mut::<T>().field = value`. Not elegant, but correct and safe.

3. **`pyo3-arrow` API**: The `PyRecordBatch` type from `pyo3-arrow` handles Arrow C Data Interface export. Verify the exact API for converting `arrow::RecordBatch` → Python object. May need `into_pyobject` or `to_pyarrow`.

4. **Boolean columns**: Arrow's `BooleanArray` uses bit-packing, but Minkowski's `CellState(bool)` stores one byte per bool. The bridge must convert between these representations.

5. **Query → Polars**: The path is RecordBatch → PyArrow Table → `pl.DataFrame()`. The `pyarrow.Table.from_batches([batch])` call creates the table. Verify this works with pyo3-arrow's output type.
