# ADR-014: PyO3-Arrow Bridge

**Status:** Accepted
**Date:** 2026-03-07

## Context

Minkowski's column-oriented archetype storage is structurally similar to Arrow's columnar format — both store typed arrays of values contiguously in memory. Exposing the ECS to Python enables Jupyter-based visualization, data science workflows, and hybrid Rust/Python simulation patterns where Rust handles hot-loop physics and Python handles structural decisions (spawn, despawn, cross-archetype queries).

The bridge must cross two boundaries: Rust-to-Python (PyO3) and ECS-to-dataframe (Arrow). The key tension is between type safety (Rust components are generic, statically typed) and dynamism (Python users specify components by name at runtime).

## Decision

A three-layer architecture bridges the ECS to Python:

1. **SchemaRegistry** maps component names to `ComponentSchema` descriptors containing a `ComponentId`, struct size, and `FieldMapping`s (Arrow column name, `DataType`, byte offset within the Rust struct). Components are `#[repr(C)]` or `#[repr(transparent)]` so field offsets are stable and known at compile time via `std::mem::offset_of!`.

2. **Arrow bridge** (`query_to_record_batch`) scans archetypes matching the requested components, copies raw bytes from BlobVec columns into per-field byte buffers via `archetype_column_ptr`, then reinterprets those buffers as typed Arrow arrays. The result is a standard Arrow `RecordBatch` with an `entity_id: UInt64` column prepended.

3. **PyO3 surface** (`PyWorld`) exposes `query()` → Polars DataFrame and `query_arrow()` → PyArrow RecordBatch. Transfer from Rust to Python uses pyo3-arrow's C Data Interface — zero-copy across the FFI boundary. The `spawn`/`spawn_batch`/`write_column` write path uses string-keyed dispatch to typed Rust calls.

**Key insight: the copy budget is exactly one — BlobVec bytes are memcpy'd into Arrow arrays (one copy), then Arrow-to-Python is zero-copy via the C Data Interface. This is the minimum possible for a type-erased columnar store.**

Reducers and spatial indexes are separate pyclasses (`PyReducerRegistry`, `PySpatialGrid`) that compose externally with `PyWorld`, following the same composition-over-integration pattern as the Rust API. A `CircuitSim` pyclass demonstrates non-ECS simulation exposed through the same Arrow waveform interface.

## Alternatives Considered

- **Automatic schema derivation** (proc macro generates `ComponentSchema` from struct definitions) — rejected because the field-to-column name mapping is a design decision (`Position.x` → `pos_x`), not a mechanical derivation. A macro would either use field names verbatim (poor Python ergonomics) or require annotation attributes (same effort as the current registry, more complexity).

- **Generic query API** (Python passes component types or schemas dynamically, no dispatch tables) — rejected because Rust's `world.spawn()` and `world.get_mut()` require static type parameters. The dispatch tables in `spawn_typed` and `write_field_to_entity` are the unavoidable cost of bridging static Rust generics to dynamic Python. Each new component combination requires a dispatch arm — this is explicit and auditable, unlike a proc macro that hides the mapping.

- **serde-based serialization** (serialize components to JSON/MessagePack, deserialize in Python) — rejected because it adds a second copy and per-value overhead. The byte-offset approach copies raw struct memory directly into Arrow buffers, avoiding any serialization framework.

- **Direct BlobVec exposure** (mmap or share BlobVec pointers with Python) — rejected because BlobVec uses 64-byte-aligned allocations with Rust's allocator. Sharing raw pointers across FFI requires lifetime management that PyO3's GIL model doesn't support safely. The one-copy approach is both safe and fast enough (memcpy is bandwidth-bound, not latency-bound at ECS scale).

## Consequences

- Adding a new component to the Python bridge requires: a Rust struct in `components.rs`, a `register_schema!` call, builder/write dispatch arms in `pyworld.rs`, and spawn dispatch arms for each bundle combination — approximately 10-20 lines per component, all mechanical.
- Queries return Polars DataFrames by default, giving Python users familiar `.filter()`, `.select()`, `.to_numpy()` APIs without learning ECS-specific query syntax.
- The `entity_id` column in every query result enables write-back via `write_column(component, entity_ids, **kwargs)` — a columnar scatter-write pattern natural to dataframe users.
- Write path is per-entity (`get_mut` per entity per field) — adequate for Python-speed mutation but not suitable for bulk Rust-speed writes. Bulk mutation belongs in Rust reducers.
- `pyo3-arrow` and `arrow` are dependencies of `minkowski-py` only — the core ECS crate has no Arrow dependency.
- The bridge validates schemas at registration time (field offsets within struct bounds) and at query time (component existence), but trusts `archetype_column_ptr` safety — the same pointer arithmetic the Rust query system uses.
