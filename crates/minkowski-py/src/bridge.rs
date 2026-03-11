//! Arrow bridge: BlobVec column data → Arrow RecordBatch.

use crate::schema::{ComponentSchema, SchemaRegistry};
use arrow::array::{ArrayRef, BooleanArray, Float32Array, UInt8Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use minkowski::{ComponentId, World};
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
    let mut entity_ids: Vec<u64> = Vec::with_capacity(total);

    // For each field, prepare a collector that accumulates raw bytes
    struct ColumnCollector {
        data_type: DataType,
        offset: usize,
        comp_id: ComponentId,
        bytes: Vec<u8>,
    }

    let mut collectors: Vec<ColumnCollector> = Vec::new();
    for schema in &schemas {
        for fm in &schema.fields {
            collectors.push(ColumnCollector {
                data_type: fm.data_type.clone(),
                offset: fm.offset,
                comp_id: schema.component_id,
                bytes: Vec::with_capacity(total * field_byte_size(&fm.data_type)?),
            });
        }
    }

    // Iterate matching archetypes and copy data
    for &arch_idx in &matching_archs {
        let entities = world.archetype_entities(arch_idx);
        let len = entities.len();
        entity_ids.extend(entities.iter().map(|e| e.to_bits()));

        for collector in &mut collectors {
            let field_size = field_byte_size(&collector.data_type)?;
            for row in 0..len {
                // SAFETY: arch_idx and comp_id are valid (we checked contains above),
                // row < len. We only read, never write.
                let ptr = unsafe { world.archetype_column_ptr(arch_idx, collector.comp_id, row) };
                let field_ptr = unsafe { ptr.add(collector.offset) };
                let slice = unsafe { std::slice::from_raw_parts(field_ptr, field_size) };
                collector.bytes.extend_from_slice(slice);
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
            DataType::UInt8 => Arc::new(UInt8Array::from(collector.bytes.clone())),
            DataType::UInt64 => {
                let values: Vec<u64> = collector
                    .bytes
                    .chunks_exact(8)
                    .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Arc::new(UInt64Array::from(values))
            }
            DataType::Boolean => {
                let values: Vec<bool> = collector.bytes.iter().map(|&b| b != 0).collect();
                Arc::new(BooleanArray::from(values))
            }
            dt => return Err(format!("unsupported data type: {dt:?}")),
        };
        columns.push(array);
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| format!("arrow error: {e}"))
}

/// Returns the byte size of a single value for the given Arrow data type.
pub(crate) fn field_byte_size(dt: &DataType) -> Result<usize, String> {
    match dt {
        DataType::Float32 => Ok(4),
        DataType::UInt32 => Ok(4),
        DataType::UInt8 => Ok(1),
        DataType::UInt64 => Ok(8),
        DataType::Boolean => Ok(1), // stored as u8 in Rust component
        _ => Err(format!("unsupported data type: {dt:?}")),
    }
}
