//! Maps Rust component types to Arrow field names, types, and byte offsets.

use arrow::datatypes::DataType;
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

/// Maps one Arrow column to a byte range inside a Rust component struct.
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
        for fm in &schema.fields {
            let fs = crate::bridge::field_byte_size(&fm.data_type).expect("validated data type");
            assert!(
                fm.offset + fs <= schema.size,
                "field '{}' offset {} + size {} exceeds component '{}' size {}",
                fm.column_name,
                fm.offset,
                fs,
                schema.name,
                schema.size
            );
        }
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
        names.sort_unstable();
        names
    }

    #[allow(dead_code)]
    pub fn name_for_id(&self, id: ComponentId) -> Option<&'static str> {
        self.id_to_name.get(&id).copied()
    }
}
