//! Rust component types and their Arrow schema mappings.

use crate::schema::{ComponentSchema, FieldMapping, SchemaRegistry};
use arrow::datatypes::DataType;
use minkowski::World;

// ── Component structs ──

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Acceleration {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mass(pub f32);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct CellState(pub bool);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Heading(pub f32);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Energy(pub f32);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Health(pub u32);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Faction(pub u8);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct WormSize(pub f32);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Nutrition(pub f32);

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

    register_schema!(
        registry,
        world,
        "Position",
        Position,
        [
            ("pos_x", DataType::Float32, offset_of!(Position, x)),
            ("pos_y", DataType::Float32, offset_of!(Position, y)),
        ]
    );
    register_schema!(
        registry,
        world,
        "Velocity",
        Velocity,
        [
            ("vel_x", DataType::Float32, offset_of!(Velocity, x)),
            ("vel_y", DataType::Float32, offset_of!(Velocity, y)),
        ]
    );
    register_schema!(
        registry,
        world,
        "Acceleration",
        Acceleration,
        [
            ("acc_x", DataType::Float32, offset_of!(Acceleration, x)),
            ("acc_y", DataType::Float32, offset_of!(Acceleration, y)),
        ]
    );
    register_schema!(
        registry,
        world,
        "Mass",
        Mass,
        [("mass", DataType::Float32, 0),]
    );
    register_schema!(
        registry,
        world,
        "CellState",
        CellState,
        [("alive", DataType::Boolean, 0),]
    );
    register_schema!(
        registry,
        world,
        "Heading",
        Heading,
        [("heading", DataType::Float32, 0),]
    );
    register_schema!(
        registry,
        world,
        "Energy",
        Energy,
        [("energy", DataType::Float32, 0),]
    );
    register_schema!(
        registry,
        world,
        "Health",
        Health,
        [("health", DataType::UInt32, 0),]
    );
    register_schema!(
        registry,
        world,
        "Faction",
        Faction,
        [("faction", DataType::UInt8, 0),]
    );
    register_schema!(
        registry,
        world,
        "WormSize",
        WormSize,
        [("worm_size", DataType::Float32, 0),]
    );
    register_schema!(
        registry,
        world,
        "Nutrition",
        Nutrition,
        [("nutrition", DataType::Float32, 0),]
    );
}
