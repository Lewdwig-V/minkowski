// WorldQuery is a pub trait that references pub(crate) types (Archetype, Tick)
// in its signatures. Users never implement WorldQuery — they compose existing
// impls via tuples. Suppress until a proper public API facade is built.
#![allow(private_interfaces)]

// Allow the derive macro's generated code (which references `::minkowski::*`)
// to resolve when used inside this crate's own tests.
extern crate self as minkowski;

pub mod bundle;
pub mod changeset;
pub mod command;
pub mod component;
pub mod entity;
pub mod index;
pub mod query;
pub mod storage;
pub mod table;
pub(crate) mod tick;
pub mod world;

pub use changeset::EnumChangeSet;
pub use command::CommandBuffer;
pub use component::ComponentId;
pub use entity::Entity;
pub use index::SpatialIndex;
pub use minkowski_derive::Table;
pub use query::fetch::Changed;
pub use world::World;
