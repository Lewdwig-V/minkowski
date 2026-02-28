// WorldQuery and Bundle are pub traits that reference pub(crate) types in their signatures.
// Proper fix is to make the public API facade consistent; suppress for now.
#![allow(private_interfaces)]

// Allow the derive macro's generated code (which references `::minkowski::*`)
// to resolve when used inside this crate's own tests.
extern crate self as minkowski;

pub mod bundle;
pub mod changeset;
pub mod command;
pub mod component;
pub mod entity;
pub mod query;
pub mod storage;
pub mod table;
pub mod world;

pub use changeset::EnumChangeSet;
pub use command::CommandBuffer;
pub use entity::Entity;
pub use minkowski_derive::Table;
pub use world::World;
