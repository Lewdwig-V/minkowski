// WorldQuery and Bundle are pub traits that reference pub(crate) types in their signatures.
// Proper fix is to make the public API facade consistent; suppress for now.
#![allow(private_interfaces)]

pub mod bundle;
pub mod command;
pub mod component;
pub mod entity;
pub mod query;
pub mod storage;
pub mod world;

pub use command::CommandBuffer;
pub use entity::Entity;
pub use world::World;
