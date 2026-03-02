// WorldQuery is a pub trait that references pub(crate) types (Archetype, Tick)
// in its signatures. Users never implement WorldQuery — they compose existing
// impls via tuples. Suppress until a proper public API facade is built.
#![allow(private_interfaces)]

// Allow the derive macro's generated code (which references `::minkowski::*`)
// to resolve when used inside this crate's own tests.
extern crate self as minkowski;

pub mod access;
pub mod bundle;
pub mod changeset;
pub mod command;
pub mod component;
pub mod entity;
pub mod index;
pub(crate) mod lock_table;
pub mod query;
pub mod storage;
pub mod table;
pub(crate) mod tick;
pub mod transaction;
pub mod world;

pub use access::Access;
pub use changeset::EnumChangeSet;
pub use command::CommandBuffer;
pub use component::ComponentId;
pub use entity::Entity;
pub use index::SpatialIndex;
pub use minkowski_derive::Table;
pub use query::fetch::{Changed, ReadOnlyWorldQuery};
pub use transaction::{
    Conflict, Optimistic, OptimisticTx, Pessimistic, PessimisticTx, Sequential, SequentialTx,
    TransactionStrategy,
};
pub use world::World;
