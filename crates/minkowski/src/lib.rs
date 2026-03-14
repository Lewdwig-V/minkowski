//! # Minkowski
//!
//! A column-oriented database engine built on an archetype Entity-Component System.
//!
//! Minkowski combines the runtime flexibility of entity-component systems with
//! the performance characteristics of analytical databases: cache-friendly
//! columnar storage, SIMD-friendly iteration, and compile-time schema
//! declarations via [`Table`].
//!
//! ## Core concepts
//!
//! - **[`World`]** — the central store. Holds all entities, components, and
//!   archetype metadata. Most operations start here.
//! - **[`Entity`]** — a lightweight handle (generational u64) identifying
//!   a row across archetypes.
//! - **Components** — any `'static + Send + Sync` type. Stored in contiguous
//!   columns within archetypes for cache-friendly iteration.
//! - **Queries** — [`world.query::<(&mut Pos, &Vel)>()`](World::query)
//!   iterates matching archetypes via bitset matching with incremental caching.
//!
//! ## Typed reducers
//!
//! The [`ReducerRegistry`] provides typed closures whose signatures declare
//! exactly which components they read and write. This enables:
//!
//! - **Compile-time conflict proofs** — the scheduler can verify two reducers
//!   touch disjoint data without running them.
//! - **Six handle types** — [`EntityRef`], [`EntityMut`], [`QueryRef`],
//!   [`QueryMut`], [`QueryWriter`], [`Spawner`] — each exposing only the
//!   operations the reducer declared.
//! - **Three execution models** — transactional (buffered writes),
//!   scheduled (direct `&mut World`), and dynamic (runtime-validated access
//!   via [`DynamicCtx`]).
//!
//! ## Transactions
//!
//! The [`Transact`] trait provides closure-based transactions with three
//! built-in strategies:
//!
//! - [`Sequential`] — zero-cost passthrough for single-threaded use.
//! - [`Optimistic`] — tick-based validation, retries on conflict.
//! - [`Pessimistic`] — cooperative per-column locks, guaranteed commit.
//!
//! All strategies use split-phase execution: [`Tx`] doesn't hold `&mut World`,
//! enabling concurrent reads via `tx.query(&world)` (bounded by
//! [`ReadOnlyWorldQuery`] to prevent aliased `&mut T`).
//!
//! ## Mutation
//!
//! - [`CommandBuffer`] — deferred structural changes during iteration.
//! - [`EnumChangeSet`] — data-driven mutations with automatic reverse
//!   generation for rollback. The serialization boundary for WAL persistence.
//!
//! ## Where to start
//!
//! 1. Create a [`World`] and spawn entities with component tuples.
//! 2. Query with [`world.query::<Q>()`](World::query) or register reducers
//!    on a [`ReducerRegistry`].
//! 3. For concurrency, wrap dispatch in a [`Transact`] strategy.
//! 4. For persistence, use `minkowski_persist::Durable` around any strategy.
//!
//! See the `examples/` directory for complete programs exercising every feature.

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
pub mod planner;
pub(crate) mod pool;
pub mod query;
pub mod reducer;
pub mod retention;
pub mod storage;
pub(crate) mod sync;
pub mod table;
pub mod tick;
pub mod transaction;
pub mod world;

pub use access::Access;
pub use changeset::{ApplyError, EnumChangeSet, MutationRef};
pub use command::CommandBuffer;
pub use component::ComponentId;
pub use entity::Entity;
pub use index::{
    BTreeIndex, HasBTreeIndex, HasHashIndex, HashIndex, SpatialCost, SpatialExpr, SpatialIndex,
};
pub use minkowski_derive::Table;
pub use planner::{
    CardinalityConstraint, Cost, IndexKind, Indexed, JoinKind, PlanNode, PlanWarning, PlannerError,
    Predicate, QueryPlanResult, QueryPlanner, SpatialLookupFn, SpatialPredicate,
    SubscriptionBuilder, SubscriptionError, SubscriptionPlan, TablePlanner, VecExecNode,
    VectorizeOpts, VectorizedPlan,
};
pub use pool::{HugePages, PoolExhausted};
pub use query::fetch::{Changed, ReadOnlyWorldQuery};
pub use reducer::{
    ComponentSet, Contains, DynamicCtx, DynamicReducerBuilder, DynamicReducerId, EntityMut,
    EntityRef, QueryMut, QueryReducerId, QueryRef, QueryWriter, ReducerError, ReducerId,
    ReducerInfo, ReducerRegistry, Spawner, WritableRef, WriterQuery,
};
pub use retention::Expiry;
pub use tick::ChangeTick;
pub use transaction::{
    Conflict, Optimistic, Pessimistic, Sequential, SequentialTx, Transact, TransactError, Tx,
    WorldMismatch,
};
pub use world::{DeadEntity, InsertError, QueryTickInfo, World, WorldBuilder, WorldStats};
