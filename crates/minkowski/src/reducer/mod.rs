mod dynamic;
mod handles;
#[cfg(test)]
mod tests;
mod writer;

use std::any::Any;
use std::collections::HashMap;
use std::fmt;

use crate::sync::{Arc, AtomicBool, AtomicU64, Ordering};

use crate::access::Access;
use crate::bundle::Bundle;
use crate::changeset::EnumChangeSet;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
use crate::transaction::{Conflict, Transact, TransactError, WorldMismatch};
use crate::world::World;

// ── Re-exports ──────────────────────────────────────────────────────

pub use dynamic::{ComponentSet, Contains, DynamicCtx, DynamicReducerBuilder, DynamicReducerId};
pub use handles::{EntityMut, EntityRef, QueryMut, QueryRef, Spawner};
pub use writer::{QueryWriter, WritableRef, WriterQuery};

// ── Internal re-exports for submodules ──────────────────────────────

use dynamic::{DynamicAdapter, DynamicResolved};

// ── ReducerError ─────────────────────────────────────────────────────

/// Error type for reducer dispatch and registration failures.
///
/// These are API-misuse errors that can be checked at the call site without
/// panicking. Access-boundary violations inside reducer closures (e.g.
/// reading an undeclared component in `DynamicCtx`) still panic per the
/// assert boundary rule — they indicate broken invariants, not recoverable
/// conditions.
#[derive(Debug)]
pub enum ReducerError {
    /// Attempted to call a scheduled reducer with `call()`, or a
    /// transactional reducer with `run()`.
    WrongKind {
        /// `"transactional"` or `"scheduled"`.
        expected: &'static str,
        /// `"transactional"` or `"scheduled"`.
        actual: &'static str,
    },
    /// A reducer with this name was already registered.
    DuplicateName {
        name: &'static str,
        existing_kind: &'static str,
        existing_index: usize,
    },
    /// Transaction conflict (wraps [`Conflict`]).
    TransactionConflict(Conflict),
    /// The reducer ID does not refer to a valid entry in this registry.
    /// Caused by using an ID from a different registry or after the
    /// registry has been rebuilt.
    InvalidId {
        /// `"reducer"` or `"dynamic"`.
        kind: &'static str,
        index: usize,
        max: usize,
    },
    /// The transaction strategy was used with a different World than it
    /// was created from.
    WorldMismatch(WorldMismatch),
}

impl fmt::Display for ReducerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReducerError::WrongKind { expected, actual } => {
                write!(
                    f,
                    "reducer kind mismatch: expected {expected}, got {actual}"
                )
            }
            ReducerError::DuplicateName {
                name,
                existing_kind,
                existing_index,
            } => {
                write!(
                    f,
                    "duplicate reducer name '{name}' \
                     (already registered as {existing_kind} reducer at index {existing_index})"
                )
            }
            ReducerError::TransactionConflict(c) => {
                write!(f, "transaction conflict: {c}")
            }
            ReducerError::InvalidId { kind, index, max } => {
                write!(
                    f,
                    "invalid {kind} reducer ID (index {index}, registry has {max})"
                )
            }
            ReducerError::WorldMismatch(w) => write!(f, "{w}"),
        }
    }
}

impl std::error::Error for ReducerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReducerError::TransactionConflict(c) => Some(c),
            ReducerError::WorldMismatch(w) => Some(w),
            _ => None,
        }
    }
}

impl From<TransactError> for ReducerError {
    fn from(e: TransactError) -> Self {
        match e {
            TransactError::Conflict(c) => ReducerError::TransactionConflict(c),
            TransactError::WorldMismatch(w) => ReducerError::WorldMismatch(w),
        }
    }
}

/// Introspection descriptor for a registered reducer.
///
/// Returned by [`ReducerRegistry::reducer_info`], [`query_reducer_info`](ReducerRegistry::query_reducer_info),
/// and [`dynamic_reducer_info`](ReducerRegistry::dynamic_reducer_info).
#[derive(Debug, Clone)]
pub struct ReducerInfo {
    /// Registration name.
    pub name: &'static str,
    /// `"transactional"`, `"scheduled"`, or `"dynamic"`.
    pub kind: &'static str,
    /// Component-level access bitsets.
    pub access: Access,
    /// Whether this reducer has `Changed<T>` tick tracking.
    pub has_change_tracking: bool,
    /// Whether this reducer declares despawn capability.
    pub can_despawn: bool,
}

// ── Internal types ──────────────────────────────────────────────────

/// Pre-resolved ComponentIds created once at registration time.
/// `Contains<T, INDEX>` positions index into the inner Vec.
pub(crate) struct ResolvedComponents(pub(crate) Vec<ComponentId>);

/// Restricted view of `&mut World` for transactional adapters. Exposes
/// only what reducers legitimately need — archetype iteration, component
/// registry, tick advancement — without the full `World` mutation API.
///
/// Prevents transactional closures from calling `world.spawn()`,
/// `world.insert()`, or `world.query::<(&mut T,)>()` directly,
/// which would bypass the ChangeSet and break optimistic validation.
pub(crate) struct TransactionalWorld<'a>(pub(crate) &'a mut World);

impl TransactionalWorld<'_> {
    /// Reborrow as `&World` for read-only access (entity reducers, spawners).
    pub(crate) fn as_ref(&self) -> &World {
        self.0
    }
}

impl std::ops::Deref for TransactionalWorld<'_> {
    type Target = World;
    fn deref(&self) -> &World {
        self.0
    }
}

/// Type-erased transactional reducer adapter. Receives changeset + allocated list
/// (from Tx), a restricted world view, resolved IDs, and type-erased args.
type TransactionalAdapter = Box<
    dyn Fn(
            &mut EnumChangeSet,
            &mut Vec<Entity>,
            &mut TransactionalWorld<'_>,
            &ResolvedComponents,
            &dyn Any,
        ) + Send
        + Sync,
>;

/// Type-erased scheduled reducer adapter.
type ScheduledAdapter = Box<dyn Fn(&mut World, &dyn Any) + Send + Sync>;

/// Two execution models for reducers.
enum ReducerKind {
    /// Runs inside `strategy.transact()`. Entity + args from call site.
    Transactional(TransactionalAdapter),
    /// Runs with direct `&mut World`.
    Scheduled(ScheduledAdapter),
}

struct ReducerEntry {
    name: &'static str,
    access: Access,
    resolved: ResolvedComponents,
    kind: ReducerKind,
    /// Per-reducer tick for `Changed<T>` support in `QueryWriter`.
    /// `None` for non-query-writer reducers.
    last_read_tick: Option<Arc<AtomicU64>>,
    /// Set to `true` by `for_each`/`count` — tick only advances if query ran.
    queried: Option<Arc<AtomicBool>>,
}

pub(super) struct DynamicReducerEntry {
    pub(super) name: &'static str,
    pub(super) resolved: DynamicResolved,
    pub(super) closure: DynamicAdapter,
    pub(super) last_read_tick: Arc<AtomicU64>,
}

/// Discriminant for the `by_name` lookup table.
#[derive(Clone, Copy)]
pub(super) enum ReducerSlot {
    Unified(usize),
    Dynamic(usize),
}

// ── ReducerRegistry ──────────────────────────────────────────────────

/// Typed handle for dispatching transactional reducers via [`ReducerRegistry::call`].
///
/// Obtained from [`ReducerRegistry::register_entity`],
/// [`register_entity_despawn`](ReducerRegistry::register_entity_despawn),
/// [`register_spawner`](ReducerRegistry::register_spawner), or
/// [`register_query_writer`](ReducerRegistry::register_query_writer).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ReducerId(pub(crate) usize);

impl ReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Typed handle for dispatching scheduled query reducers via [`ReducerRegistry::run`].
///
/// Obtained from [`ReducerRegistry::register_query`] or
/// [`register_query_ref`](ReducerRegistry::register_query_ref).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QueryReducerId(pub(crate) usize);

impl QueryReducerId {
    /// Raw index for serialization / external storage.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Central registry for typed reducer closures with conflict analysis.
///
/// Owns closures, [`Access`] metadata, and pre-resolved `ComponentId`s.
/// Composes with [`World`] and [`Transact`] strategies the same way
/// [`SpatialIndex`](crate::SpatialIndex) composes with World — no World API growth.
///
/// ## Registration
///
/// - [`register_entity`](ReducerRegistry::register_entity) / [`register_entity_despawn`](ReducerRegistry::register_entity_despawn) — single-entity read-write via [`EntityMut`]
/// - [`register_entity_ref`](ReducerRegistry::register_entity_ref) — single-entity read-only via [`EntityRef`]
/// - [`register_spawner`](ReducerRegistry::register_spawner) — entity creation via [`Spawner`]
/// - [`register_query_writer`](ReducerRegistry::register_query_writer) — buffered query iteration via [`QueryWriter`]
/// - [`register_query`](ReducerRegistry::register_query) — direct mutable iteration via [`QueryMut`]
/// - [`register_query_ref`](ReducerRegistry::register_query_ref) — read-only iteration via [`QueryRef`]
/// - [`dynamic`](ReducerRegistry::dynamic) — runtime-validated access via [`DynamicReducerBuilder`]
///
/// ## Dispatch
///
/// - [`call`](ReducerRegistry::call) — transactional reducers (entity, spawner, query writer), runs through `strategy.transact()`
/// - [`run`](ReducerRegistry::run) — scheduled query reducers, direct `&mut World`
/// - [`dynamic_call`](ReducerRegistry::dynamic_call) — dynamic reducers, routes through `strategy.transact()`
///
/// ## Conflict analysis
///
/// - [`reducer_access`](ReducerRegistry::reducer_access) / [`query_reducer_access`](ReducerRegistry::query_reducer_access) / [`dynamic_access`](ReducerRegistry::dynamic_access) — retrieve [`Access`] bitsets for scheduler conflict detection
/// - [`reducer_id_by_name`](ReducerRegistry::reducer_id_by_name) / [`query_reducer_id_by_name`](ReducerRegistry::query_reducer_id_by_name) / [`dynamic_id_by_name`](ReducerRegistry::dynamic_id_by_name) — name-based lookup for network dispatch
pub struct ReducerRegistry {
    reducers: Vec<ReducerEntry>,
    pub(super) dynamic_reducers: Vec<DynamicReducerEntry>,
    pub(super) by_name: HashMap<&'static str, ReducerSlot>,
}

impl ReducerRegistry {
    pub fn new() -> Self {
        Self {
            reducers: Vec::new(),
            dynamic_reducers: Vec::new(),
            by_name: HashMap::new(),
        }
    }

    // ── Transactional registration ───────────────────────────────

    /// Register an entity reducer: `f(EntityMut<C>, args)`.
    /// At dispatch, call with `(entity, args)` as the args tuple.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_entity<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        C: ComponentSet,
        Args: Clone + 'static,
        F: Fn(EntityMut<'_, C>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(C::resolve(&mut world.components));
        // EntityMut can both read (get) and write (set) all components in C.
        let reads = C::access(&mut world.components, true);
        let writes = C::access(&mut world.components, false);
        let access = reads.merge(&writes);

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, _allocated, tw, resolved, args_any| {
                let (entity, args) = args_any
                    .downcast_ref::<(Entity, Args)>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected (Entity, {})",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref(), false);
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register an entity reducer with despawn capability.
    /// Same as `register_entity`, but `EntityMut::despawn()` is enabled
    /// and the Access includes the despawn flag.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_entity_despawn<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        C: ComponentSet,
        Args: Clone + 'static,
        F: Fn(EntityMut<'_, C>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(C::resolve(&mut world.components));
        let reads = C::access(&mut world.components, true);
        let writes = C::access(&mut world.components, false);
        let mut access = reads.merge(&writes);
        access.set_despawns();

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, _allocated, tw, resolved, args_any| {
                let (entity, args) = args_any
                    .downcast_ref::<(Entity, Args)>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected (Entity, {})",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = EntityMut::<C>::new(entity, resolved, changeset, tw.as_ref(), true);
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register a read-only entity reducer: `f(EntityRef<C>, args)`.
    /// At dispatch, call with `(entity, args)` as the args tuple.
    ///
    /// Unlike [`register_entity`](Self::register_entity), this provides
    /// read-only access via [`EntityRef`] — no writes are buffered, and
    /// the access metadata reflects reads only. Use this when the reducer
    /// only needs to inspect component values without modifying them.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_entity_ref<C, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        C: ComponentSet,
        Args: Clone + 'static,
        F: Fn(EntityRef<'_, C>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(C::resolve(&mut world.components));
        // EntityRef is read-only — no write access needed.
        let access = C::access(&mut world.components, true);

        let adapter: TransactionalAdapter =
            Box::new(move |_changeset, _allocated, tw, resolved, args_any| {
                let (entity, args) = args_any
                    .downcast_ref::<(Entity, Args)>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected (Entity, {})",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = EntityRef::<C>::new(entity, resolved, tw.as_ref());
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register a spawner reducer: `f(Spawner<B>, args)`.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_spawner<B, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        B: Bundle,
        Args: Clone + 'static,
        F: Fn(Spawner<'_, B>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(B::component_ids(&mut world.components));
        let access = Access::empty(); // spawner creates new entities, no column conflicts

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, allocated, tw, _resolved, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected {}",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let handle = Spawner::<B>::new(changeset, allocated, tw.as_ref());
                f(handle, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            None,
            None,
        )
    }

    /// Register a query writer reducer: `f(QueryWriter<Q>, args)`.
    ///
    /// Iterates matching archetypes with buffered writes. `&T` reads directly
    /// from columns; `&mut T` produces `WritableRef<T>` that buffers into the
    /// transaction's changeset. Column ticks are NOT advanced during iteration
    /// (avoiding self-conflict with optimistic validation). Changes are applied
    /// atomically on commit.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_query_writer<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<ReducerId, ReducerError>
    where
        Q: WriterQuery + 'static,
        Args: Clone + 'static,
        F: Fn(QueryWriter<'_, Q>, Args) + Send + Sync + 'static,
    {
        Q::register(&mut world.components);
        let resolved = ResolvedComponents(Vec::new());
        let access = Access::of::<Q>(world);
        let last_read_tick = Arc::new(AtomicU64::new(0));
        let tick_ref = last_read_tick.clone();
        let queried = Arc::new(AtomicBool::new(false));
        let queried_ref = queried.clone();

        let adapter: TransactionalAdapter =
            Box::new(move |changeset, _allocated, tw, _resolved, args_any| {
                let args = args_any
                    .downcast_ref::<Args>()
                    .unwrap_or_else(|| {
                        panic!(
                            "reducer args type mismatch: expected {}",
                            std::any::type_name::<Args>()
                        )
                    })
                    .clone();
                let cs_ptr: *mut EnumChangeSet = changeset;
                let qw = QueryWriter::<Q>::new(tw.0, cs_ptr, &tick_ref, &queried_ref);
                f(qw, args);
            });

        self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Transactional(adapter),
            Some(last_read_tick),
            Some(queried),
        )
    }

    // ── Scheduled registration ───────────────────────────────────

    /// Register a mutable query reducer: `f(QueryMut<Q>, args)`.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_query<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<QueryReducerId, ReducerError>
    where
        Q: WorldQuery + 'static,
        Args: Clone + 'static,
        F: Fn(QueryMut<'_, Q>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(Vec::new());
        let access = Access::of::<Q>(world);

        let adapter: ScheduledAdapter = Box::new(move |world, args_any| {
            let args = args_any
                .downcast_ref::<Args>()
                .unwrap_or_else(|| {
                    panic!(
                        "reducer args type mismatch: expected {}",
                        std::any::type_name::<Args>()
                    )
                })
                .clone();
            let qm = QueryMut::<Q>::new(world);
            f(qm, args);
        });

        let id = self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Scheduled(adapter),
            None,
            None,
        )?;
        Ok(QueryReducerId(id.0))
    }

    /// Register a read-only query reducer: `f(QueryRef<Q>, args)`.
    ///
    /// Uses the full query path with filter support (`Changed<T>` works).
    /// The `ReadOnlyWorldQuery` bound prevents `&mut T` access.
    ///
    /// Returns `Err(ReducerError::DuplicateName)` if the name is already registered.
    pub fn register_query_ref<Q, Args, F>(
        &mut self,
        world: &mut World,
        name: &'static str,
        f: F,
    ) -> Result<QueryReducerId, ReducerError>
    where
        Q: ReadOnlyWorldQuery + 'static,
        Args: Clone + 'static,
        F: Fn(QueryRef<'_, Q>, Args) + Send + Sync + 'static,
    {
        let resolved = ResolvedComponents(Vec::new());
        let access = Access::of::<Q>(world);

        let adapter: ScheduledAdapter = Box::new(move |world, args_any| {
            let args = args_any
                .downcast_ref::<Args>()
                .unwrap_or_else(|| {
                    panic!(
                        "reducer args type mismatch: expected {}",
                        std::any::type_name::<Args>()
                    )
                })
                .clone();
            let qr = QueryRef::<Q>::new(world);
            f(qr, args);
        });

        let id = self.push_entry(
            name,
            access,
            resolved,
            ReducerKind::Scheduled(adapter),
            None,
            None,
        )?;
        Ok(QueryReducerId(id.0))
    }

    // ── Built-in reducers ────────────────────────────────────────

    /// Register the built-in retention reducer that despawns expired entities.
    ///
    /// Each dispatch decrements every [`Expiry`](crate::Expiry) counter by one.
    /// Entities whose counter reaches zero are batch-despawned. The user
    /// controls how often retention runs — each call to `run()` is one
    /// "retention cycle."
    ///
    /// # Panics
    /// Panics if a reducer named `"__retention"` is already registered.
    pub fn retention(&mut self, world: &mut World) -> QueryReducerId {
        let expiry_id = world.register_component::<crate::retention::Expiry>();
        let mut access = Access::empty();
        access.add_write(expiry_id);
        access.set_despawns();

        let resolved = ResolvedComponents(Vec::new());

        let adapter: ScheduledAdapter = Box::new(|world, _args_any| {
            // Decrement all Expiry counters and collect entities that hit zero.
            let mut expired: Vec<Entity> = Vec::new();
            world
                .query::<(Entity, &mut crate::retention::Expiry)>()
                .for_each(|(entity, expiry)| {
                    expiry.tick();
                    if expiry.is_expired() {
                        expired.push(entity);
                    }
                });
            if !expired.is_empty() {
                world.despawn_batch(&expired);
            }
        });

        // Double-underscore prefix marks engine-built-in reducers.
        // push_entry rejects duplicate names, preventing user reducers
        // from colliding with built-ins.
        let id = self
            .push_entry(
                "__retention",
                access,
                resolved,
                ReducerKind::Scheduled(adapter),
                None,
                None,
            )
            .expect("__retention reducer name conflict");
        QueryReducerId(id.0)
    }

    // ── Dynamic registration ────────────────────────────────────

    /// Start building a dynamic reducer. Returns a builder that lets you
    /// declare which components the closure may read, write, or spawn.
    pub fn dynamic<'a>(
        &'a mut self,
        name: &'static str,
        world: &'a mut World,
    ) -> DynamicReducerBuilder<'a> {
        DynamicReducerBuilder {
            registry: self,
            world,
            name,
            access: Access::empty(),
            entries: Vec::new(),
            spawn_bundles: std::collections::HashSet::new(),
            remove_ids: std::collections::HashSet::new(),
        }
    }

    // ── Dispatch ─────────────────────────────────────────────────

    /// Call a transactional reducer (entity, spawner, or query writer).
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds,
    /// `Err(ReducerError::WrongKind)` if the ID points to a scheduled
    /// reducer, or `Err(ReducerError::TransactionConflict)` if the
    /// transaction strategy detects a conflict.
    pub fn call<S: Transact, Args: Clone + 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: ReducerId,
        args: Args,
    ) -> Result<(), ReducerError> {
        let entry = self.get_entry(id.0)?;
        let adapter = match &entry.kind {
            ReducerKind::Transactional(f) => f,
            ReducerKind::Scheduled(_) => {
                return Err(ReducerError::WrongKind {
                    expected: "transactional",
                    actual: "scheduled",
                });
            }
        };
        let access = &entry.access;
        let resolved = &entry.resolved;

        let tick_arc = entry.last_read_tick.clone();
        let queried_flag = entry.queried.clone();
        // Reset the queried flag before each call so we only advance the
        // tick if for_each/count actually runs during this invocation.
        if let Some(q) = &queried_flag {
            q.store(false, Ordering::Relaxed);
        }
        let result = strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let mut tw = TransactionalWorld(world);
            adapter(changeset, allocated, &mut tw, resolved, &args);
        });
        // Update last_read_tick AFTER the changeset is applied (by transact),
        // but only if for_each/count was actually called during this invocation.
        if result.is_ok()
            && let Some(arc) = &tick_arc
            && queried_flag
                .as_ref()
                .is_none_or(|q| q.load(Ordering::Relaxed))
        {
            let new_tick = world.next_tick();
            arc.store(new_tick.raw(), Ordering::Relaxed);
        }
        result.map_err(ReducerError::from)
    }

    /// Run a scheduled query reducer directly. Caller guarantees exclusivity.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds,
    /// or `Err(ReducerError::WrongKind)` if the ID points to a
    /// transactional reducer.
    pub fn run<Args: Clone + 'static>(
        &self,
        world: &mut World,
        id: QueryReducerId,
        args: Args,
    ) -> Result<(), ReducerError> {
        let entry = self.get_entry(id.0)?;
        match &entry.kind {
            ReducerKind::Scheduled(f) => {
                f(world, &args);
                Ok(())
            }
            ReducerKind::Transactional(_) => Err(ReducerError::WrongKind {
                expected: "scheduled",
                actual: "transactional",
            }),
        }
    }

    /// Look up a transactional reducer by name. Returns `None` if the name
    /// is not registered, points to a scheduled reducer, or is a dynamic reducer.
    pub fn reducer_id_by_name(&self, name: &str) -> Option<ReducerId> {
        let &slot = self.by_name.get(name)?;
        match slot {
            ReducerSlot::Unified(idx) => match &self.reducers[idx].kind {
                ReducerKind::Transactional(_) => Some(ReducerId(idx)),
                ReducerKind::Scheduled(_) => None,
            },
            ReducerSlot::Dynamic(_) => None,
        }
    }

    /// Look up a scheduled query reducer by name. Returns `None` if the name
    /// is not registered, points to a transactional reducer, or is a dynamic reducer.
    pub fn query_reducer_id_by_name(&self, name: &str) -> Option<QueryReducerId> {
        let &slot = self.by_name.get(name)?;
        match slot {
            ReducerSlot::Unified(idx) => match &self.reducers[idx].kind {
                ReducerKind::Scheduled(_) => Some(QueryReducerId(idx)),
                ReducerKind::Transactional(_) => None,
            },
            ReducerSlot::Dynamic(_) => None,
        }
    }

    /// Access metadata for a transactional reducer.
    ///
    /// # Panics
    /// Panics if the ID is out of bounds. Use [`reducer_info`](Self::reducer_info)
    /// for a fallible alternative.
    pub fn reducer_access(&self, id: ReducerId) -> &Access {
        &self.reducers[id.0].access
    }

    /// Access metadata for a scheduled query reducer.
    ///
    /// # Panics
    /// Panics if the ID is out of bounds.
    pub fn query_reducer_access(&self, id: QueryReducerId) -> &Access {
        &self.reducers[id.0].access
    }

    /// Access metadata by raw index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn access(&self, idx: usize) -> &Access {
        &self.reducers[idx].access
    }

    // ── Dynamic dispatch ────────────────────────────────────────

    /// Call a dynamic reducer with a chosen transaction strategy.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds,
    /// or `Err(ReducerError::TransactionConflict)` if the transaction
    /// strategy detects a conflict.
    pub fn dynamic_call<S: Transact, Args: 'static>(
        &self,
        strategy: &S,
        world: &mut World,
        id: DynamicReducerId,
        args: &Args,
    ) -> Result<(), ReducerError> {
        let entry = self.get_dynamic_entry(id.0)?;
        let closure = &entry.closure;
        let resolved = &entry.resolved;
        let access = resolved.access();
        let tick_arc = entry.last_read_tick.clone();
        let queried = Arc::new(AtomicBool::new(false));

        let result = strategy.transact(world, access, |tx, world| {
            let (changeset, allocated) = tx.reducer_parts();
            let world_ref: &World = world;
            let mut ctx = DynamicCtx::new(
                world_ref, changeset, allocated, resolved, &tick_arc, &queried,
            );
            closure(&mut ctx, args);
        });
        // Update last_read_tick AFTER the changeset is applied (by transact),
        // but only if for_each was actually called during this invocation.
        if result.is_ok() && queried.load(Ordering::Relaxed) {
            let new_tick = world.next_tick();
            entry
                .last_read_tick
                .store(new_tick.raw(), Ordering::Relaxed);
        }
        result.map_err(ReducerError::from)
    }

    /// Look up a dynamic reducer by name.
    pub fn dynamic_id_by_name(&self, name: &str) -> Option<DynamicReducerId> {
        let &slot = self.by_name.get(name)?;
        match slot {
            ReducerSlot::Dynamic(idx) => Some(DynamicReducerId(idx)),
            ReducerSlot::Unified(_) => None,
        }
    }

    /// Access metadata for a dynamic reducer.
    ///
    /// # Panics
    /// Panics if the ID is out of bounds.
    pub fn dynamic_access(&self, id: DynamicReducerId) -> &Access {
        self.dynamic_reducers[id.0].resolved.access()
    }

    // ── Introspection ─────────────────────────────────────────────

    /// Introspection for a transactional reducer.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds.
    pub fn reducer_info(&self, id: ReducerId) -> Result<ReducerInfo, ReducerError> {
        let entry = self.get_entry(id.0)?;
        let kind = match &entry.kind {
            ReducerKind::Transactional(_) => "transactional",
            ReducerKind::Scheduled(_) => "scheduled",
        };
        Ok(ReducerInfo {
            name: entry.name,
            kind,
            access: entry.access.clone(),
            has_change_tracking: entry.last_read_tick.is_some(),
            can_despawn: entry.access.despawns(),
        })
    }

    /// Introspection for a scheduled query reducer.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds.
    pub fn query_reducer_info(&self, id: QueryReducerId) -> Result<ReducerInfo, ReducerError> {
        self.reducer_info(ReducerId(id.0))
    }

    /// Introspection for a dynamic reducer.
    ///
    /// Returns `Err(ReducerError::InvalidId)` if the ID is out of bounds.
    pub fn dynamic_reducer_info(&self, id: DynamicReducerId) -> Result<ReducerInfo, ReducerError> {
        let entry = self.get_dynamic_entry(id.0)?;
        let access = entry.resolved.access().clone();
        let can_despawn = access.despawns();
        Ok(ReducerInfo {
            name: entry.name,
            kind: "dynamic",
            access,
            has_change_tracking: true, // dynamic reducers always have tick tracking
            can_despawn,
        })
    }

    /// Number of registered unified reducers (transactional + scheduled).
    pub fn reducer_count(&self) -> usize {
        self.reducers.len()
    }

    /// Number of registered dynamic reducers.
    pub fn dynamic_reducer_count(&self) -> usize {
        self.dynamic_reducers.len()
    }

    /// Iterate all registered reducer names and their slot kinds.
    pub fn registered_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_name.keys().copied()
    }

    // ── Internal ─────────────────────────────────────────────────

    fn get_entry(&self, index: usize) -> Result<&ReducerEntry, ReducerError> {
        self.reducers.get(index).ok_or(ReducerError::InvalidId {
            kind: "reducer",
            index,
            max: self.reducers.len(),
        })
    }

    fn get_dynamic_entry(&self, index: usize) -> Result<&DynamicReducerEntry, ReducerError> {
        self.dynamic_reducers
            .get(index)
            .ok_or(ReducerError::InvalidId {
                kind: "dynamic",
                index,
                max: self.dynamic_reducers.len(),
            })
    }

    fn push_entry(
        &mut self,
        name: &'static str,
        access: Access,
        resolved: ResolvedComponents,
        kind: ReducerKind,
        last_read_tick: Option<Arc<AtomicU64>>,
        queried: Option<Arc<AtomicBool>>,
    ) -> Result<ReducerId, ReducerError> {
        let id = self.reducers.len();
        if let Some(slot) = self.by_name.get(name) {
            let (existing_kind, existing_index) = match slot {
                ReducerSlot::Unified(idx) => ("unified", *idx),
                ReducerSlot::Dynamic(idx) => ("dynamic", *idx),
            };
            return Err(ReducerError::DuplicateName {
                name,
                existing_kind,
                existing_index,
            });
        }
        self.by_name.insert(name, ReducerSlot::Unified(id));
        self.reducers.push(ReducerEntry {
            name,
            access,
            resolved,
            kind,
            last_read_tick,
            queried,
        });
        Ok(ReducerId(id))
    }
}

impl Default for ReducerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
