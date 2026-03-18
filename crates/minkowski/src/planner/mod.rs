//! Query planner for composing index-driven lookups, joins, and full scans
//! into optimized execution plans.
//!
//! The planner is designed for an in-memory ECS where data already lives in L1/L2
//! cache. Planning overhead is kept to O(indexes + predicates). Plans are
//! executable against live world data: scan-only plans without a spatial driver
//! use zero-alloc `execute_stream` with fused filter closures; join plans use a
//! scratch-buffer intersection model.
//!
//! All plans execute via chunked, slice-based iteration over 64-byte-aligned
//! columns. LLVM auto-vectorizes loops over these contiguous slices — there
//! is no separate "scalar" execution path. Cost estimates reflect batch
//! amortization, branchless filter eligibility, and cache-partitioned joins.
//!
//! # Plan Nodes
//!
//! - **Scan** (ChunkedScan): one batch per archetype, cache-line aligned
//! - **IndexLookup** (IndexGather): batch entity fetch via BTree/Hash index
//! - **SpatialLookup** (SpatialGather): spatial index query (within, intersects)
//! - **Filter**: predicate applied to contiguous slices (branchless when possible)
//! - **HashJoin** (PartitionedHashJoin): build side partitioned for L2 residency
//! - **NestedLoopJoin** (BatchNestedLoopJoin): fallback for small cardinalities
//!
//! ```rust,ignore
//! let plan = planner
//!     .scan::<(&Pos, &Vel)>()
//!     .filter(Predicate::range::<Score>(Score(10)..Score(50)))
//!     .build();
//!
//! println!("{}", plan.explain());
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use minkowski::planner::{QueryPlanner, Predicate, JoinKind};
//!
//! let mut planner = QueryPlanner::new(&world);
//!
//! // Register available indexes (Arc-wrapped for live reads at execution time)
//! planner.add_btree_index::<Score>(&score_index, &world);
//! planner.add_hash_index::<Team>(&team_index, &world);
//!
//! // Build a plan
//! let plan = planner
//!     .scan::<(&Pos, &Vel, &Score)>()
//!     .filter(Predicate::range::<Score>(Score(10)..Score(50)))
//!     .join::<(&Team, &Name)>(JoinKind::Inner)
//!     .build();
//!
//! // Inspect the plan
//! println!("{}", plan.explain());
//!
//! // Check warnings (missing indexes, etc.)
//! for warning in plan.warnings() {
//!     eprintln!("WARN: {}", warning);
//! }
//! ```
//!
//! # Subscription Queries
//!
//! [`SubscriptionBuilder`] requires that every predicate is backed by an index,
//! enforced at compile time via the `Indexed<T>` witness type. This ensures
//! the database can push updates without scanning entire tables. The result is
//! a [`QueryPlanResult`] with full execution support.
//!
//! ```rust,ignore
//! let sub = planner
//!     .subscribe::<(Changed<Score>, &Pos, &Score)>()
//!     .where_eq(Indexed::btree(&score_index), Predicate::eq(Score(42)))
//!     .build()
//!     .unwrap();
//! ```

mod aggregate;
mod builder;
mod cost;
mod error;
mod exec;
mod node;
mod predicate;
mod scratch;
mod table;
mod traits;

#[cfg(test)]
mod tests;

// ── Re-exports ──────────────────────────────────────────────────────

pub use aggregate::{AggregateExpr, AggregateOp, AggregateResult};
pub use builder::{Indexed, ScanBuilder, SubscriptionBuilder};
pub use cost::{CardinalityConstraint, Cost};
pub use error::{PlanExecError, PlannerError, SubscriptionError};
pub use exec::QueryPlanResult;
pub use node::{JoinKind, PlanNode, PlanWarning};
pub use predicate::{IndexKind, Predicate, SpatialPredicate};
pub use table::TablePlanner;
pub use traits::AsEntityRef;

// Internal re-exports for tests
#[cfg(test)]
pub(crate) use cost::VectorizeOpts;

// ── Type aliases (used across submodules) ────────────────────────────

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::Arc;

use fixedbitset::FixedBitSet;

use crate::component::{Component, ComponentRegistry};
use crate::entity::Entity;
use crate::index::{BTreeIndex, HashIndex, SpatialCost, SpatialExpr, SpatialIndex};
use crate::storage::archetype::Archetype;
use crate::tick::Tick;
use crate::world::{World, WorldId};

use aggregate::{AggregateAccum, BatchExtractor};
use predicate::{IndexDescriptor, PredicateKind, PredicateLookupFn};
use scratch::ScratchBuffer;

// ── Execution engine ─────────────────────────────────────────────────

/// Type-erased scan execution captured at `build()` time.
/// The monomorphic `Q` iteration code is baked in at compile time.
/// Takes `&World` (shared ref) because the compiled scan only reads archetype
/// data. The outer `execute_stream` method takes `&mut World` for tick advancement
/// (via `next_tick()`), then reborrows as `&World` for the closure.
pub(super) type CompiledForEach = Box<dyn FnMut(&World, Tick, &mut dyn FnMut(Entity))>;

/// Read-only variant for transactional reads via `query_raw`.
/// Receives the plan's `last_read_tick` for `Changed<T>` filtering but does
/// not advance it — repeated calls see the same change window.
pub(super) type CompiledForEachRaw = Box<dyn FnMut(&World, Tick, &mut dyn FnMut(Entity))>;

/// Compiled aggregate scan: iterates matching archetypes/index results and
/// calls batch extractors directly. Bypasses per-entity callbacks entirely.
pub(super) type CompiledAggScan =
    Box<dyn FnMut(&World, Tick, &mut [Option<Box<dyn BatchExtractor>>], &mut [AggregateAccum])>;

/// Type-erased index lookup function: return matching entities from the index.
/// Returns `Arc<[Entity]>` to avoid cloning the entity list on every call —
/// callers that need mutation (filter, sort) pay the allocation cost only when
/// they actually mutate.
pub(super) type IndexLookupFn = Arc<dyn Fn() -> Arc<[Entity]> + Send + Sync>;

/// Type-erased filter function: given `&World` and `Entity`, return true if
/// the entity passes the predicate.
// PERF: Arc<dyn Fn> prevents SIMD vectorization of filter loops (per-entity
// vtable call). Inherent to type-erased plan composition — monomorphic
// filters would require codegen per plan.
pub(super) type FilterFn = Arc<dyn Fn(&World, Entity) -> bool + Send + Sync>;

/// Column-aware filter function: given a world reference, archetype, and the
/// archetype's entity list, AND the result into a pre-initialized boolean
/// mask. The mask is `arch.len()` elements long and pre-filled with `true`.
/// The filter sets `mask[i] = false` for entities that do not pass.
///
/// For dense components, operates on contiguous typed column slices —
/// sequential memory access, one `ComponentId` resolution per archetype.
/// For sparse components, falls back to per-entity `World::get()`.
pub(super) type ColumnFilterFn =
    Arc<dyn Fn(&World, &Archetype, &[Entity], &mut [bool]) + Send + Sync>;

/// Type-erased closure that collects entities from a scan/index into a
/// [`ScratchBuffer`]. Used by join plans to gather left and right entity sets.
pub(super) type EntityCollector = Box<dyn FnMut(&World, Tick, &mut ScratchBuffer)>;

/// Type-erased entity-reference extractor: given `&World` and `Entity`, reads
/// the entity-reference component and returns the referenced `Entity`.
/// Returns `None` if the source entity is dead or doesn't have the component.
pub(super) type EntityRefExtractor = Arc<dyn Fn(&World, Entity) -> Option<Entity> + Send + Sync>;

/// Returns true if every component in `changed` has a column in the archetype
/// whose tick is newer than `tick`. When `changed` is empty (no `Changed<T>`
/// terms), returns true immediately. For archetype-scan paths the column will
/// always be present because `required` is checked first. For index-gather
/// paths, the column may be absent if the entity's archetype does not contain
/// the component; `is_some_and` handles this by returning false.
#[inline]
pub(super) fn passes_change_filter(
    arch: &crate::storage::archetype::Archetype,
    changed: &FixedBitSet,
    tick: Tick,
) -> bool {
    changed.is_clear()
        || changed.ones().all(|bit| {
            arch.column_index(bit)
                .is_some_and(|col| arch.columns[col].changed_tick.is_newer_than(tick))
        })
}

/// Descriptor for a registered spatial index.
pub(super) struct SpatialIndexDescriptor {
    pub(super) component_name: &'static str,
    /// The spatial index, behind Arc for shared access at execution time.
    pub(super) index: Arc<dyn SpatialIndex + Send + Sync>,
}

impl fmt::Debug for SpatialIndexDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialIndexDescriptor")
            .field("component_name", &self.component_name)
            .finish_non_exhaustive()
    }
}

/// Result of checking whether a spatial index can accelerate a predicate.
pub(super) enum SpatialLookupResult {
    /// The index can accelerate the expression at the given cost.
    Accelerated(
        &'static str,
        SpatialCost,
        Arc<dyn SpatialIndex + Send + Sync>,
    ),
    /// A spatial index is registered but it declined the expression.
    Declined(String),
    /// No spatial index is registered for this component type.
    NoIndex,
}

/// Compiled push-based query planner that composes index lookups, filters, and
/// joins into cost-optimized execution plans.
///
/// The planner operates on metadata only — it never touches actual component
/// data during planning. Registering indexes captures type-erased lookup
/// closures. Plan compilation is O(predicates × indexes).
///
/// # Index Selection
///
/// When a predicate targets a component with a registered index:
/// - **BTreeIndex**: used for both equality and range predicates
/// - **HashIndex**: used for equality predicates only
///
/// If no index is available, the planner falls back to a full scan + filter
/// and emits a [`PlanWarning::MissingIndex`].
///
/// # Join Optimization
///
/// The planner selects between hash join and nested-loop join based on
/// estimated cardinalities:
/// - **Hash join** when either side has ≥ 64 rows (for inner joins, the
///   smaller side becomes the build table for cache locality; for left
///   joins, the semantic left side is always preserved as the build side)
/// - **Nested-loop join** for very small cardinalities
///
/// Join order is determined by intermediate result size — the smallest
/// intermediate result drives subsequent joins.
pub struct QueryPlanner<'w> {
    pub(super) indexes: HashMap<TypeId, IndexDescriptor>,
    pub(super) spatial_indexes: HashMap<TypeId, SpatialIndexDescriptor>,
    pub(super) total_entities: usize,
    /// Component registry for resolving query type → component bitset.
    pub(super) components: &'w ComponentRegistry,
    /// Per-component entity counts: `component_entity_counts[comp_id]` holds the
    /// total number of entities across non-empty archetypes that contain that
    /// component.  Built once in `new()` so that `estimate_matching_entities()`
    /// can answer in O(|required|) instead of scanning every archetype.
    pub(super) component_entity_counts: Vec<usize>,
    pub(super) world_id: WorldId,
    pub(super) _world: PhantomData<&'w World>,
}

impl<'w> QueryPlanner<'w> {
    /// Create a new planner from the current world state.
    ///
    /// Captures the total entity count for cost estimation. The world is not
    /// borrowed beyond this call.
    pub fn new(world: &'w World) -> Self {
        // Single archetype walk to build per-component entity counts.
        // This turns every subsequent estimate_matching_entities() call from
        // O(archetypes) into O(|required_components|).
        let mut component_entity_counts = Vec::new();
        for arch in &world.archetypes.archetypes {
            if arch.is_empty() {
                continue;
            }
            let n = arch.entities.len();
            for comp_id in arch.component_ids.ones() {
                if comp_id >= component_entity_counts.len() {
                    component_entity_counts.resize(comp_id + 1, 0);
                }
                component_entity_counts[comp_id] += n;
            }
        }

        QueryPlanner {
            indexes: HashMap::new(),
            spatial_indexes: HashMap::new(),
            total_entities: world.entity_count(),
            components: &world.components,
            component_entity_counts,
            world_id: world.world_id(),
            _world: PhantomData,
        }
    }

    /// Estimate entities matching a required component bitset.
    ///
    /// Uses the precomputed per-component entity counts to answer in
    /// O(|required|) — the minimum entity count across required components.
    /// This is an upper bound on the true count (intersection ≤ smallest
    /// individual set), which is acceptable for cost-based plan selection.
    pub(super) fn estimate_matching_entities(&self, required: &FixedBitSet) -> usize {
        let mut min_count = usize::MAX;
        for comp_id in required.ones() {
            let count = self
                .component_entity_counts
                .get(comp_id)
                .copied()
                .unwrap_or(0);
            min_count = min_count.min(count);
        }
        // Empty required set → all entities match.
        if min_count == usize::MAX {
            self.total_entities
        } else {
            min_count
        }
    }

    /// Register a `BTreeIndex` for cost-based index selection.
    ///
    /// Returns `Err(PlannerError::UnregisteredComponent)` if `T` has not been
    /// registered as a component in `world`.
    pub fn add_btree_index<T: Component + Ord + Clone>(
        &mut self,
        index: &Arc<BTreeIndex<T>>,
        world: &World,
    ) -> Result<(), PlannerError> {
        if world.component_id::<T>().is_none() {
            return Err(PlannerError::UnregisteredComponent(
                std::any::type_name::<T>(),
            ));
        }
        let index = index.clone();

        // all_entities_fn: iterate live index at execution time (not a snapshot).
        let idx_all = Arc::clone(&index);
        let all_fn: IndexLookupFn = Arc::new(move || {
            let (tree, _, _) = idx_all.as_raw_parts();
            tree.values()
                .flat_map(|ents| ents.iter().copied())
                .collect()
        });

        // eq_lookup_fn: O(log n) point lookup on live index.
        let idx_eq = Arc::clone(&index);
        let eq_fn: PredicateLookupFn = Arc::new(move |value: &dyn std::any::Any| {
            let key = value
                .downcast_ref::<T>()
                .expect("eq_lookup_fn: type mismatch in downcast");
            idx_eq.get(key).iter().copied().collect()
        });

        // range_lookup_fn: O(log n + k) range scan on live index.
        let idx_range = Arc::clone(&index);
        let range_fn: PredicateLookupFn = Arc::new(move |value: &dyn std::any::Any| {
            let (lo, hi) = value
                .downcast_ref::<(Bound<T>, Bound<T>)>()
                .expect("range_lookup_fn: type mismatch in downcast");
            idx_range
                .range((lo.clone(), hi.clone()))
                .flat_map(|(_, ents)| ents.iter().copied())
                .collect()
        });

        self.indexes.insert(
            TypeId::of::<T>(),
            IndexDescriptor {
                component_name: std::any::type_name::<T>(),
                kind: IndexKind::BTree,
                all_entities_fn: Some(all_fn),
                eq_lookup_fn: Some(eq_fn),
                range_lookup_fn: Some(range_fn),
            },
        );
        Ok(())
    }

    /// Register a `HashIndex` for cost-based index selection.
    ///
    /// Returns `Err(PlannerError::UnregisteredComponent)` if `T` has not been
    /// registered as a component in `world`.
    pub fn add_hash_index<T: Component + std::hash::Hash + Eq + Clone>(
        &mut self,
        index: &Arc<HashIndex<T>>,
        world: &World,
    ) -> Result<(), PlannerError> {
        if world.component_id::<T>().is_none() {
            return Err(PlannerError::UnregisteredComponent(
                std::any::type_name::<T>(),
            ));
        }
        let index = index.clone();

        // all_entities_fn: iterate live index at execution time (not a snapshot).
        let idx_all = Arc::clone(&index);
        let all_fn: IndexLookupFn = Arc::new(move || {
            let (map, _, _) = idx_all.as_raw_parts();
            map.values().flat_map(|ents| ents.iter().copied()).collect()
        });

        // eq_lookup_fn: O(1) point lookup on live index.
        let idx_eq = Arc::clone(&index);
        let eq_fn: PredicateLookupFn = Arc::new(move |value: &dyn std::any::Any| {
            let key = value
                .downcast_ref::<T>()
                .expect("eq_lookup_fn: type mismatch in downcast");
            idx_eq.get(key).iter().copied().collect()
        });

        // Only insert if no BTree index is already registered for this type
        // (BTree is strictly more capable than Hash).
        self.indexes
            .entry(TypeId::of::<T>())
            .or_insert(IndexDescriptor {
                component_name: std::any::type_name::<T>(),
                kind: IndexKind::Hash,
                all_entities_fn: Some(all_fn),
                eq_lookup_fn: Some(eq_fn),
                range_lookup_fn: None, // Hash doesn't support range queries.
            });
        Ok(())
    }

    /// Register a spatial index for a component type.
    ///
    /// The planner will call [`SpatialIndex::supports`] to check whether the
    /// index can accelerate spatial predicates on `T`. Multiple spatial indexes
    /// can be registered for different component types, but only one per type.
    ///
    /// Returns `Err(PlannerError::UnregisteredComponent)` if `T` has not been
    /// registered as a component in `world`.
    pub fn add_spatial_index<T: Component>(
        &mut self,
        index: Arc<dyn SpatialIndex + Send + Sync>,
        world: &World,
    ) -> Result<(), PlannerError> {
        if world.component_id::<T>().is_none() {
            return Err(PlannerError::UnregisteredComponent(
                std::any::type_name::<T>(),
            ));
        }
        self.spatial_indexes.insert(
            TypeId::of::<T>(),
            SpatialIndexDescriptor {
                component_name: std::any::type_name::<T>(),
                index,
            },
        );
        Ok(())
    }

    /// Start building a scan plan for query type `Q`.
    pub fn scan<Q: crate::query::fetch::WorldQuery + 'static>(&'w self) -> ScanBuilder<'w> {
        let required = Q::required_ids(self.components);
        let changed = Q::changed_ids(self.components);
        let required_for_spatial = required.clone();
        let changed_for_spatial = changed.clone();
        let required_for_each = required.clone();
        let changed_for_each = changed.clone();
        let required_for_each_raw = required.clone();
        let changed_for_each_raw = changed.clone();
        let left_required = required.clone();
        let left_changed = changed.clone();
        // Estimate entities matching the scan's required components.
        let estimated_rows = self.estimate_matching_entities(&required);
        ScanBuilder {
            planner: self,
            world_id: self.world_id,
            query_name: std::any::type_name::<Q>(),
            estimated_rows,
            predicates: Vec::new(),
            joins: Vec::new(),
            er_joins: Vec::new(),
            last_join: None,
            deferred_warnings: Vec::new(),
            compile_for_each: Some(Box::new(move || {
                let required = required_for_each;
                let changed = changed_for_each;
                Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        for arch in &world.archetypes.archetypes {
                            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                                continue;
                            }
                            if !passes_change_filter(arch, &changed, tick) {
                                continue;
                            }
                            for &entity in &arch.entities {
                                callback(entity);
                            }
                        }
                    },
                )
            })),
            compile_for_each_raw: Some(Box::new(move || {
                let required = required_for_each_raw;
                let changed = changed_for_each_raw;
                Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        for arch in &world.archetypes.archetypes {
                            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                                continue;
                            }
                            if !passes_change_filter(arch, &changed, tick) {
                                continue;
                            }
                            for &entity in &arch.entities {
                                callback(entity);
                            }
                        }
                    },
                )
            })),
            left_required: Some(left_required),
            left_changed: Some(left_changed),
            required_for_spatial: Some(required_for_spatial),
            changed_for_spatial: Some(changed_for_spatial),
            aggregates: Vec::new(),
        }
    }

    /// Start building a scan plan with an explicit row estimate.
    ///
    /// Use this when you know the approximate result size from domain
    /// knowledge (e.g., "there are ~500 active players").
    pub fn scan_with_estimate<Q: crate::query::fetch::WorldQuery + 'static>(
        &'w self,
        estimated_rows: usize,
    ) -> ScanBuilder<'w> {
        let required = Q::required_ids(self.components);
        let changed = Q::changed_ids(self.components);
        let required_for_spatial = required.clone();
        let changed_for_spatial = changed.clone();
        let required_for_each = required.clone();
        let changed_for_each = changed.clone();
        let required_for_each_raw = required.clone();
        let changed_for_each_raw = changed.clone();
        let left_required = required.clone();
        let left_changed = changed.clone();
        ScanBuilder {
            planner: self,
            world_id: self.world_id,
            query_name: std::any::type_name::<Q>(),
            estimated_rows,
            predicates: Vec::new(),
            joins: Vec::new(),
            er_joins: Vec::new(),
            last_join: None,
            deferred_warnings: Vec::new(),
            compile_for_each: Some(Box::new(move || {
                let required = required_for_each;
                let changed = changed_for_each;
                Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        for arch in &world.archetypes.archetypes {
                            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                                continue;
                            }
                            if !passes_change_filter(arch, &changed, tick) {
                                continue;
                            }
                            for &entity in &arch.entities {
                                callback(entity);
                            }
                        }
                    },
                )
            })),
            compile_for_each_raw: Some(Box::new(move || {
                let required = required_for_each_raw;
                let changed = changed_for_each_raw;
                Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        for arch in &world.archetypes.archetypes {
                            if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                                continue;
                            }
                            if !passes_change_filter(arch, &changed, tick) {
                                continue;
                            }
                            for &entity in &arch.entities {
                                callback(entity);
                            }
                        }
                    },
                )
            })),
            left_required: Some(left_required),
            left_changed: Some(left_changed),
            required_for_spatial: Some(required_for_spatial),
            changed_for_spatial: Some(changed_for_spatial),
            aggregates: Vec::new(),
        }
    }

    /// Start building a subscription plan (compiler-enforced indexes).
    pub fn subscribe<Q: crate::query::fetch::WorldQuery + 'static>(
        &'w self,
    ) -> SubscriptionBuilder<'w> {
        let changed = Q::changed_ids(self.components);
        let has_changed_filter = !changed.is_clear();
        SubscriptionBuilder {
            scan: self.scan::<Q>(),
            errors: Vec::new(),
            has_predicates: false,
            attempted_predicates: false,
            has_changed_filter,
        }
    }

    /// Find the best index for a predicate, if one exists.
    pub(super) fn find_best_index(&self, pred: &Predicate) -> Option<&IndexDescriptor> {
        let idx = self.indexes.get(&pred.component_type)?;

        match idx.kind {
            IndexKind::BTree if pred.can_use_btree() => Some(idx),
            IndexKind::Hash if pred.can_use_hash() => Some(idx),
            _ => None,
        }
    }

    /// Check if a spatial predicate can be accelerated by a registered spatial index.
    ///
    /// Returns a three-state result distinguishing "accelerated", "index
    /// exists but declined", and "no index registered".
    pub(super) fn find_spatial_index(&self, pred: &Predicate) -> SpatialLookupResult {
        let PredicateKind::Spatial(sp) = &pred.kind else {
            return SpatialLookupResult::NoIndex;
        };
        let Some(desc) = self.spatial_indexes.get(&pred.component_type) else {
            return SpatialLookupResult::NoIndex;
        };
        let expr: SpatialExpr = sp.into();
        match desc.index.supports(&expr) {
            Some(cost) => {
                SpatialLookupResult::Accelerated(desc.component_name, cost, Arc::clone(&desc.index))
            }
            None => SpatialLookupResult::Declined(sp.to_string()),
        }
    }

    /// Generate warnings for predicates that can't use an index.
    pub(super) fn warn_missing_index(&self, pred: &Predicate, warnings: &mut Vec<PlanWarning>) {
        match &pred.kind {
            PredicateKind::Custom(_) | PredicateKind::CustomColumn(_) => {
                // Custom predicates never use indexes — no warning needed.
            }
            PredicateKind::Eq => {
                if self.indexes.contains_key(&pred.component_type) {
                    // We have an index but it wasn't usable — shouldn't happen
                    // for Eq since both BTree and Hash support it.
                    debug_assert!(
                        false,
                        "index for `{}` exists but could not serve Eq predicate",
                        pred.component_name
                    );
                } else {
                    warnings.push(PlanWarning::MissingIndex {
                        component_name: pred.component_name,
                        predicate_kind: "equality",
                        suggestion: "add a HashIndex<T> or BTreeIndex<T>",
                    });
                }
            }
            PredicateKind::Range => {
                if let Some(idx) = self.indexes.get(&pred.component_type) {
                    if idx.kind == IndexKind::Hash {
                        warnings.push(PlanWarning::IndexKindMismatch {
                            component_name: pred.component_name,
                            have: "Hash",
                            need: "BTree (for range queries)",
                        });
                    }
                } else {
                    warnings.push(PlanWarning::MissingIndex {
                        component_name: pred.component_name,
                        predicate_kind: "range",
                        suggestion: "add a BTreeIndex<T>",
                    });
                }
            }
            PredicateKind::Spatial(_) => {
                // Only called for NoIndex — Declined is handled by the caller
                // with a distinct SpatialIndexDeclined warning.
                warnings.push(PlanWarning::MissingIndex {
                    component_name: pred.component_name,
                    predicate_kind: "spatial",
                    suggestion: "add a SpatialIndex via add_spatial_index::<T>()",
                });
            }
        }
    }
}
