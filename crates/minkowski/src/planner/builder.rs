use std::any::TypeId;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use fixedbitset::FixedBitSet;

use crate::component::Component;
use crate::entity::Entity;
use crate::index::{BTreeIndex, HashIndex, SpatialCost, SpatialExpr, SpatialIndex};
use crate::tick::Tick;
use crate::world::{World, WorldId};

use super::aggregate::{AggregateAccum, AggregateExpr, AggregateOp, BatchExtractor};
use super::cost::{Cost, VectorizeOpts};
use super::error::{PlannerError, SubscriptionError};
use super::exec::QueryPlanResult;
use super::node::{JoinKind, PlanNode, PlanWarning};
use super::predicate::{IndexDescriptor, IndexKind, Predicate, PredicateKind};
use super::scratch::ScratchBuffer;
use super::traits::AsEntityRef;
use super::{
    ColumnFilterFn, CompiledAggScan, CompiledForEach, CompiledForEachRaw, EntityCollector,
    EntityRefExtractor, FilterFn, IndexLookupFn, QueryPlanner, SpatialLookupResult,
    passes_change_filter,
};

// ── Indexed<T> — compile-time proof that an index exists ─────────────

/// Compile-time witness that an index exists for component `T`.
///
/// Used by [`SubscriptionBuilder`] to enforce that every predicate in a
/// subscription query is backed by an index, ensuring the database can
/// push updates without scanning entire tables.
///
/// Cannot be constructed directly — only obtained by passing a reference
/// to a `BTreeIndex<T>` or `HashIndex<T>`.
pub struct Indexed<T: Component> {
    pub(super) kind: IndexKind,
    pub(super) cardinality: usize,
    pub(super) _marker: PhantomData<T>,
}

impl<T: Component + Ord + Clone> Indexed<T> {
    /// Create an index witness from a `BTreeIndex`.
    pub fn btree(index: &BTreeIndex<T>) -> Self {
        Indexed {
            kind: IndexKind::BTree,
            cardinality: index.len(),
            _marker: PhantomData,
        }
    }
}

impl<T: Component + std::hash::Hash + Eq + Clone> Indexed<T> {
    /// Create an index witness from a `HashIndex`.
    pub fn hash(index: &HashIndex<T>) -> Self {
        Indexed {
            kind: IndexKind::Hash,
            cardinality: index.len(),
            _marker: PhantomData,
        }
    }
}

impl<T: Component> Clone for Indexed<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Component> Copy for Indexed<T> {}

impl<T: Component> fmt::Debug for Indexed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Indexed")
            .field("type", &std::any::type_name::<T>())
            .field("kind", &self.kind)
            .field("cardinality", &self.cardinality)
            .finish()
    }
}

// ── Subscription builder ─────────────────────────────────────────────

/// Builder for subscription queries that enforces every predicate has an index.
///
/// Unlike `ScanBuilder`, this uses the type system to guarantee that the
/// resulting plan can push updates without full table scans. Every call to
/// `where_eq` or `where_range` requires an `Indexed<T>` witness, which can
/// only be obtained from an actual index instance.
///
/// Produces a [`QueryPlanResult`] with full execution support (`execute_collect`,
/// `execute_stream`, `execute_stream_raw`), backed by `IndexDriver` for index-gather
/// execution.
pub struct SubscriptionBuilder<'w> {
    pub(super) scan: ScanBuilder<'w>,
    pub(super) errors: Vec<SubscriptionError>,
    pub(super) has_predicates: bool,
    pub(super) attempted_predicates: bool,
    pub(super) has_changed_filter: bool,
}

impl SubscriptionBuilder<'_> {
    /// Add an equality predicate backed by a proven index.
    ///
    /// # Errors
    ///
    /// Returns [`SubscriptionError::ComponentMismatch`] (via [`build`](Self::build))
    /// if the predicate's component type does not match the witness type `T`.
    ///
    /// Returns [`SubscriptionError::PredicateKindMismatch`] (via [`build`](Self::build))
    /// if the predicate is not an equality predicate.
    pub fn where_eq<T: Component>(mut self, _witness: Indexed<T>, predicate: Predicate) -> Self {
        self.attempted_predicates = true;
        let witness_type = std::any::type_name::<T>();
        if predicate.component_type != TypeId::of::<T>() {
            self.errors.push(SubscriptionError::ComponentMismatch {
                witness_type,
                predicate_type: predicate.component_name,
            });
            return self;
        }
        if !matches!(predicate.kind, PredicateKind::Eq) {
            self.errors.push(SubscriptionError::PredicateKindMismatch {
                expected: "Eq",
                component_name: witness_type,
            });
            return self;
        }
        self.has_predicates = true;
        self.scan = self.scan.filter(predicate);
        self
    }

    /// Add a range predicate backed by a proven BTree index.
    ///
    /// # Errors
    ///
    /// Returns [`SubscriptionError::HashIndexOnRange`] (via [`build`](Self::build))
    /// if `witness` was created from a `HashIndex`. Hash indexes support only
    /// exact-match lookups — they cannot answer range queries. Use a
    /// `BTreeIndex` instead.
    ///
    /// Returns [`SubscriptionError::ComponentMismatch`] (via [`build`](Self::build))
    /// if the predicate's component type does not match the witness type `T`.
    ///
    /// Returns [`SubscriptionError::PredicateKindMismatch`] (via [`build`](Self::build))
    /// if the predicate is not a range predicate.
    pub fn where_range<T: Component + Ord + Clone>(
        mut self,
        witness: Indexed<T>,
        predicate: Predicate,
    ) -> Self {
        self.attempted_predicates = true;
        let witness_type = std::any::type_name::<T>();
        if witness.kind == IndexKind::Hash {
            self.errors.push(SubscriptionError::HashIndexOnRange {
                component_name: witness_type,
            });
            return self;
        }
        if predicate.component_type != TypeId::of::<T>() {
            self.errors.push(SubscriptionError::ComponentMismatch {
                witness_type,
                predicate_type: predicate.component_name,
            });
            return self;
        }
        if !matches!(predicate.kind, PredicateKind::Range) {
            self.errors.push(SubscriptionError::PredicateKindMismatch {
                expected: "Range",
                component_name: witness_type,
            });
            return self;
        }
        self.has_predicates = true;
        self.scan = self.scan.filter(predicate);
        self
    }

    /// Compile the subscription plan. Every predicate is guaranteed to have
    /// an index, so the plan never falls back to a full scan for filtering.
    ///
    /// # Errors
    ///
    /// Returns all [`SubscriptionError`]s if any predicates were invalid
    /// (e.g. a Hash index used with `where_range`, or a type mismatch).
    pub fn build(mut self) -> Result<QueryPlanResult, Vec<SubscriptionError>> {
        // Subscription queries require Changed<T> in the query type so that
        // unchanged archetypes are skipped. Without it the subscription
        // matches all entities every time — indistinguishable from a scan.
        if !self.has_changed_filter {
            self.errors.push(SubscriptionError::NoChangedFilter);
        }
        // Only report NoPredicates if the user never attempted to add any.
        // If they attempted but all failed validation, the specific errors
        // are already in self.errors — NoPredicates would be spurious.
        if !self.has_predicates && !self.attempted_predicates {
            self.errors.push(SubscriptionError::NoPredicates);
        }
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let result = self.scan.build();
        // Enforce the "all predicates indexed" guarantee. The Indexed<T>
        // witness proves an index object exists, but ScanBuilder only uses
        // indexes registered with the planner. If the user forgot to call
        // add_btree_index/add_hash_index, the plan silently degrades to a
        // full scan — catch this here and fail loudly.
        let index_errors: Vec<SubscriptionError> = result
            .warnings()
            .iter()
            .filter_map(|w| match w {
                PlanWarning::MissingIndex { component_name, .. } => {
                    Some(SubscriptionError::IndexNotRegistered { component_name })
                }
                _ => None,
            })
            .collect();
        if !index_errors.is_empty() {
            return Err(index_errors);
        }
        Ok(result)
    }
}

/// A single join step: collect right-side entities and intersect with the
/// accumulated left result.
pub(super) struct JoinStep {
    pub(super) right_collector: EntityCollector,
    pub(super) join_kind: JoinKind,
    /// Number of hash partitions for cache-aware join execution.
    /// When > 1, entities are bucketed by `Entity::to_bits() % partitions`
    /// so each partition fits in L2 cache during intersection.
    pub(super) partitions: usize,
}

/// A single ER join step: for each left entity, follow the entity reference
/// and check membership in the right set.
pub(super) struct ErJoinStep {
    /// Collects entities matching the right-side query (the "target" entities).
    pub(super) right_collector: EntityCollector,
    /// Extracts the referenced `Entity` from a left-side entity's component.
    pub(super) ref_extractor: EntityRefExtractor,
    pub(super) join_kind: JoinKind,
}

/// Execution state for join plans. The left collector populates the initial
/// entity set, then each `JoinStep` iteratively applies one join. Supports
/// arbitrary join chains: `A JOIN B JOIN C` becomes
/// `left_collector(A) → step[0](B) → step[1](C)`.
///
/// ER join steps execute after all regular join steps (the builder accepts
/// `join()` and `er_join()` in any order; `build()` sequences them). Each ER step builds
/// a `HashSet<Entity>` from the right side, then filters left-side entities
/// by probing via entity reference extraction. Left ER joins short-circuit
/// and skip right-side collection entirely.
pub(super) struct JoinExec {
    pub(super) left_collector: EntityCollector,
    pub(super) steps: Vec<JoinStep>,
    pub(super) er_steps: Vec<ErJoinStep>,
}

/// Index-gather validation pipeline: validate candidate entities from an index
/// lookup, filtering by liveness, required components, change detection, and
/// user predicates. Calls `emit` for each surviving entity.
///
/// Uses a 1-element archetype cache to amortise the `is_subset` and
/// `passes_change_filter` checks when consecutive candidates share the same
/// archetype — common in spatial indexes that return physically clustered
/// entities. Cache hit turns those per-entity bitset operations into a single
/// `ArchetypeId` integer comparison.
#[inline]
fn gather_index_candidates(
    world: &World,
    candidates: &[Entity],
    required: &FixedBitSet,
    changed: &FixedBitSet,
    tick: Tick,
    filters: &[FilterFn],
    mut emit: impl FnMut(Entity),
) {
    // Archetype cache: (archetype_id, passed_archetype_checks).
    // Avoids re-checking the same archetype when consecutive entities share one.
    // Safe to cache: &World guarantees no column ticks advance during this call.
    let mut cached_arch: Option<(usize, bool)> = None;
    let has_changed = !changed.is_clear();

    for &entity in candidates {
        let Some(loc) = world.validate_entity(entity) else {
            continue;
        };
        let arch_idx = loc.archetype_id.0;

        // Check archetype cache: if we've already validated this archetype,
        // skip the bitset subset and change-filter checks entirely.
        let arch_ok = match cached_arch {
            Some((cached_id, ok)) if cached_id == arch_idx => ok,
            _ => {
                let arch = &world.archetypes.archetypes[arch_idx];
                let ok = required.is_subset(&arch.component_ids)
                    && (!has_changed || passes_change_filter(arch, changed, tick));
                cached_arch = Some((arch_idx, ok));
                ok
            }
        };

        if !arch_ok {
            continue;
        }
        if filters.is_empty() || filters.iter().all(|f| f(world, entity)) {
            emit(entity);
        }
    }
}

/// Batched index-gather for aggregate execution. Groups validated candidates
/// by archetype and calls batch extractors per archetype run.
///
/// Uses the same 1-element archetype cache as `gather_index_candidates`.
/// When the archetype changes (or at the end), the accumulated row buffer
/// is flushed through `process_rows` on each extractor.
#[inline]
fn gather_index_batched(
    world: &World,
    candidates: &[Entity],
    required: &FixedBitSet,
    changed: &FixedBitSet,
    tick: Tick,
    filters: &[FilterFn],
    extractors: &mut [Option<Box<dyn BatchExtractor>>],
    accums: &mut [AggregateAccum],
) {
    let mut cached_arch: Option<(usize, bool)> = None;
    let has_changed = !changed.is_clear();
    // Row buffer for the current archetype run.
    let mut row_buf: Vec<usize> = Vec::new();
    let mut current_arch_idx: Option<usize> = None;
    // Per-extractor bind status: true if bind_archetype succeeded for the
    // current archetype. Prevents stale/null pointer dereference when the
    // aggregate component is absent from an archetype.
    let mut bound: Vec<bool> = vec![false; extractors.len()];

    for &entity in candidates {
        let Some(loc) = world.validate_entity(entity) else {
            continue;
        };
        let arch_idx = loc.archetype_id.0;

        let arch_ok = match cached_arch {
            Some((cached_id, ok)) if cached_id == arch_idx => ok,
            _ => {
                let arch = &world.archetypes.archetypes[arch_idx];
                let ok = required.is_subset(&arch.component_ids)
                    && (!has_changed || passes_change_filter(arch, changed, tick));
                cached_arch = Some((arch_idx, ok));
                ok
            }
        };

        if !arch_ok {
            continue;
        }
        if !filters.is_empty() && !filters.iter().all(|f| f(world, entity)) {
            continue;
        }

        // Archetype changed — flush accumulated rows, rebind extractors.
        if current_arch_idx != Some(arch_idx) {
            if !row_buf.is_empty() {
                flush_row_batch(extractors, accums, &bound, &row_buf);
                row_buf.clear();
            }
            current_arch_idx = Some(arch_idx);
            let arch = &world.archetypes.archetypes[arch_idx];
            for (i, (ext, accum)) in extractors.iter_mut().zip(accums.iter()).enumerate() {
                if accum.op != AggregateOp::Count {
                    bound[i] = ext.as_mut().is_some_and(|e| e.bind_archetype(arch));
                } else {
                    bound[i] = false; // Count doesn't need binding
                }
            }
        }
        row_buf.push(loc.row);
    }

    // Flush final batch.
    if !row_buf.is_empty() {
        flush_row_batch(extractors, accums, &bound, &row_buf);
    }
}

/// Flush a row buffer through batch extractors and accumulators.
/// Only processes extractors whose `bound` flag is true (component
/// present in the current archetype).
#[inline]
fn flush_row_batch(
    extractors: &mut [Option<Box<dyn BatchExtractor>>],
    accums: &mut [AggregateAccum],
    bound: &[bool],
    rows: &[usize],
) {
    for (i, (accum, ext)) in accums.iter_mut().zip(extractors.iter_mut()).enumerate() {
        if accum.op == AggregateOp::Count {
            accum.count += rows.len() as u64;
        } else if bound[i]
            && let Some(e) = ext
        {
            e.process_rows(rows, accum);
        }
    }
}

/// Collect all entities from archetypes matching a component bitset
/// into a scratch buffer, skipping archetypes whose changed columns
/// are not newer than `tick`.
fn collect_matching_entities(
    world: &World,
    required: &FixedBitSet,
    changed: &FixedBitSet,
    tick: Tick,
    scratch: &mut ScratchBuffer,
) {
    for arch in &world.archetypes.archetypes {
        if arch.is_empty() || !required.is_subset(&arch.component_ids) {
            continue;
        }
        if !passes_change_filter(arch, changed, tick) {
            continue;
        }
        for &entity in &arch.entities {
            scratch.push(entity);
        }
    }
}

// ── Scan builder ─────────────────────────────────────────────────────

/// Carries the spatial index and expression resolved during Phase 3
/// (driver selection) to Phase 7 (join collectors) and Phase 8 (closure compilation).
struct SpatialDriver {
    expr: SpatialExpr,
    index: Arc<dyn SpatialIndex + Send + Sync>,
}

/// Carries a pre-bound index lookup function from Phase 3 (driver selection)
/// to Phase 7 (join collectors) and Phase 8 (closure compilation).
///
/// The lookup function and predicate value are bound together at construction
/// time, so the execution path never sees `dyn Any`.
struct IndexDriver {
    lookup_fn: IndexLookupFn,
}

/// Tracks which kind of join was most recently added to the builder,
/// so `with_right_estimate` can target the correct one.
pub(super) enum LastJoinKind {
    Regular(usize),
    Er(usize),
}

/// Builder for a single-table scan with optional predicates and joins.
pub struct ScanBuilder<'w> {
    pub(super) planner: &'w QueryPlanner<'w>,
    pub(super) world_id: WorldId,
    pub(super) query_name: &'static str,
    pub(super) estimated_rows: usize,
    pub(super) predicates: Vec<Predicate>,
    pub(super) joins: Vec<JoinSpec>,
    pub(super) er_joins: Vec<ErJoinSpec>,
    /// Which join was most recently added (for `with_right_estimate` targeting).
    pub(super) last_join: Option<LastJoinKind>,
    /// Warnings collected during builder calls (e.g. unregistered ER component).
    pub(super) deferred_warnings: Vec<PlanWarning>,
    /// Factory that produces a [`CompiledForEach`] closure. Captured from
    /// `scan::<Q>()` while Q is still in scope.
    pub(super) compile_for_each: Option<Box<dyn FnOnce() -> CompiledForEach>>,
    /// Factory that produces a [`CompiledForEachRaw`] closure for read-only
    /// transactional access via `&World`.
    pub(super) compile_for_each_raw: Option<Box<dyn FnOnce() -> CompiledForEachRaw>>,
    /// Required component bitset for left-side entity collection in join plans.
    pub(super) left_required: Option<FixedBitSet>,
    /// Changed component bitset for left-side change detection in join plans.
    pub(super) left_changed: Option<FixedBitSet>,
    /// Required component bitset for spatial index-gather path.
    pub(super) required_for_spatial: Option<FixedBitSet>,
    /// Changed component bitset for spatial index-gather path.
    pub(super) changed_for_spatial: Option<FixedBitSet>,
    /// Aggregate expressions to compute over matched entities.
    pub(super) aggregates: Vec<AggregateExpr>,
}
pub(super) struct JoinSpec {
    right_query_name: &'static str,
    right_estimated_rows: usize,
    join_kind: JoinKind,
    /// Required component bitset for right-side entity collection.
    right_required: FixedBitSet,
    /// Changed component bitset for right-side change detection.
    right_changed: FixedBitSet,
}

/// Specification for an ER (Entity-Relationship) join.
pub(super) struct ErJoinSpec {
    /// Name of the entity-reference component type (e.g. `"Parent"`).
    ref_component_name: &'static str,
    /// Name of the right-side query type.
    right_query_name: &'static str,
    right_estimated_rows: usize,
    join_kind: JoinKind,
    /// Required component bitset for right-side entity collection.
    right_required: FixedBitSet,
    /// Changed component bitset for right-side change detection.
    right_changed: FixedBitSet,
    /// Type-erased extractor for the entity reference.
    ref_extractor: EntityRefExtractor,
}

impl ScanBuilder<'_> {
    /// Add a filter predicate. The planner will automatically determine
    /// whether to push it into an index lookup or apply it as a post-filter.
    pub fn filter(mut self, predicate: Predicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Add an aggregate expression to compute over matched entities.
    ///
    /// Multiple aggregates can be chained. The plan will produce an
    /// `Aggregate` node that computes all expressions in a single pass.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let plan = planner
    ///     .scan::<(&Score,)>()
    ///     .aggregate(AggregateExpr::count())
    ///     .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
    ///     .build();
    /// ```
    pub fn aggregate(mut self, expr: AggregateExpr) -> Self {
        self.aggregates.push(expr);
        self
    }

    /// Add a join with another query type.
    ///
    /// The planner will choose between hash join and nested-loop join based
    /// on estimated cardinalities.
    ///
    /// Regular joins and ER joins may be added in any order — `build()`
    /// always executes regular joins first (sorted intersection), then ER
    /// joins (hash probe on entity references).
    pub fn join<Q: crate::query::fetch::WorldQuery + 'static>(
        mut self,
        join_kind: JoinKind,
    ) -> Self {
        let required = Q::required_ids(self.planner.components);
        // Estimate entities matching the right query's components.
        let right_rows = self.planner.estimate_matching_entities(&required);
        let changed = Q::changed_ids(self.planner.components);
        let idx = self.joins.len();
        self.joins.push(JoinSpec {
            right_query_name: std::any::type_name::<Q>(),
            right_estimated_rows: right_rows,
            join_kind,
            right_required: required,
            right_changed: changed,
        });
        self.last_join = Some(LastJoinKind::Regular(idx));
        self
    }

    /// Set explicit row estimate for the most recently added join's right side.
    ///
    /// Returns `Err(PlannerError::BuilderOrder)` if called before any
    /// `join()` or `er_join()`.
    pub fn with_right_estimate(mut self, rows: usize) -> Result<Self, PlannerError> {
        match self.last_join {
            Some(LastJoinKind::Regular(idx)) => {
                self.joins[idx].right_estimated_rows = rows;
            }
            Some(LastJoinKind::Er(idx)) => {
                self.er_joins[idx].right_estimated_rows = rows;
            }
            None => {
                return Err(PlannerError::BuilderOrder(
                    "with_right_estimate() called before any join() or er_join()".into(),
                ));
            }
        }
        Ok(self)
    }

    /// Add an ER (Entity-Relationship) join.
    ///
    /// Unlike [`join`](Self::join), which intersects two entity sets by
    /// identity, an ER join follows entity references: for each left-side
    /// entity, it reads component `R` (which implements [`AsEntityRef`]),
    /// extracts the referenced `Entity`, and checks whether that target
    /// entity is in the right-side set (matching query `Q`).
    ///
    /// This implements a streaming hash join: the right side is collected
    /// into a `HashSet<Entity>`, then each left entity probes the set via
    /// its entity reference.
    ///
    /// # Type Parameters
    ///
    /// - `R`: The entity-reference component on the left side (implements
    ///   [`AsEntityRef`]).
    /// - `Q`: The query defining which components the *referenced* entity
    ///   must have.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // "For each entity with a Parent component, keep only those whose
    /// //  parent entity has Pos and Name."
    /// let plan = planner
    ///     .scan::<(&ChildTag, &Parent)>()
    ///     .er_join::<Parent, (&Pos, &Name)>(JoinKind::Inner)?
    ///     .build();
    /// # Ok::<(), minkowski::PlannerError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Err(PlannerError::UnregisteredComponent)` if the reference
    /// component `R` has not been registered in the World. Register it with
    /// `world.register_component::<R>()` before building the plan.
    pub fn er_join<R, Q>(mut self, join_kind: JoinKind) -> Result<Self, PlannerError>
    where
        R: AsEntityRef,
        Q: crate::query::fetch::WorldQuery + 'static,
    {
        let ref_comp_id =
            self.planner
                .components
                .id::<R>()
                .ok_or(PlannerError::UnregisteredComponent(
                    std::any::type_name::<R>(),
                ))?;

        let required = Q::required_ids(self.planner.components);
        // Estimate entities matching the right query's components.
        let right_rows = self.planner.estimate_matching_entities(&required);
        let changed = Q::changed_ids(self.planner.components);

        let ref_extractor: EntityRefExtractor = Arc::new(move |world: &World, entity: Entity| {
            let r: &R = world.get_by_id(entity, ref_comp_id)?;
            Some(r.entity_ref())
        });

        let idx = self.er_joins.len();
        self.er_joins.push(ErJoinSpec {
            ref_component_name: std::any::type_name::<R>(),
            right_query_name: std::any::type_name::<Q>(),
            right_estimated_rows: right_rows,
            join_kind,
            right_required: required,
            right_changed: changed,
            ref_extractor,
        });
        self.last_join = Some(LastJoinKind::Er(idx));
        Ok(self)
    }

    /// Compile the scan into an optimized execution plan.
    pub fn build(mut self) -> QueryPlanResult {
        let mut warnings = std::mem::take(&mut self.deferred_warnings);

        // Phase 1: Classify predicates — index-driven vs spatial vs post-filter.
        let mut index_preds: Vec<(Predicate, &IndexDescriptor)> = Vec::new();
        let mut spatial_preds: Vec<(Predicate, SpatialCost, Arc<dyn SpatialIndex + Send + Sync>)> =
            Vec::new();
        let mut filter_preds = Vec::new();
        let planner = self.planner;

        for pred in self.predicates {
            if pred.can_use_spatial() {
                match planner.find_spatial_index(&pred) {
                    SpatialLookupResult::Accelerated(_name, cost, index) => {
                        spatial_preds.push((pred, cost, index));
                    }
                    SpatialLookupResult::Declined(expression) => {
                        warnings.push(PlanWarning::SpatialIndexDeclined {
                            component_name: pred.component_name,
                            expression,
                        });
                        filter_preds.push(pred);
                    }
                    SpatialLookupResult::NoIndex => {
                        planner.warn_missing_index(&pred, &mut warnings);
                        filter_preds.push(pred);
                    }
                }
            } else if let Some(idx) = planner.find_best_index(&pred) {
                index_preds.push((pred, idx));
            } else {
                // Generate warnings for missing indexes.
                planner.warn_missing_index(&pred, &mut warnings);
                filter_preds.push(pred);
            }
        }

        // Phase 2: Order index lookups by selectivity (most selective first).
        index_preds.sort_by(|a, b| a.0.selectivity.total_cmp(&b.0.selectivity));
        // Sort spatial predicates by total cost (cpu), not estimated_rows alone.
        // A predicate with fewer rows but much higher CPU should not beat one
        // with more rows but lower total cost.
        spatial_preds.sort_by(|a, b| {
            Cost::spatial_lookup(&a.1)
                .total()
                .total_cmp(&Cost::spatial_lookup(&b.1).total())
        });

        // Collect all filter closures for compiled scan fusion.
        // Must happen before Phase 4 consumes filter_preds.
        // Debug-assert that no Eq/Range predicate has filter_fn: None — if
        // a predicate appears in the plan but has no executable filter, the
        // plan's EXPLAIN would show a filter node that doesn't actually run.
        debug_assert!(
            index_preds.iter().all(|(p, _)| p.filter_fn.is_some()
                || matches!(
                    p.kind,
                    PredicateKind::Custom(_) | PredicateKind::CustomColumn(_)
                )),
            "Eq/Range predicate with filter_fn: None — plan would show filter but not apply it"
        );
        debug_assert!(
            filter_preds.iter().all(|p| p.filter_fn.is_some()
                || matches!(
                    p.kind,
                    PredicateKind::Custom(_) | PredicateKind::CustomColumn(_)
                )),
            "Eq/Range predicate with filter_fn: None — plan would show filter but not apply it"
        );
        debug_assert!(
            spatial_preds.iter().all(|(p, _, _)| p.filter_fn.is_some()),
            "Spatial predicate with filter_fn: None — plan would show filter but not apply it"
        );
        let all_filter_fns: Vec<FilterFn> = index_preds
            .iter()
            .filter_map(|(p, _)| p.filter_fn.as_ref().map(Arc::clone))
            .chain(
                spatial_preds
                    .iter()
                    .filter_map(|(p, _, _)| p.filter_fn.as_ref().map(Arc::clone)),
            )
            .chain(
                filter_preds
                    .iter()
                    .filter_map(|p| p.filter_fn.as_ref().map(Arc::clone)),
            )
            .collect();
        // Capture before all_filter_fns is moved into compiled closures.
        let has_any_filters = !all_filter_fns.is_empty();

        // Collect column-aware filter closures from filter predicates.
        // Column filters enable the scan_required fast path even when custom
        // predicates are present, by operating on typed column slices per
        // archetype instead of per-entity world.get() dispatch.
        let all_column_filter_fns: Vec<ColumnFilterFn> = filter_preds
            .iter()
            .filter_map(|p| p.column_filter_fn.as_ref().map(Arc::clone))
            .collect();
        // All filters are column-aware when every filter_pred has a column_filter_fn
        // and there are no index/spatial predicates contributing per-entity filters.
        let all_filters_are_column = !filter_preds.is_empty()
            && index_preds.is_empty()
            && spatial_preds.is_empty()
            && filter_preds.len() == all_column_filter_fns.len();

        // Phase 3: Build the logical plan tree.
        //
        // Cost-based selection: compare best spatial index cost against
        // best BTree/Hash index cost, and use whichever is cheaper as the
        // driving access. Remaining predicates become post-filters.
        let mut node: PlanNode;

        // Determine driving access: compare full candidate plan costs
        // including downstream filters, not just the driving access alone.
        // A spatial index with high estimated_rows but low CPU can still lose
        // to a selective BTree if the BTree's post-filter cost is lower overall.
        let spatial_plan_cost = spatial_preds.first().map(|(_, first_sc, _)| {
            let mut cost = Cost::spatial_lookup(first_sc);
            for (pred, _, _) in spatial_preds.iter().skip(1) {
                cost = Cost::filter(cost, pred.selectivity);
            }
            for (pred, _) in &index_preds {
                cost = Cost::filter(cost, pred.selectivity);
            }
            cost.total()
        });

        let index_plan_cost = index_preds.first().map(|(first_pred, _)| {
            let mut cost = Cost::index_lookup(first_pred.selectivity, self.estimated_rows);
            for (pred, _) in index_preds.iter().skip(1) {
                cost = Cost::filter(cost, pred.selectivity);
            }
            for (pred, _, _) in &spatial_preds {
                cost = Cost::filter(cost, pred.selectivity);
            }
            cost.total()
        });

        let use_spatial_driver = match (spatial_plan_cost, index_plan_cost) {
            (Some(sc), Some(ic)) => sc <= ic,
            (Some(_), None) => true,
            _ => false,
        };

        // Compute spatial driver if spatial is the best driving access.
        let spatial_driver = if use_spatial_driver && !spatial_preds.is_empty() {
            let (first_pred, _, first_index) = &spatial_preds[0];
            let PredicateKind::Spatial(sp) = &first_pred.kind else {
                unreachable!("spatial_preds only contains Spatial predicates");
            };
            Some(SpatialDriver {
                expr: sp.into(),
                index: Arc::clone(first_index),
            })
        } else {
            None
        };

        // Compute index driver — pre-binds lookup fn + value for Phase 7/8.
        // Only used when no spatial driver is selected (spatial takes priority).
        let index_driver = if spatial_driver.is_none() {
            if let Some((first_pred, first_idx)) = index_preds.first() {
                let lookup_fn = match first_pred.kind {
                    PredicateKind::Eq => first_idx.eq_lookup_fn.as_ref(),
                    PredicateKind::Range => first_idx.range_lookup_fn.as_ref(),
                    _ => None,
                };
                if let (Some(fn_ref), Some(value)) = (lookup_fn, &first_pred.lookup_value) {
                    let bound_fn = Arc::clone(fn_ref);
                    let bound_value = Arc::clone(value);
                    Some(IndexDriver {
                        lookup_fn: Arc::new(move || bound_fn(&*bound_value)),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if use_spatial_driver && !spatial_preds.is_empty() {
            // Driving access is a spatial lookup.
            let (first_pred, first_cost, _) = &spatial_preds[0];
            let est = first_cost.estimated_rows.max(1.0) as usize;
            let base_cost = Cost::spatial_lookup(first_cost);
            let sort_overhead = (est as f64).log2().max(1.0);
            node = PlanNode::SpatialLookup {
                component_name: first_pred.component_name,
                predicate: format!("{:?}", first_pred),
                estimated_rows: est,
                cost: Cost {
                    rows: base_cost.rows,
                    cpu: base_cost.cpu * 0.9 + sort_overhead,
                },
            };

            // Additional spatial predicates become filters.
            for (pred, _, _) in spatial_preds.iter().skip(1) {
                let parent_cost = node.cost();
                node = PlanNode::Filter {
                    predicate: format!("{:?}", pred),
                    selectivity: pred.selectivity,
                    branchless: false, // spatial predicates are not branchless
                    cost: Cost::filter_with_branchless(parent_cost, pred.selectivity, false),
                    child: Box::new(node),
                };
            }

            // All index predicates become filters too.
            for (pred, _idx) in &index_preds {
                let bl = pred.is_branchless_eligible();
                let parent_cost = node.cost();
                node = PlanNode::Filter {
                    predicate: format!("{:?}", pred),
                    selectivity: pred.selectivity,
                    branchless: bl,
                    cost: Cost::filter_with_branchless(parent_cost, pred.selectivity, bl),
                    child: Box::new(node),
                };
            }
        } else if let Some((first_pred, first_idx)) = index_preds.first() {
            // Driving access is an index lookup.
            let est = (self.estimated_rows as f64 * first_pred.selectivity).max(1.0) as usize;
            let base_cost = Cost::index_lookup(first_pred.selectivity, self.estimated_rows);
            let sort_overhead = (est as f64).log2().max(1.0);
            node = PlanNode::IndexLookup {
                index_kind: first_idx.kind,
                component_name: first_pred.component_name,
                predicate: format!("{:?}", first_pred),
                estimated_rows: est,
                cost: Cost {
                    rows: base_cost.rows,
                    cpu: base_cost.cpu + sort_overhead,
                },
            };

            // Additional index predicates become filters.
            for (pred, _idx) in index_preds.iter().skip(1) {
                let bl = pred.is_branchless_eligible();
                let parent_cost = node.cost();
                node = PlanNode::Filter {
                    predicate: format!("{:?}", pred),
                    selectivity: pred.selectivity,
                    branchless: bl,
                    cost: Cost::filter_with_branchless(parent_cost, pred.selectivity, bl),
                    child: Box::new(node),
                };
            }

            // Spatial predicates that aren't the driver become filters.
            for (pred, _, _) in &spatial_preds {
                let parent_cost = node.cost();
                node = PlanNode::Filter {
                    predicate: format!("{:?}", pred),
                    selectivity: pred.selectivity,
                    branchless: false,
                    cost: Cost::filter_with_branchless(parent_cost, pred.selectivity, false),
                    child: Box::new(node),
                };
            }
        } else {
            // No usable index — full scan.
            let opts = VectorizeOpts::default();
            let base_cost = Cost::scan(self.estimated_rows);
            node = PlanNode::Scan {
                query_name: self.query_name,
                estimated_rows: self.estimated_rows,
                avg_chunk_size: self.estimated_rows.min(opts.target_chunk_rows).max(1),
                cost: Cost {
                    rows: base_cost.rows,
                    cpu: base_cost.cpu * 0.9,
                },
            };
        }

        // Phase 4: Apply remaining filter predicates.
        for pred in filter_preds {
            let bl = pred.is_branchless_eligible();
            let parent_cost = node.cost();
            node = PlanNode::Filter {
                predicate: format!("{:?}", pred),
                selectivity: pred.selectivity,
                branchless: bl,
                cost: Cost::filter_with_branchless(parent_cost, pred.selectivity, bl),
                child: Box::new(node),
            };
        }

        // Phase 4b: Join elimination — merge simple inner joins into scan.
        // An inner join with no predicates is pure component-presence filtering,
        // equivalent to widening the scan's required_ids. Eliminates run_join()
        // materialization, sort, and intersection for the common case.
        let mut any_eliminated = false;
        if let Some(ref mut left_req) = self.left_required {
            let left_chg = self.left_changed.get_or_insert_with(FixedBitSet::new);
            self.joins.retain(|join| {
                if join.join_kind == JoinKind::Inner {
                    left_req.grow(join.right_required.len());
                    left_req.union_with(&join.right_required);
                    left_chg.grow(join.right_changed.len());
                    left_chg.union_with(&join.right_changed);
                    warnings.push(PlanWarning::JoinEliminated {
                        right_name: join.right_query_name,
                    });
                    any_eliminated = true;
                    false
                } else {
                    true
                }
            });
        }

        // If any joins were eliminated, the compile_for_each factories hold
        // stale bitsets (captured at scan::<Q>() time with narrower requirements).
        // Replace them with new factories that capture the merged bitsets.
        if any_eliminated {
            let merged_req = self.left_required.clone().unwrap();
            let merged_chg = self.left_changed.clone().unwrap_or_default();
            let merged_req2 = merged_req.clone();
            let merged_chg2 = merged_chg.clone();

            self.compile_for_each = Some(Box::new(move || {
                Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        for arch in &world.archetypes.archetypes {
                            if arch.is_empty() || !merged_req.is_subset(&arch.component_ids) {
                                continue;
                            }
                            if !passes_change_filter(arch, &merged_chg, tick) {
                                continue;
                            }
                            for &entity in &arch.entities {
                                callback(entity);
                            }
                        }
                    },
                )
            }));
            self.compile_for_each_raw = Some(Box::new(move || {
                Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        for arch in &world.archetypes.archetypes {
                            if arch.is_empty() || !merged_req2.is_subset(&arch.component_ids) {
                                continue;
                            }
                            if !passes_change_filter(arch, &merged_chg2, tick) {
                                continue;
                            }
                            for &entity in &arch.entities {
                                callback(entity);
                            }
                        }
                    },
                )
            }));

            // Also update spatial bitsets if present.
            if let Some(ref mut spatial_req) = self.required_for_spatial
                && let Some(ref left_req) = self.left_required
            {
                spatial_req.grow(left_req.len());
                spatial_req.union_with(left_req);
            }
            if let Some(ref mut spatial_chg) = self.changed_for_spatial
                && let Some(ref left_chg) = self.left_changed
            {
                spatial_chg.grow(left_chg.len());
                spatial_chg.union_with(left_chg);
            }
        }

        let opts = VectorizeOpts::default();

        // Phase 5: Join ordering — smallest intermediate result drives the
        // left side; each join's output becomes the next left input.
        for join in &self.joins {
            let base_cost = Cost::scan(join.right_estimated_rows);
            let right_node = PlanNode::Scan {
                query_name: join.right_query_name,
                estimated_rows: join.right_estimated_rows,
                avg_chunk_size: join.right_estimated_rows.min(opts.target_chunk_rows).max(1),
                cost: Cost {
                    rows: base_cost.rows,
                    cpu: base_cost.cpu * 0.9,
                },
            };

            let left_cost = node.cost();
            let right_cost = right_node.cost();

            // Choose join strategy based on cardinality.
            let use_hash = left_cost.rows >= 64.0 || right_cost.rows >= 64.0;

            if use_hash {
                let (build, probe) =
                    if join.join_kind == JoinKind::Inner && left_cost.rows > right_cost.rows {
                        (right_node, node)
                    } else {
                        (node, right_node)
                    };
                let base_cost = Cost::hash_join(build.cost(), probe.cost());

                // Partition count: build side should fit in L2 cache.
                let build_bytes = build.cost().rows as usize * opts.avg_component_bytes;
                let l2 = opts.l2_cache_bytes.max(1);
                let partitions = build_bytes.div_ceil(l2).max(1);
                let partition_factor = if partitions > 1 { 0.7 } else { 0.9 };

                let vec_cost = Cost {
                    rows: base_cost.rows,
                    cpu: (build.cost().cpu + probe.cost().cpu)
                        + (base_cost.cpu - build.cost().cpu - probe.cost().cpu) * partition_factor,
                };

                node = PlanNode::HashJoin {
                    left: Box::new(build),
                    right: Box::new(probe),
                    join_kind: join.join_kind,
                    partitions,
                    cost: vec_cost,
                };
            } else {
                if left_cost.rows > right_cost.rows {
                    warnings.push(PlanWarning::UnindexedJoin {
                        left_name: self.query_name,
                        right_name: join.right_query_name,
                    });
                }
                let base_cost = Cost::nested_loop_join(left_cost, right_cost);
                node = PlanNode::NestedLoopJoin {
                    left: Box::new(node),
                    right: Box::new(right_node),
                    join_kind: join.join_kind,
                    cost: Cost {
                        rows: base_cost.rows,
                        cpu: base_cost.cpu * 0.95,
                    },
                };
            }
        }

        // Phase 5b: ER join plan nodes — streaming hash join on entity references.
        for er_join in &self.er_joins {
            let base_cost = Cost::scan(er_join.right_estimated_rows);
            let right_node = PlanNode::Scan {
                query_name: er_join.right_query_name,
                estimated_rows: er_join.right_estimated_rows,
                avg_chunk_size: er_join
                    .right_estimated_rows
                    .min(opts.target_chunk_rows)
                    .max(1),
                cost: Cost {
                    rows: base_cost.rows,
                    cpu: base_cost.cpu * 0.9,
                },
            };

            let left_cost = node.cost();
            let right_cost = right_node.cost();

            // ER joins always use hash join semantics: build a HashSet from
            // the right side, then probe with entity references from the left.
            // Cost model: right build + left probe (one hash lookup per left entity).
            let join_cost = Cost {
                rows: if er_join.join_kind == JoinKind::Inner {
                    // Conservative default: assume ~50% of references point to
                    // valid right entities. Override with `with_right_estimate`
                    // when domain-specific selectivity is known.
                    left_cost.rows * 0.5
                } else {
                    // Left join: all left entities survive, no right-side
                    // collection needed (short-circuited in run_join).
                    left_cost.rows
                },
                cpu: if er_join.join_kind == JoinKind::Left {
                    // Left ER join skips right-side collection entirely.
                    0.0
                } else {
                    // hash build (right scan) + probes (1.2x per left entity:
                    // hash lookup + ref extraction overhead)
                    right_cost.cpu + left_cost.rows * 1.2
                },
            };

            node = PlanNode::ErJoin {
                left: Box::new(node),
                right: Box::new(right_node),
                join_kind: er_join.join_kind,
                ref_component: er_join.ref_component_name,
                cost: join_cost,
            };
        }

        // Phase 7: Build join execution state if joins or ER joins are present.
        // Captures a left collector + one JoinStep per join, supporting
        // multi-join chains (A JOIN B JOIN C).
        let has_any_joins = !self.joins.is_empty() || !self.er_joins.is_empty();
        let join_exec = if has_any_joins {
            let left_required = self
                .left_required
                .clone()
                .expect("join plan requires left_required bitset");
            let left_changed = self.left_changed.clone().unwrap_or_default();
            let left_filters: Vec<FilterFn> = all_filter_fns.iter().map(Arc::clone).collect();
            let left_collector: EntityCollector = if let Some(ref driver) = spatial_driver {
                // Index-gather path: call the spatial index directly instead of
                // archetype scanning. Mirrors the Phase 8 index-gather closure.
                let index = Arc::clone(&driver.index);
                let expr = driver.expr.clone();
                let left_required_for_index = left_required.clone();
                let left_changed_for_index = left_changed.clone();
                Box::new(
                    move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                        let candidates = index.query(&expr);
                        gather_index_candidates(
                            world,
                            &candidates,
                            &left_required_for_index,
                            &left_changed_for_index,
                            tick,
                            &left_filters,
                            |entity| scratch.push(entity),
                        );
                    },
                )
            } else if let Some(ref driver) = index_driver {
                // BTree/Hash index-gather path: use the pre-bound lookup function
                // instead of archetype scanning. Mirrors the Phase 8 index-gather
                // closure for the left-side of a join.
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let left_required_for_index = left_required.clone();
                let left_changed_for_index = left_changed.clone();
                let left_filters_for_index: Vec<FilterFn> =
                    left_filters.iter().map(Arc::clone).collect();
                Box::new(
                    move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                        let candidates = lookup_fn();
                        gather_index_candidates(
                            world,
                            &candidates,
                            &left_required_for_index,
                            &left_changed_for_index,
                            tick,
                            &left_filters_for_index,
                            |entity| scratch.push(entity),
                        );
                    },
                )
            } else {
                // Archetype-scan path (unchanged).
                Box::new(
                    move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                        if left_filters.is_empty() {
                            collect_matching_entities(
                                world,
                                &left_required,
                                &left_changed,
                                tick,
                                scratch,
                            );
                        } else {
                            for arch in &world.archetypes.archetypes {
                                if arch.is_empty() || !left_required.is_subset(&arch.component_ids)
                                {
                                    continue;
                                }
                                if !passes_change_filter(arch, &left_changed, tick) {
                                    continue;
                                }
                                for &entity in &arch.entities {
                                    if left_filters.iter().all(|f| f(world, entity)) {
                                        scratch.push(entity);
                                    }
                                }
                            }
                        }
                    },
                )
            };

            let steps: Vec<JoinStep> = self
                .joins
                .iter()
                .map(|join| {
                    let right_required = join.right_required.clone();
                    let right_changed = join.right_changed.clone();
                    // Compute partition count using the same formula as Phase 5:
                    // build side = smaller of left and right estimates.
                    let left_rows = self.estimated_rows;
                    let right_rows = join.right_estimated_rows;
                    let build_rows = left_rows.min(right_rows);
                    let build_bytes = build_rows.saturating_mul(opts.avg_component_bytes);
                    let l2 = opts.l2_cache_bytes.max(1);
                    let partitions = build_bytes.div_ceil(l2).max(1);
                    JoinStep {
                        right_collector: Box::new(
                            move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                                collect_matching_entities(
                                    world,
                                    &right_required,
                                    &right_changed,
                                    tick,
                                    scratch,
                                );
                            },
                        ),
                        join_kind: join.join_kind,
                        partitions,
                    }
                })
                .collect();

            let er_steps: Vec<ErJoinStep> = self
                .er_joins
                .iter()
                .map(|er_join| {
                    let right_required = er_join.right_required.clone();
                    let right_changed = er_join.right_changed.clone();
                    let ref_extractor = Arc::clone(&er_join.ref_extractor);
                    ErJoinStep {
                        right_collector: Box::new(
                            move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                                collect_matching_entities(
                                    world,
                                    &right_required,
                                    &right_changed,
                                    tick,
                                    scratch,
                                );
                            },
                        ),
                        ref_extractor,
                        join_kind: er_join.join_kind,
                    }
                })
                .collect();

            Some(JoinExec {
                left_collector,
                steps,
                er_steps,
            })
        } else {
            None
        };

        // Phase 8: Compile execute_stream / execute_stream_raw closures for scan-only plans.
        // Enabled when there are no joins — predicates are fused as
        // per-entity filters into the scan closure.
        //
        // When a spatial driver is available, an index-gather path is compiled
        // instead of the usual archetype scan: the lookup function is called to
        // obtain candidate entities, which are then filtered for liveness,
        // Changed<T> compliance, and any remaining post-filter predicates.
        // Clone filter fns for the raw variant (Arc clone is a ref-count bump).
        let all_filter_fns_raw: Vec<FilterFn> = all_filter_fns.iter().map(Arc::clone).collect();

        // Take the required and changed bitsets; clone for the raw variant,
        // the index-driver path, and the aggregate batch path.
        let required_for_index = self.required_for_spatial.take().unwrap_or_default();
        let required_for_index_idx = required_for_index.clone();
        let required_for_index_raw = required_for_index.clone();
        let required_for_index_idx_raw = required_for_index.clone();
        let required_for_agg = required_for_index.clone();
        let required_for_agg_raw = required_for_index.clone();
        let changed_for_index = self.changed_for_spatial.take().unwrap_or_default();
        let changed_for_index_idx = changed_for_index.clone();
        let changed_for_index_raw = changed_for_index.clone();
        let changed_for_index_idx_raw = changed_for_index.clone();
        let changed_for_agg = changed_for_index.clone();
        let changed_for_agg_raw = changed_for_index.clone();
        // Clone filter fns for aggregate batch paths.
        let agg_filter_fns: Vec<FilterFn> = all_filter_fns.iter().map(Arc::clone).collect();
        let agg_filter_fns_raw: Vec<FilterFn> = all_filter_fns.iter().map(Arc::clone).collect();

        // Stash flags before the factories are consumed by .map().
        let has_scan_factory = self.compile_for_each.is_some();
        let has_scan_factory_raw = self.compile_for_each_raw.is_some();

        let compiled_for_each = if !has_any_joins {
            if let Some(ref driver) = spatial_driver {
                // Spatial index-gather path: call the spatial index directly
                // instead of scanning archetypes.
                let index = Arc::clone(&driver.index);
                let expr = driver.expr.clone();
                let required = required_for_index;
                let changed = changed_for_index;
                Some(Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        let candidates = index.query(&expr);
                        gather_index_candidates(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &all_filter_fns,
                            callback,
                        );
                    },
                ) as CompiledForEach)
            } else if let Some(ref driver) = index_driver {
                // BTree/Hash index-gather path — same validation pipeline as
                // the spatial path but driven by a pre-bound predicate lookup.
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let required = required_for_index_idx;
                let changed = changed_for_index_idx;
                Some(Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        let candidates = lookup_fn();
                        gather_index_candidates(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &all_filter_fns,
                            callback,
                        );
                    },
                ) as CompiledForEach)
            } else {
                // Existing archetype-scan path (unchanged).
                self.compile_for_each.map(|factory| {
                    let mut scan_fn = factory();

                    if all_filter_fns.is_empty() {
                        scan_fn
                    } else {
                        Box::new(
                            move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                                scan_fn(world, tick, &mut |entity: Entity| {
                                    if all_filter_fns.iter().all(|f| f(world, entity)) {
                                        callback(entity);
                                    }
                                });
                            },
                        )
                    }
                })
            }
        } else {
            None
        };

        let compiled_for_each_raw = if !has_any_joins {
            if let Some(ref driver) = spatial_driver {
                // Spatial index-gather path for the raw (transactional read) variant.
                let index = Arc::clone(&driver.index);
                let expr = driver.expr.clone();
                let required = required_for_index_raw;
                let changed = changed_for_index_raw;
                Some(Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        let candidates = index.query(&expr);
                        gather_index_candidates(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &all_filter_fns_raw,
                            callback,
                        );
                    },
                ) as CompiledForEachRaw)
            } else if let Some(ref driver) = index_driver {
                // BTree/Hash index-gather path for the raw variant.
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let required = required_for_index_idx_raw;
                let changed = changed_for_index_idx_raw;
                Some(Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        let candidates = lookup_fn();
                        gather_index_candidates(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &all_filter_fns_raw,
                            callback,
                        );
                    },
                ) as CompiledForEachRaw)
            } else {
                // Existing archetype-scan path for the raw variant (unchanged).
                self.compile_for_each_raw.map(|factory| {
                    let mut scan_fn = factory();

                    if all_filter_fns_raw.is_empty() {
                        scan_fn
                    } else {
                        Box::new(
                            move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                                scan_fn(world, tick, &mut |entity: Entity| {
                                    if all_filter_fns_raw.iter().all(|f| f(world, entity)) {
                                        callback(entity);
                                    }
                                });
                            },
                        )
                    }
                })
            }
        } else {
            None
        };

        // Phase 8b: Compile batch aggregate scan closures.
        //
        // These bypass the entity-callback pattern entirely — they iterate
        // archetypes or index results and call batch extractors directly.
        let compiled_agg_scan: Option<CompiledAggScan> = if !has_any_joins {
            if let Some(ref driver) = spatial_driver {
                let index = Arc::clone(&driver.index);
                let expr = driver.expr.clone();
                let required = required_for_agg;
                let changed = changed_for_agg;
                let filters = agg_filter_fns;
                Some(Box::new(
                    move |world: &World,
                          tick: Tick,
                          extractors: &mut [Option<Box<dyn BatchExtractor>>],
                          accums: &mut [AggregateAccum]| {
                        let candidates = index.query(&expr);
                        gather_index_batched(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &filters,
                            extractors,
                            accums,
                        );
                    },
                ))
            } else if let Some(ref driver) = index_driver {
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let required = required_for_agg;
                let changed = changed_for_agg;
                let filters = agg_filter_fns;
                Some(Box::new(
                    move |world: &World,
                          tick: Tick,
                          extractors: &mut [Option<Box<dyn BatchExtractor>>],
                          accums: &mut [AggregateAccum]| {
                        let candidates = lookup_fn();
                        gather_index_batched(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &filters,
                            extractors,
                            accums,
                        );
                    },
                ))
            } else {
                // Archetype scan: use batch path only when no post-filters.
                // When filters exist, fall through to compiled_for_each per-entity
                // path (filters require per-entity evaluation which defeats batching).
                if has_scan_factory && agg_filter_fns.is_empty() {
                    let required = required_for_agg;
                    let changed = changed_for_agg;
                    Some(Box::new(
                        move |world: &World,
                              tick: Tick,
                              extractors: &mut [Option<Box<dyn BatchExtractor>>],
                              accums: &mut [AggregateAccum]| {
                            for arch in &world.archetypes.archetypes {
                                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                                    continue;
                                }
                                if !passes_change_filter(arch, &changed, tick) {
                                    continue;
                                }
                                // Bind extractors and track which ones succeeded.
                                // An extractor fails to bind when the aggregate
                                // component is absent from this archetype (e.g.,
                                // scan::<&Pos> with sum::<Score> on an archetype
                                // that has Pos but not Score).
                                let count = arch.len();
                                for (accum, ext) in accums.iter_mut().zip(extractors.iter_mut()) {
                                    if accum.op == AggregateOp::Count {
                                        accum.count += count as u64;
                                    } else if let Some(e) = ext
                                        && e.bind_archetype(arch)
                                    {
                                        e.process_all(count, accum);
                                    }
                                }
                            }
                        },
                    ) as CompiledAggScan)
                } else {
                    None
                }
            }
        } else {
            None
        };

        let compiled_agg_scan_raw: Option<CompiledAggScan> = if !has_any_joins {
            if let Some(ref driver) = spatial_driver {
                let index = Arc::clone(&driver.index);
                let expr = driver.expr.clone();
                let required = required_for_agg_raw;
                let changed = changed_for_agg_raw;
                let filters = agg_filter_fns_raw;
                Some(Box::new(
                    move |world: &World,
                          tick: Tick,
                          extractors: &mut [Option<Box<dyn BatchExtractor>>],
                          accums: &mut [AggregateAccum]| {
                        let candidates = index.query(&expr);
                        gather_index_batched(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &filters,
                            extractors,
                            accums,
                        );
                    },
                ))
            } else if let Some(ref driver) = index_driver {
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let required = required_for_agg_raw;
                let changed = changed_for_agg_raw;
                let filters = agg_filter_fns_raw;
                Some(Box::new(
                    move |world: &World,
                          tick: Tick,
                          extractors: &mut [Option<Box<dyn BatchExtractor>>],
                          accums: &mut [AggregateAccum]| {
                        let candidates = lookup_fn();
                        gather_index_batched(
                            world,
                            &candidates,
                            &required,
                            &changed,
                            tick,
                            &filters,
                            extractors,
                            accums,
                        );
                    },
                ))
            } else {
                // Archetype scan for raw path: batch only without filters.
                if has_scan_factory_raw && agg_filter_fns_raw.is_empty() {
                    let required = required_for_agg_raw;
                    let changed = changed_for_agg_raw;
                    Some(Box::new(
                        move |world: &World,
                              tick: Tick,
                              extractors: &mut [Option<Box<dyn BatchExtractor>>],
                              accums: &mut [AggregateAccum]| {
                            for arch in &world.archetypes.archetypes {
                                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                                    continue;
                                }
                                if !passes_change_filter(arch, &changed, tick) {
                                    continue;
                                }
                                for (ext, accum) in extractors.iter_mut().zip(accums.iter()) {
                                    if accum.op != AggregateOp::Count
                                        && let Some(e) = ext
                                    {
                                        e.bind_archetype(arch);
                                    }
                                }
                                let count = arch.len();
                                for (accum, ext) in accums.iter_mut().zip(extractors.iter_mut()) {
                                    if accum.op == AggregateOp::Count {
                                        accum.count += count as u64;
                                    } else if let Some(e) = ext {
                                        e.process_all(count, accum);
                                    }
                                }
                            }
                        },
                    ) as CompiledAggScan)
                } else {
                    None
                }
            }
        } else {
            None
        };

        // Phase 9a: Resolve batch factory builders → batch factories.
        // This is the deferred ComponentId resolution: AggregateExpr constructors
        // capture the component type (via generics), and here we resolve to
        // ComponentId via the registry.
        let has_agg_scan = compiled_agg_scan.is_some();
        for expr in &mut self.aggregates {
            if let Some(builder) = expr.batch_factory_builder.take() {
                expr.batch_factory = builder(self.planner.components);
            }
            // If the batch scan path is active but a value-based aggregate has
            // no batch factory AND entities exist, the component type should be
            // registered. An empty world legitimately has unregistered types.
            debug_assert!(
                expr.op == AggregateOp::Count
                    || expr.batch_factory.is_some()
                    || !has_agg_scan
                    || self.planner.total_entities == 0,
                "aggregate {:?} has no batch factory but batch scan is active — \
                 component type may not be registered",
                expr.label,
            );
        }

        // Phase 9b: Wrap with aggregate node if aggregates are present.
        let aggregate_exprs = self.aggregates;
        // Capture pre-aggregate cardinality for scratch sizing — aggregate
        // nodes reduce rows to 1, but the scratch buffer stores full entity
        // sets from joins/scans underneath.
        let pre_agg_rows = node.cost().rows;
        if !aggregate_exprs.is_empty() {
            // Warn on duplicate labels — get_by_label returns the first match.
            let mut seen = std::collections::HashSet::new();
            for expr in &aggregate_exprs {
                if !seen.insert(&expr.label) {
                    warnings.push(PlanWarning::DuplicateAggregateLabel {
                        label: expr.label.clone(),
                    });
                }
            }
            let agg_labels: Vec<String> = aggregate_exprs.iter().map(|a| a.label.clone()).collect();
            // Aggregate cost: child cost + constant per-row overhead per aggregate.
            let child_cost = node.cost();
            let agg_cost = Cost {
                rows: 1.0, // aggregates produce a single result row
                cpu: child_cost.cpu + child_cost.rows * 0.1 * aggregate_exprs.len() as f64,
            };
            node = PlanNode::Aggregate {
                child: Box::new(node),
                aggregates: agg_labels,
                cost: agg_cost,
            };
        }

        // Phase 10: Pre-size scratch buffer from pre-aggregate cardinality.
        let scratch = if has_any_joins {
            let est = pre_agg_rows as usize;
            Some(ScratchBuffer::new(est * 3)) // room for left + right + output
        } else {
            Some(ScratchBuffer::new(pre_agg_rows as usize))
        };

        // Pre-allocate batch extractors and accumulators so aggregate
        // doesn't heap-allocate on every call.
        let cached_batch_extractors: Vec<Option<Box<dyn BatchExtractor>>> = aggregate_exprs
            .iter()
            .map(|expr| expr.batch_factory.as_ref().map(|f| f()))
            .collect();
        let cached_batch_extractors_raw: Vec<Option<Box<dyn BatchExtractor>>> = aggregate_exprs
            .iter()
            .map(|expr| expr.batch_factory.as_ref().map(|f| f()))
            .collect();
        let cached_accums: Vec<AggregateAccum> = aggregate_exprs
            .iter()
            .map(|expr| AggregateAccum::new(expr.op, expr.label.clone()))
            .collect();

        QueryPlanResult {
            root: node,
            join_exec,
            compiled_for_each,
            compiled_for_each_raw,
            scratch,
            opts,
            warnings,
            last_read_tick: Tick::default(),
            world_id: self.world_id,
            aggregate_exprs,
            compiled_agg_scan,
            compiled_agg_scan_raw,
            cached_batch_extractors,
            cached_batch_extractors_raw,
            cached_accums,
            row_indices: Vec::new(),
            // Direct archetype iteration: enabled for pure scans with no
            // per-entity filters, indexes, or spatial drivers. Also enabled
            // when all filters are column-aware (CustomColumn predicates).
            scan_required: if !has_any_joins
                && index_driver.is_none()
                && spatial_driver.is_none()
                && (!has_any_filters || all_filters_are_column)
            {
                self.left_required.clone()
            } else {
                None
            },
            scan_changed: self.left_changed.clone().unwrap_or_default(),
            scan_column_filters: if all_filters_are_column {
                all_column_filter_fns
            } else {
                Vec::new()
            },
            column_filter_mask: Vec::new(),
            filtered_entities: Vec::new(),
        }
    }
}
