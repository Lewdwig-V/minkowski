use std::collections::HashSet;
use std::fmt::{self, Write as _};
use std::sync::Arc;

use fixedbitset::FixedBitSet;

use crate::entity::Entity;
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
use crate::tick::Tick;
use crate::transaction::WorldMismatch;
use crate::world::{World, WorldId};

use super::aggregate::{
    AccumGuard, AggregateAccum, AggregateExpr, AggregateOp, AggregateResult, BatchExtractor,
    ValueExtractor,
};
use super::builder::JoinExec;
use super::cost::{CardinalityConstraint, Cost, VectorizeOpts};
use super::node::{JoinKind, PlanNode, PlanWarning};
use super::scratch::ScratchBuffer;
use super::{
    ColumnFilterFn, CompiledAggScan, CompiledForEach, CompiledForEachRaw, PlanExecError,
    passes_change_filter,
};

/// A compiled query execution plan.
pub struct QueryPlanResult {
    pub(super) root: PlanNode,
    pub(super) join_exec: Option<JoinExec>,
    pub(super) compiled_for_each: Option<CompiledForEach>,
    pub(super) compiled_for_each_raw: Option<CompiledForEachRaw>,
    pub(super) scratch: Option<ScratchBuffer>,
    pub(super) opts: VectorizeOpts,
    pub(super) warnings: Vec<PlanWarning>,
    pub(super) last_read_tick: Tick,
    pub(super) world_id: WorldId,
    /// Aggregate expressions for `aggregate()`. Empty if no aggregates.
    pub(super) aggregate_exprs: Vec<AggregateExpr>,
    /// Compiled batch aggregate scan (for `aggregate`).
    pub(super) compiled_agg_scan: Option<CompiledAggScan>,
    /// Compiled batch aggregate scan for raw path (for `aggregate_raw`).
    pub(super) compiled_agg_scan_raw: Option<CompiledAggScan>,
    /// Reusable buffer for row indices in batch execution methods.
    /// Cleared and repopulated on each `execute_stream_join_chunk` call.
    pub(super) row_indices: Vec<usize>,
    /// Cached batch extractors for `aggregate`. Initialized at
    /// build time and reused across calls to avoid per-call heap allocation.
    pub(super) cached_batch_extractors: Vec<Option<Box<dyn BatchExtractor>>>,
    /// Same for the raw (transactional) path.
    pub(super) cached_batch_extractors_raw: Vec<Option<Box<dyn BatchExtractor>>>,
    /// Cached accumulators reused across calls (avoids String clones per call).
    pub(super) cached_accums: Vec<AggregateAccum>,
    /// Component requirements for the direct archetype iteration fast path.
    /// `Some` when the plan is scan-only with no per-entity filter closures.
    /// `None` when the plan has joins, index/spatial drivers, or per-entity
    /// (non-column) custom filter closures.
    pub(super) scan_required: Option<FixedBitSet>,
    /// Changed-component bitset for the direct archetype iteration fast path.
    pub(super) scan_changed: FixedBitSet,
    /// Column-aware filter closures for the scan fast path.
    /// Applied per-archetype on typed column slices instead of per-entity
    /// `world.get()` dispatch. Non-empty only when `scan_required.is_some()`
    /// and the plan has `CustomColumn` predicates.
    pub(super) scan_column_filters: Vec<ColumnFilterFn>,
    /// Reusable boolean mask for column filter evaluation.
    /// Avoids per-archetype allocation in the scan fast path.
    pub(super) column_filter_mask: Vec<bool>,
    /// Reusable entity buffer for column-filtered join_chunk paths.
    /// Avoids per-archetype allocation when column filters reduce the
    /// entity set before passing to the chunk callback.
    pub(super) filtered_entities: Vec<Entity>,
}

impl QueryPlanResult {
    /// The plan root. Use this for introspection (matching on
    /// `PlanNode` variants to inspect index selection, join strategy, etc.).
    pub fn root(&self) -> &PlanNode {
        &self.root
    }

    /// Total estimated cost of the execution plan.
    pub fn cost(&self) -> Cost {
        self.root.cost()
    }

    /// Diagnostics generated during compilation.
    pub fn warnings(&self) -> &[PlanWarning] {
        &self.warnings
    }

    /// Human-readable execution plan.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Execution Plan ===\n");
        let _ = write!(out, "{}", self.root);
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                let _ = writeln!(out, "  - {w}");
            }
        }
        let _ = write!(
            out,
            "\nL2 cache budget: {} KiB, target chunk: {} rows\n",
            self.opts.l2_cache_bytes / 1024,
            self.opts.target_chunk_rows,
        );
        let _ = writeln!(
            out,
            "Estimated: {:.0} rows, {:.1} cpu",
            self.root.cost().rows,
            self.root.cost().cpu
        );
        out
    }

    /// Run join collectors into the scratch buffer.
    ///
    /// The scratch buffer is cleared before use. Join steps are applied
    /// sequentially: left collector populates the initial set, then each
    /// step collects right-side entities and intersects (inner join) or
    /// preserves left (left join). The result is left in the scratch buffer.
    ///
    /// This is pure computation over `&World` — the scratch is plan-local
    /// and invisible to the conflict model. All six execution methods
    /// (`execute_collect`, `execute_collect_raw`, `execute_stream`, `execute_stream_raw`,
    /// `aggregate`, `aggregate_raw`) delegate here for
    /// join plans.
    fn run_join(&mut self, world: &World) {
        let tick = self.last_read_tick;
        let join = self
            .join_exec
            .as_mut()
            .expect("run_join called without join_exec");
        let scratch = self
            .scratch
            .as_mut()
            .expect("run_join requires a scratch buffer");
        scratch.clear();

        // Collect initial left-side entities.
        (join.left_collector)(world, tick, scratch);

        // Apply each regular join step. After each step the result becomes the
        // left side for the next step, enabling A JOIN B JOIN C chains.
        for step in &mut join.steps {
            let left_len = scratch.len();
            (step.right_collector)(world, tick, scratch);
            match step.join_kind {
                JoinKind::Inner => {
                    let match_count = if step.partitions > 1 {
                        // Cache-aware path: partition entities into L2-sized
                        // buckets so each intersection fits in cache.
                        scratch
                            .partitioned_intersection(left_len, step.partitions)
                            .len()
                    } else {
                        scratch.sorted_intersection(left_len).len()
                    };
                    if match_count > 0 {
                        let total = scratch.entities.len();
                        scratch.entities.copy_within(total - match_count.., 0);
                    }
                    scratch.entities.truncate(match_count);
                }
                JoinKind::Left => {
                    // Keep only left entities (discard right).
                    scratch.entities.truncate(left_len);
                }
            }
        }

        // Apply ER join steps: streaming hash join on entity references.
        // For each ER step: collect right-side entities into a HashSet,
        // then filter left entities by checking if their entity reference
        // target is in the set.
        for er_step in &mut join.er_steps {
            match er_step.join_kind {
                JoinKind::Inner => {
                    // Build phase: collect right-side entities into a HashSet
                    // for O(1) probing.
                    let mut right_scratch = ScratchBuffer::new(128);
                    (er_step.right_collector)(world, tick, &mut right_scratch);
                    let right_set: HashSet<Entity> =
                        right_scratch.entities.iter().copied().collect();

                    // Probe phase: retain left entities whose entity reference
                    // target is in the right set.
                    let ref_extractor = &er_step.ref_extractor;
                    scratch.entities.retain(|&entity| {
                        ref_extractor(world, entity)
                            .is_some_and(|target| right_set.contains(&target))
                    });
                }
                JoinKind::Left => {
                    // Left join: all left entities pass through regardless of
                    // whether their reference target is in the right set.
                    // Skip right-side collection entirely to avoid wasted work.
                }
            }
        }
    }

    /// Execute the plan against a live world, returning matching entities.
    ///
    /// For join plans, entities are collected from both sides into an internal
    /// scratch buffer, then joined via sorted intersection (inner join) or
    /// left-side preservation (left join). For ER join plans, left-side
    /// entities are further filtered by probing a hash set of right-side
    /// entities via entity references. The buffer is reused across calls
    /// to amortize allocation.
    ///
    /// For scan-only plans without joins, this collects entities into the
    /// scratch buffer using the compiled scan closure (with filter fusion).
    /// Prefer [`execute_stream`](Self::execute_stream) for scan-only plans to avoid the
    /// intermediate buffer entirely.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    ///
    /// # Panics
    ///
    /// Panics if the plan has no scratch buffer (should not happen for plans
    /// built via [`ScanBuilder::build`]).
    pub fn execute_collect(&mut self, world: &mut World) -> Result<&[Entity], PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }

        // Fast path: scan-only plan with no custom predicates or index drivers.
        // Bypass CompiledForEach and push entities directly into scratch.
        if self.scan_required.is_some() {
            let Self {
                scan_required,
                scan_changed,
                scan_column_filters,
                column_filter_mask,
                last_read_tick,
                scratch,
                ..
            } = self;
            let required = scan_required.as_ref().unwrap();
            let scratch = scratch
                .as_mut()
                .expect("execute_collect() requires a plan with a scratch buffer");
            scratch.clear();
            let tick = *last_read_tick;
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !passes_change_filter(arch, scan_changed, tick) {
                    continue;
                }
                if scan_column_filters.is_empty() {
                    for &entity in &arch.entities {
                        scratch.push(entity);
                    }
                } else {
                    column_filter_mask.clear();
                    column_filter_mask.resize(arch.len(), true);
                    for cf in scan_column_filters.iter() {
                        cf(world, arch, &arch.entities, column_filter_mask);
                    }
                    for (i, &entity) in arch.entities.iter().enumerate() {
                        if column_filter_mask[i] {
                            scratch.push(entity);
                        }
                    }
                }
            }
            *last_read_tick = world.next_tick();
            return Ok(scratch.as_slice());
        }

        if self.join_exec.is_some() {
            self.run_join(&*world);
            self.last_read_tick = world.next_tick();
            Ok(self.scratch.as_ref().unwrap().as_slice())
        } else if let Some(compiled) = &mut self.compiled_for_each {
            let scratch = self
                .scratch
                .as_mut()
                .expect("execute_collect() requires a plan with a scratch buffer");
            scratch.clear();
            let tick = self.last_read_tick;
            compiled(&*world, tick, &mut |entity: Entity| {
                scratch.push(entity);
            });
            self.last_read_tick = world.next_tick();
            Ok(scratch.as_slice())
        } else {
            panic!(
                "execute_collect() called on a plan with no join executor and no compiled scan — \
                 this indicates a bug in plan compilation"
            );
        }
    }

    /// Execute the plan with read-only world access, returning matching entities.
    ///
    /// Like [`execute_collect`](Self::execute_collect) but takes `&World` instead of `&mut World`.
    /// No tick advancement, no query cache mutation. Safe for use inside
    /// transactions where only `&World` is available.
    ///
    /// Supports both scan-only and join plans. Join execution uses the plan's
    /// internal scratch buffer — this is pure computation, invisible to the
    /// conflict model. The scratch is plan-local and ephemeral: it exists for
    /// the duration of the join and is never observable by other reducers.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    ///
    /// # Panics
    ///
    /// Panics if the plan has no scratch buffer (should not happen for plans
    /// built via [`ScanBuilder::build`]).
    pub fn execute_collect_raw(&mut self, world: &World) -> Result<&[Entity], PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }

        // Fast path: scan-only plan (no tick advancement).
        if self.scan_required.is_some() {
            let Self {
                scan_required,
                scan_changed,
                scan_column_filters,
                column_filter_mask,
                last_read_tick,
                scratch,
                ..
            } = self;
            let required = scan_required.as_ref().unwrap();
            let scratch = scratch
                .as_mut()
                .expect("execute_collect_raw() requires a plan with a scratch buffer");
            scratch.clear();
            let tick = *last_read_tick;
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !passes_change_filter(arch, scan_changed, tick) {
                    continue;
                }
                if scan_column_filters.is_empty() {
                    for &entity in &arch.entities {
                        scratch.push(entity);
                    }
                } else {
                    column_filter_mask.clear();
                    column_filter_mask.resize(arch.len(), true);
                    for cf in scan_column_filters.iter() {
                        cf(world, arch, &arch.entities, column_filter_mask);
                    }
                    for (i, &entity) in arch.entities.iter().enumerate() {
                        if column_filter_mask[i] {
                            scratch.push(entity);
                        }
                    }
                }
            }
            return Ok(scratch.as_slice());
        }

        if self.join_exec.is_some() {
            self.run_join(world);
            Ok(self.scratch.as_ref().unwrap().as_slice())
        } else if let Some(compiled) = &mut self.compiled_for_each_raw {
            let scratch = self
                .scratch
                .as_mut()
                .expect("execute_collect_raw() requires a plan with a scratch buffer");
            scratch.clear();
            let tick = self.last_read_tick;
            compiled(world, tick, &mut |entity: Entity| {
                scratch.push(entity);
            });
            Ok(scratch.as_slice())
        } else {
            panic!(
                "execute_collect_raw() called on a plan with no join executor and no compiled scan — \
                 this indicates a bug in plan compilation"
            );
        }
    }

    /// Execute the plan, calling `callback` for each matching entity.
    ///
    /// For scan-only plans (no joins) without a spatial index driver, this
    /// compiles to archetype iteration with no intermediate allocation.
    /// When a spatial driver is present, the lookup function allocates
    /// a candidate list per call.
    ///
    /// For join plans, entities are materialised into the plan's internal
    /// scratch buffer first (sorted intersection for inner joins, left-side
    /// preservation for left joins), then the callback is called for each
    /// result entity. The materialisation is invisible to the conflict model —
    /// the scratch is plan-local computation, not shared state.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    pub fn execute_stream(
        &mut self,
        world: &mut World,
        mut callback: impl FnMut(Entity),
    ) -> Result<(), PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }
        // Fast path: scan-only plan with no per-entity filter closures.
        // Walk archetypes directly — the callback is monomorphic (`impl FnMut`),
        // so this avoids the double trait-object dispatch through CompiledForEach
        // (`Box<dyn FnMut>` outer + `&mut dyn FnMut` inner).
        // When column filters are present, they are applied per-archetype on
        // typed column slices instead of per-entity world.get() dispatch.
        if self.scan_required.is_some() {
            let Self {
                scan_required,
                scan_changed,
                scan_column_filters,
                column_filter_mask,
                last_read_tick,
                ..
            } = self;
            let required = scan_required.as_ref().unwrap();
            let tick = *last_read_tick;
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !passes_change_filter(arch, scan_changed, tick) {
                    continue;
                }
                if scan_column_filters.is_empty() {
                    for &entity in &arch.entities {
                        callback(entity);
                    }
                } else {
                    column_filter_mask.clear();
                    column_filter_mask.resize(arch.len(), true);
                    for cf in scan_column_filters.iter() {
                        cf(world, arch, &arch.entities, column_filter_mask);
                    }
                    for (i, &entity) in arch.entities.iter().enumerate() {
                        if column_filter_mask[i] {
                            callback(entity);
                        }
                    }
                }
            }
            *last_read_tick = world.next_tick();
            return Ok(());
        }
        if self.join_exec.is_some() {
            self.run_join(&*world);
            for &entity in self.scratch.as_ref().unwrap().as_slice() {
                callback(entity);
            }
            self.last_read_tick = world.next_tick();
        } else if let Some(compiled) = self.compiled_for_each.as_mut() {
            let tick = self.last_read_tick;
            compiled(&*world, tick, &mut callback);
            self.last_read_tick = world.next_tick();
        } else {
            panic!(
                "execute_stream() called on a plan with no join executor and no compiled scan — \
                 this indicates a bug in plan compilation"
            );
        }
        Ok(())
    }

    /// Execute the plan with read-only world access.
    ///
    /// For use inside transactions where only `&World` is available.
    /// No query cache mutation, no tick advancement. Requires the plan's
    /// query to be `ReadOnlyWorldQuery`.
    ///
    /// For scan-only plans this is a streaming pass with no intermediate
    /// allocation. For join plans, entities are materialised into the plan's
    /// internal scratch buffer before being streamed through the callback.
    /// The scratch is plan-local computation — ephemeral, plan-owned, and
    /// invisible to the transaction's conflict model. This is the same
    /// principle as pipeline-internal buffers: they hold data during execution
    /// but are not observable outside the pipeline.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    pub fn execute_stream_raw(
        &mut self,
        world: &World,
        mut callback: impl FnMut(Entity),
    ) -> Result<(), PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }
        // Fast path: scan-only plan (same as execute_stream but no tick advancement).
        if self.scan_required.is_some() {
            let Self {
                scan_required,
                scan_changed,
                scan_column_filters,
                column_filter_mask,
                last_read_tick,
                ..
            } = self;
            let required = scan_required.as_ref().unwrap();
            let tick = *last_read_tick;
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !passes_change_filter(arch, scan_changed, tick) {
                    continue;
                }
                if scan_column_filters.is_empty() {
                    for &entity in &arch.entities {
                        callback(entity);
                    }
                } else {
                    column_filter_mask.clear();
                    column_filter_mask.resize(arch.len(), true);
                    for cf in scan_column_filters.iter() {
                        cf(world, arch, &arch.entities, column_filter_mask);
                    }
                    for (i, &entity) in arch.entities.iter().enumerate() {
                        if column_filter_mask[i] {
                            callback(entity);
                        }
                    }
                }
            }
            return Ok(());
        }
        if self.join_exec.is_some() {
            self.run_join(world);
            for &entity in self.scratch.as_ref().unwrap().as_slice() {
                callback(entity);
            }
        } else if let Some(compiled) = self.compiled_for_each_raw.as_mut() {
            let tick = self.last_read_tick;
            compiled(world, tick, &mut callback);
        } else {
            panic!(
                "execute_stream_raw() called on a plan with no join executor and no compiled scan — \
                 this indicates a bug in plan compilation"
            );
        }
        Ok(())
    }

    /// Execute the plan with archetype-sorted batch extraction.
    ///
    /// After join materialisation (or scan collection), entities are sorted
    /// by `(ArchetypeId, Row)` so that consecutive entities share the same
    /// archetype. For each archetype run, `Q::init_fetch` is called once
    /// to resolve column pointers, then `Q::fetch` is called per entity
    /// with just a pointer offset — no generation check, no TypeId hash,
    /// no column search.
    ///
    /// `Q` is specified at the call site and validated at runtime against
    /// each archetype's component set. Returns `Err(ComponentMismatch)` if
    /// `Q`'s required components are missing from any matched archetype.
    ///
    /// Advances the read tick (same as `execute_stream`).
    pub fn execute_stream_batched<Q, F>(
        &mut self,
        world: &mut World,
        mut callback: F,
    ) -> Result<(), PlanExecError>
    where
        Q: WorldQuery,
        F: FnMut(Entity, Q::Item<'_>),
    {
        // Mark mutable columns changed before iteration (mirrors world.query()).
        let mutable = Q::mutable_ids(&world.components);
        if !mutable.is_empty() {
            let tick = world.next_tick();
            for arch in &mut world.archetypes.archetypes {
                if arch.is_empty()
                    || !Q::required_ids(&world.components).is_subset(&arch.component_ids)
                {
                    continue;
                }
                for comp_id in mutable.ones() {
                    if let Some(col_idx) = arch.column_index(comp_id) {
                        arch.columns[col_idx].mark_changed(tick);
                    }
                }
            }
        }
        self.execute_stream_batched_inner::<Q, F>(world, &mut callback)?;
        self.last_read_tick = world.next_tick();
        Ok(())
    }

    /// Read-only variant of [`execute_stream_batched`](Self::execute_stream_batched).
    /// No tick advancement.
    /// Safe for use inside transactions where only `&World` is available.
    pub fn execute_stream_batched_raw<Q, F>(
        &mut self,
        world: &World,
        mut callback: F,
    ) -> Result<(), PlanExecError>
    where
        Q: ReadOnlyWorldQuery,
        F: FnMut(Entity, Q::Item<'_>),
    {
        self.execute_stream_batched_inner::<Q, F>(world, &mut callback)
    }

    /// Shared implementation for `execute_stream_batched` and `execute_stream_batched_raw`.
    fn execute_stream_batched_inner<Q, F>(
        &mut self,
        world: &World,
        callback: &mut F,
    ) -> Result<(), PlanExecError>
    where
        Q: WorldQuery,
        F: FnMut(Entity, Q::Item<'_>),
    {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }

        // Fast path: scan-only plan.
        // Walk archetypes directly — no ScratchBuffer, no sort.
        if let Some(ref required) = self.scan_required {
            let changed = &self.scan_changed;
            let tick = self.last_read_tick;
            let q_required = Q::required_ids(&world.components);
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !passes_change_filter(arch, changed, tick) {
                    continue;
                }
                if !q_required.is_subset(&arch.component_ids) {
                    return Err(PlanExecError::ComponentMismatch {
                        query: std::any::type_name::<Q>(),
                        archetype_id: arch.id,
                    });
                }
                let fetch = Q::init_fetch(arch, &world.components);
                if self.scan_column_filters.is_empty() {
                    for (row, &entity) in arch.entities.iter().enumerate() {
                        let item = unsafe { Q::fetch(&fetch, row) };
                        callback(entity, item);
                    }
                } else {
                    self.column_filter_mask.clear();
                    self.column_filter_mask.resize(arch.len(), true);
                    for cf in &self.scan_column_filters {
                        cf(world, arch, &arch.entities, &mut self.column_filter_mask);
                    }
                    for (row, &entity) in arch.entities.iter().enumerate() {
                        if self.column_filter_mask[row] {
                            let item = unsafe { Q::fetch(&fetch, row) };
                            callback(entity, item);
                        }
                    }
                }
            }
            return Ok(());
        }

        // Phase 1: Populate scratch buffer.
        if self.join_exec.is_some() {
            self.run_join(world);
        } else if let Some(compiled) = &mut self.compiled_for_each_raw {
            let scratch = self
                .scratch
                .as_mut()
                .expect("execute_stream_batched requires a scratch buffer");
            scratch.clear();
            let tick = self.last_read_tick;
            compiled(world, tick, &mut |entity: Entity| {
                scratch.push(entity);
            });
        } else {
            // Scan-only plan with no compiled closure — empty result.
            // Tick advancement is the caller's responsibility.
            return Ok(());
        }

        // Phase 2: Sort by (archetype_id, row).
        let scratch = self
            .scratch
            .as_mut()
            .expect("execute_stream_batched requires a scratch buffer");
        scratch.sort_by_archetype(&world.entity_locations);

        // Phase 3: Walk archetype runs with pre-resolved fetch.
        let entities = scratch.as_slice();
        if entities.is_empty() {
            // Tick advancement is the caller's responsibility.
            return Ok(());
        }

        // Guard against aliased &mut T from duplicate entities (would be UB).
        debug_assert!(
            entities.windows(2).all(|w| w[0] != w[1]),
            "duplicate entity in batch buffer — would alias &mut T"
        );

        let required = Q::required_ids(&world.components);
        let mut run_start = 0;

        while run_start < entities.len() {
            let loc = world.entity_locations[entities[run_start].index() as usize]
                .expect("sorted entity has no location");
            let arch_id = loc.archetype_id;
            let archetype = &world.archetypes.archetypes[arch_id.0];

            // Validate Q's required components are present in this archetype.
            if !required.is_subset(&archetype.component_ids) {
                return Err(PlanExecError::ComponentMismatch {
                    query: std::any::type_name::<Q>(),
                    archetype_id: arch_id,
                });
            }

            let fetch = Q::init_fetch(archetype, &world.components);

            // Find end of this archetype run.
            let mut run_end = run_start + 1;
            while run_end < entities.len() {
                let next_loc = world.entity_locations[entities[run_end].index() as usize]
                    .expect("sorted entity has no location");
                if next_loc.archetype_id != arch_id {
                    break;
                }
                run_end += 1;
            }

            // Iterate entities in this run.
            for &entity in &entities[run_start..run_end] {
                let row = world.entity_locations[entity.index() as usize]
                    .expect("sorted entity has no location")
                    .row;
                debug_assert!(
                    row < archetype.len(),
                    "row {row} >= archetype len {}",
                    archetype.len()
                );
                let item = unsafe { Q::fetch(&fetch, row) };
                callback(entity, item);
            }

            run_start = run_end;
        }

        Ok(())
    }

    /// Execute the plan with archetype-chunked slice extraction.
    ///
    /// After join materialisation and archetype sorting, the callback
    /// receives per-archetype chunks containing:
    /// - `&[Entity]` — matched entities (sorted by row within this archetype)
    /// - `&[usize]` — row indices into the column slices
    /// - `Q::Slice<'_>` — full column slice for the archetype
    ///
    /// The callback can iterate `rows` and index into the slice:
    /// `for (i, &row) in rows.iter().enumerate() { let val = slice[row]; }`
    ///
    /// This enables SIMD-friendly access patterns on join results.
    /// Advances the read tick.
    pub fn execute_stream_join_chunk<Q, F>(
        &mut self,
        world: &mut World,
        mut callback: F,
    ) -> Result<(), PlanExecError>
    where
        Q: WorldQuery,
        F: FnMut(&[Entity], &[usize], Q::Slice<'_>),
    {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }

        // Mark mutable columns changed before iteration (mirrors world.query()).
        let mutable = Q::mutable_ids(&world.components);
        if !mutable.is_empty() {
            let tick = world.next_tick();
            for arch in &mut world.archetypes.archetypes {
                if arch.is_empty()
                    || !Q::required_ids(&world.components).is_subset(&arch.component_ids)
                {
                    continue;
                }
                for comp_id in mutable.ones() {
                    if let Some(col_idx) = arch.column_index(comp_id) {
                        arch.columns[col_idx].mark_changed(tick);
                    }
                }
            }
        }

        // Fast path: scan-only plan.
        // Walk archetypes directly — no ScratchBuffer, no sort.
        // Destructure self to borrow scan_required + row_indices disjointly.
        if self.scan_required.is_some() {
            let Self {
                scan_required,
                scan_changed,
                scan_column_filters,
                column_filter_mask,
                filtered_entities,
                last_read_tick,
                row_indices,
                ..
            } = self;
            let required = scan_required.as_ref().unwrap();
            let tick = *last_read_tick;
            let q_required = Q::required_ids(&world.components);
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !passes_change_filter(arch, scan_changed, tick) {
                    continue;
                }
                if !q_required.is_subset(&arch.component_ids) {
                    return Err(PlanExecError::ComponentMismatch {
                        query: std::any::type_name::<Q>(),
                        archetype_id: arch.id,
                    });
                }
                let fetch = Q::init_fetch(arch, &world.components);
                let slice = unsafe { Q::as_slice(&fetch, arch.len()) };
                if scan_column_filters.is_empty() {
                    row_indices.clear();
                    row_indices.extend(0..arch.len());
                    callback(&arch.entities, row_indices, slice);
                } else {
                    column_filter_mask.clear();
                    column_filter_mask.resize(arch.len(), true);
                    for cf in scan_column_filters.iter() {
                        cf(world, arch, &arch.entities, column_filter_mask);
                    }
                    row_indices.clear();
                    row_indices.extend((0..arch.len()).filter(|&i| column_filter_mask[i]));
                    if !row_indices.is_empty() {
                        filtered_entities.clear();
                        filtered_entities.extend(row_indices.iter().map(|&row| arch.entities[row]));
                        callback(filtered_entities, row_indices, slice);
                    }
                }
            }
            *last_read_tick = world.next_tick();
            return Ok(());
        }

        // Phase 1: Populate scratch buffer.
        if self.join_exec.is_some() {
            self.run_join(&*world);
        } else if let Some(compiled) = &mut self.compiled_for_each_raw {
            let scratch = self
                .scratch
                .as_mut()
                .expect("execute_stream_join_chunk requires a scratch buffer");
            scratch.clear();
            let tick = self.last_read_tick;
            compiled(&*world, tick, &mut |entity: Entity| {
                scratch.push(entity);
            });
        } else {
            // Scan-only plan with no compiled closure — empty result.
            self.last_read_tick = world.next_tick();
            return Ok(());
        }

        // Phase 2: Sort by (archetype_id, row).
        let scratch = self
            .scratch
            .as_mut()
            .expect("execute_stream_join_chunk requires a scratch buffer");
        scratch.sort_by_archetype(&world.entity_locations);

        // Early exit if scratch is empty (before destructuring self).
        if scratch.as_slice().is_empty() {
            self.last_read_tick = world.next_tick();
            return Ok(());
        }

        // Reborrow disjoint fields to avoid borrow conflict between
        // scratch.as_slice() (borrows self.scratch) and self.row_indices.
        let Self {
            scratch,
            row_indices,
            last_read_tick,
            ..
        } = self;
        let scratch = scratch.as_ref().expect("scratch buffer disappeared");
        let entities = scratch.as_slice();

        let required = Q::required_ids(&world.components);
        let mut run_start = 0;

        while run_start < entities.len() {
            let loc = world.entity_locations[entities[run_start].index() as usize]
                .expect("sorted entity has no location");
            let arch_id = loc.archetype_id;
            let archetype = &world.archetypes.archetypes[arch_id.0];

            // Validate Q's required components.
            if !required.is_subset(&archetype.component_ids) {
                return Err(PlanExecError::ComponentMismatch {
                    query: std::any::type_name::<Q>(),
                    archetype_id: arch_id,
                });
            }

            let fetch = Q::init_fetch(archetype, &world.components);

            // Find end of this archetype run.
            let mut run_end = run_start + 1;
            while run_end < entities.len() {
                let next_loc = world.entity_locations[entities[run_end].index() as usize]
                    .expect("sorted entity has no location");
                if next_loc.archetype_id != arch_id {
                    break;
                }
                run_end += 1;
            }

            // Collect row indices for this run.
            row_indices.clear();
            for &entity in &entities[run_start..run_end] {
                let row = world.entity_locations[entity.index() as usize]
                    .expect("sorted entity has no location")
                    .row;
                debug_assert!(row < archetype.len());
                row_indices.push(row);
            }

            let slice = unsafe { Q::as_slice(&fetch, archetype.len()) };
            callback(&entities[run_start..run_end], row_indices, slice);

            run_start = run_end;
        }

        *last_read_tick = world.next_tick();
        Ok(())
    }

    /// The `WorldId` this plan was built from.
    pub fn world_id(&self) -> WorldId {
        self.world_id
    }

    /// Returns `true` if this plan has aggregate expressions.
    pub fn has_aggregates(&self) -> bool {
        !self.aggregate_exprs.is_empty()
    }

    /// Execute aggregate functions over matched entities, returning computed values.
    ///
    /// This runs the underlying scan/filter/join plan and feeds each matching
    /// entity through the aggregate accumulators in a single pass.
    ///
    /// Returns an empty [`AggregateResult`] if no aggregate expressions were
    /// added during plan construction.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    pub fn aggregate(&mut self, world: &mut World) -> Result<AggregateResult, PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }
        if self.aggregate_exprs.is_empty() {
            return Ok(AggregateResult { values: Vec::new() });
        }

        // Reset cached accumulators instead of allocating new ones.
        for accum in &mut self.cached_accums {
            accum.reset();
        }

        let tick = self.last_read_tick;

        // Fast path: batch aggregate scan (chunk-at-a-time, no per-entity world.get).
        if let Some(compiled) = &mut self.compiled_agg_scan {
            compiled(
                &*world,
                tick,
                &mut self.cached_batch_extractors,
                &mut self.cached_accums,
            );
        } else if let Some(compiled) = &mut self.compiled_for_each {
            // Fallback: per-entity extraction (used for filter plans or when
            // batch factories are unavailable). AccumGuard moves accums out
            // to avoid borrow conflict and restores them on drop (including
            // unwind from user-supplied extractors/filters).
            let mut guard = AccumGuard::take(&mut self.cached_accums);
            let extractors: Vec<(AggregateOp, Option<ValueExtractor>)> = self
                .aggregate_exprs
                .iter()
                .map(|expr| (expr.op, expr.extractor.as_ref().map(Arc::clone)))
                .collect();
            let accums = guard.accums_mut();
            compiled(&*world, tick, &mut |entity: Entity| {
                for (i, (op, extractor)) in extractors.iter().enumerate() {
                    if *op == AggregateOp::Count {
                        accums[i].feed_count();
                    } else if let Some(ext) = extractor
                        && let Some(val) = ext(world, entity)
                    {
                        accums[i].feed(val);
                    }
                }
            });
            drop(guard);
        } else if self.join_exec.is_some() {
            // Join plans: materialise into scratch, then aggregate per-entity.
            self.run_join(&*world);
            let extractors: Vec<(AggregateOp, Option<ValueExtractor>)> = self
                .aggregate_exprs
                .iter()
                .map(|expr| (expr.op, expr.extractor.as_ref().map(Arc::clone)))
                .collect();
            for &entity in self.scratch.as_ref().unwrap().as_slice() {
                for (i, (op, extractor)) in extractors.iter().enumerate() {
                    if *op == AggregateOp::Count {
                        self.cached_accums[i].feed_count();
                    } else if let Some(ext) = extractor
                        && let Some(val) = ext(world, entity)
                    {
                        self.cached_accums[i].feed(val);
                    }
                }
            }
        } else {
            panic!("aggregate() called on a plan with no compiled scan and no join executor");
        }

        self.last_read_tick = world.next_tick();

        let values = self
            .cached_accums
            .iter()
            .map(|a| (a.label.clone(), a.finish()))
            .collect();
        Ok(AggregateResult { values })
    }

    /// Execute aggregate functions with read-only world access.
    ///
    /// For use inside transactions where only `&World` is available.
    /// No tick advancement. Supports both scan-only and join plans.
    ///
    /// Returns an empty [`AggregateResult`] if no aggregate expressions were
    /// added during plan construction.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    pub fn aggregate_raw(&mut self, world: &World) -> Result<AggregateResult, PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }
        if self.aggregate_exprs.is_empty() {
            return Ok(AggregateResult { values: Vec::new() });
        }

        // Reset cached accumulators.
        for accum in &mut self.cached_accums {
            accum.reset();
        }

        let tick = self.last_read_tick;

        // Fast path: batch aggregate scan.
        if let Some(compiled) = &mut self.compiled_agg_scan_raw {
            compiled(
                world,
                tick,
                &mut self.cached_batch_extractors_raw,
                &mut self.cached_accums,
            );
        } else if let Some(compiled) = &mut self.compiled_for_each_raw {
            // Fallback: per-entity extraction. AccumGuard moves accums out
            // to avoid borrow conflict and restores them on drop (including
            // unwind from user-supplied extractors/filters).
            let mut guard = AccumGuard::take(&mut self.cached_accums);
            let extractors: Vec<(AggregateOp, Option<ValueExtractor>)> = self
                .aggregate_exprs
                .iter()
                .map(|expr| (expr.op, expr.extractor.as_ref().map(Arc::clone)))
                .collect();
            let accums = guard.accums_mut();
            compiled(world, tick, &mut |entity: Entity| {
                for (i, (op, extractor)) in extractors.iter().enumerate() {
                    if *op == AggregateOp::Count {
                        accums[i].feed_count();
                    } else if let Some(ext) = extractor
                        && let Some(val) = ext(world, entity)
                    {
                        accums[i].feed(val);
                    }
                }
            });
            drop(guard);
        } else if self.join_exec.is_some() {
            // Join path: materialise into scratch, then aggregate per-entity.
            self.run_join(world);
            let extractors: Vec<(AggregateOp, Option<ValueExtractor>)> = self
                .aggregate_exprs
                .iter()
                .map(|expr| (expr.op, expr.extractor.as_ref().map(Arc::clone)))
                .collect();
            for &entity in self.scratch.as_ref().unwrap().as_slice() {
                for (i, (op, extractor)) in extractors.iter().enumerate() {
                    if *op == AggregateOp::Count {
                        self.cached_accums[i].feed_count();
                    } else if let Some(ext) = extractor
                        && let Some(val) = ext(world, entity)
                    {
                        self.cached_accums[i].feed(val);
                    }
                }
            }
        }

        let values = self
            .cached_accums
            .iter()
            .map(|a| (a.label.clone(), a.finish()))
            .collect();
        Ok(AggregateResult { values })
    }
}

impl fmt::Display for QueryPlanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain())
    }
}

impl fmt::Debug for QueryPlanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryPlanResult")
            .field("root", &self.root)
            .field("has_join_exec", &self.join_exec.is_some())
            .field("has_compiled_for_each", &self.compiled_for_each.is_some())
            .field(
                "has_compiled_for_each_raw",
                &self.compiled_for_each_raw.is_some(),
            )
            .field("has_scratch", &self.scratch.is_some())
            .field("opts", &self.opts)
            .field("warnings", &self.warnings)
            .field("last_read_tick", &self.last_read_tick)
            .field("world_id", &self.world_id)
            .field("aggregate_count", &self.aggregate_exprs.len())
            .field("has_compiled_agg_scan", &self.compiled_agg_scan.is_some())
            .field(
                "has_compiled_agg_scan_raw",
                &self.compiled_agg_scan_raw.is_some(),
            )
            .field(
                "cached_batch_extractors_len",
                &self.cached_batch_extractors.len(),
            )
            .field(
                "cached_batch_extractors_raw_len",
                &self.cached_batch_extractors_raw.len(),
            )
            .field("cached_accums_len", &self.cached_accums.len())
            .field("row_indices_cap", &self.row_indices.capacity())
            .field("has_scan_required", &self.scan_required.is_some())
            .field("scan_column_filters", &self.scan_column_filters.len())
            .field("scan_changed", &self.scan_changed)
            .field(
                "column_filter_mask_cap",
                &self.column_filter_mask.capacity(),
            )
            .field("filtered_entities_cap", &self.filtered_entities.capacity())
            .finish()
    }
}

impl QueryPlanResult {
    /// Validate this plan against cardinality constraints.
    ///
    /// Returns a list of constraint violations (empty = plan is valid).
    pub fn validate_constraints(
        &self,
        constraints: &[(&str, CardinalityConstraint)],
    ) -> Vec<String> {
        let mut violations = Vec::new();
        let est = self.root().estimated_rows();
        for (name, constraint) in constraints {
            if !constraint.satisfied_by(est) {
                violations.push(format!(
                    "constraint `{name}` violated: estimated {est:.0} rows, expected {constraint:?}"
                ));
            }
        }
        violations
    }

    /// Compare this plan with another and return a reference to the cheaper one.
    pub fn cheaper<'a>(&'a self, other: &'a Self) -> &'a Self {
        if self.cost().total() <= other.cost().total() {
            self
        } else {
            other
        }
    }
}
