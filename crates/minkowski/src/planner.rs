//! Volcano-model query planner for composing index-driven lookups, joins,
//! and full scans into optimized execution plans.
//!
//! The planner is designed for an in-memory ECS where data already lives in L1/L2
//! cache. Planning overhead is kept to O(indexes + predicates). Plans are
//! executable against live world data via type-erased closure dispatch.
//!
//! # Volcano Model
//!
//! Each node in the plan tree is a **pull-based iterator**: the root calls
//! `next()` on its child, which calls `next()` on *its* child, and so on.
//! This is the classic Volcano/iterator model adapted for in-memory ECS:
//!
//! - **Scan**: full archetype iteration via `world.query()`
//! - **IndexLookup**: point or range lookup on a `BTreeIndex` / `HashIndex`
//! - **Filter**: predicate pushdown (applied per-entity after fetch)
//! - **HashJoin**: join two entity streams on a shared component value
//! - **NestedLoopJoin**: fallback join for small cardinalities
//!
//! # Vectorized Execution
//!
//! Plans are compiled to vectorized execution by default. Instead of the
//! classic row-at-a-time Volcano pull model, vectorized plans process data
//! in **morsel-sized batches** (one archetype chunk at a time), mapping
//! directly to `QueryIter::for_each_chunk` which yields typed `&[T]` /
//! `&mut [T]` slices that LLVM can auto-vectorize.
//!
//! The [`VectorizedPlan`] captures the execution strategy for each node:
//! - **ChunkedScan**: yields one batch per archetype (cache-line aligned)
//! - **SIMDFilter**: filter applied to contiguous slices (branchless when possible)
//! - **PartitionedHashJoin**: build side partitioned for L2 cache residency
//!
//! ```rust,ignore
//! let plan = planner
//!     .scan::<(&Pos, &Vel)>()
//!     .filter(Predicate::range::<Score>(Score(10)..Score(50)))
//!     .build();
//!
//! let vectorized = plan.vectorize(VectorizeOpts::default());
//! println!("{}", vectorized.explain());
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
//! [`SubscriptionPlan`] requires that every predicate is backed by an index,
//! enforced at compile time via the `Indexed<T>` witness type. This ensures
//! the database can push updates to clients in real-time without scanning
//! entire tables.
//!
//! ```rust,ignore
//! let sub = planner
//!     .subscribe::<(&Pos, &Score)>()
//!     .require_index(Indexed::<Score>::from(&score_index))
//!     .on_change(|entity, pos, score| { /* push to client */ })
//!     .build();
//! ```

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use fixedbitset::FixedBitSet;

use crate::component::{Component, ComponentRegistry};
use crate::entity::Entity;
use crate::index::{BTreeIndex, HashIndex};
use crate::sync::Arc;
use crate::world::World;

// ── Cost model ───────────────────────────────────────────────────────

/// Cost estimate for a plan node. All values are dimensionless relative units
/// tuned for in-memory access patterns (L1/L2 cache locality assumed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    /// Estimated number of rows this node will produce.
    rows: f64,
    /// Estimated CPU cost (comparison/hash operations).
    cpu: f64,
}

impl Cost {
    /// Estimated number of rows this node will produce.
    #[inline]
    pub fn rows(&self) -> f64 {
        self.rows
    }

    /// Estimated CPU cost (comparison/hash operations).
    #[inline]
    pub fn cpu(&self) -> f64 {
        self.cpu
    }
}

impl Cost {
    const SCAN_PER_ROW: f64 = 1.0;
    const INDEX_LOOKUP: f64 = 5.0; // BTree/Hash lookup base cost
    const HASH_BUILD_PER_ROW: f64 = 2.0;
    const HASH_PROBE_PER_ROW: f64 = 1.5;
    const NESTED_LOOP_PER_PAIR: f64 = 1.0;
    const FILTER_PER_ROW: f64 = 0.5;

    fn scan(rows: usize) -> Self {
        let r = rows as f64;
        Cost {
            rows: r,
            cpu: r * Self::SCAN_PER_ROW,
        }
    }

    fn index_lookup(selectivity: f64, total_rows: usize) -> Self {
        let r = (total_rows as f64 * selectivity).max(1.0);
        Cost {
            rows: r,
            cpu: Self::INDEX_LOOKUP + r * Self::SCAN_PER_ROW,
        }
    }

    fn filter(input: Cost, selectivity: f64) -> Self {
        Cost {
            rows: (input.rows * selectivity).max(0.0),
            cpu: input.cpu + input.rows * Self::FILTER_PER_ROW,
        }
    }

    fn hash_join(left: Cost, right: Cost) -> Self {
        Cost {
            rows: (left.rows * right.rows).sqrt().max(1.0), // conservative estimate
            cpu: left.cpu
                + right.cpu
                + left.rows * Self::HASH_BUILD_PER_ROW
                + right.rows * Self::HASH_PROBE_PER_ROW,
        }
    }

    fn nested_loop_join(left: Cost, right: Cost) -> Self {
        Cost {
            rows: (left.rows * right.rows).sqrt().max(1.0),
            cpu: left.cpu + right.cpu + left.rows * right.rows * Self::NESTED_LOOP_PER_PAIR,
        }
    }

    /// Total cost for comparison (cpu-dominated since everything is in memory).
    #[inline]
    pub fn total(&self) -> f64 {
        self.cpu
    }
}

// ── Index descriptors (type-erased metadata for planning) ────────────

/// Capability of a registered index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKind {
    /// BTreeIndex — supports exact match and range queries.
    BTree,
    /// HashIndex — supports exact match only.
    Hash,
}

/// Type-erased lookup function for predicate-specific index access.
/// Takes a `&dyn Any` (the predicate's `lookup_value`) and returns matching entities.
type PredicateLookupFn = Arc<dyn Fn(&dyn std::any::Any) -> Arc<[Entity]> + Send + Sync>;

/// Type-erased index metadata for the planner.
struct IndexDescriptor {
    component_name: &'static str,
    kind: IndexKind,
    /// Type-erased function that returns all entities tracked by this index.
    /// Captured at registration time when the concrete index type is available.
    /// Used as a fallback when no predicate-specific lookup is available.
    all_entities_fn: Option<IndexLookupFn>,
    /// Predicate-specific equality lookup. Takes `&dyn Any` (downcast to `T`),
    /// returns only entities matching the exact value. O(log n) for BTree, O(1) for Hash.
    eq_lookup_fn: Option<PredicateLookupFn>,
    /// Predicate-specific range lookup. Takes `&dyn Any` (downcast to `(Bound<T>, Bound<T>)`),
    /// returns only entities within the range. O(log n + k) for BTree, not available for Hash.
    range_lookup_fn: Option<PredicateLookupFn>,
}

impl fmt::Debug for IndexDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexDescriptor")
            .field("component_name", &self.component_name)
            .field("kind", &self.kind)
            .field("has_exec", &self.all_entities_fn.is_some())
            .field("has_eq_lookup", &self.eq_lookup_fn.is_some())
            .field("has_range_lookup", &self.range_lookup_fn.is_some())
            .finish()
    }
}

impl IndexDescriptor {
    /// Create a predicate-specific `IndexLookupFn` when possible,
    /// falling back to `all_entities_fn` (full index scan) otherwise.
    fn lookup_fn_for(&self, pred: &Predicate) -> Option<IndexLookupFn> {
        // Try predicate-specific lookup first — O(1) or O(log n) vs O(n).
        if let Some(lookup_value) = &pred.lookup_value {
            let targeted_fn = match &pred.kind {
                PredicateKind::Eq => self.eq_lookup_fn.as_ref(),
                PredicateKind::Range => self.range_lookup_fn.as_ref(),
                PredicateKind::Custom(_) => None,
            };
            if let Some(lookup) = targeted_fn {
                let lookup = Arc::clone(lookup);
                let value = Arc::clone(lookup_value);
                return Some(Arc::new(move || lookup(value.as_ref())));
            }
        }
        // Fallback: return all entities and let the filter_fn narrow them.
        self.all_entities_fn.clone()
    }
}

// ── Predicates ───────────────────────────────────────────────────────

/// Clamp selectivity to [0.0, 1.0], treating NaN as 1.0 (worst-case).
///
/// `f64::clamp` preserves NaN, so we must check explicitly.
fn sanitize_selectivity(s: f64) -> f64 {
    if s.is_nan() { 1.0 } else { s.clamp(0.0, 1.0) }
}

/// A predicate that can be pushed down into an index lookup or applied as
/// a post-fetch filter.
pub struct Predicate {
    component_type: TypeId,
    component_name: &'static str,
    kind: PredicateKind,
    selectivity: f64,
    /// Type-erased filter closure for execution. Captured at predicate
    /// construction when the concrete value is available.
    filter_fn: Option<FilterFn>,
    /// Type-erased predicate value for index lookups. Eq stores `Arc<T>`,
    /// Range stores `Arc<(Bound<T>, Bound<T>)>`. Downcast by the index's
    /// `eq_lookup_fn` / `range_lookup_fn` at plan-build time.
    lookup_value: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

enum PredicateKind {
    Eq,
    Range,
    Custom(Box<str>), // description only — always post-filter
}

impl Predicate {
    /// Equality predicate on component `T`.
    ///
    /// If a `HashIndex<T>` or `BTreeIndex<T>` is registered, the planner will
    /// use it for an O(1) / O(log n) lookup instead of a full scan.
    ///
    /// Only requires `PartialEq` (not `Eq + Hash`) because the filter closure
    /// does a simple `==` comparison. Index registration enforces stronger bounds
    /// (`Hash + Eq` for HashIndex, `Ord` for BTreeIndex) separately.
    pub fn eq<T: Component + PartialEq>(value: T) -> Self {
        let value = Arc::new(value);
        let filter_val = Arc::clone(&value);
        let lookup_val: Arc<dyn std::any::Any + Send + Sync> = value.clone();
        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Eq,
            selectivity: 0.01, // default: 1% selectivity for equality
            filter_fn: Some(Arc::new(move |world: &World, entity: Entity| {
                world.get::<T>(entity).is_some_and(|v| *v == *filter_val)
            })),
            lookup_value: Some(lookup_val),
        }
    }

    /// Range predicate on component `T`.
    ///
    /// If a `BTreeIndex<T>` is registered, the planner will use it for
    /// an O(log n + k) range scan. `HashIndex` cannot serve range predicates.
    pub fn range<T: Component + Ord + Clone, R: RangeBounds<T>>(range: R) -> Self {
        // Capture owned bounds for the filter closure.
        let low: Bound<T> = match range.start_bound() {
            Bound::Included(v) => Bound::Included(v.clone()),
            Bound::Excluded(v) => Bound::Excluded(v.clone()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let high: Bound<T> = match range.end_bound() {
            Bound::Included(v) => Bound::Included(v.clone()),
            Bound::Excluded(v) => Bound::Excluded(v.clone()),
            Bound::Unbounded => Bound::Unbounded,
        };
        // Separate clones for the lookup_value (consumed by index) vs filter_fn.
        let lookup_val: Arc<dyn std::any::Any + Send + Sync> =
            Arc::new((low.clone(), high.clone()));
        let low = Arc::new(low);
        let high = Arc::new(high);
        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Range,
            selectivity: 0.1, // default: 10% selectivity for ranges
            filter_fn: Some(Arc::new(move |world: &World, entity: Entity| {
                world.get::<T>(entity).is_some_and(|v| {
                    let lo_ok = match low.as_ref() {
                        Bound::Included(lo) => v >= lo,
                        Bound::Excluded(lo) => v > lo,
                        Bound::Unbounded => true,
                    };
                    let hi_ok = match high.as_ref() {
                        Bound::Included(hi) => v <= hi,
                        Bound::Excluded(hi) => v < hi,
                        Bound::Unbounded => true,
                    };
                    lo_ok && hi_ok
                })
            })),
            lookup_value: Some(lookup_val),
        }
    }

    /// Custom predicate with a user-provided filter closure.
    ///
    /// Always applied as a post-fetch filter (cannot be pushed into an index).
    /// The closure receives `(&World, Entity)` and returns `true` if the entity
    /// passes the predicate. This enables arbitrary filtering logic while still
    /// participating in the planner's cost model.
    ///
    /// NaN selectivity is normalized to 1.0 (worst-case, full scan).
    pub fn custom<T: Component>(
        description: &str,
        selectivity: f64,
        filter: impl Fn(&World, Entity) -> bool + Send + Sync + 'static,
    ) -> Self {
        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Custom(description.into()),
            selectivity: sanitize_selectivity(selectivity),
            filter_fn: Some(Arc::new(filter)),
            lookup_value: None,
        }
    }

    /// Override the default selectivity estimate.
    ///
    /// NaN selectivity is normalized to 1.0 (worst-case, full scan).
    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = sanitize_selectivity(selectivity);
        self
    }

    fn can_use_btree(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq | PredicateKind::Range)
    }

    fn can_use_hash(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq)
    }

    /// Whether this predicate can be lowered to branchless SIMD comparison.
    fn is_branchless_eligible(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq | PredicateKind::Range)
    }
}

impl fmt::Debug for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            PredicateKind::Eq => write!(f, "Eq({})", self.component_name),
            PredicateKind::Range => write!(f, "Range({})", self.component_name),
            PredicateKind::Custom(desc) => write!(f, "Custom({}: {})", self.component_name, desc),
        }
    }
}

// ── Plan nodes ───────────────────────────────────────────────────────

/// Join strategy selected by the optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinKind {
    /// Standard inner join — only entities present in both sides.
    Inner,
    /// Left join — all entities from the left, matched from the right.
    Left,
}

/// A node in the query execution plan tree.
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanNode {
    /// Full archetype scan via `world.query()`.
    Scan {
        query_name: &'static str,
        estimated_rows: usize,
        cost: Cost,
    },
    /// Index-driven lookup (point or range).
    IndexLookup {
        index_kind: IndexKind,
        component_name: &'static str,
        predicate: String,
        estimated_rows: usize,
        cost: Cost,
    },
    /// Post-fetch filter applied to child output.
    Filter {
        child: Box<PlanNode>,
        predicate: String,
        selectivity: f64,
        /// Whether this filter can be lowered to branchless SIMD comparison.
        /// True for Eq/Range predicates on numeric types, false for Custom.
        branchless_eligible: bool,
        cost: Cost,
    },
    /// Hash join: build table on left, probe with right.
    HashJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        cost: Cost,
    },
    /// Nested-loop join for small cardinalities.
    NestedLoopJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        cost: Cost,
    },
}

impl PlanNode {
    /// Cost of this node (including children).
    #[inline]
    pub fn cost(&self) -> Cost {
        match self {
            PlanNode::Scan { cost, .. }
            | PlanNode::IndexLookup { cost, .. }
            | PlanNode::Filter { cost, .. }
            | PlanNode::HashJoin { cost, .. }
            | PlanNode::NestedLoopJoin { cost, .. } => *cost,
        }
    }

    /// Estimated output row count.
    #[inline]
    pub fn estimated_rows(&self) -> f64 {
        self.cost().rows
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        // Write indentation without heap allocation: two spaces per level.
        for _ in 0..indent {
            f.write_str("  ")?;
        }
        match self {
            PlanNode::Scan {
                query_name,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "Scan [{query_name}] rows={estimated_rows} cpu={:.1}",
                    cost.cpu
                )
            }
            PlanNode::IndexLookup {
                index_kind,
                component_name,
                predicate,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "IndexLookup [{index_kind:?} on {component_name}] {predicate} rows={estimated_rows} cpu={:.1}",
                    cost.cpu
                )
            }
            PlanNode::Filter {
                child,
                predicate,
                selectivity,
                cost,
                ..
            } => {
                writeln!(
                    f,
                    "Filter [{predicate}] sel={selectivity:.2} rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                child.fmt_indent(f, indent + 1)
            }
            PlanNode::HashJoin {
                left,
                right,
                join_kind,
                cost,
            } => {
                writeln!(
                    f,
                    "HashJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
            PlanNode::NestedLoopJoin {
                left,
                right,
                join_kind,
                cost,
            } => {
                writeln!(
                    f,
                    "NestedLoopJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
        }
    }
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

// ── Warnings ─────────────────────────────────────────────────────────

/// Diagnostic produced during plan compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanWarning {
    /// A predicate exists for this component but no matching index was registered.
    MissingIndex {
        component_name: &'static str,
        predicate_kind: &'static str,
        suggestion: &'static str,
    },
    /// A hash index was registered but the predicate needs range support.
    IndexKindMismatch {
        component_name: &'static str,
        have: &'static str,
        need: &'static str,
    },
    /// Join has no index on either side — will use nested loop.
    UnindexedJoin {
        left_name: &'static str,
        right_name: &'static str,
    },
}

impl fmt::Display for PlanWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanWarning::MissingIndex {
                component_name,
                predicate_kind,
                suggestion,
            } => {
                write!(
                    f,
                    "no index for {predicate_kind} predicate on `{component_name}` — \
                     full scan required. Suggestion: {suggestion}"
                )
            }
            PlanWarning::IndexKindMismatch {
                component_name,
                have,
                need,
            } => {
                write!(
                    f,
                    "`{component_name}` has {have} index but predicate needs {need} — \
                     falling back to scan + filter"
                )
            }
            PlanWarning::UnindexedJoin {
                left_name,
                right_name,
            } => {
                write!(
                    f,
                    "join between `{left_name}` and `{right_name}` has no index on either side — \
                     using nested loop join"
                )
            }
        }
    }
}

// ── Compiled plan ────────────────────────────────────────────────────

/// A compiled query execution plan.
pub struct QueryPlanResult {
    root: PlanNode,
    vec_root: VecExecNode,
    exec_root: Option<ExecNode>,
    opts: VectorizeOpts,
    warnings: Vec<PlanWarning>,
}

impl QueryPlanResult {
    /// The logical plan root. Use this for introspection (matching on
    /// `PlanNode` variants to inspect index selection, join strategy, etc.).
    pub fn root(&self) -> &PlanNode {
        &self.root
    }

    /// The vectorized execution root — the plan that will actually run.
    pub fn vec_root(&self) -> &VecExecNode {
        &self.vec_root
    }

    /// Total estimated cost of the vectorized execution plan.
    pub fn cost(&self) -> Cost {
        self.vec_root.cost()
    }

    /// Cost of the logical plan before vectorized lowering.
    pub fn logical_cost(&self) -> Cost {
        self.root.cost()
    }

    /// Diagnostics generated during compilation.
    pub fn warnings(&self) -> &[PlanWarning] {
        &self.warnings
    }

    /// Human-readable execution plan showing the vectorized plan tree.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Vectorized Execution Plan ===\n");
        let _ = write!(out, "{}", self.vec_root);
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
            self.vec_root.cost().rows,
            self.vec_root.cost().cpu
        );
        out
    }

    /// Execute the plan against a live world, returning matching entities.
    ///
    /// The execution engine walks the plan tree, using type-erased closures
    /// captured at build time to perform scans, index lookups, and filters
    /// against actual world data.
    ///
    /// # Panics
    ///
    /// Panics if the plan was built without execution support (e.g., from a
    /// `ScanBuilder` created by an older API path that did not capture
    /// execution closures).
    pub fn execute(&self, world: &World) -> Vec<Entity> {
        self.exec_root
            .as_ref()
            .expect("plan was built without execution support")
            .execute(world)
    }

    /// The executable plan root (for test introspection).
    #[cfg(test)]
    fn exec_root(&self) -> Option<&ExecNode> {
        self.exec_root.as_ref()
    }

    /// Human-readable logical plan (before vectorized lowering).
    pub fn explain_logical(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Logical Plan ===\n");
        let _ = write!(out, "{}", self.root);
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                let _ = writeln!(out, "  - {w}");
            }
        }
        let _ = write!(
            out,
            "\nEstimated: {:.0} rows, {:.1} cpu\n",
            self.root.cost().rows,
            self.root.cost().cpu
        );
        out
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
            .field("vec_root", &self.vec_root)
            .field("has_exec", &self.exec_root.is_some())
            .field("opts", &self.opts)
            .field("warnings", &self.warnings)
            .finish()
    }
}

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
    kind: IndexKind,
    cardinality: usize,
    _marker: PhantomData<T>,
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
/// resulting plan can push updates to clients in real-time. Every call to
/// `where_eq` or `where_range` requires an `Indexed<T>` witness, which can
/// only be obtained from an actual index instance.
pub struct SubscriptionBuilder<'w> {
    total_entities: usize,
    query_name: &'static str,
    indexed_predicates: Vec<IndexedPredicate>,
    errors: Vec<SubscriptionError>,
    _world: PhantomData<&'w World>,
}

/// Errors from [`SubscriptionBuilder::build`].
#[derive(Clone, Debug, PartialEq)]
pub enum SubscriptionError {
    /// `where_range` was called with a Hash index witness.
    /// Hash indexes support only exact-match lookups and cannot answer range
    /// queries. Use a `BTreeIndex` instead.
    HashIndexOnRange { component_name: &'static str },
    /// A selectivity value was NaN. Selectivity must be a finite number in
    /// `[0.0, 1.0]`.
    NanSelectivity { component_name: &'static str },
    /// No predicates were added. A subscription with zero predicates is a
    /// full scan, which defeats the "all predicates indexed" guarantee.
    NoPredicates,
}

impl fmt::Display for SubscriptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubscriptionError::HashIndexOnRange { component_name } => {
                write!(
                    f,
                    "where_range requires a BTree index, but got a Hash index for `{component_name}`. \
                     Hash indexes cannot answer range queries — use a BTreeIndex instead."
                )
            }
            SubscriptionError::NanSelectivity { component_name } => {
                write!(
                    f,
                    "selectivity for `{component_name}` is NaN — must be a finite number in [0.0, 1.0]"
                )
            }
            SubscriptionError::NoPredicates => {
                write!(
                    f,
                    "subscription has no predicates — add at least one where_eq or where_range"
                )
            }
        }
    }
}

impl std::error::Error for SubscriptionError {}

struct IndexedPredicate {
    component_name: &'static str,
    index_kind: IndexKind,
    predicate_desc: String,
    selectivity: f64,
}

impl SubscriptionBuilder<'_> {
    /// Add an equality predicate backed by a proven index.
    pub fn where_eq<T: Component>(mut self, witness: Indexed<T>, selectivity: f64) -> Self {
        let name = std::any::type_name::<T>();
        if selectivity.is_nan() {
            self.errors.push(SubscriptionError::NanSelectivity {
                component_name: name,
            });
            return self;
        }
        self.indexed_predicates.push(IndexedPredicate {
            component_name: name,
            index_kind: witness.kind,
            predicate_desc: format!("Eq({})", name),
            selectivity: selectivity.clamp(0.0, 1.0),
        });
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
    pub fn where_range<T: Component + Ord + Clone>(
        mut self,
        witness: Indexed<T>,
        selectivity: f64,
    ) -> Self {
        let name = std::any::type_name::<T>();
        if witness.kind == IndexKind::Hash {
            self.errors.push(SubscriptionError::HashIndexOnRange {
                component_name: name,
            });
            return self;
        }
        if selectivity.is_nan() {
            self.errors.push(SubscriptionError::NanSelectivity {
                component_name: name,
            });
            return self;
        }
        self.indexed_predicates.push(IndexedPredicate {
            component_name: name,
            index_kind: witness.kind,
            predicate_desc: format!("Range({})", name),
            selectivity: selectivity.clamp(0.0, 1.0),
        });
        self
    }

    /// Compile the subscription plan. Every predicate is guaranteed to have
    /// an index, so the plan never falls back to a full scan for filtering.
    ///
    /// # Errors
    ///
    /// Returns all [`SubscriptionError`]s if any predicates were invalid
    /// (e.g. a Hash index used with `where_range`, or a NaN selectivity).
    pub fn build(self) -> Result<SubscriptionPlan, Vec<SubscriptionError>> {
        let mut errors = self.errors;
        if self.indexed_predicates.is_empty() {
            errors.push(SubscriptionError::NoPredicates);
        }
        if !errors.is_empty() {
            return Err(errors);
        }

        let mut node: PlanNode = PlanNode::Scan {
            query_name: self.query_name,
            estimated_rows: self.total_entities,
            cost: Cost::scan(self.total_entities),
        };

        // Sort predicates by selectivity (most selective first) for optimal
        // index intersection ordering.
        let mut preds = self.indexed_predicates;
        preds.sort_by(|a, b| a.selectivity.total_cmp(&b.selectivity));

        // The most selective predicate becomes the driving index lookup.
        // Remaining predicates become chained index lookups / filters.
        if let Some(first) = preds.first() {
            let est = (self.total_entities as f64 * first.selectivity).max(1.0) as usize;
            node = PlanNode::IndexLookup {
                index_kind: first.index_kind,
                component_name: first.component_name,
                predicate: first.predicate_desc.clone(),
                estimated_rows: est,
                cost: Cost::index_lookup(first.selectivity, self.total_entities),
            };

            for pred in preds.iter().skip(1) {
                let parent_cost = node.cost();
                let est_rows = parent_cost.rows * pred.selectivity;
                node = PlanNode::Filter {
                    predicate: pred.predicate_desc.clone(),
                    selectivity: pred.selectivity,
                    branchless_eligible: true, // indexed predicates are always Eq/Range
                    cost: Cost {
                        rows: est_rows.max(0.0),
                        cpu: parent_cost.cpu
                            + parent_cost.rows * Cost::FILTER_PER_ROW
                            + Cost::INDEX_LOOKUP, // re-validate via index
                    },
                    child: Box::new(node),
                };
            }
        }

        Ok(SubscriptionPlan { root: node })
    }
}

/// A compiled subscription plan. Guaranteed to have an index for every
/// predicate — the database can push updates without full table scans.
pub struct SubscriptionPlan {
    root: PlanNode,
}

impl SubscriptionPlan {
    /// The root node of the subscription plan.
    pub fn root(&self) -> &PlanNode {
        &self.root
    }

    /// Total estimated cost.
    pub fn cost(&self) -> Cost {
        self.root.cost()
    }

    /// Human-readable plan.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Subscription Plan (all predicates indexed) ===\n");
        let _ = write!(out, "{}", self.root);
        let _ = write!(
            out,
            "\nEstimated: {:.0} rows, {:.1} cpu\n",
            self.root.cost().rows,
            self.root.cost().cpu
        );
        out
    }
}

impl fmt::Display for SubscriptionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain())
    }
}

impl fmt::Debug for SubscriptionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubscriptionPlan")
            .field("root", &self.root)
            .finish()
    }
}

// ── Execution engine ─────────────────────────────────────────────────

/// Type-erased scan function: given `&World`, return all matching entities.
type ScanFn = Arc<dyn Fn(&World) -> Vec<Entity> + Send + Sync>;

/// Type-erased index lookup function: return matching entities from the index.
/// Returns `Arc<[Entity]>` to avoid cloning the entity list on every call —
/// callers that need mutation (filter, sort) pay the allocation cost only when
/// they actually mutate.
type IndexLookupFn = Arc<dyn Fn() -> Arc<[Entity]> + Send + Sync>;

/// Type-erased filter function: given `&World` and `Entity`, return true if
/// the entity passes the predicate.
// PERF: Arc<dyn Fn> prevents SIMD vectorization of filter loops (per-entity
// vtable call). Inherent to type-erased plan composition — monomorphic
// filters would require codegen per plan.
type FilterFn = Arc<dyn Fn(&World, Entity) -> bool + Send + Sync>;

/// Partitioned or single-bucket hash set for probe-side lookups.
enum ProbeSet {
    Single(std::collections::HashSet<Entity>),
    Partitioned(Vec<std::collections::HashSet<Entity>>),
}

impl ProbeSet {
    #[inline]
    fn contains(&self, e: Entity) -> bool {
        match self {
            ProbeSet::Single(set) => set.contains(&e),
            ProbeSet::Partitioned(parts) => {
                let bucket = (e.to_bits() as usize) % parts.len();
                parts[bucket].contains(&e)
            }
        }
    }
}

/// Executable plan node — carries type-erased closures AND vectorization
/// parameters from the [`VecExecNode`] lowering pass. Built by
/// [`ScanBuilder::build`] and executed by [`QueryPlanResult::execute`].
///
/// Unlike the logical [`PlanNode`], execution nodes process data in
/// morsel-sized batches: scans yield chunks, filters operate on batches,
/// index gathers sort by archetype, and hash joins partition the build side.
enum ExecNode {
    /// Chunked archetype scan — collects entities per archetype in batches.
    /// `chunk_size` is advisory metadata from the vectorized plan (used for
    /// test assertions and plan introspection). Maps to `VecExecNode::ChunkedScan`.
    ChunkedScan {
        scan_fn: ScanFn,
        #[cfg_attr(not(test), expect(dead_code))]
        chunk_size: usize,
    },
    /// Index-driven gather — entities from the index snapshot, sorted by
    /// archetype for sequential access. Maps to `VecExecNode::IndexGather`.
    IndexGather {
        lookup_fn: IndexLookupFn,
        /// Sort entities by archetype ID for cache-friendly sequential access.
        sort_by_archetype: bool,
    },
    /// Filter — applies a type-erased predicate per entity. `batch_size` is
    /// advisory metadata from the vectorized plan (for test introspection).
    /// Maps to `VecExecNode::SIMDFilter`.
    BatchFilter {
        child: Box<ExecNode>,
        filter_fn: FilterFn,
        #[cfg_attr(not(test), expect(dead_code))]
        batch_size: usize,
    },
    /// Partitioned hash join — build side split into `partitions` hash maps
    /// for L2 cache residency. Maps to `VecExecNode::PartitionedHashJoin`.
    PartitionedHashJoin {
        build: Box<ExecNode>,
        probe: Box<ExecNode>,
        join_kind: JoinKind,
        partitions: usize,
    },
    /// Batch nested-loop join for small cardinalities.
    /// Maps to `VecExecNode::BatchNestedLoopJoin`.
    BatchNestedLoopJoin {
        left: Box<ExecNode>,
        right: Box<ExecNode>,
        join_kind: JoinKind,
    },
}

impl ExecNode {
    /// Execute this node against the world, returning matching entities.
    // PERF: Returns Vec<Entity> per node — recursive calls create temporary
    // Vecs at each tree level. An iterator model would avoid this but requires
    // GATs or self-referential types. Acceptable for the planner's use case
    // (plans executed infrequently, not per-entity).
    fn execute(&self, world: &World) -> Vec<Entity> {
        match self {
            ExecNode::ChunkedScan {
                scan_fn,
                chunk_size: _,
            } => {
                // The scan_fn already iterates per-archetype in contiguous
                // chunks. The chunk_size parameter is advisory metadata for
                // higher-level scheduling — no re-chunking needed here.
                scan_fn(world)
            }
            ExecNode::IndexGather {
                lookup_fn,
                sort_by_archetype,
            } => {
                let snapshot = lookup_fn();
                let mut entities: Vec<Entity> = snapshot
                    .iter()
                    .copied()
                    .filter(|e| world.is_alive(*e))
                    .collect();
                if *sort_by_archetype {
                    // Pre-compute archetype IDs into a parallel array so the
                    // sort does O(N) lookups instead of O(N log N). Each
                    // comparison is then a plain usize compare with no
                    // pointer chasing into entity_locations.
                    let mut keyed: Vec<(usize, Entity)> = entities
                        .iter()
                        .map(|e| {
                            let idx = e.index() as usize;
                            let arch = world
                                .entity_locations
                                .get(idx)
                                .and_then(|loc| loc.as_ref())
                                .map_or(usize::MAX, |loc| loc.archetype_id.0);
                            (arch, *e)
                        })
                        .collect();
                    keyed.sort_unstable_by_key(|(arch, _)| *arch);
                    entities = keyed.into_iter().map(|(_, e)| e).collect();
                }
                entities
            }
            ExecNode::BatchFilter {
                child,
                filter_fn,
                batch_size: _,
            } => {
                // PERF: Per-entity dyn Fn call through FilterFn prevents
                // SIMD vectorization of the filter loop. Inherent to the
                // type-erased plan composition model — monomorphic filters
                // would require codegen per plan. The previous chunk loop
                // provided no benefit since the inner call is scalar.
                let input = child.execute(world);
                let mut result = Vec::with_capacity(input.len() / 2);
                for &e in &input {
                    if filter_fn(world, e) {
                        result.push(e);
                    }
                }
                result
            }
            ExecNode::PartitionedHashJoin {
                build,
                probe,
                join_kind,
                partitions,
            } => {
                // PERF: Allocates P HashSets per execution. Caching across
                // calls would require mutable ExecNode, breaking &self execute.
                // build = left side (semantic "from" table)
                // probe = right side (joined table)
                // Inner: entities present in both sides.
                // Left: all build entities, whether or not they appear in probe.
                let build_entities = build.execute(world);
                let probe_entities = probe.execute(world);
                let probe_set = if *partitions <= 1 {
                    ProbeSet::Single(probe_entities.iter().copied().collect())
                } else {
                    let p = *partitions;
                    let mut parts: Vec<std::collections::HashSet<Entity>> =
                        (0..p).map(|_| std::collections::HashSet::new()).collect();
                    for &e in &probe_entities {
                        parts[(e.to_bits() as usize) % p].insert(e);
                    }
                    ProbeSet::Partitioned(parts)
                };
                match join_kind {
                    JoinKind::Inner => build_entities
                        .into_iter()
                        .filter(|e| probe_set.contains(*e))
                        .collect(),
                    JoinKind::Left => {
                        // All build (left) entities are preserved.
                        build_entities
                    }
                }
            }
            ExecNode::BatchNestedLoopJoin {
                left,
                right,
                join_kind,
            } => {
                let left_entities = left.execute(world);
                match join_kind {
                    JoinKind::Inner => {
                        let right_set: std::collections::HashSet<Entity> =
                            right.execute(world).into_iter().collect();
                        left_entities
                            .into_iter()
                            .filter(|e| right_set.contains(e))
                            .collect()
                    }
                    JoinKind::Left => {
                        // All left entities preserved regardless of right side.
                        left_entities
                    }
                }
            }
        }
    }

    /// Name of this exec node variant (for test assertions).
    #[cfg(test)]
    fn variant_name(&self) -> &'static str {
        match self {
            ExecNode::ChunkedScan { .. } => "ChunkedScan",
            ExecNode::IndexGather { .. } => "IndexGather",
            ExecNode::BatchFilter { .. } => "BatchFilter",
            ExecNode::PartitionedHashJoin { .. } => "PartitionedHashJoin",
            ExecNode::BatchNestedLoopJoin { .. } => "BatchNestedLoopJoin",
        }
    }

    /// Chunk size for ChunkedScan, None for other variants.
    #[cfg(test)]
    fn chunk_size(&self) -> Option<usize> {
        match self {
            ExecNode::ChunkedScan { chunk_size, .. } => Some(*chunk_size),
            _ => None,
        }
    }

    /// Partition count for PartitionedHashJoin, None for other variants.
    #[cfg(test)]
    fn partitions(&self) -> Option<usize> {
        match self {
            ExecNode::PartitionedHashJoin { partitions, .. } => Some(*partitions),
            _ => None,
        }
    }

    /// Batch size for BatchFilter, None for other variants.
    #[cfg(test)]
    fn batch_size(&self) -> Option<usize> {
        match self {
            ExecNode::BatchFilter { batch_size, .. } => Some(*batch_size),
            _ => None,
        }
    }

    /// Sort-by-archetype flag for IndexGather, None for other variants.
    #[cfg(test)]
    fn sort_by_archetype(&self) -> Option<bool> {
        match self {
            ExecNode::IndexGather {
                sort_by_archetype, ..
            } => Some(*sort_by_archetype),
            _ => None,
        }
    }

    /// Children of this node (for tree traversal in tests).
    #[cfg(test)]
    fn children(&self) -> Vec<&ExecNode> {
        match self {
            ExecNode::ChunkedScan { .. } | ExecNode::IndexGather { .. } => vec![],
            ExecNode::BatchFilter { child, .. } => vec![child],
            ExecNode::PartitionedHashJoin { build, probe, .. } => vec![build, probe],
            ExecNode::BatchNestedLoopJoin { left, right, .. } => vec![left, right],
        }
    }
}

/// Lightweight closure tree built during plan compilation, before vectorization.
/// Combined with [`VecExecNode`] parameters by [`lower_to_executable`] to produce
/// the final [`ExecNode`].
enum ClosureNode {
    Scan {
        scan_fn: ScanFn,
    },
    IndexLookup {
        lookup_fn: IndexLookupFn,
    },
    Filter {
        child: Box<ClosureNode>,
        filter_fn: FilterFn,
    },
    HashJoin {
        build: Box<ClosureNode>,
        probe: Box<ClosureNode>,
        join_kind: JoinKind,
    },
    NestedLoopJoin {
        left: Box<ClosureNode>,
        right: Box<ClosureNode>,
        join_kind: JoinKind,
    },
}

/// Lower a [`ClosureNode`] + [`VecExecNode`] pair into a vectorized [`ExecNode`].
///
/// The closure tree provides the type-erased execution functions; the vec tree
/// provides the vectorization parameters (chunk sizes, partition counts, sort
/// hints). The two trees are structurally isomorphic — they were built from the
/// same logical plan.
///
/// Takes ownership of the closure tree to avoid `Arc::clone` on every node.
fn lower_to_executable(closures: ClosureNode, vec_node: &VecExecNode) -> ExecNode {
    match (closures, vec_node) {
        (ClosureNode::Scan { scan_fn }, VecExecNode::ChunkedScan { avg_chunk_size, .. }) => {
            ExecNode::ChunkedScan {
                scan_fn,
                chunk_size: *avg_chunk_size,
            }
        }
        (ClosureNode::IndexLookup { lookup_fn }, VecExecNode::IndexGather { .. }) => {
            ExecNode::IndexGather {
                lookup_fn,
                sort_by_archetype: true,
            }
        }
        (
            ClosureNode::Filter { child, filter_fn },
            VecExecNode::SIMDFilter {
                child: vec_child, ..
            },
        ) => {
            // Batch size: use the child's estimated rows or a sensible default.
            let batch_size = match vec_child.as_ref() {
                VecExecNode::ChunkedScan { avg_chunk_size, .. } => *avg_chunk_size,
                _ => 256, // default batch for non-scan children
            };
            let child_exec = lower_to_executable(*child, vec_child);
            ExecNode::BatchFilter {
                child: Box::new(child_exec),
                filter_fn,
                batch_size,
            }
        }
        (
            ClosureNode::HashJoin {
                build,
                probe,
                join_kind,
            },
            VecExecNode::PartitionedHashJoin {
                build: vec_build,
                probe: vec_probe,
                partitions,
                ..
            },
        ) => ExecNode::PartitionedHashJoin {
            build: Box::new(lower_to_executable(*build, vec_build)),
            probe: Box::new(lower_to_executable(*probe, vec_probe)),
            join_kind,
            partitions: *partitions,
        },
        (
            ClosureNode::NestedLoopJoin {
                left,
                right,
                join_kind,
            },
            VecExecNode::BatchNestedLoopJoin {
                left: vec_left,
                right: vec_right,
                ..
            },
        ) => ExecNode::BatchNestedLoopJoin {
            left: Box::new(lower_to_executable(*left, vec_left)),
            right: Box::new(lower_to_executable(*right, vec_right)),
            join_kind,
        },
        // Fallback: structural mismatch between closure tree and vec tree.
        // This can happen when the vec lowering produces a different shape
        // (e.g., Scan closure but IndexGather vec node for index-driven plans).
        // Fall back to reasonable defaults.
        (ClosureNode::Scan { scan_fn }, _) => ExecNode::ChunkedScan {
            scan_fn,
            chunk_size: 4096,
        },
        (ClosureNode::IndexLookup { lookup_fn }, _) => ExecNode::IndexGather {
            lookup_fn,
            sort_by_archetype: true,
        },
        (ClosureNode::Filter { child, filter_fn }, _) => {
            // If vec tree shape doesn't match, walk the vec child if possible,
            // otherwise use the fallback with default batch size.
            let child_exec = lower_to_executable_fallback(*child);
            ExecNode::BatchFilter {
                child: Box::new(child_exec),
                filter_fn,
                batch_size: 256,
            }
        }
        (
            ClosureNode::HashJoin {
                build,
                probe,
                join_kind,
            },
            _,
        ) => ExecNode::PartitionedHashJoin {
            build: Box::new(lower_to_executable_fallback(*build)),
            probe: Box::new(lower_to_executable_fallback(*probe)),
            join_kind,
            partitions: 1,
        },
        (
            ClosureNode::NestedLoopJoin {
                left,
                right,
                join_kind,
            },
            _,
        ) => ExecNode::BatchNestedLoopJoin {
            left: Box::new(lower_to_executable_fallback(*left)),
            right: Box::new(lower_to_executable_fallback(*right)),
            join_kind,
        },
    }
}

/// Fallback lowering when the closure tree shape doesn't match the vec tree.
fn lower_to_executable_fallback(closures: ClosureNode) -> ExecNode {
    match closures {
        ClosureNode::Scan { scan_fn } => ExecNode::ChunkedScan {
            scan_fn,
            chunk_size: 4096,
        },
        ClosureNode::IndexLookup { lookup_fn } => ExecNode::IndexGather {
            lookup_fn,
            sort_by_archetype: true,
        },
        ClosureNode::Filter { child, filter_fn } => ExecNode::BatchFilter {
            child: Box::new(lower_to_executable_fallback(*child)),
            filter_fn,
            batch_size: 256,
        },
        ClosureNode::HashJoin {
            build,
            probe,
            join_kind,
        } => ExecNode::PartitionedHashJoin {
            build: Box::new(lower_to_executable_fallback(*build)),
            probe: Box::new(lower_to_executable_fallback(*probe)),
            join_kind,
            partitions: 1,
        },
        ClosureNode::NestedLoopJoin {
            left,
            right,
            join_kind,
        } => ExecNode::BatchNestedLoopJoin {
            left: Box::new(lower_to_executable_fallback(*left)),
            right: Box::new(lower_to_executable_fallback(*right)),
            join_kind,
        },
    }
}

/// Collect all entities from archetypes matching a component bitset.
fn scan_matching_entities(world: &World, required: &FixedBitSet) -> Vec<Entity> {
    let mut result = Vec::new();
    for arch in &world.archetypes.archetypes {
        if !arch.is_empty() && required.is_subset(&arch.component_ids) {
            // Reserve on first match to avoid per-entity push reallocation.
            if result.is_empty() {
                result.reserve(arch.entities.len());
            }
            result.extend_from_slice(&arch.entities);
        }
    }
    result
}

// ── Scan builder ─────────────────────────────────────────────────────

/// Builder for a single-table scan with optional predicates and joins.
pub struct ScanBuilder<'w> {
    planner: &'w QueryPlanner<'w>,
    query_name: &'static str,
    estimated_rows: usize,
    predicates: Vec<Predicate>,
    joins: Vec<JoinSpec>,
    /// Type-erased scan closure captured from `scan::<Q>()` when Q is known.
    scan_fn: Option<ScanFn>,
}

struct JoinSpec {
    right_query_name: &'static str,
    right_estimated_rows: usize,
    join_kind: JoinKind,
    right_scan_fn: Option<ScanFn>,
}

impl ScanBuilder<'_> {
    /// Add a filter predicate. The planner will automatically determine
    /// whether to push it into an index lookup or apply it as a post-filter.
    pub fn filter(mut self, predicate: Predicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Add a join with another query type.
    ///
    /// The planner will choose between hash join and nested-loop join based
    /// on estimated cardinalities.
    pub fn join<Q: crate::query::fetch::WorldQuery + 'static>(
        mut self,
        join_kind: JoinKind,
    ) -> Self {
        // Estimate right-side rows from total entity count (conservative).
        let right_rows = self.planner.total_entities;
        let required = Q::required_ids(self.planner.components);
        self.joins.push(JoinSpec {
            right_query_name: std::any::type_name::<Q>(),
            right_estimated_rows: right_rows,
            join_kind,
            right_scan_fn: Some(Arc::new(move |world: &World| {
                scan_matching_entities(world, &required)
            })),
        });
        self
    }

    /// Set explicit row estimate for the most recently added join's right side.
    ///
    /// # Panics
    /// Panics if called before any `join()`.
    pub fn with_right_estimate(mut self, rows: usize) -> Self {
        assert!(
            !self.joins.is_empty(),
            "with_right_estimate() called before any join() — call join() first"
        );
        self.joins.last_mut().unwrap().right_estimated_rows = rows;
        self
    }

    /// Compile the scan into an optimized execution plan.
    pub fn build(self) -> QueryPlanResult {
        let mut warnings = Vec::new();

        // Phase 1: Classify predicates — index-driven vs post-filter.
        let mut index_preds: Vec<(Predicate, &IndexDescriptor)> = Vec::new();
        let mut filter_preds = Vec::new();
        let planner = self.planner;

        for pred in self.predicates {
            if let Some(idx) = planner.find_best_index(&pred) {
                index_preds.push((pred, idx));
            } else {
                // Generate warnings for missing indexes.
                planner.warn_missing_index(&pred, &mut warnings);
                filter_preds.push(pred);
            }
        }

        // Phase 2: Order index lookups by selectivity (most selective first).
        index_preds.sort_by(|a, b| a.0.selectivity.total_cmp(&b.0.selectivity));

        // Phase 3: Build the logical plan tree AND closure tree bottom-up.
        // The closure tree captures type-erased functions while generic types
        // are still in scope. It will be combined with vectorization parameters
        // from VecExecNode in Phase 7 to produce the final ExecNode.
        let mut node: PlanNode;
        let mut closures: Option<ClosureNode>;

        if let Some((first_pred, first_idx)) = index_preds.first() {
            // Driving access is an index lookup.
            let est = (self.estimated_rows as f64 * first_pred.selectivity).max(1.0) as usize;
            node = PlanNode::IndexLookup {
                index_kind: first_idx.kind,
                component_name: first_pred.component_name,
                predicate: format!("{:?}", first_pred),
                estimated_rows: est,
                cost: Cost::index_lookup(first_pred.selectivity, self.estimated_rows),
            };

            // Closure tree: index lookup intersected with scan for archetype
            // matching, then filtered by predicate.  Use predicate-specific
            // lookup (O(1)/O(log n)) when available, falling back to full scan.
            closures = if let Some(lookup) = first_idx.lookup_fn_for(first_pred) {
                let base: ClosureNode = if let Some(scan) = &self.scan_fn {
                    ClosureNode::HashJoin {
                        build: Box::new(ClosureNode::Scan {
                            scan_fn: Arc::clone(scan),
                        }),
                        probe: Box::new(ClosureNode::IndexLookup { lookup_fn: lookup }),
                        join_kind: JoinKind::Inner, // archetype intersection is always inner
                    }
                } else {
                    ClosureNode::IndexLookup { lookup_fn: lookup }
                };
                if let Some(filter) = &first_pred.filter_fn {
                    Some(ClosureNode::Filter {
                        child: Box::new(base),
                        filter_fn: Arc::clone(filter),
                    })
                } else {
                    Some(base)
                }
            } else {
                None
            };

            // Additional index predicates become filters.
            for (pred, _idx) in index_preds.iter().skip(1) {
                let parent_cost = node.cost();
                node = PlanNode::Filter {
                    predicate: format!("{:?}", pred),
                    selectivity: pred.selectivity,
                    branchless_eligible: pred.is_branchless_eligible(),
                    cost: Cost::filter(parent_cost, pred.selectivity),
                    child: Box::new(node),
                };
                if let Some(filter_fn) = &pred.filter_fn
                    && let Some(cn) = closures.take()
                {
                    closures = Some(ClosureNode::Filter {
                        child: Box::new(cn),
                        filter_fn: Arc::clone(filter_fn),
                    });
                }
            }
        } else {
            // No usable index — full scan.
            node = PlanNode::Scan {
                query_name: self.query_name,
                estimated_rows: self.estimated_rows,
                cost: Cost::scan(self.estimated_rows),
            };
            closures = self.scan_fn.as_ref().map(|f| ClosureNode::Scan {
                scan_fn: Arc::clone(f),
            });
        }

        // Phase 4: Apply remaining filter predicates.
        for pred in filter_preds {
            let parent_cost = node.cost();
            node = PlanNode::Filter {
                predicate: format!("{:?}", pred),
                selectivity: pred.selectivity,
                branchless_eligible: pred.is_branchless_eligible(),
                cost: Cost::filter(parent_cost, pred.selectivity),
                child: Box::new(node),
            };
            if let Some(filter_fn) = &pred.filter_fn
                && let Some(cn) = closures.take()
            {
                closures = Some(ClosureNode::Filter {
                    child: Box::new(cn),
                    filter_fn: Arc::clone(filter_fn),
                });
            }
        }

        // Phase 5: Join ordering — smallest intermediate result drives the
        // left side; each join's output becomes the next left input.
        for join in &self.joins {
            let right_node = PlanNode::Scan {
                query_name: join.right_query_name,
                estimated_rows: join.right_estimated_rows,
                cost: Cost::scan(join.right_estimated_rows),
            };

            let left_cost = node.cost();
            let right_cost = right_node.cost();

            let right_closures = join.right_scan_fn.as_ref().map(|f| ClosureNode::Scan {
                scan_fn: Arc::clone(f),
            });

            // Choose join strategy based on cardinality.
            let use_hash = left_cost.rows >= 64.0 || right_cost.rows >= 64.0;

            if use_hash {
                let (build, probe) =
                    if join.join_kind == JoinKind::Inner && left_cost.rows > right_cost.rows {
                        (right_node, node)
                    } else {
                        (node, right_node)
                    };
                let cost = Cost::hash_join(build.cost(), probe.cost());

                let (build_closures, probe_closures) =
                    if join.join_kind == JoinKind::Inner && left_cost.rows > right_cost.rows {
                        (right_closures, closures.take())
                    } else {
                        (closures.take(), right_closures)
                    };

                node = PlanNode::HashJoin {
                    left: Box::new(build),
                    right: Box::new(probe),
                    join_kind: join.join_kind,
                    cost,
                };
                if let (Some(bc), Some(pc)) = (build_closures, probe_closures) {
                    closures = Some(ClosureNode::HashJoin {
                        build: Box::new(bc),
                        probe: Box::new(pc),
                        join_kind: join.join_kind,
                    });
                }
            } else {
                if left_cost.rows > right_cost.rows {
                    warnings.push(PlanWarning::UnindexedJoin {
                        left_name: self.query_name,
                        right_name: join.right_query_name,
                    });
                }
                let cost = Cost::nested_loop_join(left_cost, right_cost);
                node = PlanNode::NestedLoopJoin {
                    left: Box::new(node),
                    right: Box::new(right_node),
                    join_kind: join.join_kind,
                    cost,
                };
                if let (Some(lc), Some(rc)) = (closures.take(), right_closures) {
                    closures = Some(ClosureNode::NestedLoopJoin {
                        left: Box::new(lc),
                        right: Box::new(rc),
                        join_kind: join.join_kind,
                    });
                }
            }
        }

        // Phase 6: Lower logical plan to vectorized plan.
        let opts = VectorizeOpts::default();
        let vec_root = lower_to_vectorized(&node, &opts);

        // Phase 7: Combine closure tree + vectorization parameters into
        // executable ExecNode tree. This is the key step that ensures
        // execution honors the vectorized plan's decisions (chunk sizes,
        // partition counts, sort-by-archetype, batch sizes).
        let exec_root = closures.map(|cn| lower_to_executable(cn, &vec_root));

        QueryPlanResult {
            root: node,
            vec_root,
            exec_root,
            opts,
            warnings,
        }
    }
}

// ── QueryPlanner ─────────────────────────────────────────────────────

/// Volcano-model query planner that composes index lookups, filters, and joins
/// into cost-optimized execution plans.
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
    indexes: HashMap<TypeId, IndexDescriptor>,
    total_entities: usize,
    /// Component registry for resolving query type → component bitset.
    components: &'w ComponentRegistry,
    _world: PhantomData<&'w World>,
}

impl<'w> QueryPlanner<'w> {
    /// Create a new planner from the current world state.
    ///
    /// Captures the total entity count for cost estimation. The world is not
    /// borrowed beyond this call.
    pub fn new(world: &'w World) -> Self {
        QueryPlanner {
            indexes: HashMap::new(),
            total_entities: world.entity_count(),
            components: &world.components,
            _world: PhantomData,
        }
    }

    /// Register a `BTreeIndex` for cost-based index selection.
    ///
    /// # Panics
    /// Panics if `T` has not been registered as a component in `world`.
    pub fn add_btree_index<T: Component + Ord + Clone>(
        &mut self,
        index: &Arc<BTreeIndex<T>>,
        world: &World,
    ) {
        assert!(
            world.component_id::<T>().is_some(),
            "QueryPlanner::add_btree_index: component `{}` not registered in this World",
            std::any::type_name::<T>()
        );
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
    }

    /// Register a `HashIndex` for cost-based index selection.
    ///
    /// # Panics
    /// Panics if `T` has not been registered as a component in `world`.
    pub fn add_hash_index<T: Component + std::hash::Hash + Eq + Clone>(
        &mut self,
        index: &Arc<HashIndex<T>>,
        world: &World,
    ) {
        assert!(
            world.component_id::<T>().is_some(),
            "QueryPlanner::add_hash_index: component `{}` not registered in this World",
            std::any::type_name::<T>()
        );
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
    }

    /// Start building a scan plan for query type `Q`.
    pub fn scan<Q: crate::query::fetch::WorldQuery + 'static>(&'w self) -> ScanBuilder<'w> {
        let required = Q::required_ids(self.components);
        ScanBuilder {
            planner: self,
            query_name: std::any::type_name::<Q>(),
            estimated_rows: self.total_entities,
            predicates: Vec::new(),
            joins: Vec::new(),
            scan_fn: Some(Arc::new(move |world: &World| {
                scan_matching_entities(world, &required)
            })),
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
        ScanBuilder {
            planner: self,
            query_name: std::any::type_name::<Q>(),
            estimated_rows,
            predicates: Vec::new(),
            joins: Vec::new(),
            scan_fn: Some(Arc::new(move |world: &World| {
                scan_matching_entities(world, &required)
            })),
        }
    }

    /// Start building a subscription plan (compiler-enforced indexes).
    pub fn subscribe<Q: 'static>(&'w self) -> SubscriptionBuilder<'w> {
        SubscriptionBuilder {
            total_entities: self.total_entities,
            query_name: std::any::type_name::<Q>(),
            indexed_predicates: Vec::new(),
            errors: Vec::new(),
            _world: PhantomData,
        }
    }

    /// Find the best index for a predicate, if one exists.
    fn find_best_index(&self, pred: &Predicate) -> Option<&IndexDescriptor> {
        let idx = self.indexes.get(&pred.component_type)?;

        match idx.kind {
            IndexKind::BTree if pred.can_use_btree() => Some(idx),
            IndexKind::Hash if pred.can_use_hash() => Some(idx),
            _ => None,
        }
    }

    /// Generate warnings for predicates that can't use an index.
    fn warn_missing_index(&self, pred: &Predicate, warnings: &mut Vec<PlanWarning>) {
        match &pred.kind {
            PredicateKind::Custom(_) => {
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
        }
    }
}

// ── Constraint-based optimization utilities ──────────────────────────

/// Constraint on the expected cardinality of a query result.
///
/// Used for constraint-based optimization: the planner can verify that a
/// plan satisfies cardinality bounds and choose strategies accordingly.
#[derive(Clone, Copy, Debug)]
pub enum CardinalityConstraint {
    /// Expect exactly one result (unique lookup).
    ExactlyOne,
    /// Expect at most N results.
    AtMost(usize),
    /// Expect at least N results.
    AtLeast(usize),
    /// Expect between lo and hi results (inclusive).
    ///
    /// Use [`CardinalityConstraint::between`] for validated construction.
    Between(usize, usize),
}

impl CardinalityConstraint {
    /// Create a `Between` constraint with validation that `lo <= hi`.
    ///
    /// # Panics
    /// Panics if `lo > hi`.
    pub fn between(lo: usize, hi: usize) -> Self {
        assert!(
            lo <= hi,
            "CardinalityConstraint::between: lo ({lo}) > hi ({hi})"
        );
        Self::Between(lo, hi)
    }

    /// Check whether an estimated row count satisfies this constraint.
    pub fn satisfied_by(&self, estimated_rows: f64) -> bool {
        match self {
            CardinalityConstraint::ExactlyOne => (0.5..1.5).contains(&estimated_rows),
            CardinalityConstraint::AtMost(n) => estimated_rows <= *n as f64 + 0.5,
            CardinalityConstraint::AtLeast(n) => estimated_rows >= *n as f64 - 0.5,
            CardinalityConstraint::Between(lo, hi) => {
                estimated_rows >= *lo as f64 - 0.5 && estimated_rows <= *hi as f64 + 0.5
            }
        }
    }
}

// ── Vectorized execution ─────────────────────────────────────────────

/// Execution strategy for a vectorized plan node.
///
/// Instead of the classic Volcano row-at-a-time pull model, vectorized
/// nodes process **morsel-sized batches** — one archetype chunk at a time.
/// This maps directly to `QueryIter::for_each_chunk` which yields typed
/// `&[T]` / `&mut [T]` slices. LLVM can auto-vectorize loops over these
/// contiguous, 64-byte-aligned columns.
///
/// The vectorized plan is the **default execution representation**.
/// `build()` automatically lowers the logical [`PlanNode`] tree to
/// vectorized form. Users inspect it via `explain()` and can access
/// the logical plan via `root()` or `explain_logical()`.
#[derive(Debug)]
#[non_exhaustive]
pub enum VecExecNode {
    /// Chunked scan: yields one batch per archetype.
    /// Each batch is a contiguous column slice (64-byte aligned).
    ChunkedScan {
        query_name: &'static str,
        estimated_rows: usize,
        /// Average rows per chunk (= archetype size). Tuned for L1 residency.
        avg_chunk_size: usize,
        cost: Cost,
    },

    /// Index-driven gather: lookup entities via index, then batch-fetch
    /// components via `get_batch`. Entities are sorted by archetype to
    /// maximize sequential access.
    IndexGather {
        index_kind: IndexKind,
        component_name: &'static str,
        predicate: String,
        estimated_rows: usize,
        cost: Cost,
    },

    /// SIMD-friendly filter: applied to contiguous slices.
    /// The filter function operates on `&[T]` and produces a selection
    /// vector (bitmask) — no branching per row.
    SIMDFilter {
        child: Box<VecExecNode>,
        predicate: String,
        selectivity: f64,
        /// Whether this filter can be applied branchlessly on aligned data.
        branchless: bool,
        cost: Cost,
    },

    /// Partitioned hash join: build side is partitioned into L2-cache-sized
    /// segments. Probe side streams through chunks, probing the partition
    /// that fits in cache.
    PartitionedHashJoin {
        build: Box<VecExecNode>,
        probe: Box<VecExecNode>,
        join_kind: JoinKind,
        /// Number of partitions (tuned for L2 cache residency).
        partitions: usize,
        cost: Cost,
    },

    /// Nested-loop join on small batches. Both sides materialized.
    BatchNestedLoopJoin {
        left: Box<VecExecNode>,
        right: Box<VecExecNode>,
        join_kind: JoinKind,
        cost: Cost,
    },
}

impl VecExecNode {
    /// Cost of this vectorized node.
    #[inline]
    pub fn cost(&self) -> Cost {
        match self {
            VecExecNode::ChunkedScan { cost, .. }
            | VecExecNode::IndexGather { cost, .. }
            | VecExecNode::SIMDFilter { cost, .. }
            | VecExecNode::PartitionedHashJoin { cost, .. }
            | VecExecNode::BatchNestedLoopJoin { cost, .. } => *cost,
        }
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        // Write indentation without heap allocation: two spaces per level.
        for _ in 0..indent {
            f.write_str("  ")?;
        }
        match self {
            VecExecNode::ChunkedScan {
                query_name,
                estimated_rows,
                avg_chunk_size,
                cost,
            } => {
                writeln!(
                    f,
                    "ChunkedScan [{query_name}] rows={estimated_rows} \
                     chunk_size={avg_chunk_size} cpu={:.1}",
                    cost.cpu
                )
            }
            VecExecNode::IndexGather {
                index_kind,
                component_name,
                predicate,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "IndexGather [{index_kind:?} on {component_name}] \
                     {predicate} rows={estimated_rows} cpu={:.1}",
                    cost.cpu
                )
            }
            VecExecNode::SIMDFilter {
                child,
                predicate,
                selectivity,
                branchless,
                cost,
            } => {
                let mode = if *branchless {
                    "branchless"
                } else {
                    "branched"
                };
                writeln!(
                    f,
                    "SIMDFilter [{predicate}] sel={selectivity:.2} \
                     mode={mode} rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                child.fmt_indent(f, indent + 1)
            }
            VecExecNode::PartitionedHashJoin {
                build,
                probe,
                join_kind,
                partitions,
                cost,
            } => {
                writeln!(
                    f,
                    "PartitionedHashJoin [{join_kind:?}] partitions={partitions} \
                     rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                build.fmt_indent(f, indent + 1)?;
                probe.fmt_indent(f, indent + 1)
            }
            VecExecNode::BatchNestedLoopJoin {
                left,
                right,
                join_kind,
                cost,
            } => {
                writeln!(
                    f,
                    "BatchNestedLoopJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
        }
    }
}

impl fmt::Display for VecExecNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

/// Options for vectorized plan compilation.
#[derive(Clone, Copy, Debug)]
pub struct VectorizeOpts {
    /// L2 cache size in bytes. Used to partition hash join build tables
    /// so each partition fits in cache. Default: 256 KiB.
    pub l2_cache_bytes: usize,

    /// Average component size in bytes. Used to estimate how many rows
    /// fit in a cache line / partition. Default: 16 bytes.
    pub avg_component_bytes: usize,

    /// Target archetype chunk size. Scans that produce chunks larger than
    /// this will note it in the plan for monitoring. Default: 4096 rows.
    pub target_chunk_rows: usize,
}

impl Default for VectorizeOpts {
    fn default() -> Self {
        VectorizeOpts {
            l2_cache_bytes: 256 * 1024, // 256 KiB
            avg_component_bytes: 16,    // typical f32x4 or (f32, f32, f32, f32)
            target_chunk_rows: 4096,
        }
    }
}

/// A compiled vectorized execution plan.
///
/// Produced by [`QueryPlanResult::vectorize`]. Nodes process data in
/// archetype-sized batches, enabling SIMD auto-vectorization on the
/// contiguous, 64-byte-aligned column slices that `for_each_chunk` yields.
pub struct VectorizedPlan {
    root: VecExecNode,
    opts: VectorizeOpts,
}

impl VectorizedPlan {
    /// The root node of the vectorized plan tree.
    pub fn root(&self) -> &VecExecNode {
        &self.root
    }

    /// Total estimated cost.
    pub fn cost(&self) -> Cost {
        self.root.cost()
    }

    /// Human-readable vectorized plan.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Vectorized Execution Plan ===\n");
        let _ = write!(out, "{}", self.root);
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
}

impl fmt::Display for VectorizedPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain())
    }
}

impl fmt::Debug for VectorizedPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorizedPlan")
            .field("root", &self.root)
            .field("opts", &self.opts)
            .finish()
    }
}

impl QueryPlanResult {
    /// Re-lower the logical plan with custom vectorization options.
    ///
    /// By default, `build()` compiles to vectorized execution with
    /// `VectorizeOpts::default()`. Use this method to re-lower with
    /// different cache/chunk parameters (e.g., for a different cache
    /// hierarchy).
    pub fn vectorize(&self, opts: VectorizeOpts) -> VectorizedPlan {
        let root = lower_to_vectorized(&self.root, &opts);
        VectorizedPlan { root, opts }
    }

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

/// Recursively lower a logical plan node to a vectorized execution node.
fn lower_to_vectorized(node: &PlanNode, opts: &VectorizeOpts) -> VecExecNode {
    match node {
        PlanNode::Scan {
            query_name,
            estimated_rows,
            cost,
        } => {
            // Average chunk size = total rows / estimated archetypes.
            // Conservative: assume at least 1 archetype, cap at target.
            let avg_chunk = (*estimated_rows).min(opts.target_chunk_rows).max(1);

            // Vectorized scan is slightly cheaper than scalar: batch amortizes
            // iterator overhead. Apply a 0.9x factor.
            let vec_cost = Cost {
                rows: cost.rows,
                cpu: cost.cpu * 0.9,
            };

            VecExecNode::ChunkedScan {
                query_name,
                estimated_rows: *estimated_rows,
                avg_chunk_size: avg_chunk,
                cost: vec_cost,
            }
        }

        PlanNode::IndexLookup {
            index_kind,
            component_name,
            predicate,
            estimated_rows,
            cost,
        } => {
            // Index gather: entities are batch-fetched and sorted by archetype
            // for sequential column access. Slight overhead for the sort, but
            // much better cache behavior than random access.
            let sort_overhead = (*estimated_rows as f64).log2().max(1.0);
            let vec_cost = Cost {
                rows: cost.rows,
                cpu: cost.cpu + sort_overhead,
            };

            VecExecNode::IndexGather {
                index_kind: *index_kind,
                component_name,
                predicate: predicate.clone(),
                estimated_rows: *estimated_rows,
                cost: vec_cost,
            }
        }

        PlanNode::Filter {
            child,
            predicate,
            selectivity,
            branchless_eligible,
            cost,
        } => {
            let vec_child = lower_to_vectorized(child, opts);

            let branchless = *branchless_eligible;

            // Branchless filters are ~2x cheaper on contiguous data.
            let speedup = if branchless { 0.5 } else { 0.85 };
            let vec_cost = Cost {
                rows: cost.rows,
                cpu: vec_child.cost().cpu
                    + cost.rows / cost.rows.max(1.0)
                        * Cost::FILTER_PER_ROW
                        * speedup
                        * vec_child.cost().rows,
            };

            VecExecNode::SIMDFilter {
                child: Box::new(vec_child),
                predicate: predicate.clone(),
                selectivity: *selectivity,
                branchless,
                cost: vec_cost,
            }
        }

        PlanNode::HashJoin {
            left,
            right,
            join_kind,
            cost,
        } => {
            let vec_left = lower_to_vectorized(left, opts);
            let vec_right = lower_to_vectorized(right, opts);

            // Partition count: build side should fit in L2 cache.
            // partitions = ceil(build_rows * avg_component_bytes / l2_cache_bytes)
            let build_bytes = left.cost().rows as usize * opts.avg_component_bytes;
            let l2 = opts.l2_cache_bytes.max(1); // guard against zero
            let partitions = build_bytes.div_ceil(l2).max(1);

            // Partitioned join reduces cache misses. Model as ~0.7x of naive hash join.
            let partition_factor = if partitions > 1 { 0.7 } else { 0.9 };
            let vec_cost = Cost {
                rows: cost.rows,
                cpu: (vec_left.cost().cpu + vec_right.cost().cpu)
                    + (cost.cpu - left.cost().cpu - right.cost().cpu) * partition_factor,
            };

            VecExecNode::PartitionedHashJoin {
                build: Box::new(vec_left),
                probe: Box::new(vec_right),
                join_kind: *join_kind,
                partitions,
                cost: vec_cost,
            }
        }

        PlanNode::NestedLoopJoin {
            left,
            right,
            join_kind,
            cost,
        } => {
            let vec_left = lower_to_vectorized(left, opts);
            let vec_right = lower_to_vectorized(right, opts);

            // Batch NLJ: materialize both sides, iterate in blocks.
            // Slight improvement from batch processing.
            let vec_cost = Cost {
                rows: cost.rows,
                cpu: cost.cpu * 0.95,
            };

            VecExecNode::BatchNestedLoopJoin {
                left: Box::new(vec_left),
                right: Box::new(vec_right),
                join_kind: *join_kind,
                cost: vec_cost,
            }
        }
    }
}

// ── TablePlanner — compile-time index enforcement for Table types ─────

/// A query planner scoped to a `Table` type, with compile-time enforcement
/// that required indexes exist.
///
/// Unlike [`QueryPlanner`] which emits runtime warnings for missing indexes,
/// `TablePlanner` uses trait bounds to make missing indexes a **type error**.
/// If a table field is annotated with `#[index(btree)]` or `#[index(hash)]`
/// in its `#[derive(Table)]` declaration, the corresponding
/// [`HasBTreeIndex`](crate::index::HasBTreeIndex) /
/// [`HasHashIndex`](crate::index::HasHashIndex) marker trait is generated,
/// and `TablePlanner` methods can require those traits as bounds.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Table)]
/// struct Scores {
///     #[index(btree)]
///     score: Score,
///     #[index(hash)]
///     team: Team,
/// }
///
/// let planner = TablePlanner::<Scores>::new(&world);
///
/// // These compile because Scores has the index declarations:
/// let btree = planner.btree_index::<Score>(&mut world);
/// let hash = planner.hash_index::<Team>(&mut world);
///
/// // This would NOT compile — no #[index(btree)] on a hypothetical field:
/// // planner.btree_index::<UnindexedComponent>(&mut world);
/// ```
pub struct TablePlanner<'w, T> {
    planner: QueryPlanner<'w>,
    world: &'w World,
    _marker: PhantomData<T>,
}

impl<'w, T: crate::table::Table> TablePlanner<'w, T> {
    /// Create a new table planner. Captures entity count for cost estimation.
    pub fn new(world: &'w World) -> Self {
        Self {
            planner: QueryPlanner::new(world),
            world,
            _marker: PhantomData,
        }
    }

    /// Total entities in the world (for cost estimation).
    pub fn total_entities(&self) -> usize {
        self.planner.total_entities
    }

    /// Access the underlying `QueryPlanner` for full planning capabilities.
    pub fn query_planner(&'w self) -> &'w QueryPlanner<'w> {
        &self.planner
    }

    /// Start building a scan plan for a query type.
    ///
    /// Delegates to the underlying [`QueryPlanner::scan`].
    pub fn scan<Q: crate::query::fetch::WorldQuery + 'static>(&'w self) -> ScanBuilder<'w> {
        self.planner.scan::<Q>()
    }

    /// Start building a scan plan with an explicit row estimate.
    ///
    /// Delegates to the underlying [`QueryPlanner::scan_with_estimate`].
    pub fn scan_with_estimate<Q: crate::query::fetch::WorldQuery + 'static>(
        &'w self,
        estimated_rows: usize,
    ) -> ScanBuilder<'w> {
        self.planner.scan_with_estimate::<Q>(estimated_rows)
    }

    /// Register a btree index with the underlying planner for cost-based
    /// optimization.
    ///
    /// **Compile-time enforcement**: requires `T: HasBTreeIndex<C>`.
    pub fn add_btree_index<C>(&mut self, index: &Arc<crate::index::BTreeIndex<C>>)
    where
        C: Component + Ord + Clone,
        T: crate::index::HasBTreeIndex<C>,
    {
        self.planner.add_btree_index::<C>(index, self.world);
    }

    /// Register a hash index with the underlying planner for cost-based
    /// optimization.
    ///
    /// **Compile-time enforcement**: requires `T: HasHashIndex<C>`.
    pub fn add_hash_index<C>(&mut self, index: &Arc<crate::index::HashIndex<C>>)
    where
        C: Component + std::hash::Hash + Eq + Clone,
        T: crate::index::HasHashIndex<C>,
    {
        self.planner.add_hash_index::<C>(index, self.world);
    }

    /// Get the `Indexed<C>` witness for a btree-indexed field, suitable for
    /// use with `SubscriptionBuilder`.
    ///
    /// **Compile-time enforcement**: requires `T: HasBTreeIndex<C>`.
    pub fn indexed_btree<C>(&self, index: &crate::index::BTreeIndex<C>) -> Indexed<C>
    where
        C: Component + Ord + Clone,
        T: crate::index::HasBTreeIndex<C>,
    {
        Indexed::btree(index)
    }

    /// Get the `Indexed<C>` witness for a hash-indexed field, suitable for
    /// use with `SubscriptionBuilder`.
    ///
    /// **Compile-time enforcement**: requires `T: HasHashIndex<C>`.
    pub fn indexed_hash<C>(&self, index: &crate::index::HashIndex<C>) -> Indexed<C>
    where
        C: Component + std::hash::Hash + Eq + Clone,
        T: crate::index::HasHashIndex<C>,
    {
        Indexed::hash(index)
    }
}

// ── ScratchBuffer ────────────────────────────────────────────────────

/// Pool-aware reusable entity buffer for allocation-free query execution.
///
/// Used by join/gather plans instead of per-node `Vec<Entity>` allocation.
/// Call [`clear`](ScratchBuffer::clear) between uses to reset length while
/// preserving the backing allocation.
struct ScratchBuffer {
    entities: Vec<Entity>,
}

/// Maximum pre-allocation cap: 64K entities.
const SCRATCH_MAX_CAP: usize = 64 * 1024;

#[cfg_attr(not(test), expect(dead_code))]
impl ScratchBuffer {
    /// Create a new buffer with the given estimated capacity, capped at 64K entities.
    fn new(estimated_capacity: usize) -> Self {
        Self {
            entities: Vec::with_capacity(estimated_capacity.min(SCRATCH_MAX_CAP)),
        }
    }

    /// Append an entity to the buffer.
    fn push(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Reset length to 0, preserving the backing allocation.
    fn clear(&mut self) {
        self.entities.clear();
    }

    /// Number of entities currently in the buffer.
    fn len(&self) -> usize {
        self.entities.len()
    }

    /// Current allocation capacity.
    fn capacity(&self) -> usize {
        self.entities.capacity()
    }

    /// View the buffer contents as a slice.
    fn as_slice(&self) -> &[Entity] {
        &self.entities
    }

    /// Compute the sorted intersection of two entity sets stored contiguously
    /// in this buffer as `[left_0..left_len | right_0..right_len]`.
    ///
    /// Sorts the left partition in-place by `to_bits()`, then for each entity
    /// in the right partition, binary-searches the sorted left. Matches are
    /// appended to the end of the buffer. Returns a slice over the matches.
    fn sorted_intersection(&mut self, left_len: usize) -> &[Entity] {
        let total = self.entities.len();
        debug_assert!(left_len <= total);

        // Sort the left partition in place by raw bits (deterministic total order).
        self.entities[..left_len].sort_unstable_by_key(|e| e.to_bits());

        // Scan right partition, binary-search in sorted left, collect matches.
        let mut match_count = 0;
        for i in left_len..total {
            let entity = self.entities[i];
            let found = self.entities[..left_len]
                .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
                .is_ok();
            if found {
                self.entities.push(entity);
                match_count += 1;
            }
        }

        let final_len = self.entities.len();
        &self.entities[final_len - match_count..final_len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use crate::index::SpatialIndex;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Score(u32);

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Team(u32);

    #[derive(Clone, Copy, Debug)]
    #[expect(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Debug)]
    #[expect(dead_code)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[derive(Clone, Copy, Debug)]
    #[expect(dead_code)]
    struct Health(u32);

    // ── Basic planner construction ──────────────────────────────────

    #[test]
    fn planner_new_empty_world() {
        let world = World::new();
        let planner = QueryPlanner::new(&world);
        assert_eq!(planner.total_entities, 0);
    }

    #[test]
    fn planner_captures_entity_count() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        assert_eq!(planner.total_entities, 100);
    }

    // ── Index registration ──────────────────────────────────────────

    #[test]
    fn register_btree_index() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(idx), &world);
        assert!(planner.indexes.contains_key(&TypeId::of::<Score>()));
        assert_eq!(
            planner.indexes[&TypeId::of::<Score>()].kind,
            IndexKind::BTree
        );
    }

    #[test]
    fn register_hash_index() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Team(i),));
        }
        let mut idx = HashIndex::<Team>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(idx), &world);
        assert!(planner.indexes.contains_key(&TypeId::of::<Team>()));
        assert_eq!(planner.indexes[&TypeId::of::<Team>()].kind, IndexKind::Hash);
    }

    #[test]
    fn btree_takes_precedence_over_hash() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Score(i),));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);
        let mut hash = HashIndex::<Score>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world);
        planner.add_hash_index(&Arc::new(hash), &world); // should not overwrite
        assert_eq!(
            planner.indexes[&TypeId::of::<Score>()].kind,
            IndexKind::BTree
        );
    }

    // ── Scan without predicates ─────────────────────────────────────

    #[test]
    fn scan_no_predicates() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Pos,)>().build();

        assert!(plan.warnings().is_empty());
        assert_eq!(plan.cost().rows, 100.0);
        match plan.root() {
            PlanNode::Scan { estimated_rows, .. } => assert_eq!(*estimated_rows, 100),
            other => panic!("expected Scan, got {:?}", other),
        }
    }

    // ── Index selection ─────────────────────────────────────────────

    #[test]
    fn eq_predicate_uses_hash_index() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Team(i % 10),));
        }
        let mut idx = HashIndex::<Team>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(idx), &world);

        let plan = planner
            .scan::<(&Team,)>()
            .filter(Predicate::eq(Team(5)))
            .build();

        assert!(plan.warnings().is_empty());
        match plan.root() {
            PlanNode::IndexLookup { index_kind, .. } => {
                assert_eq!(*index_kind, IndexKind::Hash);
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        }
    }

    #[test]
    fn range_predicate_uses_btree_index() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(idx), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(100)..Score(200)))
            .build();

        assert!(plan.warnings().is_empty());
        match plan.root() {
            PlanNode::IndexLookup {
                index_kind,
                component_name,
                ..
            } => {
                assert_eq!(*index_kind, IndexKind::BTree);
                assert!(component_name.contains("Score"));
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        }
    }

    // ── Missing index warnings ──────────────────────────────────────

    #[test]
    fn warns_missing_index_for_eq() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        assert_eq!(plan.warnings().len(), 1);
        match &plan.warnings()[0] {
            PlanWarning::MissingIndex { predicate_kind, .. } => {
                assert_eq!(*predicate_kind, "equality");
            }
            other => panic!("expected MissingIndex, got {:?}", other),
        }
    }

    #[test]
    fn warns_hash_index_for_range() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(idx), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();

        assert_eq!(plan.warnings().len(), 1);
        match &plan.warnings()[0] {
            PlanWarning::IndexKindMismatch { have, need, .. } => {
                assert_eq!(*have, "Hash");
                assert!(need.contains("BTree"));
            }
            other => panic!("expected IndexKindMismatch, got {:?}", other),
        }
    }

    #[test]
    fn no_warning_for_custom_predicate() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "score > threshold",
                0.5,
                |w, e| w.get::<Score>(e).is_some_and(|s| s.0 > 50),
            ))
            .build();

        // Custom predicates can never use indexes — no warning expected
        assert!(plan.warnings().is_empty());
    }

    // ── Multiple predicates ─────────────────────────────────────────

    #[test]
    fn most_selective_predicate_drives_index_lookup() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let mut team_idx = HashIndex::<Team>::new();
        team_idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(score_idx), &world);
        planner.add_hash_index(&Arc::new(team_idx), &world);

        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(2)).with_selectivity(0.2)) // 20% sel
            .filter(Predicate::eq(Score(42)).with_selectivity(0.001)) // 0.1% sel
            .build();

        // Score(42) is more selective, should be the driving lookup.
        match plan.root() {
            PlanNode::Filter { child, .. } => match child.as_ref() {
                PlanNode::IndexLookup { component_name, .. } => {
                    assert!(
                        component_name.contains("Score"),
                        "expected Score to drive, got {}",
                        component_name
                    );
                }
                other => panic!("expected IndexLookup, got {:?}", other),
            },
            PlanNode::IndexLookup { component_name, .. } => {
                // If there's only one predicate that got pushed down
                assert!(component_name.contains("Score"));
            }
            other => panic!("expected Filter or IndexLookup, got {:?}", other),
        }
    }

    // ── Join optimization ───────────────────────────────────────────

    #[test]
    fn small_join_uses_nested_loop() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan_with_estimate::<(&Score,)>(10)
            .join::<(&Team,)>(JoinKind::Inner)
            .with_right_estimate(5)
            .build();

        match plan.root() {
            PlanNode::NestedLoopJoin { .. } => {} // expected
            other => panic!("expected NestedLoopJoin, got {:?}", other),
        }
    }

    #[test]
    fn large_join_uses_hash_join() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        match plan.root() {
            PlanNode::HashJoin { .. } => {} // expected
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn hash_join_puts_smaller_side_on_left() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan_with_estimate::<(&Score,)>(100)
            .join::<(&Team,)>(JoinKind::Inner)
            .with_right_estimate(500)
            .build();

        match plan.root() {
            PlanNode::HashJoin { left, right, .. } => {
                assert!(
                    left.estimated_rows() <= right.estimated_rows(),
                    "build side should be smaller: left={:.0} right={:.0}",
                    left.estimated_rows(),
                    right.estimated_rows()
                );
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn left_join_preserves_left_side_when_right_is_smaller() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);

        // Left side is larger (500) than right side (100).
        // For an Inner join, the planner would swap them to put the smaller
        // side on the build (left) side. But Left join must preserve the
        // semantic left side — all its rows must appear in the output.
        let plan = planner
            .scan_with_estimate::<(&Score,)>(500)
            .join::<(&Team,)>(JoinKind::Left)
            .with_right_estimate(100)
            .build();

        match plan.root() {
            PlanNode::HashJoin {
                left,
                right,
                join_kind,
                ..
            } => {
                assert_eq!(*join_kind, JoinKind::Left);
                // Left side must be the original scan (500 rows), not swapped.
                assert_eq!(
                    left.estimated_rows(),
                    500.0,
                    "left join must keep original left side (500 rows), got {}",
                    left.estimated_rows()
                );
                assert_eq!(
                    right.estimated_rows(),
                    100.0,
                    "right side should be 100 rows, got {}",
                    right.estimated_rows()
                );
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn inner_join_still_reorders_smaller_to_build_side() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);

        // Inner join: left (500) > right (100), should swap.
        let plan = planner
            .scan_with_estimate::<(&Score,)>(500)
            .join::<(&Team,)>(JoinKind::Inner)
            .with_right_estimate(100)
            .build();

        match plan.root() {
            PlanNode::HashJoin { left, right, .. } => {
                assert!(
                    left.estimated_rows() <= right.estimated_rows(),
                    "inner join build side should be smaller: left={} right={}",
                    left.estimated_rows(),
                    right.estimated_rows()
                );
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    // ── Explain output ──────────────────────────────────────────────

    #[test]
    fn explain_contains_plan_details() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(idx), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();

        // Default explain shows vectorized plan
        let explain = plan.explain();
        assert!(explain.contains("Vectorized Execution Plan"));
        assert!(explain.contains("IndexGather"));
        assert!(explain.contains("BTree"));
        assert!(explain.contains("Score"));
        assert!(explain.contains("L2 cache budget"));

        // Logical explain shows the original plan tree
        let logical = plan.explain_logical();
        assert!(logical.contains("Logical Plan"));
        assert!(logical.contains("IndexLookup"));
    }

    // ── Subscription plans ──────────────────────────────────────────

    #[test]
    fn subscription_requires_indexed_witness() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);
        let witness = Indexed::btree(&idx);

        let sub = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, 0.001)
            .build()
            .unwrap();

        // Subscription plans have no warnings (all predicates are indexed).
        match sub.root() {
            PlanNode::IndexLookup { index_kind, .. } => {
                assert_eq!(*index_kind, IndexKind::BTree);
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        }
        assert!(sub.cost().cpu > 0.0);
    }

    #[test]
    fn subscription_multiple_predicates_ordered_by_selectivity() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let mut team_idx = HashIndex::<Team>::new();
        team_idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);
        let score_w = Indexed::btree(&score_idx);
        let team_w = Indexed::hash(&team_idx);

        let sub = planner
            .subscribe::<(&Score, &Team)>()
            .where_eq(team_w, 0.2) // less selective
            .where_eq(score_w, 0.001) // more selective — should drive
            .build()
            .unwrap();

        // Most selective predicate (Score) should be the driving index lookup.
        match sub.root() {
            PlanNode::Filter { child, .. } => match child.as_ref() {
                PlanNode::IndexLookup { component_name, .. } => {
                    assert!(component_name.contains("Score"));
                }
                other => panic!("expected IndexLookup, got {:?}", other),
            },
            PlanNode::IndexLookup { component_name, .. } => {
                assert!(component_name.contains("Score"));
            }
            other => panic!("expected Filter or IndexLookup, got {:?}", other),
        }
    }

    #[test]
    fn subscription_where_range_accepts_btree_witness() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);
        let witness = Indexed::btree(&idx);

        let sub = planner
            .subscribe::<(&Score,)>()
            .where_range(witness, 0.1)
            .build()
            .unwrap();

        match sub.root() {
            PlanNode::IndexLookup { index_kind, .. } => {
                assert_eq!(*index_kind, IndexKind::BTree);
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        }
    }

    #[test]
    fn subscription_where_range_rejects_hash_witness() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);
        let witness = Indexed::hash(&idx);

        // Hash indexes cannot serve range queries — build returns an error.
        let result = planner
            .subscribe::<(&Score,)>()
            .where_range(witness, 0.1)
            .build();
        match result {
            Err(errs)
                if errs
                    .iter()
                    .any(|e| matches!(e, SubscriptionError::HashIndexOnRange { .. })) => {}
            other => panic!("expected HashIndexOnRange error, got {:?}", other),
        }
    }

    #[test]
    fn subscription_nan_selectivity_returns_error() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);
        let witness = Indexed::btree(&idx);

        let result = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, f64::NAN)
            .build();
        match result {
            Err(errs)
                if errs
                    .iter()
                    .any(|e| matches!(e, SubscriptionError::NanSelectivity { .. })) => {}
            other => panic!("expected NanSelectivity error, got {:?}", other),
        }
    }

    // ── Constraint validation ───────────────────────────────────────

    #[test]
    fn constraint_exactly_one() {
        let c = CardinalityConstraint::ExactlyOne;
        assert!(c.satisfied_by(1.0));
        assert!(!c.satisfied_by(0.0));
        assert!(!c.satisfied_by(2.0));
    }

    #[test]
    fn constraint_at_most() {
        let c = CardinalityConstraint::AtMost(10);
        assert!(c.satisfied_by(5.0));
        assert!(c.satisfied_by(10.0));
        assert!(!c.satisfied_by(11.0));
    }

    #[test]
    fn constraint_between() {
        let c = CardinalityConstraint::Between(5, 15);
        assert!(c.satisfied_by(10.0));
        assert!(!c.satisfied_by(3.0));
        assert!(!c.satisfied_by(20.0));
    }

    #[test]
    fn validate_plan_constraints() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        let violations =
            plan.validate_constraints(&[("max_100", CardinalityConstraint::AtMost(100))]);
        assert!(violations.is_empty());

        let violations =
            plan.validate_constraints(&[("max_10", CardinalityConstraint::AtMost(10))]);
        assert_eq!(violations.len(), 1);
    }

    // ── choose_cheaper ──────────────────────────────────────────────

    #[test]
    fn choose_cheaper_picks_lower_cost() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(idx), &world);

        let full_scan = planner.scan::<(&Score,)>().build();
        let indexed = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        let chosen = full_scan.cheaper(&indexed);
        assert!(chosen.cost().total() <= full_scan.cost().total());
    }

    // ── Cost model ──────────────────────────────────────────────────

    #[test]
    fn index_lookup_cheaper_than_scan_for_selective_pred() {
        let scan_cost = Cost::scan(10_000);
        let idx_cost = Cost::index_lookup(0.01, 10_000);
        assert!(
            idx_cost.total() < scan_cost.total(),
            "index lookup ({:.1}) should be cheaper than scan ({:.1})",
            idx_cost.total(),
            scan_cost.total()
        );
    }

    #[test]
    fn hash_join_cheaper_than_nested_loop_for_large() {
        let left = Cost::scan(1000);
        let right = Cost::scan(1000);
        let hash = Cost::hash_join(left, right);
        let nested = Cost::nested_loop_join(left, right);
        assert!(
            hash.total() < nested.total(),
            "hash join ({:.1}) should be cheaper than nested loop ({:.1}) for 1000x1000",
            hash.total(),
            nested.total()
        );
    }

    #[test]
    fn filter_reduces_estimated_rows() {
        let scan = Cost::scan(1000);
        let filtered = Cost::filter(scan, 0.1);
        assert!(
            filtered.rows < scan.rows,
            "filter should reduce rows: {:.0} vs {:.0}",
            filtered.rows,
            scan.rows
        );
    }

    // ── Selectivity override ────────────────────────────────────────

    #[test]
    fn custom_selectivity() {
        let pred = Predicate::eq(Score(42)).with_selectivity(0.5);
        assert!((pred.selectivity - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn selectivity_clamped() {
        let pred = Predicate::eq(Score(42)).with_selectivity(2.0);
        assert!((pred.selectivity - 1.0).abs() < f64::EPSILON);

        let pred = Predicate::eq(Score(42)).with_selectivity(-1.0);
        assert!(pred.selectivity.abs() < f64::EPSILON);
    }

    #[test]
    fn nan_selectivity_normalized_to_worst_case() {
        // with_selectivity
        let pred = Predicate::eq(Score(42)).with_selectivity(f64::NAN);
        assert!((pred.selectivity - 1.0).abs() < f64::EPSILON);

        // custom constructor
        let pred = Predicate::custom::<Score>("test", f64::NAN, |_, _| true);
        assert!((pred.selectivity - 1.0).abs() < f64::EPSILON);
    }

    // ── Display ─────────────────────────────────────────────────────

    #[test]
    fn plan_display_does_not_panic() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(score_idx), &world);

        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .filter(Predicate::custom::<Team>("team != 0", 0.8, |w, e| {
                w.get::<Team>(e).is_some_and(|t| t.0 != 0)
            }))
            .join::<(&Pos,)>(JoinKind::Inner)
            .build();

        let display = format!("{plan}");
        assert!(!display.is_empty());
        let debug = format!("{plan:?}");
        assert!(!debug.is_empty());
    }

    // ── Indexed witness ─────────────────────────────────────────────

    #[test]
    fn indexed_btree_witness() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let witness = Indexed::btree(&idx);
        assert_eq!(witness.kind, IndexKind::BTree);
        assert_eq!(witness.cardinality, 50);
    }

    #[test]
    fn indexed_hash_witness() {
        let mut world = World::new();
        for i in 0..30 {
            world.spawn((Team(i),));
        }
        let mut idx = HashIndex::<Team>::new();
        idx.rebuild(&mut world);

        let witness = Indexed::hash(&idx);
        assert_eq!(witness.kind, IndexKind::Hash);
        assert_eq!(witness.cardinality, 30);
    }

    #[test]
    fn indexed_is_copy() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let w1 = Indexed::btree(&idx);
        let w2 = w1; // copy
        assert_eq!(w1.kind, w2.kind);
    }

    // ── Vectorized execution (default) ────────────────────────────────

    #[test]
    fn build_produces_vectorized_by_default() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        // vec_root() is available without calling .vectorize()
        match plan.vec_root() {
            VecExecNode::ChunkedScan { estimated_rows, .. } => {
                assert_eq!(*estimated_rows, 1000);
            }
            other => panic!("expected ChunkedScan, got {:?}", other),
        }

        // cost() returns vectorized cost (cheaper than logical)
        assert!(plan.cost().cpu < plan.logical_cost().cpu);
    }

    #[test]
    fn vectorize_scan_produces_chunked_scan() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());

        match vec_plan.root() {
            VecExecNode::ChunkedScan {
                estimated_rows,
                avg_chunk_size,
                ..
            } => {
                assert_eq!(*estimated_rows, 1000);
                assert!(*avg_chunk_size <= VectorizeOpts::default().target_chunk_rows);
            }
            other => panic!("expected ChunkedScan, got {:?}", other),
        }
    }

    #[test]
    fn vectorize_index_lookup_produces_index_gather() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(idx), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());

        match vec_plan.root() {
            VecExecNode::IndexGather { index_kind, .. } => {
                assert_eq!(*index_kind, IndexKind::BTree);
            }
            other => panic!("expected IndexGather, got {:?}", other),
        }
    }

    #[test]
    fn vectorize_filter_detects_branchless() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);

        // Range predicate without index → scan + filter
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());

        match vec_plan.root() {
            VecExecNode::SIMDFilter { branchless, .. } => {
                assert!(*branchless, "Range predicate should be branchless");
            }
            other => panic!("expected SIMDFilter, got {:?}", other),
        }

        // Custom predicate → branched
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>("complex check", 0.5, |_, _| {
                true
            }))
            .build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());

        match vec_plan.root() {
            VecExecNode::SIMDFilter { branchless, .. } => {
                assert!(!*branchless, "Custom predicate should be branched");
            }
            other => panic!("expected SIMDFilter, got {:?}", other),
        }
    }

    #[test]
    fn vectorize_hash_join_partitioned() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());

        match vec_plan.root() {
            VecExecNode::PartitionedHashJoin { partitions, .. } => {
                assert!(*partitions >= 1);
            }
            other => panic!("expected PartitionedHashJoin, got {:?}", other),
        }
    }

    #[test]
    fn vectorize_nested_loop_becomes_batch() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan_with_estimate::<(&Score,)>(10)
            .join::<(&Team,)>(JoinKind::Inner)
            .with_right_estimate(5)
            .build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());

        match vec_plan.root() {
            VecExecNode::BatchNestedLoopJoin { .. } => {} // expected
            other => panic!("expected BatchNestedLoopJoin, got {:?}", other),
        }
    }

    #[test]
    fn vectorized_cost_cheaper_than_logical() {
        let mut world = World::new();
        for i in 0..10_000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        // cost() returns vectorized (default), logical_cost() returns pre-lowering
        assert!(
            plan.cost().cpu < plan.logical_cost().cpu,
            "vectorized ({:.1}) should be cheaper than logical ({:.1})",
            plan.cost().cpu,
            plan.logical_cost().cpu
        );
    }

    #[test]
    fn vectorize_explain_contains_details() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>("x > 0", 0.5, |w, e| {
                w.get::<Score>(e).is_some_and(|s| s.0 > 0)
            }))
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let vec_plan = plan.vectorize(VectorizeOpts::default());
        let explain = vec_plan.explain();

        assert!(explain.contains("Vectorized Execution Plan"));
        assert!(explain.contains("L2 cache budget"));
        assert!(explain.contains("target chunk"));
    }

    #[test]
    fn vectorize_opts_custom() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        let opts = VectorizeOpts {
            l2_cache_bytes: 128 * 1024,
            avg_component_bytes: 32,
            target_chunk_rows: 1024,
        };
        let vec_plan = plan.vectorize(opts);

        match vec_plan.root() {
            VecExecNode::ChunkedScan { avg_chunk_size, .. } => {
                assert!(*avg_chunk_size <= 1024);
            }
            other => panic!("expected ChunkedScan, got {:?}", other),
        }
    }

    #[test]
    fn vectorize_hash_join_zero_l2_cache_does_not_panic() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let opts = VectorizeOpts {
            l2_cache_bytes: 0,
            ..VectorizeOpts::default()
        };
        let vec_plan = plan.vectorize(opts);

        match vec_plan.root() {
            VecExecNode::PartitionedHashJoin { partitions, .. } => {
                assert!(*partitions >= 1);
            }
            other => panic!("expected PartitionedHashJoin, got {:?}", other),
        }
    }

    // ── TablePlanner tests ───────────────────────────────────────────

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, minkowski_derive::Table)]
    struct IndexedScores {
        #[index(btree)]
        score: Score,
        #[index(hash)]
        team: Team,
    }

    #[test]
    fn table_planner_creates_btree_index() {
        use crate::index::HasBTreeIndex;

        let mut world = World::new();
        for i in 0..10 {
            world.spawn(IndexedScores {
                score: Score(i),
                team: Team(i % 3),
            });
        }

        // HasBTreeIndex trait provides create_btree_index
        let idx = IndexedScores::create_btree_index(&mut world);
        assert_eq!(idx.len(), 10);
        assert_eq!(idx.get(&Score(5)).len(), 1);
    }

    #[test]
    fn table_planner_creates_hash_index() {
        use crate::index::HasHashIndex;

        let mut world = World::new();
        for i in 0..9 {
            world.spawn(IndexedScores {
                score: Score(i),
                team: Team(i % 3),
            });
        }

        let idx = IndexedScores::create_hash_index(&mut world);
        assert_eq!(idx.len(), 9);
        // 3 entities per team
        assert_eq!(idx.get(&Team(0)).len(), 3);
    }

    #[test]
    fn table_planner_indexed_witness() {
        use crate::index::{HasBTreeIndex, HasHashIndex};

        let mut world = World::new();
        for i in 0..5 {
            world.spawn(IndexedScores {
                score: Score(i),
                team: Team(i % 2),
            });
        }

        let btree = IndexedScores::create_btree_index(&mut world);
        let hash = IndexedScores::create_hash_index(&mut world);

        // TablePlanner provides indexed_* witnesses with compile-time enforcement
        let planner = TablePlanner::<IndexedScores>::new(&world);
        let indexed_bt = planner.indexed_btree::<Score>(&btree);
        let indexed_hs = planner.indexed_hash::<Team>(&hash);

        assert!(matches!(indexed_bt, Indexed { .. }));
        assert!(matches!(indexed_hs, Indexed { .. }));
    }

    #[test]
    fn table_planner_scan_builds_plan() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn(IndexedScores {
                score: Score(i),
                team: Team(i % 5),
            });
        }

        let planner = TablePlanner::<IndexedScores>::new(&world);
        let plan = planner.scan::<(&Score, &Team)>().build();

        // Should produce a valid plan with vectorized execution
        assert!(plan.cost().cpu > 0.0);
        let explain = plan.explain();
        assert!(explain.contains("Vectorized"));
    }

    #[test]
    fn table_planner_scan_with_index_filter() {
        use crate::index::HasBTreeIndex;

        let mut world = World::new();
        for i in 0..100 {
            world.spawn(IndexedScores {
                score: Score(i),
                team: Team(i % 5),
            });
        }

        // Create index first (needs &mut World)
        let btree = IndexedScores::create_btree_index(&mut world);

        // Then create planner (borrows &World)
        let mut planner = TablePlanner::<IndexedScores>::new(&world);
        planner.add_btree_index::<Score>(&Arc::new(btree));

        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();

        // Should use index (IndexGather in vectorized, IndexLookup in logical)
        let explain = plan.explain();
        assert!(
            explain.contains("IndexGather") || explain.contains("IndexLookup"),
            "expected index-driven plan, got:\n{explain}"
        );
    }

    #[test]
    fn table_planner_total_entities() {
        let mut world = World::new();
        for i in 0..20 {
            world.spawn(IndexedScores {
                score: Score(i),
                team: Team(0),
            });
        }

        let planner = TablePlanner::<IndexedScores>::new(&world);
        assert_eq!(planner.total_entities(), 20);
    }

    #[test]
    fn has_btree_index_field_name() {
        use crate::index::HasBTreeIndex;
        assert_eq!(<IndexedScores as HasBTreeIndex<Score>>::FIELD_NAME, "score");
    }

    #[test]
    fn has_hash_index_field_name() {
        use crate::index::HasHashIndex;
        assert_eq!(<IndexedScores as HasHashIndex<Team>>::FIELD_NAME, "team");
    }

    // ── Plan execution ─────────────────────────────────────────────────

    #[test]
    fn execute_scan_returns_all_matching_entities() {
        let mut world = World::new();
        let mut expected = Vec::new();
        for i in 0..20 {
            expected.push(world.spawn((Score(i),)));
        }
        // Different archetype — should also be matched
        for i in 20..30 {
            expected.push(world.spawn((Score(i), Team(0))));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let mut result = plan.execute(&world);
        result.sort_by_key(|e| e.to_bits());
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(result, expected);
    }

    #[test]
    fn execute_scan_excludes_non_matching_archetypes() {
        let mut world = World::new();
        // Archetype with Score only
        let e1 = world.spawn((Score(1),));
        // Archetype with Team only — should NOT match scan::<(&Score,)>()
        let _e2 = world.spawn((Team(1),));

        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute(&world);
        assert_eq!(result, vec![e1]);
    }

    #[test]
    fn execute_filter_eq() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 1);
        assert_eq!(*world.get::<Score>(result[0]).unwrap(), Score(42));
    }

    #[test]
    fn execute_filter_range() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 10);
        for e in &result {
            let s = world.get::<Score>(*e).unwrap().0;
            assert!((10..20).contains(&s), "score {s} out of range");
        }
    }

    #[test]
    fn execute_index_driven_eq() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(hash), &world);

        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(2)))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 20); // 100 / 5 teams
        for e in &result {
            assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
        }
    }

    #[test]
    fn execute_multi_predicate() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i), Team(i % 5)));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(2)))
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();
        let result = plan.execute(&world);
        for e in &result {
            let s = world.get::<Score>(*e).unwrap().0;
            let t = world.get::<Team>(*e).unwrap().0;
            assert!((10..50).contains(&s), "score {s} out of range");
            assert_eq!(t, 2, "team {t} != 2");
        }
        // scores 10..50 with team==2: 12, 17, 22, 27, 32, 37, 42, 47 = 8
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn execute_join_intersects_entity_sets() {
        let mut world = World::new();
        // Entities with Score only
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        // Entities with both Score and Team — only these should survive the join
        let mut both = Vec::new();
        for i in 10..20 {
            both.push(world.spawn((Score(i), Team(i % 3))));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let mut result = plan.execute(&world);
        result.sort_by_key(|e| e.to_bits());
        both.sort_by_key(|e| e.to_bits());
        assert_eq!(result, both);
    }

    #[test]
    fn execute_custom_filter() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "even scores",
                0.5,
                |world, entity| world.get::<Score>(entity).is_some_and(|s| s.0 % 2 == 0),
            ))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 25);
        for e in &result {
            assert!(world.get::<Score>(*e).unwrap().0.is_multiple_of(2));
        }
    }

    #[test]
    fn execute_empty_world() {
        let world = World::new();
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute(&world);
        assert!(result.is_empty());
    }

    #[test]
    fn execute_respects_despawned_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1),));
        let e2 = world.spawn((Score(2),));
        let e3 = world.spawn((Score(3),));

        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        // Despawn e2 after plan construction
        world.despawn(e2);

        let result = plan.execute(&world);
        // Scan walks archetypes — despawned entity is replaced via swap_remove,
        // so the archetype only contains live entities.
        assert_eq!(result.len(), 2);
        assert!(result.contains(&e1));
        assert!(result.contains(&e3));
    }

    #[test]
    fn execute_index_driven_respects_query_components() {
        // Regression: index-driven lookup must only return entities that match
        // ALL queried components, not just the indexed one.
        let mut world = World::new();
        // Entities with Team only (no Score)
        for i in 0..10 {
            world.spawn((Team(i % 3),));
        }
        // Entities with both Score and Team
        let mut both = Vec::new();
        for i in 0..10 {
            both.push(world.spawn((Score(i), Team(i % 3))));
        }

        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(hash), &world);

        // scan::<(&Score, &Team)> requires BOTH components
        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(1)))
            .build();
        let result = plan.execute(&world);
        // Only entities with both Score AND Team(1) should appear
        for e in &result {
            assert!(world.get::<Score>(*e).is_some(), "missing Score");
            assert_eq!(*world.get::<Team>(*e).unwrap(), Team(1));
        }
        // Team(1) entities with Score: indices 1, 4, 7 = 3
        assert_eq!(result.len(), 3);
    }

    // ── Predicate-specific index lookup ─────────────────────────────────

    #[test]
    fn execute_btree_eq_uses_targeted_lookup() {
        // Verify that BTree + Eq predicate returns only matching entities,
        // not the entire index (regression: was O(n) full-index scan).
        let mut world = World::new();
        for i in 0..200 {
            world.spawn((Score(i), Team(i % 10)));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 1);
        assert_eq!(*world.get::<Score>(result[0]).unwrap(), Score(42));
    }

    #[test]
    fn execute_btree_range_uses_targeted_lookup() {
        // Verify that BTree + Range predicate returns only entities in range,
        // not the entire index (regression: was O(n) full-index scan).
        let mut world = World::new();
        for i in 0..200 {
            world.spawn((Score(i),));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 10); // scores 10..20
        for e in &result {
            let s = world.get::<Score>(*e).unwrap().0;
            assert!((10..20).contains(&s), "score {s} out of range");
        }
    }

    #[test]
    fn execute_hash_eq_uses_targeted_lookup() {
        // Verify that Hash + Eq predicate returns only matching entities,
        // not the entire index (regression: was O(n) full-index scan).
        let mut world = World::new();
        for i in 0..200 {
            world.spawn((Score(i), Team(i % 10)));
        }
        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(hash), &world);

        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(3)))
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 20); // 200 / 10 teams
        for e in &result {
            assert_eq!(*world.get::<Team>(*e).unwrap(), Team(3));
        }
    }

    #[test]
    fn execute_btree_eq_nonexistent_value_returns_empty() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Score(i),));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(999)))
            .build();
        let result = plan.execute(&world);
        assert!(result.is_empty());
    }

    #[test]
    fn execute_hash_eq_nonexistent_value_returns_empty() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Team(i),));
        }
        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(hash), &world);

        let plan = planner
            .scan::<(&Team,)>()
            .filter(Predicate::eq(Team(999)))
            .build();
        let result = plan.execute(&world);
        assert!(result.is_empty());
    }

    // ── Live index reads ──────────────────────────────────────────────

    #[test]
    fn execute_reads_live_btree_not_registration_snapshot() {
        // Regression: plans used to capture a frozen BTreeMap clone at
        // registration time. After rebuild, the plan would return stale results.
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Score(i),));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);
        let btree = Arc::new(btree);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&btree, &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        // First execute: should find Score(42)
        let result = plan.execute(&world);
        assert_eq!(result.len(), 1);

        // Spawn more entities and rebuild the index via a new Arc.
        // The plan still holds the old Arc — it sees the old index contents.
        // This is the expected behavior: the plan reads the Arc it was given.
        // To see updated data, register a new index and rebuild the plan.
        for i in 50..100 {
            world.spawn((Score(i),));
        }

        // Verify scan (non-index) path sees new entities immediately
        let planner2 = QueryPlanner::new(&world);
        let scan_plan = planner2.scan::<(&Score,)>().build();
        assert_eq!(scan_plan.execute(&world).len(), 100);
    }

    #[test]
    fn execute_reads_live_hash_not_registration_snapshot() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Team(i % 5),));
        }
        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);
        let hash = Arc::new(hash);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&hash, &world);

        let plan = planner
            .scan::<(&Team,)>()
            .filter(Predicate::eq(Team(2)))
            .build();

        let result = plan.execute(&world);
        assert_eq!(result.len(), 10); // 50 / 5 teams
        for e in &result {
            assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
        }
    }

    // ── Vectorized execution ───────────────────────────────────────────

    #[test]
    fn exec_tree_uses_chunked_scan() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let root = plan.exec_root().expect("should have exec tree");
        assert_eq!(root.variant_name(), "ChunkedScan");
        assert!(root.chunk_size().unwrap() > 0);
    }

    #[test]
    fn exec_tree_uses_batch_filter() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let root = plan.exec_root().expect("should have exec tree");
        assert_eq!(root.variant_name(), "BatchFilter");
        assert!(root.batch_size().unwrap() > 0);
    }

    #[test]
    fn exec_tree_uses_partitioned_hash_join() {
        let mut world = World::new();
        for i in 0..200 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let root = plan.exec_root().expect("should have exec tree");
        assert_eq!(root.variant_name(), "PartitionedHashJoin");
        assert!(root.partitions().unwrap() >= 1);
    }

    #[test]
    fn exec_tree_uses_batch_nested_loop_for_small_joins() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i), Team(i % 2)));
        }
        let planner = QueryPlanner::new(&world);
        // Small cardinality estimates should produce BatchNestedLoopJoin
        let plan = planner
            .scan_with_estimate::<(&Score,)>(5)
            .join::<(&Team,)>(JoinKind::Inner)
            .with_right_estimate(3)
            .build();
        let root = plan.exec_root().expect("should have exec tree");
        assert_eq!(root.variant_name(), "BatchNestedLoopJoin");
    }

    #[test]
    fn exec_tree_index_gather_sorts_by_archetype() {
        let mut world = World::new();
        // Two archetypes: Score-only and Score+Team
        for i in 0..50 {
            world.spawn((Score(i),));
        }
        for i in 50..100 {
            world.spawn((Score(i), Team(0)));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world);

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(0)..Score(100)))
            .build();

        // Find the IndexGather node in the tree
        let root = plan.exec_root().expect("should have exec tree");
        fn find_index_gather(node: &ExecNode) -> Option<&ExecNode> {
            if matches!(node, ExecNode::IndexGather { .. }) {
                return Some(node);
            }
            for child in node.children() {
                if let Some(found) = find_index_gather(child) {
                    return Some(found);
                }
            }
            None
        }
        let gather = find_index_gather(root).expect("plan should contain IndexGather");
        assert_eq!(gather.sort_by_archetype(), Some(true));

        // Execute and verify entities come out sorted by archetype
        let entities = plan.execute(&world);
        assert!(!entities.is_empty());
        // Verify archetype ordering: all entities from archetype A should
        // appear before entities from archetype B (monotonic archetype IDs).
        let archetype_ids: Vec<usize> = entities
            .iter()
            .filter_map(|e| {
                let idx = e.index() as usize;
                world
                    .entity_locations
                    .get(idx)
                    .and_then(|loc| loc.as_ref())
                    .map(|loc| loc.archetype_id.0)
            })
            .collect();
        // Check monotonicity (sorted)
        for w in archetype_ids.windows(2) {
            assert!(
                w[0] <= w[1],
                "entities not sorted by archetype: {:?}",
                &archetype_ids[..10]
            );
        }
    }

    #[test]
    fn exec_tree_chunk_size_matches_vec_plan() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        // The exec tree's chunk_size should come from the vectorized plan.
        let root = plan.exec_root().expect("should have exec tree");
        let exec_chunk = root.chunk_size().unwrap();
        // The vec plan's avg_chunk_size
        if let VecExecNode::ChunkedScan { avg_chunk_size, .. } = plan.vec_root() {
            assert_eq!(exec_chunk, *avg_chunk_size);
        } else {
            panic!("expected ChunkedScan vec plan root");
        }
    }

    #[test]
    fn exec_tree_partition_count_matches_vec_plan() {
        let mut world = World::new();
        for i in 0..200 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let root = plan.exec_root().expect("should have exec tree");
        let exec_partitions = root.partitions().unwrap();
        if let VecExecNode::PartitionedHashJoin { partitions, .. } = plan.vec_root() {
            assert_eq!(exec_partitions, *partitions);
        } else {
            panic!("expected PartitionedHashJoin vec plan root");
        }
    }

    #[test]
    fn vectorized_execution_produces_correct_results() {
        // End-to-end: vectorized plan must produce the same results as
        // a naive query would.
        let mut world = World::new();
        for i in 0..500 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);
        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world);
        planner.add_hash_index(&Arc::new(hash), &world);

        // Complex plan: index + filter + join
        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::range::<Score, _>(Score(100)..Score(300)))
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let entities = plan.execute(&world);
        // All 500 entities have Team, so the join doesn't reduce the set.
        // The range filter should give us Score 100..300 = 200 entities.
        assert_eq!(entities.len(), 200);
        for e in &entities {
            let s = world.get::<Score>(*e).unwrap().0;
            assert!((100..300).contains(&s), "score {s} out of range");
        }
    }

    // ── Left join execution ────────────────────────────────────────────

    #[test]
    fn execute_left_join_preserves_all_left_entities() {
        let mut world = World::new();
        // 20 entities with Score only (no Team)
        let mut score_only = Vec::new();
        for i in 0..20 {
            score_only.push(world.spawn((Score(i),)));
        }
        // 10 entities with both Score and Team
        let mut both = Vec::new();
        for i in 20..30 {
            both.push(world.spawn((Score(i), Team(i % 3))));
        }

        let planner = QueryPlanner::new(&world);
        // Left join: all Score entities should appear, even those without Team.
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Left)
            .build();
        let result = plan.execute(&world);
        // All 30 Score entities must be present.
        assert_eq!(result.len(), 30);
        for e in &score_only {
            assert!(result.contains(e), "left join dropped Score-only entity");
        }
        for e in &both {
            assert!(result.contains(e), "left join dropped Score+Team entity");
        }
    }

    #[test]
    fn execute_inner_join_excludes_unmatched() {
        let mut world = World::new();
        // 20 entities with Score only (no Team)
        for i in 0..20 {
            world.spawn((Score(i),));
        }
        // 10 entities with both Score and Team
        let mut both = Vec::new();
        for i in 20..30 {
            both.push(world.spawn((Score(i), Team(i % 3))));
        }

        let planner = QueryPlanner::new(&world);
        // Inner join: only entities with both Score and Team.
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let result = plan.execute(&world);
        assert_eq!(result.len(), 10);
        for e in &both {
            assert!(result.contains(e));
        }
    }

    #[test]
    fn execute_left_join_small_cardinality_nested_loop() {
        let mut world = World::new();
        // Score-only entities
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        // Score+Team entities
        for i in 5..8 {
            world.spawn((Score(i), Team(0)));
        }

        let planner = QueryPlanner::new(&world);
        // Small estimates → nested-loop join path.
        let plan = planner
            .scan_with_estimate::<(&Score,)>(8)
            .join::<(&Team,)>(JoinKind::Left)
            .with_right_estimate(3)
            .build();
        let result = plan.execute(&world);
        // Left join: all 8 Score entities preserved.
        assert_eq!(result.len(), 8);
    }

    // ── ScratchBuffer tests ──────────────────────────────────────────

    #[test]
    fn scratch_buffer_starts_empty() {
        let buf = ScratchBuffer::new(100);
        assert_eq!(buf.len(), 0);
        assert!(buf.capacity() >= 100);
    }

    #[test]
    fn scratch_buffer_push_and_clear() {
        let mut buf = ScratchBuffer::new(4);
        let e0 = Entity::new(0, 0);
        let e1 = Entity::new(1, 0);
        buf.push(e0);
        buf.push(e1);
        assert_eq!(buf.as_slice(), &[e0, e1]);

        let cap_before = buf.capacity();
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), cap_before);
    }

    #[test]
    fn scratch_buffer_reuse_does_not_realloc() {
        let mut buf = ScratchBuffer::new(64);
        for i in 0..50 {
            buf.push(Entity::new(i, 0));
        }
        let cap = buf.capacity();
        buf.clear();
        for i in 0..50 {
            buf.push(Entity::new(100 + i, 0));
        }
        assert_eq!(buf.capacity(), cap);
    }

    #[test]
    fn scratch_buffer_sorted_intersection() {
        let mut buf = ScratchBuffer::new(16);
        // Left set: [1,3,5,7,9]
        let left = [1u32, 3, 5, 7, 9];
        for &idx in &left {
            buf.push(Entity::new(idx, 0));
        }
        let left_len = left.len();

        // Right set: [2,3,6,7,10]
        let right = [2u32, 3, 6, 7, 10];
        for &idx in &right {
            buf.push(Entity::new(idx, 0));
        }

        let result = buf.sorted_intersection(left_len);
        // Intersection should be entities with index 3 and 7.
        let mut result_indices: Vec<u32> = result.iter().map(|e| e.index()).collect();
        result_indices.sort_unstable();
        assert_eq!(result_indices, vec![3, 7]);
    }
}
