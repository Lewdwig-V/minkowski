//! Volcano-model query planner for composing index-driven lookups, joins,
//! and full scans into optimized execution plans.
//!
//! The planner is designed for an in-memory ECS where data already lives in L1/L2
//! cache. Planning overhead is kept to O(indexes + predicates) — no heap
//! allocations during plan compilation, no dynamic dispatch during execution.
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
//! // Register available indexes
//! planner.add_btree_index::<Score>(&score_index);
//! planner.add_hash_index::<Team>(&team_index);
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

use crate::component::{Component, ComponentId};
use crate::index::{BTreeIndex, HashIndex};
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
    pub fn rows(&self) -> f64 {
        self.rows
    }

    /// Estimated CPU cost (comparison/hash operations).
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

/// Type-erased index metadata for the planner.
#[derive(Clone, Debug)]
#[expect(dead_code)]
struct IndexDescriptor {
    component_type: TypeId,
    component_id: ComponentId,
    component_name: &'static str,
    kind: IndexKind,
    cardinality: usize, // number of entities in the index
}

// ── Predicates ───────────────────────────────────────────────────────

/// A predicate that can be pushed down into an index lookup or applied as
/// a post-fetch filter.
pub struct Predicate {
    component_type: TypeId,
    component_name: &'static str,
    kind: PredicateKind,
    selectivity: f64,
}

enum PredicateKind {
    Eq(#[expect(dead_code)] String), // debug representation — reserved for enhanced EXPLAIN
    Range(#[expect(dead_code)] String), // debug representation — reserved for enhanced EXPLAIN
    Custom(Box<str>),                // description only — always post-filter
}

impl Predicate {
    /// Equality predicate on component `T`.
    ///
    /// If a `HashIndex<T>` or `BTreeIndex<T>` is registered, the planner will
    /// use it for an O(1) / O(log n) lookup instead of a full scan.
    pub fn eq<T: Component + std::fmt::Debug>(value: T) -> Self {
        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Eq(format!("{value:?}")),
            selectivity: 0.01, // default: 1% selectivity for equality
        }
    }

    /// Range predicate on component `T`.
    ///
    /// If a `BTreeIndex<T>` is registered, the planner will use it for
    /// an O(log n + k) range scan. `HashIndex` cannot serve range predicates.
    pub fn range<T: Component + std::fmt::Debug, R: RangeBounds<T>>(range: R) -> Self {
        fn bound_str<T: std::fmt::Debug>(bound: Bound<&T>) -> String {
            match bound {
                Bound::Included(v) => format!("{v:?}"),
                Bound::Excluded(v) => format!("{v:?}"),
                Bound::Unbounded => "..".into(),
            }
        }
        let desc = format!(
            "{}..{}",
            bound_str(range.start_bound()),
            bound_str(range.end_bound())
        );
        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Range(desc),
            selectivity: 0.1, // default: 10% selectivity for ranges
        }
    }

    /// Custom predicate with a user-provided description.
    /// Always applied as a post-fetch filter (cannot be pushed into an index).
    pub fn custom<T: Component>(description: &str, selectivity: f64) -> Self {
        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Custom(description.into()),
            selectivity: selectivity.clamp(0.0, 1.0),
        }
    }

    /// Override the default selectivity estimate.
    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity.clamp(0.0, 1.0);
        self
    }

    fn can_use_btree(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq(_) | PredicateKind::Range(_))
    }

    fn can_use_hash(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq(_))
    }

    /// Whether this predicate can be lowered to branchless SIMD comparison.
    fn is_branchless_eligible(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq(_) | PredicateKind::Range(_))
    }
}

impl fmt::Debug for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            PredicateKind::Eq(_) => write!(f, "Eq({})", self.component_name),
            PredicateKind::Range(_) => write!(f, "Range({})", self.component_name),
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
    pub fn estimated_rows(&self) -> f64 {
        self.cost().rows
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        match self {
            PlanNode::Scan {
                query_name,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "{pad}Scan [{query_name}] rows={estimated_rows} cpu={:.1}",
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
                    "{pad}IndexLookup [{index_kind:?} on {component_name}] {predicate} rows={estimated_rows} cpu={:.1}",
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
                    "{pad}Filter [{predicate}] sel={selectivity:.2} rows={:.0} cpu={:.1}",
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
                    "{pad}HashJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
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
                    "{pad}NestedLoopJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
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
        }
    }
}

impl std::error::Error for SubscriptionError {}

#[expect(dead_code)]
struct IndexedPredicate {
    component_name: &'static str,
    index_kind: IndexKind,
    predicate_desc: String,
    selectivity: f64,
    cardinality: usize,
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
            cardinality: witness.cardinality,
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
            cardinality: witness.cardinality,
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
        if !self.errors.is_empty() {
            return Err(self.errors);
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

// ── Scan builder ─────────────────────────────────────────────────────

/// Builder for a single-table scan with optional predicates and joins.
pub struct ScanBuilder<'w> {
    planner: &'w QueryPlanner<'w>,
    query_name: &'static str,
    estimated_rows: usize,
    predicates: Vec<Predicate>,
    joins: Vec<JoinSpec>,
}

struct JoinSpec {
    right_query_name: &'static str,
    right_estimated_rows: usize,
    join_kind: JoinKind,
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
    pub fn join<Q: 'static>(mut self, join_kind: JoinKind) -> Self {
        // Estimate right-side rows from total entity count (conservative).
        let right_rows = self.planner.total_entities;
        self.joins.push(JoinSpec {
            right_query_name: std::any::type_name::<Q>(),
            right_estimated_rows: right_rows,
            join_kind,
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
        let mut index_preds = Vec::new();
        let mut filter_preds = Vec::new();

        for pred in self.predicates {
            if let Some(idx) = self.planner.find_best_index(&pred) {
                index_preds.push((pred, idx));
            } else {
                // Generate warnings for missing indexes.
                self.planner.warn_missing_index(&pred, &mut warnings);
                filter_preds.push(pred);
            }
        }

        // Phase 2: Order index lookups by selectivity (most selective first).
        index_preds.sort_by(|a, b| a.0.selectivity.total_cmp(&b.0.selectivity));

        // Phase 3: Build the plan tree bottom-up.
        let mut node: PlanNode;

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

            // Additional index predicates become filters (could be index
            // intersections in a more advanced planner).
            for (pred, _idx) in index_preds.iter().skip(1) {
                let parent_cost = node.cost();
                node = PlanNode::Filter {
                    predicate: format!("{:?}", pred),
                    selectivity: pred.selectivity,
                    branchless_eligible: pred.is_branchless_eligible(),
                    cost: Cost::filter(parent_cost, pred.selectivity),
                    child: Box::new(node),
                };
            }
        } else {
            // No usable index — full scan.
            node = PlanNode::Scan {
                query_name: self.query_name,
                estimated_rows: self.estimated_rows,
                cost: Cost::scan(self.estimated_rows),
            };
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

            // Choose join strategy based on cardinality.
            // Hash join is better when either side is large (>= 64 rows).
            // Nested loop is better for very small cardinalities.
            let use_hash = left_cost.rows >= 64.0 || right_cost.rows >= 64.0;

            if use_hash {
                // For inner joins, put smaller side on left (build side)
                // for cache locality. Left joins are not commutative — the
                // semantic left side must stay on the left to preserve all
                // its rows, so we only reorder for Inner.
                let (build, probe) =
                    if join.join_kind == JoinKind::Inner && left_cost.rows > right_cost.rows {
                        (right_node, node)
                    } else {
                        (node, right_node)
                    };
                let cost = Cost::hash_join(build.cost(), probe.cost());
                node = PlanNode::HashJoin {
                    left: Box::new(build),
                    right: Box::new(probe),
                    join_kind: join.join_kind,
                    cost,
                };
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
            }
        }

        let opts = VectorizeOpts::default();
        let vec_root = lower_to_vectorized(&node, &opts);
        QueryPlanResult {
            root: node,
            vec_root,
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
/// data. Registering indexes is O(1) (captures cardinality + kind), and plan
/// compilation is O(predicates × indexes) with no heap allocation beyond the
/// plan tree itself.
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
            _world: PhantomData,
        }
    }

    /// Register a `BTreeIndex` for cost-based index selection.
    ///
    /// # Panics
    /// Panics if `T` has not been registered as a component in `world`.
    pub fn add_btree_index<T: Component + Ord + Clone>(
        &mut self,
        index: &BTreeIndex<T>,
        world: &World,
    ) {
        let comp_id = world.component_id::<T>().unwrap_or_else(|| {
            panic!(
                "QueryPlanner::add_btree_index: component `{}` not registered in this World",
                std::any::type_name::<T>()
            )
        });
        self.indexes.insert(
            TypeId::of::<T>(),
            IndexDescriptor {
                component_type: TypeId::of::<T>(),
                component_id: comp_id,
                component_name: std::any::type_name::<T>(),
                kind: IndexKind::BTree,
                cardinality: index.len(),
            },
        );
    }

    /// Register a `HashIndex` for cost-based index selection.
    ///
    /// # Panics
    /// Panics if `T` has not been registered as a component in `world`.
    pub fn add_hash_index<T: Component + std::hash::Hash + Eq + Clone>(
        &mut self,
        index: &HashIndex<T>,
        world: &World,
    ) {
        let comp_id = world.component_id::<T>().unwrap_or_else(|| {
            panic!(
                "QueryPlanner::add_hash_index: component `{}` not registered in this World",
                std::any::type_name::<T>()
            )
        });
        // Only insert if no BTree index is already registered for this type
        // (BTree is strictly more capable than Hash).
        self.indexes
            .entry(TypeId::of::<T>())
            .or_insert(IndexDescriptor {
                component_type: TypeId::of::<T>(),
                component_id: comp_id,
                component_name: std::any::type_name::<T>(),
                kind: IndexKind::Hash,
                cardinality: index.len(),
            });
    }

    /// Start building a scan plan for query type `Q`.
    pub fn scan<Q: 'static>(&'w self) -> ScanBuilder<'w> {
        ScanBuilder {
            planner: self,
            query_name: std::any::type_name::<Q>(),
            estimated_rows: self.total_entities,
            predicates: Vec::new(),
            joins: Vec::new(),
        }
    }

    /// Start building a scan plan with an explicit row estimate.
    ///
    /// Use this when you know the approximate result size from domain
    /// knowledge (e.g., "there are ~500 active players").
    pub fn scan_with_estimate<Q: 'static>(&'w self, estimated_rows: usize) -> ScanBuilder<'w> {
        ScanBuilder {
            planner: self,
            query_name: std::any::type_name::<Q>(),
            estimated_rows,
            predicates: Vec::new(),
            joins: Vec::new(),
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
    fn find_best_index(&self, pred: &Predicate) -> Option<IndexDescriptor> {
        let idx = self.indexes.get(&pred.component_type)?;

        match idx.kind {
            IndexKind::BTree => {
                if pred.can_use_btree() {
                    Some(idx.clone())
                } else {
                    None
                }
            }
            IndexKind::Hash => {
                if pred.can_use_hash() {
                    Some(idx.clone())
                } else {
                    None // range predicate on hash index — can't use it
                }
            }
        }
    }

    /// Generate warnings for predicates that can't use an index.
    fn warn_missing_index(&self, pred: &Predicate, warnings: &mut Vec<PlanWarning>) {
        match &pred.kind {
            PredicateKind::Custom(_) => {
                // Custom predicates never use indexes — no warning needed.
            }
            PredicateKind::Eq(_) => {
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
            PredicateKind::Range(_) => {
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
}

impl CardinalityConstraint {
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
        let pad = "  ".repeat(indent);
        match self {
            VecExecNode::ChunkedScan {
                query_name,
                estimated_rows,
                avg_chunk_size,
                cost,
            } => {
                writeln!(
                    f,
                    "{pad}ChunkedScan [{query_name}] rows={estimated_rows} \
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
                    "{pad}IndexGather [{index_kind:?} on {component_name}] \
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
                    "{pad}SIMDFilter [{predicate}] sel={selectivity:.2} \
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
                    "{pad}PartitionedHashJoin [{join_kind:?}] partitions={partitions} \
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
                    "{pad}BatchNestedLoopJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
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
            let partitions = (build_bytes / opts.l2_cache_bytes).max(1);

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
    pub fn scan<Q: 'static>(&'w self) -> ScanBuilder<'w> {
        self.planner.scan::<Q>()
    }

    /// Start building a scan plan with an explicit row estimate.
    ///
    /// Delegates to the underlying [`QueryPlanner::scan_with_estimate`].
    pub fn scan_with_estimate<Q: 'static>(&'w self, estimated_rows: usize) -> ScanBuilder<'w> {
        self.planner.scan_with_estimate::<Q>(estimated_rows)
    }

    /// Register a btree index with the underlying planner for cost-based
    /// optimization.
    ///
    /// **Compile-time enforcement**: requires `T: HasBTreeIndex<C>`.
    pub fn add_btree_index<C>(&mut self, index: &crate::index::BTreeIndex<C>)
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
    pub fn add_hash_index<C>(&mut self, index: &crate::index::HashIndex<C>)
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
        planner.add_btree_index(&idx, &world);
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
        planner.add_hash_index(&idx, &world);
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
        planner.add_btree_index(&btree, &world);
        planner.add_hash_index(&hash, &world); // should not overwrite
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
        planner.add_hash_index(&idx, &world);

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
        planner.add_btree_index(&idx, &world);

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
        planner.add_hash_index(&idx, &world);

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
            .filter(Predicate::custom::<Score>("score > threshold", 0.5))
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
        planner.add_btree_index(&score_idx, &world);
        planner.add_hash_index(&team_idx, &world);

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
        planner.add_btree_index(&idx, &world);

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
        planner.add_btree_index(&idx, &world);

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
        planner.add_btree_index(&score_idx, &world);

        let plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .filter(Predicate::custom::<Team>("team != 0", 0.8))
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
        planner.add_btree_index(&idx, &world);

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
            .filter(Predicate::custom::<Score>("complex check", 0.5))
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
            .filter(Predicate::custom::<Score>("x > 0", 0.5))
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
        planner.add_btree_index::<Score>(&btree);

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
}
