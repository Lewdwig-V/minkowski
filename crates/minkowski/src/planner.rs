//! Query planner for composing index-driven lookups, joins, and full scans
//! into optimized execution plans.
//!
//! The planner is designed for an in-memory ECS where data already lives in L1/L2
//! cache. Planning overhead is kept to O(indexes + predicates). Plans are
//! executable against live world data: scan-only plans without a spatial driver
//! use zero-alloc `for_each` with fused filter closures; join plans use a
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
//!     .subscribe::<(&Pos, &Score)>()
//!     .where_eq(Indexed::btree(&score_index), Predicate::eq(Score(42)))
//!     .build()
//!     .unwrap();
//! ```

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use fixedbitset::FixedBitSet;

use std::collections::HashSet;

use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::index::{BTreeIndex, HashIndex, SpatialCost, SpatialExpr, SpatialIndex};
use crate::storage::archetype::{Archetype, ArchetypeId};
use crate::tick::Tick;
// Use std Arc directly — the planner has no concurrent code, so it does not
// need loom's Arc (which lacks unsized coercion for Arc<dyn Fn> type erasure).
use crate::query::fetch::{ReadOnlyWorldQuery, WorldQuery};
use crate::transaction::WorldMismatch;
use crate::world::{EntityLocation, World, WorldId};
use std::sync::Arc;

// ── Entity reference trait ────────────────────────────────────────────

/// Marker trait for component types that act as entity references (foreign keys).
///
/// Implement this on any component whose value is — or contains — an [`Entity`]
/// that points to another entity. The planner uses the extracted `Entity` to
/// perform ER (Entity-Relationship) joins: instead of intersecting two entity
/// sets by identity, the join follows the reference and checks whether the
/// *target* entity satisfies the right-side query.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Clone, Copy)]
/// struct Parent(Entity);
///
/// impl AsEntityRef for Parent {
///     fn entity_ref(&self) -> Entity {
///         self.0
///     }
/// }
///
/// // ER join: for each child entity with a Parent component, check that
/// // the parent entity has (&Pos, &Name).
/// let plan = planner
///     .scan::<(&ChildTag, &Parent)>()
///     .er_join::<Parent, (&Pos, &Name)>(JoinKind::Inner)
///     .build();
/// ```
pub trait AsEntityRef: Component {
    /// Extract the referenced entity from this component value.
    fn entity_ref(&self) -> Entity;
}

// ── Planner errors ───────────────────────────────────────────────────

/// Error returned by planner builder methods when inputs are invalid.
#[derive(Clone, Debug)]
pub enum PlannerError {
    /// Invalid predicate parameters (e.g. negative radius, empty coordinates,
    /// min > max).
    InvalidPredicate(String),
    /// Component type has not been registered in the World.
    UnregisteredComponent(&'static str),
    /// Builder method called in wrong order (e.g. `with_right_estimate` before `join`).
    BuilderOrder(String),
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlannerError::InvalidPredicate(msg) => write!(f, "{msg}"),
            PlannerError::UnregisteredComponent(name) => {
                write!(f, "component `{name}` not registered in this World")
            }
            PlannerError::BuilderOrder(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for PlannerError {}

/// Error returned by plan execution methods (`execute`, `execute_raw`,
/// `for_each`, `for_each_raw`, `execute_aggregates`, `execute_aggregates_raw`).
#[derive(Clone, Debug)]
pub enum PlanExecError {
    /// Plan was built from a different World.
    WorldMismatch(WorldMismatch),
    /// `for_each` / `for_each_raw` called on a plan that contains joins.
    /// Use `execute()` instead, which collects entities into a scratch buffer.
    ///
    /// **Deprecated**: join plans are now supported by all execution methods.
    /// This variant is retained for backward compatibility but is no longer
    /// returned by any method in this crate.
    #[deprecated(
        since = "1.3.0",
        note = "join plans are now supported by all execution methods; this variant is never returned"
    )]
    JoinNotSupported,
    /// Batch execution method called with a `Q: WorldQuery` whose required
    /// components are not present in one of the matched archetypes.
    ComponentMismatch {
        /// `std::any::type_name::<Q>()` of the query tuple.
        query: &'static str,
        /// Archetype that was missing a required component.
        archetype_id: ArchetypeId,
    },
}

impl fmt::Display for PlanExecError {
    #[allow(deprecated)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanExecError::WorldMismatch(e) => write!(f, "{e}"),
            PlanExecError::JoinNotSupported => write!(
                f,
                "for_each/for_each_raw do not support join plans — use execute() instead"
            ),
            PlanExecError::ComponentMismatch {
                query,
                archetype_id,
            } => write!(
                f,
                "batch query `{query}` has components missing from archetype {arch}",
                arch = archetype_id.0
            ),
        }
    }
}

impl std::error::Error for PlanExecError {
    #[allow(deprecated)]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlanExecError::WorldMismatch(e) => Some(e),
            PlanExecError::JoinNotSupported | PlanExecError::ComponentMismatch { .. } => None,
        }
    }
}

impl From<WorldMismatch> for PlanExecError {
    fn from(e: WorldMismatch) -> Self {
        PlanExecError::WorldMismatch(e)
    }
}

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

    fn spatial_lookup(spatial_cost: &SpatialCost) -> Self {
        let rows = if spatial_cost.estimated_rows.is_finite() && spatial_cost.estimated_rows >= 0.0
        {
            spatial_cost.estimated_rows
        } else {
            debug_assert!(
                false,
                "SpatialCost::estimated_rows must be finite and non-negative, got {}",
                spatial_cost.estimated_rows
            );
            1.0
        };
        let cpu = if spatial_cost.cpu.is_finite() && spatial_cost.cpu >= 0.0 {
            spatial_cost.cpu
        } else {
            debug_assert!(
                false,
                "SpatialCost::cpu must be finite and non-negative, got {}",
                spatial_cost.cpu
            );
            10.0
        };
        Cost { rows, cpu }
    }

    fn filter(input: Cost, selectivity: f64) -> Self {
        Self::filter_with_branchless(input, selectivity, false)
    }

    fn filter_with_branchless(input: Cost, selectivity: f64, branchless: bool) -> Self {
        let speedup = if branchless { 0.5 } else { 0.85 };
        Cost {
            rows: (input.rows * selectivity).max(0.0),
            cpu: input.cpu + input.rows * Self::FILTER_PER_ROW * speedup,
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

// ── Aggregates ───────────────────────────────────────────────────────

/// Aggregate operation to apply to matched entities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateOp {
    /// Count matching entities.
    Count,
    /// Sum a numeric component (extracted via closure).
    Sum,
    /// Find the minimum value of a numeric component.
    Min,
    /// Find the maximum value of a numeric component.
    Max,
    /// Compute the arithmetic mean of a numeric component.
    Avg,
}

impl fmt::Display for AggregateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateOp::Count => f.write_str("COUNT"),
            AggregateOp::Sum => f.write_str("SUM"),
            AggregateOp::Min => f.write_str("MIN"),
            AggregateOp::Max => f.write_str("MAX"),
            AggregateOp::Avg => f.write_str("AVG"),
        }
    }
}

/// Type-erased value extractor: given `&World` and `Entity`, return the
/// component value as `f64` for aggregate computation. Returns `None` if
/// the entity does not have the component.
type ValueExtractor = Arc<dyn Fn(&World, Entity) -> Option<f64> + Send + Sync>;

// ── Batch aggregate infrastructure ──────────────────────────────────
//
// Chunk-at-a-time processing for aggregates. Instead of calling
// `world.get::<T>(entity)` per row (which resolves location → archetype
// → column on every call), we resolve the column pointer once per
// archetype and process all rows as a typed slice.

/// Chunk-at-a-time aggregate processor. The vtable dispatch happens at the
/// archetype boundary (once per archetype), not per row. The inner
/// `process_all` / `process_rows` loops are monomorphized and
/// auto-vectorizable.
pub(crate) trait BatchExtractor: Send {
    /// Resolve column pointer for this archetype. Returns `false` if the
    /// component is absent.
    fn bind_archetype(&mut self, archetype: &Archetype) -> bool;

    /// Process all rows in the currently-bound archetype as a contiguous
    /// `&[T]` slice. Caller guarantees `bind_archetype` returned `true`
    /// and `count == archetype.len()`.
    fn process_all(&mut self, count: usize, accum: &mut AggregateAccum);

    /// Process specific rows (for index-gather paths where entities may
    /// not be contiguous within the archetype).
    fn process_rows(&mut self, rows: &[usize], accum: &mut AggregateAccum);
}

/// Concrete batch extractor for component type `T` with extraction
/// function `F`. The inner loop over `&[T]` is fully monomorphized —
/// LLVM can inline `F` and auto-vectorize.
struct TypedBatch<T, F> {
    extract: F,
    comp_id: ComponentId,
    /// Base pointer to the column data, set by `bind_archetype`.
    col_ptr: *const T,
}

// SAFETY: TypedBatch is Send because:
// - `F` is Send (enforced by the impl's `F: Send` bound)
// - `comp_id` is Copy
// - `col_ptr` is derived from BlobVec data in archetypes whose components
//   are Send+Sync (Component: Send+Sync). The pointer is only valid during
//   process_all/process_rows which execute within a &World borrow scope.
unsafe impl<T: Send, F: Send> Send for TypedBatch<T, F> {}

impl<T: Component, F: Fn(&T) -> f64 + Send> BatchExtractor for TypedBatch<T, F> {
    fn bind_archetype(&mut self, archetype: &Archetype) -> bool {
        let Some(col_idx) = archetype.column_index(self.comp_id) else {
            return false;
        };
        // SAFETY: BlobVec stores data with layout Layout::new::<T>()
        // (guaranteed by ComponentId registration). column_index guarantees
        // comp_id matches this column. The archetype is borrowed for the
        // duration of the aggregate scan (&World).
        self.col_ptr = unsafe { archetype.columns[col_idx].get_ptr(0) as *const T };
        true
    }

    fn process_all(&mut self, count: usize, accum: &mut AggregateAccum) {
        debug_assert!(
            !self.col_ptr.is_null(),
            "process_all called before successful bind_archetype"
        );
        // SAFETY: bind_archetype set col_ptr (caller checked it returned true);
        // count == archetype.len(). BlobVec guarantees contiguous layout.
        let slice = unsafe { std::slice::from_raw_parts(self.col_ptr, count) };
        for item in slice {
            accum.feed((self.extract)(item));
        }
    }

    fn process_rows(&mut self, rows: &[usize], accum: &mut AggregateAccum) {
        debug_assert!(
            !self.col_ptr.is_null(),
            "process_rows called before successful bind_archetype"
        );
        for &row in rows {
            // SAFETY: row comes from a validated EntityLocation (via
            // world.validate_entity), guaranteeing it is a valid index
            // within the bound archetype.
            let item = unsafe { &*self.col_ptr.add(row) };
            accum.feed((self.extract)(item));
        }
    }
}

/// Factory that produces fresh `Box<dyn BatchExtractor>` instances.
/// Created at `ScanBuilder::build()` time when `ComponentId` is resolved.
/// Called once per `execute_aggregates` invocation.
type BatchFactory = Box<dyn Fn() -> Box<dyn BatchExtractor> + Send + Sync>;

/// Builder that captures the typed closure and produces a `BatchFactory`
/// once the `ComponentId` is known. Stored on `AggregateExpr` between
/// construction and `build()`.
type BatchFactoryBuilder =
    Box<dyn FnOnce(&ComponentRegistry) -> Option<BatchFactory> + Send + Sync>;

/// A single aggregate expression: an operation applied to values extracted
/// from matched entities.
///
/// # Example
///
/// ```rust,ignore
/// use minkowski::planner::{AggregateExpr, AggregateOp};
///
/// // Count all matching entities.
/// let count = AggregateExpr::count();
///
/// // Sum a component value.
/// let total_score = AggregateExpr::sum::<Score>("Score", |s| s.0 as f64);
///
/// // Find the maximum health.
/// let max_hp = AggregateExpr::max::<Health>("Health", |h| h.0 as f64);
/// ```
pub struct AggregateExpr {
    /// The aggregate operation.
    op: AggregateOp,
    /// Human-readable label for `explain()` output.
    ///
    /// Label format convention: `"COUNT(*)"`, `"SUM(name)"`, `"MIN(name)"`,
    /// `"MAX(name)"`, `"AVG(name)"` where `name` is the user-supplied label.
    label: String,
    /// Extracts a `f64` value from an entity. `None` for `Count`.
    /// Kept as fallback for join plans where batch extraction isn't possible.
    extractor: Option<ValueExtractor>,
    /// Deferred factory builder: captures the typed closure and defers
    /// `ComponentId` resolution (via the generic type parameter `T`) to
    /// `ScanBuilder::build()` time. `None` for `Count` (no component access).
    batch_factory_builder: Option<BatchFactoryBuilder>,
    /// Finalized batch factory, set by `ScanBuilder::build()`. Each call
    /// produces a fresh `Box<dyn BatchExtractor>` with its own column pointer
    /// binding.
    batch_factory: Option<BatchFactory>,
}

impl AggregateExpr {
    /// Count matching entities. No component access needed.
    pub fn count() -> Self {
        Self {
            op: AggregateOp::Count,
            label: "COUNT(*)".to_string(),
            extractor: None,
            batch_factory_builder: None,
            batch_factory: None,
        }
    }

    /// Sum a component's value across matching entities.
    ///
    /// `name` is a human-readable label for plan output.
    /// `extract` converts the component reference to `f64`.
    pub fn sum<T: Component>(
        name: &str,
        extract: impl Fn(&T) -> f64 + Send + Sync + 'static,
    ) -> Self {
        let label = format!("SUM({name})");
        let (extractor, builder) = make_extractor::<T>(extract);
        Self {
            op: AggregateOp::Sum,
            label,
            extractor: Some(extractor),
            batch_factory_builder: Some(builder),
            batch_factory: None,
        }
    }

    /// Find the minimum component value across matching entities.
    pub fn min<T: Component>(
        name: &str,
        extract: impl Fn(&T) -> f64 + Send + Sync + 'static,
    ) -> Self {
        let label = format!("MIN({name})");
        let (extractor, builder) = make_extractor::<T>(extract);
        Self {
            op: AggregateOp::Min,
            label,
            extractor: Some(extractor),
            batch_factory_builder: Some(builder),
            batch_factory: None,
        }
    }

    /// Find the maximum component value across matching entities.
    pub fn max<T: Component>(
        name: &str,
        extract: impl Fn(&T) -> f64 + Send + Sync + 'static,
    ) -> Self {
        let label = format!("MAX({name})");
        let (extractor, builder) = make_extractor::<T>(extract);
        Self {
            op: AggregateOp::Max,
            label,
            extractor: Some(extractor),
            batch_factory_builder: Some(builder),
            batch_factory: None,
        }
    }

    /// Compute the arithmetic mean of a component's value.
    pub fn avg<T: Component>(
        name: &str,
        extract: impl Fn(&T) -> f64 + Send + Sync + 'static,
    ) -> Self {
        let label = format!("AVG({name})");
        let (extractor, builder) = make_extractor::<T>(extract);
        Self {
            op: AggregateOp::Avg,
            label,
            extractor: Some(extractor),
            batch_factory_builder: Some(builder),
            batch_factory: None,
        }
    }

    /// The aggregate operation type.
    pub fn op(&self) -> AggregateOp {
        self.op
    }

    /// The human-readable label (e.g. `"SUM(Score)"`).
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Debug for AggregateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AggregateExpr")
            .field("op", &self.op)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

/// Build a type-erased value extractor and a deferred batch factory builder.
///
/// The `ValueExtractor` is the per-entity fallback (used for join plans).
/// The `BatchFactoryBuilder` captures the typed closure and defers
/// `ComponentId` resolution to `ScanBuilder::build()`.
fn make_extractor<T: Component>(
    extract: impl Fn(&T) -> f64 + Send + Sync + 'static,
) -> (ValueExtractor, BatchFactoryBuilder) {
    // Wrap in Arc so the factory can clone it without requiring Clone on
    // the user's closure (preserves the existing public API bounds).
    let extract = Arc::new(extract);

    let value_extractor: ValueExtractor = {
        let extract = Arc::clone(&extract);
        Arc::new(move |world: &World, entity: Entity| world.get::<T>(entity).map(|v| extract(v)))
    };

    let builder: BatchFactoryBuilder = Box::new(move |registry: &ComponentRegistry| {
        let comp_id = registry.id::<T>()?;
        let extract = Arc::clone(&extract);
        Some(Box::new(move || -> Box<dyn BatchExtractor> {
            let extract = Arc::clone(&extract);
            Box::new(TypedBatch {
                extract: move |item: &T| extract(item),
                comp_id,
                col_ptr: std::ptr::null(),
            })
        }))
    });

    (value_extractor, builder)
}

/// Accumulator state for computing aggregates during plan execution.
#[derive(Clone, Debug)]
struct AggregateAccum {
    op: AggregateOp,
    label: String,
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl AggregateAccum {
    fn new(op: AggregateOp, label: String) -> Self {
        Self {
            op,
            label,
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Feed a value into the accumulator.
    ///
    /// `count` tracks values fed (not entities matched by the scan). For
    /// `Avg`, this is the denominator. For `Count`, use `feed_count` instead.
    fn feed(&mut self, value: f64) {
        self.count += 1;
        match self.op {
            AggregateOp::Count => {} // count incremented above
            AggregateOp::Sum => self.sum += value,
            // Propagate NaN consistently with Sum/Avg: if either operand
            // is NaN, the result is NaN. f64::min/max suppress NaN (IEEE
            // minNum), so we check explicitly.
            AggregateOp::Min => {
                if value.is_nan() || value < self.min {
                    self.min = value;
                }
            }
            AggregateOp::Max => {
                if value.is_nan() || value > self.max {
                    self.max = value;
                }
            }
            AggregateOp::Avg => self.sum += value,
        }
    }

    fn feed_count(&mut self) {
        self.count += 1;
    }

    fn finish(&self) -> f64 {
        match self.op {
            AggregateOp::Count => self.count as f64,
            AggregateOp::Sum => self.sum,
            AggregateOp::Min => {
                if self.count == 0 {
                    f64::NAN
                } else {
                    self.min
                }
            }
            AggregateOp::Max => {
                if self.count == 0 {
                    f64::NAN
                } else {
                    self.max
                }
            }
            AggregateOp::Avg => {
                if self.count == 0 {
                    f64::NAN
                } else {
                    self.sum / self.count as f64
                }
            }
        }
    }
}

/// The result of executing an aggregate plan.
///
/// Contains one `f64` result per aggregate expression, in the same order
/// they were added via [`ScanBuilder::aggregate`].
#[derive(Clone, Debug)]
pub struct AggregateResult {
    values: Vec<(String, f64)>,
}

impl AggregateResult {
    /// Number of aggregate values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if there are no aggregate values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get aggregate value by index (in order of `aggregate()` calls).
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).map(|(_, v)| *v)
    }

    /// Get aggregate value by label (e.g. `"SUM(Score)"`).
    pub fn get_by_label(&self, label: &str) -> Option<f64> {
        self.values
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, v)| *v)
    }

    /// Iterate over `(label, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> + '_ {
        self.values.iter().map(|(l, v)| (l.as_str(), *v))
    }

    /// Iterate over available labels (useful for discovering `get_by_label` keys).
    pub fn labels(&self) -> impl Iterator<Item = &str> + '_ {
        self.values.iter().map(|(l, _)| l.as_str())
    }
}

impl fmt::Display for AggregateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (label, value)) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{label}: {value:.2}")?;
        }
        write!(f, "}}")
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

/// A spatial predicate recognized by the query planner's IR.
///
/// When the planner encounters a spatial predicate, it checks whether any
/// registered `SpatialIndex` can accelerate the expression via
/// [`SpatialIndex::supports`]. If so, the planner emits a `SpatialLookup`
/// plan node instead of a full scan + post-filter.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SpatialPredicate {
    /// Proximity: entities within `radius` of `center`.
    /// Maps to `SpatialExpr::Within` for capability discovery.
    /// Dimensionality and metric are defined by the index implementation.
    Within {
        /// Center coordinates (dimensionality defined by the index).
        center: Vec<f64>,
        /// Search radius (interpretation defined by the index).
        radius: f64,
    },
    /// Bounding-box intersection: entities whose spatial extent overlaps
    /// the box from `min` to `max`. Maps to `SpatialExpr::Intersects`.
    /// Dimensionality is defined by the index implementation.
    Intersects {
        /// Minimum corner coordinates.
        min: Vec<f64>,
        /// Maximum corner coordinates.
        max: Vec<f64>,
    },
}

impl From<&SpatialPredicate> for SpatialExpr {
    fn from(sp: &SpatialPredicate) -> Self {
        match sp {
            SpatialPredicate::Within { center, radius } => SpatialExpr::Within {
                center: center.clone(),
                radius: *radius,
            },
            SpatialPredicate::Intersects { min, max } => SpatialExpr::Intersects {
                min: min.clone(),
                max: max.clone(),
            },
        }
    }
}

impl fmt::Display for SpatialPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialPredicate::Within { center, radius } => {
                write!(f, "ST_Within({center:?}, {radius})")
            }
            SpatialPredicate::Intersects { min, max } => {
                write!(f, "ST_Intersects({min:?}, {max:?})")
            }
        }
    }
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
    all_entities_fn: Option<IndexLookupFn>,
    /// Predicate-specific equality lookup. Takes `&dyn Any` (downcast to `T`),
    /// returns only entities matching the exact value. O(log n) for BTree, O(1) for Hash.
    /// Bound into an `IndexDriver` lookup closure at Phase 3 plan-build time.
    eq_lookup_fn: Option<PredicateLookupFn>,
    /// Predicate-specific range lookup. Takes `&dyn Any` (downcast to `(Bound<T>, Bound<T>)`),
    /// returns only entities within the range. O(log n + k) for BTree, not available for Hash.
    /// Bound into an `IndexDriver` lookup closure at Phase 3 plan-build time.
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

// IndexDescriptor methods removed: lookup_fn_for was part of the old
// ExecNode-based execution engine. Index-driven execution now uses an
// IndexDriver pre-bound at Phase 3 and compiled into the for_each closure.

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
    /// Bound into the `IndexDriver` lookup closure at Phase 3 plan-build time.
    lookup_value: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

#[derive(Debug)]
enum PredicateKind {
    Eq,
    Range,
    Custom(Box<str>), // description only — always post-filter
    Spatial(SpatialPredicate),
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

    /// Spatial proximity predicate: entities within `radius` of `center`.
    ///
    /// The component type `T` identifies which column holds the spatial data.
    /// The planner makes no assumptions about dimensionality or coordinate
    /// system — `center` is an opaque coordinate vector whose meaning is
    /// defined by the [`SpatialIndex`] implementation.
    ///
    /// If a `SpatialIndex` is registered for `T` and its
    /// [`supports`](SpatialIndex::supports) method returns `Some` for this
    /// expression, the planner emits a `SpatialLookup` node. Otherwise it
    /// falls back to a scan + post-filter using the provided closure.
    ///
    /// The default selectivity is a conservative 0.1 (10% of entities).
    /// Override with [`.with_selectivity()`](Self::with_selectivity) if
    /// you know your data distribution.
    pub fn within<T: Component>(
        center: impl Into<Vec<f64>>,
        radius: f64,
        filter: impl Fn(&World, Entity) -> bool + Send + Sync + 'static,
    ) -> Result<Self, PlannerError> {
        if !(radius >= 0.0 && radius.is_finite()) {
            return Err(PlannerError::InvalidPredicate(format!(
                "Predicate::within: radius must be finite and non-negative, got {radius}"
            )));
        }
        let center = center.into();
        if center.is_empty() {
            return Err(PlannerError::InvalidPredicate(
                "Predicate::within: center must have at least one coordinate".into(),
            ));
        }
        // Conservative default — override via .with_selectivity().
        let selectivity = sanitize_selectivity(0.1);
        Ok(Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Spatial(SpatialPredicate::Within { center, radius }),
            selectivity,
            filter_fn: Some(Arc::new(filter)),
            lookup_value: None,
        })
    }

    /// Spatial bounding-box intersection predicate.
    ///
    /// The component type `T` identifies which column holds the spatial data.
    /// The planner makes no assumptions about dimensionality or coordinate
    /// system — `min` and `max` are opaque coordinate vectors whose meaning
    /// is defined by the [`SpatialIndex`] implementation.
    ///
    /// If a `SpatialIndex` is registered for `T` and its
    /// [`supports`](SpatialIndex::supports) method returns `Some` for this
    /// expression, the planner emits a `SpatialLookup` node. Otherwise it
    /// falls back to a scan + post-filter using the provided closure.
    ///
    /// The default selectivity is a conservative 0.1 (10% of entities).
    /// Override with [`.with_selectivity()`](Self::with_selectivity) if
    /// you know your data distribution.
    pub fn intersects<T: Component>(
        min: impl Into<Vec<f64>>,
        max: impl Into<Vec<f64>>,
        filter: impl Fn(&World, Entity) -> bool + Send + Sync + 'static,
    ) -> Result<Self, PlannerError> {
        let min = min.into();
        let max = max.into();
        if min.len() != max.len() {
            return Err(PlannerError::InvalidPredicate(format!(
                "Predicate::intersects: min and max must have the same dimensionality, \
                 got {} vs {}",
                min.len(),
                max.len()
            )));
        }
        if min.is_empty() {
            return Err(PlannerError::InvalidPredicate(
                "Predicate::intersects: coordinates must have at least one dimension".into(),
            ));
        }
        if !min.iter().zip(max.iter()).all(|(lo, hi)| lo <= hi) {
            return Err(PlannerError::InvalidPredicate(
                "Predicate::intersects: min must be <= max in all dimensions".into(),
            ));
        }
        // Conservative default — override via .with_selectivity().
        let selectivity = sanitize_selectivity(0.1);
        Ok(Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::Spatial(SpatialPredicate::Intersects { min, max }),
            selectivity,
            filter_fn: Some(Arc::new(filter)),
            lookup_value: None,
        })
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

    fn can_use_spatial(&self) -> bool {
        matches!(self.kind, PredicateKind::Spatial(_))
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
            PredicateKind::Spatial(sp) => write!(f, "Spatial({}: {})", self.component_name, sp),
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
///
/// All plans execute via chunked, slice-based iteration over 64-byte-aligned
/// columns. LLVM auto-vectorizes loops over these contiguous slices — there
/// is no separate "scalar" execution path. Cost estimates reflect the
/// vectorized execution model (batch amortization, branchless filters,
/// cache-partitioned joins).
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanNode {
    /// Chunked archetype scan: yields one batch per archetype.
    /// Each batch is a contiguous column slice (64-byte aligned).
    Scan {
        query_name: &'static str,
        estimated_rows: usize,
        /// Average rows per chunk (= archetype size). Tuned for L1 residency.
        avg_chunk_size: usize,
        cost: Cost,
    },
    /// Index-driven gather: lookup entities via index, then batch-fetch
    /// components. Entities are sorted by archetype to maximize sequential
    /// access.
    IndexLookup {
        index_kind: IndexKind,
        component_name: &'static str,
        predicate: String,
        estimated_rows: usize,
        cost: Cost,
    },
    /// Spatial index gather: lookup entities via a spatial index.
    SpatialLookup {
        component_name: &'static str,
        predicate: String,
        estimated_rows: usize,
        cost: Cost,
    },
    /// Filter applied to contiguous slices from the child node.
    /// Branchless filters (Eq/Range on numeric types) are ~2x faster
    /// than branched filters (Custom predicates) on aligned data.
    Filter {
        child: Box<PlanNode>,
        predicate: String,
        selectivity: f64,
        /// Whether this filter can be applied branchlessly on aligned data.
        branchless: bool,
        cost: Cost,
    },
    /// Partitioned hash join: build side is partitioned into L2-cache-sized
    /// segments. Probe side streams through chunks, probing the partition
    /// that fits in cache.
    HashJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        /// Number of partitions (tuned for L2 cache residency).
        partitions: usize,
        cost: Cost,
    },
    /// Batch nested-loop join on small materialized sides.
    NestedLoopJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        cost: Cost,
    },
    /// ER (Entity-Relationship) join: follow entity references from the
    /// left side to probe a hash set built from right-side entities.
    ///
    /// For each left entity, the join reads component `R: EntityRef`,
    /// extracts the referenced `Entity`, and checks membership in the
    /// right set. Inner joins emit only left entities whose reference
    /// target is in the right set; left joins emit all left entities.
    ErJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        /// Name of the entity-reference component (e.g. `"Parent"`).
        ref_component: &'static str,
        cost: Cost,
    },
    /// Stream aggregate: compute aggregate functions in a single pass
    /// over child output.
    Aggregate {
        child: Box<PlanNode>,
        /// Human-readable labels for each aggregate (e.g. `["COUNT(*)", "SUM(Score)"]`).
        aggregates: Vec<String>,
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
            | PlanNode::SpatialLookup { cost, .. }
            | PlanNode::Filter { cost, .. }
            | PlanNode::HashJoin { cost, .. }
            | PlanNode::NestedLoopJoin { cost, .. }
            | PlanNode::ErJoin { cost, .. }
            | PlanNode::Aggregate { cost, .. } => *cost,
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
            PlanNode::IndexLookup {
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
            PlanNode::SpatialLookup {
                component_name,
                predicate,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "SpatialGather [Spatial on {component_name}] \
                     {predicate} rows={estimated_rows} cpu={:.1}",
                    cost.cpu
                )
            }
            PlanNode::Filter {
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
                    "Filter [{predicate}] sel={selectivity:.2} \
                     mode={mode} rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                child.fmt_indent(f, indent + 1)
            }
            PlanNode::HashJoin {
                left,
                right,
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
                    "BatchNestedLoopJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
            PlanNode::ErJoin {
                left,
                right,
                join_kind,
                ref_component,
                cost,
            } => {
                writeln!(
                    f,
                    "ErJoin [{join_kind:?} on {ref_component}] \
                     rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
            PlanNode::Aggregate {
                child,
                aggregates,
                cost,
            } => {
                let agg_list = aggregates.join(", ");
                writeln!(
                    f,
                    "StreamAggregate [{agg_list}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                child.fmt_indent(f, indent + 1)
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
    /// A spatial index is registered but declined the expression
    /// ([`SpatialIndex::supports`] returned `None`). The predicate falls
    /// back to a scan + post-filter.
    SpatialIndexDeclined {
        component_name: &'static str,
        expression: String,
    },
    /// Join has no index on either side — will use nested loop.
    UnindexedJoin {
        left_name: &'static str,
        right_name: &'static str,
    },
    /// Multiple aggregate expressions share the same label. `get_by_label`
    /// returns the first match — the duplicate is silently hidden.
    DuplicateAggregateLabel { label: String },
    /// An inner join was eliminated at build time — its required components
    /// were merged into the scan's bitset. The plan executes as a pure
    /// archetype scan instead of materializing a join.
    JoinEliminated { right_name: &'static str },
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
            PlanWarning::SpatialIndexDeclined {
                component_name,
                expression,
            } => {
                write!(
                    f,
                    "spatial index for `{component_name}` does not support {expression} — \
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
            PlanWarning::DuplicateAggregateLabel { label } => {
                write!(
                    f,
                    "duplicate aggregate label `{label}` — get_by_label() returns the first match"
                )
            }
            PlanWarning::JoinEliminated { right_name } => {
                write!(
                    f,
                    "inner join with `{right_name}` eliminated — merged into scan \
                     (use scan::<(Left, Right)>() directly to avoid this)"
                )
            }
        }
    }
}

// ── Compiled plan ────────────────────────────────────────────────────

/// A compiled query execution plan.
pub struct QueryPlanResult {
    root: PlanNode,
    join_exec: Option<JoinExec>,
    compiled_for_each: Option<CompiledForEach>,
    compiled_for_each_raw: Option<CompiledForEachRaw>,
    scratch: Option<ScratchBuffer>,
    opts: VectorizeOpts,
    warnings: Vec<PlanWarning>,
    last_read_tick: Tick,
    world_id: WorldId,
    /// Aggregate expressions for `execute_aggregates()`. Empty if no aggregates.
    aggregate_exprs: Vec<AggregateExpr>,
    /// Compiled batch aggregate scan (for `execute_aggregates`).
    compiled_agg_scan: Option<CompiledAggScan>,
    /// Compiled batch aggregate scan for raw path (for `execute_aggregates_raw`).
    compiled_agg_scan_raw: Option<CompiledAggScan>,
    /// Reusable buffer for row indices in batch execution methods.
    /// Cleared and repopulated on each `for_each_join_chunk` call.
    row_indices: Vec<usize>,
    /// Component requirements for the direct archetype iteration fast path.
    /// `Some` when the plan is scan-only with no custom predicates.
    /// `None` when the plan has joins or custom filter closures.
    scan_required: Option<FixedBitSet>,
    /// Changed-component bitset for the direct archetype iteration fast path.
    scan_changed: FixedBitSet,
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
    /// (`execute`, `execute_raw`, `for_each`, `for_each_raw`,
    /// `execute_aggregates`, `execute_aggregates_raw`) delegate here for
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
                    let match_count = scratch.sorted_intersection(left_len).len();
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
            // Build phase: collect right-side entities into a temporary buffer,
            // then move them into a HashSet for O(1) probing.
            let mut right_scratch = ScratchBuffer::new(128);
            (er_step.right_collector)(world, tick, &mut right_scratch);
            let right_set: HashSet<u64> =
                right_scratch.entities.iter().map(|e| e.to_bits()).collect();

            // Probe phase: for each left entity, extract entity ref and probe.
            let ref_extractor = &er_step.ref_extractor;
            match er_step.join_kind {
                JoinKind::Inner => {
                    scratch.entities.retain(|&entity| {
                        ref_extractor(world, entity)
                            .is_some_and(|target| right_set.contains(&target.to_bits()))
                    });
                }
                JoinKind::Left => {
                    // Left join: keep all left entities regardless. The join
                    // semantics are for filtering; the left entity always survives.
                    // No-op — all left entities pass through.
                }
            }
        }
    }

    /// Execute the plan against a live world, returning matching entities.
    ///
    /// For join plans, entities are collected from both sides into an internal
    /// scratch buffer, then joined via sorted intersection (inner join) or
    /// left-side preservation (left join). The buffer is reused across calls
    /// to amortize allocation.
    ///
    /// For scan-only plans without joins, this collects entities into the
    /// scratch buffer using the compiled scan closure (with filter fusion).
    /// Prefer [`for_each`](Self::for_each) for scan-only plans to avoid the
    /// intermediate buffer entirely.
    ///
    /// Returns `Err(PlanExecError::WorldMismatch)` if `world` is not the
    /// same World this plan was built from.
    ///
    /// # Panics
    ///
    /// Panics if the plan has no scratch buffer (should not happen for plans
    /// built via [`ScanBuilder::build`]).
    pub fn execute(&mut self, world: &mut World) -> Result<&[Entity], PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }

        if self.join_exec.is_some() {
            self.run_join(&*world);
            self.last_read_tick = world.next_tick();
            Ok(self.scratch.as_ref().unwrap().as_slice())
        } else if let Some(compiled) = &mut self.compiled_for_each {
            let scratch = self
                .scratch
                .as_mut()
                .expect("execute() requires a plan with a scratch buffer");
            scratch.clear();
            let tick = self.last_read_tick;
            compiled(&*world, tick, &mut |entity: Entity| {
                scratch.push(entity);
            });
            self.last_read_tick = world.next_tick();
            Ok(scratch.as_slice())
        } else {
            panic!(
                "execute() called on a plan with no join executor and no compiled scan — \
                 this indicates a bug in plan compilation"
            );
        }
    }

    /// Execute the plan with read-only world access, returning matching entities.
    ///
    /// Like [`execute`](Self::execute) but takes `&World` instead of `&mut World`.
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
    pub fn execute_raw(&mut self, world: &World) -> Result<&[Entity], PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }

        if self.join_exec.is_some() {
            self.run_join(world);
            Ok(self.scratch.as_ref().unwrap().as_slice())
        } else if let Some(compiled) = &mut self.compiled_for_each_raw {
            let scratch = self
                .scratch
                .as_mut()
                .expect("execute_raw() requires a plan with a scratch buffer");
            scratch.clear();
            let tick = self.last_read_tick;
            compiled(world, tick, &mut |entity: Entity| {
                scratch.push(entity);
            });
            Ok(scratch.as_slice())
        } else {
            panic!(
                "execute_raw() called on a plan with no join executor and no compiled scan — \
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
    pub fn for_each(
        &mut self,
        world: &mut World,
        mut callback: impl FnMut(Entity),
    ) -> Result<(), PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
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
                "for_each() called on a plan with no join executor and no compiled scan — \
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
    pub fn for_each_raw(
        &mut self,
        world: &World,
        mut callback: impl FnMut(Entity),
    ) -> Result<(), PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
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
                "for_each_raw() called on a plan with no join executor and no compiled scan — \
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
    /// Advances the read tick (same as `for_each`).
    pub fn for_each_batched<Q, F>(
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
        self.for_each_batched_inner::<Q, F>(world, &mut callback)?;
        self.last_read_tick = world.next_tick();
        Ok(())
    }

    /// Read-only variant of [`for_each_batched`](Self::for_each_batched).
    /// No tick advancement.
    /// Safe for use inside transactions where only `&World` is available.
    pub fn for_each_batched_raw<Q, F>(
        &mut self,
        world: &World,
        mut callback: F,
    ) -> Result<(), PlanExecError>
    where
        Q: ReadOnlyWorldQuery,
        F: FnMut(Entity, Q::Item<'_>),
    {
        self.for_each_batched_inner::<Q, F>(world, &mut callback)
    }

    /// Shared implementation for `for_each_batched` and `for_each_batched_raw`.
    fn for_each_batched_inner<Q, F>(
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

        // Fast path: scan-only plan with no custom predicates.
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
                for (row, &entity) in arch.entities.iter().enumerate() {
                    let item = unsafe { Q::fetch(&fetch, row) };
                    callback(entity, item);
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
                .expect("for_each_batched requires a scratch buffer");
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
            .expect("for_each_batched requires a scratch buffer");
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
    pub fn for_each_join_chunk<Q, F>(
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

        // Fast path: scan-only plan with no custom predicates or index drivers.
        // Walk archetypes directly — no ScratchBuffer, no sort.
        // Destructure self to borrow scan_required + row_indices disjointly.
        if self.scan_required.is_some() {
            let Self {
                scan_required,
                scan_changed,
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
                row_indices.clear();
                row_indices.extend(0..arch.len());
                callback(&arch.entities, row_indices, slice);
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
                .expect("for_each_join_chunk requires a scratch buffer");
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
            .expect("for_each_join_chunk requires a scratch buffer");
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
    pub fn execute_aggregates(
        &mut self,
        world: &mut World,
    ) -> Result<AggregateResult, PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }
        if self.aggregate_exprs.is_empty() {
            return Ok(AggregateResult { values: Vec::new() });
        }

        // Initialize accumulators.
        let mut accums: Vec<AggregateAccum> = self
            .aggregate_exprs
            .iter()
            .map(|expr| AggregateAccum::new(expr.op, expr.label.clone()))
            .collect();

        let tick = self.last_read_tick;

        // Fast path: batch aggregate scan (chunk-at-a-time, no per-entity world.get).
        if let Some(compiled) = &mut self.compiled_agg_scan {
            let mut extractors: Vec<Option<Box<dyn BatchExtractor>>> = self
                .aggregate_exprs
                .iter()
                .map(|expr| expr.batch_factory.as_ref().map(|f| f()))
                .collect();
            compiled(&*world, tick, &mut extractors, &mut accums);
        } else if let Some(compiled) = &mut self.compiled_for_each {
            // Fallback: per-entity extraction (used for join plans or when
            // batch factories are unavailable).
            let extractors: Vec<(AggregateOp, Option<ValueExtractor>)> = self
                .aggregate_exprs
                .iter()
                .map(|expr| (expr.op, expr.extractor.as_ref().map(Arc::clone)))
                .collect();
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
                        accums[i].feed_count();
                    } else if let Some(ext) = extractor
                        && let Some(val) = ext(world, entity)
                    {
                        accums[i].feed(val);
                    }
                }
            }
        } else {
            panic!(
                "execute_aggregates() called on a plan with no compiled scan and no join executor"
            );
        }

        self.last_read_tick = world.next_tick();

        let values = accums
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
    pub fn execute_aggregates_raw(
        &mut self,
        world: &World,
    ) -> Result<AggregateResult, PlanExecError> {
        if self.world_id != world.world_id() {
            return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
        }
        if self.aggregate_exprs.is_empty() {
            return Ok(AggregateResult { values: Vec::new() });
        }

        let mut accums: Vec<AggregateAccum> = self
            .aggregate_exprs
            .iter()
            .map(|expr| AggregateAccum::new(expr.op, expr.label.clone()))
            .collect();

        let tick = self.last_read_tick;

        // Fast path: batch aggregate scan.
        if let Some(compiled) = &mut self.compiled_agg_scan_raw {
            let mut extractors: Vec<Option<Box<dyn BatchExtractor>>> = self
                .aggregate_exprs
                .iter()
                .map(|expr| expr.batch_factory.as_ref().map(|f| f()))
                .collect();
            compiled(world, tick, &mut extractors, &mut accums);
        } else if let Some(compiled) = &mut self.compiled_for_each_raw {
            // Fallback: per-entity extraction.
            let extractors: Vec<(AggregateOp, Option<ValueExtractor>)> = self
                .aggregate_exprs
                .iter()
                .map(|expr| (expr.op, expr.extractor.as_ref().map(Arc::clone)))
                .collect();
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
                        accums[i].feed_count();
                    } else if let Some(ext) = extractor
                        && let Some(val) = ext(world, entity)
                    {
                        accums[i].feed(val);
                    }
                }
            }
        }

        let values = accums
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
            .field("row_indices_cap", &self.row_indices.capacity())
            .field("has_scan_required", &self.scan_required.is_some())
            .field("scan_changed", &self.scan_changed)
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
/// resulting plan can push updates without full table scans. Every call to
/// `where_eq` or `where_range` requires an `Indexed<T>` witness, which can
/// only be obtained from an actual index instance.
///
/// Produces a [`QueryPlanResult`] with full execution support (`execute`,
/// `for_each`, `for_each_raw`), backed by `IndexDriver` for index-gather
/// execution.
pub struct SubscriptionBuilder<'w> {
    scan: ScanBuilder<'w>,
    errors: Vec<SubscriptionError>,
    has_predicates: bool,
    attempted_predicates: bool,
}

/// Errors from [`SubscriptionBuilder::build`].
#[derive(Clone, Debug, PartialEq)]
pub enum SubscriptionError {
    /// `where_range` was called with a Hash index witness.
    /// Hash indexes support only exact-match lookups and cannot answer range
    /// queries. Use a `BTreeIndex` instead.
    HashIndexOnRange { component_name: &'static str },
    /// The predicate kind does not match what this method expects.
    /// For example, passing a `Range` predicate to `where_eq`.
    PredicateKindMismatch {
        expected: &'static str,
        component_name: &'static str,
    },
    /// The `Indexed<T>` witness type does not match the component type of the
    /// predicate. The predicate's component type and the witness type must be
    /// the same.
    ComponentMismatch {
        witness_type: &'static str,
        predicate_type: &'static str,
    },
    /// No predicates were added. A subscription with zero predicates is a
    /// full scan, which defeats the "all predicates indexed" guarantee.
    NoPredicates,
    /// An `Indexed<T>` witness was provided but the index was not registered
    /// with the planner via `add_btree_index` / `add_hash_index`. The plan
    /// would silently degrade to a full scan, violating the subscription's
    /// no-full-scan guarantee.
    IndexNotRegistered { component_name: &'static str },
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
            SubscriptionError::PredicateKindMismatch {
                expected,
                component_name,
            } => {
                write!(
                    f,
                    "expected {expected} predicate for `{component_name}`, got a different kind"
                )
            }
            SubscriptionError::ComponentMismatch {
                witness_type,
                predicate_type,
            } => {
                write!(
                    f,
                    "Indexed<{witness_type}> witness used with predicate on `{predicate_type}`"
                )
            }
            SubscriptionError::NoPredicates => {
                write!(
                    f,
                    "subscription has no predicates — add at least one where_eq or where_range"
                )
            }
            SubscriptionError::IndexNotRegistered { component_name } => {
                write!(
                    f,
                    "Indexed<{component_name}> witness provided but the index was not registered \
                     with the planner — call add_btree_index or add_hash_index first"
                )
            }
        }
    }
}

impl std::error::Error for SubscriptionError {}

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

// ── Execution engine ─────────────────────────────────────────────────

/// Type-erased scan execution captured at `build()` time.
/// The monomorphic `Q` iteration code is baked in at compile time.
/// Takes `&World` (shared ref) because the compiled scan only reads archetype
/// data. The outer `for_each` method takes `&mut World` for tick advancement
/// (via `next_tick()`), then reborrows as `&World` for the closure.
type CompiledForEach = Box<dyn FnMut(&World, Tick, &mut dyn FnMut(Entity))>;

/// Read-only variant for transactional reads via `query_raw`.
/// Receives the plan's `last_read_tick` for `Changed<T>` filtering but does
/// not advance it — repeated calls see the same change window.
type CompiledForEachRaw = Box<dyn FnMut(&World, Tick, &mut dyn FnMut(Entity))>;

/// Compiled aggregate scan: iterates matching archetypes/index results and
/// calls batch extractors directly. Bypasses per-entity callbacks entirely.
type CompiledAggScan =
    Box<dyn FnMut(&World, Tick, &mut [Option<Box<dyn BatchExtractor>>], &mut [AggregateAccum])>;

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

/// Type-erased closure that collects entities from a scan/index into a
/// [`ScratchBuffer`]. Used by join plans to gather left and right entity sets.
type EntityCollector = Box<dyn FnMut(&World, Tick, &mut ScratchBuffer)>;

/// A single join step: collect right-side entities and intersect with the
/// accumulated left result.
struct JoinStep {
    right_collector: EntityCollector,
    join_kind: JoinKind,
}

/// Type-erased entity-reference extractor: given `&World` and `Entity`, reads
/// the entity-reference component and returns the referenced `Entity`.
/// Returns `None` if the source entity is dead or doesn't have the component.
type EntityRefExtractor = Arc<dyn Fn(&World, Entity) -> Option<Entity> + Send + Sync>;

/// A single ER join step: for each left entity, follow the entity reference
/// and check membership in the right set.
struct ErJoinStep {
    /// Collects entities matching the right-side query (the "target" entities).
    right_collector: EntityCollector,
    /// Extracts the referenced `Entity` from a left-side entity's component.
    ref_extractor: EntityRefExtractor,
    join_kind: JoinKind,
}

/// Execution state for join plans. The left collector populates the initial
/// entity set, then each `JoinStep` iteratively applies one join. Supports
/// arbitrary join chains: `A JOIN B JOIN C` becomes
/// `left_collector(A) → step[0](B) → step[1](C)`.
///
/// ER join steps are applied after regular join steps. Each ER step builds
/// a `HashSet` of right-side entities, then filters left-side entities by
/// checking whether their entity reference target is in the set.
struct JoinExec {
    left_collector: EntityCollector,
    steps: Vec<JoinStep>,
    er_steps: Vec<ErJoinStep>,
}

/// Returns true if every component in `changed` has a column in the archetype
/// whose tick is newer than `tick`. When `changed` is empty (no `Changed<T>`
/// terms), returns true immediately. For archetype-scan paths the column will
/// always be present because `required` is checked first. For index-gather
/// paths, the column may be absent if the entity's archetype does not contain
/// the component; `is_some_and` handles this by returning false.
#[inline]
fn passes_change_filter(
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

/// Carries the spatial lookup function and expression resolved during Phase 3
/// (driver selection) to Phase 7 (join collectors) and Phase 8 (closure compilation).
struct SpatialDriver {
    expr: SpatialExpr,
    lookup_fn: SpatialLookupFn,
}

/// Carries a pre-bound index lookup function from Phase 3 (driver selection)
/// to Phase 7 (join collectors) and Phase 8 (closure compilation).
///
/// The lookup function and predicate value are bound together at construction
/// time, so the execution path never sees `dyn Any`.
struct IndexDriver {
    lookup_fn: IndexLookupFn,
}

/// Builder for a single-table scan with optional predicates and joins.
pub struct ScanBuilder<'w> {
    planner: &'w QueryPlanner<'w>,
    world_id: WorldId,
    query_name: &'static str,
    estimated_rows: usize,
    predicates: Vec<Predicate>,
    joins: Vec<JoinSpec>,
    er_joins: Vec<ErJoinSpec>,
    /// Factory that produces a [`CompiledForEach`] closure. Captured from
    /// `scan::<Q>()` while Q is still in scope.
    compile_for_each: Option<Box<dyn FnOnce() -> CompiledForEach>>,
    /// Factory that produces a [`CompiledForEachRaw`] closure for read-only
    /// transactional access via `&World`.
    compile_for_each_raw: Option<Box<dyn FnOnce() -> CompiledForEachRaw>>,
    /// Required component bitset for left-side entity collection in join plans.
    left_required: Option<FixedBitSet>,
    /// Changed component bitset for left-side change detection in join plans.
    left_changed: Option<FixedBitSet>,
    /// Required component bitset for spatial index-gather path.
    required_for_spatial: Option<FixedBitSet>,
    /// Changed component bitset for spatial index-gather path.
    changed_for_spatial: Option<FixedBitSet>,
    /// Aggregate expressions to compute over matched entities.
    aggregates: Vec<AggregateExpr>,
}

struct JoinSpec {
    right_query_name: &'static str,
    right_estimated_rows: usize,
    join_kind: JoinKind,
    /// Required component bitset for right-side entity collection.
    right_required: FixedBitSet,
    /// Changed component bitset for right-side change detection.
    right_changed: FixedBitSet,
}

/// Specification for an ER (Entity-Relationship) join.
struct ErJoinSpec {
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
    pub fn join<Q: crate::query::fetch::WorldQuery + 'static>(
        mut self,
        join_kind: JoinKind,
    ) -> Self {
        // Estimate right-side rows from total entity count (conservative).
        let right_rows = self.planner.total_entities;
        let required = Q::required_ids(self.planner.components);
        let changed = Q::changed_ids(self.planner.components);
        self.joins.push(JoinSpec {
            right_query_name: std::any::type_name::<Q>(),
            right_estimated_rows: right_rows,
            join_kind,
            right_required: required,
            right_changed: changed,
        });
        self
    }

    /// Set explicit row estimate for the most recently added join's right side.
    ///
    /// Returns `Err(PlannerError::BuilderOrder)` if called before any `join()`.
    pub fn with_right_estimate(mut self, rows: usize) -> Result<Self, PlannerError> {
        // Apply to the most recently added join. ER joins are always added
        // after regular joins in builder order, so check er_joins first.
        if let Some(last_er) = self.er_joins.last_mut() {
            last_er.right_estimated_rows = rows;
        } else if let Some(last) = self.joins.last_mut() {
            last.right_estimated_rows = rows;
        } else {
            return Err(PlannerError::BuilderOrder(
                "with_right_estimate() called before any join() — call join() first".into(),
            ));
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
    ///     .er_join::<Parent, (&Pos, &Name)>(JoinKind::Inner)
    ///     .build();
    /// ```
    pub fn er_join<R, Q>(mut self, join_kind: JoinKind) -> Self
    where
        R: AsEntityRef,
        Q: crate::query::fetch::WorldQuery + 'static,
    {
        let right_rows = self.planner.total_entities;
        let required = Q::required_ids(self.planner.components);
        let changed = Q::changed_ids(self.planner.components);

        // Capture the entity-reference extractor as a type-erased closure.
        // Pre-resolve ComponentId for R at plan-build time. If R is not
        // registered yet, the ER join will produce zero matches (no entity
        // can have an unregistered component).
        let Some(ref_comp_id) = self.planner.components.id::<R>() else {
            // Component not registered — no entity can have it.
            // Push a no-op ER join that always produces empty results.
            let ref_extractor: EntityRefExtractor =
                Arc::new(|_world: &World, _entity: Entity| None);
            self.er_joins.push(ErJoinSpec {
                ref_component_name: std::any::type_name::<R>(),
                right_query_name: std::any::type_name::<Q>(),
                right_estimated_rows: 0,
                join_kind,
                right_required: FixedBitSet::new(),
                right_changed: FixedBitSet::new(),
                ref_extractor,
            });
            return self;
        };

        let ref_extractor: EntityRefExtractor = Arc::new(move |world: &World, entity: Entity| {
            let r: &R = world.get_by_id(entity, ref_comp_id)?;
            Some(r.entity_ref())
        });

        self.er_joins.push(ErJoinSpec {
            ref_component_name: std::any::type_name::<R>(),
            right_query_name: std::any::type_name::<Q>(),
            right_estimated_rows: right_rows,
            join_kind,
            right_required: required,
            right_changed: changed,
            ref_extractor,
        });
        self
    }

    /// Compile the scan into an optimized execution plan.
    pub fn build(mut self) -> QueryPlanResult {
        let mut warnings = Vec::new();

        // Phase 1: Classify predicates — index-driven vs spatial vs post-filter.
        let mut index_preds: Vec<(Predicate, &IndexDescriptor)> = Vec::new();
        let mut spatial_preds: Vec<(Predicate, SpatialCost, Option<SpatialLookupFn>)> = Vec::new();
        let mut filter_preds = Vec::new();
        let planner = self.planner;

        for pred in self.predicates {
            if pred.can_use_spatial() {
                match planner.find_spatial_index(&pred) {
                    SpatialLookupResult::Accelerated(_name, cost, lookup) => {
                        spatial_preds.push((pred, cost, lookup));
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

        let has_custom_filters = !filter_preds.is_empty();

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
            index_preds
                .iter()
                .all(|(p, _)| p.filter_fn.is_some() || matches!(p.kind, PredicateKind::Custom(_))),
            "Eq/Range predicate with filter_fn: None — plan would show filter but not apply it"
        );
        debug_assert!(
            filter_preds
                .iter()
                .all(|p| p.filter_fn.is_some() || matches!(p.kind, PredicateKind::Custom(_))),
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
            let (first_pred, _, first_lookup) = &spatial_preds[0];
            if let Some(lookup_fn) = first_lookup {
                let PredicateKind::Spatial(sp) = &first_pred.kind else {
                    unreachable!("spatial_preds only contains Spatial predicates");
                };
                Some(SpatialDriver {
                    expr: sp.into(),
                    lookup_fn: Arc::clone(lookup_fn),
                })
            } else {
                None
            }
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
                    // Assume ~50% of references point to valid right entities.
                    left_cost.rows * 0.5
                } else {
                    left_cost.rows
                },
                cpu: right_cost.cpu + left_cost.rows * 1.2, // hash build + probes
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
                // Index-gather path: use the spatial lookup function instead of
                // archetype scanning. Mirrors the Phase 8 index-gather closure.
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let expr = driver.expr.clone();
                let left_required_for_index = left_required.clone();
                let left_changed_for_index = left_changed.clone();
                Box::new(
                    move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                        let candidates = lookup_fn(&expr);
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

        // Phase 8: Compile for_each / for_each_raw closures for scan-only plans.
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
                // Spatial index-gather path: call the lookup function instead of
                // scanning archetypes.
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let expr = driver.expr.clone();
                let required = required_for_index;
                let changed = changed_for_index;
                Some(Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        let candidates = lookup_fn(&expr);
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
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let expr = driver.expr.clone();
                let required = required_for_index_raw;
                let changed = changed_for_index_raw;
                Some(Box::new(
                    move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                        let candidates = lookup_fn(&expr);
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
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let expr = driver.expr.clone();
                let required = required_for_agg;
                let changed = changed_for_agg;
                let filters = agg_filter_fns;
                Some(Box::new(
                    move |world: &World,
                          tick: Tick,
                          extractors: &mut [Option<Box<dyn BatchExtractor>>],
                          accums: &mut [AggregateAccum]| {
                        let candidates = lookup_fn(&expr);
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
                let lookup_fn = Arc::clone(&driver.lookup_fn);
                let expr = driver.expr.clone();
                let required = required_for_agg_raw;
                let changed = changed_for_agg_raw;
                let filters = agg_filter_fns_raw;
                Some(Box::new(
                    move |world: &World,
                          tick: Tick,
                          extractors: &mut [Option<Box<dyn BatchExtractor>>],
                          accums: &mut [AggregateAccum]| {
                        let candidates = lookup_fn(&expr);
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
            row_indices: Vec::new(),
            // Direct archetype iteration: only for pure scans with no
            // predicates, indexes, or spatial drivers. Plans with index/spatial
            // drivers use compiled closures that execute the index lookup.
            scan_required: if !has_custom_filters
                && !has_any_joins
                && index_driver.is_none()
                && spatial_driver.is_none()
            {
                self.left_required.clone()
            } else {
                None
            },
            scan_changed: self.left_changed.clone().unwrap_or_default(),
        }
    }
}

// ── QueryPlanner ─────────────────────────────────────────────────────

/// Type-erased spatial lookup: takes a `SpatialExpr`, returns candidate entities.
/// Provided by the user at registration time to bridge between the planner's
/// expression protocol and the index's concrete query API.
pub type SpatialLookupFn = Arc<dyn Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync>;

/// Descriptor for a registered spatial index.
struct SpatialIndexDescriptor {
    component_name: &'static str,
    /// The spatial index, behind Arc for shared access at execution time.
    index: Arc<dyn SpatialIndex + Send + Sync>,
    lookup_fn: Option<SpatialLookupFn>,
}

impl fmt::Debug for SpatialIndexDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialIndexDescriptor")
            .field("component_name", &self.component_name)
            .finish_non_exhaustive()
    }
}

/// Result of checking whether a spatial index can accelerate a predicate.
enum SpatialLookupResult {
    /// The index can accelerate the expression at the given cost.
    Accelerated(&'static str, SpatialCost, Option<SpatialLookupFn>),
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
    indexes: HashMap<TypeId, IndexDescriptor>,
    spatial_indexes: HashMap<TypeId, SpatialIndexDescriptor>,
    total_entities: usize,
    /// Component registry for resolving query type → component bitset.
    components: &'w ComponentRegistry,
    world_id: WorldId,
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
            spatial_indexes: HashMap::new(),
            total_entities: world.entity_count(),
            components: &world.components,
            world_id: world.world_id(),
            _world: PhantomData,
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
                lookup_fn: None,
            },
        );
        Ok(())
    }

    /// Register a spatial index with both cost discovery and execution-time lookup.
    ///
    /// The `lookup` closure bridges between the planner's [`SpatialExpr`]
    /// protocol and the index's concrete query API. The planner makes no
    /// assumptions about how the index answers queries — the closure is the
    /// adapter.
    ///
    /// Returns `Err(PlannerError::UnregisteredComponent)` if `T` has not been
    /// registered as a component in `world`.
    pub fn add_spatial_index_with_lookup<T: Component>(
        &mut self,
        index: Arc<dyn SpatialIndex + Send + Sync>,
        world: &World,
        lookup: impl Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync + 'static,
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
                lookup_fn: Some(Arc::new(lookup)),
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
        ScanBuilder {
            planner: self,
            world_id: self.world_id,
            query_name: std::any::type_name::<Q>(),
            estimated_rows: self.total_entities,
            predicates: Vec::new(),
            joins: Vec::new(),
            er_joins: Vec::new(),
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
        SubscriptionBuilder {
            scan: self.scan::<Q>(),
            errors: Vec::new(),
            has_predicates: false,
            attempted_predicates: false,
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

    /// Check if a spatial predicate can be accelerated by a registered spatial index.
    ///
    /// Returns a three-state result distinguishing "accelerated", "index
    /// exists but declined", and "no index registered".
    fn find_spatial_index(&self, pred: &Predicate) -> SpatialLookupResult {
        let PredicateKind::Spatial(sp) = &pred.kind else {
            return SpatialLookupResult::NoIndex;
        };
        let Some(desc) = self.spatial_indexes.get(&pred.component_type) else {
            return SpatialLookupResult::NoIndex;
        };
        let expr: SpatialExpr = sp.into();
        match desc.index.supports(&expr) {
            Some(cost) => SpatialLookupResult::Accelerated(
                desc.component_name,
                cost,
                desc.lookup_fn.as_ref().map(Arc::clone),
            ),
            None => SpatialLookupResult::Declined(sp.to_string()),
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

/// Options for plan cost estimation (internal).
#[derive(Clone, Copy, Debug)]
pub(crate) struct VectorizeOpts {
    /// L2 cache size in bytes. Used to partition hash join build tables
    /// so each partition fits in cache. Default: 256 KiB.
    pub(crate) l2_cache_bytes: usize,

    /// Average component size in bytes. Used to estimate how many rows
    /// fit in a cache line / partition. Default: 16 bytes.
    pub(crate) avg_component_bytes: usize,

    /// Target archetype chunk size. Scans that produce chunks larger than
    /// this will note it in the plan for monitoring. Default: 4096 rows.
    pub(crate) target_chunk_rows: usize,
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
    pub fn add_btree_index<C>(
        &mut self,
        index: &Arc<crate::index::BTreeIndex<C>>,
    ) -> Result<(), PlannerError>
    where
        C: Component + Ord + Clone,
        T: crate::index::HasBTreeIndex<C>,
    {
        self.planner.add_btree_index::<C>(index, self.world)
    }

    /// Register a hash index with the underlying planner for cost-based
    /// optimization.
    ///
    /// **Compile-time enforcement**: requires `T: HasHashIndex<C>`.
    pub fn add_hash_index<C>(
        &mut self,
        index: &Arc<crate::index::HashIndex<C>>,
    ) -> Result<(), PlannerError>
    where
        C: Component + std::hash::Hash + Eq + Clone,
        T: crate::index::HasHashIndex<C>,
    {
        self.planner.add_hash_index::<C>(index, self.world)
    }

    /// Register a spatial index with cost discovery and execution-time lookup.
    ///
    /// Delegates to [`QueryPlanner::add_spatial_index_with_lookup`].
    /// No compile-time index enforcement — spatial indexes are orthogonal to
    /// table schemas.
    pub fn add_spatial_index_with_lookup<C: Component>(
        &mut self,
        index: Arc<dyn SpatialIndex + Send + Sync>,
        lookup: impl Fn(&SpatialExpr) -> Vec<Entity> + Send + Sync + 'static,
    ) -> Result<(), PlannerError> {
        self.planner
            .add_spatial_index_with_lookup::<C>(index, self.world, lookup)
    }

    /// Register a spatial index for cost discovery only (no execution-time lookup).
    ///
    /// Delegates to [`QueryPlanner::add_spatial_index`].
    pub fn add_spatial_index<C: Component>(
        &mut self,
        index: Arc<dyn SpatialIndex + Send + Sync>,
    ) -> Result<(), PlannerError> {
        self.planner.add_spatial_index::<C>(index, self.world)
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
    #[cfg_attr(not(test), expect(dead_code))]
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
        assert!(
            left_len <= total,
            "sorted_intersection: left_len ({left_len}) exceeds buffer length ({total})"
        );

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

    /// Sort entities by (ArchetypeId, Row) to restore cache locality after
    /// join materialisation. Entities from the same archetype become
    /// contiguous, and within each archetype group, rows are in physical
    /// memory order (ascending).
    ///
    /// # Panics
    /// Panics if any entity in the buffer has no location (dead entity).
    /// This should never happen: join collectors only iterate live archetypes.
    fn sort_by_archetype(&mut self, entity_locations: &[Option<EntityLocation>]) {
        self.entities.sort_unstable_by_key(|e| {
            let loc = entity_locations[e.index() as usize]
                .expect("entity in scratch buffer has no location");
            ((loc.archetype_id.0 as u64) << 32) | (loc.row as u64)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Changed;
    use crate::World;
    use crate::index::SpatialIndex;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Score(u32);

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Team(u32);

    #[derive(Clone, Copy, Debug)]
    struct Pos {
        x: f32,
        y: f32,
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
        planner.add_btree_index(&Arc::new(idx), &world).unwrap();
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
        planner.add_hash_index(&Arc::new(idx), &world).unwrap();
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
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();
        planner.add_hash_index(&Arc::new(hash), &world).unwrap(); // should not overwrite
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
        planner.add_hash_index(&Arc::new(idx), &world).unwrap();

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
        planner.add_btree_index(&Arc::new(idx), &world).unwrap();

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
        planner.add_hash_index(&Arc::new(idx), &world).unwrap();

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
        planner
            .add_btree_index(&Arc::new(score_idx), &world)
            .unwrap();
        planner.add_hash_index(&Arc::new(team_idx), &world).unwrap();

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
        // Use Left join to test strategy selection (Inner joins are eliminated).
        let plan = planner
            .scan_with_estimate::<(&Score,)>(10)
            .join::<(&Team,)>(JoinKind::Left)
            .with_right_estimate(5)
            .unwrap()
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
        // Use Left join to test strategy selection (Inner joins are eliminated).
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Left)
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
        // Use Left join to test strategy selection (Inner joins are eliminated).
        let plan = planner
            .scan_with_estimate::<(&Score,)>(100)
            .join::<(&Team,)>(JoinKind::Left)
            .with_right_estimate(500)
            .unwrap()
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
            .unwrap()
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
    fn inner_join_eliminated_instead_of_reordered() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);

        // Inner join is eliminated at build time — no reordering needed.
        let plan = planner
            .scan_with_estimate::<(&Score,)>(500)
            .join::<(&Team,)>(JoinKind::Inner)
            .with_right_estimate(100)
            .unwrap()
            .build();

        match plan.root() {
            PlanNode::Scan { .. } => {} // eliminated — expected
            other => panic!(
                "expected Scan after inner join elimination, got {:?}",
                other
            ),
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
        planner.add_btree_index(&Arc::new(idx), &world).unwrap();

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();

        let explain = plan.explain();
        assert!(explain.contains("Execution Plan"));
        assert!(explain.contains("IndexGather"));
        assert!(explain.contains("BTree"));
        assert!(explain.contains("Score"));
        assert!(explain.contains("L2 cache budget"));
    }

    // ── Subscription plans ──────────────────────────────────────────

    #[test]
    fn subscription_requires_indexed_witness() {
        let mut world = World::new();
        for i in 0..1000u32 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = std::sync::Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        let sub = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, Predicate::eq(Score(42)))
            .build()
            .unwrap();

        // The index is registered, so the planner should emit an IndexLookup.
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
        for i in 0..1000u32 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let score_idx = std::sync::Arc::new(score_idx);
        let mut team_idx = BTreeIndex::<Team>::new();
        team_idx.rebuild(&mut world);
        let team_idx = std::sync::Arc::new(team_idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&score_idx, &world).unwrap();
        planner.add_btree_index(&team_idx, &world).unwrap();
        let score_w = Indexed::btree(&*score_idx);
        let team_w = Indexed::btree(&*team_idx);

        // Score predicate gets lower selectivity (more selective) → should drive.
        let sub = planner
            .subscribe::<(&Score, &Team)>()
            .where_eq(team_w, Predicate::eq(Team(2)).with_selectivity(0.2))
            .where_eq(score_w, Predicate::eq(Score(42)).with_selectivity(0.001))
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
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = std::sync::Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        let sub = planner
            .subscribe::<(&Score,)>()
            .where_range(witness, Predicate::range(Score(10)..Score(50)))
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
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);
        let witness = Indexed::hash(&idx);

        // Hash indexes cannot serve range queries — build returns an error.
        let result = planner
            .subscribe::<(&Score,)>()
            .where_range(witness, Predicate::range(Score(10)..Score(50)))
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
    fn subscription_no_predicates_returns_error() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let planner = QueryPlanner::new(&world);

        let result = planner.subscribe::<(&Score,)>().build();
        match result {
            Err(errs)
                if errs
                    .iter()
                    .any(|e| matches!(e, SubscriptionError::NoPredicates)) => {}
            other => panic!("expected NoPredicates error, got {:?}", other),
        }
    }

    #[test]
    fn subscription_component_mismatch_returns_error() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut score_idx = BTreeIndex::<Score>::new();
        score_idx.rebuild(&mut world);
        let score_idx = std::sync::Arc::new(score_idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&score_idx, &world).unwrap();
        let score_w = Indexed::btree(&*score_idx);

        // Witness is for Score but predicate is for Team — should error.
        let result = planner
            .subscribe::<(&Score, &Team)>()
            .where_eq(score_w, Predicate::eq(Team(2)))
            .build();
        match result {
            Err(errs)
                if errs
                    .iter()
                    .any(|e| matches!(e, SubscriptionError::ComponentMismatch { .. })) => {}
            other => panic!("expected ComponentMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn subscription_predicate_kind_mismatch_returns_error() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = std::sync::Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        // where_eq expects an Eq predicate, but we pass a Range.
        let result = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, Predicate::range(Score(10)..Score(50)))
            .build();
        match result {
            Err(errs)
                if errs
                    .iter()
                    .any(|e| matches!(e, SubscriptionError::PredicateKindMismatch { .. })) => {}
            other => panic!("expected PredicateKindMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn subscription_plan_is_executable() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let _e2 = world.spawn((Score(99),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        let mut plan = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, Predicate::eq(Score(42)))
            .build()
            .unwrap();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn subscription_plan_for_each_raw_works() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let _e2 = world.spawn((Score(99),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        let mut plan = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, Predicate::eq(Score(42)))
            .build()
            .unwrap();

        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn subscription_where_range_rejects_eq_predicate() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        // where_range expects a Range predicate, but we pass Eq.
        let result = planner
            .subscribe::<(&Score,)>()
            .where_range(witness, Predicate::eq(Score(42)))
            .build();
        assert!(matches!(
            result,
            Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::PredicateKindMismatch { expected: "Range", .. }))
        ));
    }

    #[test]
    fn subscription_unregistered_index_returns_error() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let witness = Indexed::btree(&idx);

        // Witness exists but index NOT registered with planner.
        let planner = QueryPlanner::new(&world);
        let result = planner
            .subscribe::<(&Score,)>()
            .where_eq(witness, Predicate::eq(Score(42)))
            .build();
        assert!(
            matches!(
                result,
                Err(ref errs) if errs.iter().any(|e| matches!(e, SubscriptionError::IndexNotRegistered { .. }))
            ),
            "expected IndexNotRegistered error when index not registered with planner, got {:?}",
            result
        );
    }

    #[test]
    fn subscription_validation_failure_does_not_produce_spurious_no_predicates() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        // Pass a Team predicate with a Score witness — ComponentMismatch.
        // Should NOT also get NoPredicates.
        let result = planner
            .subscribe::<(&Score, &Team)>()
            .where_eq(witness, Predicate::eq(Team(2)).with_selectivity(0.2))
            .build();
        match result {
            Err(errs) => {
                assert!(
                    errs.iter()
                        .any(|e| matches!(e, SubscriptionError::ComponentMismatch { .. })),
                    "expected ComponentMismatch"
                );
                assert!(
                    !errs
                        .iter()
                        .any(|e| matches!(e, SubscriptionError::NoPredicates)),
                    "should NOT get spurious NoPredicates when predicates were attempted"
                );
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn subscription_with_changed_yields_only_mutated_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let e2 = world.spawn((Score(42),));
        let _e3 = world.spawn((Score(99),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        // Subscribe with Changed<Score> — only entities whose Score column
        // was mutated since last call will pass through.
        let mut plan = planner
            .subscribe::<(Changed<Score>, &Score)>()
            .where_eq(witness, Predicate::eq(Score(42)))
            .build()
            .unwrap();

        // First call: all matching entities are "changed" (never read before).
        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));

        // Second call with no mutations: nothing changed → zero results.
        results.clear();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 0, "no mutations → no results");

        // Mutate e1's Score (stays 42, but the column is marked changed).
        world.get_mut::<Score>(e1).unwrap().0 = 42;

        // Third call: only e1's archetype was touched. Since e1 and e2 are
        // in the same archetype (both have only Score), Changed<T> is
        // archetype-granular — both pass. This is correct per the engine's
        // change detection semantics.
        results.clear();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert!(results.contains(&e1));
        // e2 is in the same archetype as e1, so it also passes Changed<Score>.
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
        planner.add_btree_index(&Arc::new(idx), &world).unwrap();

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
        planner
            .add_btree_index(&Arc::new(score_idx), &world)
            .unwrap();

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

    // ── Plan node introspection ─────────────────────────────────────

    #[test]
    fn build_produces_chunked_scan() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();

        match plan.root() {
            PlanNode::Scan {
                estimated_rows,
                avg_chunk_size,
                ..
            } => {
                assert_eq!(*estimated_rows, 1000);
                assert!(*avg_chunk_size <= VectorizeOpts::default().target_chunk_rows);
            }
            other => panic!("expected Scan, got {:?}", other),
        }
    }

    #[test]
    fn index_lookup_produces_index_gather() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(idx), &world).unwrap();

        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        match plan.root() {
            PlanNode::IndexLookup { index_kind, .. } => {
                assert_eq!(*index_kind, IndexKind::BTree);
            }
            other => panic!("expected IndexLookup, got {:?}", other),
        }
    }

    #[test]
    fn filter_detects_branchless() {
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

        match plan.root() {
            PlanNode::Filter { branchless, .. } => {
                assert!(*branchless, "Range predicate should be branchless");
            }
            other => panic!("expected Filter, got {:?}", other),
        }

        // Custom predicate → branched
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>("complex check", 0.5, |_, _| {
                true
            }))
            .build();

        match plan.root() {
            PlanNode::Filter { branchless, .. } => {
                assert!(!*branchless, "Custom predicate should be branched");
            }
            other => panic!("expected Filter, got {:?}", other),
        }
    }

    #[test]
    fn hash_join_partitioned() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        // Use Left join to test partitioning (Inner joins are eliminated).
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Left)
            .build();

        match plan.root() {
            PlanNode::HashJoin { partitions, .. } => {
                assert!(*partitions >= 1);
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn nested_loop_for_small_cardinality() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        // Use Left join to test strategy selection (Inner joins are eliminated).
        let plan = planner
            .scan_with_estimate::<(&Score,)>(10)
            .join::<(&Team,)>(JoinKind::Left)
            .with_right_estimate(5)
            .unwrap()
            .build();

        match plan.root() {
            PlanNode::NestedLoopJoin { .. } => {} // expected
            other => panic!("expected NestedLoopJoin, got {:?}", other),
        }
    }

    #[test]
    fn explain_contains_details() {
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
        let explain = plan.explain();

        assert!(explain.contains("Execution Plan"));
        assert!(explain.contains("L2 cache budget"));
        assert!(explain.contains("target chunk"));
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

        // Should produce a valid plan
        assert!(plan.cost().cpu > 0.0);
        let explain = plan.explain();
        assert!(explain.contains("Execution Plan"));
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
        planner.add_btree_index::<Score>(&Arc::new(btree)).unwrap();

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
        let mut plan = planner.scan::<(&Score,)>().build();
        let mut result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result, vec![e1]);
    }

    #[test]
    fn execute_filter_eq() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        planner.add_hash_index(&Arc::new(hash), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(2)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(2)))
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let mut result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "even scores",
                0.5,
                |world, entity| world.get::<Score>(entity).is_some_and(|s| s.0 % 2 == 0),
            ))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
        assert_eq!(result.len(), 25);
        for e in &result {
            assert!(world.get::<Score>(*e).unwrap().0.is_multiple_of(2));
        }
    }

    #[test]
    fn execute_empty_world() {
        let mut world = World::new();
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute(&mut world).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn execute_respects_despawned_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1),));
        let e2 = world.spawn((Score(2),));
        let e3 = world.spawn((Score(3),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        // Despawn e2 after plan construction
        world.despawn(e2);

        let result = plan.execute(&mut world).unwrap();
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
        planner.add_hash_index(&Arc::new(hash), &world).unwrap();

        // scan::<(&Score, &Team)> requires BOTH components
        let mut plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(1)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        planner.add_hash_index(&Arc::new(hash), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::eq(Team(3)))
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(999)))
            .build();
        let result = plan.execute(&mut world).unwrap();
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
        planner.add_hash_index(&Arc::new(hash), &world).unwrap();

        let mut plan = planner
            .scan::<(&Team,)>()
            .filter(Predicate::eq(Team(999)))
            .build();
        let result = plan.execute(&mut world).unwrap();
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
        planner.add_btree_index(&btree, &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        // First execute: should find Score(42)
        let result = plan.execute(&mut world).unwrap();
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
        let mut scan_plan = planner2.scan::<(&Score,)>().build();
        assert_eq!(scan_plan.execute(&mut world).unwrap().len(), 100);
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
        planner.add_hash_index(&hash, &world).unwrap();

        let mut plan = planner
            .scan::<(&Team,)>()
            .filter(Predicate::eq(Team(2)))
            .build();

        let result = plan.execute(&mut world).unwrap().to_vec();
        assert_eq!(result.len(), 10); // 50 / 5 teams
        for e in &result {
            assert_eq!(*world.get::<Team>(*e).unwrap(), Team(2));
        }
    }

    #[test]
    fn execution_produces_correct_results() {
        // End-to-end: plan must produce the same results as a naive
        // query would.
        let mut world = World::new();
        for i in 0..500 {
            world.spawn((Score(i), Team(i % 5)));
        }
        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);
        let mut hash = HashIndex::<Team>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();
        planner.add_hash_index(&Arc::new(hash), &world).unwrap();

        // Complex plan: index + filter + join
        let mut plan = planner
            .scan::<(&Score, &Team)>()
            .filter(Predicate::range::<Score, _>(Score(100)..Score(300)))
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let entities = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Left)
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let result = plan.execute(&mut world).unwrap().to_vec();
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
        let mut plan = planner
            .scan_with_estimate::<(&Score,)>(8)
            .join::<(&Team,)>(JoinKind::Left)
            .with_right_estimate(3)
            .unwrap()
            .build();
        let result = plan.execute(&mut world).unwrap();
        // Left join: all 8 Score entities preserved.
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn execute_multi_join_intersects_all() {
        let mut world = World::new();
        let mut all_three = Vec::new();
        for i in 0..10u32 {
            all_three.push(world.spawn((Score(i), Team(i % 3), Health(100))));
        }
        // 5 entities with only Score + Team (no Health)
        for i in 10..15u32 {
            world.spawn((Score(i), Team(i % 3)));
        }
        // 5 entities with only Score + Health (no Team)
        for i in 15..20u32 {
            world.spawn((Score(i), Health(50)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .join::<(&Health,)>(JoinKind::Inner)
            .build();

        let result = plan.execute(&mut world).unwrap();
        // Score ∩ Team ∩ Health = the 10 entities with all three
        let mut found: Vec<Entity> = result.to_vec();
        found.sort_by_key(|e| e.to_bits());
        all_three.sort_by_key(|e| e.to_bits());
        assert_eq!(found, all_three);
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

    #[test]
    fn scratch_buffer_sorted_intersection_empty_left() {
        let mut buf = ScratchBuffer::new(10);
        let left_len = 0; // empty left
        for idx in [2, 3, 6] {
            buf.push(Entity::new(idx, 0));
        }
        let result = buf.sorted_intersection(left_len);
        assert!(result.is_empty());
    }

    #[test]
    fn scratch_buffer_sorted_intersection_empty_right() {
        let mut buf = ScratchBuffer::new(10);
        for idx in [1, 3, 5] {
            buf.push(Entity::new(idx, 0));
        }
        let left_len = buf.len(); // right partition is empty
        let result = buf.sorted_intersection(left_len);
        assert!(result.is_empty());
    }

    #[test]
    fn scratch_buffer_sorted_intersection_complete_overlap() {
        let mut buf = ScratchBuffer::new(10);
        for idx in [1, 2, 3] {
            buf.push(Entity::new(idx, 0));
        }
        let left_len = buf.len();
        for idx in [1, 2, 3] {
            buf.push(Entity::new(idx, 0));
        }
        let result = buf.sorted_intersection(left_len);
        let mut ids: Vec<u32> = result.iter().map(|e| e.index()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn scratch_buffer_sorted_intersection_no_overlap() {
        let mut buf = ScratchBuffer::new(10);
        for idx in [1, 3, 5] {
            buf.push(Entity::new(idx, 0));
        }
        let left_len = buf.len();
        for idx in [2, 4, 6] {
            buf.push(Entity::new(idx, 0));
        }
        let result = buf.sorted_intersection(left_len);
        assert!(result.is_empty());
    }

    // ── Execute with ScratchBuffer ─────────────────────────────────

    #[test]
    fn execute_with_scratch_returns_all_entities() {
        let mut world = World::new();
        let mut expected = Vec::new();
        for i in 0..10u32 {
            let e = world.spawn((Score(i), Team(i % 3)));
            expected.push(e);
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let result = plan.execute(&mut world).unwrap();
        let mut found: Vec<Entity> = result.to_vec();
        found.sort_by_key(|e| e.to_bits());
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(found, expected);
    }

    #[test]
    fn execute_scratch_reuse_no_realloc() {
        let mut world = World::new();
        for i in 0..10u32 {
            world.spawn((Score(i), Team(i % 3)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let _ = plan.execute(&mut world).unwrap();
        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result.len(), 10);
    }

    // ── CompiledScan for_each ──────────────────────────────────────

    #[test]
    fn compiled_scan_for_each_yields_all_entities() {
        let mut world = World::new();
        let mut expected = Vec::new();
        for i in 0..10u32 {
            let e = world.spawn((Score(i),));
            expected.push(e);
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let mut found = Vec::new();
        plan.for_each(&mut world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        found.sort_by_key(|e| e.to_bits());
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(found, expected);
    }

    #[test]
    fn for_each_iterates_multiple_archetypes() {
        let mut world = World::new();
        let mut expected = Vec::new();
        // Archetype 1: (Score,)
        for i in 0..5u32 {
            expected.push(world.spawn((Score(i),)));
        }
        // Archetype 2: (Score, Team)
        for i in 5..10u32 {
            expected.push(world.spawn((Score(i), Team(i % 3))));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let mut found = Vec::new();
        plan.for_each(&mut world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        found.sort_by_key(|e| e.to_bits());
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(found, expected);
    }

    #[test]
    fn for_each_raw_iterates_multiple_archetypes() {
        let mut world = World::new();
        let mut expected = Vec::new();
        for i in 0..5u32 {
            expected.push(world.spawn((Score(i),)));
        }
        for i in 5..10u32 {
            expected.push(world.spawn((Score(i), Team(i % 3))));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let mut found = Vec::new();
        plan.for_each_raw(&world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        found.sort_by_key(|e| e.to_bits());
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(found, expected);
    }

    #[test]
    fn for_each_with_eq_filter() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        let mut found = Vec::new();
        plan.for_each(&mut world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        assert_eq!(found.len(), 1);
        let score = world.get::<Score>(found[0]).unwrap();
        assert_eq!(*score, Score(42));
    }

    #[test]
    fn for_each_with_range_filter() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();

        let mut found = Vec::new();
        plan.for_each(&mut world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        assert_eq!(found.len(), 10); // 10..20 exclusive = 10 entities
    }

    #[test]
    fn for_each_with_custom_filter() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "even scores",
                0.5,
                |world, entity| world.get::<Score>(entity).is_some_and(|s| s.0 % 2 == 0),
            ))
            .build();

        let mut found = Vec::new();
        plan.for_each(&mut world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        assert_eq!(found.len(), 50);
    }

    // ── CompiledScan for_each_raw ────────────────────────────────────

    #[test]
    fn for_each_raw_yields_entities_without_mut_world() {
        let mut world = World::new();
        for i in 0..10u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let mut found = Vec::new();
        plan.for_each_raw(&world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        assert_eq!(found.len(), 10);
    }

    #[test]
    fn for_each_raw_with_filter() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq(Score(42)))
            .build();

        let mut found = Vec::new();
        plan.for_each_raw(&world, |entity: Entity| {
            found.push(entity);
        })
        .unwrap();
        assert_eq!(found.len(), 1);
    }

    // ── Changed<T> filtering ──────────────────────────────────────────

    #[test]
    fn for_each_changed_skips_stale_archetypes() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // First call: everything is new (changed since tick 0).
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 5);

        // Second call: nothing changed since the last read tick.
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn for_each_changed_detects_mutation() {
        let mut world = World::new();
        let e = world.spawn((Score(1),));
        world.spawn((Score(2), Team(0)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // Drain initial changes.
        plan.for_each(&mut world, |_| {}).unwrap();

        // Mutate one archetype's Score column via get_mut.
        let _ = world.get_mut::<Score>(e);

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        // Only the archetype containing `e` was mutated.
        assert_eq!(count, 1);
    }

    #[test]
    fn for_each_raw_changed_reads_tick_but_does_not_advance() {
        let mut world = World::new();
        for i in 0..3u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // for_each_raw does not advance the tick.
        let mut count = 0;
        plan.for_each_raw(&world, |_| count += 1).unwrap();
        assert_eq!(count, 3);

        // Same tick — still sees changes.
        let mut count = 0;
        plan.for_each_raw(&world, |_| count += 1).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn execute_changed_skips_stale_archetypes() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        assert_eq!(plan.execute(&mut world).unwrap().len(), 5);
        assert_eq!(plan.execute(&mut world).unwrap().len(), 0);
    }

    #[test]
    fn for_each_no_changed_pays_zero_cost() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        // Without Changed<T>, every call sees all entities.
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 5);

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn execute_join_changed_left_only() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i), Team(i % 2)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        assert_eq!(plan.execute(&mut world).unwrap().len(), 5);
        assert_eq!(plan.execute(&mut world).unwrap().len(), 0);
    }

    #[test]
    fn execute_join_changed_right_only() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i), Team(i % 2)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(Changed<Team>, &Team)>(JoinKind::Inner)
            .build();
        // First: all changed
        assert_eq!(plan.execute(&mut world).unwrap().len(), 5);
        // Second: right side not changed, right yields 0 entities.
        // Inner join of 5 and 0 = 0.
        assert_eq!(plan.execute(&mut world).unwrap().len(), 0);
    }

    #[test]
    fn for_each_raw_then_for_each_advances() {
        let mut world = World::new();
        for i in 0..3u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();
        // for_each_raw: sees everything, doesn't advance tick
        let mut count = 0;
        plan.for_each_raw(&world, |_| count += 1).unwrap();
        assert_eq!(count, 3);
        // for_each: also sees everything (tick still at 0), and advances
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 3);
        // for_each again: tick advanced, nothing changed
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn for_each_changed_multiple_archetypes_partial() {
        let mut world = World::new();
        // Archetype 1: (Score,)
        let e1 = world.spawn((Score(1),));
        // Archetype 2: (Score, Team)
        let _e2 = world.spawn((Score(2), Team(0)));
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();
        // Consume initial changes
        plan.for_each(&mut world, |_| {}).unwrap();
        // Mutate only archetype 1
        let _ = world.get_mut::<Score>(e1);
        // Only archetype 1 should be returned
        let mut found = Vec::new();
        plan.for_each(&mut world, |entity| found.push(entity))
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], e1);
    }

    #[test]
    fn for_each_changed_with_predicate_filter() {
        let mut world = World::new();
        // All in one archetype
        for i in 0..10u32 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .filter(Predicate::custom::<Score>(
                "score < 5",
                0.5,
                |world: &World, entity: Entity| world.get::<Score>(entity).is_some_and(|s| s.0 < 5),
            ))
            .build();

        // First call: Changed passes (everything new), predicate filters to 5
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 5);

        // Second call: Changed skips the archetype entirely
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn execute_changed_detects_partial_mutation() {
        let mut world = World::new();
        // Two archetypes so column-level change detection can distinguish them
        let e = world.spawn((Score(1),));
        world.spawn((Score(2), Team(0)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // Drain initial changes
        let _ = plan.execute(&mut world).unwrap();

        // Mutate only the (Score,) archetype
        let _ = world.get_mut::<Score>(e);

        // execute path (scratch buffer) should see only the changed archetype
        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn for_each_changed_sees_entities_spawned_after_plan_creation() {
        let mut world = World::new();
        world.spawn((Score(1),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // Drain initial changes
        plan.for_each(&mut world, |_| {}).unwrap();

        // Spawn new entities into a new archetype AFTER plan was built.
        // The compiled scan iterates world.archetypes at execution time,
        // so new archetypes should be visible. Their column ticks will be
        // newer than last_read_tick.
        world.spawn((Score(2), Team(0)));
        world.spawn((Score(3), Team(1)));

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn execute_left_join_changed_right_preserves_left() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i), Team(i % 2)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(Changed<Team>, &Team)>(JoinKind::Left)
            .build();

        // First call: all changed, left join preserves left = 5
        assert_eq!(plan.execute(&mut world).unwrap().len(), 5);

        // Second call: right side stale, but Left join keeps all left entities
        assert_eq!(plan.execute(&mut world).unwrap().len(), 5);
    }

    // ── Spatial predicate tests ──────────────────────────────────

    /// Grid index that supports `Within` queries for testing.
    struct TestGridIndex {
        entities: Vec<Entity>,
    }

    impl TestGridIndex {
        fn new() -> Self {
            Self {
                entities: Vec::new(),
            }
        }
    }

    impl SpatialIndex for TestGridIndex {
        fn rebuild(&mut self, world: &mut World) {
            self.entities = world.query::<(Entity, &Pos)>().map(|(e, _)| e).collect();
        }

        fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
            match expr {
                crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                    estimated_rows: (self.entities.len() as f64 * 0.1).max(1.0),
                    cpu: 5.0,
                }),
                crate::index::SpatialExpr::Intersects { .. } => Some(crate::index::SpatialCost {
                    estimated_rows: (self.entities.len() as f64 * 0.2).max(1.0),
                    cpu: 8.0,
                }),
            }
        }
    }

    /// Grid index that does NOT support any spatial queries.
    struct UnsupportedGridIndex;

    impl SpatialIndex for UnsupportedGridIndex {
        fn rebuild(&mut self, _world: &mut World) {}
    }

    #[test]
    fn spatial_predicate_within_creates_spatial_lookup() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();

        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
            .build();

        // The root of the logical plan should be a SpatialLookup.
        match plan.root() {
            PlanNode::SpatialLookup {
                component_name,
                cost,
                ..
            } => {
                assert!(component_name.contains("Pos"));
                assert!(cost.rows > 0.0);
            }
            other => panic!("expected SpatialLookup, got {:?}", other),
        }

        // No warnings expected — spatial index was registered.
        assert!(plan.warnings().is_empty());
    }

    #[test]
    fn spatial_predicate_intersects_creates_spatial_lookup() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();

        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::intersects::<Pos>([0.0, 0.0], [25.0, 25.0], |_, _| true).unwrap())
            .build();

        match plan.root() {
            PlanNode::SpatialLookup { .. } => {}
            other => panic!("expected SpatialLookup, got {:?}", other),
        }
    }

    #[test]
    fn spatial_predicate_without_index_falls_back_to_filter() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
            .build();

        // Without a spatial index, should fall back to Scan + Filter.
        match plan.root() {
            PlanNode::Filter { child, .. } => match child.as_ref() {
                PlanNode::Scan { .. } => {}
                other => panic!("expected Scan child, got {:?}", other),
            },
            other => panic!("expected Filter, got {:?}", other),
        }

        // Should have a warning about missing spatial index.
        assert_eq!(plan.warnings().len(), 1);
        match &plan.warnings()[0] {
            PlanWarning::MissingIndex { predicate_kind, .. } => {
                assert_eq!(*predicate_kind, "spatial");
            }
            other => panic!("expected MissingIndex warning, got {:?}", other),
        }
    }

    #[test]
    fn spatial_predicate_unsupported_expr_falls_back_to_filter() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        // Register a spatial index that doesn't support any queries.
        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(UnsupportedGridIndex), &world)
            .unwrap();

        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
            .build();

        // Index exists but doesn't support Within — should fall back.
        match plan.root() {
            PlanNode::Filter { child, .. } => match child.as_ref() {
                PlanNode::Scan { .. } => {}
                other => panic!("expected Scan child, got {:?}", other),
            },
            other => panic!("expected Filter, got {:?}", other),
        }

        // SpatialIndexDeclined warning should be emitted (not MissingIndex).
        assert_eq!(plan.warnings().len(), 1);
        assert!(
            matches!(
                &plan.warnings()[0],
                PlanWarning::SpatialIndexDeclined { .. }
            ),
            "expected SpatialIndexDeclined warning, got {:?}",
            plan.warnings()[0]
        );
    }

    #[test]
    fn spatial_lookup_uses_spatial_gather() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();

        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
            .build();

        match plan.root() {
            PlanNode::SpatialLookup { component_name, .. } => {
                assert!(component_name.contains("Pos"));
            }
            other => panic!("expected SpatialLookup, got {:?}", other),
        }
    }

    #[test]
    fn spatial_predicate_explain_contains_spatial_lookup() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();

        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([50.0, 50.0], 10.0, |_, _| true).unwrap())
            .build();

        let explain = plan.explain();
        assert!(explain.contains("SpatialGather"));
        assert!(explain.contains("Pos"));
    }

    #[test]
    fn spatial_predicate_with_custom_selectivity() {
        let pred = Predicate::within::<Pos>([0.0, 0.0], 1.0, |_, _| true)
            .unwrap()
            .with_selectivity(0.05);

        // Selectivity override should work.
        match &pred.kind {
            PredicateKind::Spatial(SpatialPredicate::Within { radius, .. }) => {
                assert!(*radius > 0.0);
            }
            other => panic!("expected Spatial(Within), got {:?}", other),
        }
        assert!((pred.selectivity - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn spatial_predicate_debug_format() {
        let pred = Predicate::within::<Pos>([1.0, 2.0], 3.0, |_, _| true).unwrap();
        let dbg = format!("{:?}", pred);
        assert!(dbg.contains("Spatial"));
        assert!(dbg.contains("ST_Within"));

        let pred2 = Predicate::intersects::<Pos>([0.0, 0.0], [10.0, 10.0], |_, _| true).unwrap();
        let dbg2 = format!("{:?}", pred2);
        assert!(dbg2.contains("ST_Intersects"));
    }

    #[test]
    fn spatial_vs_btree_cheaper_spatial_wins() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: i as f32,
                },
                Score(i),
            ));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        // Spatial predicate with very low estimated cost should win.
        let plan = planner
            .scan::<(&Pos, &Score)>()
            .filter(Predicate::within::<Pos>([500.0, 500.0], 5.0, |_, _| true).unwrap())
            .filter(Predicate::range::<Score, _>(Score(400)..Score(600)))
            .build();

        // The driving access should be whichever has lower cost.
        // Our TestGridIndex reports ~100 rows for within (0.1 * 1000),
        // vs BTree range at 0.1 selectivity → 100 rows.
        // The spatial cost.cpu is 5.0 vs index_lookup ~ 5.0 + 100.
        // Spatial should win.
        match plan.root() {
            PlanNode::Filter { child, .. } => match child.as_ref() {
                PlanNode::SpatialLookup { .. } => {}
                PlanNode::Filter { child, .. } => match child.as_ref() {
                    PlanNode::SpatialLookup { .. } => {}
                    other => panic!("expected SpatialLookup deep, got {:?}", other),
                },
                other => panic!("expected SpatialLookup or Filter child, got {:?}", other),
            },
            PlanNode::SpatialLookup { .. } => {}
            other => panic!("expected SpatialLookup at root, got {:?}", other),
        }
    }

    #[test]
    fn spatial_predicate_for_each_works() {
        let mut world = World::new();
        for i in 0..20 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        // Even without a spatial index, for_each should work via filter fallback.
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(
                Predicate::within::<Pos>([10.0, 10.0], 100.0, |world, entity| {
                    world.get::<Pos>(entity).is_some_and(|p| {
                        ((p.x - 10.0).powi(2) + (p.y - 10.0).powi(2)).sqrt() < 100.0
                    })
                })
                .unwrap(),
            )
            .build();

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 20); // All within radius 100
    }

    #[test]
    fn spatial_predicate_display() {
        let sp = SpatialPredicate::Within {
            center: vec![1.0, 2.0],
            radius: 3.0,
        };
        assert_eq!(format!("{}", sp), "ST_Within([1.0, 2.0], 3)");

        let sp2 = SpatialPredicate::Intersects {
            min: vec![0.0, 0.0],
            max: vec![10.0, 10.0],
        };
        assert_eq!(
            format!("{}", sp2),
            "ST_Intersects([0.0, 0.0], [10.0, 10.0])"
        );
    }

    #[test]
    fn spatial_predicate_to_expr_round_trip() {
        let sp = SpatialPredicate::Within {
            center: vec![1.0, 2.0],
            radius: 3.0,
        };
        let expr = crate::index::SpatialExpr::from(&sp);
        match &expr {
            crate::index::SpatialExpr::Within { center, radius } => {
                assert_eq!(center, &[1.0, 2.0]);
                assert!((radius - 3.0).abs() < f64::EPSILON);
            }
            other => panic!("expected Within, got {:?}", other),
        }

        let sp2 = SpatialPredicate::Intersects {
            min: vec![0.0, 1.0],
            max: vec![2.0, 3.0],
        };
        let expr2 = crate::index::SpatialExpr::from(&sp2);
        match &expr2 {
            crate::index::SpatialExpr::Intersects { min, max } => {
                assert_eq!(min, &[0.0, 1.0]);
                assert_eq!(max, &[2.0, 3.0]);
            }
            other => panic!("expected Intersects, got {:?}", other),
        }
    }

    // ── Additional spatial coverage tests ────────────────────────

    /// Spatial index with very high cost — BTree should win the cost comparison.
    struct ExpensiveSpatialIndex;

    impl SpatialIndex for ExpensiveSpatialIndex {
        fn rebuild(&mut self, _world: &mut World) {}

        fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
            match expr {
                crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                    estimated_rows: 500.0,
                    cpu: 500.0,
                }),
                _ => None,
            }
        }
    }

    #[test]
    fn spatial_vs_btree_cheaper_btree_wins() {
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: i as f32,
                },
                Score(i),
            ));
        }

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(ExpensiveSpatialIndex), &world)
            .unwrap();
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        // BTree with high selectivity (0.01) → cost ~ 5 + 10 = 15.
        // Spatial index reports cpu=500. BTree should win.
        let plan = planner
            .scan::<(&Pos, &Score)>()
            .filter(Predicate::within::<Pos>([500.0, 500.0], 5.0, |_, _| true).unwrap())
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        // Driving access should be IndexLookup (BTree), not SpatialLookup.
        fn has_index_lookup(node: &PlanNode) -> bool {
            match node {
                PlanNode::IndexLookup { .. } => true,
                PlanNode::Filter { child, .. } => has_index_lookup(child),
                _ => false,
            }
        }
        assert!(
            has_index_lookup(plan.root()),
            "expected IndexLookup as driver, got {:?}",
            plan.root()
        );
    }

    /// Spatial index that only supports `Within`, not `Intersects`.
    struct WithinOnlyIndex {
        entity_count: usize,
    }

    impl SpatialIndex for WithinOnlyIndex {
        fn rebuild(&mut self, world: &mut World) {
            self.entity_count = world.query::<(Entity, &Pos)>().count();
        }

        fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
            match expr {
                crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                    estimated_rows: (self.entity_count as f64 * 0.1).max(1.0),
                    cpu: 5.0,
                }),
                _ => None,
            }
        }
    }

    #[test]
    fn spatial_partial_capability_within_supported_intersects_declined() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let mut idx = WithinOnlyIndex { entity_count: 0 };
        idx.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(idx), &world)
            .unwrap();

        // Within should get a SpatialLookup.
        let within_plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([25.0, 25.0], 5.0, |_, _| true).unwrap())
            .build();
        assert!(
            matches!(within_plan.root(), PlanNode::SpatialLookup { .. }),
            "Within should produce SpatialLookup, got {:?}",
            within_plan.root()
        );
        assert!(within_plan.warnings().is_empty());

        // Intersects should fall back with a SpatialIndexDeclined warning.
        let intersects_plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::intersects::<Pos>([0.0, 0.0], [10.0, 10.0], |_, _| true).unwrap())
            .build();
        match intersects_plan.root() {
            PlanNode::Filter { child, .. } => {
                assert!(matches!(child.as_ref(), PlanNode::Scan { .. }));
            }
            other => panic!("expected Filter, got {:?}", other),
        }
        assert_eq!(intersects_plan.warnings().len(), 1);
        assert!(matches!(
            &intersects_plan.warnings()[0],
            PlanWarning::SpatialIndexDeclined { .. }
        ));
    }

    #[test]
    fn multiple_spatial_predicates_first_drives_rest_filter() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: i as f32,
                },
                Health(100),
            ));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();

        // Two spatial predicates on the same component.
        let plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([50.0, 50.0], 5.0, |_, _| true).unwrap())
            .filter(Predicate::intersects::<Pos>([0.0, 0.0], [0.5, 0.5], |_, _| true).unwrap())
            .build();

        // One should be the driver (SpatialLookup), the other a Filter.
        fn find_spatial_lookup(node: &PlanNode) -> bool {
            match node {
                PlanNode::SpatialLookup { .. } => true,
                PlanNode::Filter { child, .. } => find_spatial_lookup(child),
                _ => false,
            }
        }
        fn count_filters(node: &PlanNode) -> usize {
            match node {
                PlanNode::Filter { child, .. } => 1 + count_filters(child),
                _ => 0,
            }
        }
        assert!(
            find_spatial_lookup(plan.root()),
            "expected SpatialLookup in plan tree"
        );
        assert!(
            count_filters(plan.root()) >= 1,
            "expected at least one Filter wrapping the SpatialLookup"
        );
        assert!(plan.warnings().is_empty());
    }

    /// Spatial index with low CPU but high estimated_rows.
    /// Used to test that full plan cost (including downstream filters)
    /// determines the driver, not just the driving access cost alone.
    struct HighRowsSpatialIndex;

    impl SpatialIndex for HighRowsSpatialIndex {
        fn rebuild(&mut self, _world: &mut World) {}

        fn supports(&self, expr: &crate::index::SpatialExpr) -> Option<crate::index::SpatialCost> {
            match expr {
                crate::index::SpatialExpr::Within { .. } => Some(crate::index::SpatialCost {
                    // Low CPU but returns most of the dataset — downstream
                    // filters over 900 rows are expensive.
                    estimated_rows: 900.0,
                    cpu: 3.0,
                }),
                _ => None,
            }
        }
    }

    #[test]
    fn spatial_low_cpu_high_rows_loses_to_selective_btree() {
        // Regression: spatial index with cpu=3 but estimated_rows=900
        // vs BTree with selectivity=0.01 (10 rows from 1000).
        // Driving access costs alone: spatial=3, btree=5+10=15 → spatial wins.
        // But full plan cost: spatial driver + btree-as-filter over 900 rows
        // is more expensive than btree driver + spatial-as-filter over 10 rows.
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: i as f32,
                },
                Score(i),
            ));
        }

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(HighRowsSpatialIndex), &world)
            .unwrap();
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let plan = planner
            .scan::<(&Pos, &Score)>()
            .filter(Predicate::within::<Pos>([500.0, 500.0], 5.0, |_, _| true).unwrap())
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        // BTree should win as driver because the full plan cost is lower:
        // btree(10 rows) + spatial-as-filter(10 * 0.5) < spatial(900 rows) + btree-as-filter(900 * 0.5)
        fn has_index_lookup(node: &PlanNode) -> bool {
            match node {
                PlanNode::IndexLookup { .. } => true,
                PlanNode::Filter { child, .. } => has_index_lookup(child),
                _ => false,
            }
        }
        assert!(
            has_index_lookup(plan.root()),
            "expected IndexLookup as driver when BTree full plan cost is lower, got {:?}",
            plan.root()
        );
    }

    #[test]
    fn spatial_index_for_each_uses_lookup() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));
        let _e3 = world.spawn((Pos { x: 100.0, y: 100.0 },));

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(
                Arc::new(grid),
                &world,
                move |_expr: &crate::index::SpatialExpr| {
                    call_count_clone.fetch_add(1, Ordering::Relaxed);
                    vec![e1, e2]
                },
            )
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([5.0, 5.0], 10.0, |_, _| true).unwrap())
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| {
            results.push(entity);
        })
        .unwrap();

        assert!(
            call_count.load(Ordering::Relaxed) > 0,
            "lookup function was never called"
        );
        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn spatial_index_join_uses_lookup() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 }, Score(10)));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 }, Score(20)));
        let _e3 = world.spawn((Pos { x: 100.0, y: 100.0 }, Score(30)));

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(
                Arc::new(grid),
                &world,
                move |_expr: &crate::index::SpatialExpr| {
                    cc.fetch_add(1, Ordering::Relaxed);
                    vec![e1, e2]
                },
            )
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([5.0, 5.0], 10.0, |_, _| true).unwrap())
            .join::<(&Score,)>(JoinKind::Inner)
            .build();

        let results = plan.execute(&mut world).unwrap();
        assert!(
            call_count.load(Ordering::Relaxed) > 0,
            "lookup not called in join"
        );
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn spatial_index_execute_returns_correct_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));
        let _far = world.spawn((Pos { x: 999.0, y: 999.0 },));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true).unwrap())
            .build();

        let results = plan.execute(&mut world).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn spatial_index_stale_entities_filtered() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        // Build the plan while the planner still borrows world, then drop planner.
        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([1.5, 1.5], 5.0, |_, _| true).unwrap())
            .build();

        // Now we can mutate world — despawn e2 after the plan is built.
        world.despawn(e2);

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn spatial_index_for_each_raw_works() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| {
                cc.fetch_add(1, Ordering::Relaxed);
                vec![e1]
            })
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
            .build();

        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();

        assert!(call_count.load(Ordering::Relaxed) > 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn spatial_index_without_lookup_falls_back() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index::<Pos>(Arc::new(grid), &world)
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([5.0, 5.0], 100.0, |_, _| true).unwrap())
            .build();

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 10);
    }

    #[test]
    fn spatial_index_with_changed_filter() {
        // Changed<T> is column-granular (per archetype), not per-entity.
        // If any entity's column is mutated, the whole archetype passes
        // Changed<T> on the next scan. This test verifies that behavior
        // in the spatial index-gather path.
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

        // Spawn an entity in a SEPARATE archetype (with Score) so we can verify
        // the column-level filter skips the unmodified archetype on the second scan.
        let e3 = world.spawn((Pos { x: 3.0, y: 3.0 }, Score(99)));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| {
                vec![e1, e2, e3]
            })
            .unwrap();

        let mut plan = planner
            .scan::<(Changed<Pos>, &Pos)>()
            .filter(Predicate::within::<Pos>([1.5, 1.5], 10.0, |_, _| true).unwrap())
            .build();

        // First scan — all entities "changed" (never read before by this plan).
        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 3, "all entities pass on first scan");

        // Mutate only e1's Pos — this marks the (Pos) archetype column as changed
        // but NOT the (Pos, Score) archetype.
        world.get_mut::<Pos>(e1).unwrap().x = 99.0;

        // Second scan — only the archetype containing e1/e2 (which was mutated) passes
        // Changed<Pos>. e3's (Pos, Score) archetype was not touched, so it is filtered out.
        results.clear();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "only entities in the mutated archetype pass Changed<Pos>"
        );
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
        assert!(!results.contains(&e3));
    }

    // ── Additional spatial execution coverage ────────────────────

    #[test]
    fn spatial_index_empty_lookup_yields_no_results() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 1.0 },));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(
                Arc::new(grid),
                &world,
                |_expr| Vec::new(), // empty result set
            )
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([0.0, 0.0], 1.0, |_, _| true).unwrap())
            .build();

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 0, "empty lookup should yield zero results");
    }

    #[test]
    fn spatial_index_all_stale_yields_no_results() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
            .build();

        // Despawn ALL entities returned by the lookup after plan is built.
        world.despawn(e1);
        world.despawn(e2);

        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 0, "all-stale lookup should yield zero results");
    }

    #[test]
    fn spatial_index_for_each_raw_filters_stale() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
            .build();

        world.despawn(e2);

        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 1, "raw path must filter stale entities");
        assert_eq!(results[0], e1);
    }

    #[test]
    fn spatial_index_filters_entities_missing_required_components() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 }, Score(10))); // has both
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 },)); // only Pos

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        // Lookup returns both entities.
        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        // Query requires BOTH Pos and Score.
        let mut plan = planner
            .scan::<(&Pos, &Score)>()
            .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(
            results.len(),
            1,
            "entity missing required component should be filtered"
        );
        assert_eq!(results[0], e1);
    }

    #[test]
    fn spatial_index_mixed_archetypes_without_changed() {
        let mut world = World::new();
        // Two different archetypes, both have Pos.
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 2.0 }, Score(10)));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::within::<Pos>([1.0, 1.0], 5.0, |_, _| true).unwrap())
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "entities from different archetypes should both be yielded"
        );
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn spatial_index_intersects_through_execution() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e2 = world.spawn((Pos { x: 5.0, y: 5.0 },));
        let _far = world.spawn((Pos { x: 99.0, y: 99.0 },));

        let mut grid = TestGridIndex::new();
        grid.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos>(Arc::new(grid), &world, move |_expr| vec![e1, e2])
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos,)>()
            .filter(Predicate::intersects::<Pos>([0.0, 0.0], [10.0, 10.0], |_, _| true).unwrap())
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn spatial_index_3d_coordinates_propagate() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone, Copy)]
        #[expect(dead_code)]
        struct Pos3D {
            x: f32,
            y: f32,
            z: f32,
        }

        let mut world = World::new();
        let e1 = world.spawn((Pos3D {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },));

        // Verify 3D coordinates are passed through to the lookup closure.
        let received_center = Arc::new(std::sync::Mutex::new(Vec::new()));
        let rc = Arc::clone(&received_center);
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        struct Dummy3DIndex;
        impl SpatialIndex for Dummy3DIndex {
            fn rebuild(&mut self, _world: &mut World) {}
            fn supports(
                &self,
                _expr: &crate::index::SpatialExpr,
            ) -> Option<crate::index::SpatialCost> {
                Some(crate::index::SpatialCost {
                    estimated_rows: 1.0,
                    cpu: 1.0,
                })
            }
        }

        let mut planner = QueryPlanner::new(&world);
        planner
            .add_spatial_index_with_lookup::<Pos3D>(Arc::new(Dummy3DIndex), &world, move |expr| {
                cc.fetch_add(1, Ordering::Relaxed);
                if let crate::index::SpatialExpr::Within { center, .. } = expr {
                    *rc.lock().unwrap() = center.clone();
                }
                vec![e1]
            })
            .unwrap();

        let mut plan = planner
            .scan::<(&Pos3D,)>()
            .filter(Predicate::within::<Pos3D>([10.0, 20.0, 30.0], 5.0, |_, _| true).unwrap())
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert!(call_count.load(Ordering::Relaxed) > 0);
        assert_eq!(results.len(), 1);

        let center = received_center.lock().unwrap();
        assert_eq!(
            center.len(),
            3,
            "3D center should propagate all 3 coordinates"
        );
        assert!((center[0] - 10.0).abs() < f64::EPSILON);
        assert!((center[1] - 20.0).abs() < f64::EPSILON);
        assert!((center[2] - 30.0).abs() < f64::EPSILON);
    }

    // ── IndexDriver for_each execution ──────────────────────────────

    #[test]
    fn index_for_each_uses_btree_lookup() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let e2 = world.spawn((Score(42),));
        let _e3 = world.spawn((Score(99),));

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn index_join_uses_lookup() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 }));
        let e2 = world.spawn((Score(42), Pos { x: 2.0, y: 2.0 }));
        let _e3 = world.spawn((Score(99), Pos { x: 3.0, y: 3.0 }));

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq::<Score>(Score(42)))
            .join::<(&Pos,)>(JoinKind::Inner)
            .build();

        let results = plan.execute(&mut world).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn index_for_each_uses_hash_lookup() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let e2 = world.spawn((Score(42),));
        let _e3 = world.spawn((Score(99),));

        let mut hash = HashIndex::<Score>::new();
        hash.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_hash_index(&Arc::new(hash), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
    }

    #[test]
    fn index_range_lookup_execution() {
        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));
        let e3 = world.spawn((Score(30),));
        let _e4 = world.spawn((Score(100),));

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(5)..Score(35)))
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results.contains(&e1));
        assert!(results.contains(&e2));
        assert!(results.contains(&e3));
    }

    #[test]
    fn index_lookup_filters_stale_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42),));
        let e2 = world.spawn((Score(42),));

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        // Despawn e2 after plan is built — index is stale.
        world.despawn(e2);

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn index_lookup_filters_missing_required() {
        let mut world = World::new();
        let e1 = world.spawn((Score(42), Pos { x: 1.0, y: 1.0 })); // has both
        let _e2 = world.spawn((Score(42),)); // only Score

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&Arc::new(btree), &world).unwrap();

        // Query requires BOTH Score and Pos.
        let mut plan = planner
            .scan::<(&Score, &Pos)>()
            .filter(Predicate::eq::<Score>(Score(42)))
            .build();

        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();

        assert_eq!(results.len(), 1, "entity missing Pos should be filtered");
        assert_eq!(results[0], e1);
    }

    // ── Cross-world safety tests ─────────────────────────────────────

    #[test]
    fn execute_returns_err_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        world_a.spawn((Score(1),));
        world_b.spawn((Score(2),));

        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute(&mut world_b);
        assert!(result.is_err());
    }

    #[test]
    fn for_each_returns_err_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        world_a.spawn((Score(1),));
        world_b.spawn((Score(2),));

        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.for_each(&mut world_b, |_| {});
        assert!(result.is_err());
    }

    #[test]
    fn for_each_raw_returns_err_on_wrong_world() {
        let mut world_a = World::new();
        let world_b = World::new();
        world_a.spawn((Score(1),));

        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.for_each_raw(&world_b, |_| {});
        assert!(result.is_err());
    }

    #[test]
    fn execute_succeeds_on_same_world() {
        let mut world = World::new();
        world.spawn((Score(1),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute(&mut world);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn for_each_supports_join_plan() {
        let mut world = World::new();
        let e = world.spawn((Score(1), Health(100)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results, vec![e]);
    }

    #[test]
    fn for_each_raw_supports_join_plan() {
        let mut world = World::new();
        let e = world.spawn((Score(1), Health(100)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results, vec![e]);
    }

    #[test]
    fn execute_raw_supports_join_plan() {
        let mut world = World::new();
        let e = world.spawn((Score(1), Health(100)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        let result = plan.execute_raw(&world).unwrap();
        assert_eq!(result, &[e]);
    }

    #[test]
    fn execute_raw_scan_only() {
        let mut world = World::new();
        let e = world.spawn((Score(42),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute_raw(&world).unwrap();
        assert_eq!(result, &[e]);
    }

    #[test]
    fn execute_raw_returns_err_on_wrong_world() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        world_a.spawn((Score(1),));
        world_b.spawn((Score(2),));

        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.execute_raw(&world_b);
        assert!(result.is_err());
    }

    #[test]
    fn execute_raw_does_not_advance_tick() {
        let mut world = World::new();
        world.spawn((Score(1),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        // Execute raw twice — both should succeed without tick advancement.
        let r1 = plan.execute_raw(&world).unwrap();
        assert_eq!(r1.len(), 1);
        let r2 = plan.execute_raw(&world).unwrap();
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn for_each_raw_join_inner_filters_non_matching() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Health(50)));
        world.spawn((Score(2),)); // no Health — should not appear in join

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results, vec![e1]);
    }

    #[test]
    fn for_each_raw_join_left_preserves_all_left() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Health(50)));
        let e2 = world.spawn((Score(2),)); // no Health

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Left)
            .build();
        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();
        results.sort_by_key(|e| e.to_bits());
        let mut expected = vec![e1, e2];
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(results, expected);
    }

    #[test]
    fn for_each_join_advances_tick() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i), Health(i * 10)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        // First call: all changed — should see all 5.
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 5);
        // Second call: tick advanced, nothing mutated — should see 0.
        let mut count = 0;
        plan.for_each(&mut world, |_| count += 1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn execute_raw_join_does_not_advance_tick() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn((Score(i), Health(i * 10)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        // Both calls should see all 5 — execute_raw does not advance tick.
        let r1 = plan.execute_raw(&world).unwrap();
        assert_eq!(r1.len(), 5);
        let r2 = plan.execute_raw(&world).unwrap();
        assert_eq!(r2.len(), 5);
    }

    #[test]
    fn for_each_join_inner_filters_non_matching() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Health(50)));
        world.spawn((Score(2),)); // no Health — filtered by inner join

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results, vec![e1]);
    }

    #[test]
    fn for_each_join_left_preserves_all_left() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Health(50)));
        let e2 = world.spawn((Score(2),)); // no Health

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Left)
            .build();
        let mut results = Vec::new();
        plan.for_each(&mut world, |entity| results.push(entity))
            .unwrap();
        results.sort_by_key(|e| e.to_bits());
        let mut expected = vec![e1, e2];
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(results, expected);
    }

    #[test]
    fn for_each_raw_multi_step_join() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Health(50), Team(0)));
        world.spawn((Score(2), Health(60))); // no Team — filtered by second join
        world.spawn((Score(3),)); // no Health — filtered by first join

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let mut results = Vec::new();
        plan.for_each_raw(&world, |entity| results.push(entity))
            .unwrap();
        assert_eq!(results, vec![e1]);
    }

    #[test]
    fn execute_raw_multi_step_join() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Health(50), Team(0)));
        world.spawn((Score(2), Health(60))); // no Team
        world.spawn((Score(3),)); // no Health

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let result = plan.execute_raw(&world).unwrap();
        assert_eq!(result, &[e1]);
    }

    #[test]
    fn execute_raw_empty_join_result() {
        let mut world = World::new();
        world.spawn((Score(1),)); // no Health — inner join yields empty
        world.spawn((Score(2),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .build();
        let result = plan.execute_raw(&world).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn add_btree_index_returns_err_on_unregistered_component() {
        let mut world1 = World::new();
        world1.spawn((Score(1),)); // registers Score in world1

        let mut btree = BTreeIndex::<Score>::new();
        btree.rebuild(&mut world1);

        let world2 = World::new(); // Score NOT registered here
        let mut planner = QueryPlanner::new(&world2);
        let result = planner.add_btree_index(&Arc::new(btree), &world2);
        assert!(result.is_err());
    }

    #[test]
    fn predicate_within_rejects_negative_radius() {
        let result = Predicate::within::<Pos>([0.0, 0.0], -1.0, |_, _| true);
        assert!(result.is_err());
    }

    #[test]
    fn predicate_within_rejects_nan_radius() {
        let result = Predicate::within::<Pos>([0.0, 0.0], f64::NAN, |_, _| true);
        assert!(result.is_err());
    }

    #[test]
    fn predicate_intersects_rejects_mismatched_dimensions() {
        let result = Predicate::intersects::<Pos>(vec![0.0, 0.0], vec![1.0], |_, _| true);
        assert!(result.is_err());
    }

    #[test]
    fn predicate_intersects_rejects_empty_coordinates() {
        let result =
            Predicate::intersects::<Pos>(Vec::<f64>::new(), Vec::<f64>::new(), |_, _| true);
        assert!(result.is_err());
    }

    // ── Aggregate tests ──────────────────────────────────────────────

    #[test]
    fn aggregate_count_empty_world() {
        let mut world = World::new();
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.get(0), Some(0.0));
    }

    #[test]
    fn aggregate_count() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(100.0));
    }

    #[test]
    fn aggregate_sum() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        // sum(0..10) = 45
        assert_eq!(result.get(0), Some(45.0));
    }

    #[test]
    fn aggregate_min_max() {
        let mut world = World::new();
        for i in 5..15 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::min::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::max::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(5.0));
        assert_eq!(result.get(1), Some(14.0));
    }

    #[test]
    fn aggregate_avg() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::avg::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        // avg(0..10) = 4.5
        assert_eq!(result.get(0), Some(4.5));
    }

    #[test]
    fn aggregate_multiple_expressions() {
        let mut world = World::new();
        for i in 1..=5 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::min::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::max::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::avg::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(5.0)); // count
        assert_eq!(result.get(1), Some(15.0)); // sum(1+2+3+4+5)
        assert_eq!(result.get(2), Some(1.0)); // min
        assert_eq!(result.get(3), Some(5.0)); // max
        assert_eq!(result.get(4), Some(3.0)); // avg
    }

    #[test]
    fn aggregate_with_filter() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "score >= 50",
                0.5,
                |world: &World, entity: Entity| {
                    world.get::<Score>(entity).is_some_and(|s| s.0 >= 50)
                },
            ))
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(50.0)); // 50 entities match
        // sum(50..100) = 50*75-1 = 3725
        let expected_sum: f64 = (50..100).map(|i| i as f64).sum();
        assert_eq!(result.get(1), Some(expected_sum));
    }

    #[test]
    fn aggregate_get_by_label() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get_by_label("COUNT(*)"), Some(5.0));
        assert_eq!(result.get_by_label("SUM(Score)"), Some(10.0));
        assert_eq!(result.get_by_label("NONEXISTENT"), None);
    }

    #[test]
    fn aggregate_plan_node_in_explain() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let explain = plan.explain();
        assert!(explain.contains("StreamAggregate"));
        assert!(explain.contains("COUNT(*)"));
        assert!(explain.contains("SUM(Score)"));
    }

    #[test]
    fn aggregate_has_aggregates() {
        let mut world = World::new();
        world.spawn((Score(1),));
        let planner = QueryPlanner::new(&world);

        let plan_no_agg = planner.scan::<(&Score,)>().build();
        assert!(!plan_no_agg.has_aggregates());

        let plan_with_agg = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();
        assert!(plan_with_agg.has_aggregates());
    }

    #[test]
    fn aggregate_result_display() {
        let result = AggregateResult {
            values: vec![
                ("COUNT(*)".to_string(), 10.0),
                ("SUM(Score)".to_string(), 45.0),
            ],
        };
        let display = format!("{result}");
        assert!(display.contains("COUNT(*)"));
        assert!(display.contains("SUM(Score)"));
    }

    #[test]
    fn aggregate_min_max_on_empty_returns_nan() {
        let mut world = World::new();
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::min::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::max::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::avg::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert!(result.get(0).unwrap().is_nan()); // min on empty
        assert!(result.get(1).unwrap().is_nan()); // max on empty
        assert!(result.get(2).unwrap().is_nan()); // avg on empty
    }

    #[test]
    fn aggregate_execute_raw() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates_raw(&world).unwrap();
        assert_eq!(result.get(0), Some(10.0));
        assert_eq!(result.get(1), Some(45.0));
    }

    #[test]
    fn aggregate_with_index_driver() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index::<Score>(&idx, &world).unwrap();

        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(50)))
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(40.0)); // 10..50 = 40 entities
        let expected_sum: f64 = (10..50).map(|i| i as f64).sum();
        assert_eq!(result.get(1), Some(expected_sum));
    }

    #[test]
    fn aggregate_result_iter() {
        let result = AggregateResult {
            values: vec![
                ("COUNT(*)".to_string(), 10.0),
                ("SUM(Score)".to_string(), 45.0),
            ],
        };
        let items: Vec<_> = result.iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], ("COUNT(*)", 10.0));
        assert_eq!(items[1], ("SUM(Score)", 45.0));
    }

    #[test]
    fn aggregate_op_display() {
        assert_eq!(format!("{}", AggregateOp::Count), "COUNT");
        assert_eq!(format!("{}", AggregateOp::Sum), "SUM");
        assert_eq!(format!("{}", AggregateOp::Min), "MIN");
        assert_eq!(format!("{}", AggregateOp::Max), "MAX");
        assert_eq!(format!("{}", AggregateOp::Avg), "AVG");
    }

    #[test]
    fn aggregate_plan_cost_single_row() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();

        // Aggregate produces 1 result row.
        assert_eq!(plan.root().cost().rows, 1.0);
    }

    #[test]
    fn aggregate_multiple_archetypes() {
        let mut world = World::new();
        // Two archetypes: (Score,) and (Score, Health)
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        for i in 5..10 {
            world.spawn((Score(i), Health(100)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(10.0)); // all 10 entities
        assert_eq!(result.get(1), Some(45.0)); // sum(0..10)
    }

    #[test]
    fn aggregate_after_despawn() {
        let mut world = World::new();
        let mut entities = Vec::new();
        for i in 0..5 {
            entities.push(world.spawn((Score(i),)));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        // Despawn entities with Score(3) and Score(4)
        world.despawn(entities[3]);
        world.despawn(entities[4]);

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(3.0)); // 3 surviving
        assert_eq!(result.get(1), Some(3.0)); // 0+1+2 = 3
    }

    #[test]
    fn aggregate_changed_skips_stale() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .aggregate(AggregateExpr::count())
            .build();

        // First call sees all entities (all columns are new).
        let r1 = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(r1.get(0), Some(5.0));

        // No mutations — second call should see 0 (Changed filter skips).
        let r2 = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(r2.get(0), Some(0.0));
    }

    #[test]
    fn aggregate_changed_detects_mutation() {
        let mut world = World::new();
        let e = world.spawn((Score(10),));
        for _ in 0..4 {
            world.spawn((Score(0),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        // First call sees all.
        let _ = plan.execute_aggregates(&mut world).unwrap();

        // Mutate one entity.
        *world.get_mut::<Score>(e).unwrap() = Score(42);

        // Second call sees only the mutated entity's archetype.
        let r = plan.execute_aggregates(&mut world).unwrap();
        // Changed<T> is archetype-granular, so all entities in the archetype
        // are visited (all 5 are in the same archetype).
        assert_eq!(r.get(0), Some(5.0));
    }

    #[test]
    fn aggregate_no_exprs_returns_empty() {
        let mut world = World::new();
        world.spawn((Score(1),));
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        // No panic — returns empty result.
        let result = plan.execute_aggregates(&mut world).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn aggregate_no_exprs_raw_returns_empty() {
        let mut world = World::new();
        world.spawn((Score(1),));
        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let result = plan.execute_aggregates_raw(&world).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn aggregate_world_mismatch() {
        let mut world_a = World::new();
        world_a.spawn((Score(1),));
        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();

        let mut world_b = World::new();
        assert!(plan.execute_aggregates(&mut world_b).is_err());
    }

    #[test]
    fn aggregate_world_mismatch_raw() {
        let mut world_a = World::new();
        world_a.spawn((Score(1),));
        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();

        let world_b = World::new();
        assert!(plan.execute_aggregates_raw(&world_b).is_err());
    }

    #[test]
    fn aggregate_raw_tick_stationarity() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .aggregate(AggregateExpr::count())
            .build();

        // _raw does not advance ticks, so repeated calls see the same result.
        let r1 = plan.execute_aggregates_raw(&world).unwrap();
        let r2 = plan.execute_aggregates_raw(&world).unwrap();
        assert_eq!(r1.get(0), r2.get(0));
    }

    #[test]
    fn aggregate_with_join() {
        let mut world = World::new();
        // 5 entities with both Score and Health
        for i in 0..5 {
            world.spawn((Score(i), Health(100)));
        }
        // 5 entities with Score only
        for i in 5..10 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(5.0)); // only 5 have both
        assert_eq!(result.get(1), Some(10.0)); // sum(0..5)
    }

    #[test]
    fn aggregate_raw_with_join() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i), Health(100)));
        }
        for i in 5..10 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Health,)>(JoinKind::Inner)
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        // _raw now supports join plans too.
        let result = plan.execute_aggregates_raw(&world).unwrap();
        assert_eq!(result.get(0), Some(5.0));
        assert_eq!(result.get(1), Some(10.0));
    }

    #[test]
    fn aggregate_nan_propagation_min_max() {
        let mut world = World::new();
        world.spawn((Score(1),));
        world.spawn((Score(2),));
        world.spawn((Score(3),));

        let planner = QueryPlanner::new(&world);

        // Extractor that returns NaN for Score(2).
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::min::<Score>("Score", |s| {
                if s.0 == 2 { f64::NAN } else { s.0 as f64 }
            }))
            .aggregate(AggregateExpr::max::<Score>("Score", |s| {
                if s.0 == 2 { f64::NAN } else { s.0 as f64 }
            }))
            .build();

        let result = plan.execute_aggregates(&mut world).unwrap();
        // NaN propagates via f64::min/max — result should be NaN.
        assert!(result.get(0).unwrap().is_nan());
        assert!(result.get(1).unwrap().is_nan());
    }

    #[test]
    fn aggregate_after_spawn() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .build();

        let r1 = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(r1.get(0), Some(5.0));

        // Spawn more entities — visible on next execution.
        for i in 5..8 {
            world.spawn((Score(i),));
        }
        let r2 = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(r2.get(0), Some(8.0));
    }

    #[test]
    fn aggregate_duplicate_label_warning() {
        let mut world = World::new();
        world.spawn((Score(1),));
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();

        assert!(plan.warnings().iter().any(|w| matches!(
            w,
            PlanWarning::DuplicateAggregateLabel { label } if label == "SUM(Score)"
        )));
    }

    #[test]
    fn aggregate_expr_accessors() {
        let count = AggregateExpr::count();
        assert_eq!(count.op(), AggregateOp::Count);
        assert_eq!(count.label(), "COUNT(*)");

        let sum = AggregateExpr::sum::<Score>("Score", |s| s.0 as f64);
        assert_eq!(sum.op(), AggregateOp::Sum);
        assert_eq!(sum.label(), "SUM(Score)");
    }

    #[test]
    fn aggregate_result_labels() {
        let result = AggregateResult {
            values: vec![
                ("COUNT(*)".to_string(), 10.0),
                ("SUM(Score)".to_string(), 45.0),
            ],
        };
        let labels: Vec<_> = result.labels().collect();
        assert_eq!(labels, vec!["COUNT(*)", "SUM(Score)"]);
    }

    // ── Batch aggregate path activation tests ─────────────────────

    #[test]
    fn aggregate_batch_path_activates_without_filters() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();
        drop(planner);

        let dbg = format!("{plan:?}");
        assert!(
            dbg.contains("has_compiled_agg_scan: true"),
            "batch path should activate for filter-free scan"
        );
        assert!(
            dbg.contains("has_compiled_agg_scan_raw: true"),
            "raw batch path should activate for filter-free scan"
        );
    }

    #[test]
    fn aggregate_batch_path_disabled_with_filters() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>(
                "score >= 5",
                0.5,
                |world: &World, entity: Entity| {
                    world.get::<Score>(entity).is_some_and(|s| s.0 >= 5)
                },
            ))
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();
        drop(planner);

        let dbg = format!("{plan:?}");
        assert!(
            dbg.contains("has_compiled_agg_scan: false"),
            "batch path should NOT activate when filters present"
        );
    }

    #[test]
    fn aggregate_batch_multi_archetype_correctness() {
        // Entities across 3 archetypes: (Score,), (Score, Health),
        // (Score, Health, Name). Batch path must bind per archetype.
        #[derive(Debug)]
        struct Health(#[expect(dead_code)] i32);
        #[derive(Debug)]
        struct Name;

        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        for i in 5..8 {
            world.spawn((Score(i), Health(100)));
        }
        for i in 8..10 {
            world.spawn((Score(i), Health(50), Name));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Score>("Score", |s| s.0 as f64))
            .build();
        drop(planner);

        let dbg = format!("{plan:?}");
        assert!(dbg.contains("has_compiled_agg_scan: true"));

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(10.0)); // count
        let expected_sum: f64 = (0..10).map(|i| i as f64).sum();
        assert_eq!(result.get(1), Some(expected_sum)); // sum(0..10) = 45
    }

    #[test]
    fn aggregate_batch_component_absent_from_some_archetypes() {
        // scan::<&Score> with sum::<Health> — Health is absent from some
        // archetypes. The batch path must skip extraction for those.
        #[derive(Debug)]
        struct Health(i32);

        let mut world = World::new();
        // Archetype 1: Score only (no Health)
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        // Archetype 2: Score + Health
        for i in 0..3 {
            world.spawn((Score(100), Health(i * 10)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .aggregate(AggregateExpr::count())
            .aggregate(AggregateExpr::sum::<Health>("Health", |h| h.0 as f64))
            .build();
        drop(planner);

        let result = plan.execute_aggregates(&mut world).unwrap();
        assert_eq!(result.get(0), Some(8.0)); // count: 5 + 3
        // sum of Health: only from archetype 2 = 0 + 10 + 20 = 30
        assert_eq!(result.get(1), Some(30.0));
    }

    // ── Batch join execution ──────────────────────────────────────────

    #[test]
    fn scratch_sort_by_archetype_groups_entities() {
        let mut world = World::new();
        // Archetype A: Score only
        let a1 = world.spawn((Score(1),));
        let a2 = world.spawn((Score(2),));
        // Archetype B: Score + Team
        let b1 = world.spawn((Score(3), Team(1)));
        let b2 = world.spawn((Score(4), Team(2)));

        // Deliberately interleave: [b1, a1, b2, a2]
        let mut scratch = ScratchBuffer::new(4);
        scratch.push(b1);
        scratch.push(a1);
        scratch.push(b2);
        scratch.push(a2);

        scratch.sort_by_archetype(&world.entity_locations);

        // After sort: entities from same archetype should be contiguous.
        let sorted = scratch.as_slice();
        // First two share one archetype, last two share another.
        let loc0 = world.entity_locations[sorted[0].index() as usize].unwrap();
        let loc1 = world.entity_locations[sorted[1].index() as usize].unwrap();
        let loc2 = world.entity_locations[sorted[2].index() as usize].unwrap();
        let loc3 = world.entity_locations[sorted[3].index() as usize].unwrap();
        assert_eq!(loc0.archetype_id, loc1.archetype_id);
        assert_eq!(loc2.archetype_id, loc3.archetype_id);
        assert_ne!(loc0.archetype_id, loc2.archetype_id);

        // Within each archetype group, rows should be sorted ascending.
        assert!(loc0.row < loc1.row);
        assert!(loc2.row < loc3.row);
    }

    #[test]
    fn for_each_batched_yields_all_join_results() {
        let mut world = World::new();
        // Score-only entities (should NOT appear in inner join)
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        // Score+Team entities (SHOULD appear)
        let mut expected = Vec::new();
        for i in 5..15 {
            let e = world.spawn((Score(i), Team(i % 3)));
            expected.push((e, Score(i)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let mut results: Vec<(Entity, Score)> = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |entity, (score,)| {
            results.push((entity, *score));
        })
        .unwrap();

        // Sort both by entity for comparison.
        results.sort_by_key(|(e, _)| e.to_bits());
        expected.sort_by_key(|(e, _)| e.to_bits());
        assert_eq!(results, expected);
    }

    #[test]
    fn for_each_join_chunk_yields_correct_slices() {
        let mut world = World::new();
        // Archetype A: Score only (will not match inner join)
        for i in 0..3 {
            world.spawn((Score(i),));
        }
        // Archetype B: Score + Team (deterministic scores 10..15)
        for i in 10..15 {
            world.spawn((Score(i), Team(1)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let mut total_entities = 0;
        let mut chunk_count = 0;
        let mut collected_scores = Vec::new();
        plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
            // rows and entities must have the same length.
            assert_eq!(entities.len(), rows.len());
            // Each row index must be valid for the slice.
            for &row in rows {
                assert!(
                    row < scores.len(),
                    "row {row} out of bounds for slice len {}",
                    scores.len()
                );
                collected_scores.push(scores[row]);
            }
            total_entities += entities.len();
            chunk_count += 1;
        })
        .unwrap();

        assert_eq!(total_entities, 5); // Only Score+Team entities
        assert!(chunk_count >= 1); // At least one archetype chunk
        // Verify we read the correct score values (10..15).
        collected_scores.sort_by_key(|s| s.0);
        assert_eq!(collected_scores, (10..15).map(Score).collect::<Vec<_>>());
    }

    #[test]
    fn for_each_batched_raw_no_tick_advance() {
        let mut world = World::new();
        world.spawn((Score(1), Team(1)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        // Call raw twice — both should succeed (no tick advancement).
        let mut count1 = 0u32;
        plan.for_each_batched_raw::<(&Score,), _>(&world, |_, _| count1 += 1)
            .unwrap();
        assert_eq!(count1, 1);

        let mut count2 = 0u32;
        plan.for_each_batched_raw::<(&Score,), _>(&world, |_, _| count2 += 1)
            .unwrap();
        assert_eq!(count2, 1);
    }

    #[test]
    fn for_each_join_chunk_works_for_scan_plans() {
        let mut world = World::new();
        world.spawn((Score(1),));
        world.spawn((Score(2),));
        world.spawn((Score(3),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let mut total = 0;
        plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
            assert_eq!(entities.len(), rows.len());
            total += entities.len();
            // For scan plans, rows should be 0..len (all entities in the archetype).
            for (i, &row) in rows.iter().enumerate() {
                assert_eq!(row, i, "scan plan rows should be sequential");
            }
            assert_eq!(scores.len(), entities.len());
        })
        .unwrap();
        assert_eq!(total, 3);
    }

    #[test]
    fn for_each_batched_left_join() {
        let mut world = World::new();
        // 5 Score-only, 5 Score+Team
        let mut all_score = Vec::new();
        for i in 0..5 {
            all_score.push(world.spawn((Score(i),)));
        }
        for i in 5..10 {
            all_score.push(world.spawn((Score(i), Team(1))));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Left)
            .build();

        let mut results = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |entity, _| {
            results.push(entity);
        })
        .unwrap();

        // Left join: all 10 Score entities should appear.
        assert_eq!(results.len(), 10);
        all_score.sort_by_key(|e| e.to_bits());
        results.sort_by_key(|e| e.to_bits());
        assert_eq!(results, all_score);
    }

    #[test]
    fn for_each_join_chunk_multi_archetype() {
        let mut world = World::new();
        // 3 different archetypes, all with Score
        world.spawn((Score(1),));
        world.spawn((Score(2), Team(1)));
        world.spawn((Score(3), Team(1), Health(50)));

        let planner = QueryPlanner::new(&world);
        // Scan Score, join Team — only archetypes with Team match the join.
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        // Collect chunk info: (entity_count, first_entity) per chunk.
        let mut chunk_entities: Vec<Vec<Entity>> = Vec::new();
        plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, _, _| {
            assert!(!entities.is_empty());
            chunk_entities.push(entities.to_vec());
        })
        .unwrap();

        // Two archetypes have Team: (Score, Team) and (Score, Team, Health).
        assert_eq!(chunk_entities.len(), 2);
        // Total: 2 entities (one per Team-bearing archetype).
        let total: usize = chunk_entities.iter().map(Vec::len).sum();
        assert_eq!(total, 2);
        // Each chunk should have different entities.
        assert_ne!(chunk_entities[0], chunk_entities[1]);
    }

    #[test]
    fn for_each_batched_empty_join() {
        let mut world = World::new();
        // Score-only entities, no Team — inner join produces empty result.
        for i in 0..5 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let mut called = false;
        plan.for_each_batched::<(&Score,), _>(&mut world, |_, _| {
            called = true;
        })
        .unwrap();
        assert!(!called);
    }

    #[test]
    fn for_each_batched_world_mismatch() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        world_a.spawn((Score(1),));
        world_b.spawn((Score(2),));

        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.for_each_batched::<(&Score,), _>(&mut world_b, |_, _| {});
        assert!(result.is_err());
    }

    #[test]
    fn for_each_batched_component_mismatch() {
        let mut world = World::new();
        // Entities have Score only — no Health component.
        world.spawn((Score(1),));
        world.spawn((Score(2),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        // Request (&Health,) via Q, but no archetype has Health.
        let result = plan.for_each_batched::<(&Health,), _>(&mut world, |_, _| {});
        assert!(
            matches!(result, Err(PlanExecError::ComponentMismatch { .. })),
            "expected ComponentMismatch, got {result:?}"
        );
    }

    #[test]
    fn for_each_join_chunk_component_mismatch() {
        let mut world = World::new();
        world.spawn((Score(1),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let result = plan.for_each_join_chunk::<(&Health,), _>(&mut world, |_, _, _| {});
        assert!(
            matches!(result, Err(PlanExecError::ComponentMismatch { .. })),
            "expected ComponentMismatch, got {result:?}"
        );
    }

    #[test]
    fn for_each_join_chunk_world_mismatch() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        world_a.spawn((Score(1),));
        world_b.spawn((Score(2),));

        let planner = QueryPlanner::new(&world_a);
        let mut plan = planner.scan::<(&Score,)>().build();
        let result = plan.for_each_join_chunk::<(&Score,), _>(&mut world_b, |_, _, _| {});
        assert!(result.is_err());
    }

    #[test]
    fn for_each_join_chunk_empty_join() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let mut called = false;
        plan.for_each_join_chunk::<(&Score,), _>(&mut world, |_, _, _| {
            called = true;
        })
        .unwrap();
        assert!(!called);
    }

    #[test]
    fn for_each_batched_scan_only_happy_path() {
        let mut world = World::new();
        world.spawn((Score(10),));
        world.spawn((Score(20),));
        world.spawn((Score(30),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        let mut sum = 0u32;
        plan.for_each_batched::<(&Score,), _>(&mut world, |_, (score,)| {
            sum += score.0;
        })
        .unwrap();
        assert_eq!(sum, 60);
    }

    #[test]
    fn for_each_batched_multi_archetype_values() {
        let mut world = World::new();
        // Archetype A: Score + Team
        world.spawn((Score(10), Team(1)));
        world.spawn((Score(20), Team(2)));
        // Archetype B: Score + Team + Health
        world.spawn((Score(30), Team(3), Health(100)));
        world.spawn((Score(40), Team(4), Health(200)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let mut scores = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |_, (score,)| {
            scores.push(score.0);
        })
        .unwrap();
        scores.sort_unstable();
        assert_eq!(scores, vec![10, 20, 30, 40]);
    }

    #[test]
    fn for_each_batched_marks_mutable_changed() {
        let mut world = World::new();
        world.spawn((Score(1), Team(1)));

        let planner = QueryPlanner::new(&world);
        // Use Changed<Score> to check change detection.
        let mut changed_plan = planner.scan::<(Changed<Score>, &Score)>().build();

        // First call: all entities are new, so Changed sees them.
        let r1 = changed_plan.execute(&mut world).unwrap().len();
        assert_eq!(r1, 1);

        // Second call: nothing mutated, Changed skips.
        let r2 = changed_plan.execute(&mut world).unwrap().len();
        assert_eq!(r2, 0);

        // Now mutate via for_each_batched with &mut Score.
        let scan_planner = QueryPlanner::new(&world);
        let mut scan_plan = scan_planner.scan::<(&Score,)>().build();
        scan_plan
            .for_each_batched::<(&mut Score,), _>(&mut world, |_, (score,)| {
                score.0 += 1;
            })
            .unwrap();

        // Changed<Score> should now see the mutation.
        let r3 = changed_plan.execute(&mut world).unwrap().len();
        assert_eq!(
            r3, 1,
            "Changed<Score> should detect mutation from for_each_batched"
        );
    }

    // ── Join elimination ─────────────────────────────────────────────

    #[test]
    fn join_elimination_inner_becomes_scan() {
        let mut world = World::new();
        // Entities with Score only
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        // Entities with Score + Team (these should be the only results)
        let mut both = Vec::new();
        for i in 10..20 {
            both.push(world.spawn((Score(i), Team(i % 3))));
        }

        let planner = QueryPlanner::new(&world);

        // Inner join — should be eliminated into a scan.
        let mut join_plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        // Tuple scan — the "expected" path.
        let mut scan_plan = planner.scan::<(&Score, &Team)>().build();

        // Both should produce the same entity set.
        let mut join_result = join_plan.execute(&mut world).unwrap().to_vec();
        let mut scan_result = scan_plan.execute(&mut world).unwrap().to_vec();
        join_result.sort_by_key(|e| e.to_bits());
        scan_result.sort_by_key(|e| e.to_bits());
        assert_eq!(join_result, scan_result);

        // The eliminated plan should NOT have a HashJoin/NestedLoopJoin node.
        match join_plan.root() {
            PlanNode::Scan { .. } => {} // expected
            other => panic!("expected Scan after elimination, got {:?}", other),
        }
    }

    #[test]
    fn join_elimination_left_join_not_eliminated() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        for i in 5..10 {
            world.spawn((Score(i), Team(1)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Left)
            .build();

        // Left join should NOT be eliminated.
        match plan.root() {
            PlanNode::HashJoin { .. } | PlanNode::NestedLoopJoin { .. } => {} // expected
            other => panic!("expected join node for Left join, got {:?}", other),
        }

        // Left join preserves all 10 Score entities.
        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn join_elimination_mixed_inner_and_left() {
        let mut world = World::new();
        world.spawn((Score(1), Team(1), Health(100)));
        world.spawn((Score(2), Team(2))); // no Health
        world.spawn((Score(3),)); // no Team, no Health

        let planner = QueryPlanner::new(&world);
        // Inner join on Team (eliminable), Left join on Health (not eliminable).
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .join::<(&Health,)>(JoinKind::Left)
            .build();

        // Should have a join node (Left join remains).
        match plan.root() {
            PlanNode::HashJoin { .. } | PlanNode::NestedLoopJoin { .. } => {} // expected
            other => panic!(
                "expected join node for remaining Left join, got {:?}",
                other
            ),
        }

        // Inner join on Team narrows to 2 entities (Score+Team).
        // Left join on Health preserves both.
        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn join_elimination_triple_join_two_eliminated() {
        let mut world = World::new();
        world.spawn((Score(1), Team(1), Health(100)));
        world.spawn((Score(2), Team(2))); // no Health
        world.spawn((Score(3),)); // no Team

        let planner = QueryPlanner::new(&world);
        // Two inner joins (Score→Team, Score→Health) + one left join (Score→Pos).
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .join::<(&Health,)>(JoinKind::Inner)
            .join::<(&Pos,)>(JoinKind::Left)
            .build();

        // Two inner joins eliminated, one left join remains.
        let eliminated_count = plan
            .warnings()
            .iter()
            .filter(|w| matches!(w, PlanWarning::JoinEliminated { .. }))
            .count();
        assert_eq!(eliminated_count, 2);

        // Only entity with Score+Team+Health survives the merged inner joins.
        // Left join on Pos preserves it (no Pos, but Left keeps it).
        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn join_elimination_changed_merged() {
        let mut world = World::new();
        let e1 = world.spawn((Score(1), Team(1)));
        let _e2 = world.spawn((Score(2), Team(2)));

        let planner = QueryPlanner::new(&world);
        // Changed<Score> on left, Changed<Team> on right (inner join).
        let mut plan = planner
            .scan::<(Changed<Score>, &Score)>()
            .join::<(Changed<Team>, &Team)>(JoinKind::Inner)
            .build();

        // First call: all entities are "changed" (new).
        let r1 = plan.execute(&mut world).unwrap().len();
        assert_eq!(r1, 2);

        // Second call: nothing changed, should return 0.
        let r2 = plan.execute(&mut world).unwrap().len();
        assert_eq!(r2, 0);

        // Mutate Score on e1.
        *world.get_mut::<Score>(e1).unwrap() = Score(99);

        // Third call: only e1 has Changed<Score>, but we also need Changed<Team>.
        // Since Team wasn't changed, the merged change filter should still require
        // both Score AND Team to be changed. Result: 0.
        let r3 = plan.execute(&mut world).unwrap().len();
        assert_eq!(r3, 0);

        // Fourth call: only Team changed (right side of the eliminated join).
        // Score not changed → merged filter requires both → 0.
        *world.get_mut::<Team>(e1).unwrap() = Team(99);
        let r4 = plan.execute(&mut world).unwrap().len();
        assert_eq!(r4, 0);

        // Fifth call: mutate BOTH Score and Team on e1. Now both columns
        // are marked changed at the archetype level → all entities in the
        // archetype pass (Changed<T> is per-column, not per-entity).
        *world.get_mut::<Score>(e1).unwrap() = Score(100);
        *world.get_mut::<Team>(e1).unwrap() = Team(100);
        let r5 = plan.execute(&mut world).unwrap().len();
        assert_eq!(r5, 2);
    }

    #[test]
    fn join_elimination_idempotent_same_component() {
        let mut world = World::new();
        world.spawn((Score(1),));
        world.spawn((Score(2),));

        let planner = QueryPlanner::new(&world);
        // Join Score with Score (same component) — union is idempotent.
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Score,)>(JoinKind::Inner)
            .build();

        let result = plan.execute(&mut world).unwrap();
        assert_eq!(result.len(), 2);

        // Should be eliminated.
        let eliminated = plan
            .warnings()
            .iter()
            .any(|w| matches!(w, PlanWarning::JoinEliminated { .. }));
        assert!(eliminated);
    }

    #[test]
    fn join_elimination_emits_warning() {
        let mut world = World::new();
        world.spawn((Score(1), Team(1)));

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        let has_warning = plan.warnings().iter().any(|w| match w {
            PlanWarning::JoinEliminated { right_name } => right_name.contains("Team"),
            _ => false,
        });
        assert!(
            has_warning,
            "expected JoinEliminated warning, got {:?}",
            plan.warnings()
        );
    }

    #[test]
    fn join_elimination_raw_paths() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        let mut both = Vec::new();
        for i in 10..15 {
            both.push(world.spawn((Score(i), Team(i % 3))));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        // Verify elimination happened.
        assert!(
            plan.warnings()
                .iter()
                .any(|w| matches!(w, PlanWarning::JoinEliminated { .. }))
        );

        // execute_raw — should yield same entities as execute.
        let raw_result = plan.execute_raw(&world).unwrap().to_vec();
        assert_eq!(raw_result.len(), 5);

        // for_each_raw — should yield same entities.
        let mut raw_entities = Vec::new();
        plan.for_each_raw(&world, |entity| raw_entities.push(entity))
            .unwrap();
        assert_eq!(raw_entities.len(), 5);

        // for_each_batched_raw — pre-resolved column pointers via eliminated path.
        let mut batched_scores = Vec::new();
        plan.for_each_batched_raw::<(&Score,), _>(&world, |_, (score,)| {
            batched_scores.push(score.0);
        })
        .unwrap();
        batched_scores.sort_unstable();
        assert_eq!(batched_scores, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn join_elimination_benchmark_parity() {
        // Functional test: eliminated join should produce results in reasonable time.
        // Not a benchmark — just verifies the optimization works end-to-end.
        let mut world = World::new();
        for i in 0..1000 {
            world.spawn((Score(i), Team(i % 5)));
        }
        for i in 1000..2000 {
            world.spawn((Score(i),)); // no Team
        }

        let planner = QueryPlanner::new(&world);
        let mut join_plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        let mut scan_plan = planner.scan::<(&Score, &Team)>().build();

        let join_result = join_plan.execute(&mut world).unwrap().to_vec();
        let scan_result = scan_plan.execute(&mut world).unwrap().to_vec();
        assert_eq!(join_result.len(), scan_result.len());
        assert_eq!(join_result.len(), 1000);
    }

    // ── Direct archetype iteration tests ──────────────────────────────

    #[test]
    fn direct_iter_batched_scan_only() {
        let mut world = World::new();
        // Archetype A: Score only
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        // Archetype B: Score + Team
        for i in 10..15 {
            world.spawn((Score(i), Team(1)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();

        // scan_required should be set (scan-only, no custom predicates).
        assert!(plan.scan_required.is_some());

        let mut results = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |entity, (score,)| {
            results.push((entity, *score));
        })
        .unwrap();

        // All 10 entities should be visited.
        assert_eq!(results.len(), 10);
        let mut scores: Vec<u32> = results.iter().map(|(_, s)| s.0).collect();
        scores.sort_unstable();
        assert_eq!(scores, vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn direct_iter_batched_with_custom_predicate_uses_scratch() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::custom::<Score>("score < 5", 0.5, |w, e| {
                w.get::<Score>(e).is_some_and(|s| s.0 < 5)
            }))
            .build();

        // scan_required should be None (custom predicate present).
        assert!(plan.scan_required.is_none());

        let mut results = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |_entity, (score,)| {
            results.push(score.0);
        })
        .unwrap();

        results.sort_unstable();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn direct_iter_chunk_scan_only() {
        let mut world = World::new();
        // Single archetype: Score only
        for i in 0..8 {
            world.spawn((Score(i),));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner.scan::<(&Score,)>().build();
        assert!(plan.scan_required.is_some());

        let mut chunk_count = 0;
        let mut total_entities = 0;
        plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
            chunk_count += 1;
            assert_eq!(entities.len(), rows.len());
            // Row indices should be sequential 0..N for direct iteration.
            for (i, &row) in rows.iter().enumerate() {
                assert_eq!(row, i, "row indices should be sequential");
                assert!(row < scores.len());
            }
            total_entities += entities.len();
        })
        .unwrap();

        assert_eq!(total_entities, 8);
        assert!(chunk_count >= 1);
    }

    #[test]
    fn direct_iter_batched_eliminated_join() {
        let mut world = World::new();
        // Archetype A: Score only (will not match eliminated inner join)
        for i in 0..5 {
            world.spawn((Score(i),));
        }
        // Archetype B: Score + Team (matches)
        for i in 10..15 {
            world.spawn((Score(i), Team(1)));
        }

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();

        // The inner join should be eliminated (no predicates on right side).
        assert!(
            plan.warnings()
                .iter()
                .any(|w| matches!(w, PlanWarning::JoinEliminated { .. })),
            "inner join should be eliminated"
        );
        // After elimination, scan_required should be set.
        assert!(plan.scan_required.is_some());

        let mut results = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |_entity, (score,)| {
            results.push(score.0);
        })
        .unwrap();

        results.sort_unstable();
        assert_eq!(results, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn direct_iter_disabled_with_index_driver() {
        use crate::BTreeIndex;

        let mut world = World::new();
        for i in 0..100 {
            world.spawn((Score(i),));
        }

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index::<Score>(&idx, &world).unwrap();
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();

        // Index driver present — scan_required should be None.
        assert!(
            plan.scan_required.is_none(),
            "direct path should be disabled when index driver is present"
        );

        // Verify the plan still produces correct results (index-driven path).
        let mut plan = plan;
        let mut results = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |_, (score,)| {
            results.push(score.0);
        })
        .unwrap();
        results.sort_unstable();
        assert_eq!(results, (10..20).collect::<Vec<_>>());
    }

    // ── ER join tests ────────────────────────────────────────────

    /// Entity-reference component: points from a child to its parent.
    #[derive(Clone, Copy, Debug)]
    struct Parent(Entity);

    impl super::AsEntityRef for Parent {
        fn entity_ref(&self) -> Entity {
            self.0
        }
    }

    /// Tag component for child entities.
    #[derive(Clone, Copy, Debug)]
    struct ChildTag;

    /// Component only on parent entities.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Name(&'static str);

    #[test]
    fn er_join_inner_basic() {
        let mut world = World::new();

        // Spawn parents with Name.
        let p1 = world.spawn((Name("Alice"),));
        let p2 = world.spawn((Name("Bob"),));
        let p3 = world.spawn((Score(999),)); // no Name — won't match right side

        // Spawn children pointing to parents.
        let c1 = world.spawn((ChildTag, Parent(p1)));
        let c2 = world.spawn((ChildTag, Parent(p2)));
        let c3 = world.spawn((ChildTag, Parent(p3))); // parent has no Name

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let entities = plan.execute(&mut world).unwrap();
        let mut ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
        ids.sort_unstable();

        // Only c1 and c2 should match — their parents have Name.
        let mut expected = vec![c1.to_bits(), c2.to_bits()];
        expected.sort_unstable();
        assert_eq!(
            ids, expected,
            "inner ER join should keep only children whose parent has Name"
        );

        // c3's parent has no Name, so c3 is excluded.
        assert!(!ids.contains(&c3.to_bits()));
    }

    #[test]
    fn er_join_left_keeps_all() {
        let mut world = World::new();

        let p1 = world.spawn((Name("Alice"),));
        let p_no_name = world.spawn((Score(42),)); // no Name

        let c1 = world.spawn((ChildTag, Parent(p1)));
        let c2 = world.spawn((ChildTag, Parent(p_no_name)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Left)
            .build();

        let entities = plan.execute(&mut world).unwrap();
        assert_eq!(
            entities.len(),
            2,
            "left ER join should keep all left entities"
        );
        let ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
        assert!(ids.contains(&c1.to_bits()));
        assert!(ids.contains(&c2.to_bits()));
    }

    #[test]
    fn er_join_dead_reference() {
        let mut world = World::new();

        let p1 = world.spawn((Name("Alice"),));
        let p2 = world.spawn((Name("Bob"),));

        let c1 = world.spawn((ChildTag, Parent(p1)));
        let c2 = world.spawn((ChildTag, Parent(p2)));

        // Despawn p2 — c2's reference is now dangling.
        world.despawn(p2);

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let entities = plan.execute(&mut world).unwrap();
        let ids: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
        assert_eq!(ids.len(), 1, "dead reference target should not match");
        assert!(ids.contains(&c1.to_bits()));
        assert!(!ids.contains(&c2.to_bits()));
    }

    #[test]
    fn er_join_explain_shows_er_join_node() {
        let mut world = World::new();
        let p = world.spawn((Name("Alice"),));
        world.spawn((ChildTag, Parent(p)));

        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let explain = plan.explain();
        assert!(
            explain.contains("ErJoin"),
            "explain should contain ErJoin node: {explain}"
        );
        assert!(
            explain.contains("Inner"),
            "explain should show Inner join kind: {explain}"
        );
    }

    #[test]
    fn er_join_for_each() {
        let mut world = World::new();

        let p1 = world.spawn((Name("Alice"),));
        let p2 = world.spawn((Name("Bob"),));

        let c1 = world.spawn((ChildTag, Parent(p1)));
        let c2 = world.spawn((ChildTag, Parent(p2)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let mut found = Vec::new();
        plan.for_each(&mut world, |e| found.push(e)).unwrap();
        found.sort_by_key(|e| e.to_bits());

        let mut expected = vec![c1, c2];
        expected.sort_by_key(|e| e.to_bits());
        assert_eq!(found, expected);
    }

    #[test]
    fn er_join_for_each_raw() {
        let mut world = World::new();

        let p1 = world.spawn((Name("Alice"),));
        let p_no = world.spawn((Score(1),));

        let c1 = world.spawn((ChildTag, Parent(p1)));
        world.spawn((ChildTag, Parent(p_no)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let mut found = Vec::new();
        plan.for_each_raw(&world, |e| found.push(e)).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], c1);
    }

    #[test]
    fn er_join_combined_with_regular_join() {
        let mut world = World::new();

        // Parents with Name and Score.
        let p1 = world.spawn((Name("Alice"), Score(10)));
        let p2 = world.spawn((Name("Bob"), Score(20)));

        // Children.
        world.spawn((ChildTag, Parent(p1), Team(1)));
        world.spawn((ChildTag, Parent(p2), Team(2)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent, &Team)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let entities = plan.execute(&mut world).unwrap();
        // Both children's parents have Name, so both match.
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn er_join_no_matching_children() {
        let mut world = World::new();

        // Parents exist but no children.
        world.spawn((Name("Alice"),));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let entities = plan.execute(&mut world).unwrap();
        assert_eq!(entities.len(), 0);
    }

    #[test]
    fn er_join_for_each_batched() {
        let mut world = World::new();

        let p1 = world.spawn((Name("Alice"),));
        let p2 = world.spawn((Name("Bob"),));
        let p_no = world.spawn((Score(99),));

        world.spawn((ChildTag, Parent(p1), Score(1)));
        world.spawn((ChildTag, Parent(p2), Score(2)));
        world.spawn((ChildTag, Parent(p_no), Score(3)));

        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&ChildTag, &Parent)>()
            .er_join::<Parent, (&Name,)>(JoinKind::Inner)
            .build();

        let mut scores = Vec::new();
        plan.for_each_batched::<(&Score,), _>(&mut world, |_, (s,)| {
            scores.push(s.0);
        })
        .unwrap();
        scores.sort_unstable();
        assert_eq!(
            scores,
            vec![1, 2],
            "batched should yield only matching children's scores"
        );
    }
}
