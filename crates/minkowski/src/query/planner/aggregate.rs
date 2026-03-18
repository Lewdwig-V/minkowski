use std::fmt;
use std::sync::Arc;

use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::storage::archetype::Archetype;
use crate::world::World;

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
pub(super) type ValueExtractor = Arc<dyn Fn(&World, Entity) -> Option<f64> + Send + Sync>;

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
        // Hoist the op-match outside the inner loop so LLVM sees a tight
        // branchless loop per variant and can auto-vectorize.
        match accum.op {
            AggregateOp::Count => {
                accum.count += count as u64;
            }
            AggregateOp::Sum | AggregateOp::Avg => {
                let mut local_sum = 0.0f64;
                for item in slice {
                    local_sum += (self.extract)(item);
                }
                accum.count += count as u64;
                accum.sum += local_sum;
            }
            AggregateOp::Min => {
                let mut local_min = accum.min;
                for item in slice {
                    let v = (self.extract)(item);
                    if v.is_nan() || v < local_min {
                        local_min = v;
                    }
                }
                accum.count += count as u64;
                accum.min = local_min;
            }
            AggregateOp::Max => {
                let mut local_max = accum.max;
                for item in slice {
                    let v = (self.extract)(item);
                    if v.is_nan() || v > local_max {
                        local_max = v;
                    }
                }
                accum.count += count as u64;
                accum.max = local_max;
            }
        }
    }

    fn process_rows(&mut self, rows: &[usize], accum: &mut AggregateAccum) {
        debug_assert!(
            !self.col_ptr.is_null(),
            "process_rows called before successful bind_archetype"
        );
        // Same op-hoisting strategy as process_all for index-gather paths.
        match accum.op {
            AggregateOp::Count => {
                accum.count += rows.len() as u64;
            }
            AggregateOp::Sum | AggregateOp::Avg => {
                let mut local_sum = 0.0f64;
                for &row in rows {
                    // SAFETY: row comes from a validated EntityLocation (via
                    // world.validate_entity), guaranteeing it is a valid index
                    // within the bound archetype.
                    let item = unsafe { &*self.col_ptr.add(row) };
                    local_sum += (self.extract)(item);
                }
                accum.count += rows.len() as u64;
                accum.sum += local_sum;
            }
            AggregateOp::Min => {
                let mut local_min = accum.min;
                for &row in rows {
                    let item = unsafe { &*self.col_ptr.add(row) };
                    let v = (self.extract)(item);
                    if v.is_nan() || v < local_min {
                        local_min = v;
                    }
                }
                accum.count += rows.len() as u64;
                accum.min = local_min;
            }
            AggregateOp::Max => {
                let mut local_max = accum.max;
                for &row in rows {
                    let item = unsafe { &*self.col_ptr.add(row) };
                    let v = (self.extract)(item);
                    if v.is_nan() || v > local_max {
                        local_max = v;
                    }
                }
                accum.count += rows.len() as u64;
                accum.max = local_max;
            }
        }
    }
}

/// Factory that produces fresh `Box<dyn BatchExtractor>` instances.
/// Created at `ScanBuilder::build()` time when `ComponentId` is resolved.
/// Called once per `aggregate` invocation.
pub(super) type BatchFactory = Box<dyn Fn() -> Box<dyn BatchExtractor> + Send + Sync>;

/// Builder that captures the typed closure and produces a `BatchFactory`
/// once the `ComponentId` is known. Stored on `AggregateExpr` between
/// construction and `build()`.
pub(super) type BatchFactoryBuilder =
    Box<dyn FnOnce(&ComponentRegistry) -> Option<BatchFactory> + Send + Sync>;

/// A single aggregate expression: an operation applied to values extracted
/// from matched entities.
///
/// # Example
///
/// ```rust,ignore
/// use minkowski::{AggregateExpr, AggregateOp};
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
    pub(super) op: AggregateOp,
    /// Human-readable label for `explain()` output.
    ///
    /// Label format convention: `"COUNT(*)"`, `"SUM(name)"`, `"MIN(name)"`,
    /// `"MAX(name)"`, `"AVG(name)"` where `name` is the user-supplied label.
    pub(super) label: String,
    /// Extracts a `f64` value from an entity. `None` for `Count`.
    /// Kept as fallback for join plans where batch extraction isn't possible.
    pub(super) extractor: Option<ValueExtractor>,
    /// Deferred factory builder: captures the typed closure and defers
    /// `ComponentId` resolution (via the generic type parameter `T`) to
    /// `ScanBuilder::build()` time. `None` for `Count` (no component access).
    pub(super) batch_factory_builder: Option<BatchFactoryBuilder>,
    /// Finalized batch factory, set by `ScanBuilder::build()`. Each call
    /// produces a fresh `Box<dyn BatchExtractor>` with its own column pointer
    /// binding.
    pub(super) batch_factory: Option<BatchFactory>,
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
pub(super) fn make_extractor<T: Component>(
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
pub(super) struct AggregateAccum {
    pub(super) op: AggregateOp,
    pub(super) label: String,
    pub(super) count: u64,
    pub(super) sum: f64,
    pub(super) min: f64,
    pub(super) max: f64,
}

impl AggregateAccum {
    pub(super) fn new(op: AggregateOp, label: String) -> Self {
        Self {
            op,
            label,
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Reset accumulator to initial state, reusing the existing label allocation.
    pub(super) fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    /// Feed a value into the accumulator.
    ///
    /// `count` tracks values fed (not entities matched by the scan). For
    /// `Avg`, this is the denominator. For `Count`, use `feed_count` instead.
    pub(super) fn feed(&mut self, value: f64) {
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

    pub(super) fn feed_count(&mut self) {
        self.count += 1;
    }

    pub(super) fn finish(&self) -> f64 {
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

/// RAII guard that restores a `Vec<AggregateAccum>` to its owner on drop.
/// Used when accumulators must be temporarily moved out of `QueryPlanResult`
/// to satisfy borrow-splitting requirements (e.g., the `compiled_for_each`
/// callback borrows both the closure and the accums). If the callback panics,
/// Drop restores the field so the plan remains usable after catch_unwind.
pub(super) struct AccumGuard<'a> {
    slot: &'a mut Vec<AggregateAccum>,
    accums: Option<Vec<AggregateAccum>>,
}

impl<'a> AccumGuard<'a> {
    pub(super) fn take(slot: &'a mut Vec<AggregateAccum>) -> Self {
        let accums = Some(std::mem::take(slot));
        Self { slot, accums }
    }

    /// Borrow the accumulators mutably.
    pub(super) fn accums_mut(&mut self) -> &mut Vec<AggregateAccum> {
        self.accums.as_mut().unwrap()
    }
}

impl Drop for AccumGuard<'_> {
    fn drop(&mut self) {
        if let Some(accums) = self.accums.take() {
            *self.slot = accums;
        }
    }
}

/// The result of executing an aggregate plan.
///
/// Contains one `f64` result per aggregate expression, in the same order
/// they were added via [`ScanBuilder::aggregate`].
#[derive(Clone, Debug)]
pub struct AggregateResult {
    pub(super) values: Vec<(String, f64)>,
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
