use std::any::TypeId;
use std::fmt;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use crate::component::Component;
use crate::entity::Entity;
use crate::index::SpatialExpr;
use crate::storage::archetype::Archetype;
use crate::world::World;

use super::error::PlannerError;
use super::{ColumnFilterFn, FilterFn, IndexLookupFn};

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
pub(super) type PredicateLookupFn = Arc<dyn Fn(&dyn std::any::Any) -> Arc<[Entity]> + Send + Sync>;

/// Type-erased index metadata for the planner.
pub(super) struct IndexDescriptor {
    pub(super) component_name: &'static str,
    pub(super) kind: IndexKind,
    /// Type-erased function that returns all entities tracked by this index.
    /// Captured at registration time when the concrete index type is available.
    pub(super) all_entities_fn: Option<IndexLookupFn>,
    /// Predicate-specific equality lookup. Takes `&dyn Any` (downcast to `T`),
    /// returns only entities matching the exact value. O(log n) for BTree, O(1) for Hash.
    /// Bound into an `IndexDriver` lookup closure at Phase 3 plan-build time.
    pub(super) eq_lookup_fn: Option<PredicateLookupFn>,
    /// Predicate-specific range lookup. Takes `&dyn Any` (downcast to `(Bound<T>, Bound<T>)`),
    /// returns only entities within the range. O(log n + k) for BTree, not available for Hash.
    /// Bound into an `IndexDriver` lookup closure at Phase 3 plan-build time.
    pub(super) range_lookup_fn: Option<PredicateLookupFn>,
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
// IndexDriver pre-bound at Phase 3 and compiled into the execute_stream closure.

// ── Predicates ───────────────────────────────────────────────────────

/// Clamp selectivity to [0.0, 1.0], treating NaN as 1.0 (worst-case).
///
/// `f64::clamp` preserves NaN, so we must check explicitly.
pub(super) fn sanitize_selectivity(s: f64) -> f64 {
    if s.is_nan() { 1.0 } else { s.clamp(0.0, 1.0) }
}

/// A predicate that can be pushed down into an index lookup or applied as
/// a post-fetch filter.
pub struct Predicate {
    pub(super) component_type: TypeId,
    pub(super) component_name: &'static str,
    pub(super) kind: PredicateKind,
    pub(super) selectivity: f64,
    /// Type-erased filter closure for execution. Captured at predicate
    /// construction when the concrete value is available.
    pub(super) filter_fn: Option<FilterFn>,
    /// Column-aware filter closure. When present, the plan can evaluate the
    /// filter by iterating contiguous column slices per archetype instead of
    /// per-entity `world.get()` dispatch. See [`Predicate::custom_column`].
    pub(super) column_filter_fn: Option<ColumnFilterFn>,
    /// Type-erased predicate value for index lookups. Eq stores `Arc<T>`,
    /// Range stores `Arc<(Bound<T>, Bound<T>)>`. Downcast by the index's
    /// `eq_lookup_fn` / `range_lookup_fn` at plan-build time.
    /// Bound into the `IndexDriver` lookup closure at Phase 3 plan-build time.
    pub(super) lookup_value: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

#[derive(Debug)]
pub(super) enum PredicateKind {
    Eq,
    Range,
    Custom(Box<str>),       // description only — always post-filter
    CustomColumn(Box<str>), // column-aware custom — always post-filter
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
            column_filter_fn: None,
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
            column_filter_fn: None,
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
            column_filter_fn: None,
            lookup_value: None,
        }
    }

    /// Column-aware custom predicate with a per-element filter closure.
    ///
    /// Like [`custom`](Self::custom), always applied as a post-fetch filter
    /// (cannot be pushed into an index). However, instead of receiving
    /// `(&World, Entity)` and paying per-entity `world.get::<T>()` overhead,
    /// the closure receives `&T` directly from the contiguous column slice.
    ///
    /// During scan execution, the planner iterates archetype columns and
    /// applies the predicate per element with sequential memory access — one
    /// `ComponentId` resolution per archetype instead of per entity. This
    /// eliminates the dominant cost of `Predicate::custom` for scan plans.
    ///
    /// NaN selectivity is normalized to 1.0 (worst-case, full scan).
    ///
    /// # Example
    ///
    /// ```
    /// # use minkowski::planner::Predicate;
    /// # #[derive(Clone, Copy)] struct Score(u32);
    /// let pred = Predicate::custom_column::<Score>("score < 5000", 0.5, |s| s.0 < 5000);
    /// ```
    pub fn custom_column<T: Component>(
        description: &str,
        selectivity: f64,
        filter: impl Fn(&T) -> bool + Send + Sync + 'static,
    ) -> Self {
        let filter = Arc::new(filter);

        // Column-aware filter: iterates the typed column slice per archetype.
        let col_filter = {
            let filter = Arc::clone(&filter);
            Arc::new(
                move |world: &World, arch: &Archetype, entities: &[Entity], mask: &mut [bool]| {
                    debug_assert_eq!(mask.len(), arch.len());
                    debug_assert_eq!(mask.len(), entities.len());

                    let Some(comp_id) = world.components.id::<T>() else {
                        // Component not registered — no entity can pass.
                        for m in &mut *mask {
                            *m = false;
                        }
                        return;
                    };
                    if let Some(col_idx) = arch.column_index(comp_id) {
                        // Dense path: iterate contiguous column slice.
                        if arch.is_empty() {
                            return;
                        }
                        debug_assert_eq!(
                            arch.columns[col_idx].item_layout,
                            std::alloc::Layout::new::<T>(),
                            "BlobVec layout mismatch in custom_column filter for {}",
                            std::any::type_name::<T>()
                        );
                        let ptr = unsafe { arch.columns[col_idx].get_ptr(0) as *const T };
                        let slice = unsafe { std::slice::from_raw_parts(ptr, arch.len()) };
                        for (m, val) in mask.iter_mut().zip(slice) {
                            if *m && !filter(val) {
                                *m = false;
                            }
                        }
                    } else if world.components.is_sparse(comp_id) {
                        // Sparse fallback: per-entity world.get().
                        for (m, &entity) in mask.iter_mut().zip(entities) {
                            if *m {
                                let pass = world
                                    .sparse
                                    .get::<T>(comp_id, entity)
                                    .is_some_and(|v| filter(v));
                                if !pass {
                                    *m = false;
                                }
                            }
                        }
                    } else {
                        // Archetype doesn't have this component and it's not
                        // sparse — no entity can pass.
                        for m in &mut *mask {
                            *m = false;
                        }
                    }
                },
            ) as ColumnFilterFn
        };

        // Fallback per-entity filter for index-driven paths.
        let per_entity_filter: FilterFn = {
            let filter = Arc::clone(&filter);
            Arc::new(move |world: &World, entity: Entity| {
                world.get::<T>(entity).is_some_and(|v| filter(v))
            })
        };

        Predicate {
            component_type: TypeId::of::<T>(),
            component_name: std::any::type_name::<T>(),
            kind: PredicateKind::CustomColumn(description.into()),
            selectivity: sanitize_selectivity(selectivity),
            filter_fn: Some(per_entity_filter),
            column_filter_fn: Some(col_filter),
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
            column_filter_fn: None,
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
            column_filter_fn: None,
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

    pub(super) fn can_use_btree(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq | PredicateKind::Range)
    }

    pub(super) fn can_use_hash(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq)
    }

    pub(super) fn can_use_spatial(&self) -> bool {
        matches!(self.kind, PredicateKind::Spatial(_))
    }

    /// Whether this predicate can be lowered to branchless SIMD comparison.
    pub(super) fn is_branchless_eligible(&self) -> bool {
        matches!(self.kind, PredicateKind::Eq | PredicateKind::Range)
    }
}

impl fmt::Debug for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            PredicateKind::Eq => write!(f, "Eq({})", self.component_name),
            PredicateKind::Range => write!(f, "Range({})", self.component_name),
            PredicateKind::Custom(desc) => write!(f, "Custom({}: {})", self.component_name, desc),
            PredicateKind::CustomColumn(desc) => {
                write!(f, "CustomColumn({}: {})", self.component_name, desc)
            }
            PredicateKind::Spatial(sp) => write!(f, "Spatial({}: {})", self.component_name, sp),
        }
    }
}
