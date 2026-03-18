use std::marker::PhantomData;
use std::sync::Arc;

use crate::component::Component;
use crate::index::SpatialIndex;
use crate::world::World;

use super::{Indexed, PlannerError, QueryPlanner, ScanBuilder};

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

    /// Register a spatial index for cost discovery and execution.
    ///
    /// Delegates to [`QueryPlanner::add_spatial_index`].
    /// No compile-time index enforcement — spatial indexes are orthogonal to
    /// table schemas.
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
