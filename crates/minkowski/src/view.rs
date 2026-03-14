//! Materialized query views — cached, debounced query snapshots for
//! real-time computed data.
//!
//! A [`MaterializedView`] wraps a [`QueryPlanResult`] (typically from a
//! [`SubscriptionBuilder`]) and caches the matching entity list. On each
//! [`refresh`](MaterializedView::refresh) call it re-executes the plan, but
//! only if the underlying data has changed (via `Changed<T>` in the plan's
//! query type) **and** the configurable debounce threshold has been met.
//!
//! This is the read-side complement to subscription queries: where
//! `SubscriptionBuilder` guarantees that every predicate is index-backed,
//! `MaterializedView` guarantees that the result is cached and debounced.
//!
//! # Debouncing
//!
//! Two debounce modes are supported:
//!
//! - **Tick-based** ([`DebouncePolicy::EveryNTicks`]): refresh at most once
//!   per N calls to `refresh`. Useful when `refresh` is called once per frame
//!   and you want to skip expensive re-materialization on most frames.
//! - **Immediate** ([`DebouncePolicy::Immediate`]): refresh on every call if
//!   the underlying data has changed. This is the default.
//!
//! The plan's own `Changed<T>` filter provides the first layer of debouncing
//! (archetype-granular). The debounce policy adds a second layer on top.
//!
//! # Usage
//!
//! ```rust,ignore
//! // Build an index-backed subscription plan.
//! let mut planner = QueryPlanner::new(&world);
//! planner.add_btree_index(&score_index, &world).unwrap();
//! let plan = planner
//!     .subscribe::<(Changed<Score>, &Score)>()
//!     .where_eq(Indexed::btree(&score_index), Predicate::eq(Score(42)))
//!     .build()
//!     .unwrap();
//!
//! // Wrap in a materialized view with tick-based debouncing.
//! let mut view = MaterializedView::new(plan)
//!     .with_debounce(DebouncePolicy::EveryNTicks(10));
//!
//! // Per-frame: refresh and read.
//! view.refresh(&mut world).unwrap();
//! for &entity in view.entities() {
//!     let score = world.get::<Score>(entity).unwrap();
//!     // ... use the cached, debounced result
//! }
//! ```
//!
//! # Design
//!
//! - **External to World** — same composition pattern as `SpatialIndex`,
//!   `BTreeIndex`, and `ReducerRegistry`.
//! - **Owns its plan** — the view takes ownership of the `QueryPlanResult`
//!   and manages tick advancement.
//! - **Zero-copy reads** — `entities()` returns `&[Entity]`, borrowed from
//!   the plan's internal scratch buffer or the view's own cached copy.

use crate::entity::Entity;
use crate::planner::QueryPlanResult;
use crate::transaction::WorldMismatch;
use crate::world::World;

/// Controls how often a [`MaterializedView`] re-materializes its cached result.
///
/// The plan's own `Changed<T>` filter provides archetype-granular change
/// detection. The debounce policy adds a second layer that limits how often
/// the (potentially expensive) plan execution runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebouncePolicy {
    /// Refresh on every call to [`MaterializedView::refresh`] if the
    /// underlying data has changed. This is the default.
    #[default]
    Immediate,

    /// Refresh at most once per `n` calls to `refresh`. The first call
    /// always refreshes. Subsequent calls within the window return the
    /// cached result.
    ///
    /// A value of 0 is treated as 1 (immediate).
    EveryNTicks(u64),
}

/// A cached, debounced materialized view over a subscription query plan.
///
/// Wraps a [`QueryPlanResult`] and caches the matching entity list.
/// On each [`refresh`](Self::refresh) call, it re-executes the plan only
/// if the debounce threshold has been met. The plan's own `Changed<T>`
/// filter provides the first layer of change detection; the debounce
/// policy adds a second layer.
///
/// # Example
///
/// ```rust,ignore
/// let mut view = MaterializedView::new(plan);
/// view.refresh(&mut world).unwrap();
/// println!("matched {} entities", view.len());
/// ```
pub struct MaterializedView {
    plan: QueryPlanResult,
    /// Cached entity snapshot — populated on the first refresh, updated
    /// when the plan returns new results.
    cached: Vec<Entity>,
    /// Whether the view has been refreshed at least once.
    populated: bool,
    /// Debounce policy.
    policy: DebouncePolicy,
    /// Number of calls to `refresh` since the last actual re-materialization.
    ticks_since_refresh: u64,
    /// Total number of times the view has been refreshed (re-materialized).
    refresh_count: u64,
}

impl MaterializedView {
    /// Create a new materialized view wrapping the given plan.
    ///
    /// The view starts empty — call [`refresh`](Self::refresh) to populate it.
    /// The default debounce policy is [`DebouncePolicy::Immediate`].
    pub fn new(plan: QueryPlanResult) -> Self {
        Self {
            plan,
            cached: Vec::new(),
            populated: false,
            policy: DebouncePolicy::Immediate,
            ticks_since_refresh: 0,
            refresh_count: 0,
        }
    }

    /// Set the debounce policy. Returns `self` for builder-style chaining.
    pub fn with_debounce(mut self, policy: DebouncePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Refresh the materialized view if the debounce policy allows it.
    ///
    /// When the policy is [`DebouncePolicy::Immediate`], the plan is
    /// re-executed on every call (but `Changed<T>` in the plan may still
    /// short-circuit if no columns were mutated).
    ///
    /// When the policy is [`DebouncePolicy::EveryNTicks`], the plan is
    /// re-executed only when the tick counter reaches the threshold.
    ///
    /// Returns `true` if the view was actually re-materialized, `false` if
    /// the debounce policy suppressed the refresh.
    ///
    /// # Errors
    ///
    /// Returns [`WorldMismatch`] if the world is not the same one the plan
    /// was built from.
    pub fn refresh(&mut self, world: &mut World) -> Result<bool, WorldMismatch> {
        self.ticks_since_refresh += 1;

        let should_refresh = match self.policy {
            DebouncePolicy::Immediate => true,
            DebouncePolicy::EveryNTicks(n) => {
                let n = n.max(1);
                !self.populated || self.ticks_since_refresh >= n
            }
        };

        if !should_refresh {
            return Ok(false);
        }

        let entities = self.plan.execute(world)?;
        self.cached.clear();
        self.cached.extend_from_slice(entities);
        self.populated = true;
        self.ticks_since_refresh = 0;
        self.refresh_count += 1;

        Ok(true)
    }

    /// The cached entity snapshot from the last refresh.
    ///
    /// Returns an empty slice if [`refresh`](Self::refresh) has never been
    /// called.
    #[inline]
    pub fn entities(&self) -> &[Entity] {
        &self.cached
    }

    /// Number of entities in the cached snapshot.
    #[inline]
    pub fn len(&self) -> usize {
        self.cached.len()
    }

    /// Returns `true` if the cached snapshot is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cached.is_empty()
    }

    /// Returns `true` if the view has been refreshed at least once.
    #[inline]
    pub fn is_populated(&self) -> bool {
        self.populated
    }

    /// Total number of times the view has been re-materialized.
    #[inline]
    pub fn refresh_count(&self) -> u64 {
        self.refresh_count
    }

    /// The current debounce policy.
    #[inline]
    pub fn policy(&self) -> DebouncePolicy {
        self.policy
    }

    /// Change the debounce policy at runtime. Resets the tick counter.
    pub fn set_policy(&mut self, policy: DebouncePolicy) {
        self.policy = policy;
        self.ticks_since_refresh = 0;
    }

    /// Borrow the underlying plan for introspection (e.g. `explain()`).
    #[inline]
    pub fn plan(&self) -> &QueryPlanResult {
        &self.plan
    }

    /// Force a refresh on the next call to [`refresh`](Self::refresh),
    /// regardless of the debounce policy. Does not execute the plan
    /// immediately.
    pub fn invalidate(&mut self) {
        self.populated = false;
    }
}

impl std::fmt::Debug for MaterializedView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterializedView")
            .field("cached_len", &self.cached.len())
            .field("populated", &self.populated)
            .field("policy", &self.policy)
            .field("ticks_since_refresh", &self.ticks_since_refresh)
            .field("refresh_count", &self.refresh_count)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::{Predicate, QueryPlanner};
    use crate::query::fetch::Changed;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Score(u32);

    // ── Helper: build a simple scan plan ────────────────────────────────

    fn score_world_and_plan(n: u32) -> (World, QueryPlanResult) {
        let mut world = World::new();
        for i in 0..n {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        (world, plan)
    }

    // ── Construction & defaults ─────────────────────────────────────────

    #[test]
    fn new_view_is_empty_and_unpopulated() {
        let (_world, plan) = score_world_and_plan(10);
        let view = MaterializedView::new(plan);
        assert!(!view.is_populated());
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
        assert_eq!(view.entities(), &[]);
        assert_eq!(view.refresh_count(), 0);
        assert_eq!(view.policy(), DebouncePolicy::Immediate);
    }

    // ── Immediate refresh ───────────────────────────────────────────────

    #[test]
    fn immediate_refresh_populates() {
        let (mut world, plan) = score_world_and_plan(10);
        let mut view = MaterializedView::new(plan);
        let refreshed = view.refresh(&mut world).unwrap();
        assert!(refreshed);
        assert!(view.is_populated());
        assert_eq!(view.len(), 10);
        assert_eq!(view.refresh_count(), 1);
    }

    #[test]
    fn immediate_refresh_updates_on_change() {
        let (mut world, plan) = score_world_and_plan(5);
        let mut view = MaterializedView::new(plan);
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 5);

        // Spawn more entities.
        for i in 5..8 {
            world.spawn((Score(i),));
        }
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 8);
        assert_eq!(view.refresh_count(), 2);
    }

    // ── Debounced refresh ───────────────────────────────────────────────

    #[test]
    fn debounced_first_call_always_refreshes() {
        let (mut world, plan) = score_world_and_plan(10);
        let mut view = MaterializedView::new(plan).with_debounce(DebouncePolicy::EveryNTicks(5));
        let refreshed = view.refresh(&mut world).unwrap();
        assert!(refreshed);
        assert_eq!(view.len(), 10);
    }

    #[test]
    fn debounced_suppresses_within_window() {
        let (mut world, plan) = score_world_and_plan(10);
        let mut view = MaterializedView::new(plan).with_debounce(DebouncePolicy::EveryNTicks(3));
        // First call refreshes.
        view.refresh(&mut world).unwrap();
        assert_eq!(view.refresh_count(), 1);

        // Next two calls are suppressed.
        assert!(!view.refresh(&mut world).unwrap());
        assert!(!view.refresh(&mut world).unwrap());
        assert_eq!(view.refresh_count(), 1);
        // The cached data is still available.
        assert_eq!(view.len(), 10);

        // Third call triggers refresh.
        assert!(view.refresh(&mut world).unwrap());
        assert_eq!(view.refresh_count(), 2);
    }

    #[test]
    fn debounced_zero_treated_as_immediate() {
        let (mut world, plan) = score_world_and_plan(5);
        let mut view = MaterializedView::new(plan).with_debounce(DebouncePolicy::EveryNTicks(0));
        view.refresh(&mut world).unwrap();
        assert!(view.refresh(&mut world).unwrap());
        assert_eq!(view.refresh_count(), 2);
    }

    // ── Changed<T> integration ──────────────────────────────────────────

    #[test]
    fn changed_filter_skips_unchanged_archetypes() {
        let mut world = World::new();
        let entities: Vec<_> = (0u32..5).map(|i| world.spawn((Score(i),))).collect();

        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(Changed<Score>, &Score)>().build();
        let mut view = MaterializedView::new(plan);

        // First refresh sees everything (all new).
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 5);

        // No mutations → Changed<Score> filters everything.
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 0);

        // Mutate one entity → its archetype column becomes visible.
        let _ = world.get_mut::<Score>(entities[0]);
        view.refresh(&mut world).unwrap();
        // All entities in the same archetype are returned (archetype-granular).
        assert_eq!(view.len(), 5);
    }

    // ── Invalidate ──────────────────────────────────────────────────────

    #[test]
    fn invalidate_forces_next_refresh() {
        let (mut world, plan) = score_world_and_plan(5);
        let mut view = MaterializedView::new(plan).with_debounce(DebouncePolicy::EveryNTicks(100));

        view.refresh(&mut world).unwrap();
        assert_eq!(view.refresh_count(), 1);

        // Normally suppressed for 99 more calls.
        assert!(!view.refresh(&mut world).unwrap());

        // Invalidate forces the next refresh.
        view.invalidate();
        assert!(view.refresh(&mut world).unwrap());
        assert_eq!(view.refresh_count(), 2);
    }

    // ── set_policy ──────────────────────────────────────────────────────

    #[test]
    fn set_policy_resets_counter() {
        let (mut world, plan) = score_world_and_plan(5);
        let mut view = MaterializedView::new(plan).with_debounce(DebouncePolicy::EveryNTicks(2));
        view.refresh(&mut world).unwrap();

        // One tick elapsed — next would be suppressed.
        assert!(!view.refresh(&mut world).unwrap());

        // Switch to immediate — resets counter, next call refreshes.
        view.set_policy(DebouncePolicy::Immediate);
        assert!(view.refresh(&mut world).unwrap());
    }

    // ── WorldMismatch ───────────────────────────────────────────────────

    #[test]
    fn wrong_world_returns_error() {
        let (_world1, plan) = score_world_and_plan(5);
        let mut world2 = World::new();
        let mut view = MaterializedView::new(plan);
        assert!(view.refresh(&mut world2).is_err());
    }

    // ── Filtered plan with predicates ───────────────────────────────────

    #[test]
    fn filtered_plan_materializes_subset() {
        let mut world = World::new();
        for i in 0u32..100 {
            world.spawn((Score(i),));
        }
        let planner = QueryPlanner::new(&world);
        let plan = planner
            .scan::<(&Score,)>()
            .filter(Predicate::range::<Score, _>(Score(10)..Score(20)))
            .build();
        let mut view = MaterializedView::new(plan);
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 10);

        // Verify entities match.
        for &entity in view.entities() {
            let score = world.get::<Score>(entity).unwrap().0;
            assert!((10..20).contains(&score));
        }
    }

    // ── Despawned entities ──────────────────────────────────────────────

    #[test]
    fn despawned_entities_removed_on_refresh() {
        let mut world = World::new();
        let entities: Vec<_> = (0u32..5).map(|i| world.spawn((Score(i),))).collect();

        let planner = QueryPlanner::new(&world);
        let plan = planner.scan::<(&Score,)>().build();
        let mut view = MaterializedView::new(plan);
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 5);

        // Despawn two entities.
        world.despawn(entities[0]);
        world.despawn(entities[1]);
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 3);
    }

    // ── Debug output ────────────────────────────────────────────────────

    #[test]
    fn debug_format_includes_key_fields() {
        let (_world, plan) = score_world_and_plan(5);
        let view = MaterializedView::new(plan);
        let dbg = format!("{view:?}");
        assert!(dbg.contains("MaterializedView"));
        assert!(dbg.contains("cached_len: 0"));
        assert!(dbg.contains("populated: false"));
    }

    // ── Index-backed subscription view ──────────────────────────────────

    #[test]
    fn subscription_backed_view() {
        use crate::index::{BTreeIndex, SpatialIndex};
        use crate::planner::Indexed;
        use std::sync::Arc;

        let mut world = World::new();
        for i in 0u32..100 {
            world.spawn((Score(i),));
        }

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        let idx = Arc::new(idx);

        let mut planner = QueryPlanner::new(&world);
        planner.add_btree_index(&idx, &world).unwrap();
        let witness = Indexed::btree(&*idx);

        let plan = planner
            .subscribe::<(Changed<Score>, &Score)>()
            .where_eq(witness, Predicate::eq(Score(42)).with_selectivity(0.01))
            .build()
            .unwrap();

        let mut view = MaterializedView::new(plan);

        // First refresh: find Score(42).
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 1);
        assert_eq!(*world.get::<Score>(view.entities()[0]).unwrap(), Score(42));

        // No changes → empty (Changed<Score> filters).
        view.refresh(&mut world).unwrap();
        assert_eq!(view.len(), 0);
    }
}
