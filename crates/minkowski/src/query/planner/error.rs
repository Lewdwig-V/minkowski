use std::fmt;

use crate::storage::archetype::ArchetypeId;
use crate::transaction::WorldMismatch;

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

/// Error returned by plan execution methods (`execute_collect`, `execute_collect_raw`,
/// `execute_stream`, `execute_stream_raw`, `aggregate`, `aggregate_raw`).
#[derive(Clone, Debug)]
pub enum PlanExecError {
    /// Plan was built from a different World.
    WorldMismatch(WorldMismatch),
    /// `execute_stream` / `execute_stream_raw` called on a plan that contains joins.
    /// Use `execute_collect()` instead, which collects entities into a scratch buffer.
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
                "execute_stream/execute_stream_raw do not support join plans — use execute_collect() instead"
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
    /// The query type has no `Changed<T>` filter. Without change detection,
    /// the subscription matches all entities on every execution — use a
    /// regular `scan` instead, or add `Changed<T>` to the query type.
    NoChangedFilter,
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
            SubscriptionError::NoChangedFilter => {
                write!(
                    f,
                    "subscription query has no Changed<T> filter — without change detection the \
                     subscription matches all entities every time. Add Changed<T> to the query \
                     type or use a regular scan instead"
                )
            }
        }
    }
}

impl std::error::Error for SubscriptionError {}
