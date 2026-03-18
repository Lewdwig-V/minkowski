use crate::component::Component;
use crate::entity::Entity;

/// Trait for component types that act as entity references (foreign keys).
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
///     .er_join::<Parent, (&Pos, &Name)>(JoinKind::Inner)?
///     .build();
/// # Ok::<(), minkowski::PlannerError>(())
/// ```
pub trait AsEntityRef: Component {
    /// Extract the referenced entity from this component value.
    fn entity_ref(&self) -> Entity;
}
