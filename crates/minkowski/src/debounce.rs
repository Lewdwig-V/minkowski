//! Subscription debouncing — filter false positives from archetype-granular
//! `Changed<T>` detection.
//!
//! `Changed<T>` is archetype-granular: mutating one entity marks the entire
//! column as changed, so unchanged siblings in the same archetype also pass
//! the filter. A [`SubscriptionDebounce`] tracks per-entity values and
//! suppresses entities whose value has not actually changed.
//!
//! The default [`HashDebounce`] uses an in-memory `HashMap<Entity, T>`.
//! Implement the trait on your own type to back it with Redis, a persistent
//! store, or any other deduplication mechanism.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::entity::Entity;

/// Filter false positives from archetype-granular `Changed<T>` detection.
///
/// `Changed<T>` marks an entire archetype column when any entity in it is
/// mutated. A debounce filter tracks the last-seen value per entity and
/// suppresses entities whose value has not actually changed.
///
/// # Example
///
/// ```ignore
/// let mut debounce = HashDebounce::<Score>::new();
///
/// sub.for_each(&mut world, |entity| {
///     let score = world.get::<Score>(entity).unwrap();
///     if debounce.is_changed(entity, score) {
///         // genuinely new or changed — react
///     }
/// })?;
/// ```
pub trait SubscriptionDebounce<T> {
    /// Returns `true` if this entity's value is new or different from the
    /// last time this method returned `true` for it. First observation
    /// always returns `true`.
    fn is_changed(&mut self, entity: Entity, value: &T) -> bool;

    /// Stop tracking a despawned entity. Call this when you know an entity
    /// has been removed to avoid unbounded memory growth.
    fn remove(&mut self, entity: Entity);
}

/// In-memory debounce filter backed by `HashMap<Entity, T>`.
///
/// Tracks the last-seen value per entity. [`is_changed`](SubscriptionDebounce::is_changed)
/// returns `true` on first observation and whenever the value differs from
/// the stored copy.
///
/// For most workloads this is sufficient. Implement [`SubscriptionDebounce`]
/// on your own type if you need external-backed deduplication (e.g. Redis).
pub struct HashDebounce<T> {
    seen: HashMap<Entity, T>,
}

impl<T> HashDebounce<T> {
    /// Create an empty debounce filter.
    pub fn new() -> Self {
        Self {
            seen: HashMap::new(),
        }
    }

    /// Number of entities currently tracked.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Returns `true` if no entities are tracked.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Remove tracking for entities that are no longer alive.
    pub fn retain(&mut self, mut keep: impl FnMut(Entity) -> bool) {
        self.seen.retain(|&entity, _| keep(entity));
    }
}

impl<T> Default for HashDebounce<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq + Clone> SubscriptionDebounce<T> for HashDebounce<T> {
    fn is_changed(&mut self, entity: Entity, value: &T) -> bool {
        match self.seen.entry(entity) {
            Entry::Vacant(e) => {
                e.insert(value.clone());
                true
            }
            Entry::Occupied(mut e) => {
                if e.get() != value {
                    e.insert(value.clone());
                    true
                } else {
                    false
                }
            }
        }
    }

    fn remove(&mut self, entity: Entity) {
        self.seen.remove(&entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Score(u32);

    #[test]
    fn first_observation_is_always_changed() {
        let mut d = HashDebounce::<Score>::new();
        let e = Entity::from_bits(0);
        assert!(d.is_changed(e, &Score(42)));
    }

    #[test]
    fn same_value_is_not_changed() {
        let mut d = HashDebounce::<Score>::new();
        let e = Entity::from_bits(0);
        assert!(d.is_changed(e, &Score(42)));
        assert!(!d.is_changed(e, &Score(42)));
        assert!(!d.is_changed(e, &Score(42)));
    }

    #[test]
    fn different_value_is_changed() {
        let mut d = HashDebounce::<Score>::new();
        let e = Entity::from_bits(0);
        assert!(d.is_changed(e, &Score(42)));
        assert!(d.is_changed(e, &Score(99)));
        assert!(!d.is_changed(e, &Score(99)));
    }

    #[test]
    fn remove_forgets_entity() {
        let mut d = HashDebounce::<Score>::new();
        let e = Entity::from_bits(0);
        assert!(d.is_changed(e, &Score(42)));
        d.remove(e);
        // After removal, same value is "new" again.
        assert!(d.is_changed(e, &Score(42)));
    }

    #[test]
    fn retain_removes_unmatched() {
        let mut d = HashDebounce::<Score>::new();
        let e1 = Entity::from_bits(0);
        let e2 = Entity::from_bits(1);
        d.is_changed(e1, &Score(1));
        d.is_changed(e2, &Score(2));
        assert_eq!(d.len(), 2);

        d.retain(|e| e == e1);
        assert_eq!(d.len(), 1);
        // e2 was removed — re-observing is "new".
        assert!(d.is_changed(e2, &Score(2)));
    }

    #[test]
    fn different_entities_are_independent() {
        let mut d = HashDebounce::<Score>::new();
        let e1 = Entity::from_bits(0);
        let e2 = Entity::from_bits(1);
        assert!(d.is_changed(e1, &Score(42)));
        assert!(d.is_changed(e2, &Score(42)));
        assert!(!d.is_changed(e1, &Score(42)));
        assert!(!d.is_changed(e2, &Score(42)));
    }

    #[test]
    fn empty_and_len() {
        let mut d = HashDebounce::<Score>::new();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);

        let e = Entity::from_bits(0);
        d.is_changed(e, &Score(1));
        assert!(!d.is_empty());
        assert_eq!(d.len(), 1);
    }
}
