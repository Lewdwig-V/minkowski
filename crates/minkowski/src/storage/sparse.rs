use std::any::Any;
use std::collections::HashMap;

use crate::component::{Component, ComponentId};
use crate::entity::Entity;

/// Type-erased storage for sparse components. Each sparse ComponentId
/// gets a `HashMap<Entity, T>` behind a `Box<dyn Any>`.
pub(crate) struct SparseStorage {
    storages: HashMap<ComponentId, Box<dyn Any + Send + Sync>>,
}

impl SparseStorage {
    pub fn new() -> Self {
        Self { storages: HashMap::new() }
    }

    pub fn insert<T: Component>(&mut self, comp_id: ComponentId, entity: Entity, value: T) {
        let map = self.storages
            .entry(comp_id)
            .or_insert_with(|| Box::new(HashMap::<Entity, T>::new()))
            .downcast_mut::<HashMap<Entity, T>>()
            .expect("component type mismatch in sparse storage");
        map.insert(entity, value);
    }

    pub fn get<T: Component>(&self, comp_id: ComponentId, entity: Entity) -> Option<&T> {
        self.storages.get(&comp_id)?
            .downcast_ref::<HashMap<Entity, T>>()?
            .get(&entity)
    }

    pub fn get_mut<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<&mut T> {
        self.storages.get_mut(&comp_id)?
            .downcast_mut::<HashMap<Entity, T>>()?
            .get_mut(&entity)
    }

    pub fn remove<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<T> {
        self.storages.get_mut(&comp_id)?
            .downcast_mut::<HashMap<Entity, T>>()?
            .remove(&entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entity;

    #[derive(Debug, PartialEq)]
    struct Marker(u32);

    #[test]
    fn insert_and_get() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert(0, e, Marker(42));
        assert_eq!(storage.get::<Marker>(0, e), Some(&Marker(42)));
    }

    #[test]
    fn get_missing_returns_none() {
        let storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn remove() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert(0, e, Marker(42));
        let removed = storage.remove::<Marker>(0, e);
        assert_eq!(removed, Some(Marker(42)));
        assert_eq!(storage.get::<Marker>(0, e), None);
    }
}
