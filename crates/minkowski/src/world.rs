use crate::bundle::Bundle;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::{Entity, EntityAllocator};
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
use crate::storage::archetype::{Archetype, ArchetypeId, Archetypes};
use crate::storage::sparse::SparseStorage;

fn get_pair_mut(v: &mut [Archetype], a: usize, b: usize) -> (&mut Archetype, &mut Archetype) {
    assert_ne!(a, b, "cannot get mutable references to the same archetype");
    if a < b {
        let (left, right) = v.split_at_mut(b);
        (&mut left[a], &mut right[0])
    } else {
        let (left, right) = v.split_at_mut(a);
        (&mut right[0], &mut left[b])
    }
}

#[derive(Clone, Copy)]
pub(crate) struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub row: usize,
}

pub struct World {
    pub(crate) entities: EntityAllocator,
    pub(crate) archetypes: Archetypes,
    pub(crate) components: ComponentRegistry,
    pub(crate) sparse: SparseStorage,
    pub(crate) entity_locations: Vec<Option<EntityLocation>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            archetypes: Archetypes::new(),
            components: ComponentRegistry::new(),
            sparse: SparseStorage::new(),
            entity_locations: Vec::new(),
        }
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let component_ids = B::component_ids(&mut self.components);
        let arch_id = self.archetypes.get_or_create(&component_ids, &self.components);
        let entity = self.entities.alloc();
        let index = entity.index() as usize;

        if index >= self.entity_locations.len() {
            self.entity_locations.resize(index + 1, None);
        }

        let archetype = &mut self.archetypes.archetypes[arch_id.0];
        unsafe {
            bundle.put(&self.components, &mut |comp_id, ptr, _layout| {
                let col = archetype.component_index[&comp_id];
                archetype.columns[col].push(ptr as *mut u8);
            });
        }
        let row = archetype.entities.len();
        archetype.entities.push(entity);

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
        entity
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        let index = entity.index() as usize;
        let location = match self.entity_locations[index] {
            Some(loc) => loc,
            None => return false,
        };

        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        let row = location.row;

        for col in &mut archetype.columns {
            unsafe { col.swap_remove(row); }
        }

        archetype.entities.swap_remove(row);

        // Update the swapped entity's location
        if row < archetype.entities.len() {
            let swapped = archetype.entities[row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: location.archetype_id,
                row,
            });
        }

        self.entity_locations[index] = None;
        self.entities.dealloc(entity);
        true
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let comp_id = self.components.id::<T>()?;

        if self.components.is_sparse(comp_id) {
            return self.sparse.get::<T>(comp_id, entity);
        }

        let archetype = &self.archetypes.archetypes[location.archetype_id.0];
        let col_idx = archetype.component_index.get(&comp_id)?;
        unsafe {
            let ptr = archetype.columns[*col_idx].get_ptr(location.row) as *const T;
            Some(&*ptr)
        }
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let location = self.entity_locations[entity.index() as usize]?;
        let comp_id = self.components.id::<T>()?;

        if self.components.is_sparse(comp_id) {
            return self.sparse.get_mut::<T>(comp_id, entity);
        }

        let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
        let col_idx = *archetype.component_index.get(&comp_id)?;
        unsafe {
            let ptr = archetype.columns[col_idx].get_ptr(location.row) as *mut T;
            Some(&mut *ptr)
        }
    }

    pub fn query<Q: WorldQuery>(&self) -> QueryIter<'_, Q> {
        let required = Q::required_ids(&self.components);
        let fetches: Vec<_> = self.archetypes.archetypes.iter()
            .filter(|arch| {
                !arch.is_empty() && required.is_subset(&arch.component_ids)
            })
            .map(|arch| {
                let fetch = Q::init_fetch(arch, &self.components);
                (fetch, arch.len())
            })
            .collect();
        QueryIter::new(fetches)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        assert!(self.is_alive(entity), "entity is not alive");
        let index = entity.index() as usize;
        let location = self.entity_locations[index].unwrap();
        let comp_id = self.components.register::<T>();

        // If entity already has this component, overwrite in-place
        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        if src_arch.component_ids.contains(comp_id) {
            let col_idx = src_arch.component_index[&comp_id];
            unsafe {
                let ptr = src_arch.columns[col_idx].get_ptr(location.row) as *mut T;
                std::ptr::drop_in_place(ptr);
                std::ptr::write(ptr, component);
            }
            return;
        }

        // Compute target archetype: source components + new component
        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        let mut target_ids = src_arch.sorted_ids.clone();
        target_ids.push(comp_id);
        target_ids.sort_unstable();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        let target_arch_id = self.archetypes.get_or_create(&target_ids, &self.components);

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns: read ptr from source, push to target, swap_remove_no_drop source
        for (&cid, &src_col) in &src_arch.component_index {
            if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        // Write the new component into target
        let tgt_col = target_arch.component_index[&comp_id];
        unsafe {
            let comp = std::mem::ManuallyDrop::new(component);
            target_arch.columns[tgt_col].push(&*comp as *const T as *mut u8);
        }

        // Move entity tracking
        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        // Update swapped entity's location in source
        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let index = entity.index() as usize;
        let location = self.entity_locations[index]?;
        let comp_id = self.components.id::<T>()?;

        let src_arch = &self.archetypes.archetypes[location.archetype_id.0];
        if !src_arch.component_ids.contains(comp_id) {
            return None;
        }

        // Read the component value before migration
        let removed = unsafe {
            let col_idx = src_arch.component_index[&comp_id];
            let ptr = src_arch.columns[col_idx].get_ptr(location.row) as *const T;
            std::ptr::read(ptr)
        };

        // Compute target archetype: source components minus removed component
        let target_ids: Vec<ComponentId> = src_arch.sorted_ids.iter()
            .copied()
            .filter(|&id| id != comp_id)
            .collect();
        let src_arch_id = location.archetype_id;
        let src_row = location.row;

        if target_ids.is_empty() {
            // Entity has no components left — move to empty archetype
            let arch = &mut self.archetypes.archetypes[src_arch_id.0];
            // swap_remove_no_drop for the removed component (already read)
            let removed_col = arch.component_index[&comp_id];
            unsafe { arch.columns[removed_col].swap_remove_no_drop(src_row); }
            // swap_remove with drop for remaining columns
            for (&cid, &col_idx) in &arch.component_index {
                if cid != comp_id {
                    unsafe { arch.columns[col_idx].swap_remove(src_row); }
                }
            }
            arch.entities.swap_remove(src_row);
            if src_row < arch.entities.len() {
                let swapped = arch.entities[src_row];
                self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                    archetype_id: src_arch_id,
                    row: src_row,
                });
            }
            let empty_arch_id = self.archetypes.get_or_create(&[], &self.components);
            let empty_arch = &mut self.archetypes.archetypes[empty_arch_id.0];
            empty_arch.entities.push(entity);
            self.entity_locations[index] = Some(EntityLocation {
                archetype_id: empty_arch_id,
                row: empty_arch.entities.len() - 1,
            });
            return Some(removed);
        }

        let target_arch_id = self.archetypes.get_or_create(&target_ids, &self.components);

        let (src_arch, target_arch) = get_pair_mut(
            &mut self.archetypes.archetypes,
            src_arch_id.0,
            target_arch_id.0,
        );

        // Move shared columns (skip removed component)
        for (&cid, &src_col) in &src_arch.component_index {
            if cid == comp_id {
                // Already read — just discard from source
                unsafe { src_arch.columns[src_col].swap_remove_no_drop(src_row); }
            } else if let Some(&tgt_col) = target_arch.component_index.get(&cid) {
                unsafe {
                    let ptr = src_arch.columns[src_col].get_ptr(src_row);
                    target_arch.columns[tgt_col].push(ptr);
                    src_arch.columns[src_col].swap_remove_no_drop(src_row);
                }
            }
        }

        target_arch.entities.push(entity);
        let target_row = target_arch.entities.len() - 1;
        src_arch.entities.swap_remove(src_row);

        if src_row < src_arch.entities.len() {
            let swapped = src_arch.entities[src_row];
            self.entity_locations[swapped.index() as usize] = Some(EntityLocation {
                archetype_id: src_arch_id,
                row: src_row,
            });
        }

        self.entity_locations[index] = Some(EntityLocation {
            archetype_id: target_arch_id,
            row: target_row,
        });

        Some(removed)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn spawn_and_get() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn spawn_different_archetypes() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e1), None);
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e2), Some(&Vel { dx: 1.0, dy: 0.0 }));
    }

    #[test]
    fn despawn_and_is_alive() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        assert!(world.is_alive(e));
        assert!(world.despawn(e));
        assert!(!world.is_alive(e));
        assert_eq!(world.get::<Pos>(e), None);
    }

    #[test]
    fn entity_recycling() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.despawn(e1);
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        assert_eq!(e2.index(), e1.index());
        assert_ne!(e2.generation(), e1.generation());
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
    }

    #[test]
    fn get_mut() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        if let Some(pos) = world.get_mut::<Pos>(e) {
            pos.x = 10.0;
        }
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 2.0 }));
    }

    #[test]
    fn insert_new_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, Vel { dx: 3.0, dy: 4.0 });
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
    }

    #[test]
    fn insert_overwrites_existing() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert(e, Pos { x: 10.0, y: 20.0 });
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 10.0, y: 20.0 }));
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        let removed = world.remove::<Vel>(e);
        assert_eq!(removed, Some(Vel { dx: 3.0, dy: 4.0 }));
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 2.0 }));
        assert_eq!(world.get::<Vel>(e), None);
    }

    #[test]
    fn migration_preserves_other_entities() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));
        let e3 = world.spawn((Pos { x: 3.0, y: 0.0 },));

        // Migrate e1 — e3 should swap into e1's old row
        world.insert(e1, Vel { dx: 1.0, dy: 0.0 });

        // All entities still accessible with correct data
        assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 0.0 }));
        assert_eq!(world.get::<Vel>(e1), Some(&Vel { dx: 1.0, dy: 0.0 }));
        assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 2.0, y: 0.0 }));
        assert_eq!(world.get::<Pos>(e3), Some(&Pos { x: 3.0, y: 0.0 }));
    }
}
