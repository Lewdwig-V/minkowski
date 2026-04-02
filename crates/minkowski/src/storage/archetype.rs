use std::collections::HashMap;

use fixedbitset::FixedBitSet;

use super::blob_vec::BlobVec;
use crate::component::{ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::pool::SharedPool;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ArchetypeId(pub usize);

#[allow(dead_code)]
pub(crate) struct Archetype {
    pub id: ArchetypeId,
    /// Bitset of which ComponentIds this archetype contains.
    pub component_ids: FixedBitSet,
    /// Sorted list of ComponentIds (canonical key for archetype lookup).
    pub sorted_ids: Vec<ComponentId>,
    /// One BlobVec per component, in sorted_ids order.
    pub columns: Vec<BlobVec>,
    // PERF: Dense Vec indexed by ComponentId — O(1) lookup without hashing.
    // ComponentIds are assigned sequentially by ComponentRegistry, so the
    // Vec is at most max_component_id+1 entries. Replaces HashMap for
    // every init_fetch, get_mut, insert, and migration path.
    /// ComponentId -> index into columns. Dense array indexed by ComponentId.
    pub component_index: Vec<Option<usize>>,
    /// Row -> Entity mapping.
    pub entities: Vec<Entity>,
}

impl Archetype {
    pub fn new(
        id: ArchetypeId,
        sorted_component_ids: &[ComponentId],
        registry: &ComponentRegistry,
        pool: &SharedPool,
    ) -> Self {
        let max_id = sorted_component_ids.iter().copied().max().unwrap_or(0);
        let mut bitset = FixedBitSet::with_capacity(max_id + 1);
        let mut columns = Vec::with_capacity(sorted_component_ids.len());
        let mut component_index = vec![None; max_id + 1];

        for (col_idx, &comp_id) in sorted_component_ids.iter().enumerate() {
            bitset.insert(comp_id);
            let info = registry.info(comp_id);
            columns.push(BlobVec::new(info.layout, info.drop_fn, 0, pool.clone()));
            component_index[comp_id] = Some(col_idx);
        }

        Self {
            id,
            component_ids: bitset,
            sorted_ids: sorted_component_ids.to_vec(),
            columns,
            component_index,
            entities: Vec::new(),
        }
    }

    /// Look up the column index for a component. O(1) indexed access.
    #[inline]
    pub fn column_index(&self, comp_id: ComponentId) -> Option<usize> {
        self.component_index.get(comp_id).copied().flatten()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Assert that every column has exactly as many rows as `entities.len()`.
    /// Zero-cost in release builds — fires only under `debug_assertions`.
    #[inline]
    pub fn debug_assert_consistent(&self) {
        if cfg!(debug_assertions) {
            let entity_len = self.entities.len();
            for (i, col) in self.columns.iter().enumerate() {
                debug_assert_eq!(
                    col.len(),
                    entity_len,
                    "archetype {:?} column {} has {} rows but entities has {} — \
                     column/entity count mismatch after structural mutation",
                    self.id,
                    i,
                    col.len(),
                    entity_len,
                );
            }
        }
    }

    /// Returns true if any column in this archetype has dirty pages.
    pub fn any_dirty(&self) -> bool {
        self.columns.iter().any(|col| col.dirty_pages.any_dirty())
    }

    /// Clear dirty page bits for all columns. Called after a successful flush.
    pub fn clear_dirty_pages(&mut self) {
        for col in &mut self.columns {
            col.dirty_pages.clear();
        }
    }
}

/// Collection of archetypes with lookup by component set.
#[allow(clippy::struct_field_names)]
pub(crate) struct Archetypes {
    pub archetypes: Vec<Archetype>,
    by_components: HashMap<Vec<ComponentId>, ArchetypeId>,
    generation: u64,
}

impl Archetypes {
    pub fn new() -> Self {
        Self {
            archetypes: Vec::new(),
            by_components: HashMap::new(),
            generation: 0,
        }
    }

    #[allow(dead_code)]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Find or create an archetype for the given sorted component ID set.
    pub fn get_or_create(
        &mut self,
        sorted_ids: &[ComponentId],
        registry: &ComponentRegistry,
        pool: &SharedPool,
    ) -> ArchetypeId {
        if let Some(&id) = self.by_components.get(sorted_ids) {
            return id;
        }
        let id = ArchetypeId(self.archetypes.len());
        let archetype = Archetype::new(id, sorted_ids, registry, pool);
        self.archetypes.push(archetype);
        self.by_components.insert(sorted_ids.to_vec(), id);
        self.generation += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;
    use crate::entity::Entity;
    use crate::pool::default_pool;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    fn setup_registry() -> ComponentRegistry {
        ComponentRegistry::new()
    }

    #[test]
    fn archetype_creation() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let vel_id = reg.register::<Vel>();
        let mut ids = vec![pos_id, vel_id];
        ids.sort_unstable();
        let arch = Archetype::new(ArchetypeId(0), &ids, &reg, &default_pool());
        assert!(arch.component_ids.contains(pos_id));
        assert!(arch.component_ids.contains(vel_id));
        assert_eq!(arch.len(), 0);
    }

    #[test]
    fn push_and_read_row() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let mut ids = vec![pos_id];
        ids.sort_unstable();
        let mut arch = Archetype::new(ArchetypeId(0), &ids, &reg, &default_pool());

        let entity = Entity::new(0, 0);
        let mut pos = Pos { x: 1.0, y: 2.0 };
        unsafe {
            let col = arch.column_index(pos_id).unwrap();
            arch.columns[col].push(&mut pos as *mut Pos as *mut u8);
            let _ = pos;
            arch.entities.push(entity);
        }
        assert_eq!(arch.len(), 1);

        unsafe {
            let col = arch.column_index(pos_id).unwrap();
            let ptr = arch.columns[col].get_ptr(0) as *const Pos;
            assert_eq!(*ptr, Pos { x: 1.0, y: 2.0 });
        }
    }

    #[test]
    fn archetypes_get_or_create() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let vel_id = reg.register::<Vel>();

        let mut archetypes = Archetypes::new();

        let mut ids = vec![pos_id, vel_id];
        ids.sort_unstable();
        let a1 = archetypes.get_or_create(&ids, &reg, &default_pool());
        let a2 = archetypes.get_or_create(&ids, &reg, &default_pool());
        assert_eq!(a1, a2); // idempotent

        let ids2 = vec![pos_id];
        let a3 = archetypes.get_or_create(&ids2, &reg, &default_pool());
        assert_ne!(a1, a3); // different component set = different archetype
    }

    #[test]
    fn archetypes_generation_bumps_on_create() {
        let mut reg = setup_registry();
        let pos_id = reg.register::<Pos>();
        let mut archetypes = Archetypes::new();

        let gen_before = archetypes.generation();
        archetypes.get_or_create(&[pos_id], &reg, &default_pool());
        assert!(archetypes.generation() > gen_before);

        let gen_before = archetypes.generation();
        archetypes.get_or_create(&[pos_id], &reg, &default_pool()); // same, no new archetype
        assert_eq!(archetypes.generation(), gen_before);
    }
}
