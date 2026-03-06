use fixedbitset::FixedBitSet;
use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;

/// Marker trait for ECS components. Blanket-implemented for all eligible types.
pub trait Component: 'static + Send + Sync {}
impl<T: 'static + Send + Sync> Component for T {}

/// Unique identifier for a registered component type.
///
/// Assigned sequentially by the component registry. Use
/// [`World::component_id`](crate::World::component_id) to look up the ID for a type.
pub type ComponentId = usize;

#[allow(dead_code)]
pub(crate) struct ComponentInfo {
    pub id: ComponentId,
    pub name: &'static str,
    pub layout: Layout,
    pub drop_fn: Option<unsafe fn(*mut u8)>,
}

#[doc(hidden)]
pub struct ComponentRegistry {
    by_type: HashMap<TypeId, ComponentId>,
    infos: Vec<ComponentInfo>,
    sparse_set: FixedBitSet,
}

impl ComponentRegistry {
    pub(crate) fn new() -> Self {
        Self {
            by_type: HashMap::new(),
            infos: Vec::new(),
            sparse_set: FixedBitSet::new(),
        }
    }

    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.by_type.get(&type_id) {
            return id;
        }
        let id = self.infos.len();
        let drop_fn = if std::mem::needs_drop::<T>() {
            Some(Self::drop_ptr::<T> as unsafe fn(*mut u8))
        } else {
            None
        };
        self.infos.push(ComponentInfo {
            id,
            name: std::any::type_name::<T>(),
            layout: Layout::new::<T>(),
            drop_fn,
        });
        self.by_type.insert(type_id, id);
        id
    }

    /// Register a component slot using raw metadata (name + layout) without
    /// a concrete Rust type. Used during snapshot restore to fill the ID space
    /// for components that were registered in the original world but don't have
    /// a codec in the persist crate.
    ///
    /// No TypeId is recorded — the slot cannot be looked up via `id::<T>()`.
    /// No drop function — the persist crate never instantiates these placeholder
    /// components, so there is nothing to drop.
    pub fn register_raw(&mut self, name: &'static str, layout: Layout) -> ComponentId {
        let id = self.infos.len();
        self.infos.push(ComponentInfo {
            id,
            name,
            layout,
            drop_fn: None,
        });
        id
    }

    #[allow(dead_code)]
    pub(crate) fn register_sparse<T: Component>(&mut self) -> ComponentId {
        let id = self.register::<T>();
        self.sparse_set.grow(id + 1);
        self.sparse_set.insert(id);
        id
    }

    pub fn id<T: Component>(&self) -> Option<ComponentId> {
        self.by_type.get(&TypeId::of::<T>()).copied()
    }

    /// Returns the number of registered components.
    /// Also serves as the next `ComponentId` that will be assigned.
    pub(crate) fn len(&self) -> usize {
        self.infos.len()
    }

    pub(crate) fn info(&self, id: ComponentId) -> &ComponentInfo {
        &self.infos[id]
    }

    pub(crate) fn is_sparse(&self, id: ComponentId) -> bool {
        self.sparse_set.contains(id)
    }

    unsafe fn drop_ptr<T>(ptr: *mut u8) {
        std::ptr::drop_in_place(ptr as *mut T);
    }
}

/// Type-erased drop glue: calls `drop_in_place::<T>` on a raw pointer.
/// Used by `EnumChangeSet::insert_raw` and similar pre-resolved paths
/// that don't have access to `ComponentRegistry`.
///
/// # Safety
/// The pointer must point to a valid, initialized `T`.
#[allow(dead_code)]
pub(crate) unsafe fn drop_ptr<T>(ptr: *mut u8) {
    std::ptr::drop_in_place(ptr as *mut T);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position {
        _x: f32,
        _y: f32,
    }
    struct Velocity {
        _dx: f32,
        _dy: f32,
    }
    #[allow(dead_code)]
    struct Health(u32);

    #[test]
    fn register_returns_sequential_ids() {
        let mut reg = ComponentRegistry::new();
        let a = reg.register::<Position>();
        let b = reg.register::<Velocity>();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn register_is_idempotent() {
        let mut reg = ComponentRegistry::new();
        let a = reg.register::<Position>();
        let b = reg.register::<Position>();
        assert_eq!(a, b);
    }

    #[test]
    fn id_lookup() {
        let mut reg = ComponentRegistry::new();
        assert_eq!(reg.id::<Position>(), None);
        reg.register::<Position>();
        assert_eq!(reg.id::<Position>(), Some(0));
    }

    #[test]
    fn info_has_correct_layout() {
        let mut reg = ComponentRegistry::new();
        let id = reg.register::<Position>();
        let info = reg.info(id);
        assert_eq!(info.layout, std::alloc::Layout::new::<Position>());
    }

    #[test]
    fn sparse_registration() {
        let mut reg = ComponentRegistry::new();
        let id = reg.register_sparse::<Health>();
        assert!(reg.is_sparse(id));
        let pos_id = reg.register::<Position>();
        assert!(!reg.is_sparse(pos_id));
    }
}
