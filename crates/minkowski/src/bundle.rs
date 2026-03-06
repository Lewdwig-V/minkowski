//! Bundle trait for tuple-based entity construction.
//!
//! Bundles define which components an entity is spawned with. Tuple types
//! up to 12 elements implement [`Bundle`] automatically.

use crate::component::{Component, ComponentId, ComponentRegistry};
use std::alloc::Layout;

/// A collection of components that can be added to an entity.
/// Implemented for tuples of Components via macro.
///
/// # Safety
/// `put` must call `func` exactly once per component with valid pointers,
/// and must not drop the component data (ownership transfers to the caller).
pub unsafe trait Bundle: Send + Sync + 'static {
    fn component_ids(registry: &mut ComponentRegistry) -> Vec<ComponentId>;

    /// Write each component to the callback: (ComponentId, *const u8, Layout).
    /// Components are consumed -- caller takes ownership via the pointer.
    ///
    /// # Safety
    /// The caller must ensure that the `func` callback correctly takes ownership
    /// of the component data at the given pointer and does not double-free it.
    unsafe fn put(
        self,
        registry: &ComponentRegistry,
        func: &mut dyn FnMut(ComponentId, *const u8, Layout),
    );
}

macro_rules! count {
    () => { 0usize };
    ($x:ident $(, $rest:ident)*) => { 1usize + count!($($rest),*) };
}

macro_rules! impl_bundle {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: Component),*> Bundle for ($($name,)*) {
            fn component_ids(registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                let mut ids = vec![$(registry.register::<$name>()),*];
                ids.sort_unstable();
                let expected = count!($($name),*);
                ids.dedup();
                assert_eq!(ids.len(), expected, "duplicate component types in bundle");
                ids
            }

            unsafe fn put(
                self,
                registry: &ComponentRegistry,
                func: &mut dyn FnMut(ComponentId, *const u8, Layout),
            ) {
                let ($($name,)*) = self;
                $(
                    let $name = std::mem::ManuallyDrop::new($name);
                    func(
                        registry.id::<$name>().unwrap(),
                        &*$name as *const $name as *const u8,
                        Layout::new::<$name>(),
                    );
                )*
            }
        }
    };
}

impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);
impl_bundle!(A, B, C, D, E, F, G, H, I);
impl_bundle!(A, B, C, D, E, F, G, H, I, J);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct A(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct B(f32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct C(u8);

    #[test]
    fn single_component_ids() {
        let mut reg = ComponentRegistry::new();
        let ids = <(A,)>::component_ids(&mut reg);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn pair_component_ids_sorted() {
        let mut reg = ComponentRegistry::new();
        let ids = <(A, B)>::component_ids(&mut reg);
        assert_eq!(ids.len(), 2);
        assert!(ids[0] <= ids[1]);
    }

    #[test]
    fn triple_component_ids() {
        let mut reg = ComponentRegistry::new();
        let ids = <(A, B, C)>::component_ids(&mut reg);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn put_writes_correct_data() {
        let mut reg = ComponentRegistry::new();
        let _ = <(A, B)>::component_ids(&mut reg);

        let bundle = (A(42), B(3.5));
        let mut written: Vec<(ComponentId, Vec<u8>)> = Vec::new();

        unsafe {
            bundle.put(&reg, &mut |comp_id, ptr, layout| {
                let mut data = vec![0u8; layout.size()];
                std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), layout.size());
                written.push((comp_id, data));
            });
        }

        assert_eq!(written.len(), 2);
        let a_id = reg.id::<A>().unwrap();
        let a_entry = written.iter().find(|(id, _)| *id == a_id).unwrap();
        let a_val: A = unsafe { std::ptr::read_unaligned(a_entry.1.as_ptr() as *const A) };
        assert_eq!(a_val, A(42));
    }
}
