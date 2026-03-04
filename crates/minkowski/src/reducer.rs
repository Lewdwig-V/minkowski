use crate::access::Access;
use crate::component::{Component, ComponentId, ComponentRegistry};

// ── ComponentSet & Contains ──────────────────────────────────────────

/// Declares a set of component types with pre-resolved IDs.
/// Macro-generated for tuples 1–12.
pub trait ComponentSet: 'static {
    const COUNT: usize;

    /// Build an Access bitset for this component set.
    /// `read_only = true` → all components go in reads.
    /// `read_only = false` → all components go in writes.
    fn access(registry: &mut ComponentRegistry, read_only: bool) -> Access;

    /// Pre-resolve all ComponentIds (registers if needed). Returns them
    /// in positional order matching `Contains<T, INDEX>`.
    fn resolve(registry: &mut ComponentRegistry) -> Vec<ComponentId>;
}

/// Compile-time proof that `T` is at position `INDEX` in the component set.
/// The const generic disambiguates positions so that tuples like `(A, B)`
/// don't produce overlapping impls when A == B.
///
/// When calling `handle.get::<T>()`, the compiler infers INDEX from the
/// unique matching impl — no manual index needed at the call site.
pub trait Contains<T: Component, const INDEX: usize> {}

/// Pre-resolved ComponentIds created once at registration time.
/// `Contains<T, INDEX>` positions index into the inner Vec.
#[allow(dead_code)]
pub(crate) struct ResolvedComponents(pub(crate) Vec<ComponentId>);

// ── Macro ────────────────────────────────────────────────────────────

macro_rules! impl_component_set {
    ($($idx:tt: $name:ident),+) => {
        impl_component_set!(@trait $($name),+);
        impl_component_set!(@contains { $($name),+ } $($idx: $name),+);
    };

    (@trait $($name:ident),+) => {
        impl<$($name: Component),+> ComponentSet for ($($name,)+) {
            const COUNT: usize = impl_component_set!(@count $($name),+);

            fn access(registry: &mut ComponentRegistry, read_only: bool) -> Access {
                let mut access = Access::empty();
                $(
                    let id = registry.register::<$name>();
                    if read_only {
                        access.add_read(id);
                    } else {
                        access.add_write(id);
                    }
                )+
                access
            }

            fn resolve(registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                vec![$(registry.register::<$name>()),+]
            }
        }
    };

    // TT muncher: peel one Contains impl at a time, forwarding the
    // full type list in braces. Avoids cross-depth repetition issues.
    (@contains { $($all:ident),+ } $idx:tt: $target:ident, $($rest:tt)+) => {
        impl<$($all: Component),+> Contains<$target, $idx> for ($($all,)+) {}
        impl_component_set!(@contains { $($all),+ } $($rest)+);
    };
    (@contains { $($all:ident),+ } $idx:tt: $target:ident) => {
        impl<$($all: Component),+> Contains<$target, $idx> for ($($all,)+) {}
    };

    (@count $x:ident) => { 1usize };
    (@count $x:ident, $($rest:ident),+) => { 1usize + impl_component_set!(@count $($rest),+) };
}

impl_component_set!(0: A);
impl_component_set!(0: A, 1: B);
impl_component_set!(0: A, 1: B, 2: C);
impl_component_set!(0: A, 1: B, 2: C, 3: D);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_component_set!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Health(u32);

    #[test]
    fn component_set_count_single() {
        assert_eq!(<(Pos,) as ComponentSet>::COUNT, 1);
    }

    #[test]
    fn component_set_count_pair() {
        assert_eq!(<(Pos, Vel) as ComponentSet>::COUNT, 2);
    }

    #[test]
    fn component_set_resolve() {
        let mut reg = ComponentRegistry::new();
        let ids = <(Pos, Vel)>::resolve(&mut reg);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], reg.id::<Pos>().unwrap());
        assert_eq!(ids[1], reg.id::<Vel>().unwrap());
    }

    #[test]
    fn contains_bound_compiles() {
        // Verify that Contains<T, IDX> impls exist for the expected positions.
        // The functions exist purely for the compile-time bound check.
        fn assert_contains_at_0<C: Contains<Pos, 0>>(_: std::marker::PhantomData<C>) {}
        fn assert_contains_at_1<C: Contains<Vel, 1>>(_: std::marker::PhantomData<C>) {}
        assert_contains_at_0::<(Pos, Vel)>(std::marker::PhantomData);
        assert_contains_at_1::<(Pos, Vel)>(std::marker::PhantomData);
    }

    #[test]
    fn resolved_components_lookup() {
        let mut reg = ComponentRegistry::new();
        let ids = <(Pos, Vel)>::resolve(&mut reg);
        let resolved = ResolvedComponents(ids);
        // Position 0 → Pos's ComponentId, position 1 → Vel's ComponentId
        assert_eq!(resolved.0[0], reg.id::<Pos>().unwrap());
        assert_eq!(resolved.0[1], reg.id::<Vel>().unwrap());
    }

    #[test]
    fn access_read_only() {
        let mut reg = ComponentRegistry::new();
        let a = <(Pos, Vel)>::access(&mut reg, true);
        let pos_id = reg.id::<Pos>().unwrap();
        let vel_id = reg.id::<Vel>().unwrap();
        assert!(a.reads()[pos_id]);
        assert!(a.reads()[vel_id]);
        assert!(a.writes().is_empty());
    }

    #[test]
    fn access_write() {
        let mut reg = ComponentRegistry::new();
        let a = <(Pos, Vel)>::access(&mut reg, false);
        let pos_id = reg.id::<Pos>().unwrap();
        let vel_id = reg.id::<Vel>().unwrap();
        assert!(a.reads().is_empty());
        assert!(a.writes()[pos_id]);
        assert!(a.writes()[vel_id]);
    }

    #[test]
    fn access_merge_read_and_write() {
        let mut reg = ComponentRegistry::new();
        let reads = <(Pos,)>::access(&mut reg, true);
        let writes = <(Vel,)>::access(&mut reg, false);
        let merged = reads.merge(&writes);
        let pos_id = reg.id::<Pos>().unwrap();
        let vel_id = reg.id::<Vel>().unwrap();
        assert!(merged.reads()[pos_id]);
        assert!(merged.writes()[vel_id]);
    }

    #[test]
    fn triple_component_set() {
        let mut reg = ComponentRegistry::new();
        assert_eq!(<(Pos, Vel, Health) as ComponentSet>::COUNT, 3);
        let ids = <(Pos, Vel, Health)>::resolve(&mut reg);
        assert_eq!(ids.len(), 3);

        fn assert_pos_0<C: Contains<Pos, 0>>(_: std::marker::PhantomData<C>) {}
        fn assert_vel_1<C: Contains<Vel, 1>>(_: std::marker::PhantomData<C>) {}
        fn assert_health_2<C: Contains<Health, 2>>(_: std::marker::PhantomData<C>) {}
        assert_pos_0::<(Pos, Vel, Health)>(std::marker::PhantomData);
        assert_vel_1::<(Pos, Vel, Health)>(std::marker::PhantomData);
        assert_health_2::<(Pos, Vel, Health)>(std::marker::PhantomData);
    }

    /// Verify the index-inference pattern used by typed handles:
    /// `get::<T>()` calls a helper bounded by `C: Contains<T, IDX>`
    /// and the compiler infers IDX from the unique matching impl.
    #[test]
    fn index_inference() {
        fn get_index<T: Component, C, const IDX: usize>(
            resolved: &ResolvedComponents,
            _marker: std::marker::PhantomData<C>,
        ) -> ComponentId
        where
            C: Contains<T, IDX>,
        {
            resolved.0[IDX]
        }

        let mut reg = ComponentRegistry::new();
        let ids = <(Pos, Vel)>::resolve(&mut reg);
        let resolved = ResolvedComponents(ids);

        let pos_id = get_index::<Pos, (Pos, Vel), 0>(&resolved, std::marker::PhantomData);
        let vel_id = get_index::<Vel, (Pos, Vel), 1>(&resolved, std::marker::PhantomData);
        assert_eq!(pos_id, reg.id::<Pos>().unwrap());
        assert_eq!(vel_id, reg.id::<Vel>().unwrap());
    }
}
