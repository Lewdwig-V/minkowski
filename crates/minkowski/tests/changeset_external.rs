//! Integration test: exercises EnumChangeSet typed API from outside the crate.
//! This test would have caught the original ComponentId visibility bug.

use minkowski::{ComponentId, EnumChangeSet, World};

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

#[test]
fn typed_insert_remove_roundtrip() {
    let mut world = World::new();
    let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

    // Insert via typed API
    let mut cs = EnumChangeSet::new();
    cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });
    let reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));

    // Reverse undoes the insert
    let _ = reverse.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), None);
}

#[test]
fn typed_remove_roundtrip() {
    let mut world = World::new();
    let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

    let mut cs = EnumChangeSet::new();
    cs.remove::<Vel>(&mut world, e);
    let reverse = cs.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), None);

    let _ = reverse.apply(&mut world);
    assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
}

#[test]
fn component_id_lookup() {
    let mut world = World::new();
    assert_eq!(world.component_id::<Pos>(), None);

    let id: ComponentId = world.register_component::<Pos>();
    assert_eq!(world.component_id::<Pos>(), Some(id));
}
