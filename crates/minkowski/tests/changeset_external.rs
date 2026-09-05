//! Integration test: exercises EnumChangeSet typed API from outside the crate.
//! This test would have caught the original ComponentId visibility bug.

use minkowski::{ComponentId, EnumChangeSet, World};

/// Run with `cargo test -p minkowski --release --test changeset_external
/// replay_recycled_slots_scaling -- --ignored --nocapture`.
#[test]
#[ignore = "release-mode replay timing; no wall-clock threshold in CI"]
fn replay_recycled_slots_scaling() {
    use minkowski::Entity;
    use std::time::Instant;

    for n in [25_000u32, 50_000, 100_000] {
        for order in ["tail", "head", "permuted"] {
            let mut times = Vec::new();
            for _ in 0..3 {
                let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
                world.restore_allocator_state(vec![1; n as usize], (0..n).collect());
                let mut changes = EnumChangeSet::new();
                for i in 0..n {
                    let index = match order {
                        "tail" => n - 1 - i,
                        "head" => i,
                        _ => (i * 7919) % n,
                    };
                    changes.record_spawn(Entity::from_bits((1u64 << 32) | index as u64), &[]);
                }
                let start = Instant::now();
                changes.apply_replay(&mut world).unwrap();
                times.push(start.elapsed());
                assert!(world.entity_allocator_state().1.is_empty());
                assert_eq!(world.alloc_entity().index(), n);
            }
            times.sort();
            println!("{order:8} {n:6} slots: {:?} median", times[1]);
        }
    }
}

#[test]
fn replay_recycled_slots_preserves_order_and_snapshot() {
    use minkowski::Entity;
    let entity =
        |index: u32, generation: u32| Entity::from_bits(((generation as u64) << 32) | index as u64);
    let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
    world.restore_allocator_state(vec![1; 12], (0..12).collect());
    let mut changes = EnumChangeSet::new();
    for index in [11, 3, 7] {
        changes.record_spawn(entity(index, 1), &[]);
    }
    changes.apply_replay(&mut world).unwrap();
    assert_eq!(
        world.entity_allocator_state().1,
        &[0, 1, 2, 4, 5, 6, 8, 9, 10]
    );
    assert_eq!(world.alloc_entity(), entity(10, 1));
    assert!(world.despawn(entity(7, 1)));
    assert_eq!(
        world.entity_allocator_state().1,
        &[0, 1, 2, 4, 5, 6, 8, 9, 7]
    );

    let mut changes = EnumChangeSet::new();
    changes.record_spawn(entity(7, 2), &[]);
    changes.record_spawn(entity(1, 1), &[]);
    changes.record_despawn(entity(3, 1));
    changes.record_spawn(entity(3, 2), &[]);
    changes.apply_replay(&mut world).unwrap();
    let (generations, free) = world.entity_allocator_state();
    assert_eq!(free, &[0, 2, 4, 5, 6, 8, 9]);
    // Restore a different order after the old list has been indexed and read.
    let (generations, free) = (generations.to_vec(), free.iter().rev().copied().collect());
    world.restore_allocator_state(generations, free);
    for index in [4, 8, 5, 2] {
        // Separate replay calls must reuse the index, including after compaction.
        let mut changes = EnumChangeSet::new();
        changes.record_spawn(entity(index, 1), &[]);
        changes.apply_replay(&mut world).unwrap();
    }
    assert_eq!(world.entity_allocator_state().1, &[9, 6, 0]);
    for index in [0, 6, 9] {
        assert_eq!(world.alloc_entity(), entity(index, 1));
    }
    assert!(world.entity_allocator_state().1.is_empty());
    assert_eq!(world.alloc_entity().index(), 12);
}

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
fn typed_insert() {
    let mut world = World::new();
    let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

    // Insert via typed API
    let mut cs = EnumChangeSet::new();
    cs.insert::<Vel>(&mut world, e, Vel { dx: 3.0, dy: 4.0 });
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 3.0, dy: 4.0 }));
}

#[test]
fn typed_remove() {
    let mut world = World::new();
    let e = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

    let mut cs = EnumChangeSet::new();
    cs.remove::<Vel>(&mut world, e);
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Vel>(e), None);
}

#[test]
fn component_id_lookup() {
    let mut world = World::new();
    assert_eq!(world.component_id::<Pos>(), None);

    let id: ComponentId = world.register_component::<Pos>();
    assert_eq!(world.component_id::<Pos>(), Some(id));
}
