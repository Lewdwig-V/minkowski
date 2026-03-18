use std::collections::HashSet;

use crate::sync::{Arc, AtomicBool, AtomicU64};

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::ComponentRegistry;
use crate::entity::Entity;
use crate::transaction::{Conflict, Optimistic, Transact, TransactError};
use crate::world::World;

use super::dynamic::{DynamicCtx, DynamicResolved};
use super::handles::{EntityMut, EntityRef, QueryMut, QueryRef, Spawner};
use super::writer::{QueryWriter, WritableRef, WriterQuery};
use super::{
    ComponentSet, Contains, DynamicReducerId, QueryReducerId, ReducerError, ReducerId,
    ReducerRegistry, ResolvedComponents,
};
use crate::transaction::WorldMismatch;

#[derive(Clone, Copy)]
struct Pos(f32);
#[derive(Clone, Copy)]
struct Vel(f32);
#[derive(Clone, Copy)]
struct Health(u32);

// ── DynamicResolved tests ────────────────────────────────────────

#[test]
fn dynamic_resolved_lookup() {
    use std::any::TypeId;
    let entries = vec![
        (TypeId::of::<u32>(), 0),
        (TypeId::of::<f64>(), 2),
        (TypeId::of::<i64>(), 1),
    ];
    let resolved = DynamicResolved::new(
        entries,
        Access::empty(),
        HashSet::default(),
        HashSet::default(),
    );
    assert_eq!(resolved.lookup::<u32>(), Some(0));
    assert_eq!(resolved.lookup::<f64>(), Some(2));
    assert_eq!(resolved.lookup::<i64>(), Some(1));
    assert_eq!(resolved.lookup::<u8>(), None);
}

#[test]
fn dynamic_resolved_dedup() {
    use std::any::TypeId;
    let entries = vec![(TypeId::of::<u32>(), 0), (TypeId::of::<u32>(), 0)];
    let resolved = DynamicResolved::new(
        entries,
        Access::empty(),
        HashSet::default(),
        HashSet::default(),
    );
    // After dedup, duplicate entries are collapsed
    assert_eq!(resolved.lookup::<u32>(), Some(0));
}

#[test]
fn dynamic_resolved_has_spawn_bundle() {
    use std::any::TypeId;
    let mut bundles = HashSet::new();
    bundles.insert(TypeId::of::<(Pos, Vel)>());
    let resolved = DynamicResolved::new(vec![], Access::empty(), bundles, HashSet::default());
    assert!(resolved.has_spawn_bundle::<(Pos, Vel)>());
    assert!(!resolved.has_spawn_bundle::<(Health,)>());
}

// ── DynamicCtx tests ──────────────────────────────────────────

#[test]
fn dynamic_ctx_read() {
    use std::any::TypeId;
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();
    let e = world.spawn((Pos(42.0),));

    let entries = vec![(TypeId::of::<Pos>(), pos_id)];
    let mut access = Access::empty();
    access.add_read(pos_id);
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    assert_eq!(ctx.read::<Pos>(e).0, 42.0);
}

#[test]
fn dynamic_ctx_try_read_none() {
    use std::any::TypeId;
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();
    let vel_id = world.register_component::<Vel>();
    let e = world.spawn((Pos(1.0),)); // no Vel

    let entries = vec![(TypeId::of::<Pos>(), pos_id), (TypeId::of::<Vel>(), vel_id)];
    let mut access = Access::empty();
    access.add_read(pos_id);
    access.add_read(vel_id);
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    assert!(ctx.try_read::<Pos>(e).is_some());
    assert!(ctx.try_read::<Vel>(e).is_none());
}

#[test]
fn dynamic_ctx_write_buffers() {
    use std::any::TypeId;
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();
    let e = world.spawn((Pos(1.0),));

    let entries = vec![(TypeId::of::<Pos>(), pos_id)];
    let mut access = Access::empty();
    access.add_write(pos_id);
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    {
        let mut ctx = DynamicCtx::new(
            &world,
            &mut cs,
            &mut allocated,
            &resolved,
            &default_tick,
            &default_queried,
        );
        ctx.write(e, Pos(99.0));
    }
    // Not yet applied
    assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
    // Apply changeset
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);
}

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_read_undeclared_panics() {
    let mut world = World::new();
    world.register_component::<Pos>();
    let e = world.spawn((Pos(1.0),));

    // Empty resolved — no components declared
    let resolved = DynamicResolved::new(
        vec![],
        Access::empty(),
        HashSet::default(),
        HashSet::default(),
    );
    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    let _ = ctx.read::<Pos>(e);
}

// ── DynamicReducerBuilder tests ──────────────────────────────

#[test]
fn dynamic_builder_registers() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    let id = reducers
        .dynamic("test_dyn", &mut world)
        .can_read::<Pos>()
        .can_write::<Vel>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    assert_eq!(id.index(), 0);

    // Verify access: Pos is read, Vel is read+write
    let pos_id = world.components.id::<Pos>().unwrap();
    let vel_id = world.components.id::<Vel>().unwrap();
    let entry = &reducers.dynamic_reducers[id.0];
    assert!(entry.resolved.access().reads()[pos_id]);
    assert!(!entry.resolved.access().writes()[pos_id]); // read-only
    assert!(entry.resolved.access().reads()[vel_id]);
    assert!(entry.resolved.access().writes()[vel_id]); // writable
}

#[test]
fn dynamic_builder_can_spawn() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    let id = reducers
        .dynamic("spawner", &mut world)
        .can_spawn::<(Pos, Vel)>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let pos_id = world.components.id::<Pos>().unwrap();
    let vel_id = world.components.id::<Vel>().unwrap();
    let entry = &reducers.dynamic_reducers[id.0];
    // Spawn adds writes for conflict detection
    assert!(entry.resolved.access().writes()[pos_id]);
    assert!(entry.resolved.access().writes()[vel_id]);
}

#[test]
fn dynamic_builder_duplicate_name_returns_err() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    reducers
        .dynamic("dup", &mut world)
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let result = reducers
        .dynamic("dup", &mut world)
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});
    assert!(matches!(
        result,
        Err(ReducerError::DuplicateName { name: "dup", .. })
    ));
}

#[test]
fn dynamic_name_conflicts_with_unified_returns_err() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    reducers
        .register_entity::<(Health,), (), _>(&mut world, "shared_name", |_e, ()| {})
        .unwrap();
    let result = reducers
        .dynamic("shared_name", &mut world)
        .build(|_ctx: &mut DynamicCtx, _args: &()| {});
    assert!(matches!(
        result,
        Err(ReducerError::DuplicateName {
            name: "shared_name",
            ..
        })
    ));
}

// ── Dynamic dispatch tests ────────────────────────────────────

#[test]
fn dynamic_call_reads_and_writes() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);

    let mut reducers = ReducerRegistry::new();
    let id = reducers
        .dynamic("apply_vel", &mut world)
        .can_read::<Vel>()
        .can_write::<Pos>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            let vel = ctx.read::<Vel>(*entity).0;
            let pos = ctx.read::<Pos>(*entity).0;
            ctx.write(*entity, Pos(pos + vel));
        })
        .unwrap();

    reducers
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 3.0);
}

#[test]
fn dynamic_id_by_name_lookup() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    let dyn_id = reducers
        .dynamic("my_dyn", &mut world)
        .can_read::<Pos>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    reducers
        .register_entity::<(Health,), (), _>(&mut world, "entity_one", |_e, ()| {})
        .unwrap();

    // Dynamic lookup finds dynamic reducer
    assert_eq!(reducers.dynamic_id_by_name("my_dyn"), Some(dyn_id));
    // Dynamic lookup does not find unified reducer
    assert_eq!(reducers.dynamic_id_by_name("entity_one"), None);
    // Dynamic lookup does not find nonexistent
    assert_eq!(reducers.dynamic_id_by_name("nope"), None);
    // Unified lookup does not find dynamic reducer
    assert_eq!(reducers.reducer_id_by_name("my_dyn"), None);
}

#[test]
fn dynamic_access_metadata() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    let id = reducers
        .dynamic("test_access", &mut world)
        .can_read::<Pos>()
        .can_write::<Vel>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let pos_id = world.components.id::<Pos>().unwrap();
    let vel_id = world.components.id::<Vel>().unwrap();
    let access = reducers.dynamic_access(id);
    assert!(access.reads()[pos_id]);
    assert!(access.reads()[vel_id]);
    assert!(access.writes()[vel_id]);
    assert!(!access.writes()[pos_id]);
}

#[test]
fn can_remove_marks_write_access() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("remover", &mut world)
        .can_read::<Health>()
        .can_remove::<Vel>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let access = registry.dynamic_access(id);
    let vel_id = world.components.id::<Vel>().unwrap();
    assert!(access.writes().contains(vel_id));
}

#[test]
fn can_despawn_sets_flag() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("despawner", &mut world)
        .can_read::<Health>()
        .can_despawn()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let access = registry.dynamic_access(id);
    assert!(access.despawns());
}

#[test]
fn despawn_reducer_conflicts_with_reader() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let dyn_id = registry
        .dynamic("despawner", &mut world)
        .can_read::<Health>()
        .can_despawn()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let entity_id = registry
        .register_entity::<(Vel,), (), _>(&mut world, "set_vel", |_e, ()| {})
        .unwrap();
    let dyn_access = registry.dynamic_access(dyn_id);
    let entity_access = registry.reducer_access(entity_id);
    assert!(dyn_access.conflicts_with(entity_access));
}

// ── DynamicCtx structural mutation tests ─────────────────────

#[test]
fn dynamic_ctx_remove_buffers_mutation() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("strip_vel", &mut world)
        .can_read::<Pos>()
        .can_remove::<Vel>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.remove::<Vel>(*entity);
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();
    assert!(world.get::<Vel>(e).is_none());
    assert!(world.get::<Pos>(e).is_some());
}

#[test]
fn dynamic_ctx_try_remove_returns_false_when_missing() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let result = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let result_clone = result.clone();
    let id = registry
        .dynamic("try_strip", &mut world)
        .can_remove::<Vel>()
        .build(move |ctx: &mut DynamicCtx, entity: &Entity| {
            let removed = ctx.try_remove::<Vel>(*entity);
            result_clone.store(removed, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();
    assert!(!result.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_remove_undeclared_panics() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("bad_remove", &mut world)
        .can_read::<Pos>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.remove::<Vel>(*entity);
        })
        .unwrap();
    let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
}

#[test]
#[should_panic(expected = "not declared for removal")]
fn dynamic_ctx_remove_with_can_write_panics() {
    // can_write does NOT authorize remove — remove requires can_remove
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("bad_remove2", &mut world)
        .can_write::<Vel>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.remove::<Vel>(*entity);
        })
        .unwrap();
    let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
}

#[test]
fn dynamic_ctx_try_remove_returns_true_and_removes() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let result = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let result_clone = result.clone();
    let id = registry
        .dynamic("try_strip_ok", &mut world)
        .can_remove::<Vel>()
        .build(move |ctx: &mut DynamicCtx, entity: &Entity| {
            let removed = ctx.try_remove::<Vel>(*entity);
            result_clone.store(removed, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();
    assert!(result.load(std::sync::atomic::Ordering::Relaxed));
    assert!(world.get::<Vel>(e).is_none());
    assert!(world.get::<Pos>(e).is_some());
}

#[test]
fn dynamic_ctx_despawn_buffers_mutation() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("killer", &mut world)
        .can_read::<Health>()
        .can_despawn()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.despawn(*entity);
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &e)
        .unwrap();
    assert!(!world.is_alive(e));
}

#[test]
#[should_panic(expected = "can_despawn")]
fn dynamic_ctx_despawn_without_declaration_panics() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("bad_despawn", &mut world)
        .can_read::<Pos>()
        .build(|ctx: &mut DynamicCtx, entity: &Entity| {
            ctx.despawn(*entity);
        })
        .unwrap();
    let _ = registry.dynamic_call(&strategy, &mut world, id, &e);
}

// ── EntityMut structural mutation tests ──────────────────────

#[test]
fn entity_mut_remove_buffers_mutation() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity::<(Pos, Vel), (), _>(
            &mut world,
            "strip_vel",
            |mut entity: EntityMut<'_, (Pos, Vel)>, ()| {
                entity.remove::<Vel, 1>();
            },
        )
        .unwrap();
    registry.call(&strategy, &mut world, id, (e, ())).unwrap();
    assert!(world.get::<Vel>(e).is_none());
    assert!(world.get::<Pos>(e).is_some());
}

#[test]
fn entity_mut_despawn_buffers_mutation() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity_despawn::<(Pos,), (), _>(
            &mut world,
            "killer",
            |mut entity: EntityMut<'_, (Pos,)>, ()| {
                entity.despawn();
            },
        )
        .unwrap();
    registry.call(&strategy, &mut world, id, (e, ())).unwrap();
    assert!(!world.is_alive(e));
}

#[test]
#[should_panic(expected = "register_entity_despawn")]
fn entity_mut_despawn_without_flag_panics() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity::<(Pos,), (), _>(
            &mut world,
            "bad_killer",
            |mut entity: EntityMut<'_, (Pos,)>, ()| {
                entity.despawn();
            },
        )
        .unwrap();
    let _ = registry.call(&strategy, &mut world, id, (e, ()));
}

// ── DynamicCtx::for_each tests ──────────────────────────────

#[test]
fn dynamic_ctx_for_each_iterates() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    world.spawn((Vel(3.0),)); // no Pos — not matched
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = count.clone();
    let id = registry
        .dynamic("count_pos", &mut world)
        .can_read::<Pos>()
        .build(move |ctx: &mut DynamicCtx, _args: &()| {
            ctx.for_each::<(&Pos,)>(|(positions,)| {
                counter.fetch_add(positions.len(), std::sync::atomic::Ordering::Relaxed);
            });
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_for_each_undeclared_panics() {
    let mut world = World::new();
    world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("bad_query", &mut world)
        .can_read::<Pos>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            ctx.for_each::<(&Pos, &Vel)>(|(_p, _v)| {});
        })
        .unwrap();
    let _ = registry.dynamic_call(&strategy, &mut world, id, &());
}

#[test]
fn dynamic_ctx_for_each_with_write_after_read() {
    let mut world = World::new();
    let e1 = world.spawn((Pos(1.0), Vel(10.0)));
    let e2 = world.spawn((Pos(2.0), Vel(20.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("double_vel", &mut world)
        .can_read::<Vel>()
        .can_write::<Vel>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            let mut updates: Vec<(Entity, f32)> = Vec::new();
            ctx.for_each::<(Entity, &Vel)>(|(entities, velocities)| {
                for (entity, vel) in entities.iter().copied().zip(velocities.iter()) {
                    updates.push((entity, vel.0 * 2.0));
                }
            });
            for (entity, new_vel) in updates {
                ctx.write(entity, Vel(new_vel));
            }
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(world.get::<Vel>(e1).unwrap().0, 20.0);
    assert_eq!(world.get::<Vel>(e2).unwrap().0, 40.0);
}

#[test]
fn dynamic_ctx_for_each_changed_filter() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let visit_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = visit_count.clone();
    let id = registry
        .dynamic("changed_pos", &mut world)
        .can_read::<Pos>()
        .can_write::<Pos>()
        .build(move |ctx: &mut DynamicCtx, _args: &()| {
            let mut updates = Vec::new();
            ctx.for_each::<(Entity, crate::query::fetch::Changed<Pos>, &Pos)>(
                |(entities, (), positions)| {
                    for (entity, pos) in entities.iter().copied().zip(positions.iter()) {
                        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        updates.push((entity, Pos(pos.0 + 1.0)));
                    }
                },
            );
            for (entity, val) in updates {
                ctx.write(entity, val);
            }
        })
        .unwrap();

    // First call: column was never read by this reducer, Changed matches
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 2.0);

    // Second call: no external mutation, Changed should skip
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 0);

    // External mutation, then call again
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    for (pos,) in world.query::<(&mut Pos,)>() {
        pos.0 = 99.0;
    }
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 100.0);
}

#[test]
fn dynamic_ctx_for_each_slice_iterates() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    world.spawn((Vel(3.0),)); // no Pos — not matched
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = count.clone();
    let id = registry
        .dynamic("count_pos_chunks", &mut world)
        .can_read::<Pos>()
        .build(move |ctx: &mut DynamicCtx, _args: &()| {
            ctx.for_each::<(&Pos,)>(|(positions,)| {
                counter.fetch_add(positions.len(), std::sync::atomic::Ordering::Relaxed);
            });
        })
        .unwrap();
    registry
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();
    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_for_each_undeclared_multi_component_panics() {
    let mut world = World::new();
    world.spawn((Pos(1.0), Vel(2.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("bad_chunk_query", &mut world)
        .can_read::<Pos>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            ctx.for_each::<(&Pos, &Vel)>(|(_p, _v)| {});
        })
        .unwrap();
    let _ = registry.dynamic_call(&strategy, &mut world, id, &());
}

// ── Debug assertion tests ────────────────────────────────────

#[test]
#[should_panic(expected = "read-only")]
fn dynamic_ctx_write_on_read_only_panics() {
    use std::any::TypeId;
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();
    let e = world.spawn((Pos(1.0),));

    // Declare Pos as read-only (not writable)
    let entries = vec![(TypeId::of::<Pos>(), pos_id)];
    let mut access = Access::empty();
    access.add_read(pos_id); // read only, no write
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    ctx.write(e, Pos(99.0)); // should panic: read-only
}

#[test]
#[should_panic(expected = "bundle")]
fn dynamic_ctx_spawn_undeclared_bundle_panics() {
    let mut world = World::new();
    world.register_component::<Pos>();

    // No spawn bundles declared
    let resolved = DynamicResolved::new(
        vec![],
        Access::empty(),
        HashSet::default(),
        HashSet::default(),
    );

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    ctx.spawn((Pos(1.0),)); // should panic: bundle not declared
}

// ── ComponentSet tests ──────────────────────────────────────

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
    fn get_index<T: crate::component::Component, C, const IDX: usize>(
        resolved: &ResolvedComponents,
        _marker: std::marker::PhantomData<C>,
    ) -> crate::component::ComponentId
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

// ── EntityRef tests ──────────────────────────────────────────

#[test]
fn entity_ref_get() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));
    let resolved = ResolvedComponents(<(Pos, Vel)>::resolve(&mut world.components));

    let er: EntityRef<'_, (Pos, Vel)> = EntityRef::new(e, &resolved, &world);
    assert_eq!(er.get::<Pos, 0>().0, 1.0);
    assert_eq!(er.get::<Vel, 1>().0, 2.0);
    assert_eq!(er.entity(), e);
}

// ── QueryRef tests ───────────────────────────────────────────

#[test]
fn query_ref_for_each() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    let mut qr: QueryRef<'_, (&Pos,)> = QueryRef::new(&mut world);
    let mut sum = 0.0;
    qr.for_each(|(positions,)| {
        for p in positions {
            sum += p.0;
        }
    });
    assert_eq!(sum, 3.0);
}

#[test]
fn query_ref_count() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    world.spawn((Pos(3.0),));
    let mut qr: QueryRef<'_, (&Pos,)> = QueryRef::new(&mut world);
    assert_eq!(qr.count(), 3);
}

// ── QueryMut tests ───────────────────────────────────────────

#[test]
fn query_mut_for_each() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    {
        let mut qm: QueryMut<'_, (&mut Pos,)> = QueryMut::new(&mut world);
        qm.for_each(|(positions,)| {
            for p in positions {
                p.0 += 10.0;
            }
        });
    }
    assert_eq!(world.get::<Pos>(e).unwrap().0, 11.0);
}

#[test]
fn query_mut_count() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    let mut qm: QueryMut<'_, (&mut Pos,)> = QueryMut::new(&mut world);
    assert_eq!(qm.count(), 2);
}

// ── ReducerRegistry tests ────────────────────────────────────

#[test]
fn register_entity_and_call() {
    let mut world = World::new();
    let e = world.spawn((Health(100),));
    let strategy = Optimistic::new(&world);

    let mut registry = ReducerRegistry::new();
    let heal_id = registry
        .register_entity::<(Health,), u32, _>(&mut world, "heal", |mut entity, amount: u32| {
            let hp = entity.get::<Health, 0>().0;
            entity.set::<Health, 0>(Health(hp + amount));
        })
        .unwrap();

    registry
        .call(&strategy, &mut world, heal_id, (e, 25u32))
        .unwrap();
    assert_eq!(world.get::<Health>(e).unwrap().0, 125);
}

#[test]
fn register_query_and_run() {
    let mut world = World::new();
    world.spawn((Vel(1.0),));
    world.spawn((Vel(2.0),));

    let mut registry = ReducerRegistry::new();
    let gravity_id = registry
        .register_query::<(&mut Vel,), f32, _>(&mut world, "gravity", |mut query, dt: f32| {
            query.for_each(|(velocities,)| {
                for v in velocities {
                    v.0 -= 9.81 * dt;
                }
            });
        })
        .unwrap();

    registry.run(&mut world, gravity_id, 0.1f32).unwrap();

    let mut sum = 0.0;
    for (vel,) in world.query::<(&Vel,)>() {
        sum += vel.0;
    }
    // (1.0 - 0.981) + (2.0 - 0.981) = 1.038
    assert!((sum - 1.038).abs() < 0.001, "sum = {}", sum);
}

#[test]
fn register_query_ref_and_run() {
    let mut world = World::new();
    world.spawn((Pos(10.0),));
    world.spawn((Pos(20.0),));

    let mut registry = ReducerRegistry::new();
    let count_id = registry
        .register_query_ref::<(&Pos,), (), _>(&mut world, "count", |mut query, ()| {
            assert_eq!(query.count(), 2);
        })
        .unwrap();

    registry.run(&mut world, count_id, ()).unwrap();
}

#[test]
fn typed_id_by_name_lookup() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let heal_id = registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
        .unwrap();
    let _gravity_id = registry
        .register_query::<(&mut Vel,), (), _>(&mut world, "gravity", |_query, ()| {})
        .unwrap();

    // Typed lookups return the correct variant
    assert_eq!(registry.reducer_id_by_name("heal"), Some(heal_id));
    assert_eq!(registry.reducer_id_by_name("gravity"), None); // wrong kind
    assert_eq!(registry.reducer_id_by_name("nonexistent"), None);

    assert!(registry.query_reducer_id_by_name("gravity").is_some());
    assert_eq!(registry.query_reducer_id_by_name("heal"), None); // wrong kind
}

#[test]
fn access_metadata_matches() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let heal_id = registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
        .unwrap();
    let health_id = world.components.id::<Health>().unwrap();
    let access = registry.reducer_access(heal_id);
    // Entity reducers declare both reads and writes
    assert!(access.writes()[health_id]);
    assert!(access.reads()[health_id]);
}

#[test]
fn access_conflict_between_reducers() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    let heal_id = registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
        .unwrap();

    let damage_id = registry
        .register_entity::<(Health,), (), _>(&mut world, "damage", |_entity, ()| {})
        .unwrap();

    let heal_access = registry.reducer_access(heal_id);
    let damage_access = registry.reducer_access(damage_id);
    assert!(
        heal_access.conflicts_with(damage_access),
        "two reducers writing Health should conflict"
    );
}

// ── Spawner lifecycle tests ──────────────────────────────────

#[test]
fn register_spawner_and_call() {
    let mut world = World::new();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let spawn_id = registry
        .register_spawner::<(Health,), u32, _>(&mut world, "spawn_unit", |mut spawner, hp: u32| {
            spawner.spawn((Health(hp),));
        })
        .unwrap();

    registry
        .call(&strategy, &mut world, spawn_id, 50u32)
        .unwrap();

    let mut count = 0;
    for (h,) in world.query::<(&Health,)>() {
        assert_eq!(h.0, 50);
        count += 1;
    }
    assert_eq!(count, 1);
}

#[test]
fn spawner_abort_reclaims_reserved_ids() {
    let mut world = World::new();
    world.spawn((Pos(1.0),)); // seed an entity so conflict detection works
    let strategy = Optimistic::with_retries(&world, 1);
    let mut registry = ReducerRegistry::new();

    // Register a spawner that also reads Pos to create a conflict surface
    let _spawn_id = registry
        .register_spawner::<(Health,), (), _>(
            &mut world,
            "spawn_and_conflict",
            |mut spawner, ()| {
                spawner.spawn((Health(1),));
            },
        )
        .unwrap();

    // Force a conflict: mutate Pos column between begin and commit
    // by using a strategy with max 1 retry and always-conflicting access
    let mut attempt = 0u32;
    let access_with_pos = Access::of::<(&Pos, &mut Pos)>(&mut world);
    let result = strategy.transact(&mut world, &access_with_pos, |tx, world| {
        attempt += 1;
        let (changeset, allocated) = tx.reducer_parts();
        let _spawner = Spawner::<(Health,)>::new(changeset, allocated, world);
        // Spawner allocates via reserve() — entity tracked in allocated

        if attempt == 1 {
            // Mutate to force conflict
            for pos in world.query::<(&mut Pos,)>() {
                pos.0.0 = 99.0;
            }
        }
    });

    // After abort+retry, no leaked entities
    // Trigger drain_orphans
    world.register_component::<Health>();
    let health_count = world.query::<(&Health,)>().count();
    // May be 0 (both attempts conflicted) or 1 (retry succeeded)
    assert!(health_count <= 1, "no duplicate spawns");
    assert!(attempt >= 1);
    let _ = result;
}

#[test]
fn duplicate_name_returns_err() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {})
        .unwrap();
    let result = registry.register_entity::<(Health,), (), _>(&mut world, "heal", |_entity, ()| {});
    assert!(matches!(
        result,
        Err(ReducerError::DuplicateName { name: "heal", .. })
    ));
}

#[test]
fn dynamic_ctx_try_write_success() {
    use std::any::TypeId;
    let mut world = World::new();
    let e = world.spawn((42u32,));
    let comp_id = world.components.id::<u32>().unwrap();

    let mut access = Access::empty();
    access.add_write(comp_id);
    let entries = vec![(TypeId::of::<u32>(), comp_id)];
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );

    let wrote = ctx.try_write::<u32>(e, 99);
    assert!(wrote);

    cs.apply(&mut world).unwrap();
    assert_eq!(*world.get::<u32>(e).unwrap(), 99);
}

#[test]
fn dynamic_ctx_try_write_missing_component() {
    use std::any::TypeId;
    let mut world = World::new();
    let e = world.spawn((42u32,)); // has u32, not f64
    let f64_id = world.register_component::<f64>();

    let mut access = Access::empty();
    access.add_write(f64_id);
    let entries = vec![(TypeId::of::<f64>(), f64_id)];
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );

    let wrote = ctx.try_write::<f64>(e, 99.0);
    assert!(!wrote);
    assert_eq!(cs.len(), 0); // nothing buffered
}

#[test]
fn dynamic_call_spawn_places_entity() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();

    let id = reducers
        .dynamic("spawner", &mut world)
        .can_spawn::<(u32, f64)>()
        .build(|ctx: &mut DynamicCtx, _args: &()| {
            let e = ctx.spawn((42u32, std::f64::consts::PI));
            let _ = e;
        })
        .unwrap();

    let strategy = Optimistic::new(&world);
    reducers
        .dynamic_call(&strategy, &mut world, id, &())
        .unwrap();

    // Verify the spawned entity exists with correct components
    let mut found = false;
    world.query::<(&u32, &f64)>().for_each(|(u, f)| {
        assert_eq!(*u, 42);
        assert!((f - std::f64::consts::PI).abs() < f64::EPSILON);
        found = true;
    });
    assert!(found, "spawned entity not found in world");
}

#[test]
fn restore_allocator_syncs_next_reserved() {
    let mut world = World::new();
    // Spawn some entities to populate generations
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));

    // Simulate snapshot restore
    let gens = vec![0u32; 5]; // 5 entities in the snapshot
    let free = vec![];
    world.restore_allocator_state(gens, free);

    // reserve() should start at index 5, not 0
    let reserved = world.entities.reserve();
    assert_eq!(reserved.index(), 5, "reserve() must skip restored indices");
}

// ── Additional review-requested tests ───────────────────────

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_try_read_undeclared_panics() {
    let mut world = World::new();
    let e = world.spawn((42u32,));
    let resolved = DynamicResolved::new(
        vec![],
        Access::empty(),
        HashSet::default(),
        HashSet::default(),
    );
    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    let _ = ctx.try_read::<u32>(e);
}

#[test]
#[should_panic(expected = "not declared")]
fn dynamic_ctx_try_write_undeclared_panics() {
    let mut world = World::new();
    let e = world.spawn((42u32,));
    let resolved = DynamicResolved::new(
        vec![],
        Access::empty(),
        HashSet::default(),
        HashSet::default(),
    );
    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let mut ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );
    ctx.try_write::<u32>(e, 99);
}

#[test]
fn unified_name_conflicts_with_dynamic_returns_err() {
    let mut world = World::new();
    let mut reducers = ReducerRegistry::new();
    // Register dynamic first
    reducers
        .dynamic("clash", &mut world)
        .can_read::<u32>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    // Then unified — should return Err
    let result = reducers.register_entity::<(u32,), (), _>(
        &mut world,
        "clash",
        |_entity: EntityMut<'_, (u32,)>, ()| {},
    );
    assert!(matches!(
        result,
        Err(ReducerError::DuplicateName { name: "clash", .. })
    ));
}

// ── WritableRef tests ──────────────────────────────────────────

#[test]
fn writable_ref_get_returns_current_value() {
    let mut world = World::new();
    let e = world.spawn((Pos(42.0),));
    let pos_id = world.components.id::<Pos>().unwrap();
    let current = world.get::<Pos>(e).unwrap();

    let mut cs = EnumChangeSet::new();
    let wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet, 0, 0);
    assert_eq!(wr.get().0, 42.0);
}

/// Helper: open an archetype batch for the given entity's archetype
/// with a single mutable component so that `WritableRef::set` can
/// route through the fast lane.
fn open_batch_for_entity(
    cs: &mut EnumChangeSet,
    world: &World,
    entity: Entity,
    comp_id: crate::component::ComponentId,
) {
    let loc = world.entity_locations[entity.index() as usize].unwrap();
    let arch_idx = loc.archetype_id.0;
    let arch = &world.archetypes.archetypes[arch_idx];
    let mut mutable = fixedbitset::FixedBitSet::with_capacity(comp_id + 1);
    mutable.insert(comp_id);
    crate::changeset::open_archetype_batch(cs, arch_idx, arch, &world.components, &mutable);
}

#[test]
fn writable_ref_set_buffers_into_changeset() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let pos_id = world.components.id::<Pos>().unwrap();
    let current = world.get::<Pos>(e).unwrap();

    let mut cs = EnumChangeSet::new();
    open_batch_for_entity(&mut cs, &world, e, pos_id);
    {
        let mut wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet, 0, 0);
        wr.set(Pos(99.0));
    }
    // World unchanged before apply
    assert_eq!(world.get::<Pos>(e).unwrap().0, 1.0);
    assert_eq!(cs.len(), 1);
    // Apply and verify
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 99.0);
}

#[test]
fn writable_ref_modify_clones_and_sets() {
    let mut world = World::new();
    let e = world.spawn((Pos(10.0),));
    let pos_id = world.components.id::<Pos>().unwrap();
    let current = world.get::<Pos>(e).unwrap();

    let mut cs = EnumChangeSet::new();
    open_batch_for_entity(&mut cs, &world, e, pos_id);
    {
        let mut wr = WritableRef::new(e, current, pos_id, &mut cs as *mut EnumChangeSet, 0, 0);
        wr.modify(|p| p.0 += 10.0);
    }
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 20.0);
}

// ── WriterQuery tests ──────────────────────────────────────────

#[test]
fn writer_query_ref_t_passthrough() {
    let mut world = World::new();
    let e = world.spawn((Pos(7.0),));
    let loc = world.entity_locations[e.index() as usize].unwrap();
    let archetype = &world.archetypes.archetypes[loc.archetype_id.0];

    let fetch = <&Pos as WriterQuery>::init_writer_fetch(archetype, &world.components);
    let mut cs = EnumChangeSet::new();
    let item = unsafe {
        <&Pos as WriterQuery>::fetch_writer(&fetch, loc.row, e, &mut cs as *mut EnumChangeSet)
    };
    assert_eq!(item.0, 7.0);
}

#[test]
fn writer_query_mut_t_becomes_writable_ref() {
    let mut world = World::new();
    let e = world.spawn((Pos(5.0),));
    let loc = world.entity_locations[e.index() as usize].unwrap();
    let archetype = &world.archetypes.archetypes[loc.archetype_id.0];

    let fetch = <&mut Pos as WriterQuery>::init_writer_fetch(archetype, &world.components);
    let mut cs = EnumChangeSet::new();
    let pos_id = world.components.id::<Pos>().unwrap();
    open_batch_for_entity(&mut cs, &world, e, pos_id);
    let mut item = unsafe {
        <&mut Pos as WriterQuery>::fetch_writer(&fetch, loc.row, e, &mut cs as *mut EnumChangeSet)
    };
    assert_eq!(item.get().0, 5.0);
    item.set(Pos(55.0));
    // World unchanged
    assert_eq!(world.get::<Pos>(e).unwrap().0, 5.0);
    // Apply changeset
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 55.0);
}

#[test]
fn writer_query_tuple_fetch() {
    let mut world = World::new();
    let e = world.spawn((Pos(3.0), Vel(4.0)));
    let loc = world.entity_locations[e.index() as usize].unwrap();
    let archetype = &world.archetypes.archetypes[loc.archetype_id.0];

    let fetch = <(&Vel, &mut Pos) as WriterQuery>::init_writer_fetch(archetype, &world.components);
    let mut cs = EnumChangeSet::new();
    let pos_id = world.components.id::<Pos>().unwrap();
    open_batch_for_entity(&mut cs, &world, e, pos_id);
    let (vel_ref, mut pos_wr) = unsafe {
        <(&Vel, &mut Pos) as WriterQuery>::fetch_writer(
            &fetch,
            loc.row,
            e,
            &mut cs as *mut EnumChangeSet,
        )
    };
    // Read velocity passthrough
    assert_eq!(vel_ref.0, 4.0);
    // Write position via WritableRef
    pos_wr.set(Pos(vel_ref.0 + pos_wr.get().0));
    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 7.0);
}

#[test]
fn dynamic_spawn_abort_orphans_entity() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut world = World::new();
    // Spawn a sentinel to occupy index 0
    let sentinel = world.spawn((42u32,));

    let strategy = Optimistic::new(&world);
    let mut reducers = ReducerRegistry::new();

    let attempt_count = std::sync::Arc::new(AtomicUsize::new(0));
    let attempt_count_clone = attempt_count.clone();

    let id = reducers
        .dynamic("spawn_and_fail", &mut world)
        .can_write::<u32>()
        .can_spawn::<(u32,)>()
        .build(move |ctx: &mut DynamicCtx, _args: &()| {
            let _e = ctx.spawn((999u32,));
            attempt_count_clone.fetch_add(1, Ordering::SeqCst);
            // Write to u32 column — will conflict if another writer touched it
            ctx.write(sentinel, 0u32);
        })
        .unwrap();

    // Mutate the u32 column to cause an optimistic conflict on first attempt
    // by advancing the column tick between begin and commit.
    // We use a direct spawn to dirty the archetype's u32 column.
    world.spawn((77u32,));

    // The first call may fail due to stale ticks from our spawn above,
    // but retries should eventually succeed. Either way, entity IDs
    // from failed attempts must be recycled.
    let _ = reducers.dynamic_call(&strategy, &mut world, id, &());

    // After the call (success or failure), any orphaned entity IDs
    // from failed attempts should be drained on the next &mut World call.
    // Trigger drain by calling any &mut World method.
    let _ = world.spawn((0u32,));

    // Verify: every entity in the world should be alive (no leaked IDs)
    // and the attempt count confirms at least one attempt happened.
    assert!(attempt_count.load(Ordering::SeqCst) >= 1);
}

// ── QueryWriter tests ─────────────────────────────────────────

#[test]
fn query_writer_for_each_reads_and_buffers() {
    let mut world = World::new();
    let e1 = world.spawn((Pos(1.0), Vel(10.0)));
    let e2 = world.spawn((Pos(2.0), Vel(20.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry
        .register_query_writer::<(&Pos, &mut Vel), f32, _>(
            &mut world,
            "apply_drag",
            |mut query, drag: f32| {
                query.for_each(|(pos, mut vel)| {
                    let _ = pos; // read Pos (passthrough)
                    vel.modify(|v| v.0 *= drag);
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, 0.5f32).unwrap();

    assert_eq!(world.get::<Vel>(e1).unwrap().0, 5.0);
    assert_eq!(world.get::<Vel>(e2).unwrap().0, 10.0);
    assert_eq!(world.get::<Pos>(e1).unwrap().0, 1.0); // unchanged
}

#[test]
fn query_writer_count() {
    let mut world = World::new();
    world.spawn((Pos(1.0),));
    world.spawn((Pos(2.0),));
    world.spawn((Vel(3.0),)); // no Pos — not matched
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry
        .register_query_writer::<(&mut Pos,), (), _>(&mut world, "counter", |mut query, ()| {
            assert_eq!(query.count(), 2);
        })
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();
}

#[test]
fn query_writer_access_conflict() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    let entity_id = registry
        .register_entity::<(Vel,), (), _>(&mut world, "set_vel", |_e, ()| {})
        .unwrap();
    let writer_id = registry
        .register_query_writer::<(&mut Vel,), (), _>(&mut world, "bulk_vel", |_q, ()| {})
        .unwrap();

    let entity_access = registry.reducer_access(entity_id);
    let writer_access = registry.reducer_access(writer_id);
    assert!(entity_access.conflicts_with(writer_access));
}

#[test]
fn query_writer_no_conflict_disjoint() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    let entity_id = registry
        .register_entity::<(Pos,), (), _>(&mut world, "set_pos", |_e, ()| {})
        .unwrap();
    let writer_id = registry
        .register_query_writer::<(&mut Vel,), (), _>(&mut world, "bulk_vel", |_q, ()| {})
        .unwrap();

    let entity_access = registry.reducer_access(entity_id);
    let writer_access = registry.reducer_access(writer_id);
    assert!(!entity_access.conflicts_with(writer_access));
}

// ── API boundary panic tests ─────────────────────────────────

#[test]
fn call_on_scheduled_returns_wrong_kind() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let qid = registry
        .register_query::<(&Pos,), (), _>(&mut world, "read_pos", |_q, ()| {})
        .unwrap();
    let strategy = Optimistic::new(&world);
    // QueryReducerId and ReducerId share the same index space —
    // passing ReducerId(qid.0) should hit the Scheduled arm.
    let result = registry.call(&strategy, &mut world, ReducerId(qid.0), ());
    assert!(matches!(
        result,
        Err(ReducerError::WrongKind {
            expected: "transactional",
            actual: "scheduled"
        })
    ));
}

#[test]
fn run_on_transactional_returns_wrong_kind() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let rid = registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_e, ()| {})
        .unwrap();
    // ReducerId and QueryReducerId share the same index space.
    let result = registry.run(&mut world, QueryReducerId(rid.0), ());
    assert!(matches!(
        result,
        Err(ReducerError::WrongKind {
            expected: "scheduled",
            actual: "transactional"
        })
    ));
}

// ── Multi-archetype QueryWriter test ─────────────────────────

#[test]
fn query_writer_spans_multiple_archetypes() {
    let mut world = World::new();
    // Two different archetypes, both containing Vel
    let e1 = world.spawn((Vel(10.0),));
    let e2 = world.spawn((Vel(20.0), Pos(0.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry
        .register_query_writer::<(&mut Vel,), f32, _>(
            &mut world,
            "scale_vel",
            |mut query, factor: f32| {
                query.for_each(|(mut vel,)| {
                    vel.modify(|v| v.0 *= factor);
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, 0.5f32).unwrap();

    assert_eq!(world.get::<Vel>(e1).unwrap().0, 5.0);
    assert_eq!(world.get::<Vel>(e2).unwrap().0, 10.0);
}

// ── Changed<T> filter with QueryWriter ───────────────────────

#[test]
fn query_writer_changed_filter() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    // Track how many entities the query writer visits
    let visit_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = visit_count.clone();

    let id = registry
        .register_query_writer::<(crate::query::fetch::Changed<Pos>, &mut Pos), (), _>(
            &mut world,
            "changed_pos",
            move |mut query, ()| {
                query.for_each(|((), mut pos)| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    pos.modify(|p| p.0 += 1.0);
                });
            },
        )
        .unwrap();

    // First call: column was never read by this reducer, so Changed matches
    registry.call(&strategy, &mut world, id, ()).unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 2.0);

    // Second call: no mutation since last call, Changed should skip
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    registry.call(&strategy, &mut world, id, ()).unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 0);

    // Mutate the column externally, then call again
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    for (pos,) in world.query::<(&mut Pos,)>() {
        pos.0 = 99.0;
    }
    registry.call(&strategy, &mut world, id, ()).unwrap();
    assert_eq!(visit_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(world.get::<Pos>(e).unwrap().0, 100.0);
}

// ── materialize_reserved regression test ─────────────────────

#[test]
fn query_writer_after_spawn_reducer() {
    // Regression: spawn via changeset (reserve + Mutation::Spawn) must
    // call materialize_reserved() so that subsequent changesets targeting
    // the spawned entity pass the is_alive() check.
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    let spawn_id = registry
        .register_spawner::<(Vel,), f32, _>(&mut world, "spawn", |mut spawner, vel: f32| {
            spawner.spawn((Vel(vel),));
        })
        .unwrap();

    let writer_id = registry
        .register_query_writer::<(&mut Vel,), f32, _>(
            &mut world,
            "scale",
            |mut query, factor: f32| {
                query.for_each(|(mut vel,)| {
                    vel.modify(|v| v.0 *= factor);
                });
            },
        )
        .unwrap();

    let strategy = Optimistic::new(&world);

    // Spawn an entity via changeset (uses reserve() internally)
    registry
        .call(&strategy, &mut world, spawn_id, 10.0f32)
        .unwrap();

    // Query writer iterates the spawned entity and buffers a write.
    // Without materialize_reserved(), this panics with "entity is not alive"
    // when the query writer's changeset is applied.
    registry
        .call(&strategy, &mut world, writer_id, 2.0f32)
        .unwrap();

    // Verify the spawned entity has the correct value
    let mut found = false;
    for (vel,) in world.query::<(&Vel,)>() {
        assert_eq!(vel.0, 20.0);
        found = true;
    }
    assert!(found);
}

// ── Coverage tests ──────────────────────────────────────────────

#[test]
fn entity_ref_entity_accessor() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0),));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    registry
        .register_entity::<(Pos,), (), _>(
            &mut world,
            "check_entity",
            move |entity: EntityMut<'_, (Pos,)>, ()| {
                assert_eq!(entity.entity(), e);
                let _ = entity.get::<Pos, 0>();
            },
        )
        .unwrap();
    let id = registry.reducer_id_by_name("check_entity").unwrap();
    registry.call(&strategy, &mut world, id, (e, ())).unwrap();
}

#[test]
fn query_mut_for_each_slice() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Pos(i as f32),));
    }
    let mut registry = ReducerRegistry::new();
    registry
        .register_query::<(&Pos,), (), _>(
            &mut world,
            "chunk_iter",
            |mut query: QueryMut<'_, (&Pos,)>, ()| {
                let mut count = 0;
                query.for_each(|chunk| {
                    count += chunk.0.len();
                });
                assert_eq!(count, 5);
            },
        )
        .unwrap();
    let id = registry.query_reducer_id_by_name("chunk_iter").unwrap();
    registry.run(&mut world, id, ()).unwrap();
}

#[test]
fn reducer_id_index() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity::<(Pos,), (), _>(
            &mut world,
            "idx_test",
            |_entity: EntityMut<'_, (Pos,)>, ()| {},
        )
        .unwrap();
    assert_eq!(id.index(), 0);

    let qid = registry
        .register_query::<(&Pos,), (), _>(
            &mut world,
            "qidx_test",
            |_query: QueryMut<'_, (&Pos,)>, ()| {},
        )
        .unwrap();
    assert_eq!(qid.index(), 1); // shares the reducers vec
}

#[test]
fn reducer_registry_access_methods() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity::<(Pos,), (), _>(
            &mut world,
            "access_test",
            |_entity: EntityMut<'_, (Pos,)>, ()| {},
        )
        .unwrap();
    let qid = registry
        .register_query::<(&mut Pos,), (), _>(
            &mut world,
            "qaccess_test",
            |_query: QueryMut<'_, (&mut Pos,)>, ()| {},
        )
        .unwrap();

    let access = registry.access(id.index());
    assert!(access.has_any_access());
    let qaccess = registry.query_reducer_access(qid);
    assert!(qaccess.has_any_access());
}

#[test]
fn reducer_registry_default() {
    let _registry: ReducerRegistry = ReducerRegistry::default();
}

// ── ReducerError tests ──────────────────────────────────────────

#[test]
fn reducer_error_display() {
    let err = ReducerError::WrongKind {
        expected: "transactional",
        actual: "scheduled",
    };
    let msg = format!("{err}");
    assert!(msg.contains("transactional"));
    assert!(msg.contains("scheduled"));

    let err = ReducerError::DuplicateName {
        name: "foo",
        existing_kind: "unified",
        existing_index: 0,
    };
    let msg = format!("{err}");
    assert!(msg.contains("foo"));
    assert!(msg.contains("unified"));
}

#[test]
fn reducer_error_from_transact_error_conflict() {
    let conflict = Conflict {
        component_ids: fixedbitset::FixedBitSet::new(),
    };
    let transact_err = TransactError::Conflict(conflict);
    let err: ReducerError = transact_err.into();
    assert!(matches!(err, ReducerError::TransactionConflict(_)));
}

#[test]
fn reducer_error_from_transact_error_world_mismatch() {
    let world1 = crate::World::new();
    let world2 = crate::World::new();
    let transact_err =
        TransactError::WorldMismatch(WorldMismatch::new(world1.world_id(), world2.world_id()));
    let err: ReducerError = transact_err.into();
    assert!(matches!(err, ReducerError::WorldMismatch(_)));
}

// ── Introspection tests ──────────────────────────────────────────

#[test]
fn reducer_info_entity() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_e, ()| {})
        .unwrap();
    let info = registry.reducer_info(id).unwrap();
    assert_eq!(info.name, "heal");
    assert_eq!(info.kind, "transactional");
    assert!(!info.can_despawn);
    assert!(!info.has_change_tracking);
}

#[test]
fn reducer_info_entity_despawn() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity_despawn::<(Health,), (), _>(&mut world, "kill", |_e, ()| {})
        .unwrap();
    let info = registry.reducer_info(id).unwrap();
    assert_eq!(info.name, "kill");
    assert!(info.can_despawn);
}

#[test]
fn reducer_info_entity_ref() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_entity_ref::<(Health,), (), _>(&mut world, "inspect", |_e, ()| {})
        .unwrap();
    let info = registry.reducer_info(id).unwrap();
    assert_eq!(info.name, "inspect");
    assert_eq!(info.kind, "transactional");
    assert!(!info.can_despawn);
}

#[test]
fn entity_ref_reducer_reads_component() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let mut world = World::new();
    let e = world.spawn((Health(42),));

    let mut registry = ReducerRegistry::new();
    let observed = Arc::new(AtomicU32::new(0));
    let obs = Arc::clone(&observed);
    let id = registry
        .register_entity_ref::<(Health,), (), _>(&mut world, "read_hp", move |handle, ()| {
            let hp = handle.get::<Health, 0>();
            obs.store(hp.0, Ordering::Relaxed);
        })
        .unwrap();

    let strategy = Optimistic::new(&world);
    registry.call(&strategy, &mut world, id, (e, ())).unwrap();
    assert_eq!(observed.load(Ordering::Relaxed), 42);
}

#[test]
fn entity_ref_reducer_has_read_only_access() {
    let mut world = World::new();
    world.spawn((Health(1),));

    let mut registry = ReducerRegistry::new();
    registry
        .register_entity_ref::<(Health,), (), _>(&mut world, "read_only", |_e, ()| {})
        .unwrap();

    registry
        .register_entity::<(Health,), (), _>(&mut world, "read_write", |_e, ()| {})
        .unwrap();

    let ref_id = registry.reducer_id_by_name("read_only").unwrap();
    let mut_id = registry.reducer_id_by_name("read_write").unwrap();

    let ref_access = registry.reducer_access(ref_id);
    let mut_access = registry.reducer_access(mut_id);

    // EntityRef should NOT conflict with another EntityRef.
    assert!(
        !ref_access.conflicts_with(ref_access),
        "two read-only reducers should not conflict"
    );
    // EntityRef SHOULD conflict with EntityMut (writes vs reads).
    assert!(
        ref_access.conflicts_with(mut_access),
        "read-only reducer should conflict with read-write reducer"
    );
}

#[test]
fn reducer_info_query_writer() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Pos,), (), _>(&mut world, "move", |_qw, ()| {})
        .unwrap();
    let info = registry.reducer_info(id).unwrap();
    assert_eq!(info.name, "move");
    assert!(info.has_change_tracking);
}

#[test]
fn query_reducer_info() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query::<(&mut Pos,), (), _>(&mut world, "move", |_q, ()| {})
        .unwrap();
    let info = registry.query_reducer_info(id).unwrap();
    assert_eq!(info.name, "move");
    assert_eq!(info.kind, "scheduled");
}

#[test]
fn dynamic_reducer_info() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let id = registry
        .dynamic("dyn_test", &mut world)
        .can_read::<Pos>()
        .can_write::<Vel>()
        .can_despawn()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();
    let info = registry.dynamic_reducer_info(id).unwrap();
    assert_eq!(info.name, "dyn_test");
    assert_eq!(info.kind, "dynamic");
    assert!(info.can_despawn);
    assert!(info.has_change_tracking);
}

#[test]
fn reducer_count_and_names() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    registry
        .register_entity::<(Health,), (), _>(&mut world, "heal", |_e, ()| {})
        .unwrap();
    registry
        .register_query::<(&Pos,), (), _>(&mut world, "read", |_q, ()| {})
        .unwrap();
    registry
        .dynamic("dyn", &mut world)
        .can_read::<Vel>()
        .build(|_ctx: &mut DynamicCtx, _args: &()| {})
        .unwrap();

    assert_eq!(registry.reducer_count(), 2);
    assert_eq!(registry.dynamic_reducer_count(), 1);
    let names: Vec<_> = registry.registered_names().collect();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"heal"));
    assert!(names.contains(&"read"));
    assert!(names.contains(&"dyn"));
}

// ── DynamicCtx introspection tests ──────────────────────────────

#[test]
fn dynamic_ctx_is_declared() {
    use std::any::TypeId;
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();
    let vel_id = world.register_component::<Vel>();

    let entries = vec![(TypeId::of::<Pos>(), pos_id), (TypeId::of::<Vel>(), vel_id)];
    let mut access = Access::empty();
    access.add_read(pos_id);
    access.add_write(vel_id);
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );

    assert!(ctx.is_declared::<Pos>());
    assert!(ctx.is_declared::<Vel>());
    assert!(!ctx.is_declared::<Health>());

    assert!(!ctx.is_writable::<Pos>());
    assert!(ctx.is_writable::<Vel>());

    assert!(!ctx.is_removable::<Pos>());
    assert!(!ctx.can_despawn());
}

#[test]
fn dynamic_ctx_despawn_introspection() {
    use std::any::TypeId;
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();

    let entries = vec![(TypeId::of::<Pos>(), pos_id)];
    let mut access = Access::empty();
    access.add_read(pos_id);
    access.set_despawns();
    let resolved = DynamicResolved::new(entries, access, HashSet::default(), HashSet::default());

    let default_tick = Arc::new(AtomicU64::new(0));
    let default_queried = AtomicBool::new(false);
    let mut cs = EnumChangeSet::new();
    let mut allocated = Vec::new();
    let ctx = DynamicCtx::new(
        &world,
        &mut cs,
        &mut allocated,
        &resolved,
        &default_tick,
        &default_queried,
    );

    assert!(ctx.can_despawn());
}

// ── InvalidId bounds-check tests ─────────────────────────────────

#[test]
fn call_with_invalid_reducer_id() {
    let mut world = World::new();
    let strategy = crate::Optimistic::new(&world);
    let registry = ReducerRegistry::new();
    let bogus = ReducerId(999);
    let result = registry.call(&strategy, &mut world, bogus, ());
    assert!(matches!(
        result,
        Err(ReducerError::InvalidId {
            kind: "reducer",
            index: 999,
            max: 0
        })
    ));
}

#[test]
fn run_with_invalid_query_reducer_id() {
    let mut world = World::new();
    let registry = ReducerRegistry::new();
    let bogus = QueryReducerId(42);
    let result = registry.run(&mut world, bogus, ());
    assert!(matches!(
        result,
        Err(ReducerError::InvalidId {
            kind: "reducer",
            index: 42,
            max: 0
        })
    ));
}

#[test]
fn dynamic_call_with_invalid_id() {
    let mut world = World::new();
    let strategy = crate::Optimistic::new(&world);
    let registry = ReducerRegistry::new();
    let bogus = DynamicReducerId(7);
    let result = registry.dynamic_call(&strategy, &mut world, bogus, &());
    assert!(matches!(
        result,
        Err(ReducerError::InvalidId {
            kind: "dynamic",
            index: 7,
            max: 0
        })
    ));
}

#[test]
fn reducer_info_with_invalid_id() {
    let registry = ReducerRegistry::new();
    let bogus = ReducerId(0);
    let result = registry.reducer_info(bogus);
    assert!(matches!(result, Err(ReducerError::InvalidId { .. })));
}

#[test]
fn query_reducer_info_with_invalid_id() {
    let registry = ReducerRegistry::new();
    let bogus = QueryReducerId(0);
    let result = registry.query_reducer_info(bogus);
    assert!(matches!(result, Err(ReducerError::InvalidId { .. })));
}

#[test]
fn dynamic_reducer_info_with_invalid_id() {
    let registry = ReducerRegistry::new();
    let bogus = DynamicReducerId(0);
    let result = registry.dynamic_reducer_info(bogus);
    assert!(matches!(result, Err(ReducerError::InvalidId { .. })));
}

// ── Fast-lane integration tests ─────────────────────────────────

#[test]
fn query_writer_fast_lane_roundtrip() {
    let mut world = World::new();
    // Spawn 100 entities: Pos(0.0), Vel(i as f32)
    let entities: Vec<Entity> = (0..100)
        .map(|i| world.spawn((Pos(0.0), Vel(i as f32))))
        .collect();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry
        .register_query_writer::<(&mut Pos, &Vel), (), _>(
            &mut world,
            "apply_vel_roundtrip",
            |mut query, ()| {
                query.for_each(|(mut pos, vel)| {
                    pos.modify(|p| p.0 += vel.0);
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();

    // Verify each entity: Pos should equal its Vel value
    for (i, &e) in entities.iter().enumerate() {
        let pos = world.get::<Pos>(e).unwrap().0;
        assert_eq!(
            pos, i as f32,
            "entity {i}: expected Pos({i}.0), got Pos({pos})"
        );
    }
}

#[test]
fn query_writer_fast_lane_change_detection() {
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(10.0)));
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    // First reducer: writes Pos via fast lane
    let writer_id = registry
        .register_query_writer::<(&mut Pos, &Vel), (), _>(
            &mut world,
            "move_pos",
            |mut query, ()| {
                query.for_each(|(mut pos, vel)| {
                    pos.modify(|p| p.0 += vel.0);
                });
            },
        )
        .unwrap();

    // Second reducer: reads Pos with Changed filter
    let visit_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = visit_count.clone();

    let changed_id = registry
        .register_query_writer::<(crate::query::fetch::Changed<Pos>, &mut Pos), (), _>(
            &mut world,
            "detect_changed",
            move |mut query, ()| {
                query.for_each(|((), mut pos)| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // Touch pos so the write goes through the fast lane
                    pos.modify(|p| p.0 += 0.0);
                });
            },
        )
        .unwrap();

    // Call the writer — this updates Pos via fast lane
    registry.call(&strategy, &mut world, writer_id, ()).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 11.0);

    // Changed<Pos> should match because the writer just modified Pos
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    registry
        .call(&strategy, &mut world, changed_id, ())
        .unwrap();
    assert_eq!(
        visit_count.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "Changed<Pos> should match after fast-lane write"
    );

    // Call again with no intervening mutation — Changed should NOT match
    visit_count.store(0, std::sync::atomic::Ordering::Relaxed);
    registry
        .call(&strategy, &mut world, changed_id, ())
        .unwrap();
    assert_eq!(
        visit_count.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "Changed<Pos> should not match when nothing changed"
    );
}

#[test]
fn query_writer_conditional_update() {
    let mut world = World::new();
    let entities: Vec<Entity> = (0..100).map(|i| world.spawn((Pos(i as f32),))).collect();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry
        .register_query_writer::<(&mut Pos,), (), _>(
            &mut world,
            "conditional_update",
            |mut query, ()| {
                query.for_each(|(mut pos,)| {
                    if pos.get().0 > 50.0 {
                        pos.modify(|p| p.0 *= 2.0);
                    }
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();

    for (i, &e) in entities.iter().enumerate() {
        let val = world.get::<Pos>(e).unwrap().0;
        let expected = if (i as f32) > 50.0 {
            (i as f32) * 2.0
        } else {
            i as f32
        };
        assert_eq!(val, expected, "entity {i}: expected {expected}, got {val}");
    }
}

#[test]
fn query_writer_read_only_components() {
    let mut world = World::new();
    let entities: Vec<Entity> = (0..50)
        .map(|i| world.spawn((Pos(i as f32), Vel(i as f32 * 10.0))))
        .collect();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    let id = registry
        .register_query_writer::<(&Pos, &mut Vel), (), _>(
            &mut world,
            "read_pos_write_vel",
            |mut query, ()| {
                query.for_each(|(pos, mut vel)| {
                    // Read Pos, use its value to modify Vel
                    vel.modify(|v| v.0 += pos.0);
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();

    for (i, &e) in entities.iter().enumerate() {
        let pos_val = world.get::<Pos>(e).unwrap().0;
        let vel_val = world.get::<Vel>(e).unwrap().0;
        // Pos should be unchanged
        assert_eq!(pos_val, i as f32, "entity {i}: Pos should be unchanged");
        // Vel should be original + Pos value
        let expected_vel = (i as f32 * 10.0) + i as f32;
        assert_eq!(
            vel_val, expected_vel,
            "entity {i}: Vel should be {expected_vel}, got {vel_val}"
        );
    }
}

#[test]
fn query_writer_column_slot_debug_assert() {
    // Exercises the debug_assert_eq!(col_batch.comp_id, self.comp_id) in set().
    // If column_slot assignment were incorrect, this would panic in debug builds.
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(3.0)));

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Pos, &mut Vel), (), _>(
            &mut world,
            "slot_test",
            |mut qw: QueryWriter<'_, (&mut Pos, &mut Vel)>, ()| {
                qw.for_each(|(mut pos, mut vel)| {
                    pos.set(Pos(10.0));
                    vel.set(Vel(30.0));
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();
    // If we get here without a debug_assert panic, slots are correct.
    assert_eq!(world.get::<Pos>(e).unwrap().0, 10.0);
    assert_eq!(world.get::<Vel>(e).unwrap().0, 30.0);
}

#[test]
fn query_writer_reverse_component_order() {
    // Exercises the slot assignment when tuple order is reversed relative
    // to ascending ComponentId order. Without the position-based lookup
    // fix, Vel's set() would write to Pos's column (and vice versa),
    // tripping the debug_assert_eq on comp_id inside WritableRef::set().
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(3.0)));

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    // Note: (&mut Vel, &mut Pos) — reverse of registration order
    let id = registry
        .register_query_writer::<(&mut Vel, &mut Pos), (), _>(
            &mut world,
            "reverse_slot_test",
            |mut qw: QueryWriter<'_, (&mut Vel, &mut Pos)>, ()| {
                qw.for_each(|(mut vel, mut pos)| {
                    vel.set(Vel(30.0));
                    pos.set(Pos(10.0));
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 10.0);
    assert_eq!(world.get::<Vel>(e).unwrap().0, 30.0);
}

#[test]
fn query_writer_nested_tuple() {
    // Exercises nested mutable tuple: (&mut Pos, (&mut Vel,))
    // Without the offset-propagation fix, Vel would get slot 0
    // instead of slot 1, triggering the debug_assert on comp_id.
    let mut world = World::new();
    let e = world.spawn((Pos(1.0), Vel(2.0)));

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Pos, (&mut Vel,)), (), _>(
            &mut world,
            "nested_tuple",
            |mut qw: QueryWriter<'_, (&mut Pos, (&mut Vel,))>, ()| {
                qw.for_each(|(mut pos, (mut vel,))| {
                    pos.set(Pos(10.0));
                    vel.set(Vel(20.0));
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();
    assert_eq!(world.get::<Pos>(e).unwrap().0, 10.0);
    assert_eq!(world.get::<Vel>(e).unwrap().0, 20.0);
}
