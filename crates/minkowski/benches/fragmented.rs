use criterion::{criterion_group, criterion_main, Criterion};

// 20 different component types for archetype fragmentation
macro_rules! define_components {
    ($($name:ident),*) => { $( #[derive(Clone, Copy)] struct $name(f32); )* };
}
define_components!(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19);

#[derive(Clone, Copy)] struct Position { x: f32, y: f32 }

fn fragmented_iterate_minkowski(c: &mut Criterion) {
    let mut world = minkowski::World::new();
    // Spawn 500 entities across 20 different archetypes, all with Position
    for i in 0..500 {
        match i % 20 {
            0 => { world.spawn((Position { x: i as f32, y: 0.0 }, C0(0.0))); },
            1 => { world.spawn((Position { x: i as f32, y: 0.0 }, C1(0.0))); },
            2 => { world.spawn((Position { x: i as f32, y: 0.0 }, C2(0.0))); },
            3 => { world.spawn((Position { x: i as f32, y: 0.0 }, C3(0.0))); },
            4 => { world.spawn((Position { x: i as f32, y: 0.0 }, C4(0.0))); },
            5 => { world.spawn((Position { x: i as f32, y: 0.0 }, C5(0.0))); },
            6 => { world.spawn((Position { x: i as f32, y: 0.0 }, C6(0.0))); },
            7 => { world.spawn((Position { x: i as f32, y: 0.0 }, C7(0.0))); },
            8 => { world.spawn((Position { x: i as f32, y: 0.0 }, C8(0.0))); },
            9 => { world.spawn((Position { x: i as f32, y: 0.0 }, C9(0.0))); },
            10 => { world.spawn((Position { x: i as f32, y: 0.0 }, C10(0.0))); },
            11 => { world.spawn((Position { x: i as f32, y: 0.0 }, C11(0.0))); },
            12 => { world.spawn((Position { x: i as f32, y: 0.0 }, C12(0.0))); },
            13 => { world.spawn((Position { x: i as f32, y: 0.0 }, C13(0.0))); },
            14 => { world.spawn((Position { x: i as f32, y: 0.0 }, C14(0.0))); },
            15 => { world.spawn((Position { x: i as f32, y: 0.0 }, C15(0.0))); },
            16 => { world.spawn((Position { x: i as f32, y: 0.0 }, C16(0.0))); },
            17 => { world.spawn((Position { x: i as f32, y: 0.0 }, C17(0.0))); },
            18 => { world.spawn((Position { x: i as f32, y: 0.0 }, C18(0.0))); },
            _ => { world.spawn((Position { x: i as f32, y: 0.0 }, C19(0.0))); },
        }
    }

    c.bench_function("minkowski/fragmented_500", |b| {
        b.iter(|| {
            for pos in world.query::<&mut Position>() {
                pos.x += 1.0;
            }
        });
    });
}

criterion_group!(benches, fragmented_iterate_minkowski);
criterion_main!(benches);
