use criterion::{Criterion, criterion_group, criterion_main};
use minkowski_bench::Position;

macro_rules! define_fragments {
    ($($name:ident),*) => {
        $( #[derive(Clone, Copy)] #[expect(dead_code)] struct $name(f32); )*
    };
}

define_fragments!(
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
);

macro_rules! spawn_fragment {
    ($world:expr, $frag:ident) => {
        for i in 0..20 {
            $world.spawn((
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
                $frag(0.0),
            ));
        }
    };
}

fn spawn_fragmented_world() -> minkowski::World {
    let mut world = minkowski::World::new();
    spawn_fragment!(world, A);
    spawn_fragment!(world, B);
    spawn_fragment!(world, C);
    spawn_fragment!(world, D);
    spawn_fragment!(world, E);
    spawn_fragment!(world, F);
    spawn_fragment!(world, G);
    spawn_fragment!(world, H);
    spawn_fragment!(world, I);
    spawn_fragment!(world, J);
    spawn_fragment!(world, K);
    spawn_fragment!(world, L);
    spawn_fragment!(world, M);
    spawn_fragment!(world, N);
    spawn_fragment!(world, O);
    spawn_fragment!(world, P);
    spawn_fragment!(world, Q);
    spawn_fragment!(world, R);
    spawn_fragment!(world, S);
    spawn_fragment!(world, T);
    spawn_fragment!(world, U);
    spawn_fragment!(world, V);
    spawn_fragment!(world, W);
    spawn_fragment!(world, X);
    spawn_fragment!(world, Y);
    spawn_fragment!(world, Z);
    world
}

fn fragmented_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmented_iter");

    group.bench_function("iterate", |b| {
        let mut world = spawn_fragmented_world();
        b.iter(|| {
            for pos in world.query::<&mut Position>() {
                pos.x += 1.0;
            }
        });
    });

    group.finish();
}

criterion_group!(benches, fragmented_iter);
criterion_main!(benches);
