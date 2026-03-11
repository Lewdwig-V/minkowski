//! Microbenchmarks isolating individual EnumChangeSet costs.
//!
//! These complement the reducer benchmarks by measuring recording and apply
//! phases independently, helping pinpoint optimization targets.
//!
//! Run: cargo bench -p minkowski-bench -- changeset

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use minkowski::{EnumChangeSet, World};
use minkowski_bench::{Position, Velocity};

const N: usize = 10_000;

fn setup_world() -> (World, Vec<minkowski::Entity>) {
    let mut world = World::new();
    let entities: Vec<_> = (0..N)
        .map(|i| {
            world.spawn((
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
                Velocity {
                    dx: 1.0,
                    dy: 0.0,
                    dz: 0.0,
                },
            ))
        })
        .collect();
    (world, entities)
}

/// Measure: build a changeset with 10K insert mutations (arena alloc + vec push).
/// Isolates the recording cost from the apply cost.
fn bench_record_inserts(c: &mut Criterion) {
    let (mut world, entities) = setup_world();

    c.bench_function("changeset/record_10k_inserts", |b| {
        b.iter(|| {
            let mut cs = EnumChangeSet::new();
            for &entity in &entities {
                cs.insert::<Position>(
                    &mut world,
                    entity,
                    Position {
                        x: 99.0,
                        y: 99.0,
                        z: 99.0,
                    },
                );
            }
            cs
        });
    });
}

/// Measure: apply a pre-built changeset of 10K insert-overwrites.
/// Isolates the apply loop cost (entity lookup + memcpy + tick mark).
fn bench_apply_overwrites(c: &mut Criterion) {
    c.bench_function("changeset/apply_10k_overwrites", |b| {
        b.iter_batched(
            || {
                let (mut world, entities) = setup_world();
                let mut cs = EnumChangeSet::new();
                for &entity in &entities {
                    cs.insert::<Position>(
                        &mut world,
                        entity,
                        Position {
                            x: 99.0,
                            y: 99.0,
                            z: 99.0,
                        },
                    );
                }
                (world, cs)
            },
            |(mut world, cs)| {
                cs.apply(&mut world).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

/// Measure: full record + apply round-trip (simulates what QueryWriter does).
fn bench_record_and_apply(c: &mut Criterion) {
    let (mut world, _) = setup_world();

    c.bench_function("changeset/record_apply_10k", |b| {
        b.iter(|| {
            let mut cs = EnumChangeSet::new();
            let mut targets = Vec::with_capacity(N);
            world
                .query::<(minkowski::Entity, &Position, &Velocity)>()
                .for_each(|(entity, pos, vel)| {
                    targets.push((
                        entity,
                        Position {
                            x: pos.x + vel.dx,
                            y: pos.y + vel.dy,
                            z: pos.z + vel.dz,
                        },
                    ));
                });
            for (entity, pos) in targets {
                cs.insert::<Position>(&mut world, entity, pos);
            }
            cs.apply(&mut world).unwrap();
        });
    });
}

/// Measure: EnumChangeSet new + drop cycle (per-transaction allocation cost).
fn bench_changeset_lifecycle(c: &mut Criterion) {
    c.bench_function("changeset/new_drop_empty", |b| {
        b.iter(|| {
            let cs = EnumChangeSet::new();
            std::hint::black_box(cs);
        });
    });
}

criterion_group!(
    benches,
    bench_record_inserts,
    bench_apply_overwrites,
    bench_changeset_lifecycle,
    bench_record_and_apply,
);
criterion_main!(benches);
