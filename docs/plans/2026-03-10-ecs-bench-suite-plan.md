# Standardized ECS Benchmark Suite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `minkowski-bench` crate with 8 standardized benchmark scenarios, migrate existing ad-hoc benches, and remove the hecs dev-dependency.

**Architecture:** New crate `crates/minkowski-bench/` with shared component types in `src/lib.rs` and one criterion bench file per scenario. Depends on `minkowski` and `minkowski-persist`. Replaces `crates/minkowski/benches/`.

**Tech Stack:** Criterion 0.5 with html_reports, rkyv (via minkowski-persist), rayon (via minkowski), tempfile for serialize benchmark.

---

### Task 1: Create minkowski-bench crate scaffold

**Files:**
- Create: `crates/minkowski-bench/Cargo.toml`
- Create: `crates/minkowski-bench/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

**Step 1: Create `crates/minkowski-bench/Cargo.toml`**

```toml
[package]
name = "minkowski-bench"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
minkowski = { path = "../minkowski" }
minkowski-persist = { path = "../minkowski-persist" }
rkyv = { version = "0.8", features = ["alloc", "bytecheck"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
tempfile = "3"
```

Note: No `[[bench]]` entries yet — we'll add them one per task.

**Step 2: Create `crates/minkowski-bench/src/lib.rs`**

This is the shared component types module. All benchmarks import from here.

```rust
use rkyv::{Archive, Deserialize, Serialize};

/// 4x4 matrix — 64 bytes, cache-line sized. Used for heavy_compute (matrix inversion).
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Transform {
    pub matrix: [[f32; 4]; 4],
}

/// 3D position vector — 12 bytes.
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D rotation vector — 12 bytes.
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Rotation {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D velocity vector — 12 bytes.
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Velocity {
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
}

/// Spawn a world with `n` entities, each with (Transform, Position, Rotation, Velocity).
pub fn spawn_world(n: usize) -> minkowski::World {
    let mut world = minkowski::World::new();
    for i in 0..n {
        let f = i as f32;
        world.spawn((
            Transform {
                matrix: [[f, 0.0, 0.0, 0.0],
                         [0.0, f, 0.0, 0.0],
                         [0.0, 0.0, f, 0.0],
                         [0.0, 0.0, 0.0, 1.0]],
            },
            Position { x: f, y: f, z: f },
            Rotation { x: 0.0, y: 0.0, z: 0.0 },
            Velocity { dx: 1.0, dy: 1.0, dz: 1.0 },
        ));
    }
    world
}

/// Register all 4 component types with the codec registry.
pub fn register_codecs(codecs: &mut minkowski_persist::CodecRegistry, world: &mut minkowski::World) {
    codecs.register::<Transform>(world);
    codecs.register::<Position>(world);
    codecs.register::<Rotation>(world);
    codecs.register::<Velocity>(world);
}

/// Naive 4x4 matrix inversion via cofactor expansion.
/// Not optimized — the point is ~200 FLOPs of real work per entity.
pub fn invert_4x4(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] = m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
              - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
              + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]);

    inv[0][1] = -(m[0][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
               - m[0][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
               + m[0][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]));

    inv[0][2] = m[0][1] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
              - m[0][2] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
              + m[0][3] * (m[1][1] * m[3][2] - m[1][2] * m[3][1]);

    inv[0][3] = -(m[0][1] * (m[1][2] * m[2][3] - m[1][3] * m[2][2])
               - m[0][2] * (m[1][1] * m[2][3] - m[1][3] * m[2][1])
               + m[0][3] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]));

    let det = m[0][0] * inv[0][0] + m[0][1] * (
        -(m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
        + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])))
        + m[0][2] * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
        - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
        + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
        + m[0][3] * (-(m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
        - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
        + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])));

    if det.abs() < 1e-10 {
        return *m; // singular — return identity-ish
    }

    // For the benchmark we only need the first row to be correct —
    // the point is compute volume, not a production-grade inverse.
    let inv_det = 1.0 / det;
    for r in 0..4 {
        for c in 0..4 {
            inv[r][c] *= inv_det;
        }
    }
    inv
}
```

**Step 3: Add to workspace**

In `Cargo.toml` (workspace root), add `"crates/minkowski-bench"` to the `members` list.

**Step 4: Verify it compiles**

Run: `cargo check -p minkowski-bench`
Expected: compiles with no errors.

**Step 5: Commit**

```bash
git add crates/minkowski-bench/ Cargo.toml
git commit -m "feat: scaffold minkowski-bench crate with shared component types"
```

---

### Task 2: simple_insert benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/simple_insert.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

Append to `crates/minkowski-bench/Cargo.toml`:

```toml
[[bench]]
name = "simple_insert"
harness = false
```

**Step 2: Create `benches/simple_insert.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::{Position, Rotation, Transform, Velocity};

fn simple_insert_batch(c: &mut Criterion) {
    c.bench_function("simple_insert/batch", |b| {
        b.iter(|| {
            let mut world = minkowski::World::new();
            for i in 0..10_000 {
                let f = i as f32;
                world.spawn((
                    Transform {
                        matrix: [
                            [f, 0.0, 0.0, 0.0],
                            [0.0, f, 0.0, 0.0],
                            [0.0, 0.0, f, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    Position { x: f, y: f, z: f },
                    Rotation {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Velocity {
                        dx: 1.0,
                        dy: 1.0,
                        dz: 1.0,
                    },
                ));
            }
        });
    });
}

fn simple_insert_changeset(c: &mut Criterion) {
    c.bench_function("simple_insert/changeset", |b| {
        b.iter(|| {
            let mut world = minkowski::World::new();
            let mut cs = minkowski::EnumChangeSet::new();
            for i in 0..10_000 {
                let f = i as f32;
                cs.spawn_bundle(
                    &mut world,
                    (
                        Transform {
                            matrix: [
                                [f, 0.0, 0.0, 0.0],
                                [0.0, f, 0.0, 0.0],
                                [0.0, 0.0, f, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                        Position { x: f, y: f, z: f },
                        Rotation {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        Velocity {
                            dx: 1.0,
                            dy: 1.0,
                            dz: 1.0,
                        },
                    ),
                );
            }
            cs.apply(&mut world);
        });
    });
}

criterion_group!(benches, simple_insert_batch, simple_insert_changeset);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- simple_insert --test`
Expected: benchmark compiles and runs briefly (--test mode).

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/simple_insert.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add simple_insert scenario (batch + changeset)"
```

---

### Task 3: simple_iter benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/simple_iter.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "simple_iter"
harness = false
```

**Step 2: Create `benches/simple_iter.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::{spawn_world, Position, Velocity};

fn simple_iter_for_each(c: &mut Criterion) {
    let mut world = spawn_world(10_000);

    c.bench_function("simple_iter/for_each", |b| {
        b.iter(|| {
            for (vel, pos) in world.query::<(&mut Velocity, &Position)>() {
                vel.dx += pos.x * 0.1;
                vel.dy += pos.y * 0.1;
                vel.dz += pos.z * 0.1;
            }
        });
    });
}

fn simple_iter_for_each_chunk(c: &mut Criterion) {
    let mut world = spawn_world(10_000);

    c.bench_function("simple_iter/for_each_chunk", |b| {
        b.iter(|| {
            world
                .query::<(&mut Velocity, &Position)>()
                .for_each_chunk(|(velocities, positions)| {
                    for i in 0..velocities.len() {
                        velocities[i].dx += positions[i].x * 0.1;
                        velocities[i].dy += positions[i].y * 0.1;
                        velocities[i].dz += positions[i].z * 0.1;
                    }
                });
        });
    });
}

fn simple_iter_par_for_each(c: &mut Criterion) {
    let mut world = spawn_world(10_000);

    c.bench_function("simple_iter/par_for_each", |b| {
        b.iter(|| {
            world
                .query::<(&mut Velocity, &Position)>()
                .par_for_each(|(vel, pos)| {
                    vel.dx += pos.x * 0.1;
                    vel.dy += pos.y * 0.1;
                    vel.dz += pos.z * 0.1;
                });
        });
    });
}

criterion_group!(
    benches,
    simple_iter_for_each,
    simple_iter_for_each_chunk,
    simple_iter_par_for_each,
);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- simple_iter --test`

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/simple_iter.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add simple_iter scenario (for_each, chunk, parallel)"
```

---

### Task 4: fragmented_iter benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/fragmented_iter.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "fragmented_iter"
harness = false
```

**Step 2: Create `benches/fragmented_iter.rs`**

Uses a macro to define 26 fragment types (A–Z) matching ecs_bench_suite.

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::Position;

macro_rules! define_fragments {
    ($($name:ident),*) => {
        $( #[derive(Clone, Copy)] struct $name(f32); )*
    };
}

define_fragments!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);

fn spawn_fragmented() -> minkowski::World {
    let mut world = minkowski::World::new();
    // 20 entities per fragment type, each also has Position
    macro_rules! spawn_fragment {
        ($world:expr, $frag:ident) => {
            for i in 0..20 {
                $world.spawn((Position { x: i as f32, y: 0.0, z: 0.0 }, $frag(0.0)));
            }
        };
    }
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
    let mut world = spawn_fragmented();

    c.bench_function("fragmented_iter/520_entities_26_archetypes", |b| {
        b.iter(|| {
            for pos in world.query::<&mut Position>() {
                pos.x += 1.0;
            }
        });
    });
}

criterion_group!(benches, fragmented_iter);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- fragmented_iter --test`

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/fragmented_iter.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add fragmented_iter scenario (26 archetypes, 520 entities)"
```

---

### Task 5: heavy_compute benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/heavy_compute.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "heavy_compute"
harness = false
```

**Step 2: Create `benches/heavy_compute.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::{invert_4x4, Transform};

fn spawn_transforms(n: usize) -> minkowski::World {
    let mut world = minkowski::World::new();
    for i in 0..n {
        let f = (i + 1) as f32;
        world.spawn((Transform {
            matrix: [
                [f, 0.1, 0.2, 0.3],
                [0.0, f, 0.1, 0.2],
                [0.0, 0.0, f, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },));
    }
    world
}

fn heavy_compute_sequential(c: &mut Criterion) {
    let mut world = spawn_transforms(1_000);

    c.bench_function("heavy_compute/sequential", |b| {
        b.iter(|| {
            world
                .query::<&mut Transform>()
                .for_each_chunk(|transforms| {
                    for t in transforms.iter_mut() {
                        t.matrix = invert_4x4(&t.matrix);
                    }
                });
        });
    });
}

fn heavy_compute_parallel(c: &mut Criterion) {
    let mut world = spawn_transforms(1_000);

    c.bench_function("heavy_compute/parallel", |b| {
        b.iter(|| {
            world.query::<&mut Transform>().par_for_each(|t| {
                t.matrix = invert_4x4(&t.matrix);
            });
        });
    });
}

criterion_group!(benches, heavy_compute_sequential, heavy_compute_parallel);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- heavy_compute --test`

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/heavy_compute.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add heavy_compute scenario (4x4 matrix inversion, seq + parallel)"
```

---

### Task 6: add_remove benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/add_remove.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "add_remove"
harness = false
```

**Step 2: Create `benches/add_remove.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::{Position, Velocity};

fn add_remove(c: &mut Criterion) {
    c.bench_function("add_remove/10k", |b| {
        b.iter_batched(
            || {
                let mut world = minkowski::World::new();
                let entities: Vec<_> = (0..10_000)
                    .map(|i| {
                        world.spawn((Position {
                            x: i as f32,
                            y: 0.0,
                            z: 0.0,
                        },))
                    })
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for &e in &entities {
                    world.insert(e, (Velocity { dx: 1.0, dy: 1.0, dz: 1.0 },));
                }
                for &e in &entities {
                    world.remove::<Velocity>(e);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, add_remove);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- add_remove --test`

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/add_remove.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add add_remove scenario (10K entities, archetype migration)"
```

---

### Task 7: schedule benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/schedule.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "schedule"
harness = false
```

**Step 2: Create `benches/schedule.rs`**

Registers 5 non-conflicting `QueryMut` reducers and runs them sequentially. This measures Access conflict checking + dispatch overhead. Check `crates/minkowski/src/reducer.rs` for the `register_query` and `run` signatures.

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski::{Optimistic, QueryMut, ReducerRegistry};
use minkowski_bench::{spawn_world, Position, Rotation, Transform, Velocity};

fn schedule(c: &mut Criterion) {
    let mut world = spawn_world(10_000);
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();

    // 5 non-conflicting systems — each writes to a different component
    let id0 = registry
        .register_query::<&mut Position, (), _>(&mut world, "sys_pos", |mut q: QueryMut<'_, &mut Position>, _: ()| {
            q.for_each(|pos| pos.x += 1.0);
        })
        .unwrap();

    let id1 = registry
        .register_query::<&mut Velocity, (), _>(&mut world, "sys_vel", |mut q: QueryMut<'_, &mut Velocity>, _: ()| {
            q.for_each(|vel| vel.dx += 0.1);
        })
        .unwrap();

    let id2 = registry
        .register_query::<&mut Rotation, (), _>(&mut world, "sys_rot", |mut q: QueryMut<'_, &mut Rotation>, _: ()| {
            q.for_each(|rot| rot.x += 0.01);
        })
        .unwrap();

    let id3 = registry
        .register_query::<&mut Transform, (), _>(&mut world, "sys_transform", |mut q: QueryMut<'_, &mut Transform>, _: ()| {
            q.for_each(|t| t.matrix[0][0] += 0.001);
        })
        .unwrap();

    // 5th system reads Position (no conflict with sys_pos only if we read,
    // but QueryMut on &Position is read-only via the query)
    let id4 = registry
        .register_query::<&Position, (), _>(&mut world, "sys_read_pos", |mut q: QueryMut<'_, &Position>, _: ()| {
            q.for_each(|_pos| {});
        })
        .unwrap();

    let ids = [id0, id1, id2, id3, id4];

    c.bench_function("schedule/5_systems_10k", |b| {
        b.iter(|| {
            for &id in &ids {
                registry.run(&mut world, id, ()).unwrap();
            }
        });
    });

    drop(strategy);
}

criterion_group!(benches, schedule);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- schedule --test`

If `QueryMut<'_, &Position>` doesn't compile (QueryMut may require `&mut` components), change `id4` to use `register_query::<(&Position,), (), _>` or a different read-only component. Check `crates/minkowski/src/reducer.rs` for exact bounds.

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/schedule.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add schedule scenario (5 reducers, Access + dispatch overhead)"
```

---

### Task 8: serialize benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/serialize.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "serialize"
harness = false
```

**Step 2: Create `benches/serialize.rs`**

Reference `crates/minkowski-persist/benches/persist.rs` for API patterns. The new version uses the standardized component types and adds WAL replay.

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use minkowski_bench::{register_codecs, spawn_world, Position};
use minkowski_persist::{CodecRegistry, Snapshot, Wal, WalConfig};

fn bench_snapshot_save(c: &mut Criterion) {
    let mut world = spawn_world(1_000);
    let mut codecs = CodecRegistry::new();
    register_codecs(&mut codecs, &mut world);
    let dir = tempfile::tempdir().unwrap();

    c.bench_function("serialize/snapshot_save", |b| {
        let path = dir.path().join("bench.snap");
        b.iter(|| {
            let snap = Snapshot::new();
            snap.save(&path, &world, &codecs, 0).unwrap();
        });
    });
}

fn bench_snapshot_load(c: &mut Criterion) {
    let mut world = spawn_world(1_000);
    let mut codecs = CodecRegistry::new();
    register_codecs(&mut codecs, &mut world);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.snap");
    Snapshot::new().save(&path, &world, &codecs, 0).unwrap();

    c.bench_function("serialize/snapshot_load", |b| {
        b.iter(|| {
            let (_world, _seq) = Snapshot::new().load(&path, &codecs).unwrap();
        });
    });
}

fn bench_wal_append(c: &mut Criterion) {
    let mut world = spawn_world(1_000);
    let mut codecs = CodecRegistry::new();
    register_codecs(&mut codecs, &mut world);
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("wal");
    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

    // Grab an entity to mutate
    let entity = world.query::<minkowski::Entity>().next().unwrap();
    let mut cs = minkowski::EnumChangeSet::new();
    cs.insert::<Position>(&mut world, entity, Position { x: 999.0, y: 999.0, z: 999.0 });

    c.bench_function("serialize/wal_append", |b| {
        b.iter(|| {
            wal.append(&cs, &codecs).unwrap();
        });
    });
}

fn bench_wal_replay(c: &mut Criterion) {
    let mut world = spawn_world(1_000);
    let mut codecs = CodecRegistry::new();
    register_codecs(&mut codecs, &mut world);

    // Write 1K mutations to WAL
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("wal");
    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

    let entities: Vec<_> = world.query::<minkowski::Entity>().collect();
    for &entity in &entities {
        let mut cs = minkowski::EnumChangeSet::new();
        cs.insert::<Position>(&mut world, entity, Position { x: 0.0, y: 0.0, z: 0.0 });
        wal.append(&cs, &codecs).unwrap();
    }

    c.bench_function("serialize/wal_replay", |b| {
        b.iter_batched(
            || {
                let mut fresh_world = minkowski::World::new();
                // Need to register component types so replay can route them
                let mut fresh_codecs = CodecRegistry::new();
                register_codecs(&mut fresh_codecs, &mut fresh_world);
                let replay_wal = Wal::open(&wal_dir, &fresh_codecs, WalConfig::default()).unwrap();
                (fresh_world, fresh_codecs, replay_wal)
            },
            |(mut fresh_world, fresh_codecs, mut replay_wal)| {
                replay_wal.replay(&mut fresh_world, &fresh_codecs).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_snapshot_save,
    bench_snapshot_load,
    bench_wal_append,
    bench_wal_replay,
);
criterion_main!(benches);
```

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- serialize --test`

Note: WAL replay benchmark may need adjustment — `replay` might require entities to exist in the world first (spawned from snapshot). If so, use `iter_batched` with a snapshot-restored world as setup. Check the `replay` signature and its behavior with missing entities.

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/serialize.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: add serialize scenario (snapshot save/load, WAL append/replay)"
```

---

### Task 9: Migrate reducer benchmark

**Files:**
- Create: `crates/minkowski-bench/benches/reducer.rs` (copy from `crates/minkowski/benches/reducer.rs`)
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]`)

**Step 1: Add bench entry to Cargo.toml**

```toml
[[bench]]
name = "reducer"
harness = false
```

**Step 2: Copy existing reducer bench**

Copy `crates/minkowski/benches/reducer.rs` to `crates/minkowski-bench/benches/reducer.rs`. The only change needed: it already uses `minkowski::` imports directly, which will work since `minkowski-bench` depends on `minkowski`.

Read the source file and copy it verbatim. No changes needed unless it references types only available as dev-dependencies of the original crate.

**Step 3: Verify**

Run: `cargo bench -p minkowski-bench -- reducer --test`

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/reducer.rs crates/minkowski-bench/Cargo.toml
git commit -m "bench: migrate reducer benchmark to minkowski-bench"
```

---

### Task 10: Remove old benchmarks and hecs dependency

**Files:**
- Delete: `crates/minkowski/benches/spawn.rs`
- Delete: `crates/minkowski/benches/iterate.rs`
- Delete: `crates/minkowski/benches/parallel.rs`
- Delete: `crates/minkowski/benches/fragmented.rs`
- Delete: `crates/minkowski/benches/add_remove.rs`
- Delete: `crates/minkowski/benches/reducer.rs`
- Modify: `crates/minkowski/Cargo.toml` (remove `[[bench]]` entries and `hecs` dev-dependency)

**Step 1: Delete old bench files**

```bash
rm crates/minkowski/benches/spawn.rs
rm crates/minkowski/benches/iterate.rs
rm crates/minkowski/benches/parallel.rs
rm crates/minkowski/benches/fragmented.rs
rm crates/minkowski/benches/add_remove.rs
rm crates/minkowski/benches/reducer.rs
rmdir crates/minkowski/benches
```

**Step 2: Edit `crates/minkowski/Cargo.toml`**

Remove these sections:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
hecs = "0.10"

[[bench]]
name = "spawn"
harness = false

[[bench]]
name = "iterate"
harness = false

[[bench]]
name = "parallel"
harness = false

[[bench]]
name = "add_remove"
harness = false

[[bench]]
name = "fragmented"
harness = false

[[bench]]
name = "reducer"
harness = false
```

**Step 3: Verify everything still compiles and tests pass**

Run: `cargo check --workspace --all-targets && cargo test -p minkowski --lib`

**Step 4: Verify new benchmarks still work**

Run: `cargo bench -p minkowski-bench -- --test`
Expected: all 8 bench binaries compile and run in test mode.

**Step 5: Commit**

```bash
git add -A crates/minkowski/benches/ crates/minkowski/Cargo.toml
git commit -m "cleanup: remove old ad-hoc benchmarks and hecs dev-dependency"
```

---

### Task 11: Update CLAUDE.md and docs

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update benchmark commands in CLAUDE.md**

Replace the existing benchmark section:

```
cargo bench -p minkowski               # All criterion benchmarks (spawn, iterate, parallel, fragmented, add_remove, reducer)
cargo bench -p minkowski -- spawn      # Single benchmark
cargo bench -p minkowski-persist       # Persistence benchmarks (snapshot save/load/zero-copy, WAL append)
```

With:

```
cargo bench -p minkowski-bench                      # All standardized benchmarks (simple_insert, simple_iter, fragmented_iter, heavy_compute, add_remove, schedule, serialize, reducer)
cargo bench -p minkowski-bench -- simple_iter       # Single scenario
cargo bench -p minkowski-bench -- simple_iter/par   # Sub-benchmark filter
cargo bench -p minkowski-persist                    # Persistence-only benchmarks (snapshot save/load, WAL append)
```

Also update the Dependencies table — remove `hecs (dev)` row.

**Step 2: Verify**

Run: `cargo bench -p minkowski-bench -- --test`

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md benchmark commands for minkowski-bench"
```
