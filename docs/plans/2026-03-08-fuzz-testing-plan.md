# Fuzz Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4 cargo-fuzz targets covering World operations, query reducers, snapshot deserialization, and WAL replay.

**Architecture:** A `fuzz/` directory at the workspace root with its own `Cargo.toml` (cargo-fuzz convention). Four fuzz targets share component types and helper enums. Each target is a standalone binary in `fuzz/fuzz_targets/`.

**Tech Stack:** `cargo-fuzz`, `libfuzzer-sys`, `arbitrary` (with `derive` feature), `rkyv` (for persist component types), `tempfile` (for WAL target).

---

### Task 1: Scaffold fuzz crate and shared types

**Files:**
- Create: `fuzz/Cargo.toml`
- Create: `fuzz/fuzz_targets/common.rs`
- Create: `fuzz/fuzz_targets/fuzz_world_ops.rs` (stub)

**Step 1: Create `fuzz/Cargo.toml`**

```toml
[package]
name = "minkowski-fuzz"
version = "0.0.0"
publish = false
edition = "2021"

[package.metadata.cargo-fuzz]
# cargo-fuzz looks for this section

[dependencies]
libfuzzer-sys = "0.4"
arbitrary = { version = "1", features = ["derive"] }
minkowski = { path = "../crates/minkowski" }
minkowski-persist = { path = "../crates/minkowski-persist" }
rkyv = { version = "0.8", features = ["validation"] }
tempfile = "3"

# Prevent this crate from being included in workspace builds
# (cargo-fuzz manages its own build)
[[bin]]
name = "fuzz_world_ops"
path = "fuzz_targets/fuzz_world_ops.rs"
doc = false

[[bin]]
name = "fuzz_reducers"
path = "fuzz_targets/fuzz_reducers.rs"
doc = false

[[bin]]
name = "fuzz_snapshot_load"
path = "fuzz_targets/fuzz_snapshot_load.rs"
doc = false

[[bin]]
name = "fuzz_wal_replay"
path = "fuzz_targets/fuzz_wal_replay.rs"
doc = false
```

**Step 2: Create `fuzz/fuzz_targets/common.rs` with shared types**

```rust
use arbitrary::Arbitrary;

// --- Component types ---
// Different sizes and alignments to exercise BlobVec layout code.

#[derive(Clone, Copy, Debug, PartialEq, Arbitrary)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct A(pub u32); // 4 bytes, 4-align

#[derive(Clone, Copy, Debug, PartialEq, Arbitrary)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct B(pub u64); // 8 bytes, 8-align

#[derive(Clone, Copy, Debug, PartialEq, Arbitrary)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct C(pub [u8; 3]); // 3 bytes, 1-align (odd size)

#[derive(Clone, Copy, Debug, PartialEq, Arbitrary)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct D(pub f32); // 4 bytes, 4-align

// --- Enums for fuzz-driven selection ---

#[derive(Clone, Debug, Arbitrary)]
pub enum BundleKind {
    JustA(A),
    AB(A, B),
    AC(A, C),
    BD(B, D),
    ABC(A, B, C),
    ABCD(A, B, C, D),
}

#[derive(Clone, Debug, Arbitrary)]
pub enum CompVal {
    A(A),
    B(B),
    C(C),
    D(D),
}

#[derive(Clone, Copy, Debug, Arbitrary)]
pub enum CompKind {
    A,
    B,
    C,
    D,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
pub enum QueryShape {
    RefA,
    RefAB,
    RefABC,
    MutA,
    MutARefB,
    MutARefBC,
    RefB,
    MutD,
}

/// Spawn a bundle into the world based on the fuzzed BundleKind.
pub fn spawn_bundle(world: &mut minkowski::World, kind: &BundleKind) -> minkowski::Entity {
    match kind {
        BundleKind::JustA(a) => world.spawn((*a,)),
        BundleKind::AB(a, b) => world.spawn((*a, *b)),
        BundleKind::AC(a, c) => world.spawn((*a, *c)),
        BundleKind::BD(b, d) => world.spawn((*b, *d)),
        BundleKind::ABC(a, b, c) => world.spawn((*a, *b, *c)),
        BundleKind::ABCD(a, b, c, d) => world.spawn((*a, *b, *c, *d)),
    }
}

/// Run a query of the given shape, returning the number of matched entities.
pub fn run_query(world: &mut minkowski::World, shape: QueryShape) -> usize {
    match shape {
        QueryShape::RefA => world.query::<(&A,)>().count(),
        QueryShape::RefAB => world.query::<(&A, &B)>().count(),
        QueryShape::RefABC => world.query::<(&A, &B, &C)>().count(),
        QueryShape::MutA => world.query::<(&mut A,)>().count(),
        QueryShape::MutARefB => world.query::<(&mut A, &B)>().count(),
        QueryShape::MutARefBC => world.query::<(&mut A, &B, &C)>().count(),
        QueryShape::RefB => world.query::<(&B,)>().count(),
        QueryShape::MutD => world.query::<(&mut D,)>().count(),
    }
}

/// Assert world invariants against a live entity tracker.
pub fn assert_invariants(world: &minkowski::World, live: &[minkowski::Entity]) {
    assert_eq!(
        world.entity_count(),
        live.len(),
        "entity_count mismatch: world has {} but tracker has {}",
        world.entity_count(),
        live.len()
    );
    for (i, &entity) in live.iter().enumerate() {
        assert!(
            world.is_alive(entity),
            "entity at tracker index {i} is not alive"
        );
        assert!(
            world.is_placed(entity),
            "entity at tracker index {i} is not placed"
        );
    }
}
```

**Step 3: Create a stub `fuzz/fuzz_targets/fuzz_world_ops.rs`**

```rust
#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|_data: &[u8]| {
    // stub — will be filled in Task 2
});
```

**Step 4: Verify the fuzz crate compiles**

Run: `cd fuzz && cargo +nightly check 2>&1 | tail -5`

Expected: Compiles without errors. (May warn about unused imports — that's fine.)

**Step 5: Exclude fuzz crate from workspace**

Add `"fuzz"` to the `exclude` list in the root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/minkowski", "crates/minkowski-derive", "crates/minkowski-persist", "crates/minkowski-observe", "crates/minkowski-py", "examples"]
exclude = ["fuzz"]
```

This is required because `cargo-fuzz` manages its own build — workspace builds would fail due to the `libfuzzer-sys` runtime.

**Step 6: Verify workspace still builds**

Run: `cargo check --workspace --exclude minkowski-py 2>&1 | tail -3`

Expected: No errors.

**Step 7: Commit**

```bash
git add fuzz/ Cargo.toml
git commit -m "feat(fuzz): scaffold fuzz crate with shared types and world_ops stub"
```

---

### Task 2: Implement `fuzz_world_ops` target

**Files:**
- Modify: `fuzz/fuzz_targets/fuzz_world_ops.rs`

**Step 1: Write the full fuzz target**

```rust
#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::World;

#[derive(Clone, Debug, Arbitrary)]
enum Op {
    Spawn(BundleKind),
    Despawn(u8),
    Insert(u8, CompVal),
    Remove(u8, CompKind),
    Get(u8, CompKind),
    GetMut(u8, CompKind),
    Query(QueryShape),
    DespawnBatch(Vec<u8>),
}

fuzz_target!(|ops: Vec<Op>| {
    let mut world = World::new();
    let mut live: Vec<minkowski::Entity> = Vec::new();

    for op in &ops {
        match op {
            Op::Spawn(bundle) => {
                let e = spawn_bundle(&mut world, bundle);
                live.push(e);
            }
            Op::Despawn(idx) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live.swap_remove(i);
                    assert!(world.despawn(e));
                }
            }
            Op::Insert(idx, val) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match val {
                        CompVal::A(v) => world.insert(e, *v),
                        CompVal::B(v) => world.insert(e, *v),
                        CompVal::C(v) => world.insert(e, *v),
                        CompVal::D(v) => world.insert(e, *v),
                    }
                }
            }
            Op::Remove(idx, kind) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match kind {
                        CompKind::A => { world.remove::<A>(e); }
                        CompKind::B => { world.remove::<B>(e); }
                        CompKind::C => { world.remove::<C>(e); }
                        CompKind::D => { world.remove::<D>(e); }
                    }
                }
            }
            Op::Get(idx, kind) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match kind {
                        CompKind::A => { world.get::<A>(e); }
                        CompKind::B => { world.get::<B>(e); }
                        CompKind::C => { world.get::<C>(e); }
                        CompKind::D => { world.get::<D>(e); }
                    }
                }
            }
            Op::GetMut(idx, kind) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live[i];
                    match kind {
                        CompKind::A => { world.get_mut::<A>(e); }
                        CompKind::B => { world.get_mut::<B>(e); }
                        CompKind::C => { world.get_mut::<C>(e); }
                        CompKind::D => { world.get_mut::<D>(e); }
                    }
                }
            }
            Op::Query(shape) => {
                run_query(&mut world, *shape);
            }
            Op::DespawnBatch(indices) => {
                if !live.is_empty() {
                    let entities: Vec<_> = indices
                        .iter()
                        .map(|idx| live[*idx as usize % live.len()])
                        .collect();
                    let count = world.despawn_batch(&entities);
                    // Remove despawned entities from tracker (check is_alive)
                    live.retain(|e| world.is_alive(*e));
                    assert_eq!(world.entity_count(), live.len());
                    let _ = count;
                }
            }
        }

        assert_invariants(&world, &live);
    }

    // Cleanup: despawn everything
    for e in &live {
        world.despawn(*e);
    }
    assert_eq!(world.entity_count(), 0);
});
```

**Step 2: Verify it compiles**

Run: `cd fuzz && cargo +nightly check 2>&1 | tail -5`

Expected: Compiles.

**Step 3: Smoke test (10 second run)**

Run: `cd fuzz && cargo +nightly fuzz run fuzz_world_ops -- -max_total_time=10 2>&1 | tail -10`

Expected: Runs without crashing. Output shows `Done N runs` or similar.

**Step 4: Commit**

```bash
git add fuzz/fuzz_targets/fuzz_world_ops.rs
git commit -m "feat(fuzz): implement fuzz_world_ops target"
```

---

### Task 3: Implement `fuzz_reducers` target

**Files:**
- Create: `fuzz/fuzz_targets/fuzz_reducers.rs`

**Step 1: Write the fuzz target**

```rust
#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::World;

#[derive(Clone, Debug, Arbitrary)]
enum ReducerOp {
    QueryMut(QueryShape),
    QueryRef(QueryShape),
    ForEachChunk(QueryShape),
    SpawnMore(BundleKind),
    DespawnSome(u8),
}

/// Run a query using for_each_chunk instead of iterator.
fn run_for_each_chunk(world: &mut World, shape: QueryShape) {
    match shape {
        QueryShape::RefA => {
            world.query::<(&A,)>().for_each_chunk(|(_slices,)| {});
        }
        QueryShape::RefAB => {
            world.query::<(&A, &B)>().for_each_chunk(|(_, _)| {});
        }
        QueryShape::RefABC => {
            world.query::<(&A, &B, &C)>().for_each_chunk(|(_, _, _)| {});
        }
        QueryShape::MutA => {
            world.query::<(&mut A,)>().for_each_chunk(|(a,)| {
                for v in a.iter_mut() {
                    v.0 = v.0.wrapping_add(1);
                }
            });
        }
        QueryShape::MutARefB => {
            world
                .query::<(&mut A, &B)>()
                .for_each_chunk(|(a, _b)| {
                    for v in a.iter_mut() {
                        v.0 = v.0.wrapping_add(1);
                    }
                });
        }
        QueryShape::MutARefBC => {
            world
                .query::<(&mut A, &B, &C)>()
                .for_each_chunk(|(a, _, _)| {
                    for v in a.iter_mut() {
                        v.0 = v.0.wrapping_add(1);
                    }
                });
        }
        QueryShape::RefB => {
            world.query::<(&B,)>().for_each_chunk(|(_,)| {});
        }
        QueryShape::MutD => {
            world.query::<(&mut D,)>().for_each_chunk(|(d,)| {
                for v in d.iter_mut() {
                    v.0 += 1.0;
                }
            });
        }
    }
}

fuzz_target!(|ops: Vec<ReducerOp>| {
    let mut world = World::new();
    let mut live: Vec<minkowski::Entity> = Vec::new();

    for op in &ops {
        match op {
            ReducerOp::QueryMut(shape) => {
                // Use for_each with mutation for mut shapes
                match shape {
                    QueryShape::MutA => {
                        world.query::<(&mut A,)>().for_each(|(a,)| {
                            a.0 = a.0.wrapping_add(1);
                        });
                    }
                    QueryShape::MutARefB => {
                        world.query::<(&mut A, &B)>().for_each(|(a, _)| {
                            a.0 = a.0.wrapping_add(1);
                        });
                    }
                    QueryShape::MutARefBC => {
                        world.query::<(&mut A, &B, &C)>().for_each(|(a, _, _)| {
                            a.0 = a.0.wrapping_add(1);
                        });
                    }
                    QueryShape::MutD => {
                        world.query::<(&mut D,)>().for_each(|(d,)| {
                            d.0 += 1.0;
                        });
                    }
                    _ => {
                        run_query(&mut world, *shape);
                    }
                }
            }
            ReducerOp::QueryRef(shape) => {
                run_query(&mut world, *shape);
            }
            ReducerOp::ForEachChunk(shape) => {
                run_for_each_chunk(&mut world, *shape);
            }
            ReducerOp::SpawnMore(bundle) => {
                let e = spawn_bundle(&mut world, bundle);
                live.push(e);
            }
            ReducerOp::DespawnSome(idx) => {
                if !live.is_empty() {
                    let i = *idx as usize % live.len();
                    let e = live.swap_remove(i);
                    world.despawn(e);
                }
            }
        }

        assert_invariants(&world, &live);
    }
});
```

**Step 2: Verify it compiles**

Run: `cd fuzz && cargo +nightly check 2>&1 | tail -5`

**Step 3: Smoke test (10 second run)**

Run: `cd fuzz && cargo +nightly fuzz run fuzz_reducers -- -max_total_time=10 2>&1 | tail -10`

**Step 4: Commit**

```bash
git add fuzz/fuzz_targets/fuzz_reducers.rs
git commit -m "feat(fuzz): implement fuzz_reducers target"
```

---

### Task 4: Implement `fuzz_snapshot_load` target

**Files:**
- Create: `fuzz/fuzz_targets/fuzz_snapshot_load.rs`

**Step 1: Write the fuzz target**

```rust
#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::World;
use minkowski_persist::{CodecRegistry, Snapshot};

/// Build a CodecRegistry with all 4 component types registered.
fn make_codecs(world: &mut World) -> CodecRegistry {
    let mut codecs = CodecRegistry::new();
    codecs.register::<A>(world);
    codecs.register::<B>(world);
    codecs.register::<C>(world);
    codecs.register::<D>(world);
    codecs
}

#[derive(Clone, Debug, Arbitrary)]
struct RoundTripOps {
    spawns: Vec<BundleKind>,
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mode = data[0];
    let rest = &data[1..];

    match mode {
        // Mode 0: raw bytes — must not panic
        0 => {
            let mut world = World::new();
            let codecs = make_codecs(&mut world);
            let snap = Snapshot::new();
            // Must return Ok or Err, never panic
            let _ = snap.load_from_bytes(rest, &codecs);
        }
        // Mode 1: round-trip — build valid world, save, load, verify
        _ => {
            let Ok(ops) = arbitrary::Unstructured::new(rest).arbitrary::<RoundTripOps>() else {
                return;
            };
            if ops.spawns.is_empty() {
                return;
            }

            let mut world = World::new();
            let mut codecs = make_codecs(&mut world);

            for bundle in &ops.spawns {
                spawn_bundle(&mut world, bundle);
            }

            let snap = Snapshot::new();
            let Ok((header, bytes)) = snap.save_to_bytes(&world, &codecs, 0) else {
                return;
            };
            assert_eq!(header.entity_count, ops.spawns.len());

            // Load into fresh world with fresh codecs
            let mut world2 = World::new();
            let codecs2 = make_codecs(&mut world2);
            let Ok((mut world2, seq)) = snap.load_from_bytes(&bytes, &codecs2) else {
                panic!("round-trip load failed on bytes we just saved");
            };
            assert_eq!(seq, 0);

            // Verify entity counts match
            assert_eq!(world.entity_count(), world2.entity_count());

            // Verify component A values match
            let mut orig: Vec<u32> = world.query::<(&A,)>().map(|(a,)| a.0).collect();
            let mut loaded: Vec<u32> = world2.query::<(&A,)>().map(|(a,)| a.0).collect();
            orig.sort();
            loaded.sort();
            assert_eq!(orig, loaded);
        }
    }
});
```

**Step 2: Verify it compiles**

Run: `cd fuzz && cargo +nightly check 2>&1 | tail -5`

**Step 3: Smoke test (10 second run)**

Run: `cd fuzz && cargo +nightly fuzz run fuzz_snapshot_load -- -max_total_time=10 -max_len=65536 2>&1 | tail -10`

**Step 4: Commit**

```bash
git add fuzz/fuzz_targets/fuzz_snapshot_load.rs
git commit -m "feat(fuzz): implement fuzz_snapshot_load target"
```

---

### Task 5: Implement `fuzz_wal_replay` target

**Files:**
- Create: `fuzz/fuzz_targets/fuzz_wal_replay.rs`

**Step 1: Write the fuzz target**

```rust
#![no_main]

mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use minkowski::{EnumChangeSet, World};
use minkowski_persist::{CodecRegistry, Snapshot, Wal, WalConfig};

fn make_codecs(world: &mut World) -> CodecRegistry {
    let mut codecs = CodecRegistry::new();
    codecs.register::<A>(world);
    codecs.register::<B>(world);
    codecs.register::<C>(world);
    codecs.register::<D>(world);
    codecs
}

#[derive(Clone, Debug, Arbitrary)]
enum WalOp {
    SpawnA(A),
    SpawnAB(A, B),
    InsertB(u8, B),
    InsertD(u8, D),
}

#[derive(Clone, Debug, Arbitrary)]
struct WalRoundTrip {
    ops: Vec<WalOp>,
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mode = data[0];
    let rest = &data[1..];

    let dir = tempfile::tempdir().unwrap();

    match mode {
        // Mode 0: write raw bytes as a WAL file, try to open + replay
        0 => {
            let wal_dir = dir.path().join("raw");
            std::fs::create_dir_all(&wal_dir).unwrap();
            let seg_path = wal_dir.join("00000000.wal");
            std::fs::write(&seg_path, rest).unwrap();

            let mut world = World::new();
            let codecs = make_codecs(&mut world);
            // open + replay must not panic
            if let Ok(mut wal) = Wal::open(&wal_dir, &codecs, WalConfig::default()) {
                let _ = wal.replay_from(0, &mut world, &codecs);
            }
        }
        // Mode 1: round-trip — build valid changesets, append, replay, verify
        _ => {
            let Ok(rt) = arbitrary::Unstructured::new(rest).arbitrary::<WalRoundTrip>() else {
                return;
            };
            if rt.ops.is_empty() {
                return;
            }

            let wal_dir = dir.path().join("rt");
            std::fs::create_dir_all(&wal_dir).unwrap();

            let mut world = World::new();
            let codecs = make_codecs(&mut world);
            let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

            let mut live: Vec<minkowski::Entity> = Vec::new();

            for op in &rt.ops {
                let mut cs = EnumChangeSet::new();
                match op {
                    WalOp::SpawnA(a) => {
                        let e = world.alloc_entity();
                        cs.spawn_bundle(&mut world, e, (*a,));
                        wal.append(&cs, &codecs).unwrap();
                        cs.apply(&mut world);
                        live.push(e);
                    }
                    WalOp::SpawnAB(a, b) => {
                        let e = world.alloc_entity();
                        cs.spawn_bundle(&mut world, e, (*a, *b));
                        wal.append(&cs, &codecs).unwrap();
                        cs.apply(&mut world);
                        live.push(e);
                    }
                    WalOp::InsertB(idx, b) => {
                        if !live.is_empty() {
                            let i = *idx as usize % live.len();
                            let e = live[i];
                            cs.insert::<B>(&mut world, e, *b);
                            wal.append(&cs, &codecs).unwrap();
                            cs.apply(&mut world);
                        }
                    }
                    WalOp::InsertD(idx, d) => {
                        if !live.is_empty() {
                            let i = *idx as usize % live.len();
                            let e = live[i];
                            cs.insert::<D>(&mut world, e, *d);
                            wal.append(&cs, &codecs).unwrap();
                            cs.apply(&mut world);
                        }
                    }
                }
            }

            // Replay into fresh world
            let mut world2 = World::new();
            let codecs2 = make_codecs(&mut world2);
            let mut wal2 = Wal::open(&wal_dir, &codecs2, WalConfig::default()).unwrap();
            let _ = wal2.replay_from(0, &mut world2, &codecs2);

            // Verify entity counts match
            assert_eq!(world.entity_count(), world2.entity_count());
        }
    }
});
```

**Step 2: Verify it compiles**

Run: `cd fuzz && cargo +nightly check 2>&1 | tail -5`

**Step 3: Smoke test (10 second run)**

Run: `cd fuzz && cargo +nightly fuzz run fuzz_wal_replay -- -max_total_time=10 -max_len=65536 2>&1 | tail -10`

**Step 4: Commit**

```bash
git add fuzz/fuzz_targets/fuzz_wal_replay.rs
git commit -m "feat(fuzz): implement fuzz_wal_replay target"
```

---

### Task 6: Update CLAUDE.md and final verification

**Files:**
- Modify: `CLAUDE.md` (add fuzz commands to Build & Test section)

**Step 1: Add fuzz commands to CLAUDE.md Build & Test section**

After the Miri commands, add:

```bash
cargo +nightly fuzz run fuzz_world_ops -- -max_total_time=60     # 1 minute fuzz run
cargo +nightly fuzz run fuzz_reducers -- -max_total_time=60
cargo +nightly fuzz run fuzz_snapshot_load -- -max_total_time=60 -max_len=65536
cargo +nightly fuzz run fuzz_wal_replay -- -max_total_time=60 -max_len=65536
```

**Step 2: Run all 4 targets for 30 seconds each**

Run each:
```bash
cd fuzz
cargo +nightly fuzz run fuzz_world_ops -- -max_total_time=30
cargo +nightly fuzz run fuzz_reducers -- -max_total_time=30
cargo +nightly fuzz run fuzz_snapshot_load -- -max_total_time=30 -max_len=65536
cargo +nightly fuzz run fuzz_wal_replay -- -max_total_time=30 -max_len=65536
```

Expected: All 4 complete without crashes.

**Step 3: Verify workspace tests still pass**

Run: `cargo test -p minkowski`

Expected: All tests pass (no regressions from the fuzz crate).

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add fuzz testing commands to CLAUDE.md"
```
