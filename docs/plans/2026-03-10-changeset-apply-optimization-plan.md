# Changeset Apply Path Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change `EnumChangeSet::apply()` to stop building the reverse changeset and batch tick increments, reducing QueryWriter overhead by ~30-40%.

**Architecture:** Modify `apply()` signature from `fn apply(self, world: &mut World) -> EnumChangeSet` to `fn apply(self, world: &mut World) -> Result<(), ApplyError>`. Remove all reverse changeset construction. Advance tick once per apply call. Update all callers across 5 crates.

**Tech Stack:** Pure Rust refactor, no new dependencies.

---

### Task 1: Add `ApplyError` and change `apply()` signature

**Files:**
- Modify: `crates/minkowski/src/changeset.rs`

**Step 1: Add `ApplyError` enum**

Add this after the existing imports at the top of `changeset.rs`, near the other public types:

```rust
/// Error returned by [`EnumChangeSet::apply`] when a mutation targets
/// an invalid entity.
#[derive(Debug)]
pub enum ApplyError {
    /// Mutation targeted an entity that is no longer alive.
    DeadEntity(Entity),
    /// Spawn targeted an entity already placed in an archetype.
    AlreadyPlaced(Entity),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadEntity(e) => write!(f, "entity {:?} is not alive", e),
            Self::AlreadyPlaced(e) => write!(f, "entity {:?} is already placed", e),
        }
    }
}

impl std::error::Error for ApplyError {}
```

**Step 2: Rewrite `apply()` method**

Change the `apply` method signature and body. Key changes:
- Return `Result<(), ApplyError>` instead of `EnumChangeSet`
- Remove `let mut reverse = EnumChangeSet::new()` and all `reverse.record_*()` calls
- Replace `assert!(world.is_alive(...))` with `if !world.is_alive(...) { return Err(ApplyError::DeadEntity(...)); }`
- Replace `assert!(!world.is_placed(...))` with `if world.is_placed(...) { return Err(ApplyError::AlreadyPlaced(...)); }`
- Call `let tick = world.next_tick()` once at the top, pass `tick` to all handlers
- Return `Ok(())` at the end

In `changeset_insert_raw`: remove the `reverse` parameter and all `reverse.record_insert()` / `reverse.record_remove()` calls. Take `tick` as a parameter instead of calling `world.next_tick()`.

In `changeset_remove_raw`: same — remove `reverse` parameter and reverse recording. Take `tick` parameter.

**Step 3: Export `ApplyError`**

Add `ApplyError` to the `pub use` list in `crates/minkowski/src/lib.rs`.

**Step 4: Verify it compiles (it won't — callers need updating)**

Run: `cargo check -p minkowski 2>&1 | head -30`
Expected: compilation errors from callers that still expect `EnumChangeSet` return.

**Step 5: Commit**

```bash
git add crates/minkowski/src/changeset.rs crates/minkowski/src/lib.rs
git commit -m "feat: apply() returns Result, drops reverse changeset, batches ticks"
```

---

### Task 2: Update internal callers in minkowski crate

**Files:**
- Modify: `crates/minkowski/src/transaction.rs` (lines ~635, ~735)
- Modify: `crates/minkowski/src/reducer.rs` (test lines ~2255, ~3320, ~3507, ~3523, ~3566, ~3592)

**Step 1: Fix transaction.rs**

In `Optimistic::transact()` (~line 635) and `Pessimistic::transact()` (~line 735):

Before:
```rust
Ok(forward) => {
    tx.mark_committed();
    drop(tx);
    forward.apply(world);
    return Ok(value);
}
```

After:
```rust
Ok(forward) => {
    tx.mark_committed();
    drop(tx);
    forward.apply(world).expect("apply after commit");
    return Ok(value);
}
```

Note: `expect` is correct here — if apply fails after a successful commit, it's a programming error (the entity was valid during `try_commit` but died between commit and apply, which can't happen with `&mut World`).

**Step 2: Fix reducer.rs test callers**

All test uses of `let _reverse = cs.apply(&mut world);` become `cs.apply(&mut world).unwrap();`

Search for `_reverse = cs.apply` in reducer.rs and replace each with `.unwrap()`.

**Step 3: Verify minkowski crate compiles**

Run: `cargo check -p minkowski`

**Step 4: Commit**

```bash
git add crates/minkowski/src/transaction.rs crates/minkowski/src/reducer.rs
git commit -m "fix: update transaction and reducer callers for new apply() signature"
```

---

### Task 3: Update minkowski-persist callers

**Files:**
- Modify: `crates/minkowski-persist/src/durable.rs` (~line 91)
- Modify: `crates/minkowski-persist/src/wal.rs` (~line 295, plus tests)
- Modify: `crates/minkowski-persist/src/snapshot.rs` (tests)
- Modify: `crates/minkowski-persist/src/checkpoint.rs` (tests)
- Modify: `crates/minkowski-persist/src/replication.rs` (tests)
- Modify: `crates/minkowski-persist/src/index.rs` (tests)

**Step 1: Fix durable.rs**

Line ~91: `forward.apply(world);` → `forward.apply(world).expect("apply after WAL write");`

This is correct — WAL write succeeded, so the mutation is durable. Apply failure here would be a logic error.

**Step 2: Fix wal.rs**

Line ~295 (WAL replay): `changeset.apply(world);` → `changeset.apply(world).expect("WAL replay apply");`

WAL replay of a valid changeset should never fail — the snapshot restore should have created the entities. If it does fail, it's a corrupt WAL/snapshot, so panicking is correct.

All test uses: `cs.apply(&mut world);` → `cs.apply(&mut world).unwrap();` and `let _reverse = cs.apply(...)` → `cs.apply(...).unwrap();`

**Step 3: Fix remaining persist crate tests**

In `snapshot.rs`, `checkpoint.rs`, `replication.rs`, `index.rs`: change all `cs.apply(&mut world);` to `cs.apply(&mut world).unwrap();` and `let _reverse = cs.apply(...)` to `cs.apply(...).unwrap();`.

**Step 4: Verify persist crate compiles**

Run: `cargo check -p minkowski-persist`

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/
git commit -m "fix: update persist crate callers for new apply() signature"
```

---

### Task 4: Update examples and external tests

**Files:**
- Modify: `examples/examples/life.rs` — remove undo/redo, simplify
- Modify: `examples/examples/nbody.rs`, `examples/examples/replicate.rs`, `examples/examples/flatworm.rs`, `examples/examples/tactical.rs`, `examples/examples/observe.rs`, `examples/examples/boids.rs`
- Modify: `crates/minkowski/tests/changeset_external.rs`
- Modify: `crates/minkowski-bench/benches/simple_insert.rs`
- Modify: `fuzz/fuzz_targets/fuzz_wal_replay.rs`

**Step 1: Fix life.rs**

The `apply_updates` function currently returns `EnumChangeSet` (the reverse). Change it to return nothing:

```rust
fn apply_updates(world: &mut World, grid: &[Entity], updates: &[(usize, bool)]) {
    let mut cs = EnumChangeSet::new();
    for &(i, new_state) in updates {
        cs.insert::<CellState>(world, grid[i], CellState(new_state));
    }
    cs.apply(world).unwrap();
}
```

Remove the undo phase (~line 250-260) that calls `reverse.apply(&mut world)`. Update the main loop to not store the reverse. Update the doc comment that mentions "time-travel demo" and "undo/redo".

**Step 2: Fix other examples**

All examples use `cs.apply(&mut world);` or `cmds.apply(&mut world);` — these all become `.unwrap()` calls:
- `nbody.rs`: `cmds.apply(&mut world);` → `cmds.apply(&mut world).unwrap();`
- `replicate.rs`, `flatworm.rs`, `tactical.rs`, `observe.rs`, `boids.rs`: same pattern

Note: `CommandBuffer::apply` is NOT affected — only `EnumChangeSet::apply`. Check which examples use `EnumChangeSet` vs `CommandBuffer`. `CommandBuffer::apply` has a different signature and is untouched.

Actually, look carefully: `nbody.rs`, `boids.rs`, `flatworm.rs`, `tactical.rs` use `CommandBuffer` (named `cmds`), not `EnumChangeSet`. Only change examples that actually use `EnumChangeSet::apply`.

Grep for `EnumChangeSet` in examples to find which ones need changes. The `cmds.apply` calls are `CommandBuffer::apply` which returns `()` — leave those alone.

**Step 3: Fix changeset_external.rs**

```rust
// Before:
let reverse = cs.apply(&mut world);
let _ = reverse.apply(&mut world);

// After:
cs.apply(&mut world).unwrap();
```

Remove the reverse-apply assertions. Keep the forward-apply assertions that check the world state after apply.

**Step 4: Fix simple_insert.rs benchmark**

`let _reverse = cs.apply(&mut world);` → `cs.apply(&mut world).unwrap();`

**Step 5: Fix fuzz targets**

In `fuzz/fuzz_targets/fuzz_wal_replay.rs`: `cs.apply(&mut world);` → `let _ = cs.apply(&mut world);` (fuzz targets should not unwrap — they test invalid inputs).

**Step 6: Verify everything compiles**

Run: `cargo check --workspace --all-targets`

**Step 7: Commit**

```bash
git add examples/ crates/minkowski/tests/ crates/minkowski-bench/ fuzz/
git commit -m "fix: update examples, tests, and fuzz targets for new apply() signature"
```

---

### Task 5: Rewrite changeset tests that assert on reverse

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (test module, ~lines 1090-1750)

**Step 1: Identify tests that use the reverse**

These tests assert properties of the reverse changeset:
- `apply_spawn_and_reverse_despawns` — applies spawn, checks reverse is despawn, applies reverse
- `apply_despawn_and_reverse_respawns` — applies despawn, checks reverse is spawn
- `apply_insert_new_and_reverse_removes` — applies insert (new component), checks reverse is remove
- `apply_insert_overwrite_and_reverse_restores` — applies insert (overwrite), checks reverse restores old value
- `apply_remove_and_reverse_reinserts` — applies remove, checks reverse is insert
- `apply_empty_changeset` — applies empty, checks reverse is empty
- `apply_batch_despawns_and_reverse` — applies batch despawns, checks reverse
- Various sparse insert/remove reverse tests

**Step 2: Rewrite each test**

For each test, keep the forward-apply assertion (verify the world state changed correctly) but remove the reverse-apply part. The test name should change to remove "reverse" from the name.

For example, `apply_spawn_and_reverse_despawns` becomes `apply_spawn`:
```rust
#[test]
fn apply_spawn() {
    let mut world = World::new();
    let entity = world.alloc_entity();
    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world, entity, (Pos { x: 1.0, y: 2.0 },));
    cs.apply(&mut world).unwrap();
    assert!(world.is_alive(entity));
    assert_eq!(world.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
}
```

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- changeset`
Expected: all changeset tests pass.

**Step 4: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "test: rewrite changeset tests to remove reverse changeset assertions"
```

---

### Task 6: Update CLAUDE.md and CHANGELOG.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CHANGELOG.md`

**Step 1: Update CLAUDE.md**

In the "Mutations" section under EnumChangeSet, update the description. The current text mentions `apply()` returns a reverse `EnumChangeSet` for rollback. Change to describe the new signature.

In the "Three tiers of mutation" table, remove any mention of undo/redo.

**Step 2: Update CHANGELOG.md**

Add a new section at the top for the upcoming version (1.0.4 or whatever the next version is). Key items:
- **Breaking:** `EnumChangeSet::apply()` now returns `Result<(), ApplyError>` instead of a reverse changeset
- **Performance:** ~30-40% faster changeset apply — reverse changeset eliminated, tick increment batched
- **New:** `ApplyError` enum for fallible apply
- **Removed:** Undo/redo via reverse changeset (life example simplified)

**Step 3: Run full test suite**

Run: `cargo test -p minkowski && cargo test -p minkowski-persist`
Run: `cargo clippy --workspace --all-targets -- -D warnings`
Run: `cargo bench -p minkowski-bench -- --test`

**Step 4: Commit**

```bash
git add CLAUDE.md CHANGELOG.md
git commit -m "docs: update CLAUDE.md and CHANGELOG for apply() breaking change"
```

---

### Task 7: Benchmark and validate improvement

**Files:** None (measurement only)

**Step 1: Run the reducer benchmark**

Run: `cargo bench -p minkowski-bench -- reducer`

Record the `query_writer_10k` and `dynamic_for_each_10k` results. Compare against the baseline:
- `query_writer_10k` baseline: ~181 µs (target: ~110-130 µs)
- `dynamic_for_each_10k` baseline: ~313 µs (target: ~200-220 µs)

**Step 2: Run the simple_insert benchmark**

Run: `cargo bench -p minkowski-bench -- simple_insert`

The `changeset` sub-benchmark should also improve since it calls `apply()`.

**Step 3: Report results**

Print before/after comparison. If the improvement is less than expected, investigate — the bottleneck may be elsewhere (arena alloc, clone, entity lookup).
