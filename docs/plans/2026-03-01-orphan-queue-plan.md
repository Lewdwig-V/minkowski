# World-Owned Orphan Queue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the entity ID orphan queue from strategies to World, making drain automatic on any `&mut self` call — engine-guaranteed, no framework cooperation required.

**Architecture:** World owns an `OrphanQueue(Arc<Mutex<Vec<Entity>>>)`. Strategies clone the handle at construction (`Optimistic::new(&world)`). Transaction Drop pushes to the shared queue. A single private `drain_orphans(&mut self)` on World is called at the top of every `&mut self` entry point.

**Tech Stack:** Rust, `parking_lot::Mutex`, `Arc`

---

### Task 1: Add `OrphanQueue` type and `drain_orphans` to World

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Add the `OrphanQueue` type and field to World**

Add `use std::sync::Arc;` and `use parking_lot::Mutex;` to world.rs imports.

Define `OrphanQueue` above the `World` struct:

```rust
/// Shared queue for entity IDs orphaned by aborted transactions.
/// World owns the canonical instance; strategies clone the Arc handle.
#[derive(Clone)]
pub(crate) struct OrphanQueue(pub(crate) Arc<Mutex<Vec<Entity>>>);

impl OrphanQueue {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}
```

Add the field to `World`:

```rust
pub struct World {
    // ... existing fields ...
    pub(crate) orphan_queue: OrphanQueue,
}
```

Initialize in `World::new()`:

```rust
orphan_queue: OrphanQueue::new(),
```

**Step 2: Add `drain_orphans` and `orphan_queue` accessor**

Add these methods to `impl World`:

```rust
/// Drain orphaned entity IDs from aborted transactions.
/// Called automatically at the top of every &mut self entry point.
fn drain_orphans(&mut self) {
    let mut queue = self.orphan_queue.0.lock();
    for entity in queue.drain(..) {
        self.entities.dealloc(entity);
    }
}

/// Clone the orphan queue handle. Strategies capture this at construction
/// so that transaction Drop can push orphaned entity IDs without &mut World.
pub(crate) fn orphan_queue(&self) -> OrphanQueue {
    self.orphan_queue.clone()
}
```

**Step 3: Wire `drain_orphans` into every `&mut self` entry point**

Add `self.drain_orphans();` as the **first line** of each of these methods:

| Method | Line | Notes |
|--------|------|-------|
| `register_component` | 86 | Mutation |
| `alloc_entity` | 92 | Mutation — drain before allocating |
| `dealloc_entity` | 106 | Mutation (pub(crate)) |
| `spawn` | 117 | Mutation |
| `despawn` | 150 | Mutation |
| `get_mut` | 208 | Hands out `&mut T` |
| `query` | 228 | `&mut self` for cache |
| `query_table_raw` | 339 | `&mut self` for table cache |
| `query_table` | 346 | Calls `query_table_raw` (already drained) — **skip** |
| `query_table_mut` | 356 | Calls `query_table_raw` (already drained) — **skip** |
| `insert` | 377 | Mutation |
| `remove` | 459 | Mutation |
| `next_tick` | 74 | pub(crate), called from within other methods — **skip** (would double-drain) |
| `snapshot_column_ticks` | 599 | `&self` — **skip** |
| `check_column_conflicts` | 617 | `&self` — **skip** |
| `query_raw` | 645 | `&self` — **skip** |
| `read_all_components` | 571 | `&self` — **skip** |

Total: 9 methods get `self.drain_orphans();` as first line.

**Step 4: Verify it compiles**

Run: `cargo test -p minkowski --lib --no-run`
Expected: Compiles (tests haven't changed yet, and `OrphanQueue` is unused by strategies so far).

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add OrphanQueue to World with auto-drain on &mut self"
```

---

### Task 2: Migrate strategies from `pending_release` to `OrphanQueue`

**Files:**
- Modify: `crates/minkowski/src/transaction.rs`

**Step 1: Replace `pending_release` with `OrphanQueue` in `Optimistic`**

Change `Optimistic` struct and `new()`:

```rust
pub struct Optimistic {
    next_tx_id: AtomicU64,
    orphan_queue: crate::world::OrphanQueue,
}

impl Optimistic {
    pub fn new(world: &World) -> Self {
        Self {
            next_tx_id: AtomicU64::new(1),
            orphan_queue: world.orphan_queue(),
        }
    }
}
```

Update `Default` impl — remove it (can't impl Default without a World reference).

Update `begin()` — remove the `pending_release` drain loop (World handles it now via `drain_orphans`):

```rust
fn begin<'s>(&'s self, world: &mut World, access: &Access) -> OptimisticTx<'s> {
    let tx_id = self.next_tx_id.fetch_add(1, Ordering::Relaxed);
    let mut accessed = access.reads().clone();
    accessed.union_with(access.writes());
    let read_ticks = world.snapshot_column_ticks(&accessed);
    let archetype_count = world.archetypes.archetypes.len();
    OptimisticTx {
        tx_id,
        strategy: self,
        read_ticks,
        archetype_count,
        spawned_entities: Vec::new(),
        changeset: EnumChangeSet::new(),
    }
}
```

Update `OptimisticTx::Drop` — push to `strategy.orphan_queue` instead of `strategy.pending_release`:

```rust
impl<'s> Drop for OptimisticTx<'s> {
    fn drop(&mut self) {
        if !self.spawned_entities.is_empty() {
            self.strategy
                .orphan_queue
                .0
                .lock()
                .extend(self.spawned_entities.drain(..));
        }
    }
}
```

**Step 2: Same changes for `Pessimistic`**

```rust
pub struct Pessimistic {
    lock_table: Mutex<ColumnLockTable>,
    next_tx_id: AtomicU64,
    orphan_queue: crate::world::OrphanQueue,
}

impl Pessimistic {
    pub fn new(world: &World) -> Self {
        Self {
            lock_table: Mutex::new(ColumnLockTable::new()),
            next_tx_id: AtomicU64::new(1),
            orphan_queue: world.orphan_queue(),
        }
    }
}
```

Remove `Default` impl for `Pessimistic`.

Update `begin()` — remove the drain loop.

Update `PessimisticTx::Drop` — push to `strategy.orphan_queue.0.lock()` instead of `strategy.pending_release.lock()`.

**Step 3: Remove unused `Mutex` import if `pending_release` was the only user**

Check: `ColumnLockTable` also uses `Mutex` in `Pessimistic`, and `OrphanQueue` uses it. Keep the import.

**Step 4: Verify compilation**

Run: `cargo test -p minkowski --lib --no-run`
Expected: Compile errors in tests (they still call `::new()` without args). That's fine — we fix tests in the next task.

**Step 5: Commit**

```bash
git add crates/minkowski/src/transaction.rs
git commit -m "refactor: migrate strategies from pending_release to World's OrphanQueue"
```

---

### Task 3: Update transaction tests

**Files:**
- Modify: `crates/minkowski/src/transaction.rs` (test module)

**Step 1: Update every `Optimistic::new()` → `Optimistic::new(&world)` in tests**

There are 7 call sites in the test module (lines 464, 478, 495, 529, 600, 647 and the implicit ones). Change each to:

```rust
let strategy = Optimistic::new(&world);
```

**Step 2: Update every `Pessimistic::new()` → `Pessimistic::new(&world)` in tests**

There are 4 call sites (lines 544, 557, 573, 586, 624). Change each to:

```rust
let strategy = Pessimistic::new(&world);
```

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (176 tests).

The entity ID leak tests (`optimistic_drop_releases_spawned_entity_ids`, `pessimistic_drop_releases_spawned_entity_ids`, `optimistic_conflict_deallocates_spawned_entities`) now test the World-owned queue path — Drop pushes to the shared queue, and the next `&mut self` call on World drains it. No test logic changes needed — the observable behavior is identical.

**Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean. The removed `Default` impls may trigger clippy if anything still references them — fix if so.

**Step 5: Commit**

```bash
git add crates/minkowski/src/transaction.rs
git commit -m "test: update transaction tests for World-owned OrphanQueue"
```

---

### Task 4: Update examples

**Files:**
- Modify: `examples/examples/transaction.rs`
- Modify: `examples/examples/battle.rs`

**Step 1: Update `transaction.rs` example**

Three call sites:

Line 118: `let strategy = Optimistic::new();` → `let strategy = Optimistic::new(&world);`
Line 138: `let strategy = Optimistic::new();` → `let strategy = Optimistic::new(&world);`
Line 178: `let strategy = Pessimistic::new();` → `let strategy = Pessimistic::new(&world);`

**Step 2: Update `battle.rs` example**

Two call sites:

Line 148: `let strategy = Optimistic::new();` → `let strategy = Optimistic::new(world);`
(Note: `world` is `&mut World` in the function parameter — `&mut World` coerces to `&World` for `new(&World)`)

Line 226: `let strategy = Pessimistic::new();` → `let strategy = Pessimistic::new(world);`

**Step 3: Build examples**

Run: `cargo build -p minkowski-examples --examples`
Expected: All examples compile.

**Step 4: Run examples**

```bash
cargo run -p minkowski-examples --example transaction --release
cargo run -p minkowski-examples --example battle --release
```

Expected: Both run without errors.

**Step 5: Commit**

```bash
git add examples/examples/transaction.rs examples/examples/battle.rs
git commit -m "refactor: update examples for Optimistic::new(&world) / Pessimistic::new(&world)"
```

---

### Task 5: Update CLAUDE.md and push

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update Transaction Semantics section**

In CLAUDE.md, find the Transaction Semantics paragraph and update:

- Add: strategy constructors take `&World` to capture the orphan queue handle
- Add: `World::drain_orphans()` is called automatically on every `&mut self` method
- Remove or update any mention of drain being strategy-coupled

The relevant paragraph currently says:
> `TransactionStrategy` is a trait with one method: `begin(&mut self, world, access) -> Tx`.

Update to reflect:
> `TransactionStrategy` is a trait with one method: `begin(&self, world, access) -> Tx`. Strategies take `&World` at construction to capture a shared `OrphanQueue` handle — entity IDs from aborted transactions are automatically reclaimed by World on any `&mut self` call, engine-guaranteed with no framework cooperation required.

**Step 2: Final verification**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: All pass, clean clippy.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for World-owned orphan queue"
```

**Step 4: Push**

```bash
git push
```

---

### Verification

1. `cargo test -p minkowski --lib` — all tests pass (including entity ID leak tests)
2. `cargo clippy --workspace --all-targets -- -D warnings` — clean
3. `cargo run -p minkowski-examples --example transaction --release` — runs
4. `cargo run -p minkowski-examples --example battle --release` — runs
5. Entity leak tests verify: Drop pushes to queue, next World mutation drains it, generation is bumped
