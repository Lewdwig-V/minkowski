# Transaction Semantics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Context:** See `docs/plans/2026-03-01-transaction-semantics-design.md` for full design rationale.

**Goal:** Add a `TransactionStrategy` trait with three implementations — Sequential (zero-cost), Optimistic (tick validation), and Pessimistic (cooperative column locks) — so framework authors can choose the concurrency model that fits their workload.

**Architecture:** The trait defines `begin()` → `Tx` (associated type). Each `Tx` wraps `&mut World` + strategy-specific state (nothing / read-ticks / lock-set). Reads go through `world.query()`. Writes buffer into `EnumChangeSet`. Commit validates (optimistic) or just applies (sequential/pessimistic). Drop = abort (RAII).

**Tech Stack:** Rust, `fixedbitset` (existing dependency), `EnumChangeSet` (existing)

---

### Task 1: TransactionStrategy trait + Conflict struct + Sequential impl

**Files:**
- Create: `crates/minkowski/src/transaction.rs`
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Create `transaction.rs` with the trait, Conflict, Sequential, and SequentialTx**

```rust
use fixedbitset::FixedBitSet;

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
use crate::world::World;

/// Conflict information returned when a transaction commit fails.
pub struct Conflict {
    /// Which component columns had conflicting concurrent modifications.
    pub component_ids: FixedBitSet,
}

/// Strategy for transaction concurrency control.
///
/// The trait provides `begin()` which returns a strategy-specific transaction
/// object. The caller reads and writes through the transaction, then calls
/// `commit()` to apply changes atomically — or drops the transaction to abort.
///
/// Three built-in strategies:
/// - [`Sequential`] — zero-cost passthrough, no conflict detection
/// - [`Optimistic`] — live reads with tick-based validation at commit
/// - [`Pessimistic`] — cooperative column locks, guaranteed commit success
pub trait TransactionStrategy {
    /// The transaction object returned by `begin()`.
    type Tx<'w>;

    /// Begin a transaction. `access` declares which components will be
    /// read and written — used by Optimistic for tick snapshotting and
    /// by Pessimistic for lock acquisition.
    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> Self::Tx<'w>;
}

// ── Sequential ──────────────────────────────────────────────────────

/// Zero-cost transaction strategy. All operations delegate directly to World.
/// Commit always succeeds. No read-set, no changeset buffering, no validation.
///
/// Use when systems run sequentially and no conflict detection is needed.
pub struct Sequential;

impl TransactionStrategy for Sequential {
    type Tx<'w> = SequentialTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, _access: &Access) -> SequentialTx<'w> {
        SequentialTx { world }
    }
}

/// Transaction object for the [`Sequential`] strategy.
/// Transparent wrapper around `&mut World`.
pub struct SequentialTx<'w> {
    world: &'w mut World,
}

impl<'w> SequentialTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        self.world.spawn(bundle)
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.world.despawn(entity)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.world.insert(entity, component);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.world.remove::<T>(entity)
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.world.get_mut::<T>(entity)
    }

    /// Commit the transaction. Always succeeds for Sequential.
    /// Returns an empty reverse changeset (mutations went directly to World).
    pub fn commit(self) -> Result<EnumChangeSet, Conflict> {
        Ok(EnumChangeSet::new())
    }
}
```

**Step 2: Add module and re-exports in `lib.rs`**

Add `pub mod transaction;` alphabetically, and:

```rust
pub use transaction::{Conflict, Sequential, SequentialTx, TransactionStrategy};
```

**Step 3: Write inline tests**

Append to `transaction.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);

    #[test]
    fn sequential_query_reads_spawned_entities() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &Vel)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn sequential_mutation_is_immediate() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn((Pos(42.0),));
        // Mutation is immediate — visible through query
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        let _ = tx.commit();
    }

    #[test]
    fn sequential_commit_always_ok() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn((Pos(1.0),));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn sequential_drop_is_noop() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let mut strategy = Sequential;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.spawn((Pos(1.0),));
            // drop without commit — but mutations already applied
        }
        // Entity is in world because Sequential writes directly
        assert_eq!(world.query::<(&Pos,)>().count(), 1);
    }
}
```

**Step 4: Run tests**

```
cargo test -p minkowski --lib -- transaction
```

**Step 5: Run clippy**

```
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 6: Commit**

```
git add crates/minkowski/src/transaction.rs crates/minkowski/src/lib.rs
git commit -m "feat: TransactionStrategy trait + Sequential impl"
```

---

### Task 2: World tick-snapshotting helpers for Optimistic

**Files:**
- Modify: `crates/minkowski/src/world.rs`

The Optimistic strategy needs to snapshot column ticks at begin and check if they advanced at commit. These are `pub(crate)` helpers on World — no public API growth.

**Step 1: Add `snapshot_column_ticks` and `columns_changed_since` to World**

At the end of the `impl World` block (before the closing `}`), add:

```rust
    /// Snapshot the `changed_tick` of every column matching the given component
    /// bitset. Returns a Vec of (ArchetypeId, ComponentId, Tick) triples.
    /// Used by OptimisticTx for read-set validation.
    pub(crate) fn snapshot_column_ticks(
        &self,
        component_ids: &FixedBitSet,
    ) -> Vec<(usize, ComponentId, crate::tick::Tick)> {
        let mut ticks = Vec::new();
        for arch in &self.archetypes.archetypes {
            for comp_id in component_ids.ones() {
                if let Some(&col_idx) = arch.component_index.get(&comp_id) {
                    ticks.push((arch.id.0, comp_id, arch.columns[col_idx].changed_tick));
                }
            }
        }
        ticks
    }

    /// Check if any column in the snapshot has been modified since the
    /// recorded tick. Returns a FixedBitSet of conflicting ComponentIds.
    pub(crate) fn check_column_conflicts(
        &self,
        snapshot: &[(usize, ComponentId, crate::tick::Tick)],
    ) -> FixedBitSet {
        let mut conflicts = FixedBitSet::new();
        for &(arch_idx, comp_id, begin_tick) in snapshot {
            if let Some(arch) = self.archetypes.archetypes.get(arch_idx) {
                if let Some(&col_idx) = arch.component_index.get(&comp_id) {
                    if arch.columns[col_idx].changed_tick.is_newer_than(begin_tick) {
                        conflicts.grow(comp_id + 1);
                        conflicts.insert(comp_id);
                    }
                }
            }
        }
        conflicts
    }
```

**Step 2: Write tests in world.rs tests module**

```rust
    #[test]
    fn snapshot_column_ticks_captures_current_state() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut bits = FixedBitSet::with_capacity(pos_id + 1);
        bits.insert(pos_id);
        let snap = world.snapshot_column_ticks(&bits);
        assert!(!snap.is_empty());
    }

    #[test]
    fn check_column_conflicts_detects_mutation() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut bits = FixedBitSet::with_capacity(pos_id + 1);
        bits.insert(pos_id);

        let snap = world.snapshot_column_ticks(&bits);

        // Mutate through query — advances tick
        for pos in world.query::<(&mut Pos,)>() {
            pos.0.x = 99.0;
        }

        let conflicts = world.check_column_conflicts(&snap);
        assert!(conflicts.contains(pos_id));
    }

    #[test]
    fn check_column_conflicts_clean_when_unchanged() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut bits = FixedBitSet::with_capacity(pos_id + 1);
        bits.insert(pos_id);

        let snap = world.snapshot_column_ticks(&bits);
        // No mutation
        let conflicts = world.check_column_conflicts(&snap);
        assert!(!conflicts.contains(pos_id));
    }
```

**Step 3: Run tests + clippy**

```
cargo test -p minkowski --lib -- snapshot_column_ticks check_column_conflicts
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 4: Commit**

```
git add crates/minkowski/src/world.rs
git commit -m "feat: World tick-snapshotting helpers for optimistic transactions"
```

---

### Task 3: Optimistic impl

**Files:**
- Modify: `crates/minkowski/src/transaction.rs`
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Add Optimistic + OptimisticTx after Sequential in `transaction.rs`**

```rust
// ── Optimistic ──────────────────────────────────────────────────────

/// Optimistic transaction strategy. Reads go directly to World (zero-copy).
/// Writes buffer into an EnumChangeSet. At commit, validates that no read
/// column was modified since begin — if conflict detected, returns Err.
///
/// Best for read-heavy workloads where conflicts are rare.
pub struct Optimistic;

impl TransactionStrategy for Optimistic {
    type Tx<'w> = OptimisticTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> OptimisticTx<'w> {
        let read_ticks = world.snapshot_column_ticks(access.reads());
        OptimisticTx {
            world,
            read_ticks,
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Optimistic`] strategy.
///
/// Reads are live (direct World access). Writes are buffered in an
/// `EnumChangeSet` and applied atomically at commit. If any column
/// that was read at begin has been modified by the time commit runs,
/// the transaction aborts with a [`Conflict`].
///
/// Drop without commit discards the buffered changeset (abort).
pub struct OptimisticTx<'w> {
    world: &'w mut World,
    read_ticks: Vec<(usize, ComponentId, crate::tick::Tick)>,
    changeset: EnumChangeSet,
}

impl<'w> OptimisticTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.changeset.insert::<T>(self.world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.changeset.remove::<T>(self.world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.world.alloc_entity();
        self.changeset.spawn_bundle(self.world, entity, bundle);
        entity
    }

    /// Validate and commit. Checks that no read column was modified since
    /// begin. Returns reverse changeset on success, Conflict on failure.
    pub fn commit(self) -> Result<EnumChangeSet, Conflict> {
        let conflicts = self.world.check_column_conflicts(&self.read_ticks);
        if !conflicts.is_empty() {
            return Err(Conflict {
                component_ids: conflicts,
            });
        }
        Ok(self.changeset.apply(self.world))
    }
}
```

**Step 2: Update `lib.rs` re-exports**

```rust
pub use transaction::{
    Conflict, Optimistic, OptimisticTx, Sequential, SequentialTx, TransactionStrategy,
};
```

**Step 3: Write tests**

Add to the `tests` module in `transaction.rs`:

```rust
    #[test]
    fn optimistic_commit_succeeds_without_conflict() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        // Read pos
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        // Write vel via changeset
        tx.insert::<Vel>(e, Vel(99.0));
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn optimistic_buffered_writes_not_visible_until_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(e, Pos(99.0));
        // Query still sees old value (write is buffered)
        let pos = tx.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit();
        // After commit, world has new value
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 99.0);
    }

    #[test]
    fn optimistic_drop_aborts_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.insert::<Pos>(e, Pos(99.0));
            // drop without commit — changeset discarded
        }
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0); // unchanged
    }

    #[test]
    fn optimistic_conflict_on_read_column_mutation() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Optimistic;
        let tx = strategy.begin(&mut world, &access);
        // Simulate another transaction modifying Pos (a read column)
        // by mutating directly through world (tx holds &mut World so
        // we need to go through the tx's world reference)
        // Instead: modify via query which advances the tick
        // We can't do this with tx alive (it borrows world).
        // So we test the lower-level helpers:
        drop(tx);

        // Manual test: snapshot, mutate, check
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut read_bits = FixedBitSet::with_capacity(pos_id + 1);
        read_bits.insert(pos_id);
        let snap = world.snapshot_column_ticks(&read_bits);

        // Mutate the read column
        for pos in world.query::<(&mut Pos,)>() {
            pos.0 .0 = 99.0;
        }

        let conflicts = world.check_column_conflicts(&snap);
        assert!(conflicts.contains(pos_id));
    }

    #[test]
    fn optimistic_spawn_allocates_entity() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn((Pos(42.0),));
        let result = tx.commit();
        assert!(result.is_ok());
        assert!(world.get::<Pos>(e).is_some());
    }
```

**Step 4: Run tests + clippy**

```
cargo test -p minkowski --lib -- transaction
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 5: Commit**

```
git add crates/minkowski/src/transaction.rs crates/minkowski/src/lib.rs
git commit -m "feat: Optimistic transaction strategy with tick validation"
```

---

### Task 4: Lock table infrastructure

**Files:**
- Create: `crates/minkowski/src/lock_table.rs`
- Modify: `crates/minkowski/src/world.rs` (add `lock_table` field)
- Modify: `crates/minkowski/src/lib.rs` (add module)

**Step 1: Create `lock_table.rs`**

```rust
use std::collections::HashMap;

use fixedbitset::FixedBitSet;

use crate::component::ComponentId;
use crate::storage::archetype::ArchetypeId;

/// Cooperative per-column lock table for pessimistic transactions.
///
/// Columns are identified by (ArchetypeId, ComponentId). Shared readers
/// and exclusive writers follow standard read-write lock semantics.
/// This is not an OS mutex — it's a bookkeeping structure for cooperative
/// transaction isolation.
pub(crate) struct ColumnLockTable {
    locks: HashMap<(usize, ComponentId), ColumnLock>,
}

#[derive(Default)]
struct ColumnLock {
    readers: u32,
    writer: bool,
}

/// A set of acquired column locks. Releases all locks on drop.
pub(crate) struct ColumnLockSet {
    held: Vec<(usize, ComponentId, LockMode)>,
}

#[derive(Clone, Copy)]
enum LockMode {
    Shared,
    Exclusive,
}

/// Error returned when a lock cannot be acquired.
pub(crate) struct LockConflict {
    pub component_ids: FixedBitSet,
}

impl ColumnLockTable {
    pub fn new() -> Self {
        Self {
            locks: HashMap::new(),
        }
    }

    /// Try to acquire all locks for the given access pattern.
    /// Read components get shared locks, write components get exclusive locks.
    /// Locks are acquired in deterministic order to prevent deadlock.
    ///
    /// Returns a LockSet on success, or LockConflict listing which
    /// components couldn't be locked.
    pub fn acquire(
        &mut self,
        archetypes: &[crate::storage::archetype::Archetype],
        reads: &FixedBitSet,
        writes: &FixedBitSet,
    ) -> Result<ColumnLockSet, LockConflict> {
        // Build sorted list of (arch_id, comp_id, mode) for deterministic ordering
        let mut requests: Vec<(usize, ComponentId, LockMode)> = Vec::new();
        for arch in archetypes {
            for comp_id in reads.ones() {
                if arch.component_index.contains_key(&comp_id) {
                    requests.push((arch.id.0, comp_id, LockMode::Shared));
                }
            }
            for comp_id in writes.ones() {
                if arch.component_index.contains_key(&comp_id) {
                    requests.push((arch.id.0, comp_id, LockMode::Exclusive));
                }
            }
        }
        requests.sort_by_key(|&(a, c, _)| (a, c));
        requests.dedup_by_key(|r| (r.0, r.1)); // write supersedes read on same column

        // Try to acquire all locks
        let mut acquired = Vec::new();
        let mut conflicts = FixedBitSet::new();

        for &(arch_id, comp_id, mode) in &requests {
            let lock = self.locks.entry((arch_id, comp_id)).or_default();
            let ok = match mode {
                LockMode::Shared => !lock.writer,
                LockMode::Exclusive => !lock.writer && lock.readers == 0,
            };
            if ok {
                match mode {
                    LockMode::Shared => lock.readers += 1,
                    LockMode::Exclusive => lock.writer = true,
                }
                acquired.push((arch_id, comp_id, mode));
            } else {
                conflicts.grow(comp_id + 1);
                conflicts.insert(comp_id);
            }
        }

        if !conflicts.is_empty() {
            // Roll back acquired locks
            for &(arch_id, comp_id, mode) in &acquired {
                self.release_one(arch_id, comp_id, mode);
            }
            return Err(LockConflict { component_ids: conflicts });
        }

        Ok(ColumnLockSet { held: acquired })
    }

    /// Release all locks in a lock set.
    pub fn release(&mut self, lock_set: ColumnLockSet) {
        for (arch_id, comp_id, mode) in lock_set.held {
            self.release_one(arch_id, comp_id, mode);
        }
    }

    fn release_one(&mut self, arch_id: usize, comp_id: ComponentId, mode: LockMode) {
        if let Some(lock) = self.locks.get_mut(&(arch_id, comp_id)) {
            match mode {
                LockMode::Shared => lock.readers = lock.readers.saturating_sub(1),
                LockMode::Exclusive => lock.writer = false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;
    use crate::storage::archetype::{Archetype, ArchetypeId};

    fn setup() -> (Vec<Archetype>, ComponentId, ComponentId) {
        let mut reg = ComponentRegistry::new();
        let pos_id = reg.register::<f32>();
        let vel_id = reg.register::<u32>();
        let arch = Archetype::new(ArchetypeId(0), &[pos_id, vel_id], &reg);
        (vec![arch], pos_id, vel_id)
    }

    #[test]
    fn shared_locks_coexist() {
        let (archs, pos_id, _) = setup();
        let mut table = ColumnLockTable::new();
        let mut reads = FixedBitSet::with_capacity(pos_id + 1);
        reads.insert(pos_id);
        let writes = FixedBitSet::new();

        let lock1 = table.acquire(&archs, &reads, &writes);
        assert!(lock1.is_ok());
        let lock2 = table.acquire(&archs, &reads, &writes);
        assert!(lock2.is_ok());

        table.release(lock1.unwrap());
        table.release(lock2.unwrap());
    }

    #[test]
    fn exclusive_conflicts_with_shared() {
        let (archs, pos_id, _) = setup();
        let mut table = ColumnLockTable::new();
        let mut reads = FixedBitSet::with_capacity(pos_id + 1);
        reads.insert(pos_id);
        let mut writes = FixedBitSet::with_capacity(pos_id + 1);
        writes.insert(pos_id);
        let empty = FixedBitSet::new();

        let shared = table.acquire(&archs, &reads, &empty).unwrap();
        let exclusive = table.acquire(&archs, &empty, &writes);
        assert!(exclusive.is_err());

        table.release(shared);
        // Now exclusive should succeed
        let exclusive = table.acquire(&archs, &empty, &writes);
        assert!(exclusive.is_ok());
        table.release(exclusive.unwrap());
    }

    #[test]
    fn exclusive_conflicts_with_exclusive() {
        let (archs, pos_id, _) = setup();
        let mut table = ColumnLockTable::new();
        let mut writes = FixedBitSet::with_capacity(pos_id + 1);
        writes.insert(pos_id);
        let empty = FixedBitSet::new();

        let lock1 = table.acquire(&archs, &empty, &writes).unwrap();
        let lock2 = table.acquire(&archs, &empty, &writes);
        assert!(lock2.is_err());

        table.release(lock1);
    }

    #[test]
    fn disjoint_columns_no_conflict() {
        let (archs, pos_id, vel_id) = setup();
        let mut table = ColumnLockTable::new();
        let empty = FixedBitSet::new();
        let mut w1 = FixedBitSet::with_capacity(pos_id + 1);
        w1.insert(pos_id);
        let mut w2 = FixedBitSet::with_capacity(vel_id + 1);
        w2.insert(vel_id);

        let lock1 = table.acquire(&archs, &empty, &w1).unwrap();
        let lock2 = table.acquire(&archs, &empty, &w2);
        assert!(lock2.is_ok());

        table.release(lock1);
        table.release(lock2.unwrap());
    }

    #[test]
    fn failed_acquire_rolls_back() {
        let (archs, pos_id, vel_id) = setup();
        let mut table = ColumnLockTable::new();
        let empty = FixedBitSet::new();

        // Hold exclusive lock on vel
        let mut w_vel = FixedBitSet::with_capacity(vel_id + 1);
        w_vel.insert(vel_id);
        let vel_lock = table.acquire(&archs, &empty, &w_vel).unwrap();

        // Try to lock both pos and vel exclusively — should fail on vel
        let mut w_both = FixedBitSet::with_capacity(vel_id + 1);
        w_both.insert(pos_id);
        w_both.insert(vel_id);
        let result = table.acquire(&archs, &empty, &w_both);
        assert!(result.is_err());

        // pos should NOT be locked (rolled back)
        let mut w_pos = FixedBitSet::with_capacity(pos_id + 1);
        w_pos.insert(pos_id);
        let pos_lock = table.acquire(&archs, &empty, &w_pos);
        assert!(pos_lock.is_ok());

        table.release(vel_lock);
        table.release(pos_lock.unwrap());
    }
}
```

**Step 2: Add `lock_table` field to World**

In `crates/minkowski/src/world.rs`, add to the `World` struct:

```rust
pub(crate) lock_table: crate::lock_table::ColumnLockTable,
```

And initialize it in `World::new()`:

```rust
lock_table: crate::lock_table::ColumnLockTable::new(),
```

**Step 3: Add module in `lib.rs`**

```rust
pub(crate) mod lock_table;
```

**Step 4: Run tests + clippy**

```
cargo test -p minkowski --lib -- lock_table
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 5: Commit**

```
git add crates/minkowski/src/lock_table.rs crates/minkowski/src/world.rs crates/minkowski/src/lib.rs
git commit -m "feat: cooperative column lock table for pessimistic transactions"
```

---

### Task 5: Pessimistic impl

**Files:**
- Modify: `crates/minkowski/src/transaction.rs`
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Add Pessimistic + PessimisticTx after Optimistic in `transaction.rs`**

```rust
use crate::lock_table::ColumnLockSet;

// ── Pessimistic ─────────────────────────────────────────────────────

/// Pessimistic transaction strategy. Acquires cooperative column locks
/// at begin. Reads and writes are guaranteed conflict-free. Commit
/// always succeeds.
///
/// Best for write-heavy workloads or expensive computations where
/// optimistic retry would waste work.
pub struct Pessimistic;

impl TransactionStrategy for Pessimistic {
    type Tx<'w> = PessimisticTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> PessimisticTx<'w> {
        let locks = world
            .lock_table
            .acquire(&world.archetypes.archetypes, access.reads(), access.writes())
            .expect("pessimistic begin: lock acquisition failed (held by another transaction)");
        PessimisticTx {
            world,
            locks: Some(locks),
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Pessimistic`] strategy.
///
/// Column locks are held for the transaction's lifetime. Reads go
/// through World (protected by shared locks). Writes buffer into
/// an `EnumChangeSet`. Commit applies the changeset and releases locks.
///
/// Drop without commit releases locks and discards the changeset (abort).
pub struct PessimisticTx<'w> {
    world: &'w mut World,
    locks: Option<ColumnLockSet>,
    changeset: EnumChangeSet,
}

impl<'w> PessimisticTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.changeset.insert::<T>(self.world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.changeset.remove::<T>(self.world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.world.alloc_entity();
        self.changeset.spawn_bundle(self.world, entity, bundle);
        entity
    }

    /// Commit the transaction. Always succeeds (locks guarantee isolation).
    /// Applies the buffered changeset and releases locks.
    pub fn commit(mut self) -> Result<EnumChangeSet, Conflict> {
        let reverse = self.changeset.apply(self.world);
        // Release locks explicitly (also released on drop, but be clear)
        if let Some(locks) = self.locks.take() {
            self.world.lock_table.release(locks);
        }
        Ok(reverse)
    }
}

impl<'w> Drop for PessimisticTx<'w> {
    fn drop(&mut self) {
        // Release locks if not already released by commit
        if let Some(locks) = self.locks.take() {
            self.world.lock_table.release(locks);
        }
    }
}
```

**Step 2: Update `lib.rs` re-exports**

```rust
pub use transaction::{
    Conflict, Optimistic, OptimisticTx, Pessimistic, PessimisticTx,
    Sequential, SequentialTx, TransactionStrategy,
};
```

**Step 3: Write tests**

Add to `transaction.rs` tests:

```rust
    #[test]
    fn pessimistic_commit_always_succeeds() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Pessimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let _ = tx.query::<(&Pos,)>().count();
        tx.insert::<Vel>(e, Vel(99.0));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn pessimistic_buffered_writes_applied_at_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic;
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(e, Pos(42.0));
        // Not visible yet
        let pos = tx.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit();
        // Now visible
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 42.0);
    }

    #[test]
    fn pessimistic_drop_releases_locks() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic;
        {
            let _tx = strategy.begin(&mut world, &access);
            // drop without commit
        }
        // Locks released — can begin another transaction
        let _tx2 = strategy.begin(&mut world, &access);
        let _ = _tx2.commit();
    }

    #[test]
    fn pessimistic_spawn_visible_after_commit() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn((Pos(77.0),));
        let _ = tx.commit();
        assert!(world.get::<Pos>(e).is_some());
        assert_eq!(world.get::<Pos>(e).unwrap().0, 77.0);
    }
```

**Step 4: Run tests + clippy**

```
cargo test -p minkowski --lib -- transaction
cargo test -p minkowski --lib -- lock_table
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 5: Commit**

```
git add crates/minkowski/src/transaction.rs crates/minkowski/src/lib.rs
git commit -m "feat: Pessimistic transaction strategy with cooperative column locks"
```

---

### Task 6: Transaction example

**Files:**
- Create: `examples/examples/transaction.rs`

**Step 1: Write the example**

Demonstrates all three strategies on the same workload. Shows optimistic conflict + retry, pessimistic guaranteed commit, sequential zero-cost path. Prints which strategy was used and what happened.

The example should:
1. Spawn 100 entities with (Pos, Vel, Health)
2. Run the same "movement + health decay" logic under each strategy
3. For Optimistic: demonstrate a conflict scenario (mutate read columns between begin/commit) and retry
4. For Pessimistic: show guaranteed commit
5. For Sequential: show zero overhead
6. Print results

**Step 2: Build and run**

```
cargo build -p minkowski-examples --example transaction
cargo run -p minkowski-examples --example transaction --release
```

**Step 3: Run clippy**

```
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 4: Commit**

```
git add examples/examples/transaction.rs
git commit -m "feat: transaction example demonstrating all three strategies"
```

---

### Task 7: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

- Add example command: `cargo run -p minkowski-examples --example transaction --release   # Transaction strategies demo (3 strategies, 100 entities)`
- Add Architecture section "### Transaction Semantics" after System Scheduling Primitives
- Update Key Conventions pub API list to include `TransactionStrategy`, `Conflict`, `Sequential`, `Optimistic`, `Pessimistic`, `Access`
- Update Dependencies if any new deps (none expected)
- Add `lock_table` to pub(crate) internals list

**Step 2: Update README.md**

- Add transaction example section after scheduler example
- Update "Phase 4 — done" line to include transaction semantics
- Remove "Transaction semantics" from roadmap table

**Step 3: Commit**

```
git add CLAUDE.md README.md
git commit -m "docs: add transaction semantics to CLAUDE.md and README.md"
```

---

### Verification

1. `cargo test -p minkowski --lib` — all tests pass (existing + new transaction + lock_table tests)
2. `cargo test -p minkowski --doc` — doc tests pass
3. `cargo run -p minkowski-examples --example transaction --release` — demonstrates all 3 strategies
4. `cargo clippy --workspace --all-targets -- -D warnings` — clean
5. `cargo build -p minkowski-examples --examples` — all examples build
