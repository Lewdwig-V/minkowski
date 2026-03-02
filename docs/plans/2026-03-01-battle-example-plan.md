# Battle Example + Transaction API Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Context:** See `docs/plans/2026-03-01-battle-example-design.md` for full design rationale.

**Goal:** Refactor the transaction API so Tx types don't hold `&mut World` (enabling split-phase concurrent execution), add a shared-ref query path for transactions, move the lock table from World to Pessimistic, and add a multi-threaded battle example that stress-tests optimistic vs pessimistic under tunable conflict pressure.

**Architecture:** Tx methods take `world` as a parameter instead of holding it. `World::query_raw(&self, matched_archetypes)` provides a shared-ref read path for the parallel execute phase. The battle example uses `rayon::scope` for real threading — multiple workers read concurrently through `&World`, buffer writes into private changesets, then commit sequentially.

**Tech Stack:** Rust, `rayon` (existing dependency), `fixedbitset` (existing)

---

### Task 1: Add `World::query_raw` — shared-ref read path

**Files:**
- Modify: `crates/minkowski/src/world.rs`

The transaction read path needs `&self` on World (not `&mut self`). This method builds fetches from a pre-resolved list of archetype IDs. No cache, no ticks, no mutable column marking.

**Step 1: Add `query_raw` method to `impl World`**

```rust
    /// Shared-ref query path for transactions. No cache update, no tick
    /// advancement, no column marking. Safe for concurrent reads from
    /// multiple threads.
    ///
    /// `matched_archetype_indices` should be pre-resolved during transaction
    /// begin (which has &mut World) by scanning archetypes against the
    /// query's required component bitset.
    pub(crate) fn query_raw<Q: WorldQuery>(
        &self,
        matched_archetype_indices: &[usize],
    ) -> QueryIter<'_, Q> {
        let fetches: Vec<_> = matched_archetype_indices
            .iter()
            .filter_map(|&idx| {
                let arch = self.archetypes.archetypes.get(idx)?;
                if arch.is_empty() {
                    return None;
                }
                Some((Q::init_fetch(arch, &self.components), arch.len()))
            })
            .collect();
        QueryIter::new(fetches)
    }

    /// Match archetypes against a query's required component bitset.
    /// Returns archetype indices for use with `query_raw`.
    pub(crate) fn match_archetypes<Q: WorldQuery + 'static>(&self) -> Vec<usize> {
        let required = Q::required_ids(&self.components);
        self.archetypes
            .archetypes
            .iter()
            .filter(|arch| required.is_subset(&arch.component_ids))
            .map(|arch| arch.id.0)
            .collect()
    }
```

**Step 2: Write tests**

Add to the world.rs tests module:

```rust
    #[test]
    fn query_raw_reads_without_mutation() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        let indices = world.match_archetypes::<(&Pos,)>();

        // query_raw takes &self — no tick advancement, no cache mutation
        let count = world.query_raw::<(&Pos,)>(&indices).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn query_raw_skips_empty_archetypes() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let indices = world.match_archetypes::<(&Pos,)>();
        world.despawn(e);

        let count = world.query_raw::<(&Pos,)>(&indices).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn match_archetypes_returns_correct_indices() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));
        let indices = world.match_archetypes::<(&Pos,)>();
        // Both archetypes contain Pos
        assert_eq!(indices.len(), 2);
    }
```

**Step 3: Run tests + clippy**

```
cargo test -p minkowski --lib -- query_raw match_archetypes
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 4: Commit**

```
git add crates/minkowski/src/world.rs
git commit -m "feat: World::query_raw shared-ref read path for transactions"
```

---

### Task 2: Refactor transaction API — split-phase, strategy owns state

**Files:**
- Modify: `crates/minkowski/src/transaction.rs`
- Modify: `crates/minkowski/src/lock_table.rs`
- Modify: `crates/minkowski/src/world.rs` (remove lock_table field)
- Modify: `crates/minkowski/src/lib.rs`

This is the big refactor. All three Tx types change: methods take `world` as a parameter. The lock table moves from World to Pessimistic.

**Step 1: Move lock_table from World to Pessimistic**

In `world.rs`:
- Remove `pub(crate) lock_table: crate::lock_table::ColumnLockTable,` from the World struct
- Remove `lock_table: crate::lock_table::ColumnLockTable::new(),` from World::new()
- Remove any `#[allow(dead_code)]` annotation that was on the field

**Step 2: Rewrite `transaction.rs` completely**

Replace the entire file with the refactored API. Key changes:

- **Trait**: `type Tx<'s> where Self: 's` — tied to strategy lifetime, not world
- **SequentialTx**: Unit struct. All methods take `world: &mut World` as parameter.
- **OptimisticTx**: Holds `read_ticks`, `matched_archetypes: Vec<usize>`, `changeset`. No world reference. `query(&self, &World)` uses `world.query_raw()`. `insert/spawn(&mut self, &mut World)` buffer into changeset. `commit(self, &mut World)` validates + applies.
- **Pessimistic**: Owns `ColumnLockTable`. `begin(&mut self, &mut World)` acquires locks from `self.lock_table`.
- **PessimisticTx<'s>**: Holds `&'s mut Pessimistic` for lock release on Drop. Same method pattern as OptimisticTx.

```rust
use fixedbitset::FixedBitSet;

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::lock_table::{ColumnLockSet, ColumnLockTable};
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
    type Tx<'s>
    where
        Self: 's;

    /// Begin a transaction. `access` declares which components will be
    /// read and written — used by Optimistic for tick snapshotting and
    /// by Pessimistic for lock acquisition.
    ///
    /// World is borrowed only for setup (archetype scanning, tick capture,
    /// lock acquisition). The returned Tx does not hold a World reference.
    fn begin<'s>(&'s mut self, world: &mut World, access: &Access) -> Self::Tx<'s>;
}

// ── Sequential ──────────────────────────────────────────────────────

/// Zero-cost transaction strategy. All operations delegate directly to World.
/// Commit always succeeds. No read-set, no changeset buffering, no validation.
///
/// Use when systems run sequentially and no conflict detection is needed.
pub struct Sequential;

impl TransactionStrategy for Sequential {
    type Tx<'s> = SequentialTx;

    fn begin<'s>(&'s mut self, _world: &mut World, _access: &Access) -> SequentialTx {
        SequentialTx
    }
}

/// Transaction object for the [`Sequential`] strategy.
/// Zero-state unit struct — all methods delegate directly to World.
pub struct SequentialTx;

impl SequentialTx {
    pub fn query<Q: WorldQuery + 'static>(&self, world: &mut World) -> QueryIter<'_, Q> {
        world.query::<Q>()
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, world: &mut World, bundle: B) -> Entity {
        world.spawn(bundle)
    }

    pub fn despawn(&mut self, world: &mut World, entity: Entity) -> bool {
        world.despawn(entity)
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, component: T) {
        world.insert(entity, component);
    }

    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) -> Option<T> {
        world.remove::<T>(entity)
    }

    pub fn get_mut<T: Component>(
        &mut self,
        world: &mut World,
        entity: Entity,
    ) -> Option<&mut T> {
        world.get_mut::<T>(entity)
    }

    /// Commit the transaction. Always succeeds for Sequential.
    /// Returns an empty reverse changeset (mutations went directly to World).
    pub fn commit(self, _world: &mut World) -> Result<EnumChangeSet, Conflict> {
        Ok(EnumChangeSet::new())
    }
}

// ── Optimistic ──────────────────────────────────────────────────────

/// Optimistic transaction strategy. Reads go directly to World (zero-copy)
/// via the shared-ref `query_raw` path. Writes buffer into an EnumChangeSet.
/// At commit, validates that no read column was modified since begin.
///
/// Best for read-heavy workloads where conflicts are rare.
pub struct Optimistic;

impl TransactionStrategy for Optimistic {
    type Tx<'s> = OptimisticTx;

    fn begin<'s>(&'s mut self, world: &mut World, access: &Access) -> OptimisticTx {
        let read_ticks = world.snapshot_column_ticks(access.reads());
        let matched_archetypes = world.match_archetypes_by_bitset(access);
        OptimisticTx {
            read_ticks,
            matched_archetypes,
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Optimistic`] strategy.
///
/// Does not hold a World reference — methods take `&World` or `&mut World`
/// as parameters. Reads use `World::query_raw` (shared-ref, no tick/cache
/// mutation). Writes buffer into an `EnumChangeSet`.
///
/// Drop without commit discards the buffered changeset (abort).
pub struct OptimisticTx {
    read_ticks: Vec<(usize, ComponentId, crate::tick::Tick)>,
    matched_archetypes: Vec<usize>,
    changeset: EnumChangeSet,
}

impl OptimisticTx {
    /// Read through the transaction via shared World reference.
    /// No tick advancement, no cache mutation. Safe for concurrent reads.
    pub fn query<Q: WorldQuery + 'static>(&self, world: &World) -> QueryIter<'_, Q> {
        world.query_raw::<Q>(&self.matched_archetypes)
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        self.changeset.insert::<T>(world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        self.changeset.remove::<T>(world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(
        &mut self,
        world: &mut World,
        bundle: B,
    ) -> Entity {
        let entity = world.alloc_entity();
        self.changeset.spawn_bundle(world, entity, bundle);
        entity
    }

    /// Validate and commit. Checks that no read column was modified since
    /// begin. Returns reverse changeset on success, Conflict on failure.
    pub fn commit(self, world: &mut World) -> Result<EnumChangeSet, Conflict> {
        let conflicts = world.check_column_conflicts(&self.read_ticks);
        if !conflicts.is_empty() {
            return Err(Conflict {
                component_ids: conflicts,
            });
        }
        Ok(self.changeset.apply(world))
    }
}

// ── Pessimistic ─────────────────────────────────────────────────────

/// Pessimistic transaction strategy. Acquires cooperative column locks
/// at begin. Reads and writes are guaranteed conflict-free. Commit
/// always succeeds.
///
/// Best for write-heavy workloads or expensive computations where
/// optimistic retry would waste work.
pub struct Pessimistic {
    lock_table: ColumnLockTable,
}

impl Pessimistic {
    pub fn new() -> Self {
        Self {
            lock_table: ColumnLockTable::new(),
        }
    }
}

impl TransactionStrategy for Pessimistic {
    type Tx<'s> = PessimisticTx<'s>;

    fn begin<'s>(&'s mut self, world: &mut World, access: &Access) -> PessimisticTx<'s> {
        let locks = self
            .lock_table
            .acquire(
                &world.archetypes.archetypes,
                access.reads(),
                access.writes(),
            )
            .expect("pessimistic begin: lock acquisition failed (held by another transaction)");
        let matched_archetypes = world.match_archetypes_by_bitset(access);
        PessimisticTx {
            strategy: self,
            locks: Some(locks),
            matched_archetypes,
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Pessimistic`] strategy.
///
/// Holds a reference to the Pessimistic strategy for lock release on drop.
/// Does not hold a World reference — methods take `&World` or `&mut World`.
///
/// Drop without commit releases locks and discards the changeset (abort).
pub struct PessimisticTx<'s> {
    strategy: &'s mut Pessimistic,
    locks: Option<ColumnLockSet>,
    matched_archetypes: Vec<usize>,
    changeset: EnumChangeSet,
}

impl<'s> PessimisticTx<'s> {
    /// Read through the transaction via shared World reference.
    pub fn query<Q: WorldQuery + 'static>(&self, world: &World) -> QueryIter<'_, Q> {
        world.query_raw::<Q>(&self.matched_archetypes)
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T) {
        self.changeset.insert::<T>(world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity) {
        self.changeset.remove::<T>(world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(
        &mut self,
        world: &mut World,
        bundle: B,
    ) -> Entity {
        let entity = world.alloc_entity();
        self.changeset.spawn_bundle(world, entity, bundle);
        entity
    }

    /// Commit the transaction. Always succeeds (locks guarantee isolation).
    pub fn commit(mut self, world: &mut World) -> Result<EnumChangeSet, Conflict> {
        let changeset = std::mem::take(&mut self.changeset);
        let reverse = changeset.apply(world);
        if let Some(locks) = self.locks.take() {
            self.strategy.lock_table.release(locks);
        }
        Ok(reverse)
    }
}

impl<'s> Drop for PessimisticTx<'s> {
    fn drop(&mut self) {
        if let Some(locks) = self.locks.take() {
            self.strategy.lock_table.release(locks);
        }
    }
}
```

**Step 3: Add `match_archetypes_by_bitset` helper to World**

In world.rs, add:

```rust
    /// Match archetypes against an Access's read+write component bitset.
    /// Returns archetype indices for use with `query_raw`.
    pub(crate) fn match_archetypes_by_bitset(&self, access: &Access) -> Vec<usize> {
        // Union reads and writes to get all components the transaction touches
        let mut all = access.reads().clone();
        all.grow(access.writes().len());
        all.union_with(access.writes());
        self.archetypes
            .archetypes
            .iter()
            .filter(|arch| {
                // Include archetype if it contains ANY of the accessed components
                all.ones().any(|comp_id| arch.component_ids.contains(comp_id))
            })
            .map(|arch| arch.id.0)
            .collect()
    }
```

Note: this matches any archetype containing at least one component in the access set. The actual query type narrows further when `init_fetch` runs — archetypes missing required components will fail at fetch init. A more precise approach would pre-resolve per-query-type, but that requires knowing query types at begin time. The Access bitset is a superset that's correct and simple.

Actually, a better approach: `query_raw` should still check `required_ids.is_subset(arch.component_ids)` per archetype. The matched list from begin is just an optimization hint — `query_raw` filters further. Let me revise: don't bother with `match_archetypes_by_bitset`. Just pass all archetype indices and let `query_raw` filter by required_ids. Even simpler: query_raw can scan all archetypes itself since it takes `&self`:

Revised `query_raw`:

```rust
    pub(crate) fn query_raw<Q: WorldQuery + 'static>(&self) -> QueryIter<'_, Q> {
        let required = Q::required_ids(&self.components);
        let fetches: Vec<_> = self
            .archetypes
            .archetypes
            .iter()
            .filter(|arch| !arch.is_empty() && required.is_subset(&arch.component_ids))
            .map(|arch| (Q::init_fetch(arch, &self.components), arch.len()))
            .collect();
        QueryIter::new(fetches)
    }
```

This is simpler — no pre-resolution needed. It scans archetypes every call but that's O(archetypes) which is fine for transaction use. Remove `match_archetypes_by_bitset` and `matched_archetypes` from the Tx types.

**Step 4: Update tests in transaction.rs**

All tests need to change to pass `world` as a parameter. Replace the entire tests module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::Access;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);

    // ── Sequential tests ────────────────────────────────────────────

    #[test]
    fn sequential_query_reads_spawned_entities() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &Vel)>(&mut world);

        let mut strategy = Sequential;
        let tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>(&mut world).count();
        assert_eq!(count, 1);
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn sequential_mutation_is_immediate() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn(&mut world, (Pos(42.0),));
        let count = tx.query::<(&Pos,)>(&mut world).count();
        assert_eq!(count, 1);
        let _ = tx.commit(&mut world);
    }

    #[test]
    fn sequential_commit_always_ok() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn(&mut world, (Pos(1.0),));
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn sequential_drop_is_noop() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let mut strategy = Sequential;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.spawn(&mut world, (Pos(1.0),));
        }
        assert_eq!(world.query::<(&Pos,)>().count(), 1);
    }

    // ── Optimistic tests ────────────────────────────────────────────

    #[test]
    fn optimistic_commit_succeeds_without_conflict() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>(&world).count();
        assert_eq!(count, 1);
        tx.insert::<Vel>(&mut world, e, Vel(99.0));
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn optimistic_buffered_writes_not_visible_until_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(&mut world, e, Pos(99.0));
        // query_raw sees old value (write is buffered in changeset)
        let pos = tx.query::<(&Pos,)>(&world).next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit(&mut world);
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
            tx.insert::<Pos>(&mut world, e, Pos(99.0));
        }
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
    }

    #[test]
    fn optimistic_conflict_detected_on_read_column_mutation() {
        use fixedbitset::FixedBitSet;

        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));

        let pos_id = world.components.id::<Pos>().unwrap();
        let mut read_bits = FixedBitSet::with_capacity(pos_id + 1);
        read_bits.insert(pos_id);
        let snap = world.snapshot_column_ticks(&read_bits);

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
        let e = tx.spawn(&mut world, (Pos(42.0),));
        assert!(tx.commit(&mut world).is_ok());
        assert!(world.get::<Pos>(e).is_some());
    }

    // ── Pessimistic tests ───────────────────────────────────────────

    #[test]
    fn pessimistic_commit_always_succeeds() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Pessimistic::new();
        let mut tx = strategy.begin(&mut world, &access);
        let _ = tx.query::<(&Pos,)>(&world).count();
        tx.insert::<Vel>(&mut world, e, Vel(99.0));
        assert!(tx.commit(&mut world).is_ok());
    }

    #[test]
    fn pessimistic_buffered_writes_applied_at_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic::new();
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(&mut world, e, Pos(42.0));
        let pos = tx.query::<(&Pos,)>(&world).next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit(&mut world);
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 42.0);
    }

    #[test]
    fn pessimistic_drop_releases_locks() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic::new();
        {
            let _tx = strategy.begin(&mut world, &access);
        }
        let tx2 = strategy.begin(&mut world, &access);
        let _ = tx2.commit(&mut world);
    }

    #[test]
    fn pessimistic_spawn_visible_after_commit() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic::new();
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn(&mut world, (Pos(77.0),));
        let _ = tx.commit(&mut world);
        assert!(world.get::<Pos>(e).is_some());
        assert_eq!(world.get::<Pos>(e).unwrap().0, 77.0);
    }
}
```

**Step 5: Update lib.rs**

Remove the `lock_table` module from World's path. Update re-exports — `Pessimistic` now has a constructor:

```rust
pub use transaction::{
    Conflict, Optimistic, OptimisticTx, Pessimistic, PessimisticTx, Sequential, SequentialTx,
    TransactionStrategy,
};
```

**Step 6: Run all tests + clippy**

```
cargo test -p minkowski --lib
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 7: Commit**

```
git add crates/minkowski/src/transaction.rs crates/minkowski/src/world.rs crates/minkowski/src/lib.rs
git commit -m "refactor: split-phase transaction API, strategy owns state

Tx types no longer hold &mut World. Methods take world as parameter.
Lock table moves from World to Pessimistic. Enables concurrent read
phase for multi-threaded execution."
```

---

### Task 3: Update existing transaction example

**Files:**
- Modify: `examples/examples/transaction.rs`

Adapt the existing transaction example to the new API where all Tx methods take `world` as a parameter. `Pessimistic` is now constructed with `Pessimistic::new()`. Optimistic/Pessimistic `query()` takes `&world` (shared ref).

The structure stays the same — 4 sections (sequential, optimistic clean, optimistic conflict, pessimistic). Just update the method calls.

**Step 1: Update all method calls**

Key changes:
- `tx.query::<Q>()` → `tx.query::<Q>(&mut world)` (sequential) or `tx.query::<Q>(&world)` (optimistic/pessimistic)
- `tx.insert::<T>(e, val)` → `tx.insert::<T>(&mut world, e, val)`
- `tx.spawn(bundle)` → `tx.spawn(&mut world, bundle)`
- `tx.commit()` → `tx.commit(&mut world)`
- `Pessimistic` → `Pessimistic::new()`

**Step 2: Build and run**

```
cargo build -p minkowski-examples --example transaction
cargo run -p minkowski-examples --example transaction --release
```

**Step 3: Commit**

```
git add examples/examples/transaction.rs
git commit -m "refactor: update transaction example for split-phase API"
```

---

### Task 4: Battle example with rayon threading

**Files:**
- Create: `examples/examples/battle.rs`

A multi-threaded arena battle with tunable conflict rates. Uses `rayon::scope` for real concurrent execution during the read/compute phase.

**Components:** `Health(u32)`, `Team(u8)`, `Damage(u32)`, `Healing(u32)`

**Systems:**
- `combat` — reads Team, writes Damage on targets
- `healing` — reads Team, writes Healing on targets
- `apply_effects` — reads Damage + Healing, writes Health

**Two modes:**
- Low conflict: combat targets Team 1, healing targets Team 0 (disjoint)
- High conflict: both target overlapping entity ranges

**Threading model:**
```rust
// begin phase (sequential — needs &mut World)
let mut tx_combat = optimistic.begin(&mut world, &combat_access);
let mut tx_healing = optimistic.begin(&mut world, &healing_access);

// execute phase (parallel — &World is shareable)
// Pre-buffer all writes before the parallel phase since insert needs &mut World
// Actually: the parallel read phase computes WHAT to write (entity IDs + values)
// into local Vecs. Then the sequential write phase buffers into changesets.
rayon::scope(|s| {
    let world_ref = &world;
    s.spawn(|_| {
        // Read Team + compute damage targets → local Vec
        combat_results = tx_combat.query::<(&Team,)>(world_ref)
            .filter(...)
            .collect();
    });
    s.spawn(|_| {
        healing_results = tx_healing.query::<(&Team,)>(world_ref)
            .filter(...)
            .collect();
    });
});

// write phase (sequential — needs &mut World for changeset buffering)
for (entity, dmg) in combat_results {
    tx_combat.insert::<Damage>(&mut world, entity, dmg);
}

// commit phase (sequential)
tx_combat.commit(&mut world)?;
tx_healing.commit(&mut world)?;
```

Run 100 frames. Print conflict counts, retry counts, timing.

The example should be ~200-250 lines. Print clear output showing the strategy behavior under both conflict modes.

**Step 1: Write the example**

**Step 2: Build and run**

```
cargo build -p minkowski-examples --example battle
cargo run -p minkowski-examples --example battle --release
```

**Step 3: Run clippy**

```
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 4: Commit**

```
git add examples/examples/battle.rs
git commit -m "feat: battle example — multi-threaded transaction stress test"
```

---

### Task 5: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

- Add battle example command
- Update Transaction Semantics section to reflect split-phase API
- Update key conventions: lock table is now `pub(crate)` on Pessimistic, not on World

**Step 2: Update README.md**

- Add battle example section after transaction example
- Update transaction description to mention split-phase API and multi-threaded battle test

**Step 3: Commit**

```
git add CLAUDE.md README.md
git commit -m "docs: update for split-phase transaction API and battle example"
```

---

### Verification

1. `cargo test -p minkowski --lib` — all tests pass (existing + refactored transaction + new query_raw)
2. `cargo test -p minkowski --doc` — doc tests pass
3. `cargo run -p minkowski-examples --example transaction --release` — adapted example works
4. `cargo run -p minkowski-examples --example battle --release` — multi-threaded execution, both modes
5. `cargo clippy --workspace --all-targets -- -D warnings` — clean
6. `cargo build -p minkowski-examples --examples` — all examples build
