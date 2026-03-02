# Battle Simulation Example + Transaction API Refactor

**Date:** 2026-03-01
**Status:** Approved

## Context

The current transaction API has `Tx` types hold `&'w mut World` for their entire lifetime. This prevents split-phase execution where multiple transactions read concurrently and commit sequentially. The battle example needs this pattern to demonstrate optimistic vs pessimistic under tunable conflict pressure.

## Part 1: Transaction API Refactor

### Strategy owns the state, World stays clean

The lock table moves from World to `Pessimistic`. World has no concurrency infrastructure — it's a storage engine.

```rust
pub struct Sequential;           // stateless
pub struct Optimistic;           // stateless (tick snapshots are per-tx)
pub struct Pessimistic {         // owns the lock table
    lock_table: ColumnLockTable,
}
```

### Trait (revised)

```rust
pub trait TransactionStrategy {
    type Tx<'s> where Self: 's;

    fn begin<'s>(&'s mut self, world: &mut World, access: &Access) -> Self::Tx<'s>;
}
```

The lifetime `'s` ties the Tx to the strategy, not to World. World is borrowed transiently during `begin()` and `commit()`, not held.

### Tx types hold strategy reference, not World

```rust
pub struct SequentialTx;  // unit struct, zero state

pub struct OptimisticTx {
    read_ticks: Vec<(usize, ComponentId, Tick)>,
    changeset: EnumChangeSet,
}

pub struct PessimisticTx<'s> {
    strategy: &'s mut Pessimistic,  // for lock release on drop
    locks: Option<ColumnLockSet>,
    changeset: EnumChangeSet,
}
```

### Method signatures (all Tx types)

```rust
fn query<Q>(&mut self, world: &mut World) -> QueryIter<Q>;
fn insert<T>(&mut self, world: &mut World, entity: Entity, value: T);
fn remove<T>(&mut self, world: &mut World, entity: Entity);
fn spawn<B>(&mut self, world: &mut World, bundle: B) -> Entity;
fn commit(self, world: &mut World) -> Result<EnumChangeSet, Conflict>;
```

World is passed to each call. This enables the split-phase pattern: begin (captures state), execute (world is free between calls), commit (validates + applies).

### Drop semantics

- **SequentialTx**: No Drop needed (unit struct).
- **OptimisticTx**: EnumChangeSet Drop runs destructors on buffered values. No other cleanup.
- **PessimisticTx**: Custom Drop releases locks via `self.strategy.lock_table.release()`. Option::take pattern prevents double-release after commit.

### World changes

Remove `lock_table: ColumnLockTable` field from World. It was added prematurely — the lock table is concurrency policy, not storage infrastructure.

## Part 2: Battle Simulation Example

### `examples/examples/battle.rs`

Arena battle with tunable conflict rates. Demonstrates optimistic vs pessimistic transaction strategies.

**Components:**
- `Health(u32)` — hitpoints
- `Team(u8)` — 0 or 1
- `Damage(u32)` — pending damage
- `Healing(u32)` — pending healing

**Systems:**
- `combat` — reads Team, writes Damage on enemies
- `healing` — reads Team, writes Healing on allies
- `apply_effects` — reads Damage + Healing, writes Health

**Two modes:**

| Mode | Combat targets | Healing targets | Expected conflicts |
|---|---|---|---|
| Low conflict | Disjoint entity sets | Disjoint entity sets | Rare |
| High conflict | Overlapping entities | Overlapping entities | Frequent |

**Flow per frame:**
1. `begin()` transactions (one per system)
2. Execute each system (read World, buffer writes)
3. `commit()` sequentially
4. Optimistic: count conflicts, retry failed transactions
5. Print stats

**Output:** Conflict counts, retry counts, timing per strategy per mode. High-conflict optimistic shows retry cost. Low-conflict shows both strategies perform similarly.

**No actual threading.** Transactions run sequentially. Comments explain where `rayon::scope` would slot in.

## Dependencies

No new external dependencies. `ColumnLockTable` moves from `lock_table.rs` (stays `pub(crate)`) to being owned by `Pessimistic` instead of `World`.
