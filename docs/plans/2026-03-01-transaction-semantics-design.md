# Transaction Semantics Design

**Date:** 2026-03-01
**Status:** Approved

## Context

No single concurrency strategy is optimal for all workloads. The storage engine shouldn't decide the concurrency model any more than it should decide the system scheduling order or the spatial index type. Provide the interface. Ship implementations as defaults. Framework authors swap in their own.

The guarantee: everything in the transaction is visible atomically to other observers, or none of it is. The caller decides what "everything" means — a system invocation, a client update batch, or an arbitrary set of mutations buffered into a changeset.

### Not MVCC

This design is not simplified MVCC and is not a stepping stone toward MVCC. MVCC solves reader-writer coexistence through version chains. We solve the same problem through tick validation (optimistic) and cooperative locks (pessimistic). These are different mechanisms with different trade-offs. The implementation must be optimized for its own semantics — no version chain scaffolding, no assumptions shaped for future MVCC support.

## Design

### The Trait

```rust
pub trait TransactionStrategy {
    type Tx<'w>;

    fn begin<'w>(
        &mut self,
        world: &'w mut World,
        access: &Access,
    ) -> Self::Tx<'w>;
}
```

`begin()` takes `&mut World` (exclusive access to start the transaction) and `&Access` (declares what components this transaction touches). Returns an opaque `Tx<'w>` tied to the World's lifetime.

Each `Tx` struct exposes inherent methods (not trait methods):
- `query<Q>()` — read through the transaction
- `insert/remove/spawn/despawn` — write operations buffer into an internal `EnumChangeSet`
- `commit(self) -> Result<EnumChangeSet, Conflict>` — validate + apply

Drop without commit = abort (RAII safety).

```rust
pub struct Conflict {
    pub component_ids: FixedBitSet,
}
```

### Sequential — Zero-Cost Path

```rust
pub struct Sequential;
pub struct SequentialTx<'w> {
    world: &'w mut World,
}
```

Transparent wrapper around `&mut World`. All methods delegate directly. No read-set, no changeset buffering, no validation. The compiler should inline everything away.

- `query()` → `self.world.query()`
- `insert()` → `self.world.insert()`
- `commit()` → always returns `Ok(EnumChangeSet::new())`

Zero overhead. This is what a user pays when they don't need transactions. The `_access` parameter is unused.

### Optimistic — Read-Heavy Default

```rust
pub struct Optimistic;
pub struct OptimisticTx<'w> {
    world: &'w mut World,
    read_ticks: Vec<(ComponentId, Tick)>,
    changeset: EnumChangeSet,
}
```

**Begin:** Snapshot `changed_tick` values for all columns matching the read access bitset. Cost proportional to `|read_components| × |archetypes|`, paid once.

**Read path:** `query()` delegates to `self.world.query()` — direct pointer reads, zero copy. No per-read tracking needed because `Access` already declares which columns will be read. The read-set is the tick snapshot captured at begin.

**Write path:** Buffer into `self.changeset` via EnumChangeSet's typed API.

**Commit:** Check if any read column's `changed_tick` advanced since begin. If yes, return `Err(Conflict)`. If no, apply changeset atomically, return `Ok(reverse)`.

**Drop = abort:** EnumChangeSet destructor runs on buffered values. Read-ticks discarded. World untouched.

**Isolation level:** Read-committed, not repeatable-read. Between begin and commit, the caller sees live World state. Validation catches write-write and read-write conflicts but doesn't prevent phantom reads. This is the right trade-off for game workloads — most systems run once per frame and don't re-read.

### Pessimistic — Write-Heavy Alternative

```rust
pub struct Pessimistic;
pub struct PessimisticTx<'w> {
    world: &'w mut World,
    locks: ColumnLockSet,
    changeset: EnumChangeSet,
}
```

**Lock granularity: per-column.** A column is a `(ArchetypeId, ComponentId)` pair — one BlobVec. This matches what `Access` already describes and what change detection already tracks. Per-entity is too fine (millions of lock ops), per-archetype too coarse (locks unrelated components), per-component-type over-locks (two archetypes with Pos are independent).

**Lock table:**

```rust
pub(crate) struct ColumnLockTable {
    locks: HashMap<(ArchetypeId, ComponentId), ColumnLock>,
}

pub(crate) struct ColumnLock {
    readers: u32,
    writer: bool,
}
```

Stored on World as `pub(crate) lock_table: ColumnLockTable`, initialized empty, zero cost if unused. Not a real OS mutex — cooperative locking for interleaved transactions.

**Begin:** Acquire locks for all accessed columns upfront. Read columns get shared locks (fails if writer held). Write columns get exclusive locks (fails if any reader or writer held). Acquire in deterministic order (sorted `(ArchetypeId, ComponentId)`) to prevent deadlock.

**Read path:** `query()` delegates to `self.world.query()`. Locks guarantee no concurrent writer will modify read columns.

**Write path:** Buffer into `self.changeset`.

**Commit:** No validation needed — locks guarantee isolation. Apply changeset, release locks. Always returns `Ok(reverse)`. The `Err` branch is unreachable.

**Drop = abort:** Release all locks, run EnumChangeSet destructors. World untouched.

**When to choose:** When a transaction is expensive to compute and cheap to write. If a system spends 10ms computing physics and 0.1ms applying results, an optimistic conflict at commit means re-doing 10ms of work. Pessimistic pays lock overhead upfront but guarantees commit succeeds.

## File Layout

All in `crates/minkowski/src/transaction.rs` as a module with submodules:
- `transaction.rs` — trait, `Conflict`, `Sequential` impl
- `transaction/optimistic.rs` — `Optimistic` + `OptimisticTx`
- `transaction/pessimistic.rs` — `Pessimistic` + `PessimisticTx`
- `transaction/lock_table.rs` — `ColumnLockTable`, `ColumnLock`, `ColumnLockSet`

## Public API

| Type | Visibility | Reason |
|---|---|---|
| `TransactionStrategy` | `pub` | The trait |
| `Sequential`, `Optimistic`, `Pessimistic` | `pub` | The three strategies |
| `SequentialTx`, `OptimisticTx`, `PessimisticTx` | `pub` | Associated types must be nameable |
| `Conflict` | `pub` | Error type from commit |
| `ColumnLockTable`, `ColumnLock` | `pub(crate)` | Internal infrastructure |

Re-exported from `lib.rs`:
```rust
pub use transaction::{TransactionStrategy, Conflict, Sequential, Optimistic, Pessimistic};
```

## World Additions

Minimal:
- `pub(crate) lock_table: ColumnLockTable` on World (initialized empty, zero cost if unused)
- `pub(crate)` helpers to read column ticks by ComponentId for optimistic validation
- No new public methods on World

## Example

`examples/examples/transaction.rs` — demonstrates all three strategies on the same workload. Shows optimistic conflict + retry, pessimistic guaranteed commit, sequential zero-cost path. Similar shape to the scheduler example.

## Dependencies

No new external dependencies. Uses `FixedBitSet` (existing), `EnumChangeSet` (existing), `Tick` (existing).
