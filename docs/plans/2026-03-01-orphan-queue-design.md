# World-Owned Orphan Queue

## Problem

Entity ID cleanup after transaction abort is coupled to strategy behavior. The `pending_release: Mutex<Vec<Entity>>` queue lives on each strategy (`Optimistic`, `Pessimistic`), drained in their `begin()` implementations. This means:

- A custom `TransactionStrategy` that forgets to drain leaks IDs silently.
- A strategy dropped without another `begin()` leaks IDs.
- The invariant is "correct if all strategy impls cooperate" — not engine-guaranteed.

The engine must own this invariant. If the framework has to remember to call `drain_orphans`, someone will forget and entity IDs will silently leak.

## Design

### Shared Handle

The queue belongs on World. The problem is getting the transaction's `Drop` to reach it. The answer is a shared handle:

```
World ──owns──> OrphanQueue(Arc<Mutex<Vec<Entity>>>)
                      |
              ┌───────┴───────┐
              v               v
         Optimistic      Pessimistic
         (clone)         (clone)
              |               |
              v               v
         OptimisticTx    PessimisticTx
         (via &strategy)  (via &strategy)
```

`OrphanQueue` is `pub(crate)` — it never appears in any public type signature. Strategy constructors take `&World` and clone the queue internally.

### Lifecycle

1. **World creates** the `OrphanQueue` in `World::new()`.
2. **Strategy clones** the handle at construction: `Optimistic::new(&world)`.
3. **Transaction Drop** pushes spawned entity IDs to the shared queue (via `strategy.orphan_queue`).
4. **Transaction commit on conflict** deallocates immediately (has `&mut World`).
5. **World drains** automatically — a single private `drain_orphans(&mut self)` method called at the top of every `&mut self` entry point.

Transactions push on abort. World drains on use. No manual step. No leaked IDs. The engine guarantees it.

### Type

```rust
/// Shared queue for entity IDs orphaned by aborted transactions.
/// World owns the canonical instance; strategies clone the Arc handle.
#[derive(Clone)]
pub(crate) struct OrphanQueue(Arc<Mutex<Vec<Entity>>>);
```

### drain_orphans

A single private method on World, called at the top of every `&mut self` entry point (`spawn`, `despawn`, `insert`, `remove`, `get_mut`, `query`, `alloc_entity`). Not called from `query_raw(&self)` (read-only, no `&mut self`).

```rust
fn drain_orphans(&mut self) {
    let mut queue = self.orphan_queue.0.lock();
    for entity in queue.drain(..) {
        self.entities.dealloc(entity);
    }
}
```

Overhead on empty queue: one uncontended `parking_lot::Mutex` CAS + `Vec::is_empty()` check.

### API Changes

| Before | After |
|--------|-------|
| `Optimistic::new()` | `Optimistic::new(&world)` |
| `Pessimistic::new()` | `Pessimistic::new(&world)` |
| Strategy has `pending_release` field | Removed |
| `begin()` drains strategy queue | Removed — World auto-drains |
| `Sequential` unchanged | Unchanged (no spawns to track) |

### What Doesn't Change

- `TransactionStrategy` trait signature — still `begin(&self, &mut World, &Access) -> Tx`.
- Drop semantics — transactions still push to queue on abort.
- Commit on conflict — still deallocates immediately (has `&mut World`).
- `query_raw(&self)` — doesn't drain (read-only path, no `&mut self`).
