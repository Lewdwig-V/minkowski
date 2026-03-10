# Changeset Apply Path Optimization — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce QueryWriter overhead from ~112x slower than QueryMut to ~60-70x by eliminating the unconditional reverse changeset and batching tick increments in `EnumChangeSet::apply()`.

**Architecture:** Two targeted changes to `changeset.rs`: (1) `apply()` stops building a reverse changeset and returns `Result<(), ApplyError>` instead, (2) a single tick is advanced per `apply()` call instead of per mutation. No new abstractions or features — pure removal of unnecessary work.

**Tech Stack:** No new dependencies.

---

## Motivation

Benchmarks show QueryWriter is 112x slower than QueryMut for 10K entity iteration (181µs vs 1.6µs). The buffered-write architecture is inherent to QueryWriter's transactional/WAL semantics, but two sources of waste are not:

1. **Reverse changeset** — `apply()` unconditionally builds a reverse `EnumChangeSet` for undo/redo. This doubles the per-mutation work (extra arena alloc + Vec push + byte copy per entity). Only used by the `life.rs` example. No production caller uses it.

2. **Per-mutation tick increment** — `world.next_tick()` is an atomic `fetch_add` called per mutation. For 10K overwrites hitting the same column, only the last tick matters. One tick per `apply()` call is semantically equivalent.

## Changes

### 1. `ApplyError` enum

```rust
#[derive(Debug)]
pub enum ApplyError {
    /// Mutation targeted an entity that is no longer alive.
    DeadEntity(Entity),
    /// Spawn targeted an entity that is already placed in an archetype.
    AlreadyPlaced(Entity),
}
```

Replaces `assert!` panics in the Spawn and Insert paths. Consistent with the `ReducerError` pattern established in v1.0.1.

### 2. `apply()` signature

**Before:** `pub fn apply(mut self, world: &mut World) -> EnumChangeSet`
**After:** `pub fn apply(self, world: &mut World) -> Result<(), ApplyError>`

- No `reverse` changeset allocated or populated
- All `reverse.record_*()` calls removed from Spawn, Despawn, Insert, Remove, SparseInsert, SparseRemove handlers
- Dead entity / already-placed checks return `Err` instead of panicking

### 3. Single tick per apply

- `let tick = world.next_tick()` called once at the top of `apply()`
- Passed through to all `mark_changed(tick)` calls
- Semantically equivalent: `Changed<T>` checks "column tick > last read tick" — a batch at one tick is indistinguishable from individual ticks

### 4. Life example

Remove undo/redo from `examples/life.rs`. The example demonstrates Table queries and QueryMut — undo was a secondary feature that depended on the reverse changeset. Simpler example without it.

## Callers to update

| Caller | Current usage | Change |
|---|---|---|
| `World::insert` | Discards reverse | Handle `Result` (unwrap — internal, entity is known alive) |
| `World::remove` | Discards reverse | Same |
| `World::despawn` | Discards reverse | Same |
| `Tx::try_commit` | Discards reverse | Propagate `Result` through `Conflict` |
| `Durable::transact` | Discards reverse | Handle `Result` |
| `ReducerRegistry::call/run/dynamic_call` | Discards reverse | Handle `Result` |
| `examples/life.rs` | Uses reverse for undo | Remove undo feature |
| ~15 changeset tests | Assert on reverse | Rewrite without reverse assertions |

## Expected impact

Removing the reverse changeset eliminates ~30-50 cycles per entity (arena alloc + Vec push + copy). Batching ticks saves ~5-20 cycles per entity in atomics. Together: ~30-40% reduction in per-entity apply cost. QueryWriter ratio should drop from ~112x to ~60-70x.

The remaining gap (~60-70x) is inherent to buffered writes: clone + forward arena alloc + apply-time entity lookup + column write. This is the price of transactional atomicity and WAL compatibility.

## Non-goals

- **Reversible apply** — YAGNI. If needed later, can be added as a separate method.
- **Arena pre-sizing** — amortized O(1) via geometric growth is sufficient.
- **Drop function optimization** — branch predictor handles the `None` case; not worth the complexity.
- **DynamicCtx HashMap optimization** — separate concern, DynamicCtx's extra overhead (~8% over QueryWriter) is dominated by the changeset path we're fixing here.
