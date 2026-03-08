# Loom Concurrency Verification Design

**Goal**: Exhaustively verify Minkowski's transactional concurrency invariants using loom's deterministic schedule enumeration.

**Scope**: OrphanQueue, ColumnLockTable, and EntityAllocator — the three concurrency-critical primitives. Rayon `par_for_each` stays with TSan (loom doesn't support rayon's work-stealing scheduler).

## Sync Abstraction Layer

New module `crates/minkowski/src/sync.rs` conditionally re-exports concurrency primitives:

- `cfg(not(loom))`: direct re-export of `parking_lot::Mutex`, `std::sync::atomic::*`, `std::sync::Arc`
- `cfg(loom)`: re-export `loom::sync::Arc`, `loom::sync::atomic::*`, and a thin `Mutex<T>` newtype wrapping `loom::sync::Mutex` with infallible `.lock()` to match parking_lot's API

All internal code (`world.rs`, `lock_table.rs`, `entity.rs`, `transaction.rs`, `reducer.rs`) imports from `crate::sync` instead of directly from `parking_lot` or `std::sync::atomic`. Production builds see zero change — the re-exports are direct, no wrapper. The loom newtype only exists in loom test builds (always debug, performance irrelevant).

## Loom Test Suite

Integration test file: `crates/minkowski/tests/loom_concurrency.rs`, gated by `#[cfg(loom)]`.

### Test 1: OrphanQueue concurrent push + drain

- Spawn 2 loom threads that each push entity IDs to a shared OrphanQueue
- Main thread drains the queue via `lock().drain(..)`
- Assert: every pushed ID appears exactly once across drain results — no lost IDs, no duplicates
- Verifies the invariant that broke during design review (entity leak on abort)

### Test 2: ColumnLockTable acquire/release/upgrade

Three variants:

1. **Exclusive conflict**: Two threads attempt overlapping locks — one Shared, one Exclusive on same column. Verify mutual exclusion: never both succeed simultaneously. On failure, all partially-held locks are rolled back.
2. **Upgrade-not-downgrade**: Same thread requests both Shared and Exclusive on same column. Verify the Exclusive privilege is kept (the bug where `dedup_by_key` silently kept Shared).
3. **Deadlock freedom**: Two threads acquire locks in opposite orders. Verify no deadlock due to sorted acquisition ordering.

### Test 3: EntityAllocator::reserve contention

- N loom threads each call `reserve()` concurrently via the shared AtomicU32
- Collect all returned Entity indices
- Assert: all indices are unique (no duplicate allocation)

## Build Integration

- `loom` added as a dev-dependency of `minkowski` crate
- Not a cargo feature — tests gated by `#[cfg(loom)]`, activated via `RUSTFLAGS="--cfg loom"`
- **Not in CI** — loom's exhaustive enumeration is too slow for per-PR gates. Manual verification tool, same category as cargo-fuzz.
- Run command: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency`

## Files to Modify

- Create: `crates/minkowski/src/sync.rs`
- Create: `crates/minkowski/tests/loom_concurrency.rs`
- Modify: `crates/minkowski/src/lib.rs` (add `mod sync`)
- Modify: `crates/minkowski/src/world.rs` (import from `crate::sync`)
- Modify: `crates/minkowski/src/entity.rs` (import from `crate::sync`)
- Modify: `crates/minkowski/src/lock_table.rs` (import from `crate::sync`)
- Modify: `crates/minkowski/src/transaction.rs` (import from `crate::sync`)
- Modify: `crates/minkowski/src/reducer.rs` (import from `crate::sync`)
- Modify: `crates/minkowski/Cargo.toml` (add loom dev-dependency)
- Modify: `CLAUDE.md` (document loom command)
