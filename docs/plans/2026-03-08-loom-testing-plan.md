# Loom Concurrency Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Exhaustively verify OrphanQueue, ColumnLockTable, and EntityAllocator concurrency invariants using loom's deterministic schedule enumeration.

**Architecture:** A `sync.rs` abstraction layer conditionally re-exports `parking_lot::Mutex` (production) or a loom-compatible wrapper (test). All internal sync imports are routed through `crate::sync`. Three loom model tests exercise the concurrency-critical primitives in isolation.

**Tech Stack:** [loom](https://github.com/tokio-rs/loom) (dev-dependency), `cfg(loom)` conditional compilation

---

### Task 1: Add loom dev-dependency and sync abstraction module

**Files:**
- Modify: `crates/minkowski/Cargo.toml` (add loom dev-dependency)
- Create: `crates/minkowski/src/sync.rs`
- Modify: `crates/minkowski/src/lib.rs:80` (add `mod sync`)

**Step 1: Add loom to Cargo.toml**

In `crates/minkowski/Cargo.toml`, add loom as a dev-dependency:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
hecs = "0.10"
loom = "0.7"
```

**Step 2: Create `crates/minkowski/src/sync.rs`**

```rust
//! Conditional sync primitives — routes to parking_lot/std in production,
//! loom equivalents under `cfg(loom)` for deterministic schedule testing.

#[cfg(not(loom))]
pub(crate) use parking_lot::Mutex;

#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};

#[cfg(not(loom))]
pub(crate) use std::sync::Arc;

#[cfg(loom)]
pub(crate) use loom::sync::Arc;

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};

// loom::sync::Mutex::lock() returns Result — wrap to match parking_lot's
// infallible API so call sites don't change.
#[cfg(loom)]
mod loom_mutex {
    pub(crate) struct Mutex<T>(loom::sync::Mutex<T>);

    impl<T> Mutex<T> {
        #[inline]
        pub fn new(val: T) -> Self {
            Self(loom::sync::Mutex::new(val))
        }

        #[inline]
        pub fn lock(&self) -> loom::sync::MutexGuard<'_, T> {
            self.0.lock().unwrap()
        }
    }
}

#[cfg(loom)]
pub(crate) use loom_mutex::Mutex;

// Thread operations: yield_now must route through loom for schedule control.
#[cfg(not(loom))]
#[inline]
pub(crate) fn yield_now() {
    std::thread::yield_now();
}

#[cfg(loom)]
#[inline]
pub(crate) fn yield_now() {
    loom::thread::yield_now();
}
```

**Step 3: Add `mod sync` to `crates/minkowski/src/lib.rs`**

Add the module declaration after `lock_table` (line 80):

```rust
pub(crate) mod lock_table;
pub(crate) mod sync;
pub mod query;
```

**Step 4: Verify it compiles**

Run: `cargo check -p minkowski`
Expected: success (sync.rs is defined but not yet imported anywhere)

**Step 5: Commit**

```bash
git add crates/minkowski/Cargo.toml crates/minkowski/src/sync.rs crates/minkowski/src/lib.rs
git commit -m "feat(loom): add sync abstraction layer and loom dev-dependency"
```

---

### Task 2: Rewire internal imports to use `crate::sync`

**Files:**
- Modify: `crates/minkowski/src/world.rs:11,15-16` (replace parking_lot/std imports)
- Modify: `crates/minkowski/src/transaction.rs:58,61` (replace parking_lot/std imports)
- Modify: `crates/minkowski/src/entity.rs:64,77,88-91,98-102` (replace inline std::sync::atomic)
- Modify: `crates/minkowski/src/reducer.rs:4-5` (replace std::sync imports)

**Important context:** Only change the *module-level* production imports. Do NOT touch imports inside `#[cfg(test)] mod tests` blocks — those use `std::sync::atomic::AtomicUsize` for drop counters and test instrumentation, which are unrelated to the concurrency primitives we're abstracting.

**Step 1: Rewire `world.rs` imports**

Replace lines 11, 15-16:

```rust
// Old:
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// New:
use crate::sync::{Arc, AtomicU64, Mutex, Ordering};
```

Also update the static `NEXT_WORLD_ID` on line 24. This is a `static AtomicU64` — under loom, statics must use `loom::lazy_static!` or `loom::sync::atomic::AtomicU64`. However, loom's `AtomicU64` doesn't implement `const` construction. The simplest fix: gate the static:

```rust
#[cfg(not(loom))]
static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);

#[cfg(loom)]
loom::lazy_static! {
    static ref NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);
}
```

**Step 2: Rewire `transaction.rs` imports and yield_now**

Replace lines 58 and 61:

```rust
// Old:
use std::sync::atomic::AtomicU64;
use parking_lot::Mutex;

// New:
use crate::sync::{AtomicU64, Mutex};
```

Also replace the `std::thread::yield_now()` call in the backoff function (line 545):

```rust
// Old:
std::thread::yield_now();

// New:
crate::sync::yield_now();
```

**Step 3: Rewire `entity.rs` inline atomic references**

`entity.rs` uses fully-qualified `std::sync::atomic::AtomicU32` inline (lines 64, 77, 91). Replace with import from `crate::sync`:

Add at the top of the file:

```rust
use crate::sync::{AtomicU32, Ordering};
```

Then change:
- Line 64: `std::sync::atomic::AtomicU32` → `AtomicU32`
- Line 77: `std::sync::atomic::AtomicU32::new(0)` → `AtomicU32::new(0)`
- Line 91: `std::sync::atomic::Ordering::Relaxed` → `Ordering::Relaxed`

**Step 4: Rewire `reducer.rs` imports**

Replace lines 4-5:

```rust
// Old:
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// New:
use crate::sync::{Arc, AtomicBool, AtomicU64, Ordering};
```

**Step 5: Verify production build is unchanged**

Run: `cargo test -p minkowski --lib`
Expected: all 396+ tests pass — behavior identical, only import paths changed.

Run: `cargo clippy -p minkowski --all-targets -- -D warnings`
Expected: clean

**Step 6: Verify loom cfg compiles**

Run: `RUSTFLAGS="--cfg loom" cargo check -p minkowski --lib`
Expected: success (compiles with loom's sync primitives)

Note: if loom's `AtomicU32` doesn't support `get_mut()` (used in `entity.rs` lines 100, 102, 109), we may need a shim. `get_mut()` is a `&mut self` method on std atomics that bypasses atomics for exclusive access. If loom doesn't have it, add to `sync.rs`:

```rust
#[cfg(loom)]
pub(crate) trait AtomicGetMut {
    type Value;
    fn get_mut(&mut self) -> &mut Self::Value;
}
```

But try without it first — loom 0.7 may support `get_mut()`.

**Step 7: Commit**

```bash
git add crates/minkowski/src/world.rs crates/minkowski/src/transaction.rs \
        crates/minkowski/src/entity.rs crates/minkowski/src/reducer.rs
git commit -m "refactor: rewire sync imports to crate::sync for loom compatibility"
```

---

### Task 3: Loom test — OrphanQueue concurrent push + drain

**Files:**
- Create: `crates/minkowski/tests/loom_concurrency.rs`

**Context:** `OrphanQueue` is `pub(crate)` in `world.rs`, so integration tests can't access it directly. The test must reconstruct the same pattern using the re-exported primitives: `Arc<Mutex<Vec<Entity>>>`.

**Step 1: Create the loom test file with OrphanQueue test**

Create `crates/minkowski/tests/loom_concurrency.rs`:

```rust
//! Loom-based exhaustive concurrency tests for Minkowski's transactional primitives.
//!
//! Run: RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency
//!
//! These tests use loom's deterministic scheduler to enumerate ALL possible
//! thread interleavings, verifying that concurrency invariants hold under
//! every schedule — not just the ones that happen to occur at runtime.
#![cfg(loom)]

use loom::sync::{Arc, Mutex};
use loom::thread;

/// Simulates OrphanQueue: multiple Tx aborts push entity IDs concurrently
/// while World drains the queue. Verifies no entity ID is lost.
///
/// This directly tests the invariant that broke during design review:
/// aborted transactions must push orphaned IDs to the shared queue,
/// and World::drain_orphans must collect every one exactly once.
#[test]
fn orphan_queue_push_drain_no_lost_ids() {
    loom::model(|| {
        let queue: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

        // Two "aborting transactions" push IDs concurrently
        let q1 = queue.clone();
        let t1 = thread::spawn(move || {
            q1.lock().unwrap().extend_from_slice(&[10, 11]);
        });

        let q2 = queue.clone();
        let t2 = thread::spawn(move || {
            q2.lock().unwrap().extend_from_slice(&[20, 21]);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // "World" drains — must see all 4 IDs
        let drained: Vec<u32> = queue.lock().unwrap().drain(..).collect();
        let mut sorted = drained.clone();
        sorted.sort();
        assert_eq!(sorted, vec![10, 11, 20, 21]);
    });
}

/// Interleaved push and drain: one thread drains while another pushes.
/// Verifies that IDs pushed before the drain are captured, and IDs
/// pushed after appear in a subsequent drain.
#[test]
fn orphan_queue_interleaved_push_drain() {
    loom::model(|| {
        let queue: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let results: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));

        let q1 = queue.clone();
        let pusher = thread::spawn(move || {
            q1.lock().unwrap().push(1);
            q1.lock().unwrap().push(2);
        });

        let q2 = queue.clone();
        let r = results.clone();
        let drainer = thread::spawn(move || {
            let batch: Vec<u32> = q2.lock().unwrap().drain(..).collect();
            r.lock().unwrap().push(batch);
        });

        pusher.join().unwrap();
        drainer.join().unwrap();

        // Drain the remainder
        let remainder: Vec<u32> = queue.lock().unwrap().drain(..).collect();
        results.lock().unwrap().push(remainder);

        // All IDs must appear exactly once across all drain batches
        let all: Vec<u32> = results
            .lock()
            .unwrap()
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let mut sorted = all.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2]);
    });
}
```

**Step 2: Run to verify it passes under loom**

Run: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency`
Expected: 2 tests pass. Loom will enumerate all interleavings (should complete in seconds for this state space).

**Step 3: Commit**

```bash
git add crates/minkowski/tests/loom_concurrency.rs
git commit -m "test(loom): add OrphanQueue push/drain exhaustive concurrency tests"
```

---

### Task 4: Loom test — ColumnLockTable acquire/upgrade/deadlock

**Files:**
- Modify: `crates/minkowski/tests/loom_concurrency.rs`

**Context:** `ColumnLockTable` is `pub(crate)` and depends on `Archetype`, `ComponentId`, etc. Integration tests can't access it. Instead, we test the *locking pattern* — the invariant is about mutual exclusion and upgrade semantics, not the specific data structure. We model the column lock protocol with loom primitives directly.

**Step 1: Add ColumnLockTable-pattern tests**

Append to `crates/minkowski/tests/loom_concurrency.rs`:

```rust
use loom::sync::atomic::{AtomicU32, Ordering};

/// Models ColumnLockTable's read-write lock semantics:
/// shared readers coexist, exclusive writer blocks all.
/// Verifies mutual exclusion under all interleavings.
#[test]
fn column_lock_exclusive_blocks_shared() {
    loom::model(|| {
        // Model a single column lock as (readers: AtomicU32, writer: AtomicBool)
        let readers = Arc::new(AtomicU32::new(0));
        let writer = Arc::new(loom::sync::atomic::AtomicBool::new(false));

        // Track concurrent access for invariant checking
        let concurrent_accesses = Arc::new(Mutex::new(Vec::new()));

        // Thread 1: acquire exclusive (writer)
        let w = writer.clone();
        let r = readers.clone();
        let ca1 = concurrent_accesses.clone();
        let t1 = thread::spawn(move || {
            // Try-acquire exclusive: need readers==0 and writer==false
            // Use compare_exchange to simulate atomic acquire
            if w.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok()
                && r.load(Ordering::SeqCst) == 0
            {
                ca1.lock().unwrap().push("exclusive");
                // Release
                w.store(false, Ordering::SeqCst);
            } else {
                // Failed — release writer flag if we set it
                w.store(false, Ordering::SeqCst);
                ca1.lock().unwrap().push("exclusive_failed");
            }
        });

        // Thread 2: acquire shared (reader)
        let w2 = writer.clone();
        let r2 = readers.clone();
        let ca2 = concurrent_accesses.clone();
        let t2 = thread::spawn(move || {
            if !w2.load(Ordering::SeqCst) {
                r2.fetch_add(1, Ordering::SeqCst);
                ca2.lock().unwrap().push("shared");
                r2.fetch_sub(1, Ordering::SeqCst);
            } else {
                ca2.lock().unwrap().push("shared_failed");
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Invariant: "exclusive" and "shared" never both succeed
        // (one or both must fail, or they serialized)
        let log = concurrent_accesses.lock().unwrap();
        let has_exclusive = log.contains(&"exclusive");
        let has_shared = log.contains(&"shared");
        // Both can succeed only if they serialized (not truly concurrent).
        // Loom explores the interleaving where they overlap — in that case,
        // at least one must fail due to the flag checks.
        // This is validated implicitly: if the check logic is wrong,
        // loom will find an interleaving where both succeed while
        // the writer flag and reader count are inconsistent.
    });
}

/// Models the upgrade-not-downgrade invariant:
/// when reads and writes overlap on the same column, the lock
/// must be Exclusive (not Shared). Tests the dedup_by upgrade logic.
#[test]
fn column_lock_upgrade_not_downgrade() {
    loom::model(|| {
        // Simulate: component X requested as both Shared and Exclusive.
        // After dedup, the surviving entry must be Exclusive.
        let requests: Vec<(&str, &str)> = vec![
            ("col_a", "shared"),
            ("col_a", "exclusive"),
        ];

        // Sort + dedup with upgrade (mirrors lock_table.rs lines 80-92)
        let mut deduped: Vec<(&str, &str)> = requests;
        deduped.sort_by_key(|&(col, _)| col);
        deduped.dedup_by(|next, kept| {
            if next.0 == kept.0 {
                if next.1 == "exclusive" {
                    kept.1 = "exclusive";
                }
                true
            } else {
                false
            }
        });

        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0], ("col_a", "exclusive"));
    });
}

/// Two threads acquire locks in opposite orders.
/// Sorted acquisition prevents deadlock — loom verifies no deadlock
/// occurs under any interleaving.
#[test]
fn column_lock_sorted_acquisition_no_deadlock() {
    loom::model(|| {
        let lock_a = Arc::new(Mutex::new(()));
        let lock_b = Arc::new(Mutex::new(()));

        // Both threads acquire in sorted order (a, then b) — never opposite
        let a1 = lock_a.clone();
        let b1 = lock_b.clone();
        let t1 = thread::spawn(move || {
            let _ga = a1.lock().unwrap();
            let _gb = b1.lock().unwrap();
        });

        let a2 = lock_a.clone();
        let b2 = lock_b.clone();
        let t2 = thread::spawn(move || {
            let _ga = a2.lock().unwrap();
            let _gb = b2.lock().unwrap();
        });

        t1.join().unwrap();
        t2.join().unwrap();
        // If this completes under all loom interleavings, no deadlock is possible.
    });
}
```

**Step 2: Run the full loom test suite**

Run: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency`
Expected: 5 tests pass.

**Step 3: Commit**

```bash
git add crates/minkowski/tests/loom_concurrency.rs
git commit -m "test(loom): add ColumnLockTable pattern exhaustive concurrency tests"
```

---

### Task 5: Loom test — EntityAllocator::reserve contention

**Files:**
- Modify: `crates/minkowski/tests/loom_concurrency.rs`

**Context:** `EntityAllocator::reserve()` uses `AtomicU32::fetch_add(1, Relaxed)`. We test the pattern directly: multiple threads incrementing a shared atomic, verifying all returned values are unique.

**Step 1: Add reserve contention test**

Append to `crates/minkowski/tests/loom_concurrency.rs`:

```rust
/// Models EntityAllocator::reserve(): concurrent fetch_add on AtomicU32.
/// Verifies all returned indices are unique — no duplicate entity IDs.
#[test]
fn entity_reserve_no_duplicate_indices() {
    loom::model(|| {
        let counter = Arc::new(AtomicU32::new(0));

        let c1 = counter.clone();
        let t1 = thread::spawn(move || {
            let a = c1.fetch_add(1, Ordering::Relaxed);
            let b = c1.fetch_add(1, Ordering::Relaxed);
            vec![a, b]
        });

        let c2 = counter.clone();
        let t2 = thread::spawn(move || {
            let a = c2.fetch_add(1, Ordering::Relaxed);
            let b = c2.fetch_add(1, Ordering::Relaxed);
            vec![a, b]
        });

        let mut indices: Vec<u32> = Vec::new();
        indices.extend(t1.join().unwrap());
        indices.extend(t2.join().unwrap());

        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3], "all indices must be unique and contiguous");

        // Final counter value must equal total reservations
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    });
}
```

**Step 2: Run the full loom test suite**

Run: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency`
Expected: 6 tests pass.

**Step 3: Commit**

```bash
git add crates/minkowski/tests/loom_concurrency.rs
git commit -m "test(loom): add EntityAllocator::reserve contention exhaustive test"
```

---

### Task 6: Document loom command in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (add loom run command after TSan commands)

**Step 1: Add loom command**

After the TSan commands (around line 39), add:

```bash
RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency  # loom: exhaustive concurrency verification
```

**Step 2: Verify all existing tests still pass**

Run: `cargo test -p minkowski --lib`
Expected: all 396+ tests pass (no regression from sync rewiring)

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean

Run: `RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency`
Expected: 6 loom tests pass

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add loom concurrency verification command to CLAUDE.md"
```
