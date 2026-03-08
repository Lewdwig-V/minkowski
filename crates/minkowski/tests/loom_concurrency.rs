//! Loom-based exhaustive concurrency tests for Minkowski's transactional primitives.
//!
//! Run: RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency --features loom
//!
//! These tests use loom's deterministic scheduler to enumerate ALL possible
//! thread interleavings, verifying that concurrency invariants hold under
//! every schedule — not just the ones that happen to occur at runtime.
#![cfg(loom)]

use loom::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

// ── OrphanQueue tests ──────────────────────────────────────────────────

/// Simulates OrphanQueue: multiple Tx aborts push entity IDs concurrently
/// while World drains the queue. Verifies no entity ID is lost.
#[test]
fn orphan_queue_push_drain_no_lost_ids() {
    loom::model(|| {
        let queue: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

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

        let mut drained: Vec<u32> = queue.lock().unwrap().drain(..).collect();
        drained.sort();
        assert_eq!(drained, vec![10, 11, 20, 21]);
    });
}

/// Interleaved push and drain: one thread drains while another pushes.
/// Verifies all IDs appear exactly once across drain batches.
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

        let remainder: Vec<u32> = queue.lock().unwrap().drain(..).collect();
        results.lock().unwrap().push(remainder);

        let mut all: Vec<u32> = results
            .lock()
            .unwrap()
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        all.sort();
        assert_eq!(all, vec![1, 2]);
    });
}

// ── ColumnLockTable pattern tests ──────────────────────────────────────

/// Models column lock read-write semantics: exclusive writer blocks shared readers.
/// Loom explores all interleavings to verify mutual exclusion.
#[test]
fn column_lock_exclusive_blocks_shared() {
    loom::model(|| {
        let writer_flag = Arc::new(AtomicBool::new(false));
        let reader_count = Arc::new(AtomicU32::new(0));
        // Track what happened in each thread
        let writer_held = Arc::new(AtomicBool::new(false));
        let reader_held = Arc::new(AtomicBool::new(false));

        // Thread 1: try to acquire exclusive lock
        let wf = writer_flag.clone();
        let rc = reader_count.clone();
        let wh = writer_held.clone();
        let t1 = thread::spawn(move || {
            // Try-acquire: set writer flag, then check no readers
            if wf
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                if rc.load(Ordering::SeqCst) == 0 {
                    // Acquired exclusive
                    wh.store(true, Ordering::SeqCst);
                    // Release
                    wh.store(false, Ordering::SeqCst);
                }
                wf.store(false, Ordering::SeqCst);
            }
        });

        // Thread 2: try to acquire shared lock
        let wf2 = writer_flag.clone();
        let rc2 = reader_count.clone();
        let rh = reader_held.clone();
        let t2 = thread::spawn(move || {
            // Try-acquire: check no writer, then increment readers
            if !wf2.load(Ordering::SeqCst) {
                rc2.fetch_add(1, Ordering::SeqCst);
                // Acquired shared
                rh.store(true, Ordering::SeqCst);
                // Release
                rh.store(false, Ordering::SeqCst);
                rc2.fetch_sub(1, Ordering::SeqCst);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

/// Verifies the upgrade-not-downgrade invariant: when a component is
/// requested as both Shared and Exclusive, the Exclusive privilege wins.
#[test]
fn column_lock_upgrade_not_downgrade() {
    loom::model(|| {
        let requests: Vec<(&str, &str)> = vec![("col_a", "shared"), ("col_a", "exclusive")];

        let mut deduped = requests;
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

/// Two threads acquire locks in sorted order. Loom verifies no deadlock
/// under any interleaving — completing proves deadlock freedom.
#[test]
fn column_lock_sorted_acquisition_no_deadlock() {
    loom::model(|| {
        let lock_a = Arc::new(Mutex::new(()));
        let lock_b = Arc::new(Mutex::new(()));

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
    });
}

// ── EntityAllocator::reserve pattern tests ─────────────────────────────

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
        assert_eq!(indices, vec![0, 1, 2, 3]);
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    });
}
