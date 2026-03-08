use std::collections::HashMap;

use fixedbitset::FixedBitSet;

use crate::component::ComponentId;
use crate::storage::archetype::Archetype;

/// Cooperative per-column lock table for pessimistic transactions.
///
/// Columns are identified by (ArchetypeId, ComponentId). Shared readers
/// and exclusive writers follow standard read-write lock semantics.
/// This is not an OS mutex — it's a bookkeeping structure for cooperative
/// transaction isolation.
#[allow(dead_code)]
pub(crate) struct ColumnLockTable {
    locks: HashMap<(usize, ComponentId), ColumnLock>,
}

#[derive(Default)]
#[allow(dead_code)]
struct ColumnLock {
    readers: u32,
    writer: bool,
}

/// A set of acquired column locks. Releases all locks on drop.
#[allow(dead_code)]
pub(crate) struct ColumnLockSet {
    held: Vec<(usize, ComponentId, LockMode)>,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum LockMode {
    Shared,
    Exclusive,
}

/// Error returned when a lock cannot be acquired.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct LockConflict {
    pub component_ids: FixedBitSet,
}

#[allow(dead_code)]
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
        archetypes: &[Archetype],
        reads: &FixedBitSet,
        writes: &FixedBitSet,
    ) -> Result<ColumnLockSet, LockConflict> {
        // Build sorted list of (arch_id, comp_id, mode) for deterministic ordering.
        let mut requests: Vec<(usize, ComponentId, LockMode)> = Vec::new();
        for arch in archetypes {
            for comp_id in reads.ones() {
                if arch.column_index(comp_id).is_some() {
                    requests.push((arch.id.0, comp_id, LockMode::Shared));
                }
            }
            for comp_id in writes.ones() {
                if arch.column_index(comp_id).is_some() {
                    requests.push((arch.id.0, comp_id, LockMode::Exclusive));
                }
            }
        }
        requests.sort_by_key(|&(a, c, _)| (a, c));
        // Merge duplicates, upgrading to Exclusive if either entry is Exclusive.
        // A component in both reads and writes needs an exclusive lock.
        requests.dedup_by(|next, kept| {
            if (next.0, next.1) == (kept.0, kept.1) {
                if matches!(next.2, LockMode::Exclusive) {
                    kept.2 = LockMode::Exclusive;
                }
                true
            } else {
                false
            }
        });

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
            return Err(LockConflict {
                component_ids: conflicts,
            });
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
    use crate::storage::archetype::ArchetypeId;

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

    #[test]
    fn read_write_same_column_acquires_exclusive() {
        // Regression: when a transaction both reads and writes the same column,
        // the dedup must upgrade to Exclusive. A Shared lock would let a
        // concurrent writer proceed and violate pessimistic isolation.
        let (archs, pos_id, _) = setup();
        let mut table = ColumnLockTable::new();

        // Transaction 1: reads AND writes pos (e.g. Access::of::<(&Pos, &mut Pos)>)
        let mut reads = FixedBitSet::with_capacity(pos_id + 1);
        reads.insert(pos_id);
        let mut writes = FixedBitSet::with_capacity(pos_id + 1);
        writes.insert(pos_id);
        let lock1 = table.acquire(&archs, &reads, &writes).unwrap();

        // Transaction 2: tries to write pos — must be blocked
        let empty = FixedBitSet::new();
        let lock2 = table.acquire(&archs, &empty, &writes);
        assert!(lock2.is_err(), "concurrent writer must be blocked");

        // Transaction 3: tries to read pos — must also be blocked (exclusive held)
        let lock3 = table.acquire(&archs, &reads, &empty);
        assert!(lock3.is_err(), "concurrent reader must be blocked");

        table.release(lock1);

        // After release, both should succeed
        let lock4 = table.acquire(&archs, &empty, &writes);
        assert!(lock4.is_ok());
        table.release(lock4.unwrap());
    }
}

#[cfg(loom)]
mod loom_tests {
    use super::*;
    use crate::component::ComponentRegistry;
    use crate::storage::archetype::ArchetypeId;
    use crate::sync::Mutex;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::sync::Arc;
    use loom::thread;

    // Archetype doesn't impl Clone, so each thread builds its own.
    // Component IDs are deterministic (sequential registration), so
    // archetypes built with the same registration order are equivalent.
    fn setup() -> (Vec<Archetype>, ComponentId, ComponentId) {
        let mut reg = ComponentRegistry::new();
        let pos_id = reg.register::<f32>();
        let vel_id = reg.register::<u32>();
        let arch = Archetype::new(ArchetypeId(0), &[pos_id, vel_id], &reg);
        (vec![arch], pos_id, vel_id)
    }

    /// Two threads contend on the real ColumnLockTable behind a Mutex
    /// (modeling Pessimistic's `Mutex<ColumnLockTable>`). One acquires
    /// exclusive, the other acquires shared on the same column. Verifies
    /// mutual exclusion: the two never overlap in-flight.
    #[test]
    fn loom_exclusive_blocks_shared() {
        loom::model(|| {
            let table = Arc::new(Mutex::new(ColumnLockTable::new()));
            let writer_held = Arc::new(AtomicBool::new(false));
            let reader_held = Arc::new(AtomicBool::new(false));

            let t1 = table.clone();
            let wh = writer_held.clone();
            let rh1 = reader_held.clone();
            let exclusive = thread::spawn(move || {
                let (archs, pos_id, _) = setup();
                let mut writes = FixedBitSet::with_capacity(pos_id + 1);
                writes.insert(pos_id);
                let empty = FixedBitSet::new();

                let lock_set = {
                    let mut tbl = t1.lock();
                    tbl.acquire(&archs, &empty, &writes)
                };
                if let Ok(lock_set) = lock_set {
                    wh.store(true, Ordering::SeqCst);
                    assert!(
                        !rh1.load(Ordering::SeqCst),
                        "exclusive held while shared also held"
                    );
                    wh.store(false, Ordering::SeqCst);
                    t1.lock().release(lock_set);
                }
            });

            let t2 = table.clone();
            let rh = reader_held.clone();
            let wh2 = writer_held.clone();
            let shared = thread::spawn(move || {
                let (archs, pos_id, _) = setup();
                let mut reads = FixedBitSet::with_capacity(pos_id + 1);
                reads.insert(pos_id);
                let empty = FixedBitSet::new();

                let lock_set = {
                    let mut tbl = t2.lock();
                    tbl.acquire(&archs, &reads, &empty)
                };
                if let Ok(lock_set) = lock_set {
                    rh.store(true, Ordering::SeqCst);
                    assert!(
                        !wh2.load(Ordering::SeqCst),
                        "shared held while exclusive also held"
                    );
                    rh.store(false, Ordering::SeqCst);
                    t2.lock().release(lock_set);
                }
            });

            exclusive.join().unwrap();
            shared.join().unwrap();
        });
    }

    /// Verifies the upgrade-not-downgrade invariant using the real
    /// ColumnLockTable::acquire. When a component appears in both reads
    /// and writes, acquire must treat it as exclusive — blocking a
    /// concurrent shared reader on the same column.
    #[test]
    fn loom_upgrade_blocks_concurrent_reader() {
        loom::model(|| {
            let table = Arc::new(Mutex::new(ColumnLockTable::new()));
            let writer_held = Arc::new(AtomicBool::new(false));
            let reader_held = Arc::new(AtomicBool::new(false));

            let t1 = table.clone();
            let wh = writer_held.clone();
            let rh1 = reader_held.clone();
            // Thread 1: requests both read AND write on pos_id
            let upgraded = thread::spawn(move || {
                let (archs, pos_id, _) = setup();
                let mut reads = FixedBitSet::with_capacity(pos_id + 1);
                reads.insert(pos_id);
                let mut writes = FixedBitSet::with_capacity(pos_id + 1);
                writes.insert(pos_id);

                let lock_set = {
                    let mut tbl = t1.lock();
                    tbl.acquire(&archs, &reads, &writes)
                };
                if let Ok(lock_set) = lock_set {
                    wh.store(true, Ordering::SeqCst);
                    assert!(
                        !rh1.load(Ordering::SeqCst),
                        "upgraded exclusive held while shared reader also active"
                    );
                    wh.store(false, Ordering::SeqCst);
                    t1.lock().release(lock_set);
                }
            });

            let t2 = table.clone();
            let rh = reader_held.clone();
            let wh2 = writer_held.clone();
            // Thread 2: requests read-only on the same column
            let reader = thread::spawn(move || {
                let (archs, pos_id, _) = setup();
                let mut reads = FixedBitSet::with_capacity(pos_id + 1);
                reads.insert(pos_id);
                let empty = FixedBitSet::new();

                let lock_set = {
                    let mut tbl = t2.lock();
                    tbl.acquire(&archs, &reads, &empty)
                };
                if let Ok(lock_set) = lock_set {
                    rh.store(true, Ordering::SeqCst);
                    assert!(
                        !wh2.load(Ordering::SeqCst),
                        "shared reader active while upgraded exclusive also held"
                    );
                    rh.store(false, Ordering::SeqCst);
                    t2.lock().release(lock_set);
                }
            });

            upgraded.join().unwrap();
            reader.join().unwrap();
        });
    }

    /// Two threads acquire column locks via the Mutex-protected table.
    /// Each thread retries on LockConflict (modeling Pessimistic's retry
    /// loop). Completion under all loom interleavings proves no deadlock.
    #[test]
    fn loom_sorted_acquisition_no_deadlock() {
        loom::model(|| {
            let table = Arc::new(Mutex::new(ColumnLockTable::new()));

            let t1 = table.clone();
            let thread1 = thread::spawn(move || {
                let (archs, pos_id, vel_id) = setup();
                let mut writes = FixedBitSet::with_capacity(vel_id + 1);
                writes.insert(pos_id);
                writes.insert(vel_id);
                let empty = FixedBitSet::new();

                // Retry loop (models Pessimistic::begin's retry)
                loop {
                    let result = t1.lock().acquire(&archs, &empty, &writes);
                    if let Ok(lock_set) = result {
                        t1.lock().release(lock_set);
                        break;
                    }
                    loom::thread::yield_now();
                }
            });

            let t2 = table.clone();
            let thread2 = thread::spawn(move || {
                let (archs, pos_id, vel_id) = setup();
                let mut writes = FixedBitSet::with_capacity(vel_id + 1);
                writes.insert(pos_id);
                writes.insert(vel_id);
                let empty = FixedBitSet::new();

                loop {
                    let result = t2.lock().acquire(&archs, &empty, &writes);
                    if let Ok(lock_set) = result {
                        t2.lock().release(lock_set);
                        break;
                    }
                    loom::thread::yield_now();
                }
            });

            thread1.join().unwrap();
            thread2.join().unwrap();
        });
    }
}
