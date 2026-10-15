# Memory Management

RAM is fast, expensive, and limited. Minkowski provides four complementary features to help you do more with less.

## Pre-allocated Memory Pool

`WorldBuilder` creates a World backed by a single [mmap][mmap] region — all memory is allocated upfront at startup, [TigerBeetle][tigerbeetle]-style. After initialization, `try_spawn` returns `Err(PoolExhausted)` and `try_insert` returns `Err(InsertError)` instead of crashing the process. No dynamic allocation on the data path.

```rust
use minkowski::{World, HugePages, PoolExhausted};

let mut world = World::builder()
    .memory_budget(64 * 1024 * 1024)   // 64 MB fixed budget
    .hugepages(HugePages::Try)          // 2 MiB pages if available
    .lock_all_memory(true)              // mlockall — prevent swapping
    .build()?;

// Spawn until the pool is full — no OOM kill, just a Result
match world.try_spawn((pos, vel)) {
    Ok(entity) => { /* success */ }
    Err(PoolExhausted { .. }) => { /* shed load, evict, or report */ }
}
```

The pool is one window with two opposing frontiers. The bottom frontier grows upward and carves 64 KiB chunks on demand for five slab size classes (64 B – 64 KiB). The top frontier grows downward and serves anything larger as runs of 1 MiB units, so one column can grow to any size the budget allows. Nothing is reserved per size class up front: `PoolExhausted` means the frontiers have met (or every carved block of the needed class is in use), not that one fixed partition ran out. Large runs grow in place when free units border them, which avoids the transient 2× footprint of copy-on-grow. All worlds use the mmap-backed `SlabPool`: `World::new()` creates a 256 MiB demand-paged pool that touches no pages until the first allocation, while `WorldBuilder::memory_budget()` creates a pre-faulted pool with a specific budget and optional hugepage support. Exhaustion is deterministic for single-threaded use of a pool. When several threads share one, thread-local caches and lock order can move the exact point where `PoolExhausted` fires, so call `flush_pool_caches()` before any budget-sensitive check. See [the design doc](plans/two-frontier-pool.md) for the layout and measured utilization.

## Blob Offloading

Large per-entity assets (images, meshes, audio) shouldn't live in column storage. `BlobRef` is a lightweight component that stores an external key (S3 path, URL, content hash). The `BlobStore` trait provides a cleanup hook — after despawning entities, collect orphaned refs and let your store delete the external data.

```rust
use minkowski_persist::{BlobRef, BlobStore};

// Entity holds a key, not the bytes
world.spawn((metadata, BlobRef::new("s3://bucket/mesh-00af.bin")));

// After despawn, clean up external storage
store.on_orphaned(&orphaned_refs);
```

`BlobRef` lives in `minkowski-persist` for [rkyv][rkyv] serialization support — blob references survive snapshots and WAL replay. The engine stores only the reference; blob lifecycle is the user's responsibility (same external composition pattern as [indexes](indexing.md)).

## Retention

`Expiry` is a countdown component — it counts retention dispatches, not ticks or wall-clock time. Each call to `registry.run()` decrements all counters by one and despawns entities that reach zero. You control how often retention runs; the engine never runs it automatically.

```rust
use minkowski::{World, Expiry, ReducerRegistry};

let mut world = World::new();
let mut registry = ReducerRegistry::new();
let retention_id = registry.retention(&mut world);

// This entity survives 5 retention dispatches
world.spawn((data, Expiry::after(5)));

// Each call is one "retention cycle" — counters decrement, zeros despawn
registry.run(&mut world, retention_id, ());
```

Together, these four features let a Minkowski deployment run indefinitely within a fixed memory envelope: the pool caps total RAM, blob offloading keeps large assets external, retention prevents unbounded entity growth, and LSM compaction bounds disk-space amplification to ≤ 2×.

## LSM Disk-Space Management

The `minkowski-lsm` crate adds a fourth dimension to memory management: **incremental disk persistence with bounded space amplification**. Where the pool caps in-memory RAM, blob offloading keeps large assets external, and retention prevents unbounded entity growth, LSM compaction prevents unbounded disk usage.

### Dirty-Page Tracking

`storage::dirty_pages` (in the core crate) provides per-column page-level dirty bitsets (256 rows/page). Every `BlobVec` mutation path marks affected pages dirty. On flush, only dirty pages are written to a new sorted-run file — persistence cost is proportional to the mutation rate, not the world size.

```rust
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::types::SeqRange;

// Write only the pages that changed since last flush
let path = flush_and_record(&world, seq_range, &mut manifest, &mut log, &dir)?;
// path is Some if dirty pages were written, None if the world was clean
```

### Compaction

Each flush creates a new L1 sorted run. Over time, L1 accumulates overlapping runs for the same archetypes. `compact_one` merges runs atomically — either all input runs are replaced by the output, or none are (via the `CompactionCommit` manifest log entry).

```rust
use minkowski_lsm::{compact_one, COMPACTION_TRIGGER};

// Compact when L1 exceeds the trigger threshold
if let Some(report) = compact_one(&mut manifest, &mut log, &dir)? {
    println!("Merged {} runs → {} (freed {} bytes)",
             report.input_count, report.output_count, report.bytes_freed);
}
```

Space amplification is bounded at ≤ 2× during compaction (old and new runs coexist briefly). After compaction completes, orphaned files are cleaned up via `cleanup_orphans`.

### Crash Safety

Sorted-run files are immutable once written. A partial flush produces an orphan file not referenced by the manifest — `cleanup_orphans` removes it on the next startup. The manifest log uses CRC32 frames with atomic `CompactionCommit` entries, so recovery replays a consistent prefix of the log.

### What's Next

Phase 5 (pending) will add `LsmRecovery` (restore World from sorted runs + WAL tail) and `Durable<S>` integration (automatic LSM flush instead of full snapshots on checkpoint). Until then, the LSM crate provides flush and compaction primitives that can be called manually.

<!-- Link definitions -->
[mmap]: https://en.wikipedia.org/wiki/Mmap
[tigerbeetle]: https://tigerbeetle.com/
[rkyv]: https://github.com/rkyv/rkyv
