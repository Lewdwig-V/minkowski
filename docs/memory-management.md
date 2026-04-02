# Memory Management

RAM is fast, expensive, and limited. Minkowski provides three complementary features to help you do more with less.

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

The pool uses size-classed slab allocation (64 B to 1 MB). All worlds use mmap-backed `SlabPool` — `World::new()` creates a 256 MiB demand-paged pool, while `WorldBuilder::memory_budget()` creates a pre-faulted pool with a specific budget and optional hugepage support.

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

Together, these three features let a Minkowski deployment run indefinitely within a fixed memory envelope: the pool caps total memory, blob offloading keeps large assets external, and retention prevents unbounded entity growth.

<!-- Link definitions -->
[mmap]: https://en.wikipedia.org/wiki/Mmap
[tigerbeetle]: https://tigerbeetle.com/
[rkyv]: https://github.com/rkyv/rkyv
