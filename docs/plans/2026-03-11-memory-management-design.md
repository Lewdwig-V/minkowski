# Memory Management Design

Three independent features unified by a single theme: RAM is fast, expensive,
and limited — help users do more with less, allocate as fast as possible, and
handle exhaustion gracefully.

Design philosophy: **What Would TigerBeetle Do?** Fail fast at startup, never
at 3am. Pre-allocate everything. Return errors, don't panic.

## Feature 1: Slab Pool Allocator

### Motivation

Today every internal allocator (BlobVec, Arena, EntityAllocator, PagedSparseSet)
calls `std::alloc` directly and panics on OOM via `handle_alloc_error`. There is
no memory budget, no pre-allocation, no graceful degradation. A single runaway
archetype can exhaust system memory and crash the process.

### Design

A single `mmap`'d region pre-allocated at startup, managed via size-classed free
lists. All internal allocators draw from this pool instead of `std::alloc`.

#### Backing region: `MmapRegion`

```rust
struct MmapRegion {
    ptr: NonNull<u8>,
    size: usize,
    huge: bool, // whether hugepages were actually granted
}
```

Created via `libc::mmap` with:
- `MAP_ANONYMOUS | MAP_PRIVATE` — anonymous mapping, no file backing.
- `MAP_POPULATE` — pre-fault all pages at creation. If the system cannot back
  the mapping with physical RAM, `mmap` fails immediately. This is the
  TigerBeetle insight: fail at startup, not hours into production when pages
  fault lazily under kernel overcommit.
- `MAP_HUGETLB` (optional) — request 2MB hugepages for TLB efficiency.

TLB math for a 2 GB pool:

| Page size | Entries needed | L2 TLB (~2K entries) | Coverage |
|-----------|---------------|---------------------|----------|
| 4 KB      | 524,288       | 0.4%                | Heavy misses on random access |
| 2 MB      | 1,024         | 100%                | Full pool TLB-resident |
| 1 GB      | 2             | Trivial             | Perfect |

Linear column sweeps (`for_each_chunk`) have good spatial locality regardless.
Random access (entity lookup, archetype migration, sparse lookups) benefits
significantly from hugepages.

#### Allocator trait: `PoolAllocator`

```rust
pub trait PoolAllocator: Send + Sync {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}
```

Two implementations:

1. **`SystemAllocator`** — delegates to `std::alloc`. Current behavior. Default
   for `World::new()`.
2. **`SlabPool`** — size-classed free lists over an `MmapRegion`. Returns
   `Err(PoolExhausted)` when the pool is exhausted.

#### Size classes

Tuned for ECS component patterns:

| Class | Size | Typical use |
|-------|------|-------------|
| 0     | 64 B | Small components (Vec2, scalars, enums) |
| 1     | 256 B | Medium components, entity location entries |
| 2     | 1 KB | Sparse pages, small column segments |
| 3     | 4 KB | OS page, column growth |
| 4     | 64 KB | Large column segments |
| 5     | 1 MB | Bulk column pre-allocation |

Each size class is a singly-linked free list threaded through the free blocks
themselves (zero overhead). Allocation = pop head. Deallocation = push head.

Requests larger than 1 MB are served from a large-block region (linked list of
variable-size blocks, first-fit).

#### Builder API

```rust
let world = World::builder()
    .memory_budget(2 << 30)       // 2 GB
    .hugepages(HugePages::Try)    // Try 2MB, fall back to 4KB
    .build()?;                    // Err if mmap fails

// Default — current behavior, no budget:
let world = World::new();
```

`HugePages::Try` attempts `MAP_HUGETLB`, falls back silently.
`HugePages::Require` fails if hugepages unavailable.
`HugePages::Off` uses regular 4KB pages.

#### Error propagation

Allocation-heavy operations gain `try_` variants:

```rust
world.try_spawn((Pos(0.0), Vel(1.0)))?;  // Result<Entity, PoolExhausted>
world.spawn((Pos(0.0), Vel(1.0)));        // panics if exhausted (backwards compat)
```

Affected methods: `spawn`, `insert`, `remove` (migration allocates in target
archetype), `changeset.apply`, `CommandBuffer.apply`.

#### Subsumption of existing allocators

BlobVec and Arena currently manage their own `std::alloc` calls with growth
logic. After this change, they request/release blocks from the shared pool. The
pool is the single backing allocator — BlobVec and Arena become thin wrappers
that manage typed views over pool-allocated blocks.

#### Thread safety

`SlabPool` is `Arc<SlabPool>` owned by World and threaded to all internal
allocators. Free lists use atomic operations (lock-free push/pop) for concurrent
access from rayon `par_for_each` and transaction strategies.

#### Observability

```rust
pub struct WorldStats {
    // ... existing fields ...
    pub pool_capacity: Option<usize>,  // None if SystemAllocator
    pub pool_used: Option<usize>,
    pub pool_free: Option<usize>,
}
```

#### Platform support

- **Linux**: full support (mmap + MAP_POPULATE + MAP_HUGETLB)
- **macOS**: mmap + MAP_POPULATE equivalent (`MAP_PREFAULT`), no hugepages
- **Windows**: `VirtualAlloc` + `MEM_COMMIT` + `MEM_LARGE_PAGES`
- **Fallback**: `HeapPool` using `std::alloc` (testing, unsupported platforms)

---

## Feature 2: Blob Offloading

### Motivation

Components in Minkowski are typically small (8-32 bytes). But as a general
in-memory database, users may want to associate large binary data (images,
documents, serialized models) with entities. Storing multi-megabyte blobs in
BlobVec columns wastes pool memory and destroys cache locality for neighboring
components.

### Design

A `BlobRef` component type that holds a reference (URL/key) to data in an
external object store (S3, MinIO, local filesystem). The ECS stores only the
reference. The blob bytes never enter the World.

#### `BlobRef` component

```rust
/// Reference to an externally-stored blob.
/// The ECS stores only this reference — blob bytes live outside the World.
/// Persistence serializes the key string, not the remote blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRef(pub String);
```

`BlobRef` is a regular `Component` (`'static + Send + Sync`). Stored in
archetypes like any other component. In pool mode, only the `String` key
consumes pool memory.

#### `BlobStore` lifecycle trait

```rust
/// Lifecycle hook for external blob storage.
/// Same composition pattern as SpatialIndex — the engine provides hook points,
/// the user provides policy.
pub trait BlobStore {
    /// Called with blob references no longer attached to any live entity.
    /// The implementor is responsible for deleting the external blobs.
    fn on_orphaned(&mut self, refs: &[&BlobRef]);
}
```

Key decisions:

- **No async** — `on_orphaned` is sync. Users who need async S3 deletion use
  `block_on` or buffer keys for a background task.
- **No auto-invocation** — the engine never calls `on_orphaned`. The user writes
  a cleanup reducer or framework hook. Same responsibility model as
  `SpatialIndex::rebuild`.
- **No generic key type** — `String` covers S3 keys, MinIO paths, URLs, URNs.
  Avoids generic machinery for marginal gain.

#### Persistence integration

The codec registry serializes `BlobRef` as a normal `String` component via
rkyv. No special handling needed. On snapshot restore, blob keys are restored
but the remote blobs must still exist — restoring a snapshot does NOT restore
remote blobs.

Replication: `BlobRef` components replicate as key strings. The receiving node
must have access to the same object store (or a replica of it).

---

## Feature 3: Expiry & Retention Reducer

### Motivation

Without cleanup, entity count grows monotonically. In pool mode, this means the
pool drains until `PoolExhausted`. Even without a pool, unbounded growth is a
resource leak. Users need a built-in primitive for expiring old data.

### Design

An `Expiry` component marks entities for automatic despawn at a target tick. A
built-in `RetentionReducer` scans for expired entities and despawns them in
batch.

#### `Expiry` component

```rust
/// Marks an entity for despawn when the world tick reaches this value.
/// Set at spawn time. The tick is monotonic (from change detection),
/// not wall-clock time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expiry(pub ChangeTick);
```

For time-based TTL, users convert duration to ticks based on their tick rate:
`Expiry(ChangeTick::from_raw(now.to_raw() + ticks_per_sec * seconds))`.

#### `RetentionReducer`

A new built-in reducer type in `ReducerRegistry`:

```rust
// Registration:
let retention_id = registry.retention(&mut world);

// Dispatch (scheduled query reducer):
registry.run(retention_id, &mut world);
```

Internally:

1. Queries `(Entity, &Expiry)` — read-only scan, no mutation of `Expiry`.
2. Collects entities where `expiry.0.to_raw() <= current_tick.to_raw()`.
3. Batch despawn via `world.despawn_batch()`.

Key decisions:

- **Read-only scan** — `Expiry` is never mutated by the reducer. No per-tick
  write noise. `Changed<Expiry>` only fires on insertion.
- **User controls dispatch** — not automatic. The user calls
  `registry.run(retention_id, &mut world)` at their preferred frequency.
- **Batch despawn** — avoids archetype migration churn during the scan.
- **Composable** — `RetentionReducer` declares `Access` (reads `Expiry`,
  despawns). The scheduler detects conflicts with other reducers.

#### Memory pool interaction

In pool mode, despawning returns slab blocks to free lists. The retention
reducer is the primary mechanism for reclaiming pool memory in churny workloads.
Without cleanup, the pool drains monotonically.

---

## How the features connect

```
                    ┌─────────────────┐
                    │  Memory Pool    │
                    │  (SlabPool)     │
                    │  finite budget  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  BlobRef   │  │  Expiry +  │  │  try_spawn │
     │  offload   │  │  Retention │  │  try_insert│
     │  large     │  │  reclaim   │  │  error     │
     │  data      │  │  pool mem  │  │  propagate │
     └────────────┘  └────────────┘  └────────────┘
      keep pool       prevent         handle
      lean             drain           exhaustion
```

The pool bounds total memory. Retention reclaims it. Blob refs keep large data
out. Error propagation handles the case where all three aren't enough.

---

## Scope exclusions

- **Tiered storage** (hot/cold archetype offloading) — different problem, out of
  scope.
- **Async blob I/O** — Minkowski is sync-only. Users bridge to async runtimes.
- **Automatic blob lifecycle** — deletion is user/framework responsibility.
- **Wall-clock TTL** — ticks are monotonic counters, not time. Users map time to
  ticks.
- **Memory compaction / defragmentation** — slab free lists handle fragmentation
  within size classes. Cross-class compaction is out of scope.
