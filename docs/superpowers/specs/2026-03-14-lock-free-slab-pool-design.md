# Lock-Free Slab Pool Allocator

**Date**: 2026-03-14
**Status**: Approved
**Scope**: `crates/minkowski/src/pool.rs`, `Cargo.toml` (new `atomic` dependency)

## Problem

The slab pool allocator (`SlabPool`) uses `Mutex<Vec<*mut u8>>` per size class
for its free lists. Benchmarks show this is 4.3x slower than the system
allocator for spawn workloads and 6.3x slower for migration, even with zero
contention (single-threaded). The mutex lock/unlock overhead and Vec indirection
dominate allocation cost.

Additionally, when a size class is exhausted, the pool returns `PoolExhausted`
even when larger size classes have free blocks. This causes premature allocation
failures in workloads with non-uniform component sizes.

## Solution

Replace the mutex-guarded `Vec` free lists with lock-free intrusive linked
stacks using tagged pointers (`Atomic<u128>`), add a side table for deallocation
class routing, and implement single-step overflow from exhausted classes to the
next larger class.

## Design

### Lock-Free Intrusive Stack

Each size class's free list becomes a singly-linked stack. The stack head is a
single `Atomic<u128>` containing a tagged pointer:

```rust
#[repr(C)]
struct TaggedPtr {
    ptr: u64,   // pointer to top block (0 = empty)
    tag: u64,   // monotonic counter, incremented on every push/pop
}
```

Free blocks form an intrusive linked list: the first 8 bytes of each free block
store a `*mut u8` pointing to the next free block (null for the tail). No
separate heap allocation is needed for the list — the metadata lives inside the
free blocks themselves.

**Invariant**: `MIN_BLOCK_SIZE >= size_of::<*mut u8>()` (8 bytes). The smallest
size class is 64 bytes, so this is satisfied with room to spare. If size classes
are ever changed, this invariant must be maintained or the intrusive list will
corrupt block data.

**ABA prevention**: The 64-bit tag increments on every CAS (push or pop). Even
if a pointer is recycled to the same address, the tag will differ, causing a
stale CAS to fail. With 64 bits, wraparound requires 18 quintillion operations
per size class — effectively impossible.

**Allocate (pop)**:

```
loop {
    head = atomic_load(head_slot, Acquire)
    if head.ptr == 0 { try overflow or return Err(PoolExhausted) }
    next = ptr::read(head.ptr as *const u64)  // aligned: blocks are ≥64-byte aligned
    new_head = TaggedPtr { ptr: next, tag: head.tag + 1 }
    if CAS(head_slot, head, new_head, AcqRel, Acquire) {
        update side table
        fetch_add(used_bytes, block_size, Relaxed)
        return Ok(head.ptr)
    }
}
```

**Deallocate (push)**:

```
actual_class = side_table[(ptr - base) / MIN_BLOCK_SIZE]
loop {
    head = atomic_load(heads[actual_class], Acquire)
    write head.ptr into block's first 8 bytes  // set next-pointer
    new_head = TaggedPtr { ptr: block, tag: head.tag + 1 }
    if CAS(heads[actual_class], head, new_head, AcqRel, Acquire) {
        fetch_sub(used_bytes, SIZE_CLASSES[actual_class], Relaxed)
        return
    }
}
```

**Memory ordering**: `Acquire` on load ensures visibility of the next-pointer
written by the thread that last pushed the block. `AcqRel` on successful CAS
publishes the new next-pointer to future consumers.

**Next-pointer visibility proof**: On push (deallocate), the thread writes the
next-pointer into the block *before* the `AcqRel` CAS that publishes the block
as the new head. On pop (allocate), the `Acquire` load of the head
synchronizes-with that CAS, so the next-pointer write is guaranteed visible
before it is read. This is the standard release-acquire pair for lock-free
stacks.

**Side table write ordering**: The side table entry is written *after* the
successful CAS on allocation. At this point the block is owned by the
allocating thread (removed from the free list), so no other thread can access
the same entry. On deallocation, the side table is read *before* the CAS that
returns the block — again, single-owner. No atomics needed on side table
entries.

### Side Table for Deallocation Routing

A separate byte array tracks which size class each block was actually allocated
from. This is necessary because overflow allocations return blocks from a
different class than the caller's `Layout` would imply.

```rust
side_table: Vec<u8>  // indexed by (ptr - base) / MIN_BLOCK_SIZE
```

- Allocated once at pool construction on the system heap.
- Size: `total_bytes / MIN_BLOCK_SIZE` entries (1 byte each). For a 64 MB pool
  with 64-byte minimum block, this is 1 MB (~1.5% overhead).
- On allocation: `side_table[index] = actual_class` (the class the block came
  from, which may differ from the requested class due to overflow).
- On deallocation: `actual_class = side_table[index]` — routes the block back
  to the correct free list regardless of what `Layout` the caller passes.

The side table write (on allocation) and read (on deallocation) are not
contended — each block is owned by exactly one thread at a time between
allocation and deallocation. No atomic operations needed on the side table
entries.

### Overflow Policy

When a size class is exhausted, try the next larger class (one step up):

```
64B → 256B → 1KB → 4KB → 64KB → 1MB → PoolExhausted
```

Maximum internal fragmentation: 16x (a 4KB request served from a 64KB block).
This is justified by:

1. **Temporary nature** — overflow blocks are a pressure valve until the
   original class gets deallocations back. Steady-state workloads rarely
   overflow.
2. **Avoids kernel calls** — using existing pool memory avoids `mmap`/`brk`
   latency mid-transaction.
3. **Low frequency** — only a small fraction of active allocations live in
   overflow slots at any given time.

The side table ensures correct deallocation routing: an overflow block allocated
from class 3 (4KB) for a class 2 (1KB) request is returned to class 3's free
list, not class 2's.

### Initialization

`SlabPool::new()` chains blocks as a linked list directly in the mmap region:

```
for each size class:
    for i in 0..block_count:
        block_ptr = base + offset + (i * block_size)
        next_ptr = if i < block_count - 1 { block_ptr + block_size } else { null }
        write next_ptr at block_ptr[0..8]
    heads[class] = Atomic<u128>::new(TaggedPtr { ptr: first_block, tag: 0 })
```

All blocks are chained in address order (low → high) for cache locality on
early allocations. The linked list is fully formed before any thread can access
the pool — no synchronization needed during construction.

Side table entries are initialized to `0xFF` (unallocated sentinel). Each
allocation writes the actual class index.

### Struct Changes

```rust
struct SlabPool {
    _region: MmapRegion,
    base: *mut u8,
    total: usize,
    heads: [Atomic<u128>; NUM_SIZE_CLASSES],   // was: [Mutex<Vec<*mut u8>>; NUM_SIZE_CLASSES]
    side_table: Vec<u8>,                      // NEW: deallocation class routing
    used_bytes: AtomicUsize,                  // unchanged
}
```

`parking_lot::Mutex` is no longer used by the pool. It remains in the workspace
for `ColumnLockTable` and `OrphanQueue`.

### Loom Compatibility

The `atomic` crate's `Atomic<u128>` is not supported by loom. Under `cfg(loom)`,
the tagged pointer is replaced with `loom::sync::Mutex<u128>`:

```rust
#[cfg(not(loom))]
type AtomicHead = atomic::Atomic<u128>;

#[cfg(loom)]
type AtomicHead = loom::sync::Mutex<u128>;
```

Helper methods `load_head()` and `cas_head()` abstract over the two
implementations. The allocate/deallocate logic is written once against these
helpers.

**Loom limitation**: Under the Mutex shim, `cas_head()` always succeeds if the
comparison matches — it cannot model spurious CAS failure or the retry loop
under contention. Loom verifies *logical correctness* (no lost blocks, no
duplicates, no ABA) but not the *retry path*. The CAS retry path is covered by
std thread concurrency tests (real contention) and Miri + TSan (memory
ordering). This is the same limitation as the existing loom tests for
`crossbeam-epoch` in rayon.

### Overflow Telemetry

Each size class tracks overflow activity via two counters:

```rust
overflow_active: [AtomicU64; NUM_SIZE_CLASSES],  // currently serving overflow
overflow_total: [AtomicU64; NUM_SIZE_CLASSES],    // cumulative overflow count
```

To support decrementing `overflow_active` on deallocation, the side table entry
stores both the actual class *and* an overflow flag:

```rust
// Side table entry: bits [0..3] = actual class index, bit 7 = overflow flag
side_table: Vec<u8>,
```

On overflow allocation: set the overflow bit, increment both `overflow_active`
and `overflow_total` for the *actual* class. On deallocation: if the overflow
bit is set, decrement `overflow_active`. This gives both a real-time gauge
(`overflow_active`) and a cumulative counter (`overflow_total`).

Exposed via `SlabPool::overflow_active(class) -> u64` and
`SlabPool::overflow_total(class) -> u64`. Surfaced through `WorldStats` for
observability.

This gives users the data to tune `PROPORTIONS` — if class 4 (64KB) consistently
has high `overflow_active` counts, the user should increase class 3 (4KB)
proportion. Automated rebalancing (subdividing overflow blocks at runtime) is a
future direction that would require dynamic block management within the fixed
mmap region — see Future Directions.

## API Changes

The `PoolAllocator` trait signature is unchanged. The `deallocate` contract is
*relaxed*: the side table routes blocks to the correct free list regardless of
the `Layout` passed by the caller. Previously, passing a mismatched `Layout`
would return the block to the wrong class (silent corruption). Now it is
harmless — the side table is authoritative. The SAFETY doc on `deallocate`
should be updated to reflect this: the `Layout` parameter is accepted for API
compatibility but not used for class routing.

`WorldStats` gains `pool_overflow_active: Option<[u64; 6]>` and
`pool_overflow_total: Option<[u64; 6]>` for observability.

## Dependencies

**Added**: `atomic` crate (provides portable `Atomic<u128>` — `cmpxchg16b` on
x86-64, `ldxp`/`stxp` on AArch64). Zero transitive dependencies, `no_std`
compatible.

## Testing Strategy

### Unit Tests (replace existing)

- Allocate/deallocate round-trip per size class.
- Exhaust a class, verify `PoolExhausted`.
- Overflow: exhaust class N, verify class N+1 serves the request.
- Overflow deallocation: verify overflowed block returns to the correct class.
- Exhaust all classes including overflow, verify `PoolExhausted`.
- `used_bytes` accounting through overflow and deallocation.
- Zero-size and oversized request handling (unchanged behavior).
- Side table correctness: verify class index written on alloc, read on dealloc.

### Std Thread Concurrency Tests

- N threads (4-8) racing to allocate from one class until exhaustion. Verify:
  no duplicate pointers, total equals class capacity.
- N threads doing interleaved allocate/deallocate (1000 iterations). Verify: no
  lost blocks, `used_bytes` correct.
- Concurrent overflow: multiple threads exhaust a class simultaneously, verify
  overflow routing is consistent.

### Loom Tests (replace existing 2)

- Two threads concurrent pop from same head — no duplicates, no ABA.
- Two threads concurrent push to same head — no lost blocks.
- One push + one pop concurrently — list consistency.
- Overflow under contention: one thread exhausts a class while another
  allocates from it — verify overflow triggers correctly.

### Benchmark Validation

- `simple_insert/pool`: target within 1.5x of `simple_insert/batch` (currently
  4.3x).
- `add_remove/pool`: target within 2x of `add_remove/add_remove` (currently
  6.3x).

## Performance Target

| Benchmark | Current (Mutex) | Target (Lock-Free) | System Alloc |
|---|---|---|---|
| `simple_insert/pool` | 7.54 ms | < 2.6 ms | 1.74 ms |
| `add_remove/pool` | 8.24 ms | < 2.6 ms | 1.30 ms |

## Rollback

Single-commit revert. The `PoolAllocator` trait boundary means nothing outside
`pool.rs` is affected. The `atomic` dependency can be removed on revert.

## Non-Goals

- Lock-free `OrphanQueue` or `ColumnLockTable` — separate concern, uses
  `parking_lot::Mutex` for different reasons (cooperative locking semantics).
- Custom allocator trait (`std::alloc::Allocator`) — unstable, and our
  `PoolAllocator` trait is simpler for the use case.

## Future Directions

### Runtime Superblock Subdivision

When overflow telemetry shows persistent pressure on a size class, the pool
could subdivide a larger block into smaller ones at runtime — a slab-to-buddy
transition. The approach:

1. **Superblock model**: Divide the mmap region into fixed superblocks (e.g.
   1MB). The side table tracks class ownership per superblock.
2. **Atomic swap**: Mark the superblock as BUSY, initialize sub-slot free list,
   then atomically re-tag it to the smaller class.
3. **Hysteresis**: Only subdivide large→small within a frame. Re-merging (small
   back to large) only during a dedicated maintenance phase when all sub-slots
   are confirmed empty. This prevents oscillation.
4. **Concurrency**: A radix tree with atomic leaf pointers allows subdivision
   of one superblock while other threads allocate from other superblocks
   without contention.

This is deferred because it adds significant complexity (hierarchical bitmap or
radix tree, merge logic, oscillation prevention) and the single-step overflow
policy handles the common case. The overflow telemetry from this spec provides
the signal to decide whether subdivision is needed in practice.
