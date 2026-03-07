# Paged Sparse Set + Batched Despawn

**Date:** 2026-03-07

Two `pub(crate)` optimizations to internal storage. No public API changes except `World::despawn_batch`.

---

## 1. Paged Sparse Set

### Problem

Sparse component storage uses `HashMap<Entity, T>` behind `Box<dyn Any>`. Hash probe per lookup, allocator pressure per insert, poor cache locality during iteration. Type erasure via `dyn Any` requires a downcast on every access.

### Design

Replace `HashMap<Entity, T>` with a paged sparse set backed by BlobVec.

```rust
const PAGE_SIZE: usize = 4096;
const EMPTY: u32 = u32::MAX;

pub(crate) struct PagedSparseSet {
    pages: Vec<Option<Box<[u32; PAGE_SIZE]>>>,
    dense_entities: Vec<Entity>,
    dense_values: BlobVec,
}
```

**Page size rationale:** 4096 entries × 4 bytes = 16 KB per page. Fits in L1 data cache (32–64 KB). Entity IDs are allocated sequentially from an `AtomicU32` bump allocator, so entities spawned together have adjacent IDs. One page covers 4096 consecutive entity IDs — one cache miss amortized across a burst of related lookups.

**Operations:**

| Operation | Steps | Cost |
|-----------|-------|------|
| `get(entity)` | `pages[idx/4096]?[idx%4096]` → dense_idx, verify `dense_entities[dense_idx] == entity` (generation check), return `dense_values.get_ptr(dense_idx)` | O(1), one page touch |
| `insert(entity, ptr)` | Allocate page if needed (filled with `EMPTY`), push to `dense_entities` + `dense_values`, store dense index in page slot | O(1) amortized |
| `remove(entity)` | Lookup dense_idx, swap-remove from `dense_entities` and `dense_values`, update swapped entity's page entry to new dense_idx, set removed entity's page entry to `EMPTY` | O(1) |
| `contains(entity)` | Same as get, check existence only | O(1) |
| `iter()` | Zip `dense_entities` with `dense_values` pointer arithmetic — contiguous BlobVec slices | O(n) contiguous |
| `len()` | `dense_entities.len()` | O(1) |

**Generation check:** The dense array stores full `Entity` values (index + generation). On lookup, compare the stored Entity against the queried Entity. Generation mismatch → stale entry → return `None`. Self-contained correctness — no external `is_alive` call needed. This is engine-internal storage, not an external index.

**Page lifecycle:** Pages are allocated on first insert to that 4096-entity range, initialized to `EMPTY`. Pages are never deallocated.

### Outer Container

Since `PagedSparseSet` is a single concrete type (BlobVec is type-erased), the `Box<dyn Any>` wrapper is eliminated:

```rust
pub(crate) struct SparseStorage {
    storages: HashMap<ComponentId, PagedSparseSet>,
}
```

Direct `HashMap<ComponentId, PagedSparseSet>` lookup — no downcast. The typed wrapper methods (`get::<T>`, `insert::<T>`) become thin unsafe casts from `*mut u8` to `&T` / `&mut T`, trusting that the `ComponentId` was registered with the correct `Layout`. Same trust model as archetype column access.

### Change Detection

Not added. Sparse components remain invisible to `Changed<T>`. Current behavior preserved. Adding sparse change detection is a separate feature — ship without it, add when there's a concrete use case.

---

## 2. Batched Despawn

### Problem

`EnumChangeSet::apply` processes `Mutation::Despawn` entries one at a time. Each despawn does a swap-remove across all BlobVec columns, moving one entity per removal. For N despawns in the same archetype, that's N swap-removes with N EntityLocation updates. Worse: a swap-remove can move an entity that will itself be despawned later in the batch, causing redundant work.

### Design

Group despawns by archetype, sort rows descending, sweep back-to-front.

**Phase 1 — Capture.** For each entity being despawned, call `read_all_components(entity)` to capture component data into the reverse changeset. All captures complete before any removals begin.

**Phase 2 — Group and sort.** Build `HashMap<ArchetypeId, Vec<usize>>` mapping each archetype to the rows that need removal. Sort each row list descending.

**Phase 3 — Back-to-front sweep.** For each archetype, iterate rows from highest to lowest:

1. Drop the component data at the target row (calls `drop_fn` if registered)
2. If row ≠ last: bitwise-copy last element into the vacated slot, patch the moved entity's `EntityLocation`
3. Truncate (decrement len)

**Why back-to-front is correct:** Removing row `r` swap-removes the last element into position `r`. Processing rows descending guarantees the "last element" is always either not in the despawn set or already processed. No element is moved and then despawned — because all higher rows are gone before lower rows are touched.

**Phase 4 — Dealloc entities.** Clear each despawned entity's `EntityLocation`, call `dealloc_entity` to bump generation and push to free list.

### BlobVec Additions

Two new `pub(crate)` methods:

- `drop_in_place(row)` — calls `drop_fn` on the element at `row` without moving anything. Slot becomes logically uninitialized.
- `copy_unchecked(src_row, dst_row)` — `ptr::copy_nonoverlapping` from src to dst. No drop on either side. Caller ensures dst is uninitialized and src will not be accessed again.

These decompose the existing `swap_remove` into constituent operations for the batch loop.

### API Surface

```rust
impl World {
    /// Despawn multiple entities. Returns count of actually-despawned entities
    /// (skips dead/unplaced). Order-independent.
    pub fn despawn_batch(&mut self, entities: &[Entity]) -> usize { ... }
}
```

`EnumChangeSet::apply` partitions despawn mutations out, captures their component data in bulk, records reverse `Spawn` mutations, then calls `despawn_batch`. Remaining mutations (Spawn, Insert, Remove) are applied in original order.

Single despawns still go through `World::despawn` — no regression for the common case.

### Sparse Cleanup on Despawn

When an entity is despawned, its sparse components are eagerly removed:

```rust
impl SparseStorage {
    pub(crate) fn remove_all(&mut self, entity: Entity) {
        for set in self.storages.values_mut() {
            set.remove(entity); // O(1) page lookup, no-op if absent
        }
    }
}
```

Cost: one page lookup per sparse component type per despawned entity. For 3 sparse types and 1000 despawns: 3000 array indexes at ~3ns each ≈ 9μs. Invisible next to archetype swap-removes.

`remove` on a `PagedSparseSet` where the entity isn't present costs one page lookup, finds `EMPTY` or a generation mismatch, returns false. No allocation, no data movement.

`despawn_batch` gets the same treatment — one flat pass over the entity list after archetype removals:

```rust
for &entity in actually_despawned {
    self.sparse.remove_all(entity);
}
```

Sparse cleanup is outside the archetype loop — sparse sets aren't archetype-specific.

---

## What Doesn't Change

- Public API: `World::despawn`, `register_sparse`, `get`/`get_mut` for sparse, `iter_sparse`, `insert_sparse` — all signatures unchanged
- Archetype storage, query system, change detection, transactions, reducers — untouched
- `BlobVec::swap_remove` and `swap_remove_no_drop` — still used by migration paths

## Test Plan

### Paged Sparse Set
- Page allocation on first insert, cross-page lookup (entity IDs in different pages)
- Generation rejection (despawned entity, same index reused)
- Swap-remove dense compaction (remove middle element, verify swapped element accessible)
- Iteration over dense array (contiguous, correct entities)
- Empty set operations (get/remove/iter on empty)

### Batched Despawn
- Multiple entities in same archetype — verify all removed, survivors intact
- Multiple archetypes in one batch — verify per-archetype grouping
- Mixed alive/dead inputs — dead entities skipped, alive ones despawned
- Single-entity degenerate case — same result as `World::despawn`
- Reverse changeset correctness — apply reverse re-spawns all entities with correct components
- Mixed sparse and dense-only entities in batch:
  ```rust
  // Entity with sparse component
  let a = world.spawn((pos,));
  world.insert_sparse(a, burning);
  // Entity without sparse component
  let b = world.spawn((pos,));
  world.despawn_batch(&[a, b]);
  // Both dead, sparse storage cleaned
  ```
