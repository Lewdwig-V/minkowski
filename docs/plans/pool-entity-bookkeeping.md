# Pool-Backed Entity Bookkeeping Design

## Problem

The pool budget (`WorldBuilder::memory_budget`) covers component storage only. That scope assumed component bytes dwarf the engine's heap-side bookkeeping. `examples/memory_footprint.rs` (branch `claude/memory-footprint-measure`) measured it with a counting global allocator, and the assumption does not hold for typical ECS entities:

| Heap-side cost | Measured |
|---|---|
| Per-entity bookkeeping (generations, `entity_locations`, archetype entity lists) | ~37 B/entity, whatever the component size |
| Sparse component bookkeeping | +12 B/entity per sparse component |
| Per archetype | ~1.5 KiB |
| `BTreeIndex` / `HashIndex` | ~123 / ~137 B per entry |
| Join scratch | 8 B per result row, reused |

At 1M entities, heap/pool is 4.45 with 8 B of components, 0.56 with 64 B and 0.14 with 256 B. The crossover is around 40 B of components per entity, which is where typical entities sit (position, velocity, health, a few tags). Entities with only zero-sized components consume no pool bytes at all, so today entity count is not bounded by the budget in any way.

**Decision (option B):** move everything that scales with *entity count* into the pool. What stays on the heap then scales with archetype count, user indexes, or query results. Options A (document only) and C (bound every allocation) are under Alternatives.

## Current State

Verified by grep on `main` (`841caf8`).

| Structure | Type today | Growth sites |
|---|---|---|
| `EntityAllocator::generations` | `Vec<u32>` (`entity.rs:181`) | `alloc` push (`entity.rs:250, 268`), `materialize_reserved` resize (`entity.rs:311`), `World::restore_allocator_state` replace (`world.rs:2193`) |
| `EntityAllocator::free_list.slots` | `Vec<u32>` inside `FreeList` (`entity.rs:9`) | `dealloc` push, `adopt` extend, `remove` retain |
| `World::entity_locations` | `Vec<Option<EntityLocation>>`, 24 B/slot (`world.rs:220`) | resize at `world.rs:403, 425, 483, 542`, `changeset.rs:977`, `import.rs:264` |
| `Archetype::entities` | `pub Vec<Entity>` (`storage/archetype.rs:30`) | push at `world.rs:440, 498, 557, 1435, 1630, 1718, 1752`, `changeset.rs:990, 1205, 1285, 1325`, `archetype.rs:195`; reserve `world.rs:532`; `extend_from_slice` `import.rs:256`; `swap_remove` / `truncate` (no growth) |
| `PagedSparseSet::dense_entities` | `Vec<Entity>` (`storage/sparse.rs:41`) | parallel to `dense_values` (already a pool `BlobVec`) |
| `PagedSparseSet::pages` | `Vec<Option<Box<[u32; 4096]>>>` | one 16 KiB box per touched page |

Existing failure-mode convention, from `try_spawn` (`world.rs:466-510`): fallible entry points (`try_spawn`, `try_insert`) **preflight** every column with `try_grow` before mutating any state. Infallible ones (`spawn`, `insert`, `remove`, changeset apply) push and abort through `handle_alloc_error` if the pool is exhausted.

Readers outside `World` mostly use slices: `entity_allocator_state() -> (&[u32], &[u32])` (LSM writer and recovery), query iteration over `arch.entities`, and planner scans.

## Proposed Design

### API Surface

No public signature changes. Behavior changes:
- **`World::new()` (256 MiB) and `memory_budget` pools now also pay for entity bookkeeping**, about 37 B per entity. `try_spawn` and `try_insert` can therefore return `PoolExhausted` earlier than before. ZST-only entities now consume budget, so entity count is bounded.
- **`WorldStats::pool_used`** now includes the bookkeeping.
- Changelog entry under Core, marked as a behavior change.

### Internal Architecture

**`PoolVec<T: Copy>`** (new, `storage/pool_vec.rs`, `pub(crate)`): a growable vector in the world's `SlabPool`.
- `Deref<Target = [T]>` / `DerefMut`, so indexing, iteration and `.get()` call sites don't change.
- Fallible growth: `try_reserve(n)`, `try_push`, `try_resize(len, value)`, `try_extend_from_slice`. All grow through `SlabPool::grow`, so large lists widen in place.
- Infallible wrappers (`push`, `resize`, `reserve`, `extend_from_slice`) call `handle_alloc_error` on `PoolExhausted`, mirroring `Vec` and today's `BlobVec::push`.
- `swap_remove`, `truncate`, `retain`, `pop`, `clear` never allocate.
- `T: Copy` only (no drop glue to manage). `assert!(size_of::<T>() > 0)`. Doubling growth from 4 elements, like `BlobVec`.
- `Drop` deallocates. `Send` / `Sync` follow `T`. Holds `Arc<SlabPool>`.

Structures moved to `PoolVec`:

| Structure | Element | Notes |
|---|---|---|
| `generations` | `u32` | |
| `FreeList::slots` | `u32` | `positions` (lazy `HashMap`, built only on the `adopt`/`remove` path) and `snapshot` (lazy `as_slice` copy for persistence) stay on the heap. Both are transient and empty on the spawn/despawn path; see the matrix exemptions. |
| `entity_locations` | `Option<EntityLocation>` | |
| `Archetype::entities` | `Entity` | Field type changes from `Vec<Entity>` to `PoolVec<Entity>`; `pub` stays `pub` within the crate. |
| `PagedSparseSet::dense_entities` | `Entity` | |

**Two rules make the failure modes well defined:**

1. **Bookkeeping follows its column's failure mode.** On a path that preflights columns (`try_spawn`, `try_insert`), the same preflight also reserves every bookkeeping slot it will touch: a `generations` slot if the free list is empty, `entity_locations` up to the new index, and one target-archetype `entities` slot. Only then does it mutate anything. On an infallible path, bookkeeping growth aborts exactly as the column push already does. No path gains a new failure mode it didn't already have for columns.
2. **Despawn never allocates.** Free-list capacity is kept at least `generations.len()`: whenever `generations` grows, the free list reserves to match, inside the same preflight. `dealloc`'s push therefore never needs memory, so `despawn` and `despawn_batch` stay infallible, as they are today. This costs ≤ 4 B per entity, reserved up front.

### Data Flow

`try_spawn` → preflight: columns `try_grow`, then `entities.try_reserve_slot()` (`generations` + matching free-list capacity, only if the free list is empty), then `entity_locations.try_resize`, then `archetype.entities.try_reserve(1)` → `Err(PoolExhausted)` with no state changed. Otherwise all pushes are infallible, because capacity is already held.

### Invariant Matrix

Invariant **B1**: every structure in the table above is allocated from the world's pool. Invariant **B2**: no path reports success or failure inconsistently. A fallible path that returns `Err` has changed no state, and an infallible path aborts only where its column push would already abort. Invariant **B3**: `despawn` never allocates.

| Consumer \ structure | generations | free list | entity_locations | archetype entities | sparse dense_entities |
|---|---|---|---|---|---|
| `try_spawn` | fallible preflight (new test) | reserve with generations (new test) | fallible preflight (new test) | fallible preflight (new test) | n/a |
| `spawn` / `spawn_batch` | abort (as columns) | reserve with generations | abort | abort; `spawn_batch` reserves `lower` up front | n/a |
| `try_insert` (migration) | n/a | n/a | n/a | fallible preflight on target archetype (new test) | fallible via sparse insert path (new test) |
| `insert` / `remove` (migration) | n/a | n/a | n/a | abort (as columns) | abort |
| `despawn` / `despawn_batch` | no alloc | **no alloc (B3)**: test exhausts the pool, then despawns | no alloc | `swap_remove`, no alloc | `swap_remove`, no alloc |
| changeset apply (spawn/insert/remove/despawn) | abort (as columns) | reserve with generations | abort | abort | abort |
| `reserve` (`&self`, atomic) / `materialize_reserved` | resize: abort (as columns on apply) | reserve to match | resize: abort | n/a | n/a |
| `adopt` (replication, explicit entity) | resize: abort | `extend`: reserved with generations; `positions` HashMap exempt (lazy, adopt-only) | resize: abort | n/a | n/a |
| `import_target` / `import_page` (LSM recovery) | n/a | n/a | resize: abort | `extend_from_slice`: abort | n/a |
| `restore_allocator_state` | copy into pool: abort | copy into pool: abort | n/a | n/a | n/a |
| readers (`entity_allocator_state`, queries, planner, LSM writer) | slice via `Deref`, unchanged | `as_slice`; `snapshot` exempt (lazy copy, persistence only) | slice, unchanged | slice, unchanged | slice, unchanged |

Every "new test" cell gets a test that exhausts the pool at exactly that preflight step and asserts `Err` and an unchanged world (entity count, `entity_locations`, archetype lengths, `pool_used`).

### Convention Sheet

- `generations.len()` is the number of indices ever handed out (allocated or materialized), as today.
- `free_list.slots.capacity() >= generations.len()` after every operation that can grow `generations` (B3). It is pinned by a debug assertion in `alloc` / `materialize_reserved` / `adopt` and by `despawn_never_allocates`.
- `entity_locations.len()` may lag `generations.len()`, as today (it is resized on demand to `index + 1`). No new convention.

## Alternatives Considered

1. **A: document only.** Publish the sizing rule (~37 B/entity + ~1.5 KiB/archetype + index entries + join scratch) and keep the component-only scope. Cheap, but guarantees nothing; ZST-only entities stay unbounded.
2. **C: bound every allocation.** Route all engine allocations, including archetype metadata, query caches, indexes and planner scratch, through bounded pools or a budgeted global allocator. This is the only option that bounds process memory, but it touches most of the crate and user-owned index types, and it needs a fallible API on every query path.
3. **Global-allocator accounting** (count heap bytes against the budget without moving them). Rejected: the global allocator is process-wide, not per-world, and can't return a typed `PoolExhausted` to the right caller.

## Phase 2 (out of scope here)

- **`PagedSparseSet::pages`** (done): 16 KiB pages would land in the 64 KiB slab class (4× waste), so pages shrank to 1024 entries (4 KiB, exactly the 4 KiB class) and moved into the pool. Only the page directory stays on the heap.
- **Shrink `EntityLocation`** (done) from 16 B (24 B as `Option`) to `{ archetype_id: u32, row: u32 }` with a niche (8 B). This cuts the largest bookkeeping term by about two thirds, in the pool or on the heap. It's independent of this design, but it multiplies its effect.
- **Fix the per-transaction changeset pools** (`Tx::new` and commit's `mem::take` each build a 256 MiB default pool; ~453 µs per empty transaction). Independent bug; separate PR.

## Semantic Review

1. **Wrong World?** Each `PoolVec` holds the `Arc<SlabPool>` it was created from and deallocates into it. The structures are created in `World::new_with_pool` / `Archetype::new` from the world's own pool, the same way columns are.
2. **Drop observes inconsistent state?** `PoolVec<T: Copy>` drop only deallocates its buffer. There is no element drop glue, and the Arc keeps the pool alive.
3. **Two threads through `&self`?** `EntityAllocator::reserve(&self)` stays an atomic counter and touches no `PoolVec`. Every `PoolVec` mutation is behind `&mut World` / `&mut Archetype`. The pool itself is already `Sync`.
4. **Dedup/merge preserves the strongest invariant?** `FreeList::remove`'s `retain` compaction only shrinks `len`; capacity (B3) is unchanged.
5. **Abandoned halfway?** A fallible preflight only *reserves* capacity. If a later preflight step fails, the earlier reservations stay as unused capacity, and no length or content has changed (B2). Infallible paths abort, as today.
6. **Type bound violable by a legal generic?** `PoolVec<T: Copy>` with a runtime non-ZST assert, instantiated only with crate-internal types.
7. **Operations outside the Access bitset?** None: bookkeeping is engine-internal and not component data.

## Risks

- **Default budget headroom.** A `World::new()` that previously held N entities near 256 MiB of components now also holds ~37·N bytes of bookkeeping. LSM recovery builds its world with `World::new()` (`minkowski-lsm/src/recovery.rs:686`), so recovering a very large world can now exhaust the default pool sooner. Mitigation: changelog entry, plus a follow-up to let recovery take a `WorldBuilder`.
- **Performance.** Indexing and iteration go through `Deref` to a slice, same as `Vec`. Growth goes through `SlabPool::grow`, which is in place for large runs, versus `Vec` realloc. Benchmark gate: `simple_insert/*`, `add_remove/*`, `simple_iter/*` and `reducer/*` must stay within noise.

## Implementation Plan

1. `storage/pool_vec.rs`: `PoolVec<T: Copy>` plus unit tests (growth, in-place large growth, fallible paths leave state unchanged, drop returns `used()` to baseline) and a loom-free Miri pass.
2. `entity.rs`: `generations` and `FreeList::slots` → `PoolVec`. `EntityAllocator::new(pool)`. Fallible `try_reserve_slot` maintaining B3.
3. `world.rs` / `archetype.rs`: `entity_locations` and `Archetype::entities` → `PoolVec`. Extend the `try_spawn` / `try_insert` preflights. Infallible paths call the aborting wrappers.
4. `changeset.rs`, `import.rs`, `storage/sparse.rs` (`dense_entities`): switch to the `PoolVec` APIs.
5. `restore_allocator_state` / `entity_allocator_state`: copy in, slice out.
6. Tests for every "new test" cell in the matrix, plus `despawn_never_allocates`. Rerun `memory_footprint` to show heap per entity falling from ~37 B to about 0 and pool usage rising by the same amount.
7. Docs: `memory-management.md` "What the budget covers" table, and the changelog.
8. Gates: fmt and clippy (1.97 and 1.99), workspace tests, loom, TSan, Miri on the touched modules, `fuzz_world_ops` and `fuzz_pool_columns` (60 s each), benchmarks against `main`.
