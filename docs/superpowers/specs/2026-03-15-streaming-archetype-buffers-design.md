# Streaming Archetype Buffers for QueryWriter

**Date**: 2026-03-15
**Target**: v1.3.7
**Status**: Design approved, pending implementation

## Problem

QueryWriter's apply phase pays per-entity bookkeeping costs that dominate
execution time. Profiling shows 1.58% real work (memcpy) vs 37% per-entity
lookups (`is_alive`, `entity_locations`, `FixedBitSet::contains`,
`column_index`, `ComponentRegistry::info`). PR #125's batch-apply optimization
reduced this by 14% (84µs → 72µs) by amortizing `column_index` and `info`
across consecutive same-archetype overwrites. The remaining gap to `query_mut`
(1.6µs) is 45x.

The root cause: the apply phase re-discovers information that the recording
phase already knew. QueryWriter's `for_each` iterates archetypes — the
archetype, row, column index, and drop function are all known at write time
but discarded, forcing apply to look them up again per entity.

## Design

### Approach: Fast Lane on EnumChangeSet

Add an optional fast lane to `EnumChangeSet` — a `Vec<ArchetypeBatch>` of
pre-resolved overwrite batches. QueryWriter populates the fast lane during
`for_each` instead of pushing `Mutation::Insert` entries. The apply phase
drains the fast lane with zero per-entity lookups, then processes the regular
mutation log for spawns, despawns, removes, and migrations.

This is internal to QueryWriter. The public `EnumChangeSet` API, WAL
serialization, and external consumers are unaffected.

**Alternatives considered and rejected:**

- **Shadow columns (copy-on-write)**: Doubles memory for every mutable
  component. Wasteful for sparse updates (10 entities in a 100K archetype).
- **Direct write + undo log**: Destroys read stability during mutation phase.
  Incompatible with atomic commit semantics.
- **HashMap<ArchetypeId, Vec<Mutation>>**: HashMap overhead on every `set()`
  call. QueryWriter already iterates archetypes sequentially — streaming
  is the natural fit.
- **QueryWriter-owned sidecar buffer**: Forces `WritableRef` to carry two
  pointers (changeset + sidecar). Complicates borrow splitting.
- **Separate ArchetypeChangeSet type**: Too invasive — changes WriterQuery
  trait, reducer registration, and adapter closure signatures.

### Data Structures

All types are `pub(crate)` in `changeset.rs`:

```rust
/// A single column's worth of buffered overwrites within one archetype.
/// Column index and drop function are resolved once when the batch is
/// created, not per entity.
struct ColumnBatch {
    comp_id: ComponentId,
    col_idx: usize,
    drop_fn: Option<unsafe fn(*mut u8)>,
    layout: Layout,
    entries: Vec<(usize, Entity, usize)>,  // (row, entity, arena_offset)
}

/// All buffered overwrites for a single archetype, grouped by component.
struct ArchetypeBatch {
    arch_idx: usize,
    columns: Vec<ColumnBatch>,  // one per mutable component
}
```

New field on `EnumChangeSet`:

```rust
pub struct EnumChangeSet {
    pub(crate) mutations: Vec<Mutation>,
    pub(crate) arena: Arena,
    drop_entries: Vec<DropEntry>,
    // Streaming archetype batches from QueryWriter.
    // Applied before the regular mutation log.
    pub(crate) archetype_batches: Vec<ArchetypeBatch>,
}
```

`Vec<ArchetypeBatch>` rather than `Option<Vec<...>>` — an empty Vec is
zero-cost (no allocation until first push), and avoids a branch on every
access.

### Recording Path

QueryWriter's `for_each` archetype loop gains batch management at the
archetype boundary. When entering a new archetype, `open_archetype_batch`
resolves each mutable component's column index and drop function once,
then pushes an `ArchetypeBatch` with pre-resolved `ColumnBatch` slots.

```
for (arch_idx, arch) in archetypes.iter().enumerate() {
    if !matches_query(arch) { continue; }
    open_archetype_batch(cs, arch_idx, arch, components, mutable_ids);
    for row in 0..arch.len() {
        let item = fetch_writer(&fetch, row, entity, cs_ptr);
        f(item);
    }
}
```

`open_archetype_batch` creates `ColumnBatch` entries in `mutable_ids.ones()`
iteration order. Since `FixedBitSet::ones()` yields bits in ascending order,
this ordering is deterministic and stable.

**Mutable IDs computed once**: `mutable_ids` is computed at the top of
`for_each` and passed to both pre-allocation and `open_archetype_batch`,
avoiding redundant bitset construction per archetype.

### WritableRef Changes

`WritableRef` gains two fields threaded from the iteration loop:

```rust
pub struct WritableRef<'a, T: Component> {
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: *mut EnumChangeSet,
    row: usize,           // entity's row in archetype (free from iteration)
    column_slot: usize,   // index into ArchetypeBatch.columns (resolved per-archetype)
    _marker: PhantomData<&'a EnumChangeSet>,
}
```

- **`row`**: Already available at the `fetch_writer` call site. Threading it
  avoids the 5.91% `entity_locations` lookup tax per entity.
- **`column_slot`**: The component's position in `mutable_ids.ones()` order,
  resolved in `init_writer_fetch`. Direct index into `ArchetypeBatch.columns`
  — no linear scan, no HashMap.

**`init_writer_fetch`** resolves `column_slot` for `&mut T`:

```rust
let column_slot = mutable_ids.ones()
    .position(|id| id == comp_id.0)
    .unwrap();
```

This runs once per archetype, not per entity.

**`set()` becomes a triple-pointer-bump:**

```rust
pub fn set(&mut self, value: T) {
    let cs = unsafe { &mut *self.changeset };
    let batch = cs.archetype_batches.last_mut().unwrap();
    let col_batch = &mut batch.columns[self.column_slot];
    debug_assert_eq!(col_batch.comp_id, self.comp_id);
    debug_assert_eq!(col_batch.layout, Layout::new::<T>());

    let value = std::mem::ManuallyDrop::new(value);
    let offset = cs.arena.alloc(
        &*value as *const T as *const u8,
        Layout::new::<T>(),
    );
    col_batch.entries.push((self.row, self.entity, offset));

    if std::mem::needs_drop::<T>() {
        cs.drop_entries.push(DropEntry {
            offset,
            drop_fn: crate::component::drop_ptr::<T>,
            mutation_idx: usize::MAX,  // sentinel: fast-lane entry
        });
    }
}
```

**`modify()` inherits the fast lane**: `WritableRef::modify()` calls
`self.set(val)` internally, so it routes through the fast lane automatically.
No separate handling needed.
```

Per-entity cost: arena alloc + push `(row, entity, offset)` + conditional drop entry.
No generation check, no location lookup, no column resolution, no bitset
check.

### Apply Path

The fast lane is drained as a preamble to `apply_mutations`, before the
regular mutation log:

```rust
fn apply_mutations(&self, world: &mut World, tick: Tick)
    -> Result<(), (usize, ApplyError)>
{
    // ── Fast lane: zero per-entity lookups ──
    for batch in &self.archetype_batches {
        let arch = &mut world.archetypes.archetypes[batch.arch_idx];
        for col_batch in &batch.columns {
            if col_batch.entries.is_empty() {
                continue;
            }
            let col = &mut arch.columns[col_batch.col_idx];
            col.mark_changed(tick);
            let size = col_batch.layout.size();
            for &(row, _entity, offset) in &col_batch.entries {
                unsafe {
                    let src = self.arena.get(offset);
                    let dst = col.get_ptr(row);
                    if let Some(drop_fn) = col_batch.drop_fn {
                        drop_fn(dst);
                    }
                    std::ptr::copy_nonoverlapping(src, dst, size);
                }
            }
        }
    }

    // ── Regular path: spawns, despawns, removes, migrations ──
    let mut batch: Option<InsertBatch> = None;
    for (mutation_idx, mutation) in self.mutations.iter().enumerate() {
        // ... existing code unchanged ...
    }
    Ok(())
}
```

Per-entity apply cost: pointer arithmetic + optional drop + memcpy.

**Eliminated per entity vs current path:**

| Lookup | Cost (from profiling) | Status |
|---|---|---|
| `is_alive(entity)` | 2.41% | eliminated |
| `entity_locations[index]` | 5.91% | eliminated |
| `FixedBitSet::contains` | 9.32% | eliminated (PR #125 for batched, now all) |
| `column_index(comp_id)` | 3.17% | eliminated |
| `ComponentRegistry::info` | 4.39% | eliminated (PR #125 for batched, now all) |
| `copy_nonoverlapping` | 1.58% | retained (the real work) |

**Ordering**: Fast-lane batches (all overwrites) are applied first, then the
regular mutation log. Overwrites commute with each other. Any subsequent
mutations in the regular log see the overwritten values.

### Drop Safety

**Abort (changeset dropped without apply):** The `Drop` impl iterates all
`drop_entries` unconditionally. Fast-lane entries have `mutation_idx:
usize::MAX`, which is always included. Destructors run correctly.

**Successful apply:** `drop_entries.clear()` disarms everything. Fast-lane
values' ownership transferred to world columns. No double-free.

**Partial failure in regular mutation log:** The fast lane has already been
applied. The retain filter must exclude fast-lane entries to prevent
double-free:

```rust
Err((failed_idx, err)) => {
    self.drop_entries
        .retain(|entry| entry.mutation_idx >= failed_idx
                     && entry.mutation_idx != usize::MAX);
    Err(err)
}
```

`usize::MAX` serves double duty: on abort, `MAX >= 0` retains the entry for
cleanup. On partial failure, `MAX != MAX` excludes it because ownership
already transferred. This is a clean invariant — `usize::MAX` means
"fast-lane, applied before the regular log."

**Note:** In practice, partial failure with fast-lane entries is unlikely.
Each reducer gets its own changeset, and QueryWriter only produces overwrites.
But `EnumChangeSet` is a public type — defense-in-depth protects against
future composition patterns.

### `is_empty()` and `len()`

Updated to account for fast-lane entries:

```rust
pub fn is_empty(&self) -> bool {
    self.mutations.is_empty() && self.archetype_batches.is_empty()
}

pub fn len(&self) -> usize {
    let fast_lane_count: usize = self.archetype_batches.iter()
        .flat_map(|b| &b.columns)
        .map(|c| c.entries.len())
        .sum();
    self.mutations.len() + fast_lane_count
}
```

### `is_alive` Safety Argument

The fast lane skips `is_alive` checks. This is sound because:

1. QueryWriter's `for_each` iterates entities from archetype storage — these
   are live, placed entities by construction.
2. The changeset is not applied until `for_each` completes — no entity can
   be despawned between recording and apply.
3. The fast lane is `pub(crate)` — external code cannot inject stale handles
   into `archetype_batches`.

This does NOT apply to the regular mutation log, which remains a public API
where stale handles are possible. The `is_alive` check stays on the regular
path.

### Pre-allocation

The existing pre-allocation in `for_each` (lines 1041-1047) reserves capacity
on `mutations` and `arena`. With the fast lane:

- `mutations.reserve()` is removed — overwrites no longer use the mutations
  vec.
- `arena.reserve()` stays — values are still arena-allocated.
- `archetype_batches` does not need pre-allocation — it grows by one entry
  per matching archetype, which is typically small (1-10).

### WAL Serialization

The fast lane is invisible to WAL. The `Durable` strategy serializes
`EnumChangeSet` before `apply()`. Two options:

- **(A)** Double-write: QueryWriter writes to both fast lane and mutations.
- **(B)** On-demand drain: `fn drain_fast_lane_to_mutations(&mut self)` that
  the WAL path calls before serialization.

**Decision: (B)**. The hot path (non-durable) never calls it. The WAL path
calls it once before serialization, converting `ArchetypeBatch` entries into
synthetic `Mutation::Insert` entries. This keeps the fast lane zero-overhead
for the common case.

**`drain_fast_lane_to_mutations` implementation:**

```rust
pub(crate) fn drain_fast_lane_to_mutations(&mut self) {
    for batch in self.archetype_batches.drain(..) {
        for col_batch in batch.columns {
            for (_row, entity, offset) in col_batch.entries {
                let mutation_idx = self.mutations.len();
                self.mutations.push(Mutation::Insert {
                    entity,
                    component_id: col_batch.comp_id,
                    offset,
                    layout: col_batch.layout,
                });
                // Rebase any fast-lane drop entries for this offset
                // from usize::MAX to the real mutation_idx, so the
                // partial-failure retain logic works correctly.
                for entry in &mut self.drop_entries {
                    if entry.offset == offset
                        && entry.mutation_idx == usize::MAX
                    {
                        entry.mutation_idx = mutation_idx;
                        break;
                    }
                }
            }
        }
    }
}
```

The `Entity` stored in `ColumnBatch.entries` is what makes this possible —
without it, the drain function would need `&World` access to recover entity
handles from row indices. The 8-byte per-entry cost of storing `Entity` is
justified by keeping the drain function self-contained.

**Entries store arena offsets, not pointers**: `ColumnBatch.entries` stores
`(row, Entity, usize)` where the third element is an arena offset, not a
`*const u8`. This is critical because the arena may reallocate during
recording (when `set()` triggers `arena.alloc()` → `grow()`), invalidating
all previously issued pointers. Offsets survive reallocation. Pointers are
resolved at apply time via `self.arena.get(offset)`, when no further
allocations occur. This matches the existing `Mutation::Insert` design.

**Drop entry rebasing**: After drain, fast-lane `DropEntry` records have real
`mutation_idx` values, so the partial-failure retain filter
(`entry.mutation_idx >= failed_idx`) works correctly without the
`usize::MAX` exclusion. The WAL path and non-WAL path use different retain
logic — this is acceptable because `drain_fast_lane_to_mutations` converts
fast-lane entries into regular entries, unifying both paths.

### Read-only Components

For a query like `(&Position, &mut Velocity)`, `&Position` uses the existing
`fetch_writer` that returns a plain reference — no `WritableRef` wrapper, no
fast-lane interaction. Only `&mut T` terms carry `row` and `column_slot`.

For tuple queries `(&mut Position, &mut Velocity)`, the `WriterFetch` is a
tuple `(MutFetch<Position>, MutFetch<Velocity>)`. Each has its own
`column_slot` — resolved independently in `init_writer_fetch`. The tuple's
`fetch_writer` impl threads the same `row` to both.

## Testing Strategy

### Unit Tests (changeset.rs)

| Test | Verifies |
|---|---|
| `fast_lane_single_component` | One archetype, one mutable component, N entities |
| `fast_lane_multi_component` | `(&mut Pos, &mut Vel)` — two ColumnBatch slots |
| `fast_lane_multi_archetype` | 3 archetypes, each gets its own batch |
| `fast_lane_drop_on_abort` | `needs_drop` values cleaned up on changeset drop |
| `fast_lane_partial_failure_no_double_free` | Regular log fails after fast lane applied |
| `fast_lane_empty_batch` | Conditional update — no `set()` calls on some archetypes |
| `fast_lane_is_empty_and_len` | Accounting includes fast-lane entries |

### Integration Tests (reducer.rs)

| Test | Verifies |
|---|---|
| `query_writer_fast_lane_roundtrip` | End-to-end: register, iterate, modify, verify world state |
| `query_writer_fast_lane_change_detection` | `Changed<T>` filter matches after fast-lane apply |
| `query_writer_conditional_update` | Half entities updated, half untouched |
| `query_writer_read_only_components` | `(&Pos, &mut Vel)` — Pos unmodified |
| `query_writer_column_slot_debug_assert` | `debug_assert_eq!` on comp_id fires on misalignment |

### Miri Tests (ci/miri-subset.txt)

```
changeset::tests::fast_lane_single_component
changeset::tests::fast_lane_drop_on_abort
changeset::tests::fast_lane_partial_failure_no_double_free
reducer::tests::query_writer_fast_lane_roundtrip
```

### Benchmarks

| Benchmark | Setup |
|---|---|
| `reducer/query_writer_fast_lane_10k` | 10K entities, `(&mut Pos)`, all updated |
| `reducer/query_writer_fast_lane_multi_comp_10k` | 10K entities, `(&mut Pos, &mut Vel)` |
| `reducer/query_writer_fast_lane_sparse_update_10k` | 10K entities, 10% call `set()` |
| `reducer/query_writer_fast_lane_multi_arch` | 3 archetypes × 3K entities |

**Expected improvement**: 72µs → 55-60µs (conservative), 45-50µs
(aggressive). The remaining gap to `query_mut` (1.6µs) is structural: arena
allocation (15%) and `Vec::push` (8%) in the recording phase.

## Regression Risk: Archetype Fragmentation

The one scenario where the fast lane could regress is **highly fragmented
archetypes** — many archetypes with few entities each (e.g., 5,000 archetypes
× 2 entities vs 5 archetypes × 2,000 entities).

**Per-archetype overhead:**
- `open_archetype_batch`: one `ColumnBatch` allocation per archetype per
  mutable component
- `init_writer_fetch`: one `column_slot` resolution per archetype
- Apply: one `mark_changed` call + column pointer resolution per archetype

For 5,000 archetypes × 2 mutable components = 10,000 `ColumnBatch`
allocations + 10,000 `Vec::new()` calls, most holding only 2 entries each.
The current path has no per-archetype setup cost — it pays per entity
uniformly.

**Mitigating factors:**
- QueryWriter already pays per-archetype costs (`init_writer_fetch`,
  archetype matching) in the current code. The fast lane adds one
  `ColumnBatch` struct per component on top of existing overhead.
- `Vec::new()` does not allocate until first push. A `ColumnBatch` with
  zero entries (conditional update, archetype doesn't trigger `set()`) is
  a no-op.
- For 2-entity archetypes, the entries `Vec` holds 2 elements — a single
  16-byte heap allocation or inline on some allocators.

**Mitigation if needed:** Skip the fast lane for archetypes below a size
threshold (e.g., `arch.len() < 4`) and fall back to `insert_raw` for those
entities. This adds one branch per archetype, not per entity.

**Benchmark coverage:** `reducer/query_writer_fast_lane_multi_arch` (3
archetypes × 3K entities) tests the multi-archetype path. If fragmentation
is a concern, add a stress benchmark: 1,000 archetypes × 10 entities each.

## Files Modified

| File | Changes |
|---|---|
| `crates/minkowski/src/changeset.rs` | `ColumnBatch`, `ArchetypeBatch` structs; `archetype_batches` field; fast-lane drain in `apply_mutations`; `is_empty`/`len` updates; partial-failure retain fix; `drain_fast_lane_to_mutations` for WAL |
| `crates/minkowski/src/reducer.rs` | `WritableRef` gains `row`, `column_slot`; `set()` routes to fast lane; `init_writer_fetch` resolves `column_slot`; `for_each` calls `open_archetype_batch` at archetype boundaries; pre-allocation updated |
| `crates/minkowski-bench/benches/planner.rs` | 4 new benchmarks |
| `ci/miri-subset.txt` | 4 new entries |
