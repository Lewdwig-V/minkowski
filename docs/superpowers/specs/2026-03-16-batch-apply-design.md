# Batch Apply for EnumChangeSet

**Version**: v1.3.4
**Date**: 2026-03-16
**Status**: Approved

## Problem

`EnumChangeSet::apply_mutations` processes mutations one at a time. For
each `Mutation::Insert` (in-place overwrite), it performs per-entity:

1. `is_alive(entity)` — generation check (2.41%)
2. `entity_locations[index].unwrap()` — location lookup (5.91%)
3. `FixedBitSet::contains(comp_id)` — archetype has component? (9.32%)
4. `column_index(comp_id).unwrap()` — which column? (3.17%)
5. `ComponentRegistry::info(comp_id)` — drop_fn lookup (4.39%)
6. `BlobVec::get_ptr_mut(row, tick)` — ptr_at computation (12.82%)
7. `copy_nonoverlapping` — the actual data copy (1.58%)

Steps 3-5 are invariant across entities in the same archetype — they
resolve to the same answer 10,000 times. The memcpy is 1.58% of total
time; the per-entity bookkeeping is 37%.

Benchmark: `reducer/query_writer_10k` = 84 us (53x vs `query_mut_10k`).

## Solution: Group-and-Interleave

Walk the mutation log sequentially. Track the current batch state. When
consecutive `Insert` mutations share the same `(archetype_id, component_id)`
and the entity's archetype already contains the component (overwrite, not
migration), accumulate them into a batch. When the pattern breaks (different
archetype, different component, non-Insert mutation, or migration), flush
the batch and process the breaking mutation sequentially.

### The Algorithm

```
batch = None
for mutation in mutations:
    if mutation is Insert:
        loc = entity_locations[entity]
        if archetype has component (overwrite path):
            key = (loc.archetype_id, component_id)
            if batch.key == key:
                batch.add(row, src_ptr)
            else:
                flush(batch)
                batch = new Batch(key, resolve column + drop_fn + layout)
                batch.add(row, src_ptr)
        else:
            flush(batch)
            process_migration_sequential(mutation)
    else:
        flush(batch)
        process_sequential(mutation)
flush(batch)
```

### What `flush` does

```rust
fn flush_insert_batch(
    archetype: &mut Archetype,
    batch: &InsertBatch,
    tick: Tick,
) {
    // Resolved ONCE per batch:
    let col = &mut archetype.columns[batch.col_idx];
    col.mark_changed(tick);

    // Per entity — tight loop:
    for &(row, src) in &batch.entries {
        let dst = unsafe { col.base_ptr().add(row * batch.layout.size()) };
        if let Some(drop_fn) = batch.drop_fn {
            unsafe { drop_fn(dst); }
        }
        unsafe { ptr::copy_nonoverlapping(src, dst, batch.layout.size()); }
    }
}
```

Per-batch cost: 1 `column_index` + 1 `info` lookup + 1 `mark_changed`.
Per-entity cost: 1 `entity_locations` lookup + 1 pointer arithmetic +
1 memcpy.

Eliminated per-entity: `is_alive`, `FixedBitSet::contains`,
`column_index`, `ComponentRegistry::info` — ~17% of total time.

### Batch State

```rust
struct InsertBatch {
    arch_idx: usize,
    comp_id: ComponentId,
    col_idx: usize,
    drop_fn: Option<DropFn>,
    layout: Layout,
    entries: Vec<(usize, *const u8)>,  // (row, arena src ptr)
}
```

`entries` is a local in `apply_mutations`, cleared per flush, reused
across batches (amortized allocation).

### Ordering Guarantee

The batch never reorders mutations. It only groups *consecutive*
same-key inserts. When the pattern breaks, the batch is flushed before
the next mutation is processed. This preserves the sequential semantics
of the mutation log.

Examples:
- `[Insert(A,Pos), Insert(B,Pos), Despawn(A)]` ->
  batch [A,B] flush -> Despawn(A). A has updated Pos when despawned.
- `[Insert(A,Pos), Despawn(A), Insert(B,Pos)]` ->
  batch [A] flush -> Despawn(A) -> batch [B] flush. Correct.

### What stays sequential

- `Mutation::Spawn` — multi-component archetype resolution
- `Mutation::Despawn` — entity lifecycle mutation
- `Mutation::Remove` — archetype migration
- `Mutation::SparseInsert` / `Mutation::SparseRemove` — sparse storage
- `Mutation::Insert` where component NOT in archetype (migration)
- `Mutation::Insert` where entity is dead

### is_alive check

Checked once at batch-add time (when resolving `entity_locations`).
If the entity is dead (location is `None`), flush the current batch
and return `Err(DeadEntity)`. Dead entities in a QueryWriter changeset
indicate a bug (the entity was alive during `for_each`), so this is
defensive.

## Performance Predictions

| Benchmark | Before | After |
|---|---|---|
| `reducer/query_writer_10k` | 84 us | ~45-50 us |
| `changeset/apply_10k_overwrites` | 104 us | ~55-60 us |

The remaining cost is entity_locations lookup + ptr arithmetic + memcpy
per entity, plus the user's for_each closure (untouched).

## Test Plan

5 tests in `changeset.rs::tests`:

1. `batch_apply_overwrites_same_archetype` — 10K overwrites, verify values
2. `batch_apply_mixed_insert_despawn` — interleaved, verify ordering
3. `batch_apply_cross_archetype_flushes` — different archetypes, verify breaks
4. `batch_apply_migration_falls_through` — component not in archetype
5. `batch_apply_dead_entity_returns_error` — dead entity mid-batch

## Files Modified

| File | Change |
|---|---|
| `crates/minkowski/src/changeset.rs` | `InsertBatch` struct, `flush_insert_batch` fn, refactored `apply_mutations` |

## Non-Goals

- Sorting batch entries by row for linear stride — second-order optimization,
  measure first after batching lands.
- Batching `Spawn` mutations — different shape (multi-component), lower
  frequency.
- Changing the `Mutation` enum or `EnumChangeSet` public API.
