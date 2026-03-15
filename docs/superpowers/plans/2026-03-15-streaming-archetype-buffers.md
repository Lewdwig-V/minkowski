# Streaming Archetype Buffers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-entity bookkeeping in QueryWriter's apply phase by routing overwrites through pre-resolved per-archetype, per-component batch buffers.

**Architecture:** Add a fast lane (`Vec<ArchetypeBatch>`) to `EnumChangeSet`. QueryWriter populates it during `for_each` with pre-resolved column indices. The apply phase drains batches with zero per-entity lookups. The regular mutation log (spawns, despawns, removes) is unchanged.

**Tech Stack:** Rust (edition 2024), `fixedbitset`, `criterion` benchmarks, Miri for unsafe validation.

**Spec:** `docs/superpowers/specs/2026-03-15-streaming-archetype-buffers-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/src/changeset.rs` | Modify | `ColumnBatch`, `ArchetypeBatch` structs; `archetype_batches` field on `EnumChangeSet`; `open_archetype_batch`; fast-lane drain in `apply_mutations`; `drain_fast_lane_to_mutations`; `is_empty`/`len` updates; partial-failure retain fix |
| `crates/minkowski/src/reducer.rs` | Modify | `WritableRef` gains `row`, `column_slot`; `set()` routes to fast lane; `init_writer_fetch` resolves `column_slot`; `for_each` calls `open_archetype_batch`; pre-allocation updated; tuple macro updated |
| `crates/minkowski-persist/src/durable.rs` | Modify | Call `drain_fast_lane_to_mutations()` before WAL append |
| `crates/minkowski-bench/benches/reducer.rs` | Modify | 4 new benchmarks |
| `ci/miri-subset.txt` | Modify | 4 new entries |

---

## Chunk 1: Data Structures & Apply Path (changeset.rs)

### Task 1: Add ColumnBatch, ArchetypeBatch, and archetype_batches field

**Files:**
- Modify: `crates/minkowski/src/changeset.rs:209-243` (EnumChangeSet struct + constructors)

- [ ] **Step 1: Write the test `fast_lane_is_empty_and_len`**

In `crates/minkowski/src/changeset.rs`, in the `#[cfg(test)] mod tests` block at the bottom, add:

```rust
#[test]
fn fast_lane_is_empty_and_len() {
    use std::alloc::Layout;

    let mut cs = EnumChangeSet::new();
    assert!(cs.is_empty());
    assert_eq!(cs.len(), 0);

    // Push a fast-lane batch with 2 entries in one column
    cs.archetype_batches.push(ArchetypeBatch {
        arch_idx: 0,
        columns: vec![ColumnBatch {
            comp_id: 0,
            col_idx: 0,
            drop_fn: None,
            layout: Layout::new::<f32>(),
            entries: vec![(0, Entity::new(0, 0), 0), (1, Entity::new(1, 0), 4)],
        }],
    });

    assert!(!cs.is_empty());
    assert_eq!(cs.len(), 2);
}
```

- [ ] **Step 2: Run test — verify it fails (types don't exist yet)**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_is_empty_and_len
```

Expected: compile error — `ColumnBatch`, `ArchetypeBatch` not found, `archetype_batches` field missing.

- [ ] **Step 3: Add the data structures and field**

In `crates/minkowski/src/changeset.rs`, after the `DropEntry` struct (around line 205), add:

```rust
/// A single column's worth of buffered overwrites within one archetype.
/// Column index and drop function are resolved once when the batch is
/// created, not per entity.
pub(crate) struct ColumnBatch {
    pub(crate) comp_id: ComponentId,
    pub(crate) col_idx: usize,
    pub(crate) drop_fn: Option<unsafe fn(*mut u8)>,
    pub(crate) layout: Layout,
    /// (row, entity, arena_offset) triples. Offsets, not pointers — the
    /// arena may reallocate during recording. Resolved at apply time.
    pub(crate) entries: Vec<(usize, Entity, usize)>,
}

/// All buffered overwrites for a single archetype, grouped by component.
pub(crate) struct ArchetypeBatch {
    pub(crate) arch_idx: usize,
    pub(crate) columns: Vec<ColumnBatch>,
}
```

Add `archetype_batches` field to `EnumChangeSet` struct:

```rust
pub struct EnumChangeSet {
    pub(crate) mutations: Vec<Mutation>,
    pub(crate) arena: Arena,
    drop_entries: Vec<DropEntry>,
    /// Streaming archetype batches from QueryWriter. Applied before the
    /// regular mutation log. Empty for non-QueryWriter changesets.
    pub(crate) archetype_batches: Vec<ArchetypeBatch>,
}
```

Change `drop_entries` from private to `pub(crate)` (needed by `WritableRef::set()` in `reducer.rs`):

```rust
// Before:
    drop_entries: Vec<DropEntry>,
// After:
    pub(crate) drop_entries: Vec<DropEntry>,
```

Initialize `archetype_batches` in **all three constructors** (`new`, `with_capacity`, `new_in`):

```rust
archetype_batches: Vec::new(),
```

Update `is_empty` and `len`:

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

- [ ] **Step 4: Run test — verify it passes**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_is_empty_and_len
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat(changeset): add ColumnBatch, ArchetypeBatch, and archetype_batches field"
```

---

### Task 2: Fast-lane drain in apply_mutations

**Files:**
- Modify: `crates/minkowski/src/changeset.rs:670-864` (apply_mutations)

- [ ] **Step 1: Write test `fast_lane_single_component`**

```rust
#[test]
fn fast_lane_single_component() {
    let mut world = World::new();
    let pos_id = world.register_component::<Pos>();
    let e1 = world.spawn((Pos { x: 0.0, y: 0.0 },));
    let e2 = world.spawn((Pos { x: 0.0, y: 0.0 },));

    // Build a changeset with fast-lane entries (simulating QueryWriter)
    let mut cs = EnumChangeSet::new();

    // Allocate values in the arena
    let v1 = Pos { x: 1.0, y: 2.0 };
    let v1 = std::mem::ManuallyDrop::new(v1);
    let off1 = cs.arena.alloc(&*v1 as *const Pos as *const u8, Layout::new::<Pos>());

    let v2 = Pos { x: 3.0, y: 4.0 };
    let v2 = std::mem::ManuallyDrop::new(v2);
    let off2 = cs.arena.alloc(&*v2 as *const Pos as *const u8, Layout::new::<Pos>());

    // Look up the archetype info to build the batch
    let loc1 = world.entity_location(e1).unwrap();
    let arch = &world.archetypes.archetypes[loc1.archetype_id.0];
    let col_idx = arch.column_index(pos_id).unwrap();

    cs.archetype_batches.push(ArchetypeBatch {
        arch_idx: loc1.archetype_id.0,
        columns: vec![ColumnBatch {
            comp_id: pos_id,
            col_idx,
            drop_fn: None, // Pos is Copy
            layout: Layout::new::<Pos>(),
            entries: vec![
                (0, e1, off1),
                (1, e2, off2),
            ],
        }],
    });

    cs.apply(&mut world).unwrap();
    assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 1.0, y: 2.0 }));
    assert_eq!(world.get::<Pos>(e2), Some(&Pos { x: 3.0, y: 4.0 }));
}
```

Note: This test needs `entity_location` and `archetypes` to be accessible from tests. Check existing test patterns — changeset tests already access `world.archetypes` (e.g., line 1486). If `entity_location` is not public, use `world.entity_locations[e1.index() as usize]` directly.

- [ ] **Step 2: Run test — verify it fails**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_single_component
```

Expected: FAIL — fast-lane entries not processed, world state unchanged.

- [ ] **Step 3: Add fast-lane drain preamble to `apply_mutations`**

At the top of `apply_mutations` (line 676, after the tick parameter), before the existing `let mut batch: Option<InsertBatch> = None;` line, add:

```rust
// ── Fast lane: drain pre-resolved archetype batches ──
// Zero per-entity lookups — column index, drop_fn, layout all
// pre-resolved during recording. Only pointer arithmetic + memcpy.
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
```

- [ ] **Step 4: Run test — verify it passes**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_single_component
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat(changeset): fast-lane drain in apply_mutations"
```

---

### Task 3: Multi-component and multi-archetype tests

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (tests section)

- [ ] **Step 1: Write tests `fast_lane_multi_component` and `fast_lane_multi_archetype`**

`fast_lane_multi_component`: spawn 2 entities with `(Pos, Vel)`, build a batch with two `ColumnBatch` entries (one for Pos, one for Vel). Apply. Verify both components updated.

`fast_lane_multi_archetype`: spawn entities in 3 archetypes: `(Pos)`, `(Pos, Vel)`, `(Pos, Vel, Score)` where `Score` is a test component (use `Health` from existing tests or add `Score(f32)`). Build 3 `ArchetypeBatch` entries. Apply. Verify each archetype's entities updated correctly.

- [ ] **Step 2: Run tests — verify they pass (apply logic already handles this)**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_multi_component
cargo test -p minkowski --lib changeset::tests::fast_lane_multi_archetype
```

Expected: PASS (the apply logic from Task 2 handles any number of batches and columns).

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "test(changeset): multi-component and multi-archetype fast lane tests"
```

---

### Task 4: Drop safety — abort and partial failure

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (tests section + partial-failure retain in `apply`)

- [ ] **Step 1: Write test `fast_lane_drop_on_abort`**

Use a component with a drop counter (like existing `Tracked` in changeset tests). Push values into the fast lane via arena + `DropEntry` with `mutation_idx: usize::MAX`. Drop the changeset without calling apply. Assert drop count matches entry count.

```rust
#[test]
fn fast_lane_drop_on_abort() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let tracked_id = world.register_component::<Tracked>();
    let e = world.spawn((Pos { x: 0.0, y: 0.0 },));

    {
        let mut cs = EnumChangeSet::new();
        let val = Tracked::new(42, &drops);
        let val = std::mem::ManuallyDrop::new(val);
        let offset = cs.arena.alloc(
            &*val as *const Tracked as *const u8,
            Layout::new::<Tracked>(),
        );
        cs.drop_entries.push(DropEntry {
            offset,
            drop_fn: crate::component::drop_ptr::<Tracked>,
            mutation_idx: usize::MAX,
        });
        // Also push a batch entry (not strictly needed for drop test,
        // but verifies the full fast-lane setup doesn't interfere)
        cs.archetype_batches.push(ArchetypeBatch {
            arch_idx: 0,
            columns: vec![ColumnBatch {
                comp_id: tracked_id,
                col_idx: 0,
                drop_fn: Some(crate::component::drop_ptr::<Tracked>),
                layout: Layout::new::<Tracked>(),
                entries: vec![(0, e, offset)],
            }],
        });
        // drop without apply
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
```

Note: `DropEntry` and `drop_entries` are private. Existing tests already access them (check test patterns — if tests are inside `mod tests` within `changeset.rs`, they have access to private fields). If `DropEntry` needs `pub(crate)` or the test needs to be in-module, follow existing patterns.

- [ ] **Step 2: Run test — verify it passes (Drop impl already handles usize::MAX)**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_drop_on_abort
```

Expected: PASS (the `Drop` impl iterates all `drop_entries` unconditionally).

- [ ] **Step 3: Write test `fast_lane_partial_failure_no_double_free`**

Build a changeset with: fast-lane entries (DropEntry with `mutation_idx: usize::MAX`) + regular mutation log entries including one that will fail (e.g., `Mutation::Despawn` on a dead entity). Verify fast-lane drop entries are excluded from the retain filter.

```rust
#[test]
fn fast_lane_partial_failure_no_double_free() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let tracked_id = world.register_component::<Tracked>();
    let e1 = world.spawn((Pos { x: 0.0, y: 0.0 },));

    let mut cs = EnumChangeSet::new();

    // Fast-lane entry for e1 (will succeed)
    let val = Tracked::new(99, &drops);
    let val = std::mem::ManuallyDrop::new(val);
    let offset = cs.arena.alloc(
        &*val as *const Tracked as *const u8,
        Layout::new::<Tracked>(),
    );
    cs.drop_entries.push(DropEntry {
        offset,
        drop_fn: crate::component::drop_ptr::<Tracked>,
        mutation_idx: usize::MAX,
    });
    // Note: for this test, e1 needs Tracked in its archetype, or we use
    // a Pos overwrite. Adjust column setup to match e1's actual archetype.
    let loc = world.entity_locations[e1.index() as usize].unwrap();
    let arch = &world.archetypes.archetypes[loc.archetype_id.0];
    // Use Pos (which e1 already has) to avoid archetype mismatch
    let pos_id = world.register_component::<Pos>();
    let col_idx = arch.column_index(pos_id).unwrap();
    let pos_val = Pos { x: 99.0, y: 99.0 };
    let pos_val = std::mem::ManuallyDrop::new(pos_val);
    let pos_offset = cs.arena.alloc(
        &*pos_val as *const Pos as *const u8,
        Layout::new::<Pos>(),
    );
    cs.archetype_batches.push(ArchetypeBatch {
        arch_idx: loc.archetype_id.0,
        columns: vec![ColumnBatch {
            comp_id: pos_id,
            col_idx,
            drop_fn: None,
            layout: Layout::new::<Pos>(),
            entries: vec![(loc.row, e1, pos_offset)],
        }],
    });

    // Regular mutation that will fail: despawn a dead entity
    let dead = Entity::new(9999, 0);
    cs.record_despawn(dead);

    let result = cs.apply(&mut world);
    assert!(result.is_err());

    // Fast-lane was applied (Pos overwritten)
    assert_eq!(world.get::<Pos>(e1), Some(&Pos { x: 99.0, y: 99.0 }));

    // Tracked was pushed as a drop_entry with usize::MAX.
    // After partial failure, it should NOT be dropped again (ownership
    // transferred during fast-lane drain).
    // Since the Tracked value was in the arena but NOT actually written
    // to a column (we used Pos for the actual batch), it should still
    // be dropped by the retain filter. But the key invariant:
    // no double-free, no panic.
    // The drop count should be exactly 1 (from Drop impl cleanup).
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
```

- [ ] **Step 4: Fix the partial-failure retain filter**

In `crates/minkowski/src/changeset.rs`, in the `apply` method (around line 664), update:

```rust
// Before:
self.drop_entries
    .retain(|entry| entry.mutation_idx >= failed_idx);

// After:
self.drop_entries
    .retain(|entry| entry.mutation_idx >= failed_idx
                 && entry.mutation_idx != usize::MAX);
```

- [ ] **Step 5: Run both tests**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_drop_on_abort
cargo test -p minkowski --lib changeset::tests::fast_lane_partial_failure_no_double_free
```

Expected: PASS.

- [ ] **Step 6: Write test `fast_lane_empty_batch`**

Changeset with an `ArchetypeBatch` containing a `ColumnBatch` with zero entries. Apply should succeed, column should NOT be marked changed.

- [ ] **Step 7: Run test — verify it passes**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_empty_batch
```

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat(changeset): drop safety for fast lane — abort, partial failure, empty batch"
```

---

### Task 5: drain_fast_lane_to_mutations (WAL support)

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (new method on EnumChangeSet)

- [ ] **Step 1: Write test `fast_lane_drain_to_mutations`**

Build a changeset with fast-lane entries (including a DropEntry with `mutation_idx: usize::MAX`). Call `drain_fast_lane_to_mutations()`. Assert: `archetype_batches` is empty, `mutations` contains the expected `Mutation::Insert` entries, and `DropEntry.mutation_idx` is rebased from `usize::MAX` to the real index.

```rust
#[test]
fn fast_lane_drain_to_mutations() {
    let mut cs = EnumChangeSet::new();

    let v1 = Pos { x: 1.0, y: 2.0 };
    let v1 = std::mem::ManuallyDrop::new(v1);
    let off1 = cs.arena.alloc(&*v1 as *const Pos as *const u8, Layout::new::<Pos>());

    let pos_id = 0;
    let e1 = Entity::new(0, 0);

    cs.drop_entries.push(DropEntry {
        offset: off1,
        drop_fn: crate::component::drop_ptr::<Pos>,
        mutation_idx: usize::MAX,
    });

    cs.archetype_batches.push(ArchetypeBatch {
        arch_idx: 0,
        columns: vec![ColumnBatch {
            comp_id: pos_id,
            col_idx: 0,
            drop_fn: None,
            layout: Layout::new::<Pos>(),
            entries: vec![(0, e1, off1)],
        }],
    });

    cs.drain_fast_lane_to_mutations();

    assert!(cs.archetype_batches.is_empty());
    assert_eq!(cs.mutations.len(), 1);
    match &cs.mutations[0] {
        Mutation::Insert { entity, component_id, offset, layout } => {
            assert_eq!(*entity, e1);
            assert_eq!(*component_id, pos_id);
            assert_eq!(*offset, off1);
            assert_eq!(*layout, Layout::new::<Pos>());
        }
        _ => panic!("expected Insert mutation"),
    }
    // DropEntry rebased from usize::MAX to 0
    assert_eq!(cs.drop_entries[0].mutation_idx, 0);
}
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_drain_to_mutations
```

- [ ] **Step 3: Implement `drain_fast_lane_to_mutations`**

Add to `impl EnumChangeSet`:

```rust
/// Convert fast-lane archetype batches into regular `Mutation::Insert`
/// entries. Called by the WAL path before serialization.
///
/// Rebases `DropEntry.mutation_idx` from `usize::MAX` to real indices
/// so partial-failure retain logic works after drain.
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

- [ ] **Step 4: Run test — verify it passes**

```bash
cargo test -p minkowski --lib changeset::tests::fast_lane_drain_to_mutations
```

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat(changeset): drain_fast_lane_to_mutations for WAL serialization"
```

---

### Task 6: open_archetype_batch helper

**Files:**
- Modify: `crates/minkowski/src/changeset.rs` (new pub(crate) function)

- [ ] **Step 1: Add `open_archetype_batch` function**

This is a helper used by QueryWriter's `for_each`. Add near the `flush_insert_batch` function:

```rust
/// Open a new archetype batch on the fast lane. Resolves each mutable
/// component's column index and drop function once per archetype.
///
/// Called by `QueryWriter::for_each` at each archetype boundary.
pub(crate) fn open_archetype_batch(
    cs: &mut EnumChangeSet,
    arch_idx: usize,
    arch: &crate::storage::archetype::Archetype,
    components: &ComponentRegistry,
    mutable_ids: &fixedbitset::FixedBitSet,
) {
    let columns: Vec<ColumnBatch> = mutable_ids.ones().map(|comp_id| {
        let comp_id = ComponentId(comp_id);
        let col_idx = arch.column_index(comp_id).unwrap();
        let info = components.info(comp_id);
        ColumnBatch {
            comp_id,
            col_idx,
            drop_fn: info.drop_fn,
            layout: info.layout,
            entries: Vec::new(),
        }
    }).collect();

    cs.archetype_batches.push(ArchetypeBatch { arch_idx, columns });
}
```

Note: Check how `ComponentRegistry::info` returns data — it may return a struct with `drop_fn` and `layout` fields, or they may be accessed differently. Match the existing pattern from `changeset_insert_raw` (line 730 area).

- [ ] **Step 2: Run full changeset tests to verify no regressions**

```bash
cargo test -p minkowski --lib changeset::tests
```

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat(changeset): open_archetype_batch helper for QueryWriter"
```

---

## Chunk 2: Recording Path (reducer.rs)

### Task 7: Add `row` and `column_slot` to WritableRef

**Files:**
- Modify: `crates/minkowski/src/reducer.rs:718-768` (WritableRef struct + new/set/modify)

- [ ] **Step 1: Add `row` and `column_slot` fields to WritableRef**

```rust
pub struct WritableRef<'a, T: Component> {
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: *mut EnumChangeSet,
    row: usize,
    column_slot: usize,
    _marker: PhantomData<&'a EnumChangeSet>,
}
```

Update `WritableRef::new` to accept and store `row` and `column_slot`:

```rust
pub(crate) fn new(
    entity: Entity,
    current: &'a T,
    comp_id: ComponentId,
    changeset: *mut EnumChangeSet,
    row: usize,
    column_slot: usize,
) -> Self {
    Self {
        entity,
        current,
        comp_id,
        changeset,
        row,
        column_slot,
        _marker: PhantomData,
    }
}
```

- [ ] **Step 2: Update `set()` to use fast lane**

```rust
#[inline]
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
            mutation_idx: usize::MAX,
        });
    }
}
```

Note: `cs.drop_entries` is private. Since `WritableRef::set` accesses the changeset through a raw pointer, and `reducer.rs` is in the same crate, this should work. If `drop_entries` needs to be `pub(crate)`, update it. Also add `use std::alloc::Layout;` if not already imported in this scope.

`modify()` already calls `self.set(val)` — no changes needed.

- [ ] **Step 3: Fix compilation — update all `WritableRef::new` call sites**

The only call site is in `fetch_writer` for `&mut T` (line 847):

```rust
// Before:
WritableRef::new(entity, current, *comp_id, changeset)

// After:
WritableRef::new(entity, current, *comp_id, changeset, row, column_slot)
```

But `column_slot` isn't in the fetch yet — that's Task 8. For now, to keep things compiling, temporarily pass `0` for both and mark with a `// TODO: resolve from fetch` comment. Or combine Tasks 7 and 8.

**Decision: Combine this step with Task 8 to avoid a broken intermediate state.**

- [ ] **Step 4: Commit (combined with Task 8 below)**

---

### Task 8: Update WriterQuery trait and impls for column_slot

**Files:**
- Modify: `crates/minkowski/src/reducer.rs:780-958` (WriterQuery trait + impls + tuple macro)

- [ ] **Step 1: Update `&mut T` WriterFetch to include `column_slot`**

Change the `WriterFetch` associated type for `&mut T`:

```rust
// Before:
type WriterFetch<'a> = (ThinSlicePtr<T>, ComponentId);

// After:
type WriterFetch<'a> = (ThinSlicePtr<T>, ComponentId, usize); // +column_slot
```

Update `init_writer_fetch`:

```rust
fn init_writer_fetch<'w>(
    archetype: &'w Archetype,
    registry: &ComponentRegistry,
) -> Self::WriterFetch<'w> {
    let id = registry.id::<T>().expect("component not registered");
    let ptr = <&T as WorldQuery>::init_fetch(archetype, registry);
    // column_slot: this component's position in mutable_ids iteration order.
    // Resolved once per archetype.
    let mutable = <&mut T as WorldQuery>::mutable_ids(registry);
    let column_slot = mutable.ones()
        .position(|bit| bit == id.0)
        .unwrap();
    (ptr, id, column_slot)
}
```

Wait — for a single `&mut T`, `mutable_ids` returns a bitset with just one bit set, so `column_slot` is always 0. The slot only becomes meaningful in tuple queries. But the tuple macro delegates `init_writer_fetch` to each element, which can't know its position in the tuple's combined `mutable_ids`.

**Revised approach**: The `column_slot` must be computed at the **tuple level**, not per-element. The tuple macro's `init_writer_fetch` knows the full set. But the current trait design has each element resolve its own fetch independently.

**Simplest fix**: Have each element's `WriterFetch` store a `column_slot: usize` that defaults to 0. After the tuple constructs all fetches, a **post-init pass** sets the correct slot for each `&mut T` element based on the combined `mutable_ids` order.

**Even simpler**: Add a `set_column_slot` method to `WriterQuery` (default no-op for `&T`/`Entity`/etc., stores the value for `&mut T`). The tuple macro calls it after init.

**Simplest of all**: Store `column_slot` in the `WriterFetch` tuple element. For `&mut T`, init with 0. Add a trait method `fn with_column_slot(fetch: &mut Self::WriterFetch<'_>, slot: usize)` that `&mut T` implements to set the slot, and all others implement as no-op. The tuple macro calls this after constructing all fetches.

```rust
// Add to WriterQuery trait:
fn set_column_slot(_fetch: &mut Self::WriterFetch<'_>, _slot: usize) {}
```

For `&mut T`:

```rust
fn set_column_slot(fetch: &mut Self::WriterFetch<'_>, slot: usize) {
    fetch.2 = slot;
}
```

Update `fetch_writer` for `&mut T`:

```rust
unsafe fn fetch_writer<'w>(
    fetch: &Self::WriterFetch<'w>,
    row: usize,
    entity: Entity,
    changeset: *mut EnumChangeSet,
) -> Self::WriterItem<'w> {
    unsafe {
        let (ptr, comp_id, column_slot) = fetch;
        let current: &T = &*ptr.ptr.add(row);
        WritableRef::new(entity, current, *comp_id, changeset, row, *column_slot)
    }
}
```

Update the **tuple macro** to compute and assign slots:

```rust
macro_rules! impl_writer_query_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: WriterQuery),*> WriterQuery for ($($name,)*) {
            type WriterItem<'a> = ($($name::WriterItem<'a>,)*);
            type WriterFetch<'a> = ($($name::WriterFetch<'a>,)*);

            fn init_writer_fetch<'w>(
                archetype: &'w Archetype,
                registry: &ComponentRegistry,
            ) -> Self::WriterFetch<'w> {
                let mut fetch = ($($name::init_writer_fetch(archetype, registry),)*);
                // Assign column_slot based on position in combined mutable_ids
                let mutable = <Self as WorldQuery>::mutable_ids(registry);
                let mut slot = 0usize;
                let ($($name,)*) = &mut fetch;
                $(
                    let sub_mutable = <$name as WorldQuery>::mutable_ids(registry);
                    if sub_mutable.count_ones(..) > 0 {
                        <$name as WriterQuery>::set_column_slot($name, slot);
                        slot += sub_mutable.count_ones(..);
                    }
                )*
                debug_assert_eq!(
                    slot, mutable.count_ones(..),
                    "column_slot assignment out of sync with mutable_ids"
                );
                fetch
            }

            unsafe fn fetch_writer<'w>(
                fetch: &Self::WriterFetch<'w>,
                row: usize,
                entity: Entity,
                changeset: *mut EnumChangeSet,
            ) -> Self::WriterItem<'w> { unsafe {
                let ($($name,)*) = fetch;
                ($(<$name as WriterQuery>::fetch_writer($name, row, entity, changeset),)*)
            }}

            fn set_column_slot(_fetch: &mut Self::WriterFetch<'_>, _slot: usize) {
                // Tuple assigns slots in init_writer_fetch, not here
            }
        }
    };
}
```

- [ ] **Step 2: Verify compilation**

```bash
cargo check -p minkowski
```

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): WritableRef gains row + column_slot, set() routes to fast lane"
```

---

### Task 9: Update QueryWriter::for_each to open archetype batches

**Files:**
- Modify: `crates/minkowski/src/reducer.rs:1020-1067` (for_each method)

- [ ] **Step 1: Update `for_each` to open batches and compute mutable_ids once**

```rust
pub fn for_each(&mut self, mut f: impl FnMut(Q::WriterItem<'_>)) {
    self.queried.store(true, Ordering::Relaxed);
    let last_tick = Tick::new(self.last_read_tick.load(Ordering::Relaxed));

    let required = Q::required_ids(&self.world.components);
    let mutable = Q::mutable_ids(&self.world.components);
    let cs_ptr = self.changeset;

    // Pre-allocate arena capacity (mutations vec no longer used for overwrites)
    {
        const MAX_PREALLOC_MUTATIONS: usize = 64 * 1024;
        let mut entity_count = 0;
        for arch in &self.world.archetypes.archetypes {
            if !arch.is_empty()
                && required.is_subset(&arch.component_ids)
                && Q::matches_filters(arch, &self.world.components, last_tick)
            {
                entity_count += arch.len();
            }
        }
        if entity_count > 0 {
            let cs = unsafe { &mut *cs_ptr };
            let mutable_count = mutable.count_ones(..);
            let mutations_needed =
                (entity_count * mutable_count).min(MAX_PREALLOC_MUTATIONS);
            // Only reserve arena — mutations vec not used for fast-lane overwrites
            cs.arena.reserve(mutations_needed * 64);
        }
    }

    for (arch_idx, arch) in self.world.archetypes.archetypes.iter().enumerate() {
        if arch.is_empty() || !required.is_subset(&arch.component_ids) {
            continue;
        }
        if !Q::matches_filters(arch, &self.world.components, last_tick) {
            continue;
        }

        // Open a fast-lane batch for this archetype
        let cs = unsafe { &mut *cs_ptr };
        crate::changeset::open_archetype_batch(
            cs, arch_idx, arch, &self.world.components, &mutable,
        );

        let fetch = Q::init_writer_fetch(arch, &self.world.components);
        for row in 0..arch.len() {
            let entity = arch.entities[row];
            let item = unsafe { Q::fetch_writer(&fetch, row, entity, cs_ptr) };
            f(item);
        }
    }
}
```

- [ ] **Step 2: Run all existing reducer tests**

```bash
cargo test -p minkowski --lib reducer::tests
```

Expected: PASS — existing tests exercise QueryWriter end-to-end.

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "feat(reducer): for_each opens archetype batches, pre-alloc updated"
```

---

### Task 10: Integration tests for QueryWriter + fast lane

**Files:**
- Modify: `crates/minkowski/src/reducer.rs` (tests section)

- [ ] **Step 1: Write test `query_writer_fast_lane_roundtrip`**

Register a `QueryWriter<(&mut Pos, &Vel)>`, iterate with `modify`, verify world state matches expected values. This is the end-to-end smoke test.

```rust
#[test]
fn query_writer_fast_lane_roundtrip() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((
            Pos { x: i as f32, y: 0.0 },
            Vel { dx: 1.0, dy: 2.0 },
        ));
    }

    let strategy = crate::Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Pos, &Vel), (), _>(
            &mut world,
            "integrate",
            |mut qw: QueryWriter<'_, (&mut Pos, &Vel)>, ()| {
                qw.for_each(|(mut pos, vel)| {
                    pos.modify(|p| {
                        p.x += vel.dx;
                        p.y += vel.dy;
                    });
                });
            },
        )
        .unwrap();

    registry.call(&strategy, &mut world, id, ()).unwrap();

    // Verify first and last entity
    // All entities have same archetype, same vel
    let entities: Vec<_> = world.query::<(Entity, &Pos)>()
        .iter()
        .map(|(e, p)| (e, p.clone()))
        .collect();
    assert_eq!(entities.len(), 100);
    // Entity 0: x was 0 + 1 = 1, y was 0 + 2 = 2
    assert_eq!(entities[0].1.x, 1.0);
    assert_eq!(entities[0].1.y, 2.0);
}
```

- [ ] **Step 2: Write test `query_writer_fast_lane_change_detection`**

Call QueryWriter, then check `Changed<Pos>` matches on next tick. Matches existing change detection patterns in reducer tests.

- [ ] **Step 3: Write test `query_writer_conditional_update`**

In the `for_each` closure, only call `set`/`modify` for entities with `pos.x > 50`. Verify unchanged entities retain original values.

- [ ] **Step 4: Write test `query_writer_read_only_components`**

`(&Pos, &mut Vel)` — verify Pos is unmodified after apply.

- [ ] **Step 5: Run all tests**

```bash
cargo test -p minkowski --lib reducer::tests
```

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "test(reducer): QueryWriter fast-lane integration tests"
```

---

## Chunk 3: WAL Integration, Miri, Benchmarks

### Task 11: WAL integration — drain before append

**Files:**
- Modify: `crates/minkowski-persist/src/durable.rs:88-98`

- [ ] **Step 1: Add `drain_fast_lane_to_mutations` call before WAL append**

In `durable.rs`, in the `Ok(forward) => { ... }` branch (around line 88):

```rust
// Before:
Ok(forward) => {
    tx.mark_committed();
    drop(tx);
    // WAL write BEFORE apply — durable commit point
    self.wal
        .lock()
        .append(&forward, &self.codecs)

// After:
Ok(mut forward) => {
    tx.mark_committed();
    drop(tx);
    // Convert fast-lane batches to regular mutations for WAL serialization
    forward.drain_fast_lane_to_mutations();
    // WAL write BEFORE apply — durable commit point
    self.wal
        .lock()
        .append(&forward, &self.codecs)
```

- [ ] **Step 2: Run persist tests**

```bash
cargo test -p minkowski-persist
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski-persist/src/durable.rs
git commit -m "feat(persist): drain fast lane before WAL append"
```

---

### Task 12: Miri subset update

**Files:**
- Modify: `ci/miri-subset.txt`

- [ ] **Step 1: Add fast-lane tests to Miri subset**

Append to the changeset section:

```
changeset::tests::fast_lane_single_component
changeset::tests::fast_lane_drop_on_abort
changeset::tests::fast_lane_partial_failure_no_double_free
```

Append to a new section (or the reducer section if one exists):

```
# ── Fast-lane: streaming archetype buffers (4 selected) ────────────
reducer::tests::query_writer_fast_lane_roundtrip
```

- [ ] **Step 2: Run Miri on the new tests**

```bash
cargo +nightly miri test -p minkowski --lib -- \
    "changeset::tests::fast_lane_single_component|changeset::tests::fast_lane_drop_on_abort|changeset::tests::fast_lane_partial_failure_no_double_free|reducer::tests::query_writer_fast_lane_roundtrip"
```

Note: Miri is slow — expect 2-5 minutes per test with Tree Borrows.

- [ ] **Step 3: Commit**

```bash
git add ci/miri-subset.txt
git commit -m "ci: add fast-lane tests to Miri subset"
```

---

### Task 13: Benchmarks

**Files:**
- Modify: `crates/minkowski-bench/benches/reducer.rs`

- [ ] **Step 1: Add 4 new benchmarks**

The existing `query_writer_10k` benchmark already exercises the fast lane (it uses `QueryWriter::for_each` → `modify`). The fast lane is now the default path — no separate "fast_lane" benchmarks are needed. The existing benchmark IS the fast-lane benchmark.

Instead, add benchmarks for the new scenarios from the spec:

```rust
fn bench_query_writer_multi_comp(c: &mut Criterion) {
    let mut world = World::new();
    for i in 0..10_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0, z: 0.0 },
            Velocity { dx: 1.0, dy: 0.0, dz: 0.0 },
        ));
    }

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position, &mut Velocity), (), _>(
            &mut world,
            "update_both",
            |mut qw: QueryWriter<'_, (&mut Position, &mut Velocity)>, ()| {
                qw.for_each(|(mut pos, mut vel)| {
                    pos.modify(|p| p.x += 1.0);
                    vel.modify(|v| v.dx *= 0.99);
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_writer_multi_comp_10k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}

fn bench_query_writer_sparse_update(c: &mut Criterion) {
    let mut world = setup_world();
    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position, &Velocity), (), _>(
            &mut world,
            "sparse_update",
            |mut qw: QueryWriter<'_, (&mut Position, &Velocity)>, ()| {
                qw.for_each(|(mut pos, vel)| {
                    // Only update 10% of entities
                    if vel.dx > 0.9 {
                        pos.modify(|p| p.x += vel.dx);
                    }
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_writer_sparse_10k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}

fn bench_query_writer_multi_arch(c: &mut Criterion) {
    use minkowski_bench::Team;
    let mut world = World::new();
    // 3 archetypes × 3K entities
    for i in 0..3_000 {
        world.spawn((Position { x: i as f32, y: 0.0, z: 0.0 },));
    }
    for i in 0..3_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0, z: 0.0 },
            Velocity { dx: 1.0, dy: 0.0, dz: 0.0 },
        ));
    }
    for i in 0..3_000 {
        world.spawn((
            Position { x: i as f32, y: 0.0, z: 0.0 },
            Velocity { dx: 1.0, dy: 0.0, dz: 0.0 },
            Team(1),
        ));
    }

    let strategy = Optimistic::new(&world);
    let mut registry = ReducerRegistry::new();
    let id = registry
        .register_query_writer::<(&mut Position,), (), _>(
            &mut world,
            "multi_arch",
            |mut qw: QueryWriter<'_, (&mut Position,)>, ()| {
                qw.for_each(|(mut pos,)| {
                    pos.modify(|p| p.x += 1.0);
                });
            },
        )
        .unwrap();

    c.bench_function("reducer/query_writer_multi_arch_9k", |b| {
        b.iter(|| {
            registry.call(&strategy, &mut world, id, ()).unwrap();
        });
    });
}
```

Add to `criterion_group!`:

```rust
criterion_group!(
    benches,
    bench_query_mut,
    bench_query_writer,
    bench_query_writer_multi_comp,
    bench_query_writer_sparse_update,
    bench_query_writer_multi_arch,
    bench_dynamic_for_each,
);
```

- [ ] **Step 2: Run benchmarks to establish baseline**

```bash
cargo bench -p minkowski-bench --bench reducer
```

Record results. Compare `reducer/query_writer_10k` against previous baseline (72µs from PR #125).

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski-bench/benches/reducer.rs
git commit -m "bench: add multi-component, sparse update, multi-archetype QueryWriter benchmarks"
```

---

### Task 14: Run full test suite and verify

- [ ] **Step 1: Run all tests**

```bash
cargo test --workspace
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 3: Run formatter**

```bash
cargo fmt --check
```

- [ ] **Step 4: Fix any issues**

- [ ] **Step 5: Final commit if needed**

```bash
git commit -am "fix: address clippy/fmt issues from fast-lane implementation"
```

---

## Execution Notes

**Critical ordering**: Tasks 7-9 must be implemented together (or 7+8 combined, then 9) to avoid a broken intermediate state. `WritableRef::new` signature change (Task 7) requires `fetch_writer` update (Task 8) which requires `for_each` to open batches (Task 9). The safest path: implement all three in sequence, compile-check after each, commit once when all three pass.

**Test components**: The changeset unit tests (Tasks 2-4) need direct access to `world.archetypes`, `entity_locations`, `ColumnBatch`, `ArchetypeBatch`, and `DropEntry`. Since the tests are in `mod tests` inside `changeset.rs`, they have access to private fields. If any field access fails, add `pub(crate)` as needed.

**The existing `query_writer_10k` benchmark IS the fast-lane benchmark**: After this implementation, all QueryWriter `for_each` calls automatically use the fast lane. There's no opt-in or separate code path. The existing benchmark measures the new performance directly.

**`ComponentRegistry::info` return type**: Check the actual return type. The spec assumes it has `drop_fn` and `layout` fields. If the API differs, adjust `open_archetype_batch` accordingly.
