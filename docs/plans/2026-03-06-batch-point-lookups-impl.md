# Batch Point-Lookups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `get_batch` and `get_batch_mut` to World — batch point-lookups that group by archetype for cache locality, composing with external indexes.

**Architecture:** Two new public methods on World. `get_batch<T>(&self, &[Entity]) -> Vec<Option<&T>>` for reads, `get_batch_mut<T>(&mut self, &[Entity]) -> Vec<Option<&mut T>>` for writes. Both resolve ComponentId once, group entities by archetype, then fetch all rows per archetype. `get_batch_mut` panics on duplicate entities (aliased `&mut T` is UB — unconditional assert).

**Tech Stack:** Pure Rust, no new dependencies. Uses existing `BlobVec::get_ptr`/`get_ptr_mut`, `EntityAllocator::is_alive`, `SparseStorage::get`/`get_mut`.

**Key files:**
- Implementation: `crates/minkowski/src/world.rs` (add after `get_mut` at line ~420)
- Tests: `crates/minkowski/src/world.rs` (in existing `#[cfg(test)] mod tests` at line 962)
- Exports: `crates/minkowski/src/lib.rs` (no change needed — methods are on World which is already pub)
- Example update: `examples/examples/index.rs`
- CLAUDE.md: add to pub API notes

---

### Task 1: Write failing tests for `get_batch` (read-only)

**Files:**
- Modify: `crates/minkowski/src/world.rs:962-1632` (test module)

**Step 1: Add test component types and tests**

Add these tests at the end of the existing `mod tests` block (before the final `}`). The test module already has `Pos` and `Vel` structs defined.

```rust
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Health(u32);

    #[test]
    fn get_batch_basic() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));
        let e3 = world.spawn((Health(25),));

        let results = world.get_batch::<Health>(&[e1, e2, e3]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(&Health(100)));
        assert_eq!(results[1], Some(&Health(50)));
        assert_eq!(results[2], Some(&Health(25)));
    }

    #[test]
    fn get_batch_dead_entity() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));
        world.despawn(e1);

        let results = world.get_batch::<Health>(&[e1, e2]);
        assert_eq!(results[0], None);
        assert_eq!(results[1], Some(&Health(50)));
    }

    #[test]
    fn get_batch_missing_component() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Pos { x: 1.0, y: 2.0 },)); // No Health

        let results = world.get_batch::<Health>(&[e1, e2]);
        assert_eq!(results[0], Some(&Health(100)));
        assert_eq!(results[1], None);
    }

    #[test]
    fn get_batch_empty_input() {
        let world = World::new();
        let results = world.get_batch::<Health>(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn get_batch_unregistered_type() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
        // Health was never registered
        let results = world.get_batch::<Health>(&[e]);
        assert_eq!(results[0], None);
    }

    #[test]
    fn get_batch_multi_archetype() {
        let mut world = World::new();
        let e1 = world.spawn((Health(10),));
        let e2 = world.spawn((Health(20), Pos { x: 0.0, y: 0.0 }));
        let e3 = world.spawn((Health(30),));

        let results = world.get_batch::<Health>(&[e1, e2, e3]);
        assert_eq!(results[0], Some(&Health(10)));
        assert_eq!(results[1], Some(&Health(20)));
        assert_eq!(results[2], Some(&Health(30)));
    }

    #[test]
    fn get_batch_duplicate_entity() {
        let mut world = World::new();
        let e = world.spawn((Health(42),));

        let results = world.get_batch::<Health>(&[e, e]);
        assert_eq!(results[0], Some(&Health(42)));
        assert_eq!(results[1], Some(&Health(42)));
    }

    #[test]
    fn get_batch_preserves_order() {
        let mut world = World::new();
        let e1 = world.spawn((Health(1),));
        let e2 = world.spawn((Health(2),));
        let e3 = world.spawn((Health(3),));

        // Request in reverse order
        let results = world.get_batch::<Health>(&[e3, e1, e2]);
        assert_eq!(results[0], Some(&Health(3)));
        assert_eq!(results[1], Some(&Health(1)));
        assert_eq!(results[2], Some(&Health(2)));
    }

    #[test]
    fn get_batch_sparse() {
        let mut world = World::new();
        world.register_sparse::<Health>();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        world.insert_sparse(e1, Health(100));
        world.insert_sparse(e2, Health(50));

        let results = world.get_batch::<Health>(&[e1, e2]);
        assert_eq!(results[0], Some(&Health(100)));
        assert_eq!(results[1], Some(&Health(50)));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- get_batch 2>&1 | head -30`

Expected: Compilation error — `get_batch` method doesn't exist on World.

**Step 3: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "test: add failing tests for World::get_batch"
```

---

### Task 2: Implement `get_batch`

**Files:**
- Modify: `crates/minkowski/src/world.rs` (add after `get_mut` method, around line 420)

**Step 1: Add the implementation**

Insert after the closing `}` of `get_mut` (line ~420), before `query`:

```rust
    /// Fetch a component for multiple entities, grouped by archetype for
    /// cache locality. Returns results in the same order as the input slice.
    /// Dead entities and entities missing the component yield `None`.
    ///
    /// This amortises the per-entity overhead of [`get`](Self::get): the
    /// `ComponentId` is resolved once, entities are grouped by archetype,
    /// and all rows in each archetype are fetched together.
    pub fn get_batch<T: Component>(&self, entities: &[Entity]) -> Vec<Option<&T>> {
        let mut results = vec![None; entities.len()];
        if entities.is_empty() {
            return results;
        }

        let comp_id = match self.components.id::<T>() {
            Some(id) => id,
            None => return results,
        };

        // Sparse fast path — no archetype grouping benefit
        if self.components.is_sparse(comp_id) {
            for (i, &entity) in entities.iter().enumerate() {
                if self.entities.is_alive(entity) {
                    results[i] = self.sparse.get::<T>(comp_id, entity);
                }
            }
            return results;
        }

        // Group by archetype for cache locality.
        // Key: archetype index. Value: vec of (result_index, row).
        let mut by_arch: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (i, &entity) in entities.iter().enumerate() {
            if !self.entities.is_alive(entity) {
                continue;
            }
            if let Some(location) = self.entity_locations[entity.index() as usize] {
                by_arch
                    .entry(location.archetype_id.0)
                    .or_default()
                    .push((i, location.row));
            }
        }

        // Fetch per-archetype: one column lookup per archetype, not per entity.
        for (arch_idx, rows) in &by_arch {
            let arch = &self.archetypes.archetypes[*arch_idx];
            if let Some(&col_idx) = arch.component_index.get(&comp_id) {
                for &(result_idx, row) in rows {
                    results[result_idx] = unsafe {
                        let ptr = arch.columns[col_idx].get_ptr(row) as *const T;
                        Some(&*ptr)
                    };
                }
            }
        }

        results
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- get_batch`

Expected: All 9 `get_batch` tests pass.

**Step 3: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5`

Expected: No warnings.

**Step 4: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add World::get_batch for batched read-only point lookups"
```

---

### Task 3: Write failing tests for `get_batch_mut`

**Files:**
- Modify: `crates/minkowski/src/world.rs` (test module)

**Step 1: Add tests**

Add after the `get_batch` tests:

```rust
    #[test]
    fn get_batch_mut_basic() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));

        let results = world.get_batch_mut::<Health>(&[e1, e2]);
        assert_eq!(results.len(), 2);
        *results[0].unwrap() = Health(200);
        *results[1].unwrap() = Health(75);

        assert_eq!(world.get::<Health>(e1), Some(&Health(200)));
        assert_eq!(world.get::<Health>(e2), Some(&Health(75)));
    }

    #[test]
    fn get_batch_mut_marks_changed() {
        let mut world = World::new();
        let e = world.spawn((Health(100),));

        let tick_before = world.change_tick();
        let _results = world.get_batch_mut::<Health>(&[e]);
        let tick_after = world.change_tick();

        // Tick must have advanced (column marked + read tick)
        assert!(tick_after.0 .0 > tick_before.0 .0);
    }

    #[test]
    fn get_batch_mut_dead_entity() {
        let mut world = World::new();
        let e1 = world.spawn((Health(100),));
        let e2 = world.spawn((Health(50),));
        world.despawn(e1);

        let results = world.get_batch_mut::<Health>(&[e1, e2]);
        assert!(results[0].is_none());
        assert_eq!(*results[1].unwrap(), Health(50));
    }

    #[test]
    #[should_panic(expected = "duplicate entity")]
    fn get_batch_mut_duplicate_panics() {
        let mut world = World::new();
        let e = world.spawn((Health(42),));
        let _results = world.get_batch_mut::<Health>(&[e, e]);
    }

    #[test]
    fn get_batch_mut_sparse() {
        let mut world = World::new();
        world.register_sparse::<Health>();
        let e = world.spawn(());
        world.insert_sparse(e, Health(100));

        let results = world.get_batch_mut::<Health>(&[e]);
        *results[0].unwrap() = Health(200);

        assert_eq!(world.get::<Health>(e), Some(&Health(200)));
    }

    #[test]
    fn get_batch_mut_multi_archetype() {
        let mut world = World::new();
        let e1 = world.spawn((Health(10),));
        let e2 = world.spawn((Health(20), Pos { x: 0.0, y: 0.0 }));

        let results = world.get_batch_mut::<Health>(&[e1, e2]);
        *results[0].unwrap() = Health(11);
        *results[1].unwrap() = Health(21);

        assert_eq!(world.get::<Health>(e1), Some(&Health(11)));
        assert_eq!(world.get::<Health>(e2), Some(&Health(21)));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- get_batch_mut 2>&1 | head -30`

Expected: Compilation error — `get_batch_mut` method doesn't exist.

**Step 3: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "test: add failing tests for World::get_batch_mut"
```

---

### Task 4: Implement `get_batch_mut`

**Files:**
- Modify: `crates/minkowski/src/world.rs` (add after `get_batch`)

**Step 1: Add the implementation**

Insert after `get_batch`:

```rust
    /// Mutable batch fetch — same archetype-grouped pattern as
    /// [`get_batch`](Self::get_batch), but returns `&mut T` and marks
    /// accessed columns as changed for [`Changed<T>`](crate::query::fetch::Changed)
    /// detection.
    ///
    /// # Panics
    ///
    /// Panics if the same entity appears more than once in `entities`.
    /// Aliased `&mut T` is undefined behaviour — this check is unconditional.
    pub fn get_batch_mut<T: Component>(&mut self, entities: &[Entity]) -> Vec<Option<&mut T>> {
        self.drain_orphans();

        let mut results: Vec<Option<&mut T>> = (0..entities.len()).map(|_| None).collect();
        if entities.is_empty() {
            return results;
        }

        let comp_id = match self.components.id::<T>() {
            Some(id) => id,
            None => return results,
        };

        // Sparse fast path
        if self.components.is_sparse(comp_id) {
            // Check for duplicates — collect alive entity indices first
            let mut seen = std::collections::HashSet::with_capacity(entities.len());
            for &entity in entities {
                if self.entities.is_alive(entity) {
                    assert!(
                        seen.insert(entity),
                        "duplicate entity in get_batch_mut: {:?}",
                        entity
                    );
                }
            }
            for (i, &entity) in entities.iter().enumerate() {
                if self.entities.is_alive(entity) {
                    results[i] = self.sparse.get_mut::<T>(comp_id, entity);
                }
            }
            return results;
        }

        // Group by archetype. Detect duplicates during grouping.
        let mut seen = std::collections::HashSet::with_capacity(entities.len());
        let mut by_arch: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (i, &entity) in entities.iter().enumerate() {
            if !self.entities.is_alive(entity) {
                continue;
            }
            assert!(
                seen.insert(entity),
                "duplicate entity in get_batch_mut: {:?}",
                entity
            );
            if let Some(location) = self.entity_locations[entity.index() as usize] {
                by_arch
                    .entry(location.archetype_id.0)
                    .or_default()
                    .push((i, location.row));
            }
        }

        // Mark columns changed and fetch per-archetype.
        let tick = self.next_tick();
        for (arch_idx, rows) in &by_arch {
            let arch = &mut self.archetypes.archetypes[*arch_idx];
            if let Some(&col_idx) = arch.component_index.get(&comp_id) {
                arch.columns[col_idx].mark_changed(tick);
                for &(result_idx, row) in rows {
                    results[result_idx] = unsafe {
                        let ptr = arch.columns[col_idx].get_ptr_mut(row, tick) as *mut T;
                        Some(&mut *ptr)
                    };
                }
            }
        }

        results
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- get_batch_mut`

Expected: All 6 `get_batch_mut` tests pass (including `get_batch_mut_duplicate_panics`).

**Step 3: Run full test suite**

Run: `cargo test -p minkowski --lib`

Expected: All tests pass (no regressions).

**Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5`

Expected: No warnings.

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add World::get_batch_mut with duplicate entity detection"
```

---

### Task 5: Update index example to demonstrate batch fetch

**Files:**
- Modify: `examples/examples/index.rs`

**Step 1: Add batch fetch section**

Add before the `println!("Done.");` line at the end of `main()`:

```rust
    // -- Batch fetch: index -> get_batch composition --
    // Rebuild btree (it was rebuilt after despawn above)
    let low_scores: Vec<_> = btree
        .range(Score(10)..Score(20))
        .flat_map(|(_, entities)| entities.iter().copied())
        .collect();

    println!(
        "Batch fetch: {} entities with Score in [10..20)",
        low_scores.len()
    );

    // Batch read — one call instead of N individual get() calls.
    // Groups by archetype internally for cache locality.
    let names = world.get_batch::<Name>(&low_scores);
    let mut named = 0;
    let mut unnamed = 0;
    for name in &names {
        match name {
            Some(_) => named += 1,
            None => unnamed += 1,
        }
    }
    println!(
        "  {} with Name component, {} without (different archetypes)",
        named, unnamed
    );
    println!();
```

**Step 2: Run the example**

Run: `cargo run -p minkowski-examples --example index --release 2>&1 | tail -10`

Expected: Output includes "Batch fetch:" lines showing the composition.

**Step 3: Commit**

```bash
git add examples/examples/index.rs
git commit -m "example: demonstrate get_batch composition with index lookups"
```

---

### Task 6: Update CLAUDE.md and README roadmap

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md pub API list**

In the Key Conventions section, find the line starting with `- \`pub\` for user-facing API` and add `get_batch` / `get_batch_mut` to the World method notes. These are methods on World (already pub), so no re-export needed — but document them in the convention list.

No change needed to the pub API parenthetical — they're methods, not types. But add a note under the change detection invariant:

Find the bullet starting with `- **Change detection invariant**:` and add `get_batch_mut` to the list of paths that mark columns changed.

**Step 2: Update README roadmap**

Replace the "Query planning (Volcano model)" row:

Old: `| Query planning (Volcano model) | Optimize complex queries across indexes |`

New: `| Batch point-lookups | Archetype-grouped fetch for index-driven access patterns |`

Wait — actually the batch lookups are being implemented now. If they'll be done by the time this ships, remove the row and mention it in the features section or examples table instead. But since this plan is the implementation, update the roadmap row to reflect the next actual stretch goal. Check with the design doc — the remaining roadmap items are rkyv zero-copy snapshots and replication & sync. Just remove the query planning row since it's being implemented.

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md change detection paths, remove completed roadmap item"
```

---

### Task 7: Write ADR

**Files:**
- Create: `docs/adr/012-batch-point-lookups.md`

**Step 1: Write ADR**

```markdown
# ADR-012: Batch Point-Lookups Over Query Planner

**Status:** Accepted
**Date:** 2026-03-06

## Context

With BTreeIndex and HashIndex in the engine, the next question was how to
efficiently fetch components for entities returned by index lookups. The
roadmap listed "Query planning (Volcano model)" — composable operator trees
that would optimise query execution across indexes.

## Decision

Provide batch point-lookups (`World::get_batch`, `World::get_batch_mut`)
instead of a query planner. The user composes index queries with batch
fetches manually.

### Why not a query planner?

- **ECS queries don't join.** The Volcano model solves joining rows across
  tables with unknown cardinalities. In an ECS, an entity either has the
  components or it doesn't — the "join" is the archetype bitset match,
  already O(1) per archetype.
- **The user already knows the access pattern.** ECS systems are compiled
  functions. The developer chooses between `query` and `index.range` at
  write time. A runtime planner adds overhead without new information.
- **Archetype matching is already fast.** Bitset subset check is
  O(archetypes), not O(entities).

### What batch lookups provide

- **Amortised resolution.** ComponentId lookup, sparse check, and column
  index lookup resolve once per type/archetype, not once per entity.
- **Cache locality.** Grouping by archetype means column data access is
  sequential within each group — the prefetcher stays happy.
- **Composability.** Index narrowing followed by batch fetch is a two-step
  pipeline. No operator trees, no cost model, no planner overhead.

## Alternatives

- **Volcano-model query planner** — rejected (see above).
- **`query_entities` with pre-filtered entity set** — deferred. Would
  allow multi-component fetch in one call, but `get_batch` per component
  is sufficient until profiling shows otherwise.
- **No new API** — rejected. Per-entity `get()` has poor cache locality
  for large candidate sets from index lookups.

## Consequences

- Two new methods on World: `get_batch<T>(&self, &[Entity])` and
  `get_batch_mut<T>(&mut self, &[Entity])`.
- `get_batch_mut` panics on duplicate entities (aliased `&mut T` is UB).
- Indexes remain external — no coupling between query engine and index types.
- The user is the planner. The type system is the cost model.
```

**Step 2: Commit**

```bash
git add docs/adr/012-batch-point-lookups.md
git commit -m "docs: ADR-012 batch point-lookups over query planner"
```
