# Query Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add transparent query caching to `World::query()` so archetype matching is skipped when no new archetypes have been created since the last query.

**Architecture:** A `HashMap<TypeId, QueryCacheEntry>` on `World` stores matched archetype IDs per query type. On each `query()` call, if the archetype count hasn't grown, the cached list is used directly. If new archetypes exist, only the new ones are scanned (incremental). Empty archetypes are filtered at iteration time.

**Tech Stack:** Rust, `fixedbitset`, `std::any::TypeId`

---

### Task 1: Change `query()` signature from `&self` to `&mut self`

The cache requires mutation on `World`. This task isolates the breaking signature change so all existing tests pass before adding cache logic.

**Files:**
- Modify: `crates/minkowski/src/world.rs:154` (query method signature)
- Modify: `crates/minkowski/src/query/iter.rs:126` (iterate_empty test)

**Step 1: Change the query signature**

In `crates/minkowski/src/world.rs`, change line 154:

```rust
// Before:
pub fn query<Q: WorldQuery>(&self) -> QueryIter<'_, Q> {

// After:
pub fn query<Q: WorldQuery>(&mut self) -> QueryIter<'_, Q> {
```

No other changes to the method body.

**Step 2: Fix the `iterate_empty` test**

In `crates/minkowski/src/query/iter.rs`, line 127:

```rust
// Before:
let world = World::new();

// After:
let mut world = World::new();
```

This is the only callsite that uses an immutable `World` with `query()`. All other tests, benchmarks, and the boids example already use `let mut world`.

**Step 3: Run tests to verify no regressions**

Run: `cargo test -p minkowski --lib`
Expected: All 80 tests pass. The `&self` → `&mut self` change is source-compatible for all existing callsites (Rust allows calling `&mut self` methods on `mut` bindings).

**Step 4: Commit**

```bash
git add crates/minkowski/src/world.rs crates/minkowski/src/query/iter.rs
git commit -m "refactor: change World::query() from &self to &mut self

Prepares for transparent query caching which requires mutation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `QueryCacheEntry` struct and `query_cache` field

Pure data structure addition — no behavior change yet.

**Files:**
- Modify: `crates/minkowski/src/world.rs` (add struct, add field to World, initialize in new())

**Step 1: Add the struct and field**

At the top of `crates/minkowski/src/world.rs`, add `use std::any::TypeId;` to imports.

Add `QueryCacheEntry` struct after the `EntityLocation` struct (around line 25):

```rust
pub(crate) struct QueryCacheEntry {
    /// Archetypes whose component_ids are a superset of the query's required_ids.
    matched_ids: Vec<ArchetypeId>,
    /// Precomputed required component bitset for incremental scans.
    required: FixedBitSet,
    /// Number of archetypes when cache was last updated.
    last_archetype_count: usize,
}
```

Add `use fixedbitset::FixedBitSet;` to imports.
Add `use std::collections::HashMap;` to imports (if not already present).

Add field to `World`:
```rust
pub(crate) query_cache: HashMap<TypeId, QueryCacheEntry>,
```

Initialize in `World::new()`:
```rust
query_cache: HashMap::new(),
```

**Step 2: Run tests to verify compilation**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (struct exists but is unused — may get a dead_code warning, which is fine during development).

**Step 3: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add QueryCacheEntry struct and query_cache field on World

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Write tests for query cache correctness

These tests verify that `query()` returns correct results through the cache paths: first population, incremental update, empty archetype filtering, and independent caching per query type.

**Files:**
- Modify: `crates/minkowski/src/world.rs` (add tests to existing `mod tests`)

**Step 1: Write the tests**

Add these tests inside the existing `#[cfg(test)] mod tests` in `crates/minkowski/src/world.rs`, after the last existing test (`migration_preserves_other_entities`):

```rust
    #[test]
    fn query_cache_populated_on_first_call() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // First query populates cache; second uses it
        let count1 = world.query::<&Pos>().count();
        let count2 = world.query::<&Pos>().count();
        assert_eq!(count1, 1);
        assert_eq!(count2, 1);
    }

    #[test]
    fn query_cache_incremental_after_new_archetype() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);

        // Spawn into a NEW archetype (Pos + Vel)
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));
        // Cache must pick up the new archetype
        assert_eq!(world.query::<&Pos>().count(), 2);
    }

    #[test]
    fn query_cache_filters_empty_archetypes() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);

        world.despawn(e);
        // Archetype still exists but is empty — should not yield results
        assert_eq!(world.query::<&Pos>().count(), 0);
    }

    #[test]
    fn query_cache_independent_per_query_type() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));

        // &Pos matches both archetypes
        assert_eq!(world.query::<&Pos>().count(), 2);
        // (&Pos, &Vel) matches only the second
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 1);
    }

    #[test]
    fn query_cache_unrelated_archetype_no_false_match() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Vel>().count(), 0);

        // Spawn a Vel-only entity (new archetype, unrelated to Pos query)
        world.spawn((Vel { dx: 1.0, dy: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1); // unchanged
        assert_eq!(world.query::<&Vel>().count(), 1); // found it
    }

    #[test]
    fn query_cache_after_migration() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(world.query::<&Pos>().count(), 1);
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 0);

        // Migration creates new archetype (Pos, Vel)
        world.insert(e, Vel { dx: 1.0, dy: 0.0 });
        assert_eq!(world.query::<&Pos>().count(), 1); // still 1 entity
        assert_eq!(world.query::<(&Pos, &Vel)>().count(), 1); // now found
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- query_cache`
Expected: All 6 new tests pass (they test correctness, which already works — the cache is an optimization).

**Step 3: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "test: add query cache correctness tests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Implement the cache logic in `query()`

Wire up `QueryCacheEntry` into the `query()` method. This is the core change.

**Files:**
- Modify: `crates/minkowski/src/world.rs:154-167` (replace query method body)

**Step 1: Replace the query method body**

Replace the entire `query` method (lines 154-167) with:

```rust
    pub fn query<Q: WorldQuery>(&mut self) -> QueryIter<'_, Q> {
        let type_id = TypeId::of::<Q>();
        let total = self.archetypes.archetypes.len();

        let entry = self.query_cache.entry(type_id).or_insert_with(|| {
            QueryCacheEntry {
                matched_ids: Vec::new(),
                required: Q::required_ids(&self.components),
                last_archetype_count: 0,
            }
        });

        // Incremental scan: only check archetypes added since last cache update
        if entry.last_archetype_count < total {
            for arch in &self.archetypes.archetypes[entry.last_archetype_count..total] {
                if entry.required.is_subset(&arch.component_ids) {
                    entry.matched_ids.push(arch.id);
                }
            }
            entry.last_archetype_count = total;
        }

        // Build fetches from cached archetype list, filtering empties at iteration time
        let fetches: Vec<_> = entry
            .matched_ids
            .iter()
            .filter_map(|&aid| {
                let arch = &self.archetypes.archetypes[aid.0];
                if arch.is_empty() {
                    return None;
                }
                Some((Q::init_fetch(arch, &self.components), arch.len()))
            })
            .collect();

        QueryIter::new(fetches)
    }
```

**Step 2: Run the query cache tests**

Run: `cargo test -p minkowski --lib -- query_cache`
Expected: All 6 query cache tests pass.

**Step 3: Run the full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (82+ tests — the 80 originals plus the 6 new ones).

**Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: implement transparent query cache with incremental archetype scan

World::query() now caches matched archetype IDs per query type.
On repeat calls with no new archetypes, the archetype scan is skipped entirely.
When new archetypes are created, only the new ones are scanned.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final verification

Run the complete test suite, clippy, and Miri to verify correctness.

**Step 1: Full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass (including doc tests).

**Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 3: Miri (optional — may take a while)**

Run: `MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib`
Expected: All tests pass under Miri.

**Step 4: Run boids example to verify no regression**

Run: `cargo run -p minkowski --example boids --release 2>&1 | tail -5`
Expected: Completes successfully with frame stats.

**Step 5: Commit if any final fixes were needed**

If all clean, no commit needed — Task 4 commit is the final one.
