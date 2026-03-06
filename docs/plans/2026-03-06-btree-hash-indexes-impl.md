# B-Tree and Hash Column Indexes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `BTreeIndex<T>` and `HashIndex<T>` for O(log n) range queries and O(1) exact lookups on component values, with incremental updates via `Changed<T>`.

**Architecture:** Both types are external to World (same composition pattern as SpatialIndex). Each wraps a standard collection (`BTreeMap`/`HashMap`) plus a reverse map for incremental updates. Both implement `SpatialIndex` for lifecycle compatibility. New example exercises both index types.

**Tech Stack:** Rust std collections (`BTreeMap`, `HashMap`), existing `SpatialIndex` trait, `Changed<T>` query filter.

---

### Task 1: BTreeIndex — Tests and Core Implementation

**Files:**
- Modify: `crates/minkowski/src/index.rs`

**Step 1: Write failing tests for BTreeIndex**

Add these tests inside the existing `#[cfg(test)] mod tests` block in `index.rs` (after the existing `stale_entries_detectable_via_is_alive` test at line 127). Also add the `Score` test component near the existing `Pos` struct.

```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[test]
fn btree_rebuild_basic() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));
    let e3 = world.spawn((Score(10),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let tens = idx.get(&Score(10));
    assert_eq!(tens.len(), 2);
    assert!(tens.contains(&e1));
    assert!(tens.contains(&e3));

    let twenties = idx.get(&Score(20));
    assert_eq!(twenties.len(), 1);
    assert!(twenties.contains(&e2));

    assert!(idx.get(&Score(99)).is_empty());
}

#[test]
fn btree_range() {
    let mut world = World::new();
    world.spawn((Score(5),));
    world.spawn((Score(15),));
    world.spawn((Score(25),));
    world.spawn((Score(35),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let in_range: Vec<_> = idx.range(Score(10)..Score(30)).collect();
    assert_eq!(in_range.len(), 2);
    assert_eq!(in_range[0].0, &Score(15));
    assert_eq!(in_range[1].0, &Score(25));
}

#[test]
fn btree_empty() {
    let mut world = World::new();
    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    assert!(idx.get(&Score(1)).is_empty());
    assert_eq!(idx.range(..).count(), 0);
}

#[test]
fn btree_update_incremental() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let _e2 = world.spawn((Score(20),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);
    assert_eq!(idx.get(&Score(10)).len(), 1);

    // Mutate e1's score — triggers Changed<Score>
    *world.get_mut::<Score>(e1).unwrap() = Score(30);

    idx.update(&mut world);

    // Old value bucket should no longer contain e1
    assert!(idx.get(&Score(10)).is_empty());
    // New value bucket should contain e1
    assert_eq!(idx.get(&Score(30)).len(), 1);
    assert!(idx.get(&Score(30)).contains(&e1));
    // e2 unchanged
    assert_eq!(idx.get(&Score(20)).len(), 1);
}

#[test]
fn btree_stale_after_despawn() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    world.despawn(e1);

    // Stale entry still present
    let tens = idx.get(&Score(10));
    assert_eq!(tens.len(), 1);
    // But is_alive catches it
    assert!(!world.is_alive(tens[0]));

    // Rebuild cleans up
    idx.rebuild(&mut world);
    assert!(idx.get(&Score(10)).is_empty());
    assert_eq!(idx.get(&Score(20)).len(), 1);
    assert!(idx.get(&Score(20)).contains(&e2));
}

#[test]
fn btree_multi_archetype() {
    let mut world = World::new();
    // Score in two different archetypes
    let e1 = world.spawn((Score(10), Pos { x: 0.0, y: 0.0 }));
    let e2 = world.spawn((Score(10),)); // different archetype (no Pos)

    let mut idx = BTreeIndex::<Score>::new();
    idx.rebuild(&mut world);

    let tens = idx.get(&Score(10));
    assert_eq!(tens.len(), 2);
    assert!(tens.contains(&e1));
    assert!(tens.contains(&e2));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- btree`
Expected: FAIL — `BTreeIndex` not found.

**Step 3: Implement BTreeIndex**

Add this implementation above the `#[cfg(test)]` block in `index.rs`:

```rust
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::ops::RangeBounds;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::fetch::Changed;

/// Secondary index mapping component values to entities via a B-tree.
///
/// Supports O(log n) range queries and exact lookups. External to [`World`] —
/// call [`rebuild`](BTreeIndex::rebuild) or [`update`](BTreeIndex::update) to
/// sync with world state.
///
/// `update` uses [`Changed<T>`] to incrementally patch only the entries
/// whose component was mutably accessed since the last call. A reverse map
/// tracks each entity's last-indexed value for efficient removal.
///
/// Stale entries from despawned entities are included in results — filter
/// with [`World::is_alive`] at query time. Stale entries are cleaned up
/// on the next [`rebuild`](BTreeIndex::rebuild).
pub struct BTreeIndex<T: Component + Ord + Clone> {
    tree: BTreeMap<T, Vec<Entity>>,
    reverse: HashMap<Entity, T>,
}

impl<T: Component + Ord + Clone> BTreeIndex<T> {
    pub fn new() -> Self {
        Self {
            tree: BTreeMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Exact-match lookup. Returns the entities with this component value,
    /// or an empty slice if none.
    pub fn get(&self, value: &T) -> &[Entity] {
        self.tree.get(value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Range query over the B-tree. Returns `(value, entities)` pairs
    /// in sorted order.
    pub fn range<R: RangeBounds<T>>(
        &self,
        range: R,
    ) -> impl Iterator<Item = (&T, &[Entity])> {
        self.tree.range(range).map(|(k, v)| (k, v.as_slice()))
    }

    /// Remove an entity from its current bucket (if tracked).
    fn remove_entity(&mut self, entity: Entity) {
        if let Some(old_value) = self.reverse.remove(&entity) {
            if let Some(bucket) = self.tree.get_mut(&old_value) {
                bucket.retain(|&e| e != entity);
                if bucket.is_empty() {
                    self.tree.remove(&old_value);
                }
            }
        }
    }

    /// Insert an entity under a value in both the tree and reverse map.
    fn insert_entity(&mut self, entity: Entity, value: T) {
        self.tree
            .entry(value.clone())
            .or_insert_with(Vec::new)
            .push(entity);
        self.reverse.insert(entity, value);
    }
}

impl<T: Component + Ord + Clone> SpatialIndex for BTreeIndex<T> {
    fn rebuild(&mut self, world: &mut World) {
        self.tree.clear();
        self.reverse.clear();
        for (entity, value) in world.query::<(Entity, &T)>() {
            self.insert_entity(entity, value.clone());
        }
    }

    fn update(&mut self, world: &mut World) {
        for (entity, value, _) in world.query::<(Entity, &T, Changed<T>)>() {
            self.remove_entity(entity);
            self.insert_entity(entity, value.clone());
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- btree`
Expected: All 6 btree tests PASS.

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 6: Commit**

```bash
git add crates/minkowski/src/index.rs
git commit -m "feat: BTreeIndex — B-tree column index with incremental Changed<T> updates"
```

---

### Task 2: HashIndex — Tests and Implementation

**Files:**
- Modify: `crates/minkowski/src/index.rs`

**Step 1: Write failing tests for HashIndex**

Add these tests inside the existing `mod tests` block:

```rust
#[test]
fn hash_rebuild_basic() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let e2 = world.spawn((Score(20),));
    let e3 = world.spawn((Score(10),));

    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    let tens = idx.get(&Score(10));
    assert_eq!(tens.len(), 2);
    assert!(tens.contains(&e1));
    assert!(tens.contains(&e3));

    let twenties = idx.get(&Score(20));
    assert_eq!(twenties.len(), 1);
    assert!(twenties.contains(&e2));

    assert!(idx.get(&Score(99)).is_empty());
}

#[test]
fn hash_update_incremental() {
    let mut world = World::new();
    let e1 = world.spawn((Score(10),));
    let _e2 = world.spawn((Score(20),));

    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    *world.get_mut::<Score>(e1).unwrap() = Score(30);
    idx.update(&mut world);

    assert!(idx.get(&Score(10)).is_empty());
    assert_eq!(idx.get(&Score(30)).len(), 1);
    assert!(idx.get(&Score(30)).contains(&e1));
    assert_eq!(idx.get(&Score(20)).len(), 1);
}

#[test]
fn hash_duplicate_values() {
    let mut world = World::new();
    let e1 = world.spawn((Score(42),));
    let e2 = world.spawn((Score(42),));
    let e3 = world.spawn((Score(42),));

    let mut idx = HashIndex::<Score>::new();
    idx.rebuild(&mut world);

    let fortytwos = idx.get(&Score(42));
    assert_eq!(fortytwos.len(), 3);
    assert!(fortytwos.contains(&e1));
    assert!(fortytwos.contains(&e2));
    assert!(fortytwos.contains(&e3));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- hash_`
Expected: FAIL — `HashIndex` not found.

**Step 3: Implement HashIndex**

Add this below the `BTreeIndex` impl in `index.rs`:

```rust
/// Secondary index mapping component values to entities via a hash map.
///
/// Supports O(1) exact lookups. External to [`World`] — call
/// [`rebuild`](HashIndex::rebuild) or [`update`](HashIndex::update) to sync.
///
/// `update` uses [`Changed<T>`] to incrementally patch only changed entries.
/// A reverse map tracks each entity's last-indexed value for efficient removal.
///
/// Stale entries from despawned entities are included in results — filter
/// with [`World::is_alive`] at query time.
pub struct HashIndex<T: Component + Hash + Eq + Clone> {
    map: HashMap<T, Vec<Entity>>,
    reverse: HashMap<Entity, T>,
}

impl<T: Component + Hash + Eq + Clone> HashIndex<T> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Exact-match lookup. Returns the entities with this component value,
    /// or an empty slice if none.
    pub fn get(&self, value: &T) -> &[Entity] {
        self.map.get(value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn remove_entity(&mut self, entity: Entity) {
        if let Some(old_value) = self.reverse.remove(&entity) {
            if let Some(bucket) = self.map.get_mut(&old_value) {
                bucket.retain(|&e| e != entity);
                if bucket.is_empty() {
                    self.map.remove(&old_value);
                }
            }
        }
    }

    fn insert_entity(&mut self, entity: Entity, value: T) {
        self.map
            .entry(value.clone())
            .or_insert_with(Vec::new)
            .push(entity);
        self.reverse.insert(entity, value);
    }
}

impl<T: Component + Hash + Eq + Clone> SpatialIndex for HashIndex<T> {
    fn rebuild(&mut self, world: &mut World) {
        self.map.clear();
        self.reverse.clear();
        for (entity, value) in world.query::<(Entity, &T)>() {
            self.insert_entity(entity, value.clone());
        }
    }

    fn update(&mut self, world: &mut World) {
        for (entity, value, _) in world.query::<(Entity, &T, Changed<T>)>() {
            self.remove_entity(entity);
            self.insert_entity(entity, value.clone());
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- hash_`
Expected: All 3 hash tests PASS.

**Step 5: Run full test suite + clippy**

Run: `cargo test -p minkowski --lib && cargo clippy --workspace --all-targets -- -D warnings`
Expected: 295+ tests pass, clean clippy.

**Step 6: Commit**

```bash
git add crates/minkowski/src/index.rs
git commit -m "feat: HashIndex — hash column index with incremental Changed<T> updates"
```

---

### Task 3: SpatialIndex Trait Satisfaction Test

**Files:**
- Modify: `crates/minkowski/src/index.rs`

**Step 1: Write trait satisfaction test**

Add to `mod tests`:

```rust
#[test]
fn spatial_index_trait_satisfaction() {
    let mut world = World::new();
    world.spawn((Score(10),));
    world.spawn((Score(20),));

    // Both types satisfy SpatialIndex — can be used through trait reference
    let mut indexes: Vec<Box<dyn SpatialIndex>> = vec![
        Box::new(BTreeIndex::<Score>::new()),
        Box::new(HashIndex::<Score>::new()),
    ];

    for idx in &mut indexes {
        idx.rebuild(&mut world);
        idx.update(&mut world);
    }
    // Compiles and runs = trait satisfied
}
```

**Step 2: Run test**

Run: `cargo test -p minkowski --lib -- spatial_index_trait`
Expected: PASS.

**Step 3: Commit**

```bash
git add crates/minkowski/src/index.rs
git commit -m "test: verify BTreeIndex and HashIndex satisfy SpatialIndex trait"
```

---

### Task 4: Public Exports + Rustdocs

**Files:**
- Modify: `crates/minkowski/src/lib.rs:94` (add exports)

**Step 1: Add public exports**

Change line 94 from:
```rust
pub use index::SpatialIndex;
```
to:
```rust
pub use index::{BTreeIndex, HashIndex, SpatialIndex};
```

**Step 2: Verify cargo doc**

Run: `cargo doc -p minkowski --no-deps 2>&1 | grep warning`
Expected: Zero warnings.

**Step 3: Verify examples still compile**

Run: `cargo build -p minkowski-examples`
Expected: Clean build.

**Step 4: Commit**

```bash
git add crates/minkowski/src/lib.rs
git commit -m "feat: export BTreeIndex and HashIndex from crate root"
```

---

### Task 5: Example

**Files:**
- Create: `examples/examples/index.rs`

**Step 1: Write the example**

```rust
//! Column indexes — B-tree range queries and hash exact lookups.
//!
//! Run: cargo run -p minkowski-examples --example index --release
//!
//! Demonstrates BTreeIndex and HashIndex on a Score component:
//! - Build indexes from world state
//! - Range query (scores 50..150)
//! - Exact lookup (score == 42)
//! - Incremental update after mutations
//! - Stale entry detection after despawn

use minkowski::{BTreeIndex, Entity, HashIndex, SpatialIndex, World};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy)]
struct Name(&'static str);

fn main() {
    let mut world = World::new();

    // Spawn entities with scores across different archetypes
    let mut entities = Vec::new();
    for i in 0..200 {
        let e = if i % 2 == 0 {
            world.spawn((Score(i),))
        } else {
            // Different archetype — BTreeIndex must span both
            world.spawn((Score(i), Name("player")))
        };
        entities.push(e);
    }

    println!("Spawned {} entities with Score 0..200", entities.len());
    println!();

    // -- Build indexes --
    let mut btree = BTreeIndex::<Score>::new();
    let mut hash = HashIndex::<Score>::new();
    btree.rebuild(&mut world);
    hash.rebuild(&mut world);

    // -- Range query (B-tree) --
    let range_results: Vec<_> = btree
        .range(Score(50)..Score(55))
        .flat_map(|(_, entities)| entities.iter().copied())
        .collect();
    println!(
        "BTree range [50..55): {} entities",
        range_results.len()
    );
    for e in &range_results {
        let score = world.get::<Score>(*e).unwrap();
        println!("  entity {:?} -> score {}", e, score.0);
    }
    println!();

    // -- Exact lookup (both) --
    let btree_42 = btree.get(&Score(42));
    let hash_42 = hash.get(&Score(42));
    println!("BTree exact Score(42): {} entities", btree_42.len());
    println!("Hash  exact Score(42): {} entities", hash_42.len());
    assert_eq!(btree_42.len(), hash_42.len());
    println!();

    // -- Mutate some scores --
    println!("Mutating scores 0..10 to 1000..1010...");
    for i in 0..10 {
        *world.get_mut::<Score>(entities[i]).unwrap() = Score(1000 + i as u32);
    }

    // Incremental update
    btree.update(&mut world);
    hash.update(&mut world);

    let high_scores: Vec<_> = btree
        .range(Score(1000)..)
        .flat_map(|(_, entities)| entities.iter().copied())
        .collect();
    println!(
        "BTree range [1000..): {} entities (expected 10)",
        high_scores.len()
    );
    assert_eq!(high_scores.len(), 10);

    // Old values should be gone
    assert!(btree.get(&Score(0)).is_empty());
    assert!(hash.get(&Score(0)).is_empty());
    println!("Old value Score(0) cleared from both indexes.");
    println!();

    // -- Despawn and stale detection --
    let victim = entities[50];
    let victim_score = world.get::<Score>(victim).unwrap().0;
    println!("Despawning entity with Score({})...", victim_score);
    world.despawn(victim);

    // Stale entry still in index
    let stale = btree.get(&Score(victim_score));
    assert_eq!(stale.len(), 1);
    assert!(!world.is_alive(stale[0]));
    println!("  Stale entry present, is_alive = false");

    // Rebuild cleans up
    btree.rebuild(&mut world);
    assert!(btree.get(&Score(victim_score)).is_empty());
    println!("  After rebuild: stale entry cleaned up");
    println!();

    println!("Done.");
}
```

**Step 2: Verify example runs**

Run: `cargo run -p minkowski-examples --example index --release`
Expected: Output showing range queries, exact lookups, updates, and stale detection.

**Step 3: Commit**

```bash
git add examples/examples/index.rs
git commit -m "feat: index example — B-tree range queries and hash exact lookups"
```

---

### Task 6: CLAUDE.md + ADR + README Updates

**Files:**
- Modify: `CLAUDE.md` — add example run command, add `BTreeIndex`/`HashIndex` to public API list
- Modify: `README.md` — add index example to examples table, update features
- Create: `docs/adr/011-btree-hash-column-indexes.md`

**Step 1: Update CLAUDE.md**

Add the example run command alongside the existing examples:
```
cargo run -p minkowski-examples --example index --release   # B-tree range queries + hash exact lookups
```

Add `BTreeIndex` and `HashIndex` to the `pub` items list in the Key Conventions section.

**Step 2: Update README.md**

Add a row to the examples table:
```
| `index` | BTreeIndex range queries, HashIndex exact lookups, incremental update | `cargo run -p minkowski-examples --example index --release` |
```

In the Spatial Indexing section, add a sentence: "For column-value lookups, `BTreeIndex<T>` provides O(log n) range queries and `HashIndex<T>` provides O(1) exact match — both use `Changed<T>` for incremental updates."

**Step 3: Write ADR**

Create `docs/adr/011-btree-hash-column-indexes.md`:

```markdown
# ADR-011: B-Tree and Hash Column Indexes

**Status:** Accepted
**Date:** 2026-03-06

## Context

All queries are full-archetype scans. Database-style workloads (find entities with score > 100, look up entity by name) need secondary indexes on component values for sub-linear access.

## Decision

Two independent index types — `BTreeIndex<T: Ord>` for range queries and `HashIndex<T: Hash + Eq>` for exact lookups. Both are external to World (same composition pattern as `SpatialIndex`), wrap standard collections with a reverse map for incremental updates, and implement the `SpatialIndex` trait for lifecycle compatibility.

**Key insight: index on whole components, not fields — if you need to index a field, make it a component. This follows the ECS convention and keeps the index API simple.**

## Alternatives Considered

- Index on extracted fields via accessor function — more flexible but requires `dyn Fn` and deferred to future enhancement
- Per-archetype B-trees merged at query time — complex merge-join for minimal benefit
- World-registered indexes with auto-update hooks — violates external composition principle
- Shared `ColumnIndex` trait — query shapes too different (range vs exact)

## Consequences

- O(log n) range queries and O(1) exact lookups on component values
- Incremental updates via `Changed<T>` avoid full rescans
- Reverse map costs one HashMap entry per indexed entity
- Stale entries from despawns handled lazily via generational validation, cleaned on rebuild
- Components must implement `Ord` (B-tree) or `Hash + Eq` (hash) plus `Clone`
```

**Step 4: Verify**

Run: `cargo doc -p minkowski --no-deps 2>&1 | grep warning` — zero warnings.
Run: `cargo test -p minkowski --lib` — all tests pass.

**Step 5: Commit**

```bash
git add CLAUDE.md README.md docs/adr/011-btree-hash-column-indexes.md
git commit -m "docs: ADR-011 + CLAUDE.md/README updates for column indexes"
```

---

## Execution Notes

**Parallelization:** Tasks 1-4 are sequential (each builds on the previous). Task 5 (example) is independent of Task 4 (exports) at the code level but needs the exports to compile. Task 6 (docs) is independent of the example.

Recommended execution: Tasks 1-4 sequentially, then Tasks 5-6 can be parallelized.

**Final verification:**
```bash
cargo test -p minkowski --lib
cargo test -p minkowski --doc
cargo doc -p minkowski --no-deps 2>&1 | grep warning
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p minkowski-examples --example index --release
```
