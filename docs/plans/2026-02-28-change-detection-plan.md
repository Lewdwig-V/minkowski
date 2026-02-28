# Change Detection Ticks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-column tick tracking so queries with `Changed<T>` can skip entire archetypes whose data hasn't been mutably accessed since the query's last read.

**Architecture:** A `Tick` newtype (u32, wrapping comparison) lives on `World` (global counter) and each `BlobVec` column (changed_tick). Every mutable access path sets `changed_tick = current_tick`. `Changed<T>` implements `WorldQuery` as a filter via a new `matches_filters` trait method. `QueryCacheEntry` stores `last_read_tick` to know what's "new" per query type.

**Tech Stack:** Rust, `std::marker::PhantomData`

---

### Task 1: Create `Tick` newtype

**Files:**
- Create: `crates/minkowski/src/tick.rs`
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Create the module**

Create `crates/minkowski/src/tick.rs`:

```rust
/// Monotonic tick counter with wrapping-aware comparison.
///
/// Used for change detection: each BlobVec column stores the tick at which
/// it was last mutably accessed. Queries compare column ticks against their
/// last-read tick to skip unchanged archetypes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Tick(u32);

impl Tick {
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Wrapping-aware comparison. Returns true if `self` is more recent than `other`.
    ///
    /// Handles overflow: treats any tick within `u32::MAX / 2` distance as "recent".
    /// At 60fps this gives ~2.3 years before wraparound.
    pub fn is_newer_than(self, other: Tick) -> bool {
        self.0.wrapping_sub(other.0) < u32::MAX / 2
    }

    /// Advance the tick by one (wrapping).
    pub fn advance(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero() {
        assert_eq!(Tick::default(), Tick::new(0));
    }

    #[test]
    fn advance_increments() {
        let mut t = Tick::new(0);
        t.advance();
        assert_eq!(t, Tick::new(1));
    }

    #[test]
    fn newer_than_basic() {
        let a = Tick::new(5);
        let b = Tick::new(3);
        assert!(a.is_newer_than(b));
        assert!(!b.is_newer_than(a));
    }

    #[test]
    fn newer_than_equal_is_false() {
        let a = Tick::new(5);
        assert!(!a.is_newer_than(a));
    }

    #[test]
    fn newer_than_wrapping() {
        // Simulate wraparound: current tick just wrapped past 0
        let old = Tick::new(u32::MAX - 5);
        let new = Tick::new(3); // wrapped: 3 is "newer" than MAX-5
        assert!(new.is_newer_than(old));
        assert!(!old.is_newer_than(new));
    }

    #[test]
    fn advance_wraps() {
        let mut t = Tick::new(u32::MAX);
        t.advance();
        assert_eq!(t, Tick::new(0));
    }
}
```

**Step 2: Add module to lib.rs**

In `crates/minkowski/src/lib.rs`, add after `pub mod table;`:

```rust
pub mod tick;
```

And add to the re-exports:

```rust
pub use tick::Tick;
```

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- tick`
Expected: 6 new tests PASS.

**Step 4: Commit**

```bash
git add crates/minkowski/src/tick.rs crates/minkowski/src/lib.rs
git commit -m "feat: add Tick newtype with wrapping-aware comparison

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `changed_tick` to BlobVec

**Files:**
- Modify: `crates/minkowski/src/storage/blob_vec.rs`

**Step 1: Add the field**

Add `use crate::tick::Tick;` to imports at the top of `blob_vec.rs`.

Add `pub(crate) changed_tick: Tick,` to the `BlobVec` struct (after the `capacity` field, line 11).

Initialize it in `BlobVec::new()` — add `changed_tick: Tick::default(),` to the `Self { ... }` block (after `capacity,`).

**Step 2: Add a method to mark the column changed**

Add to `impl BlobVec` (after `alloc_align`):

```rust
    /// Mark this column as changed at the given tick.
    #[inline]
    pub(crate) fn mark_changed(&mut self, tick: Tick) {
        self.changed_tick = tick;
    }
```

**Step 3: Add test**

Add to the test module in `blob_vec.rs`:

```rust
    #[test]
    fn changed_tick_default_and_mark() {
        use crate::tick::Tick;
        let mut bv = bv_for::<u32>();
        assert_eq!(bv.changed_tick, Tick::default());
        bv.mark_changed(Tick::new(42));
        assert_eq!(bv.changed_tick, Tick::new(42));
    }
```

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/minkowski/src/storage/blob_vec.rs
git commit -m "feat: add changed_tick to BlobVec columns

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `current_tick` to World and instrument mutation sites

Add the tick counter to World and set `changed_tick` on every mutable access path.

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Add field and methods**

Add `use crate::tick::Tick;` to imports.

Add `pub(crate) current_tick: Tick,` to the `World` struct.

Initialize in `World::new()`: `current_tick: Tick::default(),`

Add methods to `impl World`:

```rust
    /// Advance the world tick. Call once per frame or system batch.
    pub fn tick(&mut self) {
        self.current_tick.advance();
    }

    /// Get the current world tick.
    pub fn current_tick(&self) -> Tick {
        self.current_tick
    }
```

**Step 2: Instrument `spawn()`**

After the bundle push loop (after line 84 `});`), add:

```rust
        // Mark all columns as changed at current tick
        for col in &mut archetype.columns {
            col.mark_changed(self.current_tick);
        }
```

**Step 3: Instrument `get_mut()`**

Before `Some(&mut *ptr)` (line 169), add the tick update:

```rust
            archetype.columns[col_idx].mark_changed(self.current_tick);
```

Note: you'll need to reorganize slightly — `archetype` is currently an immutable borrow at this point. Change the `let archetype = &mut self.archetypes...` line to use `&mut` and move the `col_idx` lookup accordingly. Actually, looking at the code, line 165 already uses `&mut`:

```rust
let archetype = &mut self.archetypes.archetypes[location.archetype_id.0];
```

So just add `archetype.columns[col_idx].mark_changed(self.current_tick);` before the unsafe block.

**Step 4: Instrument `insert()` overwrite case**

After the `std::ptr::write(ptr, component);` line (line 280), add:

```rust
            // Mark column changed (need mutable access to the archetype)
            let src_arch = &mut self.archetypes.archetypes[location.archetype_id.0];
            src_arch.columns[col_idx].mark_changed(self.current_tick);
```

Wait — there's a borrow issue here. The `src_arch` at line 274 is an immutable borrow, and we're inside an unsafe block at line 277. Let me restructure: after the return, this case returns early, so we can mark before or inside the unsafe block. The cleanest way: after `std::ptr::write(ptr, component);` and before `return;`, mark the column. But we need `&mut` access to the archetype. Since this is inside an `unsafe` block with a raw pointer, we can get `&mut` to the column via the archetype:

Actually, looking more carefully, the overwrite path reads `src_arch` as `&self.archetypes...` (immutable). We need to restructure this to get mutable access for marking. The simplest fix:

```rust
        // If entity already has this component, overwrite in-place
        {
            let src_arch = &mut self.archetypes.archetypes[location.archetype_id.0];
            if src_arch.component_ids.contains(comp_id) {
                let col_idx = src_arch.component_index[&comp_id];
                unsafe {
                    let ptr = src_arch.columns[col_idx].get_ptr(location.row) as *mut T;
                    std::ptr::drop_in_place(ptr);
                    std::ptr::write(ptr, component);
                }
                src_arch.columns[col_idx].mark_changed(self.current_tick);
                return;
            }
        }
```

**Step 5: Instrument `query()` — set tick on mutable columns before init_fetch**

This is the key change in `World::query()`. Before calling `init_fetch`, we need to set `changed_tick` on columns that will be mutably accessed. But `World::query()` doesn't know which columns are mutable — that's encoded in the query type `Q`.

The design says: "Option 2 — set tick on the column inside World::query() before calling init_fetch". To do this, we need a way for the query type to report which component IDs it mutably accesses. Add a new method to WorldQuery:

```rust
/// Returns ComponentIds that this query accesses mutably.
/// Used by change detection to mark columns as changed.
/// Default: empty (no mutable access).
fn mutable_ids(_registry: &ComponentRegistry) -> FixedBitSet {
    FixedBitSet::new()
}
```

`&mut T` returns the same as `required_ids`. `&T`, `Entity`, `Option<&T>` return empty. `Changed<T>` returns empty. Tuples union their elements' mutable_ids.

Then in `World::query()`, after the filter check, before `init_fetch`:

```rust
// Mark mutable columns as changed at current tick
let mutable = Q::mutable_ids(&self.components);
for comp_id in mutable.ones() {
    if let Some(&col_idx) = arch.component_index.get(&comp_id) {
        // Safety: we have &mut self, so no aliasing
        let arch_mut = &mut self.archetypes.archetypes[aid.0];
        arch_mut.columns[col_idx].mark_changed(self.current_tick);
    }
}
```

Wait — there's a borrow conflict. `entry` borrows `self.query_cache`, and now we need `&mut self.archetypes`. This requires restructuring the query method to avoid holding the cache entry borrow while mutating archetypes. The cleanest approach: collect the matched_ids into a local Vec (we already clone via iter), then drop the cache borrow, then iterate the local vec to build fetches.

Actually, looking at the code again — `entry.matched_ids` is iterated by reference, and `self.archetypes` is accessed immutably via `&self.archetypes.archetypes[aid.0]`. The conflict is that `entry` borrows `self.query_cache` (via `self`), and we can't mutably borrow `self.archetypes` simultaneously.

**Solution**: Clone the matched_ids and last_read_tick out of the cache entry, drop the borrow, then iterate:

```rust
let matched_ids = entry.matched_ids.clone();
let last_read_tick = entry.last_read_tick;
// Drop the mutable borrow on query_cache

let mutable = Q::mutable_ids(&self.components);

let fetches: Vec<_> = matched_ids.iter()
    .filter_map(|&aid| {
        let arch = &self.archetypes.archetypes[aid.0];
        if arch.is_empty() { return None; }
        if !Q::matches_filters(arch, &self.components, last_read_tick) {
            return None;
        }
        // Mark mutable columns changed
        for comp_id in mutable.ones() {
            if let Some(&col_idx) = arch.component_index.get(&comp_id) {
                let arch_mut = &mut self.archetypes.archetypes[aid.0];
                arch_mut.columns[col_idx].mark_changed(self.current_tick);
            }
        }
        let arch = &self.archetypes.archetypes[aid.0];
        Some((Q::init_fetch(arch, &self.components), arch.len()))
    })
    .collect();

// Update last_read_tick
if let Some(entry) = self.query_cache.get_mut(&type_id) {
    entry.last_read_tick = self.current_tick;
}
```

This is getting complex. Let the implementer work out the exact borrow restructuring — the key constraints are:
1. Clone matched_ids and last_read_tick from the cache entry before building fetches
2. Mark mutable columns via `mutable_ids()` before `init_fetch`
3. Update `last_read_tick` after fetches are built

**Step 6: Write tests**

Add to world.rs test module:

```rust
    #[test]
    fn spawn_marks_column_ticks() {
        use crate::tick::Tick;
        let mut world = World::new();
        world.tick(); // tick to 1
        world.spawn((Pos { x: 1.0, y: 0.0 },));

        // The Pos column in the archetype should have changed_tick == 1
        let arch = &world.archetypes.archetypes[0];
        for col in &arch.columns {
            assert_eq!(col.changed_tick, Tick::new(1));
        }
    }

    #[test]
    fn get_mut_marks_column_tick() {
        use crate::tick::Tick;
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.tick(); // tick to 1
        let _ = world.get_mut::<Pos>(e);

        let loc = world.entity_locations[e.index() as usize].unwrap();
        let arch = &world.archetypes.archetypes[loc.archetype_id.0];
        let comp_id = world.components.id::<Pos>().unwrap();
        let col_idx = arch.component_index[&comp_id];
        assert_eq!(arch.columns[col_idx].changed_tick, Tick::new(1));
    }
```

**Step 7: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 8: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: add current_tick to World and mark columns on mutable access

Instruments spawn, get_mut, insert, and query &mut T paths.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add `matches_filters` and `mutable_ids` to WorldQuery trait + implement `Changed<T>`

**Files:**
- Modify: `crates/minkowski/src/query/fetch.rs`

**Step 1: Add `mutable_ids` and `matches_filters` to the trait**

Add to `pub unsafe trait WorldQuery` (after `as_slice`):

```rust
    /// Returns ComponentIds that this query accesses mutably.
    /// Used by change detection to mark columns as changed before iteration.
    /// Default: empty (no mutable access).
    fn mutable_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new()
    }

    /// Archetype-level filter. Returns false to skip this archetype entirely.
    /// Used by Changed<T> to skip archetypes whose column tick is stale.
    /// Default: true (no filtering).
    fn matches_filters(
        _archetype: &Archetype,
        _registry: &ComponentRegistry,
        _last_read_tick: crate::tick::Tick,
    ) -> bool {
        true
    }
```

**Step 2: Implement `mutable_ids` for `&mut T`**

In the `&mut T` impl, override `mutable_ids`:

```rust
    fn mutable_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry) // same components, but mutable
    }
```

All other existing impls (`&T`, `Entity`, `Option<&T>`) use the defaults (empty mutable_ids, true matches_filters).

**Step 3: Update the tuple macro**

Add to the tuple macro body:

```rust
            fn mutable_ids(registry: &ComponentRegistry) -> FixedBitSet {
                let mut bits = FixedBitSet::new();
                $(
                    let sub = $name::mutable_ids(registry);
                    bits.grow(sub.len());
                    bits.union_with(&sub);
                )*
                bits
            }

            fn matches_filters(
                archetype: &Archetype,
                registry: &ComponentRegistry,
                last_read_tick: crate::tick::Tick,
            ) -> bool {
                $($name::matches_filters(archetype, registry, last_read_tick))&&*
            }
```

**Step 4: Implement `Changed<T>`**

Add after the `Option<&T>` impl, before the tuple macro:

```rust
// --- Changed<T> ---
use crate::tick::Tick;
use std::marker::PhantomData as PhantomChanged; // avoid name conflict if needed

/// Query filter that skips archetypes where component T hasn't changed
/// since the query's last read tick.
pub struct Changed<T: Component>(std::marker::PhantomData<T>);

unsafe impl<T: Component> WorldQuery for Changed<T> {
    type Item<'w> = ();
    type Fetch<'w> = ();
    type Slice<'w> = ();

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry)
    }

    fn init_fetch(_archetype: &Archetype, _registry: &ComponentRegistry) -> () {}

    unsafe fn fetch<'w>(_fetch: &(), _row: usize) -> () {}

    unsafe fn as_slice<'w>(_fetch: &(), _len: usize) -> () {}

    fn matches_filters(
        archetype: &Archetype,
        registry: &ComponentRegistry,
        last_read_tick: Tick,
    ) -> bool {
        if let Some(&comp_id) = registry.id::<T>().as_ref() {
            if let Some(&col_idx) = archetype.component_index.get(comp_id) {
                return archetype.columns[col_idx].changed_tick.is_newer_than(last_read_tick);
            }
        }
        false // component not found → skip
    }
}
```

**Step 5: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass (Changed<T> compiles but isn't used in queries yet).

**Step 6: Commit**

```bash
git add crates/minkowski/src/query/fetch.rs
git commit -m "feat: add matches_filters, mutable_ids to WorldQuery + Changed<T> filter

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wire `matches_filters` and `last_read_tick` into `World::query()`

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Add `last_read_tick` to `QueryCacheEntry`**

Add `last_read_tick: crate::tick::Tick,` to the `QueryCacheEntry` struct.

Initialize it in the `or_insert_with` closure in `query()`:

```rust
last_read_tick: Tick::default(),
```

**Step 2: Restructure `query()` to use filters and mark mutable columns**

The key change: clone `matched_ids` and `last_read_tick` from the cache entry, drop the borrow, then build fetches with filter checks and column tick marking, then update `last_read_tick`.

Replace the fetch-building section of `query()` (the part after the incremental scan, currently lines 205-218) with logic that:
1. Clones `matched_ids` and `last_read_tick` from the entry
2. Computes `mutable_ids` once
3. Iterates matched archetypes, checks `matches_filters`, marks mutable columns, builds fetches
4. Updates `last_read_tick` on the cache entry

The implementer should restructure the borrow flow to avoid holding `&mut query_cache` while accessing `&mut archetypes`. The approach: extract matched_ids + last_read_tick, drop the cache borrow, do the work, then re-borrow cache to update last_read_tick.

**Step 3: Re-export Changed from lib.rs**

Add to `crates/minkowski/src/lib.rs`:

```rust
pub use query::fetch::Changed;
```

**Step 4: Write `Changed<T>` integration tests**

Add to world.rs test module:

```rust
    use crate::query::fetch::Changed;

    #[test]
    fn changed_filter_skips_stale_archetype() {
        use crate::tick::Tick;
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.tick();

        // First query with Changed<Pos> — should find the entity (spawned this tick)
        // Actually, spawn was at tick 0, we're now at tick 1.
        // The first Changed<Pos> query's last_read_tick is Tick(0) (default).
        // Column changed_tick is Tick(0) (set during spawn at tick 0).
        // Tick(0).is_newer_than(Tick(0)) is false, so it would be skipped!
        // We need to ensure spawn happens AFTER a tick, or adjust.

        // Better approach: tick first, then spawn, then query
        let mut world = World::new();
        world.tick(); // tick 1
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        // Column changed_tick = 1, query last_read_tick starts at 0
        // Tick(1).is_newer_than(Tick(0)) = true → found
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 1);

        // Query again without changes — last_read_tick updated to 1
        // Column still at 1, last_read = 1 → Tick(1).is_newer_than(Tick(1)) = false → skip
        let count = world.query::<(Changed<Pos>,)>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn changed_filter_detects_get_mut() {
        let mut world = World::new();
        world.tick();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));

        // Consume the initial change
        let _ = world.query::<(Changed<Pos>,)>().count();

        // No changes — should skip
        world.tick();
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 0);

        // Mutate via get_mut
        world.tick();
        let _ = world.get_mut::<Pos>(e);
        assert_eq!(world.query::<(Changed<Pos>,)>().count(), 1);
    }

    #[test]
    fn changed_filter_mixed_query() {
        let mut world = World::new();
        world.tick();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 1.0, dy: 0.0 }));

        // Consume initial change
        let _ = world.query::<(&Pos, Changed<Vel>)>().count();

        // Mutate only Pos (via get_mut), not Vel
        world.tick();
        let _ = world.get_mut::<Pos>(e);

        // Changed<Vel> should skip — Vel column not touched
        assert_eq!(world.query::<(&Pos, Changed<Vel>)>().count(), 0);

        // But Changed<Pos> should find it
        assert_eq!(world.query::<(&Pos, Changed<Pos>)>().count(), 1);
    }
```

**Step 5: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 6: Commit**

```bash
git add crates/minkowski/src/world.rs crates/minkowski/src/lib.rs
git commit -m "feat: wire Changed<T> filter into query pipeline with last_read_tick

World::query() now checks matches_filters against QueryCacheEntry's
last_read_tick, and marks mutable columns via mutable_ids before
init_fetch. Re-export Changed from crate root.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Instrument changeset apply path

**Files:**
- Modify: `crates/minkowski/src/changeset.rs`

**Step 1: Mark columns in changeset_insert_raw**

In `changeset_insert_raw`, after the overwrite `copy_nonoverlapping` (line ~308) and after the migration `push` of the new column (line ~346), add:

```rust
// Mark the written column as changed
let col = &mut archetype.columns[col_idx];
col.mark_changed(world.current_tick);
```

Also in the spawn mutation handler in `EnumChangeSet::apply()`, after pushing all columns, mark them changed.

The implementer should find each `BlobVec::push` or `copy_nonoverlapping` in the changeset apply paths and add `mark_changed` calls.

**Step 2: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat: mark columns changed in changeset apply paths

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final verification

**Step 1: Full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass including doc tests.

**Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 3: Miri**

Run: `MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib`
Expected: All tests pass.

**Step 4: Boids example**

Run: `cargo run -p minkowski --example boids --release 2>&1 | tail -5`
Expected: Completes successfully (boids doesn't use Changed<T>, but all engine changes must not break it).

**Step 5: Commit if fixes needed**

If all clean, no commit needed.
