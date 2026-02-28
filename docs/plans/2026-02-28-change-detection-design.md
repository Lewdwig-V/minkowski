# Change Detection Ticks

## Problem

Every query iterates all matched entities regardless of whether their data actually changed. For systems that only need to react to changes (e.g., re-render only modified transforms, sync only dirty state), this wastes work proportional to the total entity count rather than the changed count.

## Design

Per-column tick tracking with archetype-level skip. A global tick counter on `World` advances once per frame. Each BlobVec column stores a `changed_tick` — the tick at which it was last mutably accessed. Queries with `Changed<T>` filters skip entire archetypes whose column tick is older than the query's last-read tick.

### Tick Newtype

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Tick(u32);

impl Tick {
    pub fn new(value: u32) -> Self { Self(value) }

    /// Wrapping-aware comparison. Returns true if self is more recent than other.
    /// Handles overflow: treats any tick within u32::MAX/2 distance as "recent".
    pub fn is_newer_than(self, other: Tick) -> bool {
        self.0.wrapping_sub(other.0) < u32::MAX / 2
    }

    pub fn advance(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }
}
```

The comparison method is the single source of truth for tick ordering. Raw `>` on tick values is never used — the newtype prevents it.

### World Tick Counter

```rust
pub struct World {
    // ...existing fields...
    current_tick: Tick,
}

impl World {
    /// Advance the world tick. Call once per frame or per system batch.
    pub fn tick(&mut self) { self.current_tick.advance(); }

    pub fn current_tick(&self) -> Tick { self.current_tick }
}
```

### BlobVec Column Tick

```rust
pub(crate) struct BlobVec {
    // ...existing fields...
    pub(crate) changed_tick: Tick,
}
```

Initialized to `Tick::default()` (0) on creation. Updated to `world.current_tick` whenever a mutable access path touches the column.

### Mutation Points That Update changed_tick

Marking happens **on mutable access**, not on actual write (pessimistic but zero-cost at the write site):

| Mutation path | Where tick is set |
|---|---|
| `World::spawn()` | After pushing all columns |
| `World::get_mut()` | Before returning `&mut T` |
| `World::query()` for `&mut T` | In `init_fetch` when the column pointer is obtained |
| `for_each_chunk` for `&mut T` | Same as above — `init_fetch` runs once per archetype |
| `World::insert()` overwrite | After writing the new value |
| `World::insert()` migration | On the target archetype's new column |
| `changeset_insert_raw()` | After writing the component |

**Not marked**: `&T` (read-only), `Entity`, `Option<&T>`, despawn (entity is gone — no readers care), remove (component is gone).

### Passing Tick to Mutation Sites

The tick value needs to reach into BlobVec columns. Currently, `World::query()` calls `Q::init_fetch(archetype, registry)`. For `&mut T`, this gets the column pointer — that's where we set `changed_tick`.

Options for threading the tick through:
1. Add `tick: Tick` parameter to `init_fetch` — changes trait signature
2. Set tick on the column inside `World::query()` before calling `init_fetch` — no trait change
3. Store tick on `Archetype` and let `init_fetch` read it — slightly indirect

**Choice: Option 2** — `World::query()` already knows which archetypes match and which columns are mutable. It sets `changed_tick` on the relevant columns before building fetches. No trait signature changes needed.

For `get_mut()` and `insert()`, the tick is set directly — these are `World` methods with full access.

### Changed<T> Query Filter

```rust
pub struct Changed<T: Component>(PhantomData<T>);
```

Implements `WorldQuery` as a filter — produces no data, just archetype-level skip:

```rust
unsafe impl<T: Component> WorldQuery for Changed<T> {
    type Item<'w> = ();
    type Fetch<'w> = ();
    type Slice<'w> = ();
    // required_ids: same as &T (archetype must have the component)
    // init_fetch: no-op (no data to fetch)
    // fetch: returns ()
    // matches_filters: checks column changed_tick
}
```

### WorldQuery Filter Extension

Add a new default method to `WorldQuery`:

```rust
pub unsafe trait WorldQuery {
    // ...existing methods...

    /// Archetype-level filter. Returns false to skip this archetype.
    /// Default: true (no filtering).
    fn matches_filters(
        _archetype: &Archetype,
        _registry: &ComponentRegistry,
        _last_read_tick: Tick,
    ) -> bool {
        true
    }
}
```

`Changed<T>` overrides:
```rust
fn matches_filters(archetype: &Archetype, registry: &ComponentRegistry, last_read_tick: Tick) -> bool {
    let comp_id = registry.id::<T>().unwrap();
    let col_idx = archetype.component_index[&comp_id];
    archetype.columns[col_idx].changed_tick.is_newer_than(last_read_tick)
}
```

Tuple impls: all elements must match (AND logic).

### QueryCacheEntry Last-Read Tick

```rust
pub(crate) struct QueryCacheEntry {
    matched_ids: Vec<ArchetypeId>,
    required: FixedBitSet,
    last_archetype_count: usize,
    last_read_tick: Tick,  // NEW
}
```

In `World::query()`, the fetch-building step adds the filter check:

```rust
let fetches: Vec<_> = entry.matched_ids.iter()
    .filter_map(|&aid| {
        let arch = &self.archetypes.archetypes[aid.0];
        if arch.is_empty() { return None; }
        if !Q::matches_filters(arch, &self.components, entry.last_read_tick) {
            return None;
        }
        // ... set changed_tick on mutable columns ...
        Some((Q::init_fetch(arch, &self.components), arch.len()))
    })
    .collect();

// After building fetches, update last_read_tick
entry.last_read_tick = self.current_tick;
```

### What Changed<T> Does NOT Do

- **Per-entity filtering** — it skips entire archetypes, not individual rows. If one entity in a 1000-entity archetype changes, all 1000 are iterated. Per-row filtering would require per-row tick arrays (future opt-in).
- **Tracking which fields changed** — it's per-column (per component type), not per-field.
- **Added<T>** — a separate filter for "component was added this tick" (spawn or insert). Deferred to later; requires distinguishing "added" from "mutated" ticks.

### Testing

1. **Tick wrapping**: verify `is_newer_than` works across u32::MAX boundary
2. **Column tick updates on spawn**: spawn entity, verify column tick == current_tick
3. **Column tick updates on get_mut**: verify tick updates when get_mut is called
4. **Column tick updates on query &mut T**: verify tick updates on init_fetch
5. **Changed<T> skips stale archetypes**: query with Changed<Pos>, verify archetype skipped when Pos column unchanged
6. **Changed<T> includes fresh archetypes**: modify Pos, verify Changed<Pos> query finds it
7. **last_read_tick advances**: query twice with no changes between, second query should skip
8. **Mixed query**: `(&Pos, Changed<Vel>)` — reads Pos for all, but only in archetypes where Vel changed

### Files

- Create: `crates/minkowski/src/tick.rs` — `Tick` newtype with wrapping comparison
- Modify: `crates/minkowski/src/storage/blob_vec.rs` — add `changed_tick: Tick` field
- Modify: `crates/minkowski/src/query/fetch.rs` — add `matches_filters` to `WorldQuery`, implement for all types + `Changed<T>`
- Modify: `crates/minkowski/src/world.rs` — add `current_tick` field, `tick()` method, set column ticks on mutable access, wire `matches_filters` into query pipeline, add `last_read_tick` to `QueryCacheEntry`
- Modify: `crates/minkowski/src/lib.rs` — add `pub mod tick`, re-export `Tick`, `Changed`
