# B-Tree and Hash Column Indexes — Design

**Goal:** O(log n) range queries and O(1) exact lookups on component values, external to World, with incremental updates via `Changed<T>`.

**Context:** All current queries are full-archetype scans. For database-style workloads (find entities with score > 100, look up entity by name), we need secondary indexes on component values.

---

## Core Data Structures

### `BTreeIndex<T: Component + Ord + Clone>`

Wraps `BTreeMap<T, Vec<Entity>>` plus a reverse map `HashMap<Entity, T>` for incremental updates.

- `new() -> Self`
- `rebuild(&mut self, world: &mut World)` — full scan of all archetypes containing `T`
- `update(&mut self, world: &mut World)` — incremental via `Changed<T>`, patches only changed entries using the reverse map
- `get(&self, value: &T) -> &[Entity]` — exact match
- `range(&self, range: R) -> impl Iterator<Item = (&T, &[Entity])>` — range query

### `HashIndex<T: Component + Hash + Eq + Clone>`

Wraps `HashMap<T, Vec<Entity>>` plus a reverse map `HashMap<Entity, T>`.

- `new() -> Self`
- `rebuild(&mut self, world: &mut World)` — full scan
- `update(&mut self, world: &mut World)` — incremental via `Changed<T>`
- `get(&self, value: &T) -> &[Entity]` — O(1) exact match

### Trait bounds rationale

- `Ord` (B-tree) / `Hash + Eq` (hash) — required by the underlying collections
- `Clone` — values are copied out of type-erased BlobVec columns during scan
- `Component` (`'static + Send + Sync`) — standard Minkowski requirement

Both implement `SpatialIndex` for lifecycle compatibility.

## Incremental Update Strategy

`update` uses `Changed<T>` to identify archetypes whose `T` column was mutably accessed since the last call.

1. Query `world.query::<(Entity, &T, Changed<T>)>()` — yields entities in changed archetypes only.
2. For each returned entity, look up old value in the reverse map.
3. Remove entity from old value's bucket (if present).
4. Insert entity under the new value in both the main structure and reverse map.

The reverse map (`HashMap<Entity, T>`) costs one extra entry per indexed entity. This is the standard pattern for incremental index maintenance.

**Tick tracking:** each index's `update` call goes through `world.query()` which maintains a per-query-type `last_read_tick` in `QueryCacheEntry`. Since `(Entity, &T, Changed<T>)` is a distinct query type per `T`, each index gets independent tick tracking automatically.

## Stale Entry Handling

Despawned entities leave stale entries in the index. Handled lazily:

- `get` and `range` return raw `&[Entity]` including stale handles
- Caller filters with `world.is_alive(entity)` — O(1) generation check
- `rebuild` clears everything and re-scans, cleaning up stale entries
- `update` does NOT remove despawned entities (it only sees changed archetypes, not despawns) — stale entries are cleaned on next `rebuild`

This matches the `SpatialIndex` pattern used by `SpatialGrid` and `BarnesHutTree`.

## Multi-Archetype Support

A component can exist in multiple archetypes. Both index types scan all matching archetypes during rebuild/update using `world.query::<(Entity, &T)>()` which already handles multi-archetype iteration via bitset matching.

The flat structure (one tree/map across all archetypes) keeps queries simple — no merge-join needed.

## API Location

- **File:** `crates/minkowski/src/index.rs` (extends existing module)
- **Public exports:** `BTreeIndex`, `HashIndex` added to `lib.rs`
- **No World changes** — external composition pattern preserved

## Example

New `examples/examples/index.rs` demonstrating:
- Spawn entities with `Score(u32)` component
- Build `BTreeIndex<Score>` and `HashIndex<Score>`
- Range query and exact lookup
- Mutate scores, call `update`, verify changes reflected
- Despawn entity, show `is_alive` filtering

## Testing

~12 unit tests in `index.rs`:
- Basic rebuild + get for both types
- Range queries (B-tree)
- Duplicate values (multiple entities, same value)
- Incremental update (mutate then update, verify old entries gone)
- Stale entries after despawn
- Multi-archetype indexing
- SpatialIndex trait satisfaction

## Alternatives Considered

- **Index on fields via accessor function** — more flexible but requires `dyn Fn`, deferred to future enhancement
- **Per-archetype B-trees merged at query time** — complex merge-join for minimal benefit (few archetypes share a component)
- **World-registered indexes with auto-update hooks** — violates external composition principle
- **Shared `ColumnIndex` trait** — query shapes are too different (range vs exact), forced abstraction limits both
