# Query Cache with Generation Tracking

## Problem

Every `World::query::<Q>()` call scans **all** archetypes to find matches via `required.is_subset(&arch.component_ids)`. For workloads with many archetypes and frequent queries, this linear scan dominates.

## Design

Transparent internal cache on `World` — no API change for callers (except `&self` → `&mut self` on `query()`).

### Data Structures

```rust
struct QueryCacheEntry {
    /// Archetypes whose component_ids are a superset of the query's required_ids.
    matched_ids: Vec<ArchetypeId>,
    /// Precomputed required component bitset for incremental scans.
    required: FixedBitSet,
    /// Number of archetypes when cache was last updated.
    /// New archetypes live at indices [last_archetype_count..].
    last_archetype_count: usize,
}
```

Added to `World`:
```rust
query_cache: HashMap<TypeId, QueryCacheEntry>,
```

### Query Flow

1. Look up `TypeId::of::<Q>()` in `query_cache`.
2. If missing → create entry with `last_archetype_count: 0` and `required: Q::required_ids()`.
3. If `last_archetype_count < archetypes.len()` → scan only `archetypes[last_count..]` for new matches, append to `matched_ids`, update count.
4. Build `Fetch` state from cached archetype list, filtering empty archetypes at iteration time.

### Invalidation

- **Trigger**: New archetype creation (via `spawn`, `insert`, `remove` when migrating to a new component set).
- **Mechanism**: `last_archetype_count < archetypes.len()` triggers incremental scan of only the new archetypes.
- **Empty archetypes**: Filtered at iteration time, not at cache time. No invalidation needed for spawn/despawn within existing archetypes.

### Signature Change

`World::query()` changes from `&self` to `&mut self`. This matches `query_table()`/`query_table_mut()` and is required for cache mutation. Callers cannot hold two query iterators simultaneously (already the practical reality due to borrow rules).

### What Is NOT Cached

Column pointers (`Fetch` state) are NOT cached. BlobVec can reallocate on any spawn/despawn, invalidating pointers. Only the archetype ID list is safe to cache across structural changes.

### Edge Cases

- **First query**: `last_archetype_count` starts at 0, so the full scan runs once.
- **Unrelated archetype creation**: Incremental scan checks only the new archetype, finds no match, cached list unchanged.
- **Duplicate query types**: `(&Pos, &Vel)` and `(&Vel, &Pos)` are separate TypeIds with separate cache entries. Same matches, duplicated entries. Acceptable — WorldQuery impls are separate types.
- **Archetype-only growth**: Archetypes are append-only (never removed), so the incremental scan using index ranges is always valid.

### Testing

1. **Cache hit**: Query twice with no structural changes — verify second call skips scan.
2. **Incremental update**: Query, spawn new archetype, query again — verify new archetype found.
3. **Empty archetype filtering**: Despawn all entities from an archetype — verify query skips it.
4. **Independent cache per query type**: Two different query types maintain separate caches.
5. **Generation debug assertion**: Assert generation consistency with archetype count.

### Files Modified

- `crates/minkowski/src/world.rs` — Add `query_cache` field, modify `query()` to use cache.
- `crates/minkowski/src/query/` — No changes needed (QueryIter, WorldQuery unchanged).
- `crates/minkowski/src/storage/archetype.rs` — No changes (generation already exists).
- Tests inline in `world.rs` and `query/iter.rs` — Update `&self` → `&mut self` where needed.
