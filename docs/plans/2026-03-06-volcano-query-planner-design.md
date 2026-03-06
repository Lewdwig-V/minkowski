# Batch Point-Lookup Design (was: Volcano Query Planner)

## Problem

Minkowski has secondary indexes (BTreeIndex, HashIndex) that narrow an entity set efficiently — "entities with Health < 20" via `health_index.range(..20)`. But the step after narrowing — fetching components for those entities — is a per-entity point lookup via `world.get(entity)`. Each call does:

1. `is_alive` check (generation comparison)
2. `entity_locations[idx]` lookup (entity → archetype + row)
3. `ComponentId` resolution (TypeId → id, same result every time)
4. Sparse check (is this component sparse?)
5. `archetype.component_index.get(comp_id)` (component → column index)
6. `columns[col_idx].get_ptr(row)` (raw pointer read)

For 1,000 candidate entities from an index lookup, that's 1,000 redundant ComponentId resolutions, 1,000 independent archetype accesses (poor cache locality if entities are spread across archetypes), and 1,000 sparse checks (always the same answer).

A batch version resolves the ComponentId once, groups entities by archetype, and fetches all rows in each archetype together — amortizing the per-entity overhead and improving cache locality.

## Why not a query planner?

The Volcano model (pull-based operator trees) solves a problem ECS queries don't have: joining rows across multiple tables with unknown cardinalities. In an ECS, entities aren't joined. An entity either has the components or it doesn't. The "join" is the archetype bitset match, which is already O(1) per archetype.

A database query planner exists because the cost difference between a full table scan and an index lookup can be 1,000,000×, and the planner chooses at runtime. In an ECS:

- **The user already knows the access pattern.** ECS systems are compiled functions — the developer knows at write time whether they're doing a bulk scan or a targeted lookup. The "planner" is the developer choosing between `query` and `index.range`.
- **Archetype matching is already fast.** Bitset subset check is O(archetypes), not O(entities). For 100 archetypes, it's microseconds.
- **The cost difference is modest.** Index vs scan is maybe 10-100× for typical entity counts, not 1,000,000×. A planner's overhead would consume the savings.

What's genuinely needed is efficient batch point-lookups that compose with external indexes. The user is the planner. The type system is the cost model.

## Current State

### Point lookups (`crates/minkowski/src/world.rs:361-420`)
- `world.get::<T>(entity) -> Option<&T>` — read path, no tick marking
- `world.get_mut::<T>(entity) -> Option<&mut T>` — write path, marks column changed
- `world.get_by_id::<T>(entity, comp_id)` — pub(crate), skips TypeId→ComponentId lookup
- Each call independently: is_alive → location → comp_id → sparse → column → ptr

### Index query results (`crates/minkowski/src/index.rs`)
- `BTreeIndex::range(bounds) -> &[Entity]` — unsorted, may contain stale entries
- `BTreeIndex::range_valid(bounds, world) -> impl Iterator<Item = Entity>` — filters via `world.has::<T>()`
- `HashIndex::get(value) -> &[Entity]` — same pattern
- All return entity IDs that the user then feeds to `world.get()`

### Bulk iteration (`crates/minkowski/src/world.rs:422-505`)
- `world.query::<Q>()` — iterates ALL matching archetypes, dense scan
- No way to say "iterate only these specific entities"

## Proposed Design

### API Surface

Two new methods on World:

```rust
impl World {
    /// Fetch a component for multiple entities, grouped by archetype
    /// for cache locality. Skips dead entities (returns None).
    /// Results are in the same order as the input slice.
    pub fn get_batch<T: Component>(&self, entities: &[Entity]) -> Vec<Option<&T>>;

    /// Mutable batch fetch. Marks accessed columns as changed.
    /// Results are in the same order as the input slice.
    pub fn get_batch_mut<T: Component>(&mut self, entities: &[Entity]) -> Vec<Option<&mut T>>;
}
```

### Internal Architecture

```rust
pub fn get_batch<T: Component>(&self, entities: &[Entity]) -> Vec<Option<&T>> {
    let mut results = vec![None; entities.len()];

    // Resolve ComponentId once (not per entity)
    let comp_id = match self.components.id::<T>() {
        Some(id) => id,
        None => return results, // Component type never registered
    };

    // Fast path: sparse component
    if self.components.is_sparse(comp_id) {
        for (i, &entity) in entities.iter().enumerate() {
            if self.entities.is_alive(entity) {
                results[i] = self.sparse.get::<T>(comp_id, entity);
            }
        }
        return results;
    }

    // Group by archetype for cache locality
    // (archetype_id) → Vec<(result_index, row)>
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

    // Fetch per-archetype: one column lookup per archetype, not per entity
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

`get_batch_mut` follows the same pattern but:
- Calls `self.drain_orphans()` at entry
- Uses `get_ptr_mut(row, tick)` instead of `get_ptr(row)` to mark columns changed
- Each archetype gets a fresh tick via `self.next_tick()`
- Returns `Vec<Option<&mut T>>` with appropriate lifetime

### Data Flow

Typical index-driven access pattern:

```
BTreeIndex::range(..20)         // O(log n + k) — k matching entities
  → &[Entity]
  → world.get_batch::<Pos>()    // O(k) with archetype-grouped locality
  → Vec<Option<&Pos>>           // Same order as input
  → process results
```

### Multi-component batch fetch

For fetching multiple components per entity, the user calls `get_batch` multiple times:

```rust
let entities: Vec<Entity> = health_index
    .range_valid(..Health(20), &world)
    .collect();

let healths = world.get_batch::<Health>(&entities);
let positions = world.get_batch::<Position>(&entities);

for i in 0..entities.len() {
    if let (Some(health), Some(pos)) = (healths[i], positions[i]) {
        // Process entity with both components
    }
}
```

Each `get_batch` call independently groups by archetype. For two components on the same archetypes, the grouping work is redundant. A future optimization could provide `get_batch2::<A, B>()` that groups once and fetches both columns — but this is premature until profiling shows the grouping is a bottleneck. The single-component version is the right starting point.

## Alternatives Considered

### A. Volcano-model query planner

Composable operator tree: IndexScan → Fetch → Filter with `open()/next()/close()` semantics.

**Rejected.** Solves a problem ECS doesn't have. Entity queries don't join across tables. The developer already knows the access pattern at compile time. Planner overhead would consume the savings for typical entity counts.

### B. `query_entities` — query with a pre-filtered entity set

```rust
world.query_entities::<(&Health, &Pos)>(&entities)
```

**Tradeoffs:** More ergonomic for multi-component fetch — one call instead of multiple `get_batch` calls. But requires a new WorldQuery execution path (iterate specific rows within archetypes, not all rows). More complex implementation, and the separate `get_batch` calls are fast enough for the common case.

**Deferred** — could be added later if profiling shows multi-component batch fetch is a bottleneck.

### C. No new API — keep per-entity `get()`

Users compose index + `get()` manually:

```rust
for &entity in health_index.range(..20) {
    if let Some(pos) = world.get::<Pos>(entity) { ... }
}
```

**Tradeoffs:** Zero API surface growth. But poor cache locality for many candidates — each `get()` does an independent archetype access. The batch version is strictly better when the candidate set is > ~10 entities.

**Rejected** — the performance gap is real and the batch API is simple.

## Semantic Review

### 1. Can this be called with the wrong World?

Same as `get()` — entity IDs from a different World would fail `is_alive()` (different generation array) or return wrong data (matching index but wrong archetype). No worse than the existing API. No cross-World check needed beyond what entity generations provide.

### 2. Can Drop observe inconsistent state?

No. Returns `Vec<Option<&T>>` — standard Rust references. No cleanup, no allocated engine resources.

### 3. Can two threads reach this through `&self`?

`get_batch` takes `&self` — safe for concurrent reads (same as `get`). `get_batch_mut` takes `&mut self` — exclusive access guaranteed by the borrow checker. No new concurrency concerns.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

`get_batch` (read-only): if the same entity appears twice, it appears twice in the output — two shared references to the same data, which is safe. `get_batch_mut`: duplicate entities are detected during archetype grouping and cause a panic. Aliased `&mut T` is UB, so this is an unconditional `assert!`, not `debug_assert!`.

### 5. What happens if this is abandoned halfway through?

`get_batch` computes eagerly and returns a Vec. No partial state to abandon. For `get_batch_mut`, column ticks are marked before building the result Vec — abandoning the Vec after construction leaves columns marked changed (pessimistic but safe, same as `query::<&mut T>()` dropped without iteration).

### 6. Can a type bound be violated by a legal generic instantiation?

`T: Component` is the only bound. Same as `get::<T>()`. Component is blanket-impl'd for `'static + Send + Sync`.

### 7. Does the API surface of this handle permit any operation not covered by the Access bitset?

`get_batch` is a read. `get_batch_mut` is a write. Same access profile as `get`/`get_mut`. No Access integration needed — these are direct World methods, not reducer handles.

## Implementation Plan

### Phase 1: `get_batch` (read-only)

1. **`crates/minkowski/src/world.rs`** — Add `get_batch::<T>(&self, &[Entity]) -> Vec<Option<&T>>`
   - Resolve ComponentId once
   - Sparse fast path
   - Group by archetype, fetch per-archetype

2. **Tests in `world.rs`**:
   - `get_batch_basic` — fetch known entities, verify values
   - `get_batch_dead_entity` — dead entity → None
   - `get_batch_missing_component` — entity alive but no component → None
   - `get_batch_empty_input` — empty slice → empty vec
   - `get_batch_unregistered_type` — component never registered → all None
   - `get_batch_sparse` — sparse component works
   - `get_batch_multi_archetype` — entities across archetypes
   - `get_batch_duplicate_entity` — same entity twice → same value twice
   - `get_batch_preserves_order` — output order matches input order

### Phase 2: `get_batch_mut` (mutable)

3. **`crates/minkowski/src/world.rs`** — Add `get_batch_mut::<T>(&mut self, &[Entity]) -> Vec<Option<&mut T>>`
   - Same grouping logic
   - Uses `get_ptr_mut(row, tick)` for change detection
   - Unconditional duplicate check: if same entity appears twice, panic (aliased `&mut T` is UB — this is an `assert!` boundary)

4. **Tests**:
   - `get_batch_mut_basic` — mutate via returned references
   - `get_batch_mut_marks_changed` — verify column tick advances
   - `get_batch_mut_dead_entity` — dead → None, no tick marking
   - `get_batch_mut_sparse` — sparse component works

### Phase 3: Docs + example update

5. **`examples/examples/index.rs`** — Add section demonstrating index → `get_batch` composition

6. **`CLAUDE.md`** — Add `get_batch`, `get_batch_mut` to pub API notes

7. **README.md roadmap** — Update "Query planning (Volcano model)" to reflect actual scope

### Phase 4: ADR

8. **`docs/adr/012-batch-point-lookups.md`** — Document the decision to provide batch lookups instead of a query planner, and why
