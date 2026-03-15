# Join Batch Execution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add archetype-sorted batch execution to join plans, eliminating per-entity `world.get()` overhead via pre-resolved `ThinSlicePtr` column pointers.

**Architecture:** After `run_join()` materializes entities into `ScratchBuffer`, sort by packed `(archetype_id << 32 | row)` key, then iterate in archetype runs calling `init_fetch` once per archetype and `fetch`/`as_slice` per entity. Three new methods: `for_each_batched`, `for_each_batched_raw`, `for_each_join_chunk`.

**Tech Stack:** Rust, `WorldQuery` trait (`init_fetch`/`fetch`/`as_slice`), `FixedBitSet::is_subset`, criterion benchmarks.

**Spec:** `docs/superpowers/specs/2026-03-16-join-batch-execution-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/minkowski/src/planner.rs` | All implementation: `ComponentMismatch` error, `sort_by_archetype` on `ScratchBuffer`, `row_indices: Vec<usize>` on `QueryPlanResult`, three new methods |
| `crates/minkowski-bench/benches/planner.rs` | 14 new benchmarks across 3 groups |
| `crates/minkowski-bench/src/lib.rs` | `FatData` and `Team` component types |
| `ci/miri-subset.txt` | 3 new test entries |
| `ci/run-miri-subset.sh` | 3 new `EXACT_TESTS` entries |
| `docs/perf-roadmap.md` | Updated results after benchmarking |

All changes go into existing files. No new files created.

---

## Chunk 1: Core Implementation

### Task 1: Add `ComponentMismatch` error variant

**Files:**
- Modify: `crates/minkowski/src/planner.rs:124-168` (PlanExecError enum + Display + Error impls)

- [ ] **Step 1: Add the variant to `PlanExecError`**

In `crates/minkowski/src/planner.rs`, add after the `JoinNotSupported` variant:

```rust
/// Batch execution method called with a `Q: WorldQuery` whose required
/// components are not present in one of the matched archetypes.
ComponentMismatch {
    /// `std::any::type_name::<T>()` of the missing component.
    component: &'static str,
    /// Archetype that was missing the component.
    archetype_id: ArchetypeId,
},
```

Update the `Display` impl (inside the `#[allow(deprecated)]` match):

```rust
PlanExecError::ComponentMismatch { component, archetype_id } => write!(
    f,
    "batch query component `{component}` not found in archetype {arch}",
    arch = archetype_id.0
),
```

Update the `Error::source` impl (add arm returning `None`).

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p minkowski`
Expected: PASS (new variant is unused, but `#[allow(deprecated)]` already covers the match)

Note: The `#[allow(deprecated)]` on the match will need to become `#[allow(deprecated)]` on the function or the individual arm. Check the existing pattern — the current code uses `#[allow(deprecated)]` on the `fn fmt` and `fn source` methods, which covers all arms.

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): add ComponentMismatch error variant for batch execution"
```

---

### Task 2: Add `sort_by_archetype` to `ScratchBuffer`

**Files:**
- Modify: `crates/minkowski/src/planner.rs:4180-4246` (ScratchBuffer impl block)

- [ ] **Step 1: Write the test**

Add to the `#[cfg(test)] mod tests` block in `planner.rs` (near the existing `ScratchBuffer` tests if any, otherwise at the end):

```rust
#[test]
fn scratch_sort_by_archetype_groups_entities() {
    let mut world = World::new();
    // Archetype A: Score only
    let a1 = world.spawn((Score(1),));
    let a2 = world.spawn((Score(2),));
    // Archetype B: Score + Team
    let b1 = world.spawn((Score(3), Team(1)));
    let b2 = world.spawn((Score(4), Team(2)));

    // Deliberately interleave: [b1, a1, b2, a2]
    let mut scratch = ScratchBuffer::new(4);
    scratch.push(b1);
    scratch.push(a1);
    scratch.push(b2);
    scratch.push(a2);

    scratch.sort_by_archetype(&world.entity_locations);

    // After sort: entities from same archetype should be contiguous.
    let sorted = scratch.as_slice();
    // First two share one archetype, last two share another.
    let loc0 = world.entity_locations[sorted[0].index() as usize].unwrap();
    let loc1 = world.entity_locations[sorted[1].index() as usize].unwrap();
    let loc2 = world.entity_locations[sorted[2].index() as usize].unwrap();
    let loc3 = world.entity_locations[sorted[3].index() as usize].unwrap();
    assert_eq!(loc0.archetype_id, loc1.archetype_id);
    assert_eq!(loc2.archetype_id, loc3.archetype_id);
    assert_ne!(loc0.archetype_id, loc2.archetype_id);

    // Within each archetype group, rows should be sorted ascending.
    assert!(loc0.row < loc1.row);
    assert!(loc2.row < loc3.row);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski -- scratch_sort_by_archetype_groups_entities`
Expected: FAIL — `sort_by_archetype` method does not exist.

- [ ] **Step 3: Implement `sort_by_archetype`**

Add to the `impl ScratchBuffer` block (after `sorted_intersection`):

```rust
/// Sort entities by (ArchetypeId, Row) to restore cache locality after
/// join materialisation. Entities from the same archetype become
/// contiguous, and within each archetype group, rows are in physical
/// memory order (ascending).
///
/// # Panics
/// Panics if any entity in the buffer has no location (dead entity).
/// This should never happen: join collectors only iterate live archetypes.
fn sort_by_archetype(&mut self, entity_locations: &[Option<EntityLocation>]) {
    self.entities.sort_unstable_by_key(|e| {
        let loc = entity_locations[e.index() as usize]
            .expect("join produced dead entity in scratch buffer");
        ((loc.archetype_id.0 as u64) << 32) | (loc.row as u64)
    });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski -- scratch_sort_by_archetype_groups_entities`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): add sort_by_archetype to ScratchBuffer"
```

---

### Task 3: Add `row_indices` field to `QueryPlanResult`

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1402-1418` (QueryPlanResult struct)
- Modify: `crates/minkowski/src/planner.rs` (wherever QueryPlanResult is constructed — search for `QueryPlanResult {`)

- [ ] **Step 1: Add the field**

Add after `compiled_agg_scan_raw`:

```rust
/// Reusable buffer for row indices in batch execution methods.
/// Cleared and repopulated on each `for_each_join_chunk` call.
row_indices: Vec<usize>,
```

- [ ] **Step 2: Initialize it in the constructor**

Search for `QueryPlanResult {` in the `build()` method (around line 3390-3410). Add `row_indices: Vec::new(),` to the struct literal.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p minkowski`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): add row_indices buffer to QueryPlanResult"
```

---

### Task 4: Implement `for_each_batched`

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (QueryPlanResult impl block, after `for_each_raw`)

- [ ] **Step 1: Write the test**

Add to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn for_each_batched_yields_all_join_results() {
    let mut world = World::new();
    // Score-only entities (should NOT appear in inner join)
    for i in 0..5 {
        world.spawn((Score(i),));
    }
    // Score+Team entities (SHOULD appear)
    let mut expected = Vec::new();
    for i in 5..15 {
        let e = world.spawn((Score(i), Team(i % 3)));
        expected.push((e, Score(i)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut results: Vec<(Entity, Score)> = Vec::new();
    plan.for_each_batched::<(&Score,), _>(&mut world, |entity, (score,)| {
        results.push((entity, *score));
    })
    .unwrap();

    // Sort both by entity for comparison.
    results.sort_by_key(|(e, _)| e.to_bits());
    expected.sort_by_key(|(e, _)| e.to_bits());
    assert_eq!(results, expected);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski -- for_each_batched_yields_all_join_results`
Expected: FAIL — `for_each_batched` method does not exist.

- [ ] **Step 3: Implement `for_each_batched`**

Add to the `impl QueryPlanResult` block, after the `for_each_raw` method:

```rust
/// Execute the plan with archetype-sorted batch extraction.
///
/// After join materialisation (or scan collection), entities are sorted
/// by `(ArchetypeId, Row)` so that consecutive entities share the same
/// archetype. For each archetype run, `Q::init_fetch` is called once
/// to resolve column pointers, then `Q::fetch` is called per entity
/// with just a pointer offset — no generation check, no TypeId hash,
/// no column search.
///
/// `Q` is specified at the call site and validated at runtime against
/// each archetype's component set. Returns `Err(ComponentMismatch)` if
/// `Q`'s required components are missing from any matched archetype.
///
/// Advances the read tick (same as `for_each`).
pub fn for_each_batched<Q, F>(
    &mut self,
    world: &mut World,
    mut callback: F,
) -> Result<(), PlanExecError>
where
    Q: WorldQuery,
    F: FnMut(Entity, Q::Item<'_>),
{
    self.for_each_batched_inner::<Q, F>(world, &mut callback)?;
    self.last_read_tick = world.next_tick();
    Ok(())
}

/// Read-only variant of [`for_each_batched`]. No tick advancement.
/// Safe for use inside transactions where only `&World` is available.
pub fn for_each_batched_raw<Q, F>(
    &mut self,
    world: &World,
    mut callback: F,
) -> Result<(), PlanExecError>
where
    Q: ReadOnlyWorldQuery,
    F: FnMut(Entity, Q::Item<'_>),
{
    self.for_each_batched_inner::<Q, F>(world, &mut callback)
}

/// Shared implementation for `for_each_batched` and `for_each_batched_raw`.
fn for_each_batched_inner<Q, F>(
    &mut self,
    world: &World,
    callback: &mut F,
) -> Result<(), PlanExecError>
where
    Q: WorldQuery,
    F: FnMut(Entity, Q::Item<'_>),
{
    if self.world_id != world.world_id() {
        return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
    }

    // Phase 1: Populate scratch buffer.
    if self.join_exec.is_some() {
        self.run_join(world);
    } else if let Some(compiled) = &mut self.compiled_for_each_raw {
        let scratch = self
            .scratch
            .as_mut()
            .expect("for_each_batched requires a scratch buffer");
        scratch.clear();
        let tick = self.last_read_tick;
        compiled(world, tick, &mut |entity: Entity| {
            scratch.push(entity);
        });
    } else {
        panic!(
            "for_each_batched() called on a plan with no join executor and no compiled scan"
        );
    }

    // Phase 2: Sort by (archetype_id, row).
    let scratch = self
        .scratch
        .as_mut()
        .expect("for_each_batched requires a scratch buffer");
    scratch.sort_by_archetype(&world.entity_locations);

    // Phase 3: Walk archetype runs with pre-resolved fetch.
    let entities = scratch.as_slice();
    if entities.is_empty() {
        return Ok(());
    }

    let required = Q::required_ids(&world.components);
    let mut run_start = 0;

    while run_start < entities.len() {
        let loc = world.entity_locations[entities[run_start].index() as usize]
            .expect("sorted entity has no location");
        let arch_id = loc.archetype_id;
        let archetype = &world.archetypes.archetypes[arch_id.0];

        // Validate Q's required components are present in this archetype.
        if !required.is_subset(&archetype.component_ids) {
            return Err(PlanExecError::ComponentMismatch {
                component: std::any::type_name::<Q>(),
                archetype_id: arch_id,
            });
        }

        let fetch = Q::init_fetch(archetype, &world.components);

        // Find end of this archetype run.
        let mut run_end = run_start + 1;
        while run_end < entities.len() {
            let next_loc = world.entity_locations[entities[run_end].index() as usize]
                .expect("sorted entity has no location");
            if next_loc.archetype_id != arch_id {
                break;
            }
            run_end += 1;
        }

        // Iterate entities in this run.
        for &entity in &entities[run_start..run_end] {
            let row = world.entity_locations[entity.index() as usize]
                .expect("sorted entity has no location")
                .row;
            debug_assert!(row < archetype.len(), "row {row} >= archetype len {}", archetype.len());
            let item = unsafe { Q::fetch(&fetch, row) };
            callback(entity, item);
        }

        run_start = run_end;
    }

    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski -- for_each_batched_yields_all_join_results`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): implement for_each_batched and for_each_batched_raw"
```

---

### Task 5: Implement `for_each_join_chunk`

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (QueryPlanResult impl block, after `for_each_batched_raw`)

- [ ] **Step 1: Write the test**

```rust
#[test]
fn for_each_join_chunk_yields_correct_slices() {
    let mut world = World::new();
    // Archetype A: Score only (will not match inner join)
    for i in 0..3 {
        world.spawn((Score(i),));
    }
    // Archetype B: Score + Team (deterministic scores 10..15)
    for i in 10..15 {
        world.spawn((Score(i), Team(1)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut total_entities = 0;
    let mut chunk_count = 0;
    let mut collected_scores = Vec::new();
    plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
        // rows and entities must have the same length.
        assert_eq!(entities.len(), rows.len());
        // Each row index must be valid for the slice.
        for &row in rows {
            assert!(row < scores.len(), "row {row} out of bounds for slice len {}", scores.len());
            collected_scores.push(scores[row]);
        }
        total_entities += entities.len();
        chunk_count += 1;
    })
    .unwrap();

    assert_eq!(total_entities, 5); // Only Score+Team entities
    assert!(chunk_count >= 1); // At least one archetype chunk
    // Verify we read the correct score values (10..15).
    collected_scores.sort_by_key(|s| s.0);
    assert_eq!(collected_scores, (10..15).map(Score).collect::<Vec<_>>());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski -- for_each_join_chunk_yields_correct_slices`
Expected: FAIL — method does not exist.

- [ ] **Step 3: Implement `for_each_join_chunk`**

Add to the `impl QueryPlanResult` block:

```rust
/// Execute the plan with archetype-chunked slice extraction.
///
/// After join materialisation and archetype sorting, the callback
/// receives per-archetype chunks containing:
/// - `&[Entity]` — matched entities (sorted by row within this archetype)
/// - `&[usize]` — row indices into the column slices
/// - `Q::Slice<'_>` — full column slice for the archetype
///
/// The callback can iterate `rows` and index into the slice:
/// `for (i, &row) in rows.iter().enumerate() { let val = slice[row]; }`
///
/// This enables SIMD-friendly access patterns on join results.
/// Advances the read tick.
pub fn for_each_join_chunk<Q, F>(
    &mut self,
    world: &mut World,
    mut callback: F,
) -> Result<(), PlanExecError>
where
    Q: WorldQuery,
    F: FnMut(&[Entity], &[usize], Q::Slice<'_>),
{
    if self.world_id != world.world_id() {
        return Err(WorldMismatch::new(self.world_id, world.world_id()).into());
    }

    // Phase 1: Populate scratch buffer.
    if self.join_exec.is_some() {
        self.run_join(&*world);
    } else if let Some(compiled) = &mut self.compiled_for_each_raw {
        let scratch = self
            .scratch
            .as_mut()
            .expect("for_each_join_chunk requires a scratch buffer");
        scratch.clear();
        let tick = self.last_read_tick;
        compiled(&*world, tick, &mut |entity: Entity| {
            scratch.push(entity);
        });
    } else {
        panic!(
            "for_each_join_chunk() called on a plan with no join executor and no compiled scan"
        );
    }

    // Phase 2: Sort by (archetype_id, row).
    // Destructure self into disjoint fields to allow simultaneous borrows
    // of scratch (immutable, for entity slice) and row_indices (mutable).
    let scratch = self
        .scratch
        .as_mut()
        .expect("for_each_join_chunk requires a scratch buffer");
    scratch.sort_by_archetype(&world.entity_locations);

    // Early exit if scratch is empty (before destructuring self).
    if scratch.as_slice().is_empty() {
        self.last_read_tick = world.next_tick();
        return Ok(());
    }

    // Reborrow disjoint fields to avoid borrow conflict between
    // scratch.as_slice() (borrows self.scratch) and self.row_indices.
    let Self { scratch, row_indices, last_read_tick, .. } = self;
    let scratch = scratch.as_ref().expect("scratch buffer disappeared");
    let entities = scratch.as_slice();

    let required = Q::required_ids(&world.components);
    let mut run_start = 0;

    while run_start < entities.len() {
        let loc = world.entity_locations[entities[run_start].index() as usize]
            .expect("sorted entity has no location");
        let arch_id = loc.archetype_id;
        let archetype = &world.archetypes.archetypes[arch_id.0];

        // Validate Q's required components.
        if !required.is_subset(&archetype.component_ids) {
            return Err(PlanExecError::ComponentMismatch {
                component: std::any::type_name::<Q>(),
                archetype_id: arch_id,
            });
        }

        let fetch = Q::init_fetch(archetype, &world.components);

        // Find end of this archetype run.
        let mut run_end = run_start + 1;
        while run_end < entities.len() {
            let next_loc = world.entity_locations[entities[run_end].index() as usize]
                .expect("sorted entity has no location");
            if next_loc.archetype_id != arch_id {
                break;
            }
            run_end += 1;
        }

        // Collect row indices for this run.
        row_indices.clear();
        for &entity in &entities[run_start..run_end] {
            let row = world.entity_locations[entity.index() as usize]
                .expect("sorted entity has no location")
                .row;
            debug_assert!(row < archetype.len());
            row_indices.push(row);
        }

        let slice = unsafe { Q::as_slice(&fetch, archetype.len()) };
        callback(&entities[run_start..run_end], row_indices, slice);

        run_start = run_end;
    }

    *last_read_tick = world.next_tick();
    Ok(())
}
```

**Key borrow fix**: The `let Self { scratch, row_indices, last_read_tick, .. } = self;` destructure allows the borrow checker to track `scratch`, `row_indices`, and `last_read_tick` as independent borrows. Without this, `entities` (from `self.scratch.as_slice()`) and `self.row_indices.clear()` would conflict because both go through `self`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski -- for_each_join_chunk_yields_correct_slices`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat(planner): implement for_each_join_chunk with archetype-sorted slices"
```

---

### Task 6: Additional unit tests

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (test module)

- [ ] **Step 1: Write all remaining tests**

```rust
#[test]
fn for_each_batched_raw_no_tick_advance() {
    let mut world = World::new();
    world.spawn((Score(1), Team(1)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // Call raw twice — both should succeed (no tick advancement).
    let mut count1 = 0u32;
    plan.for_each_batched_raw::<(&Score,), _>(&world, |_, _| count1 += 1)
        .unwrap();
    assert_eq!(count1, 1);

    let mut count2 = 0u32;
    plan.for_each_batched_raw::<(&Score,), _>(&world, |_, _| count2 += 1)
        .unwrap();
    assert_eq!(count2, 1);
}

#[test]
fn for_each_join_chunk_works_for_scan_plans() {
    let mut world = World::new();
    world.spawn((Score(1),));
    world.spawn((Score(2),));
    world.spawn((Score(3),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut total = 0;
    plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, rows, (scores,)| {
        assert_eq!(entities.len(), rows.len());
        total += entities.len();
        // For scan plans, rows should be 0..len (all entities in the archetype).
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(row, i, "scan plan rows should be sequential");
        }
        assert_eq!(scores.len(), entities.len());
    })
    .unwrap();
    assert_eq!(total, 3);
}

#[test]
fn for_each_batched_left_join() {
    let mut world = World::new();
    // 5 Score-only, 5 Score+Team
    let mut all_score = Vec::new();
    for i in 0..5 {
        all_score.push(world.spawn((Score(i),)));
    }
    for i in 5..10 {
        all_score.push(world.spawn((Score(i), Team(1))));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Left)
        .build();

    let mut results = Vec::new();
    plan.for_each_batched::<(&Score,), _>(&mut world, |entity, _| {
        results.push(entity);
    })
    .unwrap();

    // Left join: all 10 Score entities should appear.
    assert_eq!(results.len(), 10);
    all_score.sort_by_key(|e| e.to_bits());
    results.sort_by_key(|e| e.to_bits());
    assert_eq!(results, all_score);
}

#[test]
fn for_each_join_chunk_multi_archetype() {
    let mut world = World::new();
    // 3 different archetypes, all with Score
    world.spawn((Score(1),));
    world.spawn((Score(2), Team(1)));
    world.spawn((Score(3), Team(1), Health(50)));

    let planner = QueryPlanner::new(&world);
    // Scan Score, join Team — only archetypes with Team match the join.
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // Collect chunk info: (entity_count, first_entity) per chunk.
    let mut chunk_entities: Vec<Vec<Entity>> = Vec::new();
    plan.for_each_join_chunk::<(&Score,), _>(&mut world, |entities, _, _| {
        assert!(!entities.is_empty());
        chunk_entities.push(entities.to_vec());
    })
    .unwrap();

    // Two archetypes have Team: (Score, Team) and (Score, Team, Health).
    assert_eq!(chunk_entities.len(), 2);
    // Total: 2 entities (one per Team-bearing archetype).
    let total: usize = chunk_entities.iter().map(|c| c.len()).sum();
    assert_eq!(total, 2);
    // Each chunk should have different entities.
    assert_ne!(chunk_entities[0], chunk_entities[1]);
}

#[test]
fn for_each_batched_empty_join() {
    let mut world = World::new();
    // Score-only entities, no Team — inner join produces empty result.
    for i in 0..5 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    let mut called = false;
    plan.for_each_batched::<(&Score,), _>(&mut world, |_, _| {
        called = true;
    })
    .unwrap();
    assert!(!called);
}

#[test]
fn for_each_batched_world_mismatch() {
    let mut world_a = World::new();
    let mut world_b = World::new();
    world_a.spawn((Score(1),));
    world_b.spawn((Score(2),));

    let planner = QueryPlanner::new(&world_a);
    let mut plan = planner.scan::<(&Score,)>().build();
    let result = plan.for_each_batched::<(&Score,), _>(&mut world_b, |_, _| {});
    assert!(result.is_err());
}

#[test]
fn for_each_batched_component_mismatch() {
    let mut world = World::new();
    // Entities have Score only — no Health component.
    world.spawn((Score(1),));
    world.spawn((Score(2),));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    // Request (&Health,) via Q, but no archetype has Health.
    let result = plan.for_each_batched::<(&Health,), _>(&mut world, |_, _| {});
    assert!(
        matches!(result, Err(PlanExecError::ComponentMismatch { .. })),
        "expected ComponentMismatch, got {result:?}"
    );
}
```

- [ ] **Step 2: Run all new tests**

Run: `cargo test -p minkowski -- for_each_batched for_each_join_chunk scratch_sort`
Expected: All PASS.

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "test(planner): comprehensive tests for batch join execution"
```

---

## Chunk 2: Benchmarks & CI

### Task 7: Add `FatData` and `Team` component types to bench lib

**Files:**
- Modify: `crates/minkowski-bench/src/lib.rs`

- [ ] **Step 1: Add the types**

Add after the `Score` struct:

```rust
/// Team identifier — 4 bytes. Used for join benchmarks.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Archive, Serialize, Deserialize,
)]
#[repr(C)]
pub struct Team(pub u32);

/// Fat component — 256 bytes. Used to measure cache-miss amplification
/// on large components in join benchmarks.
#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct FatData {
    pub data: [u8; 256],
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p minkowski-bench`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski-bench/src/lib.rs
git commit -m "feat(bench): add Team and FatData component types for join benchmarks"
```

---

### Task 8: Add join benchmarks

**Files:**
- Modify: `crates/minkowski-bench/benches/planner.rs`

- [ ] **Step 1: Add imports and helper functions**

At the top of the file, update imports to include `JoinKind` and the new types:

```rust
use minkowski::{
    AggregateExpr, BTreeIndex, Changed, HashIndex, JoinKind, Predicate, QueryPlanner,
    SpatialIndex, World,
};
use minkowski_bench::{FatData, Score, Team};
```

Add helper functions after the existing `score_world`:

```rust
/// Spawn `n` entities with `Score`, with `join_pct` fraction also getting `Team`.
/// Returns (world, number_of_joined_entities).
fn join_world(n: u32, join_pct: f64) -> (World, u32) {
    let mut world = World::new();
    let threshold = (n as f64 * join_pct) as u32;
    for i in 0..n {
        if i < threshold {
            world.spawn((Score(i), Team(i % 5)));
        } else {
            world.spawn((Score(i),));
        }
    }
    (world, threshold)
}

/// Spawn `n` entities with `FatData`, with `join_pct` fraction also getting `Team`.
fn fat_join_world(n: u32, join_pct: f64) -> (World, u32) {
    let mut world = World::new();
    let threshold = (n as f64 * join_pct) as u32;
    for i in 0..n {
        let fat = FatData { data: [i as u8; 256] };
        if i < threshold {
            world.spawn((fat, Team(i % 5)));
        } else {
            world.spawn((fat,));
        }
    }
    (world, threshold)
}
```

- [ ] **Step 2: Add core join benchmarks**

Add a new function after the existing `planner` function (but before `criterion_group!`):

```rust
fn join_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("join");

    // ── Baseline: current for_each + world.get() ────────────────────
    group.bench_function("for_each_get_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.for_each(&mut world, |entity| {
                if let Some(score) = world.get::<Score>(entity) {
                    sum += score.0 as u64;
                }
            })
            .unwrap();
            sum
        });
    });

    // ── New: for_each_batched ───────────────────────────────────────
    group.bench_function("for_each_batched_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.for_each_batched::<(&Score,), _>(&mut world, |_, (score,)| {
                sum += score.0 as u64;
            })
            .unwrap();
            sum
        });
    });

    // ── New: for_each_join_chunk ────────────────────────────────────
    group.bench_function("for_each_chunk_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&Score,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.for_each_join_chunk::<(&Score,), _>(&mut world, |_, rows, (scores,)| {
                for &row in rows {
                    sum += scores[row].0 as u64;
                }
            })
            .unwrap();
            sum
        });
    });

    // ── Manual baseline: world.query() (no join) ────────────────────
    group.bench_function("manual_query_10k", |b| {
        let (mut world, _) = join_world(10_000, 0.8);

        b.iter(|| {
            let mut sum = 0u64;
            for (score, _team) in world.query::<(&Score, &Team)>() {
                sum += score.0 as u64;
            }
            sum
        });
    });

    group.finish();
}
```

**Important**: The `for_each_get_10k` benchmark has a problem — `for_each` takes a callback `FnMut(Entity)` and borrows `world` via `&mut World`. Calling `world.get()` inside the callback would require a second borrow. Check how the existing `for_each` works:

Looking at the existing code, `for_each` takes `&mut self, world: &mut World, callback: impl FnMut(Entity)`. The `world` is reborrowed inside the method. The callback captures `world` by move — but `world` is already borrowed by `for_each`.

**Fix**: Use `execute()` to collect entities, then iterate with `world.get()`:

```rust
group.bench_function("for_each_get_10k", |b| {
    let (mut world, _) = join_world(10_000, 0.8);
    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();
    drop(planner);

    b.iter(|| {
        let entities = plan.execute(&mut world).unwrap();
        let mut sum = 0u64;
        for &entity in entities {
            if let Some(score) = world.get::<Score>(entity) {
                sum += score.0 as u64;
            }
        }
        sum
    });
});
```

This is actually the realistic baseline — `execute()` + per-entity `world.get()` is what users do today with join results.

- [ ] **Step 3: Add fat struct benchmarks**

```rust
fn join_fat_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_fat");

    group.bench_function("for_each_get_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&FatData,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let entities = plan.execute(&mut world).unwrap();
            let mut sum = 0u64;
            for &entity in entities {
                if let Some(fat) = world.get::<FatData>(entity) {
                    sum += fat.data[0] as u64;
                }
            }
            sum
        });
    });

    group.bench_function("for_each_batched_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&FatData,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.for_each_batched::<(&FatData,), _>(&mut world, |_, (fat,)| {
                sum += fat.data[0] as u64;
            })
            .unwrap();
            sum
        });
    });

    group.bench_function("for_each_chunk_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);
        let planner = QueryPlanner::new(&world);
        let mut plan = planner
            .scan::<(&FatData,)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build();
        drop(planner);

        b.iter(|| {
            let mut sum = 0u64;
            plan.for_each_join_chunk::<(&FatData,), _>(&mut world, |_, rows, (fats,)| {
                for &row in rows {
                    sum += fats[row].data[0] as u64;
                }
            })
            .unwrap();
            sum
        });
    });

    group.bench_function("manual_query_10k", |b| {
        let (mut world, _) = fat_join_world(10_000, 0.8);

        b.iter(|| {
            let mut sum = 0u64;
            for (fat, _team) in world.query::<(&FatData, &Team)>() {
                sum += fat.data[0] as u64;
            }
            sum
        });
    });

    group.finish();
}
```

- [ ] **Step 4: Add selectivity sweep benchmarks**

```rust
fn join_selectivity_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_selectivity");

    for (label, pct) in [("10pct", 0.1), ("50pct", 0.5), ("90pct", 0.9)] {
        group.bench_function(&format!("get_{label}"), |b| {
            let (mut world, _) = join_world(10_000, pct);
            let planner = QueryPlanner::new(&world);
            let mut plan = planner
                .scan::<(&Score,)>()
                .join::<(&Team,)>(JoinKind::Inner)
                .build();
            drop(planner);

            b.iter(|| {
                let entities = plan.execute(&mut world).unwrap();
                let mut sum = 0u64;
                for &entity in entities {
                    if let Some(score) = world.get::<Score>(entity) {
                        sum += score.0 as u64;
                    }
                }
                sum
            });
        });

        group.bench_function(&format!("chunk_{label}"), |b| {
            let (mut world, _) = join_world(10_000, pct);
            let planner = QueryPlanner::new(&world);
            let mut plan = planner
                .scan::<(&Score,)>()
                .join::<(&Team,)>(JoinKind::Inner)
                .build();
            drop(planner);

            b.iter(|| {
                let mut sum = 0u64;
                plan.for_each_join_chunk::<(&Score,), _>(&mut world, |_, rows, (scores,)| {
                    for &row in rows {
                        sum += scores[row].0 as u64;
                    }
                })
                .unwrap();
                sum
            });
        });
    }

    group.finish();
}
```

- [ ] **Step 5: Update `criterion_group!` and `criterion_main!`**

Change the bottom of the file:

```rust
criterion_group!(benches, planner, join_benches, join_fat_benches, join_selectivity_benches);
criterion_main!(benches);
```

- [ ] **Step 6: Verify benchmarks compile**

Run: `cargo check -p minkowski-bench --benches`
Expected: PASS

- [ ] **Step 7: Run one benchmark to verify**

Run: `cargo bench -p minkowski-bench -- join/manual_query_10k`
Expected: Runs and reports a result.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski-bench/benches/planner.rs
git commit -m "bench(planner): add join, join_fat, and join_selectivity benchmark suites"
```

---

### Task 9: Update Miri subset

**Files:**
- Modify: `ci/miri-subset.txt`
- Modify: `ci/run-miri-subset.sh`

- [ ] **Step 1: Add tests to `ci/miri-subset.txt`**

Append before the closing of the file (after the Bundle section):

```
# ── Planner: batch join execution (3 selected) ────────────────────
planner::tests::for_each_batched_yields_all_join_results
planner::tests::for_each_join_chunk_yields_correct_slices
planner::tests::for_each_join_chunk_multi_archetype
```

- [ ] **Step 2: Add tests to `ci/run-miri-subset.sh`**

Add to the `EXACT_TESTS` array:

```bash
    # Planner batch join execution (3)
    "planner::tests::for_each_batched_yields_all_join_results"
    "planner::tests::for_each_join_chunk_yields_correct_slices"
    "planner::tests::for_each_join_chunk_multi_archetype"
```

- [ ] **Step 3: Commit**

```bash
git add ci/miri-subset.txt ci/run-miri-subset.sh
git commit -m "ci: add batch join execution tests to Miri subset"
```

---

### Task 10: Run full benchmarks and update perf roadmap

**Files:**
- Modify: `docs/perf-roadmap.md`

- [ ] **Step 1: Run the full join benchmark suite**

Run: `cargo bench -p minkowski-bench -- "join|join_fat|join_selectivity"`

Record all results.

- [ ] **Step 2: Update `docs/perf-roadmap.md`**

Add a new section under P1-2 (or replace it):

```markdown
### P1-4: Join batch execution — COMPLETED

**Implementation**: Archetype-sorted batch execution for join plans.
After `run_join()` materialises entities into ScratchBuffer, sort by
packed `(archetype_id << 32 | row)` key. Walk archetype runs calling
`init_fetch` once per archetype, `fetch`/`as_slice` per entity.

**Results**: [Fill in actual benchmark numbers]

| Benchmark | Time | vs baseline |
|---|---|---|
| `join/for_each_get_10k` | ? | baseline |
| `join/for_each_batched_10k` | ? | ?x faster |
| `join/for_each_chunk_10k` | ? | ?x faster |
| `join/manual_query_10k` | ? | theoretical limit |

**API**: `for_each_batched`, `for_each_batched_raw`, `for_each_join_chunk`
on `QueryPlanResult`.
```

Update the baselines table with the new benchmark results.

- [ ] **Step 3: Commit**

```bash
git add docs/perf-roadmap.md
git commit -m "docs: update perf roadmap with join batch execution results"
```

---

### Task 11: Final validation

- [ ] **Step 1: Run full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

- [ ] **Step 3: Run fmt**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

- [ ] **Step 4: Create PR**

Use the `/pr` skill to create a pull request.
