# Extended Reducers Design

**Goal:** Add structural mutations (despawn, remove) and dynamic iteration (for_each) to the reducer system.

**Architecture:** Structural mutations buffer into EnumChangeSet like existing writes — the scheduler treats them as column writes. Dynamic iteration uses typed query codepaths with runtime access validation. No new registration paths or handle types.

---

## Structural Mutations

### can_remove

`DynamicReducerBuilder::can_remove::<T>()` marks T as written in the Access bitset. Removal is a write from the scheduler's perspective — it modifies the column at commit time (archetype migration).

Enables two methods on DynamicCtx:

```rust
ctx.remove::<T>(entity)        // panics if T undeclared or entity missing T
ctx.try_remove::<T>(entity) -> bool  // returns false if entity missing T
```

Both buffer `Mutation::Remove` into the changeset. Panics if T was not declared via `can_remove`.

EntityMut also gets `remove()` — bounded by the declared component set C.

### can_despawn

`DynamicReducerBuilder::can_despawn()` sets a `despawns: bool` flag on Access. No component set parameter — despawn is a blanket "I may destroy entities."

Enables:

```rust
ctx.despawn(entity)  // panics if can_despawn() not declared
```

Buffers `Mutation::Despawn` into the changeset.

EntityMut also gets `despawn()` — requires the despawn flag at registration.

### Access conflict rule

```rust
fn conflicts_with(&self, other: &Access) -> bool {
    let column_conflict = /* existing read-write / write-write bitset logic */;
    let despawn_conflict =
        (self.despawns && other.has_any_access()) ||
        (other.despawns && self.has_any_access());
    column_conflict || despawn_conflict
}
```

Where `has_any_access()` returns true if reads or writes bitsets are non-empty.

Two despawn reducers with disjoint reads still conflict — each side's despawn flag hits the other side's non-empty component access. This is correct: despawn can destroy any entity, so it must serialize against any component reader/writer.

### Why two access categories, not three

Structural mutations (remove, despawn) are buffered in the ChangeSet and applied at commit under `&mut World`. During the parallel execution phase, no archetype changes happen. The entity doesn't move between archetype tables until the sequential commit phase.

The scheduler sees columns, not archetype topology. From its perspective, `remove<Shield>` is a write to the Shield column. No third access category needed.

The only path where structural mutations would need special treatment is direct query reducers (`register_query` / `QueryMut`), which take `&mut World` and mutate in place. But direct query reducers don't expose `world.despawn()` — the handle prevents it. Structural mutations only happen through the transactional path.

---

## Dynamic Iteration

### DynamicCtx::for_each

```rust
impl DynamicCtx {
    pub fn for_each<Q: ReadOnlyWorldQuery + 'static>(
        &mut self,
        f: impl FnMut(Q::Item<'_>),
    )
}
```

The query execution is fully typed. The "dynamic" part is only the access validation — at entry, `Q::accessed_ids` is checked as a subset of the builder-declared reads + writes. Panics if the query accesses undeclared components.

Iteration uses the typed codepath: archetype scan via `Q::required_ids`, filter via `Q::matches_filters`, fetch via `Q::init_fetch` / `Q::fetch`. Manual archetype scan (same as QueryWriter), no cache interaction, no tick marking on mutable columns.

### Changed<T> support

`Changed<T>` works naturally — it's a `ReadOnlyWorldQuery` filter. Per-reducer tick state via `Arc<AtomicU64>`, same pattern as QueryWriter. `last_read_tick` updated by `call()` after commit (not inside `for_each`), ensuring the stored tick is newer than any column tick set during changeset application.

### Write pattern

Read via typed query, write via `ctx.write()` / `ctx.remove()` / `ctx.despawn()`. No `&mut T` in the query — the query is read-only, mutations buffer through the changeset.

```rust
registry.dynamic("reaper", &mut world)
    .can_read::<Health>()
    .can_despawn()
    .build(|ctx, _trigger_entity| {
        ctx.for_each::<(Entity, &Health)>(|(&entity, health)| {
            if health.0 == 0 {
                ctx.despawn(entity);
            }
        });
    });
```

---

## What doesn't ship

- **Dynamic query reducers** as a separate registration path — `DynamicCtx::for_each` covers it.
- **Runtime access tracking / profiling** — YAGNI until there's a real scheduler.
- **Mutable iteration in DynamicCtx** — reads via typed query, writes via ctx methods.
- **`can_despawn_with::<(H,P,V)>()`** — dropped. Blanket `can_despawn()` avoids soundness risk from incorrect component set declarations.
- **Despawner handle type** — despawn lives on EntityMut and DynamicCtx.

---

## Files touched

| File | Changes |
|---|---|
| `access.rs` | Add `despawns: bool` field, update `conflicts_with`, add `has_any_access()` |
| `reducer.rs` | DynamicCtx: `for_each`, `despawn`, `remove`, `try_remove`. DynamicReducerBuilder: `can_remove`, `can_despawn`. EntityMut: `despawn`, `remove`. DynamicResolved: track remove/despawn declarations. |
| `changeset.rs` | Public `despawn(&mut self, entity)` helper |
| `examples/reducer.rs` | Demo structural mutations + dynamic iteration |
| `CLAUDE.md` | Update Access, DynamicCtx, EntityMut docs |

## Testing

- `can_remove` marks write in Access bitset
- `can_despawn` sets flag, conflicts with any non-empty access
- Two despawn reducers always conflict (each side's flag hits other's reads)
- `ctx.remove()` buffers Mutation::Remove, applied at commit
- `ctx.despawn()` buffers Mutation::Despawn, applied at commit
- `ctx.remove()` on undeclared component panics
- `ctx.despawn()` without `can_despawn()` panics
- `for_each` iterates matching entities with typed query
- `for_each` with undeclared component panics
- `for_each` with `Changed<T>` skips unchanged archetypes
- `EntityMut::despawn()` and `EntityMut::remove()` buffer correctly
