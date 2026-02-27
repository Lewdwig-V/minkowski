# Phase 2: #[derive(Table)] — Design Document

## Goal

Add a proc macro that turns a named struct into a compile-time schema declaration. The struct is simultaneously the data container (implements Bundle for spawning) and the schema (generates typed accessors with known column offsets that bypass HashMap lookups).

## What It Replaces

Phase 1 spawn and query:
```rust
// Spawn with tuples — no schema, no names
let e = world.spawn((Position(v), Rotation(q), Scale(v)));

// Query with tuples — HashMap lookup per component per archetype
for (pos, rot) in world.query::<(&mut Position, &Rotation)>() { .. }
```

Phase 2 adds:
```rust
#[derive(Table)]
struct Transform {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

// Spawn with named struct
let e = world.spawn(Transform { position: .., rotation: .., scale: .. });

// Query with typed accessors — constant column indices, no HashMap
for row in world.query_table_mut::<Transform>() {
    row.position.x += row.scale.x;
}
```

Both paths coexist. Entities spawned via Table are queryable via dynamic queries too.

## What #[derive(Table)] Generates

Given `struct Transform { position: Vec3, rotation: Quat, scale: Vec3 }`:

### 1. Bundle impl

Auto-implements `Bundle` for the struct. `component_ids()` registers each field's type as a component in field declaration order. `put()` yields each field's bytes via `ManuallyDrop`. Identical semantics to the hand-written tuple Bundle impls.

### 2. Table trait impl

```rust
pub trait Table: Bundle + Sized {
    const FIELD_COUNT: usize;
    fn register(registry: &mut ComponentRegistry) -> Vec<ComponentId>;
    fn descriptor(world: &World) -> &TableDescriptor;
}
```

- `FIELD_COUNT = 3`
- `register()` calls `registry.register::<Vec3>()`, `registry.register::<Quat>()`, `registry.register::<Vec3>()` in field order, returns the component IDs
- `descriptor()` does one-time resolution: registers components, gets/creates the archetype, maps field indices to archetype column indices, caches the result

### 3. Row reference types

```rust
struct TransformRef<'w> {
    position: &'w Vec3,
    rotation: &'w Quat,
    scale: &'w Vec3,
}

struct TransformMut<'w> {
    position: &'w mut Vec3,
    rotation: &'w mut Quat,
    scale: &'w mut Vec3,
}
```

Constructed from raw column pointers + row offset. Field access is a single pointer dereference — no HashMap, no bitset, no indirection.

## TableDescriptor

```rust
pub struct TableDescriptor {
    pub archetype_id: ArchetypeId,
    pub col_indices: Vec<usize>,
}
```

Cached in `World` in a `HashMap<TypeId, TableDescriptor>`. First call resolves, all subsequent calls return cached. Maps field index (0, 1, 2) to archetype column index.

Key invariant: field declaration order = component registration order = column index order within the table's archetype.

## Query API

```rust
impl World {
    pub fn query_table<T: Table>(&self) -> TableIter<'_, T> { .. }
    pub fn query_table_mut<T: Table>(&mut self) -> TableIterMut<'_, T> { .. }
}
```

`TableIter` holds resolved column pointers and archetype length. Each `.next()` yields a `TransformRef` / `TransformMut` by offsetting column pointers by current row.

What this skips compared to `world.query()`:
- No bitset construction
- No archetype scan/filter
- No `init_fetch` per archetype
- Direct jump to one known archetype by cached ID

## Interop

Entities spawned via `world.spawn(Transform{..})` live in a normal archetype. They are queryable via both:
- `world.query_table::<Transform>()` — fast path, typed accessors
- `world.query::<(&Position, &Rotation)>()` — dynamic path, tuple access

Same data, different access patterns. No duplication.

## Scope

**In scope:**
- `#[derive(Table)]` proc macro in `minkowski-derive`
- `Table` trait with `FIELD_COUNT`, `register()`, `descriptor()`
- `TableDescriptor` caching in World
- `Bundle` impl auto-generated for Table structs
- `TransformRef<'w>` / `TransformMut<'w>` row reference types
- `World::query_table()` / `World::query_table_mut()`
- `TableIter` / `TableIterMut`
- Tests: correct Bundle generation, descriptor caching, query_table yields correct data, mutation, interop with dynamic queries

**Non-goals:**
- No WorldQuery impl for Table types
- No par_for_each on TableIter
- No `#[index]` attributes
- No change detection
- No schema migration
