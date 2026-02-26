# Phase 1: ECS Core — Design Document

## Goal

Build the foundational ECS storage layer for minkowski — column-oriented archetype
storage with generational entity IDs, parallel iteration, and deferred mutation. This
is the game-workload-first foundation that later phases extend with compile-time schema
optimization, persistence, query planning, and transactions.

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary workload | Game ECS first | Database features layer on top once storage is solid |
| Dense storage | BlobVec (type-erased blobs) | Max performance, proven pattern, memcpy-friendly for future snapshots |
| Sparse storage | `HashMap<Entity, T>` per component | Simple, correct, upgradeable to sparse sets via profiling |
| Entity ID | 32-bit index + 32-bit generation (u64) | Standard, fits in a register, ~4B entities and reuses |
| Query matching | Bitset cold scan + cached archetype list | Bitset is trivial to implement; caching makes hot queries free |
| Parallelism | Parallel iteration, serial mutation | Columns are Send+Sync; structural mutations deferred via CommandBuffer |
| Crate layout | `minkowski` + `minkowski-derive` (placeholder) | Proc-macro crate forced separate by Rust; everything else in one crate |

## Entity & Component Model

### Entity

```
Entity = u64
  bits[0..32]:  index (slot in entity allocator)
  bits[32..64]: generation (incremented on recycle)
```

**Entity allocator**: `Vec<Generation>` for metadata + free list (stack of recycled
indices). `spawn()` pops from free list or extends the vec. `despawn()` bumps
generation and pushes to free list.

**EntityLocation**: `{ archetype_id: ArchetypeId, row: usize }` — stored in a parallel
`Vec<Option<EntityLocation>>` indexed by entity index.

### Components

- **ComponentId**: `usize`, assigned monotonically at registration time.
- **ComponentInfo**: `{ id, name: &'static str, layout: Layout, drop: Option<unsafe fn(*mut u8)> }`.
- **Registration**: `world.register_component::<T>()` returns `ComponentId`. Idempotent.
- **Component trait**: marker + `'static + Send + Sync` bounds.

### Sparse vs Dense

Per-component-type decision at registration time:
- **Dense** (default): stored in archetype BlobVec columns.
- **Sparse** (`world.register_sparse::<T>()`): stored in `HashMap<Entity, T>`.

## Archetype Storage

### Archetype

```rust
Archetype {
    id: ArchetypeId,
    component_ids: FixedBitSet,                     // which components
    columns: Vec<BlobVec>,                           // one per component
    component_index: HashMap<ComponentId, usize>,    // ComponentId -> column index
    entities: Vec<Entity>,                           // row -> entity
    len: usize,
}
```

### BlobVec

Type-erased growable array: `item_layout: Layout`, `drop_fn: Option<unsafe fn(*mut u8)>`,
`data: NonNull<u8>`, `len`, `capacity`.

API: `push(ptr)`, `swap_remove(row)`, `get_ptr(row) -> *mut u8`.

### Archetype Graph

```rust
ArchetypeGraph {
    archetypes: Vec<Archetype>,
    edges: HashMap<(ArchetypeId, ComponentId), ArchetypeEdge>,
}

ArchetypeEdge {
    add: Option<ArchetypeId>,
    remove: Option<ArchetypeId>,
}
```

Edges are lazy — created on first traversal. New archetype creation bumps
`World.archetype_generation`, which invalidates cached query state.

### Component-to-Archetype Index

```rust
component_archetypes: HashMap<ComponentId, HashSet<ArchetypeId>>
```

Inverse index for query matching: "which archetypes contain component C?"

## Query System

### Two-tier matching

**Cold path** (first run or stale):
1. Build `QueryMask`: `required: FixedBitSet`, `excluded: FixedBitSet`
2. Scan archetypes: `(arch.component_ids & required) == required && (arch.component_ids & excluded) == 0`
3. Collect into `QueryState`

**Hot path** (cached):
1. Check `QueryState.generation` vs `World.archetype_generation`
2. If stale, scan only archetypes created since last generation
3. Iterate cached archetype list

### QueryState

```rust
QueryState<Q: WorldQuery> {
    matched_archetypes: Vec<ArchetypeId>,
    generation: u64,
    component_ids: Vec<ComponentId>,
}
```

### WorldQuery trait

```rust
unsafe trait WorldQuery {
    type Item<'w>;
    type Fetch<'w>;

    fn init_fetch(world: &World, archetype: &Archetype) -> Self::Fetch<'_>;
    unsafe fn fetch(fetch: &Self::Fetch<'_>, row: usize) -> Self::Item<'_>;
    fn component_ids(world: &World) -> Vec<ComponentId>;
}
```

Implemented for `&T`, `&mut T`, `Option<&T>`, `Entity`, and tuples up to arity 12
(macro-generated).

### Iteration

```rust
// Sequential
for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
    pos.x += vel.dx;
}

// Parallel (rayon)
world.query::<(&mut Position, &Velocity)>().par_for_each(|pos, vel| {
    pos.x += vel.dx;
});
```

`par_for_each` splits archetype rows into chunks, dispatches to rayon. Sparse
`Option<&SparseComponent>` does a HashMap lookup per entity — no archetype filtering.

## World API & Mutation

### World struct

```rust
pub struct World {
    entities: EntityAllocator,
    archetypes: ArchetypeGraph,
    components: ComponentRegistry,
    sparse_storage: SparseStorage,
    archetype_generation: u64,
    entity_locations: Vec<Option<EntityLocation>>,
}
```

### Immediate mutation (serial context)

```rust
let entity = world.spawn((Position::new(0.0, 0.0), Velocity::new(1.0, 0.0)));
world.get::<Position>(entity);          // -> Option<&Position>
world.get_mut::<Position>(entity);      // -> Option<&mut Position>
world.insert(entity, Health(100));      // archetype migration
world.remove::<Health>(entity);         // archetype migration
world.despawn(entity);
```

### Deferred mutation (during iteration)

```rust
let mut commands = CommandBuffer::new();
commands.spawn((Position::new(0.0, 0.0), Velocity::new(1.0, 0.0)));
commands.despawn(entity);
commands.insert(entity, Health(100));
commands.remove::<Health>(entity);
commands.apply(&mut world);
```

CommandBuffer: growable byte buffer of encoded commands (tag + entity + component data).
Designed so Phase 4's ChangeSet can wrap rather than replace it.

### Archetype migration

`world.insert(entity, C)` moves the entire row from archetype A to A∪{C}: memcpy per
column, swap-remove from old, update EntityLocation. Graph edge caches destination.

## Module Structure

```
minkowski/
├── Cargo.toml                      # workspace
├── crates/
│   ├── minkowski/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── entity.rs           # Entity, EntityAllocator, Generation
│   │       ├── component.rs        # ComponentId, ComponentInfo, ComponentRegistry
│   │       ├── storage/
│   │       │   ├── mod.rs
│   │       │   ├── blob_vec.rs     # BlobVec
│   │       │   ├── archetype.rs    # Archetype, ArchetypeGraph
│   │       │   └── sparse.rs       # SparseStorage
│   │       ├── query/
│   │       │   ├── mod.rs
│   │       │   ├── state.rs        # QueryState, matching
│   │       │   ├── fetch.rs        # WorldQuery trait, impls
│   │       │   └── iter.rs         # QueryIter, par_for_each
│   │       ├── world.rs            # World
│   │       └── command.rs          # CommandBuffer
│   └── minkowski-derive/           # Phase 2 placeholder
│       ├── Cargo.toml
│       └── src/lib.rs
├── benches/
│   ├── spawn.rs
│   ├── iterate.rs
│   ├── parallel.rs
│   ├── add_remove.rs
│   └── fragmented.rs
└── examples/
    └── boids.rs
```

### Dependencies

- `rayon` — parallel iteration
- `fixedbitset` — archetype component masks
- `criterion` — benchmarks (dev-dependency)

## Phase 1 Deliverables

1. Core storage: BlobVec, Archetype, ArchetypeGraph, SparseStorage
2. Entity management: EntityAllocator with generational IDs
3. Component registry: ComponentId assignment, ComponentInfo
4. World: spawn, despawn, get, get_mut, insert, remove
5. Queries: WorldQuery trait, QueryState caching, sequential + parallel iteration
6. CommandBuffer: deferred mutation, apply
7. Criterion benchmarks vs hecs: spawn, iterate, parallel, add/remove, fragmented
   - Target: within 2x of hecs on all benchmarks
   - Faster on parallel iteration (or something is wrong)
8. Boids example: terminal output, 10K frames, exercises spawn/iterate/mutate/despawn
   - Prints entity count + average velocity per frame
   - Primary purpose: API ergonomics validation + integration test

## Explicit Non-Goals (Phase 1)

- No proc macros / `#[derive(Table)]`
- No persistence / WAL / snapshots
- No indexes (B-tree, hash)
- No transaction semantics
- No query planning / Volcano model
- No serialization
- No change detection ticks
- No automatic scheduler / conflict detection
