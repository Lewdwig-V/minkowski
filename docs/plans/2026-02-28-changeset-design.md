# ChangeSet — Mutation Abstraction

## Problem

All mutations today are either direct method calls (`world.spawn()`) or opaque closures (`CommandBuffer`). You can't inspect, serialize, replay, or undo them. This blocks persistence (WAL), replication (send mutations over wire), and transactions (rollback on failure).

## Design

A `ChangeSet` trait defines the interface for recording and applying structural mutations. `EnumChangeSet` is the default implementation using a `Vec<Mutation>` enum + a contiguous `Arena` for component byte data. `CommandBuffer` is refactored to be a typed facade over `EnumChangeSet`.

### ChangeSet Trait

```rust
pub trait ChangeSet: Send {
    fn spawn(&mut self, entity: Entity, components: &[(ComponentId, *const u8, Layout)]);
    fn despawn(&mut self, entity: Entity);
    fn insert(&mut self, entity: Entity, component_id: ComponentId, data: *const u8, layout: Layout);
    fn remove(&mut self, entity: Entity, component_id: ComponentId);

    /// Apply all mutations. Returns a reverse ChangeSet that undoes them.
    fn apply(self: Box<Self>, world: &mut World) -> Box<dyn ChangeSet>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```

The trait is the extension point. Today we ship `EnumChangeSet`. Future implementations (e.g. `FlatChangeLog` backed by a contiguous byte buffer for WAL-friendly workloads) implement the same trait.

### Mutation Enum

```rust
enum Mutation {
    Spawn {
        entity: Entity,
        components: Vec<(ComponentId, usize, Layout)>, // (id, arena offset, layout)
    },
    Despawn {
        entity: Entity,
    },
    Insert {
        entity: Entity,
        component_id: ComponentId,
        offset: usize,
        layout: Layout,
    },
    Remove {
        entity: Entity,
        component_id: ComponentId,
    },
}
```

Component data is **not** stored inline in the enum. Each mutation stores an offset into a shared `Arena`. This avoids per-mutation heap allocation — all component bytes live in one contiguous `Vec<u8>`.

### Arena

```rust
pub(crate) struct Arena {
    data: Vec<u8>,
}

impl Arena {
    fn alloc(&mut self, src: *const u8, layout: Layout) -> usize {
        if layout.size() == 0 { return 0; }
        let align = layout.align();
        let offset = (self.data.len() + align - 1) & !(align - 1);
        self.data.resize(offset + layout.size(), 0);
        unsafe {
            std::ptr::copy_nonoverlapping(
                src, self.data.as_mut_ptr().add(offset), layout.size()
            );
        }
        offset
    }

    fn get(&self, offset: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(offset) }
    }
}
```

**Key safety property**: Mutations store integer offsets, not pointers. The `Vec<u8>` backing the arena can grow (reallocate) freely. Pointers are only computed from offsets at `apply()` time.

### EnumChangeSet

```rust
pub struct EnumChangeSet {
    mutations: Vec<Mutation>,
    arena: Arena,
}
```

### Reverse Capture Semantics

`apply()` builds a reverse `EnumChangeSet` by capturing the data needed to undo each mutation:

| Forward | What apply() captures for reverse |
|---------|----------------------------------|
| **Spawn(E, components)** | Records `Despawn(E)` — entity ID is sufficient |
| **Despawn(E)** | Reads ALL component bytes from E's archetype into reverse arena. Records `Spawn(E, all_components)` |
| **Insert(E, C, data)** | If E already had C: copies old bytes into reverse arena, records `Insert(E, C, old_data)`. If E didn't have C: records `Remove(E, C)` |
| **Remove(E, C)** | Copies component bytes into reverse arena. Records `Insert(E, C, old_data)` |

The reverse is itself an `EnumChangeSet`. Applying the reverse produces the forward again (round-trip).

### CommandBuffer Refactor

CommandBuffer becomes a typed facade:

```rust
pub struct CommandBuffer {
    changes: EnumChangeSet,
}

impl CommandBuffer {
    pub fn spawn<B: Bundle>(&mut self, bundle: B);
    pub fn despawn(&mut self, entity: Entity);
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T);
    pub fn remove<T: Component>(&mut self, entity: Entity);

    /// Apply all commands. Returns a reverse changeset for rollback.
    pub fn apply(self, world: &mut World) -> EnumChangeSet;

    pub fn is_empty(&self) -> bool;
}
```

The typed methods handle `Bundle`/`Component` → `(ComponentId, *const u8, Layout)` conversion. Users never touch raw pointers. `apply` now returns a reverse changeset — callers can ignore it if they don't need rollback.

**Breaking change**: `apply` returns `EnumChangeSet` instead of `()`. Callers that don't need rollback add `let _ =` or `.apply()` stays the same and we add a separate `apply_reversible()`. Decision: return the reverse from `apply()` — it's zero-cost if you drop it immediately, and the API makes rollback discoverable.

### Scope

**Captured**: Structural mutations only — spawn, despawn, insert component, remove component.

**Not captured**: In-place field mutations during query iteration (e.g. `pos.x += 1.0`). These happen via direct pointer writes and require change detection (Phase 3) to track.

### Future: FlatChangeLog

When profiling shows `EnumChangeSet` allocation patterns matter, a `FlatChangeLog` implementing the same `ChangeSet` trait stores everything in a single `Vec<u8>` with a header per entry (opcode + entity + component_id + length). Maximally compact, directly writable to a WAL file. Same trait, swap implementation.

### Testing

1. **Round-trip**: apply forward, apply reverse, verify world matches original state.
2. **Spawn + reverse**: spawn via changeset, reverse despawns the entity.
3. **Despawn + reverse**: despawn via changeset, reverse respawns with all original components.
4. **Insert new + reverse**: insert new component, reverse removes it.
5. **Insert overwrite + reverse**: overwrite existing component, reverse restores old value.
6. **Remove + reverse**: remove component, reverse re-inserts it.
7. **CommandBuffer parity**: verify CommandBuffer produces same results as before refactor.
8. **Empty changeset**: apply empty changeset, world unchanged, reverse is empty.
9. **Arena alignment**: test with components of varying alignment requirements (u8, u32, u64, ZSTs).

### Files

- Create: `crates/minkowski/src/changeset.rs` — Arena, Mutation, EnumChangeSet, ChangeSet trait
- Modify: `crates/minkowski/src/command.rs` — Refactor CommandBuffer to use EnumChangeSet
- Modify: `crates/minkowski/src/lib.rs` — Add `pub mod changeset`, re-export ChangeSet types
- Modify: `crates/minkowski/src/world.rs` — May need helper methods for reading entity component data (for reverse capture)
