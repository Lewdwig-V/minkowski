# EnumChangeSet Typed API Design

## Problem

`EnumChangeSet`'s public methods (`record_insert`, `record_remove`, `record_spawn`) require `ComponentId`, but external users have no way to obtain one. `ComponentRegistry` is `pub(crate)` and `World` exposes no lookup. Crate-internal tests work because they access `world.components.register::<T>()` directly — external consumers cannot.

## Solution

Add typed safe helper methods on `EnumChangeSet` that resolve `ComponentId` internally, plus `World` methods to expose component registration for power users.

## Design

### World methods

```rust
impl World {
    /// Read-only lookup. Returns None if type was never registered.
    pub fn component_id<T: Component>(&self) -> Option<ComponentId>

    /// Register a component type, returning its id. Idempotent.
    pub fn register_component<T: Component>(&mut self) -> ComponentId
}
```

### EnumChangeSet typed helpers

```rust
impl EnumChangeSet {
    /// Record inserting a component on an entity.
    /// Auto-registers the component type. Safe — handles ManuallyDrop internally.
    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, value: T)

    /// Record removing a component from an entity.
    /// Auto-registers the component type.
    pub fn remove<T: Component>(&mut self, world: &mut World, entity: Entity)

    /// Record spawning an entity with a bundle of components.
    /// Auto-registers all component types in the bundle.
    pub fn spawn<B: Bundle>(&mut self, world: &mut World, entity: Entity, bundle: B)
}
```

`insert` uses `ManuallyDrop` to prevent double-free after arena copy. `spawn` calls `Bundle::put` directly and pushes `Mutation::Spawn` inline (bypasses `record_spawn` because arena allocation happens during the `put` callback).

Existing `record_insert`, `record_remove`, `record_spawn` remain as the raw power-user API.

### Re-exports

Add `pub use component::ComponentId` to `lib.rs` so the return type of `world.component_id::<T>()` is usable externally.

### Tests

- Migrate existing `changeset.rs` tests to use the new public API.
- Add an external integration test (outside the crate) exercising the typed API — the test that would have caught this bug originally.

## Files changed

| File | Change |
|------|--------|
| `crates/minkowski/src/world.rs` | Add `component_id<T>()` and `register_component<T>()` |
| `crates/minkowski/src/changeset.rs` | Add `insert<T>()`, `remove<T>()`, `spawn<B>()` typed helpers |
| `crates/minkowski/src/lib.rs` | Re-export `ComponentId` |
| `crates/minkowski/src/changeset.rs` tests | Migrate to new public API |
| `crates/minkowski/tests/changeset_external.rs` | New integration test |
