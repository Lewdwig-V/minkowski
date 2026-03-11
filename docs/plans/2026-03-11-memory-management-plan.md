# Memory Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TigerBeetle-style memory pooling, blob offloading, and retention
to Minkowski ECS — three independent features sharing the theme "RAM is precious."

**Architecture:** Three independent feature groups, each its own PR. Feature 1
(pool allocator) is the largest — it touches every internal allocation path.
Features 2 and 3 are small, self-contained additions that follow existing
patterns (SpatialIndex, scheduled reducers).

**Tech Stack:** Rust, `memmap2` (mmap — already a workspace dependency),
lock-free atomics, existing Minkowski reducer/trait infrastructure.

---

## Group A: Blob Offloading (Feature 2)

Smallest feature. Ships first to build confidence and provide immediate value.
No dependency on the pool allocator.

### Task A1: `BlobRef` component type + `BlobStore` trait

**Files:**
- Create: `crates/minkowski/src/blob.rs`
- Modify: `crates/minkowski/src/lib.rs` (add module + re-exports)

**Step 1: Write the failing test**

In `crates/minkowski/src/blob.rs`, create the module with tests:

```rust
//! External blob reference component and lifecycle trait.
//!
//! `BlobRef` holds a key/URL to data in an external object store (S3, MinIO,
//! local filesystem). The ECS stores only the reference — blob bytes never
//! enter the World. Same external composition pattern as [`SpatialIndex`].

/// Reference to an externally-stored blob.
///
/// The ECS stores only this key string — the actual blob bytes live in an
/// external object store. Persistence serializes the key, not the remote blob.
/// On snapshot restore, keys are restored but remote blobs must still exist.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlobRef(pub String);

impl BlobRef {
    /// Create a new blob reference from a key string.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// The key/URL string.
    pub fn key(&self) -> &str {
        &self.0
    }
}

/// Lifecycle hook for external blob storage.
///
/// The engine does **not** call these methods automatically. Users wire them
/// into cleanup reducers or framework-level hooks. Same responsibility model
/// as [`SpatialIndex::rebuild`](crate::SpatialIndex::rebuild).
///
/// # Example
///
/// ```ignore
/// struct S3Store { client: S3Client }
///
/// impl BlobStore for S3Store {
///     fn on_orphaned(&mut self, refs: &[&BlobRef]) {
///         for r in refs {
///             self.client.delete_object(r.key());
///         }
///     }
/// }
/// ```
pub trait BlobStore {
    /// Called with blob references no longer attached to any live entity.
    /// The implementor is responsible for deleting the external blobs.
    fn on_orphaned(&mut self, refs: &[&BlobRef]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    #[test]
    fn blob_ref_is_a_component() {
        let mut world = World::new();
        let e = world.spawn((BlobRef::new("s3://bucket/key.bin"),));
        let got = world.get::<BlobRef>(e).unwrap();
        assert_eq!(got.key(), "s3://bucket/key.bin");
    }

    #[test]
    fn blob_ref_survives_archetype_migration() {
        let mut world = World::new();
        let e = world.spawn((BlobRef::new("s3://a"),));
        world.insert(e, (42u32,));
        assert_eq!(world.get::<BlobRef>(e).unwrap().key(), "s3://a");
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
    }

    #[test]
    fn blob_store_receives_orphaned_refs() {
        struct TestStore { deleted: Vec<String> }
        impl BlobStore for TestStore {
            fn on_orphaned(&mut self, refs: &[&BlobRef]) {
                for r in refs {
                    self.deleted.push(r.key().to_owned());
                }
            }
        }

        let refs = vec![
            BlobRef::new("s3://a"),
            BlobRef::new("s3://b"),
        ];
        let borrowed: Vec<&BlobRef> = refs.iter().collect();
        let mut store = TestStore { deleted: vec![] };
        store.on_orphaned(&borrowed);

        assert_eq!(store.deleted, vec!["s3://a", "s3://b"]);
    }

    #[test]
    fn blob_ref_query_iteration() {
        let mut world = World::new();
        world.spawn((BlobRef::new("s3://1"),));
        world.spawn((BlobRef::new("s3://2"),));
        world.spawn((42u32,)); // different archetype, no BlobRef

        let mut keys: Vec<String> = Vec::new();
        world.query::<(&BlobRef,)>().for_each(|(r,)| {
            keys.push(r.key().to_owned());
        });
        keys.sort();
        assert_eq!(keys, vec!["s3://1", "s3://2"]);
    }
}
```

**Step 2: Wire into lib.rs**

Add to `crates/minkowski/src/lib.rs`:
- Module declaration: `mod blob;`
- Re-exports: `pub use blob::{BlobRef, BlobStore};`

**Step 3: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- blob`
Expected: 4 tests PASS.

**Step 4: Run full CI checks**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test -p minkowski --lib`
Expected: All pass, no warnings.

**Step 5: Commit**

```
feat: add BlobRef component and BlobStore lifecycle trait

External blob reference for object store offloading (S3, MinIO, etc.).
Follows SpatialIndex composition pattern — engine provides hooks, user
provides policy.
```

---

### Task A2: Blob offloading example

**Files:**
- Modify: `examples/examples/persist.rs` (or create a new `examples/examples/blob.rs`)

**Step 1: Write a minimal example**

The example demonstrates the BlobRef + BlobStore pattern:

```rust
//! Blob offloading: store large data references in the ECS, actual bytes
//! in an external store. Shows BlobRef component + BlobStore cleanup trait.

use minkowski::{BlobRef, BlobStore, Entity, World};
use std::collections::HashMap;

/// Simulated object store (in-memory HashMap standing in for S3/MinIO).
struct MemoryBlobStore {
    objects: HashMap<String, Vec<u8>>,
}

impl MemoryBlobStore {
    fn new() -> Self {
        Self { objects: HashMap::new() }
    }

    fn put(&mut self, key: &str, data: Vec<u8>) -> BlobRef {
        self.objects.insert(key.to_owned(), data);
        BlobRef::new(key)
    }

    fn get(&self, key: &str) -> Option<&[u8]> {
        self.objects.get(key).map(|v| v.as_slice())
    }
}

impl BlobStore for MemoryBlobStore {
    fn on_orphaned(&mut self, refs: &[&BlobRef]) {
        for r in refs {
            println!("  Deleting blob: {}", r.key());
            self.objects.remove(r.key());
        }
    }
}

fn main() {
    let mut world = World::new();
    let mut store = MemoryBlobStore::new();

    // Spawn entities with blob references
    println!("--- Spawning entities with blob refs ---");
    let mut entities: Vec<Entity> = Vec::new();
    for i in 0..5 {
        let key = format!("s3://bucket/object_{i}.bin");
        let data = vec![i as u8; 1024]; // 1KB "large" blob
        let blob_ref = store.put(&key, data);
        let e = world.spawn((blob_ref,));
        entities.push(e);
        println!("  Entity {i}: {key}");
    }
    println!("Store has {} objects", store.objects.len());

    // Despawn some entities — collect orphaned BlobRefs for cleanup
    println!("\n--- Despawning entities 1 and 3 ---");
    let mut orphaned = Vec::new();
    for &i in &[1usize, 3] {
        let e = entities[i];
        if let Some(blob_ref) = world.get::<BlobRef>(e) {
            orphaned.push(blob_ref.clone());
        }
        world.despawn(e);
    }

    // User calls cleanup — engine doesn't do this automatically
    println!("\n--- Running BlobStore cleanup ---");
    let borrowed: Vec<&BlobRef> = orphaned.iter().collect();
    store.on_orphaned(&borrowed);
    println!("Store now has {} objects", store.objects.len());

    // Verify remaining blobs are accessible
    println!("\n--- Remaining blob refs ---");
    world.query::<(&BlobRef,)>().for_each(|(r,)| {
        let size = store.get(r.key()).map_or(0, |b| b.len());
        println!("  {}: {} bytes", r.key(), size);
    });
}
```

**Step 2: Add to Cargo.toml if creating new example**

If creating `blob.rs`, add to `examples/Cargo.toml`:
```toml
[[example]]
name = "blob"
```

**Step 3: Run the example**

Run: `cargo run -p minkowski-examples --example blob --release`
Expected: Output shows spawning, despawning, cleanup, and remaining blobs.

**Step 4: Commit**

```
docs: add blob offloading example (BlobRef + MemoryBlobStore)
```

---

## Group B: Expiry & Retention Reducer (Feature 3)

Small feature. No dependency on the pool allocator.

### Task B1: `Expiry` component

**Files:**
- Create: `crates/minkowski/src/retention.rs`
- Modify: `crates/minkowski/src/lib.rs` (add module + re-exports)

**Step 1: Write Expiry and its tests**

In `crates/minkowski/src/retention.rs`:

```rust
//! Expiry component and retention reducer for automatic entity cleanup.
//!
//! `Expiry` marks entities for despawn at a target tick. `RetentionReducer`
//! scans for expired entities and batch-despawns them. The user controls
//! dispatch frequency — the engine never runs retention automatically.

use crate::tick::ChangeTick;

/// Marks an entity for despawn when the world tick reaches or exceeds this
/// value.
///
/// Set at spawn time via [`World::change_tick`](crate::World::change_tick):
///
/// ```ignore
/// let ttl_ticks = 1000;
/// let deadline = ChangeTick::from_raw(world.change_tick().to_raw() + ttl_ticks);
/// world.spawn((data, Expiry(deadline)));
/// ```
///
/// The tick is a monotonic u64 from change detection — **not** wall-clock time.
/// For time-based TTL, convert duration to ticks based on your simulation's
/// tick rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expiry(pub ChangeTick);

impl Expiry {
    /// Create an expiry at the given tick.
    pub fn at_tick(tick: ChangeTick) -> Self {
        Self(tick)
    }

    /// The deadline tick.
    pub fn deadline(&self) -> ChangeTick {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    #[test]
    fn expiry_is_a_component() {
        let mut world = World::new();
        let tick = world.change_tick();
        let e = world.spawn((Expiry::at_tick(tick),));
        assert_eq!(world.get::<Expiry>(e).unwrap().deadline(), tick);
    }

    #[test]
    fn expiry_round_trip() {
        let tick = ChangeTick::from_raw(42);
        let exp = Expiry::at_tick(tick);
        assert_eq!(exp.deadline().to_raw(), 42);
    }

    #[test]
    fn expiry_coexists_with_other_components() {
        let mut world = World::new();
        let tick = ChangeTick::from_raw(100);
        let e = world.spawn((42u32, Expiry::at_tick(tick)));
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
        assert_eq!(world.get::<Expiry>(e).unwrap().deadline().to_raw(), 100);
    }
}
```

**Step 2: Wire into lib.rs**

Add module declaration `mod retention;` and re-export `pub use retention::Expiry;`.

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- retention`
Expected: 3 tests PASS.

**Step 4: Commit**

```
feat: add Expiry component for tick-based entity TTL
```

---

### Task B2: `RetentionReducer` registration in `ReducerRegistry`

**Files:**
- Modify: `crates/minkowski/src/retention.rs` (add retention reducer logic)
- Modify: `crates/minkowski/src/reducer.rs` (add `retention()` registration method)
- Modify: `crates/minkowski/src/lib.rs` (re-export `RetentionReducerId` if needed)

**Step 1: Write the failing test in retention.rs**

Add to the test module in `retention.rs`:

```rust
#[test]
fn retention_despawns_expired_entities() {
    let mut world = World::new();
    let mut registry = crate::ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    // Spawn entities with varying expiry deadlines.
    // Current tick is low; set some deadlines in the past.
    let tick = world.change_tick();
    let past = ChangeTick::from_raw(0); // already expired
    let future = ChangeTick::from_raw(tick.to_raw() + 1_000_000);

    let e_expired_1 = world.spawn((Expiry::at_tick(past), 1u32));
    let e_expired_2 = world.spawn((Expiry::at_tick(past), 2u32));
    let e_alive = world.spawn((Expiry::at_tick(future), 3u32));
    let e_no_expiry = world.spawn((4u32,)); // no Expiry component

    // Run retention reducer
    registry.run(&mut world, retention_id, ()).unwrap();

    // Expired entities are gone
    assert!(!world.is_alive(e_expired_1));
    assert!(!world.is_alive(e_expired_2));
    // Living entities remain
    assert!(world.is_alive(e_alive));
    assert!(world.is_alive(e_no_expiry));
    assert_eq!(*world.get::<u32>(e_alive).unwrap(), 3);
    assert_eq!(*world.get::<u32>(e_no_expiry).unwrap(), 4);
}

#[test]
fn retention_is_idempotent() {
    let mut world = World::new();
    let mut registry = crate::ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    let past = ChangeTick::from_raw(0);
    world.spawn((Expiry::at_tick(past),));

    // Run twice — second run should be a no-op
    registry.run(&mut world, retention_id, ()).unwrap();
    registry.run(&mut world, retention_id, ()).unwrap();

    // No entities left with Expiry
    let mut count = 0;
    world.query::<(&Expiry,)>().for_each(|_| count += 1);
    assert_eq!(count, 0);
}

#[test]
fn retention_noop_when_nothing_expired() {
    let mut world = World::new();
    let mut registry = crate::ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    let future = ChangeTick::from_raw(u64::MAX);
    let e = world.spawn((Expiry::at_tick(future),));

    registry.run(&mut world, retention_id, ()).unwrap();

    assert!(world.is_alive(e));
}

#[test]
fn retention_access_declares_despawns() {
    let mut world = World::new();
    let mut registry = crate::ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    let access = registry.query_access(retention_id).unwrap();
    assert!(access.despawns);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- retention`
Expected: FAIL — `retention()` method doesn't exist yet on `ReducerRegistry`.

**Step 3: Implement the retention method on ReducerRegistry**

In `crates/minkowski/src/reducer.rs`, add a `retention` method. This follows
the `register_query` pattern but with a fixed built-in closure:

```rust
/// Register the built-in retention reducer.
///
/// The retention reducer queries all entities with an [`Expiry`] component
/// and batch-despawns those whose deadline tick has been reached.
///
/// Returns a `QueryReducerId` that can be passed to
/// [`run()`](Self::run) at whatever frequency the user chooses.
pub fn retention(&mut self, world: &mut World) -> QueryReducerId {
    use crate::retention::Expiry;

    self.register_query::<(Entity, &Expiry), (), _>(
        world,
        "__retention",
        |mut query, ()| {
            let current = query.change_tick();
            let mut to_despawn: Vec<crate::Entity> = Vec::new();
            query.for_each(|(entity, expiry)| {
                if expiry.deadline().to_raw() <= current.to_raw() {
                    to_despawn.push(entity);
                }
            });
            query.despawn_batch(&to_despawn);
        },
    )
    .expect("retention reducer registration should not fail")
}
```

**Important:** The retention reducer needs to both read entities and despawn
them. It is a `QueryMut`-style reducer because it calls `despawn_batch`.
The exact handle types depend on what `QueryMut` exposes — if `QueryMut`
doesn't expose `change_tick()` and `despawn_batch()`, you may need to use
the lower-level `ScheduledAdapter` pattern instead:

```rust
// Alternative: use raw ScheduledAdapter if QueryMut API is insufficient
let adapter: ScheduledAdapter = Box::new(move |world: &mut World, _args: &dyn Any| {
    let current = world.change_tick();
    let mut to_despawn: Vec<Entity> = Vec::new();
    world.query::<(Entity, &Expiry)>().for_each(|(entity, expiry)| {
        if expiry.deadline().to_raw() <= current.to_raw() {
            to_despawn.push(entity);
        }
    });
    world.despawn_batch(&to_despawn);
});
```

The implementation must declare `Access` with reads on `Expiry` and
`despawns: true` so the scheduler can detect conflicts.

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib -- retention`
Expected: All retention tests PASS.

**Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

**Step 6: Commit**

```
feat: add RetentionReducer for automatic expiry-based entity cleanup

Built-in scheduled reducer that batch-despawns entities whose Expiry
deadline has been reached. User controls dispatch frequency.
```

---

### Task B3: Retention example

**Files:**
- Modify: existing example or create `examples/examples/retention.rs`

**Step 1: Write example**

```rust
//! Retention: automatic entity cleanup via Expiry + RetentionReducer.
//!
//! Spawns entities with varying TTLs. Each "frame" advances the world tick
//! by running a dummy mutation, then runs the retention reducer to despawn
//! expired entities.

use minkowski::{ChangeTick, Expiry, ReducerRegistry, World};

fn main() {
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();
    let retention_id = registry.retention(&mut world);

    // Spawn entities with different TTLs (in ticks from now)
    let ttls = [10, 50, 100, 200, 500];
    let base_tick = world.change_tick().to_raw();

    println!("--- Spawning {} entities with TTLs: {:?} ---", ttls.len(), ttls);
    for &ttl in &ttls {
        let deadline = ChangeTick::from_raw(base_tick + ttl);
        world.spawn((Expiry::at_tick(deadline), ttl as u32));
    }

    // Simulate frames — each iteration advances the tick via a query
    for frame in 0..6 {
        // Advance tick by doing work (queries advance ticks)
        for _ in 0..100 {
            world.query::<(&u32,)>().for_each(|_| {});
        }

        let before = world.stats().entity_count;
        registry.run(&mut world, retention_id, ()).unwrap();
        let after = world.stats().entity_count;
        let tick = world.change_tick().to_raw();

        println!(
            "Frame {frame}: tick={tick}, entities {before} -> {after} ({} despawned)",
            before - after
        );
    }

    println!("\n--- Final entity count: {} ---", world.stats().entity_count);
}
```

**Step 2: Add to examples Cargo.toml**

```toml
[[example]]
name = "retention"
```

**Step 3: Run**

Run: `cargo run -p minkowski-examples --example retention --release`
Expected: Shows entity count decreasing as ticks advance past deadlines.

**Step 4: Commit**

```
docs: add retention example (Expiry + RetentionReducer)
```

---

## Group C: Slab Pool Allocator (Feature 1)

The largest feature. Touches internal allocation infrastructure. Each task is
independently testable.

### Task C1: `PoolAllocator` trait + `SystemAllocator` + `PoolExhausted` error

**Files:**
- Create: `crates/minkowski/src/pool.rs`
- Modify: `crates/minkowski/src/lib.rs` (add module + re-exports)

**Step 1: Define the trait and SystemAllocator**

In `crates/minkowski/src/pool.rs`:

```rust
//! Memory pool allocator trait and implementations.
//!
//! The pool allocator is the single backing allocator for all internal
//! data structures (BlobVec, Arena, entity tables, sparse pages).
//! Two implementations: `SystemAllocator` (current behavior, default)
//! and `SlabPool` (TigerBeetle-style fixed budget with mmap).

use std::alloc::Layout;
use std::ptr::NonNull;
use std::fmt;
use std::sync::Arc;

/// Error returned when the memory pool is exhausted.
#[derive(Debug, Clone)]
pub struct PoolExhausted {
    pub requested: Layout,
}

impl fmt::Display for PoolExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "memory pool exhausted: requested {} bytes (align {})",
            self.requested.size(),
            self.requested.align()
        )
    }
}

impl std::error::Error for PoolExhausted {}

/// Backing allocator for all internal ECS data structures.
///
/// # Safety
///
/// Implementations must return properly aligned, non-overlapping memory
/// regions. `deallocate` must only be called with pointers and layouts
/// previously returned by `allocate`.
pub trait PoolAllocator: Send + Sync {
    /// Allocate a block satisfying `layout`.
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted>;

    /// Return a block to the pool.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to `allocate` with a
    /// compatible layout. The caller must not use `ptr` after this call.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Total capacity in bytes, if bounded. `None` for unbounded allocators.
    fn capacity(&self) -> Option<usize> { None }

    /// Bytes currently allocated. `None` if not tracked.
    fn used(&self) -> Option<usize> { None }
}

/// Default allocator — delegates to `std::alloc`. Unbounded, panics on OOM.
/// This is the allocator used by `World::new()`.
pub(crate) struct SystemAllocator;

impl PoolAllocator for SystemAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            // Zero-size: return dangling aligned pointer
            return Ok(NonNull::new(layout.align() as *mut u8)
                .expect("alignment is non-zero"));
        }
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).ok_or_else(|| {
            // Preserve current behavior: panic on OOM for system allocator
            std::alloc::handle_alloc_error(layout)
        })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

/// Shared handle to a pool allocator. Cheaply cloneable.
pub(crate) type SharedPool = Arc<dyn PoolAllocator>;

/// Create the default (system) allocator as a shared pool.
pub(crate) fn default_pool() -> SharedPool {
    Arc::new(SystemAllocator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_allocator_allocate_and_deallocate() {
        let alloc = SystemAllocator;
        let layout = Layout::from_size_align(256, 64).unwrap();
        let ptr = alloc.allocate(layout).unwrap();
        // Write to verify it's valid memory
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0xAB, 256) };
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn system_allocator_zero_size() {
        let alloc = SystemAllocator;
        let layout = Layout::from_size_align(0, 1).unwrap();
        let ptr = alloc.allocate(layout).unwrap();
        // Should not crash on dealloc of zero-size
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn system_allocator_capacity_is_none() {
        let alloc = SystemAllocator;
        assert!(alloc.capacity().is_none());
        assert!(alloc.used().is_none());
    }

    #[test]
    fn shared_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedPool>();
    }

    #[test]
    fn pool_exhausted_display() {
        let err = PoolExhausted {
            requested: Layout::from_size_align(1024, 64).unwrap(),
        };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("64"));
    }
}
```

**Step 2: Wire into lib.rs**

Add `mod pool;` and `pub use pool::PoolExhausted;`.

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- pool`
Expected: 5 tests PASS.

**Step 4: Commit**

```
feat: add PoolAllocator trait and SystemAllocator default impl
```

---

### Task C2: Thread BlobVec through `SharedPool`

**Files:**
- Modify: `crates/minkowski/src/storage/blob_vec.rs`
- Modify: `crates/minkowski/src/storage/archetype.rs`

This is the core refactor. BlobVec currently calls `std::alloc` directly.
After this task, it uses the pool.

**Step 1: Add pool field to BlobVec**

Modify the struct to hold a `SharedPool`:

```rust
pub(crate) struct BlobVec {
    pub(crate) item_layout: Layout,
    pub(crate) drop_fn: Option<unsafe fn(*mut u8)>,
    data: NonNull<u8>,
    len: usize,
    capacity: usize,
    pub(crate) changed_tick: Tick,
    pool: SharedPool,
}
```

**Step 2: Replace all `std::alloc::alloc` calls with `pool.allocate()`**

In `BlobVec::new()`:
```rust
// Before:
let ptr = unsafe { alloc::alloc(layout) };
let data = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));

// After:
let data = pool.allocate(layout)
    .unwrap_or_else(|_| alloc::handle_alloc_error(layout));
```

In `BlobVec::grow()`:
```rust
// Before:
let new_ptr = unsafe { alloc::alloc(new_layout) };
let new_data = NonNull::new(new_ptr)
    .unwrap_or_else(|| alloc::handle_alloc_error(new_layout));
// ... copy ...
alloc::dealloc(self.data.as_ptr(), old_layout);

// After:
let new_data = self.pool.allocate(new_layout)
    .unwrap_or_else(|_| alloc::handle_alloc_error(new_layout));
// ... copy ...
unsafe { self.pool.deallocate(self.data, old_layout) };
```

In `Drop for BlobVec`:
```rust
// Before:
alloc::dealloc(self.data.as_ptr(), layout);

// After:
unsafe { self.pool.deallocate(self.data, layout) };
```

**Step 3: Add `try_grow()` variant**

Add a fallible grow method that returns `Result` instead of panicking:

```rust
pub(crate) fn try_grow(&mut self) -> Result<(), PoolExhausted> {
    // Same logic as grow() but propagates PoolExhausted instead of panicking
    let new_capacity = if self.capacity == 0 { 4 } else { self.capacity * 2 };
    let new_layout = Layout::from_size_align(
        self.item_layout.size() * new_capacity,
        self.item_layout.align().max(MIN_COLUMN_ALIGN),
    ).unwrap();
    let new_data = self.pool.allocate(new_layout)?;
    // ... copy + dealloc old ...
    Ok(())
}
```

**Step 4: Update BlobVec::new() to accept SharedPool**

```rust
pub(crate) fn new(
    item_layout: Layout,
    drop_fn: Option<unsafe fn(*mut u8)>,
    capacity: usize,
    pool: SharedPool,
) -> Self { ... }
```

**Step 5: Update all BlobVec::new() call sites**

Search for `BlobVec::new(` in the codebase. Each call site must pass the pool.
Key locations:
- `archetype.rs`: `Archetype::new()` creates BlobVecs for columns — needs pool
- `sparse.rs`: `PagedSparseSet::new()` creates a BlobVec — needs pool
- `blob_vec.rs` tests: `bv_for::<T>()` helper needs pool

For `archetype.rs`, thread the pool through:
```rust
pub(crate) fn new(
    id: ArchetypeId,
    sorted_component_ids: &[ComponentId],
    registry: &ComponentRegistry,
    pool: SharedPool,
) -> Self { ... }
```

For `Archetypes::get_or_create`, pass pool from World.

**Step 6: Run existing tests**

Run: `cargo test -p minkowski --lib`
Expected: ALL existing tests pass (no behavior change — still using
SystemAllocator via `default_pool()`).

**Step 7: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

**Step 8: Commit**

```
refactor: thread SharedPool through BlobVec allocation paths

BlobVec now uses PoolAllocator trait instead of direct std::alloc calls.
No behavior change — SystemAllocator preserves current semantics.
```

---

### Task C3: Thread Arena through `SharedPool`

**Files:**
- Modify: `crates/minkowski/src/changeset.rs`

**Step 1: Add pool field to Arena**

```rust
pub(crate) struct Arena {
    data: NonNull<u8>,
    len: usize,
    capacity: usize,
    pool: SharedPool,
}
```

**Step 2: Replace alloc/dealloc calls**

Same pattern as BlobVec — replace `alloc::alloc`/`alloc::dealloc` with
`pool.allocate()`/`pool.deallocate()`.

**Step 3: Update Arena::new() to accept SharedPool**

```rust
pub(crate) fn new(pool: SharedPool) -> Self { ... }
```

**Step 4: Update EnumChangeSet to thread pool**

`EnumChangeSet::new()` creates an Arena — needs pool parameter. This means
`EnumChangeSet::new()` needs a pool argument, or it uses `default_pool()`.

Since `EnumChangeSet` is public API, keep `EnumChangeSet::new()` using
`default_pool()` for backwards compat. Add `EnumChangeSet::new_in(pool)`.

**Step 5: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All pass.

**Step 6: Commit**

```
refactor: thread SharedPool through Arena allocation paths
```

---

### Task C4: Thread pool through World + add WorldBuilder

**Files:**
- Modify: `crates/minkowski/src/world.rs`
- Modify: `crates/minkowski/src/lib.rs` (re-export `WorldBuilder`, `HugePages`)

**Step 1: Add pool to World struct**

```rust
pub struct World {
    // ... existing fields ...
    pub(crate) pool: SharedPool,
}
```

**Step 2: Update World::new() to use default_pool()**

```rust
pub fn new() -> Self {
    let pool = default_pool();
    Self {
        // ... pass pool to Archetypes, EntityAllocator, etc. ...
        pool,
    }
}
```

**Step 3: Add WorldBuilder**

```rust
/// Builder for configuring World memory and allocation strategy.
pub struct WorldBuilder {
    memory_budget: Option<usize>,
    hugepages: HugePages,
}

/// Hugepage configuration for the memory pool.
#[derive(Clone, Copy, Debug, Default)]
pub enum HugePages {
    /// Attempt 2MB hugepages, fall back to 4KB pages silently.
    #[default]
    Try,
    /// Require hugepages — fail if unavailable.
    Require,
    /// Use regular 4KB pages only.
    Off,
}

impl WorldBuilder {
    pub fn new() -> Self {
        Self {
            memory_budget: None,
            hugepages: HugePages::default(),
        }
    }

    /// Set the total memory budget in bytes. Activates the slab pool allocator.
    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = Some(bytes);
        self
    }

    /// Configure hugepage usage (only effective with a memory budget).
    pub fn hugepages(mut self, hp: HugePages) -> Self {
        self.hugepages = hp;
        self
    }

    /// Build the World. Returns `Err` if the memory pool cannot be allocated
    /// (insufficient RAM, hugepages unavailable with `Require`, etc.).
    pub fn build(self) -> Result<World, PoolExhausted> {
        let pool: SharedPool = match self.memory_budget {
            Some(bytes) => {
                // SlabPool::new() — implemented in Task C5
                Arc::new(SlabPool::new(bytes, self.hugepages)?)
            }
            None => default_pool(),
        };
        Ok(World::new_with_pool(pool))
    }
}

impl World {
    pub fn builder() -> WorldBuilder {
        WorldBuilder::new()
    }

    pub(crate) fn new_with_pool(pool: SharedPool) -> Self {
        Self {
            // ... same as new() but uses provided pool ...
        }
    }
}
```

**Step 4: Update WorldStats**

```rust
pub struct WorldStats {
    // ... existing fields ...
    pub pool_capacity: Option<usize>,
    pub pool_used: Option<usize>,
}
```

**Step 5: Write builder tests**

```rust
#[test]
fn world_builder_default_is_system_allocator() {
    let world = World::builder().build().unwrap();
    let stats = world.stats();
    assert!(stats.pool_capacity.is_none());
}

#[test]
fn world_new_still_works() {
    let world = World::new();
    let stats = world.stats();
    assert!(stats.pool_capacity.is_none());
}
```

**Step 6: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All pass.

**Step 7: Commit**

```
feat: add WorldBuilder with memory_budget and hugepages config
```

---

### Task C5: MmapRegion + SlabPool implementation

**Files:**
- Create: `crates/minkowski/src/pool/mmap.rs` (or keep in `pool.rs`)
- Modify: `crates/minkowski/Cargo.toml` (add `memmap2` dependency)

This is the core pool implementation. The most complex single task.

**Step 1: Add memmap2 dependency**

`memmap2` is already a workspace dependency (used by `minkowski-persist`).
Add it to `crates/minkowski/Cargo.toml`:
```toml
[dependencies]
memmap2 = "0.9"
```

**Step 2: Implement MmapRegion using memmap2**

```rust
use memmap2::MmapOptions;

/// Memory region backed by mmap with optional hugepage support.
/// Uses `memmap2` for cross-platform mmap — same crate used by minkowski-persist.
struct MmapRegion {
    mmap: memmap2::MmapMut,
    huge: bool,
}

impl MmapRegion {
    fn new(size: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        let layout = Layout::from_size_align(size, 1).unwrap();

        // Try hugepages first if requested
        if matches!(hugepages, HugePages::Try | HugePages::Require) {
            let result = MmapOptions::new()
                .len(size)
                .populate()    // MAP_POPULATE — pre-fault all pages
                .huge(Some(21)) // MAP_HUGETLB with 2MB pages (2^21)
                .map_anon();   // MAP_ANONYMOUS | MAP_PRIVATE

            match result {
                Ok(mmap) => return Ok(Self { mmap, huge: true }),
                Err(_) if matches!(hugepages, HugePages::Try) => {
                    // Fall through to regular pages
                }
                Err(_) => return Err(PoolExhausted { requested: layout }),
            }
        }

        // Regular pages with MAP_POPULATE
        let mmap = MmapOptions::new()
            .len(size)
            .populate()
            .map_anon()
            .map_err(|_| PoolExhausted { requested: layout })?;

        Ok(Self { mmap, huge: false })
    }

    fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.mmap.len()
    }
}
// Drop is handled by memmap2::MmapMut (calls munmap automatically)
```

**Step 3: Implement SlabPool**

```rust
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

const NUM_SIZE_CLASSES: usize = 6;
const SIZE_CLASSES: [usize; NUM_SIZE_CLASSES] = [64, 256, 1024, 4096, 65536, 1_048_576];

/// TigerBeetle-style slab allocator backed by a single mmap'd region.
pub struct SlabPool {
    region: MmapRegion,
    free_lists: [AtomicPtr<FreeBlock>; NUM_SIZE_CLASSES],
    used_bytes: AtomicUsize,
}

#[repr(C)]
struct FreeBlock {
    next: *mut FreeBlock,
}

impl SlabPool {
    pub fn new(size: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        let region = MmapRegion::new(size, hugepages)?;

        let free_lists = std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut()));

        let pool = Self {
            region,
            free_lists,
            used_bytes: AtomicUsize::new(0),
        };

        // Partition the region into size-classed blocks and populate free lists.
        // Strategy: divide proportionally based on expected ECS usage patterns.
        pool.partition_region(size);

        Ok(pool)
    }

    fn partition_region(&self, total: usize) {
        // Proportional split: most memory goes to 4KB and 64KB classes
        // (column storage). Exact ratios are tunable.
        let proportions = [1, 2, 4, 20, 40, 33]; // percentages
        let mut offset = 0;
        let base = self.region.ptr.as_ptr();

        for (class_idx, &proportion) in proportions.iter().enumerate() {
            let class_budget = total * proportion / 100;
            let block_size = SIZE_CLASSES[class_idx];
            let block_count = class_budget / block_size;

            for i in 0..block_count {
                let block_ptr = unsafe { base.add(offset + i * block_size) } as *mut FreeBlock;
                // Push onto free list (single-threaded during init)
                unsafe {
                    (*block_ptr).next = self.free_lists[class_idx]
                        .load(Ordering::Relaxed);
                }
                self.free_lists[class_idx].store(block_ptr, Ordering::Relaxed);
            }
            offset += block_count * block_size;
        }
    }

    fn size_class_for(layout: Layout) -> Option<usize> {
        let size = layout.size().max(layout.align());
        SIZE_CLASSES.iter().position(|&s| s >= size)
    }
}

impl PoolAllocator for SlabPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8).unwrap());
        }

        let class = Self::size_class_for(layout)
            .ok_or(PoolExhausted { requested: layout })?;

        // Lock-free pop from free list
        loop {
            let head = self.free_lists[class].load(Ordering::Acquire);
            if head.is_null() {
                return Err(PoolExhausted { requested: layout });
            }
            let next = unsafe { (*head).next };
            if self.free_lists[class]
                .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.used_bytes.fetch_add(
                    SIZE_CLASSES[class], Ordering::Relaxed
                );
                return Ok(NonNull::new(head.cast()).unwrap());
            }
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        let class = match Self::size_class_for(layout) {
            Some(c) => c,
            None => return, // oversized — cannot return to free list
        };

        // Lock-free push onto free list
        let block = ptr.as_ptr() as *mut FreeBlock;
        loop {
            let head = self.free_lists[class].load(Ordering::Acquire);
            unsafe { (*block).next = head };
            if self.free_lists[class]
                .compare_exchange_weak(block, head, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.used_bytes.fetch_sub(
                    SIZE_CLASSES[class], Ordering::Relaxed
                );
                return;
            }
        }
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.region.size)
    }

    fn used(&self) -> Option<usize> {
        Some(self.used_bytes.load(Ordering::Relaxed))
    }
}

// Safety: All mutable state is behind atomics.
unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}
```

**Step 4: Write SlabPool tests**

```rust
#[test]
fn slab_pool_allocate_and_deallocate() {
    let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap(); // 4MB
    let layout = Layout::from_size_align(64, 64).unwrap();
    let ptr = pool.allocate(layout).unwrap();
    assert!(pool.used().unwrap() > 0);
    unsafe { pool.deallocate(ptr, layout) };
}

#[test]
fn slab_pool_returns_error_on_exhaustion() {
    let pool = SlabPool::new(64 * 1024, HugePages::Off).unwrap(); // 64KB — tiny
    let layout = Layout::from_size_align(64, 64).unwrap();

    // Allocate until exhausted
    let mut ptrs = Vec::new();
    loop {
        match pool.allocate(layout) {
            Ok(ptr) => ptrs.push(ptr),
            Err(_) => break,
        }
    }
    assert!(!ptrs.is_empty());

    // Deallocate one and re-allocate succeeds
    unsafe { pool.deallocate(ptrs.pop().unwrap(), layout) };
    assert!(pool.allocate(layout).is_ok());
}

#[test]
fn slab_pool_observability() {
    let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
    assert_eq!(pool.capacity(), Some(1024 * 1024));
    assert_eq!(pool.used(), Some(0));

    let layout = Layout::from_size_align(64, 64).unwrap();
    let ptr = pool.allocate(layout).unwrap();
    assert!(pool.used().unwrap() > 0);
    unsafe { pool.deallocate(ptr, layout) };
}

#[test]
fn slab_pool_concurrent_allocate() {
    use std::sync::Arc;
    let pool = Arc::new(
        SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap()
    );
    let layout = Layout::from_size_align(64, 64).unwrap();

    std::thread::scope(|s| {
        for _ in 0..4 {
            let pool = Arc::clone(&pool);
            s.spawn(move || {
                let mut ptrs = Vec::new();
                for _ in 0..100 {
                    if let Ok(ptr) = pool.allocate(layout) {
                        ptrs.push(ptr);
                    }
                }
                for ptr in ptrs {
                    unsafe { pool.deallocate(ptr, layout) };
                }
            });
        }
    });
}
```

**Step 5: Run tests**

Run: `cargo test -p minkowski --lib -- pool`
Expected: All pool tests pass including concurrent test.

**Step 6: Commit**

```
feat: add SlabPool with mmap backing + lock-free free lists
```

---

### Task C6: `try_spawn` and `try_insert` error propagation

**Files:**
- Modify: `crates/minkowski/src/world.rs`
- Modify: `crates/minkowski/src/storage/blob_vec.rs` (add `try_push`)

**Step 1: Add try_push to BlobVec**

```rust
/// Push a value, returning Err if the pool is exhausted.
pub(crate) unsafe fn try_push(
    &mut self,
    value: *const u8,
) -> Result<(), PoolExhausted> {
    if self.len == self.capacity {
        self.try_grow()?;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            value,
            self.data.as_ptr().add(self.len * self.item_layout.size()),
            self.item_layout.size(),
        );
    }
    self.len += 1;
    Ok(())
}
```

**Step 2: Add try_spawn to World**

```rust
/// Spawn an entity, returning `Err(PoolExhausted)` if the memory pool
/// cannot accommodate the new entity.
pub fn try_spawn<B: Bundle>(&mut self, bundle: B) -> Result<Entity, PoolExhausted> {
    // Same logic as spawn() but using try_ allocation methods
    // and propagating PoolExhausted
    // ...
}
```

**Step 3: Add try_insert to World**

Same pattern — wraps the archetype migration path with fallible allocation.

**Step 4: Write tests**

```rust
#[test]
fn try_spawn_succeeds_with_system_allocator() {
    let mut world = World::new();
    let e = world.try_spawn((42u32,)).unwrap();
    assert_eq!(*world.get::<u32>(e).unwrap(), 42);
}

// Integration test with SlabPool — test exhaustion:
#[test]
fn try_spawn_returns_error_on_pool_exhaustion() {
    let mut world = World::builder()
        .memory_budget(64 * 1024) // tiny pool
        .build()
        .unwrap();

    let mut count = 0;
    loop {
        match world.try_spawn((0u64, 1u64, 2u64, 3u64)) {
            Ok(_) => count += 1,
            Err(_) => break,
        }
    }
    assert!(count > 0, "should have spawned at least one entity");
}
```

**Step 5: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All pass.

**Step 6: Commit**

```
feat: add try_spawn and try_insert for fallible allocation
```

---

### Task C7: Thread pool through remaining allocators

**Files:**
- Modify: `crates/minkowski/src/storage/sparse.rs`
- Modify: `crates/minkowski/src/entity.rs` (if Vec fields need pool)

**Step 1: Thread pool through PagedSparseSet**

`PagedSparseSet` uses `Box<[u32; PAGE_SIZE]>` for pages and `BlobVec` for
dense values. The BlobVec already uses the pool (from Task C2). The Box
pages are small (16KB) and infrequent — they can remain on the system
allocator for now, or be converted to pool allocation if strict zero-malloc
is required.

For strict TigerBeetle compliance, replace `Box::new(...)` with a pool
allocation:

```rust
fn alloc_page(pool: &SharedPool) -> Result<NonNull<[u32; PAGE_SIZE]>, PoolExhausted> {
    let layout = Layout::new::<[u32; PAGE_SIZE]>();
    let ptr = pool.allocate(layout)?;
    // Initialize to EMPTY
    unsafe {
        let page = ptr.as_ptr() as *mut [u32; PAGE_SIZE];
        (*page) = [EMPTY; PAGE_SIZE];
        Ok(NonNull::new_unchecked(page))
    }
}
```

**Step 2: Thread pool through EntityAllocator Vecs**

The `generations: Vec<u32>` and `free_list: Vec<u32>` use standard Vec.
For strict zero-malloc, these would need a pool-backed Vec. This is
significant additional work — consider whether to do this in v1 or defer.

Recommendation: **defer to v2**. These Vecs grow slowly (one entry per unique
entity ever created) and are cold-path. The BlobVec columns are the hot path
and account for >95% of memory. Document the deferral.

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib`

**Step 4: Commit**

```
refactor: thread pool through sparse page allocation
```

---

### Task C8: Integration test + pool mode example

**Files:**
- Create: `examples/examples/pool.rs`
- Add integration tests to `crates/minkowski/src/pool.rs`

**Step 1: Write integration test**

```rust
#[test]
fn world_builder_with_slab_pool() {
    let mut world = World::builder()
        .memory_budget(8 * 1024 * 1024) // 8MB
        .hugepages(HugePages::Off) // no hugepages in CI
        .build()
        .unwrap();

    // Spawn entities and verify pool stats
    for i in 0..1000 {
        world.spawn((i as u32, i as f64));
    }

    let stats = world.stats();
    assert_eq!(stats.entity_count, 1000);
    assert!(stats.pool_capacity.is_some());
    assert!(stats.pool_used.unwrap() > 0);

    // Despawn and verify memory is reclaimed
    // (BlobVec doesn't shrink, but slab blocks return to free lists)
}
```

**Step 2: Write pool example**

```rust
//! Memory pool: TigerBeetle-style pre-allocated World with bounded memory.

use minkowski::{HugePages, World};

fn main() {
    println!("--- Allocating World with 16MB pool ---");
    let mut world = World::builder()
        .memory_budget(16 * 1024 * 1024)
        .hugepages(HugePages::Try)
        .build()
        .expect("failed to allocate memory pool");

    let stats = world.stats();
    println!(
        "Pool: {:.1}MB capacity, {:.1}KB used",
        stats.pool_capacity.unwrap() as f64 / 1_048_576.0,
        stats.pool_used.unwrap() as f64 / 1024.0,
    );

    // Spawn entities until pool approaches capacity
    println!("\n--- Spawning entities ---");
    let mut count = 0u32;
    loop {
        match world.try_spawn((count, count as f64)) {
            Ok(_) => count += 1,
            Err(e) => {
                println!("Pool exhausted after {count} entities: {e}");
                break;
            }
        }
        if count % 10_000 == 0 {
            let s = world.stats();
            println!(
                "  {count} entities, {:.1}KB used / {:.1}MB total",
                s.pool_used.unwrap() as f64 / 1024.0,
                s.pool_capacity.unwrap() as f64 / 1_048_576.0,
            );
        }
    }

    let stats = world.stats();
    println!(
        "\nFinal: {} entities, {:.1}MB used / {:.1}MB total",
        stats.entity_count,
        stats.pool_used.unwrap() as f64 / 1_048_576.0,
        stats.pool_capacity.unwrap() as f64 / 1_048_576.0,
    );
}
```

**Step 3: Run example**

Run: `cargo run -p minkowski-examples --example pool --release`

**Step 4: Commit**

```
docs: add pool allocator example and integration tests
```

---

### Task C9: Final CI verification

**Step 1: Full test suite**

Run: `cargo test -p minkowski`

**Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

**Step 3: Miri (if time permits)**

Run: `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- pool`

Note: Miri may not support mmap. If it doesn't, gate SlabPool tests
behind `#[cfg(not(miri))]` and test the `SystemAllocator` path under Miri.

**Step 4: Benchmarks**

Run: `cargo bench -p minkowski-bench`
Compare before/after — the `SystemAllocator` path should show zero regression.

---

## Task Dependencies

```
Group A (Blob):     A1 → A2
Group B (Retention): B1 → B2 → B3
Group C (Pool):     C1 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9
```

Groups A, B, and C are fully independent. Recommended execution order:
**A first** (smallest, highest confidence), then **B** (small, validates
reducer infrastructure), then **C** (largest, most risk).

Each group is a separate PR.

---

## Key risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| mmap not available on all platforms | Build failure on WASM | `memmap2` handles Linux/macOS/Windows; fallback `HeapPool` for WASM |
| Miri doesn't support mmap | Can't verify pool under Miri | Gate mmap tests, test allocator trait under Miri |
| SlabPool fragmentation | Pool reports free space but can't allocate | Size-class proportions tunable; document tradeoffs |
| BlobVec growth doubles into wrong size class | Wastes pool blocks | BlobVec capacity capped to size-class boundaries |
| `try_spawn` changes World API surface | Breaking for trait impls | `try_` is additive, `spawn()` unchanged |
| Retention reducer conflicts with user reducers | Unexpected scheduling | Access declares reads(Expiry) + despawns; scheduler handles |
