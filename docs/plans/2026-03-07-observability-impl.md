# Observability Companion Crate — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `WorldStats`/`WalStats` facade structs to core crates and a `minkowski-observe` companion crate that captures, diffs, and displays engine metrics.

**Architecture:** Two facade methods (`world.stats()`, `wal.stats()`) copy internal values into plain `Copy` structs. The `minkowski-observe` crate composes these into `MetricsSnapshot`, computes `MetricsDiff` from consecutive snapshots, and provides `Display` impls for human-readable output. Zero changes to engine semantics.

**Tech Stack:** Rust, no new external dependencies for observe crate. Core crate changes are additive (two structs, two methods).

---

### Task 1: `WorldStats` struct and `world.stats()`

**Files:**
- Modify: `crates/minkowski/src/world.rs` (add struct + method after the Introspection section ~line 1095)
- Modify: `crates/minkowski/src/lib.rs` (re-export `WorldStats`)

**Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `world.rs`:

```rust
#[test]
fn world_stats_reflects_state() {
    let mut world = World::new();
    let s0 = world.stats();
    assert_eq!(s0.entity_count, 0);
    assert_eq!(s0.archetype_count, 1); // empty archetype always exists
    assert_eq!(s0.component_count, 0);
    assert_eq!(s0.free_list_len, 0);
    assert_eq!(s0.query_cache_len, 0);
    assert_eq!(s0.current_tick, 0);

    #[derive(Clone, Copy)]
    struct Pos { x: f32, y: f32 }

    let e = world.spawn((Pos { x: 1.0, y: 2.0 },));
    let s1 = world.stats();
    assert_eq!(s1.entity_count, 1);
    assert!(s1.archetype_count >= 2);
    assert!(s1.component_count >= 1);
    assert!(s1.current_tick > s0.current_tick);

    world.despawn(e);
    let s2 = world.stats();
    assert_eq!(s2.entity_count, 0);
    assert_eq!(s2.free_list_len, 1);
}

#[test]
fn world_stats_query_cache_len() {
    let mut world = World::new();
    #[derive(Clone, Copy)]
    struct A(u32);
    world.spawn((A(1),));

    assert_eq!(world.stats().query_cache_len, 0);
    let _: Vec<_> = world.query::<(&A,)>().collect();
    assert_eq!(world.stats().query_cache_len, 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski --lib -- world_stats`
Expected: FAIL — `stats()` method and `WorldStats` struct don't exist

**Step 3: Write minimal implementation**

Add to `crates/minkowski/src/world.rs`, in the Introspection section (after `entity_count()`, around line 1104):

```rust
/// Read-only snapshot of engine statistics. Plain data struct — no references
/// to internal state, safe to store, serialize, or send across threads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldStats {
    pub entity_count: usize,
    pub archetype_count: usize,
    pub component_count: usize,
    pub free_list_len: usize,
    pub query_cache_len: usize,
    pub current_tick: u64,
}

impl World {
    // ... (inside the existing impl block, in the Introspection section)

    /// Snapshot of engine statistics for observability.
    pub fn stats(&self) -> WorldStats {
        WorldStats {
            entity_count: self.entity_count(),
            archetype_count: self.archetypes.archetypes.len(),
            component_count: self.components.len(),
            free_list_len: self.entities.free_list.len(),
            query_cache_len: self.query_cache.len(),
            current_tick: self.current_tick.0,
        }
    }
}
```

Add to `crates/minkowski/src/lib.rs` re-exports:

```rust
pub use world::{World, WorldStats};
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- world_stats`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs crates/minkowski/src/lib.rs
git commit -m "feat: add WorldStats facade and world.stats() method"
```

---

### Task 2: `WalStats` struct and `wal.stats()`

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs` (add struct + method near the existing public accessors ~line 349)
- Modify: `crates/minkowski-persist/src/lib.rs` (re-export `WalStats`)

**Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `wal.rs`:

```rust
#[test]
fn wal_stats_reflects_state() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(1024),
    };
    let mut wal = Wal::create(&wal_path, &codecs, config).unwrap();

    let s0 = wal.stats();
    assert_eq!(s0.next_seq, 0);
    assert_eq!(s0.segment_count, 1);
    assert_eq!(s0.oldest_seq, Some(0));
    assert_eq!(s0.bytes_since_checkpoint, 0);
    assert_eq!(s0.last_checkpoint_seq, None);
    assert!(!s0.checkpoint_needed);

    let e = world.alloc_entity();
    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
    wal.append(&cs, &codecs).unwrap();
    cs.apply(&mut world);

    let s1 = wal.stats();
    assert_eq!(s1.next_seq, 1);
    assert!(s1.bytes_since_checkpoint > 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- wal_stats`
Expected: FAIL — `stats()` method and `WalStats` struct don't exist

**Step 3: Write minimal implementation**

Add to `crates/minkowski-persist/src/wal.rs`, near the existing public accessors (around line 349):

```rust
/// Read-only snapshot of WAL statistics. Plain data struct — no references
/// to internal state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalStats {
    pub next_seq: u64,
    pub segment_count: usize,
    pub oldest_seq: Option<u64>,
    pub bytes_since_checkpoint: u64,
    pub last_checkpoint_seq: Option<u64>,
    pub checkpoint_needed: bool,
}

impl Wal {
    // ... (inside the existing impl block)

    /// Snapshot of WAL statistics for observability.
    pub fn stats(&self) -> WalStats {
        WalStats {
            next_seq: self.next_seq(),
            segment_count: self.segment_count(),
            oldest_seq: self.oldest_seq(),
            bytes_since_checkpoint: self.bytes_since_checkpoint,
            last_checkpoint_seq: self.last_checkpoint_seq(),
            checkpoint_needed: self.checkpoint_needed(),
        }
    }
}
```

Add to `crates/minkowski-persist/src/lib.rs` re-exports:

```rust
pub use wal::{Wal, WalConfig, WalError, WalStats};
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist -- wal_stats`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs crates/minkowski-persist/src/lib.rs
git commit -m "feat: add WalStats facade and wal.stats() method"
```

---

### Task 3: Scaffold `minkowski-observe` crate

**Files:**
- Create: `crates/minkowski-observe/Cargo.toml`
- Create: `crates/minkowski-observe/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Create the crate files**

`crates/minkowski-observe/Cargo.toml`:
```toml
[package]
name = "minkowski-observe"
version = "0.1.0"
edition = "2021"

[dependencies]
minkowski = { path = "../minkowski" }
minkowski-persist = { path = "../minkowski-persist" }
```

`crates/minkowski-observe/src/lib.rs`:
```rust
//! Observability companion for Minkowski ECS.
//!
//! Pure consumer crate: captures read-only stats from `World` and `Wal`,
//! diffs consecutive snapshots, and computes rates. No changes to engine
//! semantics.

pub mod snapshot;
pub mod diff;

pub use snapshot::{ArchetypeInfo, MetricsSnapshot};
pub use diff::MetricsDiff;
```

**Step 2: Add to workspace**

Modify `Cargo.toml` at workspace root — add `"crates/minkowski-observe"` to the `members` array.

**Step 3: Verify it compiles**

Run: `cargo check -p minkowski-observe`
Expected: FAIL — modules don't exist yet, but the crate structure is valid

**Step 4: Create stub modules**

`crates/minkowski-observe/src/snapshot.rs`:
```rust
//! Point-in-time metrics capture.
```

`crates/minkowski-observe/src/diff.rs`:
```rust
//! Rate computation from consecutive snapshots.
```

**Step 5: Verify it compiles**

Run: `cargo check -p minkowski-observe`
Expected: PASS (empty modules)

**Step 6: Commit**

```bash
git add crates/minkowski-observe/ Cargo.toml
git commit -m "feat: scaffold minkowski-observe crate"
```

---

### Task 4: `MetricsSnapshot` and `ArchetypeInfo`

**Files:**
- Modify: `crates/minkowski-observe/src/snapshot.rs`
- Test: inline `#[cfg(test)] mod tests` in `snapshot.rs`

**Step 1: Write the failing test**

Add to `crates/minkowski-observe/src/snapshot.rs`:

```rust
use std::time::Instant;

use minkowski::world::WorldStats;
use minkowski::{ComponentId, World};
use minkowski_persist::wal::WalStats;
use minkowski_persist::{Wal, WalConfig, CodecRegistry};

/// Per-archetype detail.
#[derive(Clone, Debug)]
pub struct ArchetypeInfo {
    pub id: usize,
    pub entity_count: usize,
    pub component_names: Vec<&'static str>,
    pub estimated_bytes: usize,
}

/// Point-in-time capture of all engine metrics.
#[derive(Clone, Debug)]
pub struct MetricsSnapshot {
    pub world: WorldStats,
    pub wal: Option<WalStats>,
    pub archetypes: Vec<ArchetypeInfo>,
    pub timestamp: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Pos { x: f32, y: f32 }

    #[derive(Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn capture_without_wal() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        world.spawn((Pos { x: 3.0, y: 4.0 },));

        let snap = MetricsSnapshot::capture(&world, None);

        assert_eq!(snap.world.entity_count, 2);
        assert!(snap.wal.is_none());
        // At least 2 archetypes: (Pos, Vel) and (Pos)
        let non_empty: Vec<_> = snap.archetypes.iter()
            .filter(|a| a.entity_count > 0)
            .collect();
        assert_eq!(non_empty.len(), 2);
    }

    #[test]
    fn capture_with_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let snap = MetricsSnapshot::capture(&world, Some(&wal));
        assert!(snap.wal.is_some());
        assert_eq!(snap.wal.unwrap().next_seq, 0);
    }

    #[test]
    fn archetype_estimated_bytes() {
        let mut world = World::new();
        for _ in 0..10 {
            world.spawn((Pos { x: 1.0, y: 2.0 },));
        }

        let snap = MetricsSnapshot::capture(&world, None);
        let pos_arch = snap.archetypes.iter()
            .find(|a| a.entity_count == 10)
            .expect("should have archetype with 10 entities");

        // Pos is 8 bytes (2 x f32), 10 entities = 80 bytes
        assert_eq!(pos_arch.estimated_bytes, 80);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-observe`
Expected: FAIL — `MetricsSnapshot::capture` doesn't exist

**Step 3: Implement `capture`**

Add to `MetricsSnapshot` impl block in `snapshot.rs`:

```rust
impl MetricsSnapshot {
    /// Capture a point-in-time snapshot of engine metrics.
    ///
    /// Pass `Some(&wal)` to include WAL stats, or `None` for World-only metrics.
    pub fn capture(world: &World, wal: Option<&Wal>) -> Self {
        let world_stats = world.stats();
        let wal_stats = wal.map(|w| w.stats());

        let mut archetypes = Vec::with_capacity(world.archetype_count());
        for arch_idx in 0..world.archetype_count() {
            let comp_ids = world.archetype_component_ids(arch_idx);
            let entity_count = world.archetype_len(arch_idx);

            let component_names: Vec<&'static str> = comp_ids
                .iter()
                .filter_map(|&id| world.component_name(id))
                .collect();

            let bytes_per_entity: usize = comp_ids
                .iter()
                .filter_map(|&id| world.component_layout(id))
                .map(|layout| layout.size())
                .sum();

            archetypes.push(ArchetypeInfo {
                id: arch_idx,
                entity_count,
                component_names,
                estimated_bytes: entity_count * bytes_per_entity,
            });
        }

        Self {
            world: world_stats,
            wal: wal_stats,
            archetypes,
            timestamp: Instant::now(),
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-observe`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add crates/minkowski-observe/src/snapshot.rs
git commit -m "feat(observe): MetricsSnapshot with capture and ArchetypeInfo"
```

---

### Task 5: `MetricsDiff`

**Files:**
- Modify: `crates/minkowski-observe/src/diff.rs`
- Test: inline `#[cfg(test)] mod tests` in `diff.rs`

**Step 1: Write the failing test**

Add to `crates/minkowski-observe/src/diff.rs`:

```rust
use std::time::Duration;
use crate::snapshot::MetricsSnapshot;

/// Rates computed from two consecutive snapshots.
#[derive(Clone, Debug)]
pub struct MetricsDiff {
    pub elapsed: Duration,
    pub entity_delta: i64,
    pub entity_churn: u64,
    pub tick_delta: u64,
    pub wal_seq_delta: u64,
    pub archetype_delta: i64,
    pub largest_archetypes: Vec<(usize, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::World;
    use std::time::Instant;

    #[derive(Clone, Copy)]
    struct Pos { x: f32, y: f32 }

    #[test]
    fn diff_entity_delta() {
        let mut world = World::new();
        let before = MetricsSnapshot::capture(&world, None);

        world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.spawn((Pos { x: 3.0, y: 4.0 },));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert_eq!(diff.entity_delta, 2);
        assert!(diff.tick_delta > 0);
        assert_eq!(diff.wal_seq_delta, 0); // no WAL
    }

    #[test]
    fn diff_churn_estimation() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));
        let before = MetricsSnapshot::capture(&world, None);

        world.despawn(e1);
        world.despawn(e2);
        world.spawn((Pos { x: 5.0, y: 6.0 },));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert_eq!(diff.entity_delta, -1); // was 2, now 1
        // Churn: 2 despawns + 1 spawn = 3
        assert!(diff.entity_churn >= 3);
    }

    #[test]
    fn diff_largest_archetypes() {
        let mut world = World::new();
        for _ in 0..10 {
            world.spawn((Pos { x: 1.0, y: 2.0 },));
        }
        let before = MetricsSnapshot::capture(&world, None);
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert!(!diff.largest_archetypes.is_empty());
        assert_eq!(diff.largest_archetypes[0].1, 10);
    }

    #[test]
    fn diff_archetype_delta() {
        let mut world = World::new();
        let before = MetricsSnapshot::capture(&world, None);

        #[derive(Clone, Copy)]
        struct Vel { dx: f32, dy: f32 }
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert!(diff.archetype_delta >= 1);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-observe`
Expected: FAIL — `MetricsDiff::compute` doesn't exist

**Step 3: Implement `compute`**

Add to `MetricsDiff` impl block in `diff.rs`:

```rust
impl MetricsDiff {
    /// Compute rates and deltas from two consecutive snapshots.
    pub fn compute(before: &MetricsSnapshot, after: &MetricsSnapshot) -> Self {
        let elapsed = after.timestamp.duration_since(before.timestamp);

        let entity_delta = after.world.entity_count as i64 - before.world.entity_count as i64;

        // Churn estimation:
        // free_list_growth ≈ despawns (each despawn pushes to free list)
        // spawns ≈ entity_delta + despawns
        let free_list_growth = after.world.free_list_len.saturating_sub(before.world.free_list_len);
        let despawns = free_list_growth as u64;
        let spawns = (entity_delta + despawns as i64).max(0) as u64;
        let entity_churn = spawns + despawns;

        let tick_delta = after.world.current_tick.saturating_sub(before.world.current_tick);

        let wal_seq_delta = match (before.wal, after.wal) {
            (Some(b), Some(a)) => a.next_seq.saturating_sub(b.next_seq),
            _ => 0,
        };

        let archetype_delta =
            after.world.archetype_count as i64 - before.world.archetype_count as i64;

        // Top 5 archetypes by entity count (from the `after` snapshot)
        let mut sorted: Vec<(usize, usize)> = after
            .archetypes
            .iter()
            .filter(|a| a.entity_count > 0)
            .map(|a| (a.id, a.entity_count))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(5);

        Self {
            elapsed,
            entity_delta,
            entity_churn,
            tick_delta,
            wal_seq_delta,
            archetype_delta,
            largest_archetypes: sorted,
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-observe`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add crates/minkowski-observe/src/diff.rs
git commit -m "feat(observe): MetricsDiff with entity churn estimation"
```

---

### Task 6: `Display` impls

**Files:**
- Modify: `crates/minkowski-observe/src/snapshot.rs` (add `Display` impl)
- Modify: `crates/minkowski-observe/src/diff.rs` (add `Display` impl)

**Step 1: Write the failing test**

Add to `snapshot.rs` tests:

```rust
#[test]
fn snapshot_display_includes_key_info() {
    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 },));

    let snap = MetricsSnapshot::capture(&world, None);
    let output = format!("{snap}");

    assert!(output.contains("entities: 1"));
    assert!(output.contains("archetypes:"));
    assert!(output.contains("tick:"));
}
```

Add to `diff.rs` tests:

```rust
#[test]
fn diff_display_includes_key_info() {
    let mut world = World::new();
    let before = MetricsSnapshot::capture(&world, None);
    world.spawn((Pos { x: 1.0, y: 2.0 },));
    let after = MetricsSnapshot::capture(&world, None);

    let diff = MetricsDiff::compute(&before, &after);
    let output = format!("{diff}");

    assert!(output.contains("entity delta:"));
    assert!(output.contains("tick delta:"));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-observe`
Expected: FAIL — `Display` not implemented

**Step 3: Implement Display**

In `snapshot.rs`:

```rust
impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--- World ---")?;
        writeln!(f, "  entities: {}  archetypes: {}  components: {}",
            self.world.entity_count, self.world.archetype_count, self.world.component_count)?;
        writeln!(f, "  free list: {}  query cache: {}  tick: {}",
            self.world.free_list_len, self.world.query_cache_len, self.world.current_tick)?;

        if let Some(ref wal) = self.wal {
            writeln!(f, "--- WAL ---")?;
            writeln!(f, "  seq: {}  segments: {}  oldest: {:?}",
                wal.next_seq, wal.segment_count, wal.oldest_seq)?;
            writeln!(f, "  checkpoint: needed={}  last={:?}  bytes_since={}",
                wal.checkpoint_needed, wal.last_checkpoint_seq, wal.bytes_since_checkpoint)?;
        }

        if !self.archetypes.is_empty() {
            writeln!(f, "--- Archetypes ---")?;
            for arch in self.archetypes.iter().filter(|a| a.entity_count > 0) {
                writeln!(f, "  [{}] {} entities, ~{} bytes — {:?}",
                    arch.id, arch.entity_count, arch.estimated_bytes, arch.component_names)?;
            }
        }

        Ok(())
    }
}
```

In `diff.rs`:

```rust
impl std::fmt::Display for MetricsDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--- Diff ({:.1?}) ---", self.elapsed)?;
        writeln!(f, "  entity delta: {:+}  churn: {}",
            self.entity_delta, self.entity_churn)?;
        writeln!(f, "  tick delta: {}  WAL seq delta: {}",
            self.tick_delta, self.wal_seq_delta)?;
        writeln!(f, "  archetype delta: {:+}", self.archetype_delta)?;

        if !self.largest_archetypes.is_empty() {
            writeln!(f, "  largest archetypes:")?;
            for (id, count) in &self.largest_archetypes {
                writeln!(f, "    [{id}] {count} entities")?;
            }
        }

        Ok(())
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-observe`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski-observe/src/snapshot.rs crates/minkowski-observe/src/diff.rs
git commit -m "feat(observe): Display impls for MetricsSnapshot and MetricsDiff"
```

---

### Task 7: Integration example

**Files:**
- Create: `examples/examples/observe.rs`
- Modify: `examples/Cargo.toml` (add `minkowski-observe` dependency)

**Step 1: Add dependency**

In `examples/Cargo.toml`, add under `[dependencies]`:
```toml
minkowski-observe = { path = "../crates/minkowski-observe" }
```

**Step 2: Write the example**

`examples/examples/observe.rs`:

```rust
//! Observability — capture, diff, and display engine metrics.
//!
//! Demonstrates MetricsSnapshot capture at two points in time, diffing
//! to compute entity churn, tick velocity, and archetype changes.
//!
//! Run: cargo run -p minkowski-examples --example observe --release

use minkowski::{EnumChangeSet, World};
use minkowski_observe::{MetricsDiff, MetricsSnapshot};
use minkowski_persist::{CodecRegistry, Wal, WalConfig};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct Vel {
    dx: f32,
    dy: f32,
}

fn main() {
    let dir = std::env::temp_dir().join("minkowski-observe-example");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_dir = dir.join("observe.wal");
    let _ = std::fs::remove_dir_all(&wal_dir);

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Vel>("vel", &mut world);

    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

    // Phase 1: initial state
    for i in 0..100 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (
                Pos { x: i as f32, y: 0.0 },
                Vel { dx: 1.0, dy: 0.5 },
            ),
        );
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let snap1 = MetricsSnapshot::capture(&world, Some(&wal));
    println!("=== Snapshot 1 ===");
    println!("{snap1}");

    // Phase 2: churn — despawn 20, spawn 50 new
    let entities: Vec<_> = world.query::<(minkowski::Entity,)>()
        .take(20)
        .map(|e| e.0)
        .collect();
    for e in entities {
        world.despawn(e);
    }

    for i in 100..150 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (Pos { x: i as f32, y: 10.0 },),  // Pos-only archetype
        );
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let snap2 = MetricsSnapshot::capture(&world, Some(&wal));
    println!("=== Snapshot 2 ===");
    println!("{snap2}");

    let diff = MetricsDiff::compute(&snap1, &snap2);
    println!("=== Diff ===");
    println!("{diff}");

    let _ = std::fs::remove_dir_all(&dir);
    println!("Done.");
}
```

**Step 3: Run the example**

Run: `cargo run -p minkowski-examples --example observe --release`
Expected: Prints three sections of formatted metrics

**Step 4: Run full test suite to verify nothing is broken**

Run: `cargo test -p minkowski --lib && cargo test -p minkowski-persist && cargo test -p minkowski-observe`
Expected: All tests pass

**Step 5: Commit**

```bash
git add examples/examples/observe.rs examples/Cargo.toml
git commit -m "feat: add observe example demonstrating metrics capture and diff"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add to the example commands section**

Add after the `tactical` example entry:
```
cargo run -p minkowski-examples --example observe --release   # Observability: MetricsSnapshot capture, diff, entity churn estimation (100 entities, 2 archetypes)
```

**Step 2: Add to the Dependencies table**

Add:
```
| `minkowski-observe` | Observability companion: metrics capture, diff, display |
```

**Step 3: Add to the Architecture section**

Update the opening line to mention the new crate:
```
Five crates: `minkowski` (core), `minkowski-derive` (`#[derive(Table)]` proc macro), `minkowski-persist` (WAL, snapshots, durable transactions), `minkowski-observe` (metrics capture and display), and `minkowski-examples` (examples as external API consumers).
```

**Step 4: Add `WorldStats` and `WalStats` to the Key Conventions pub list**

Add `WorldStats` to the minkowski pub list and `WalStats` to the persist pub exports.

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add minkowski-observe to CLAUDE.md"
```
