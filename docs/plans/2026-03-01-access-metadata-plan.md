# Access Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Context:** Examples currently live inside the `minkowski` crate (`crates/minkowski/examples/`), which means they can access `pub(crate)` internals. This has caused bugs twice — most recently, `#[derive(Table)]` in `life.rs` fails because the derive macro generates code referencing `pub(crate) ComponentRegistry`. Moving examples to an external crate makes the compiler enforce the public API boundary permanently.

**Goal:** Add an `Access` struct that extracts component-level read/write metadata from any `WorldQuery` type, with a `conflicts_with()` method for scheduler authors to detect data races between systems.

**Architecture:** `Access` wraps two `FixedBitSet`s (reads, writes) derived from `WorldQuery::required_ids()` and `mutable_ids()`. `conflicts_with()` is two bitwise ANDs — O(N/64) where N is total component count. A `scheduler` example demonstrates the full conflict analysis workflow.

**Tech Stack:** Rust, `fixedbitset` (existing dependency)

---

### Task 1: Implement `Access` struct with tests

**Files:**
- Create: `crates/minkowski/src/access.rs`
- Modify: `crates/minkowski/src/lib.rs`

**Step 1: Create `access.rs` with the struct, constructor, accessors, and `conflicts_with`**

```rust
use fixedbitset::FixedBitSet;

use crate::query::fetch::WorldQuery;
use crate::world::World;

/// Component-level access metadata for a query type.
///
/// Used by external schedulers to detect conflicts between systems.
/// Two accesses conflict if either writes a component the other reads or writes
/// (standard read-write lock rule, applied per component).
///
/// # Example
///
/// ```
/// use minkowski::{Access, World};
///
/// #[derive(Clone, Copy)]
/// struct Pos(f32);
/// #[derive(Clone, Copy)]
/// struct Vel(f32);
/// #[derive(Clone, Copy)]
/// struct Health(u32);
///
/// let mut world = World::new();
///
/// let movement = Access::of::<(&mut Pos, &Vel)>(&mut world);
/// let regen = Access::of::<(&mut Health,)>(&mut world);
/// let log = Access::of::<(&Pos,)>(&mut world);
///
/// // Disjoint writes — no conflict
/// assert!(!movement.conflicts_with(&regen));
///
/// // Read Pos vs write Pos — conflict
/// assert!(movement.conflicts_with(&log));
/// ```
pub struct Access {
    reads: FixedBitSet,
    writes: FixedBitSet,
}

impl Access {
    /// Build access metadata for a query type.
    ///
    /// Registers any unregistered component types as a side effect
    /// (same as `world.query::<Q>()`).
    pub fn of<Q: WorldQuery + 'static>(world: &mut World) -> Self {
        let required = Q::required_ids(&world.components);
        let writes = Q::mutable_ids(&world.components);

        // reads = required - writes (components read but not written)
        let mut reads = required;
        reads.difference_with(&writes);

        Self { reads, writes }
    }

    /// Components this query reads but does not write.
    pub fn reads(&self) -> &FixedBitSet {
        &self.reads
    }

    /// Components this query writes (mutably accesses).
    pub fn writes(&self) -> &FixedBitSet {
        &self.writes
    }

    /// True if these two accesses cannot safely run concurrently.
    ///
    /// Conflict rule: two accesses conflict iff either writes to a
    /// component the other reads or writes.
    pub fn conflicts_with(&self, other: &Access) -> bool {
        // Does self write anything other reads or writes?
        if self.writes.intersection(&other.reads).next().is_some() {
            return true;
        }
        if self.writes.intersection(&other.writes).next().is_some() {
            return true;
        }
        // Does other write anything self reads?
        // (other writes ∩ self writes already covered above)
        if other.writes.intersection(&self.reads).next().is_some() {
            return true;
        }
        false
    }
}
```

**Step 2: Add module and re-export in `lib.rs`**

Add `pub mod access;` after the existing module declarations, and `pub use access::Access;` in the re-exports section.

In `crates/minkowski/src/lib.rs`, add:
- `pub mod access;` between `pub mod bundle;` and... actually, alphabetically before `bundle`:

```
pub mod access;
pub mod bundle;
```

And in the re-exports:
```
pub use access::Access;
```

**Step 3: Write inline tests in `access.rs`**

Append to `access.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    struct Vel(f32);
    #[derive(Clone, Copy)]
    struct Health(u32);

    #[test]
    fn reads_and_writes_for_immutable_ref() {
        let mut world = World::new();
        let a = Access::of::<(&Pos,)>(&mut world);
        assert!(!a.reads().is_empty());
        assert!(a.writes().is_empty());
    }

    #[test]
    fn reads_and_writes_for_mutable_ref() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos,)>(&mut world);
        assert!(a.reads().is_empty()); // reads = required - writes = empty
        assert!(!a.writes().is_empty());
    }

    #[test]
    fn mixed_read_write_query() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos, &Vel)>(&mut world);
        // Pos is written, Vel is read-only
        assert!(!a.writes().is_empty());
        assert!(!a.reads().is_empty());
    }

    #[test]
    fn no_conflict_disjoint_writes() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos,)>(&mut world);
        let b = Access::of::<(&mut Health,)>(&mut world);
        assert!(!a.conflicts_with(&b));
        assert!(!b.conflicts_with(&a));
    }

    #[test]
    fn conflict_read_write_same_component() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos,)>(&mut world);
        let b = Access::of::<(&Pos,)>(&mut world);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn conflict_write_write_same_component() {
        let mut world = World::new();
        let a = Access::of::<(&mut Health,)>(&mut world);
        let b = Access::of::<(&mut Health,)>(&mut world);
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn no_conflict_read_read_same_component() {
        let mut world = World::new();
        let a = Access::of::<(&Pos,)>(&mut world);
        let b = Access::of::<(&Pos,)>(&mut world);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn conflict_is_symmetric() {
        let mut world = World::new();
        let a = Access::of::<(&mut Pos, &Vel)>(&mut world);
        let b = Access::of::<(&mut Vel,)>(&mut world);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn no_conflict_empty_access() {
        let mut world = World::new();
        let a = Access::of::<(Entity,)>(&mut world);
        let b = Access::of::<(&mut Pos,)>(&mut world);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn complex_disjoint_systems() {
        let mut world = World::new();
        // movement: reads Vel, writes Pos
        let movement = Access::of::<(&mut Pos, &Vel)>(&mut world);
        // health_regen: writes Health
        let regen = Access::of::<(&mut Health,)>(&mut world);
        // These touch completely different components
        assert!(!movement.conflicts_with(&regen));
    }
}
```

**Step 4: Run tests**

```
cargo test -p minkowski --lib -- access
```

Expected: all 10 tests pass.

**Step 5: Run clippy**

```
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 6: Commit**

```
git add crates/minkowski/src/access.rs crates/minkowski/src/lib.rs
git commit -m "feat: Access struct for query conflict detection"
```

---

### Task 2: Add `scheduler` example

**Files:**
- Create: `examples/examples/scheduler.rs`

**Step 1: Write the scheduler example**

Six systems in three batches of two. Prints conflict matrix + greedy batch assignment.

```rust
//! Minimal parallel scheduler — exercises `Access` for conflict detection.
//!
//! Run: cargo run -p minkowski-examples --example scheduler --release
//!
//! This is NOT a real framework scheduler. It demonstrates how a framework
//! author would use `Access::of` and `conflicts_with` to build one.
//!
//! Six systems, three batches:
//! - Batch 0: movement (writes Pos, reads Vel) + health_regen (writes Health)
//! - Batch 1: gravity (writes Vel) + apply_damage (writes Health)
//! - Batch 2: log_positions (reads Pos) + log_health (reads Health)
//!
//! Within each batch, systems touch disjoint component sets and could run
//! in parallel. Across batches, conflicts require sequential execution.

use minkowski::{Access, World};

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
struct Vel {
    dx: f32,
    dy: f32,
}

#[derive(Clone, Copy)]
struct Health(u32);

// ── Systems ─────────────────────────────────────────────────────────

fn movement(world: &mut World) {
    for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
        pos.x += vel.dx;
        pos.y += vel.dy;
    }
}

fn gravity(world: &mut World) {
    for vel in world.query::<(&mut Vel,)>() {
        vel.0.dy -= 9.8;
    }
}

fn health_regen(world: &mut World) {
    for hp in world.query::<(&mut Health,)>() {
        hp.0 .0 = hp.0 .0.saturating_add(1);
    }
}

fn apply_damage(world: &mut World) {
    for hp in world.query::<(&mut Health,)>() {
        hp.0 .0 = hp.0 .0.saturating_sub(5);
    }
}

fn log_positions(world: &mut World) {
    let count = world.query::<(&Pos,)>().count();
    println!("    log_positions: {count} entities");
}

fn log_health(world: &mut World) {
    let total: u32 = world.query::<(&Health,)>().map(|h| h.0 .0).sum();
    println!("    log_health: total HP = {total}");
}

// ── Greedy batch scheduler ──────────────────────────────────────────

struct SystemEntry {
    name: &'static str,
    access: Access,
    run: fn(&mut World),
}

/// Assign systems to batches using greedy graph coloring.
/// Systems in the same batch have no conflicts and could run in parallel.
fn assign_batches(systems: &[SystemEntry]) -> Vec<Vec<usize>> {
    let n = systems.len();
    let mut batch_of = vec![usize::MAX; n];
    let mut batches: Vec<Vec<usize>> = Vec::new();

    for i in 0..n {
        // Find the first batch where system i has no conflicts
        let mut assigned = false;
        for (b, batch) in batches.iter().enumerate() {
            let conflicts_with_batch = batch
                .iter()
                .any(|&j| systems[i].access.conflicts_with(&systems[j].access));
            if !conflicts_with_batch {
                batches[b].push(i);
                batch_of[i] = b;
                assigned = true;
                break;
            }
        }
        if !assigned {
            batch_of[i] = batches.len();
            batches.push(vec![i]);
        }
    }

    batches
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let mut world = World::new();

    // Spawn some entities
    for i in 0..100 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel {
                dx: 1.0,
                dy: 0.0,
            },
            Health(100),
        ));
    }

    // Register systems with their access metadata
    let systems = vec![
        SystemEntry {
            name: "movement",
            access: Access::of::<(&mut Pos, &Vel)>(&mut world),
            run: movement,
        },
        SystemEntry {
            name: "gravity",
            access: Access::of::<(&mut Vel,)>(&mut world),
            run: gravity,
        },
        SystemEntry {
            name: "health_regen",
            access: Access::of::<(&mut Health,)>(&mut world),
            run: health_regen,
        },
        SystemEntry {
            name: "apply_damage",
            access: Access::of::<(&mut Health,)>(&mut world),
            run: apply_damage,
        },
        SystemEntry {
            name: "log_positions",
            access: Access::of::<(&Pos,)>(&mut world),
            run: log_positions,
        },
        SystemEntry {
            name: "log_health",
            access: Access::of::<(&Health,)>(&mut world),
            run: log_health,
        },
    ];

    // ── Conflict matrix ─────────────────────────────────────────────

    println!("Conflict matrix:");
    println!();
    let max_name = systems.iter().map(|s| s.name.len()).max().unwrap_or(0);
    for (i, a) in systems.iter().enumerate() {
        for (j, b) in systems.iter().enumerate().skip(i + 1) {
            let tag = if a.access.conflicts_with(&b.access) {
                "CONFLICT"
            } else {
                "independent"
            };
            println!(
                "  {:>width$} <-> {:<width$}  {}",
                a.name,
                b.name,
                tag,
                width = max_name
            );
        }
    }
    println!();

    // ── Batch assignment ────────────────────────────────────────────

    let batches = assign_batches(&systems);

    println!("Batch assignment ({} batches):", batches.len());
    for (b, batch) in batches.iter().enumerate() {
        let names: Vec<_> = batch.iter().map(|&i| systems[i].name).collect();
        println!("  batch {b}: [{}]", names.join(", "));
    }
    println!();

    // ── Execute ─────────────────────────────────────────────────────

    println!("Running 10 frames:");
    for frame in 0..10 {
        for (b, batch) in batches.iter().enumerate() {
            // Within a batch, systems could run in parallel — they touch
            // disjoint component sets. A real framework would use rayon or
            // scoped threads here. We run sequentially to keep the example
            // focused on Access, not unsafe parallel execution.
            for &i in batch {
                (systems[i].run)(&mut world);
            }

            if frame == 0 {
                let names: Vec<_> = batch.iter().map(|&i| systems[i].name).collect();
                println!("  frame {frame}, batch {b}: ran [{}]", names.join(", "));
            }
        }
    }

    // Final state
    let pos_sum: f32 = world.query::<(&Pos,)>().map(|p| p.0.x).sum();
    let vel_sum: f32 = world.query::<(&Vel,)>().map(|v| v.0.dy).sum();
    let hp_sum: u32 = world.query::<(&Health,)>().map(|h| h.0 .0).sum();

    println!();
    println!("After 10 frames:");
    println!("  avg pos.x:  {:.1}", pos_sum / 100.0);
    println!("  avg vel.dy: {:.1}", vel_sum / 100.0);
    println!("  avg hp:     {:.1}", hp_sum as f32 / 100.0);
    println!();
    println!("Done.");
}
```

**Step 2: Build and run**

```
cargo build -p minkowski-examples --example scheduler
cargo run -p minkowski-examples --example scheduler --release
```

Expected output includes:
- Conflict matrix showing 7 conflicts and 8 independent pairs
- 3 batches
- 10 frames of execution

**Step 3: Run clippy**

```
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 4: Commit**

```
git add examples/examples/scheduler.rs
git commit -m "feat: scheduler example demonstrating Access conflict detection"
```

---

### Task 3: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Add the scheduler example run command alongside the existing examples:

```
cargo run -p minkowski-examples --example scheduler --release   # Access conflict detection demo (6 systems, 10 frames)
```

Update the Architecture section:
- Add a **System Scheduling Primitives** subsection under the existing sections (after Secondary Indexes):

```markdown
### System Scheduling Primitives

`Access` extracts component-level read/write metadata from any `WorldQuery` type. `Access::of::<(&mut Pos, &Vel)>(world)` returns a struct with two `FixedBitSet`s: reads (Vel) and writes (Pos). `conflicts_with()` detects whether two accesses violate the read-write lock rule — two bitwise ANDs over the component bitsets.

This is a building block for framework-level schedulers. Minkowski provides the access metadata; scheduling policy (dependency graphs, topological sort, parallel execution) is the framework's responsibility.
```

Update Key Conventions to include `Access` in the pub API list.

**Step 2: Update README.md**

Add a scheduler example section after the N-body example:

```markdown
### Scheduler example

A minimal conflict analysis demo showing how a framework author would use `Access` to detect data races between systems.

\```
$ cargo run -p minkowski-examples --example scheduler --release

Conflict matrix:
     movement <-> gravity        CONFLICT
     movement <-> health_regen   independent
  ...

Batch assignment (3 batches):
  batch 0: [movement, health_regen]
  batch 1: [gravity, apply_damage]
  batch 2: [log_positions, log_health]

Running 10 frames:
  ...
Done.
\```

Six systems demonstrate every conflict case: write/write (`health_regen` vs `apply_damage`), read/write (`log_positions` vs `movement`), disjoint writes that parallelize (`movement` + `health_regen`), and read-only systems that batch with non-overlapping writers. The greedy batcher assigns systems to 3 batches — within each batch, systems touch disjoint components and could run in parallel.
```

Update the "Phase 4 — done" line to mention Access:

```markdown
**Phase 4 — done:** `SpatialIndex` lifecycle trait, Barnes-Hut N-body example, boids grid refactored through the trait. `Access` struct for query conflict detection, scheduler example.
```

**Step 3: Run clippy**

```
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 4: Commit**

```
git add CLAUDE.md README.md
git commit -m "docs: add Access and scheduler example to CLAUDE.md and README.md"
```

---

### Verification

1. `cargo test -p minkowski --lib -- access` — all access tests pass
2. `cargo run -p minkowski-examples --example scheduler --release` — prints conflict matrix + batches + frame execution
3. `cargo test -p minkowski --lib` — all existing tests still pass
4. `cargo clippy --workspace --all-targets -- -D warnings` — clean
5. `cargo test -p minkowski` — doc test in Access passes
