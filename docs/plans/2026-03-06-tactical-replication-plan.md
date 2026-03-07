# Tactical Map Replication — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `tactical` example exercising 8 uncovered API gaps: sparse components, `par_for_each`, `Conflict` inspection, entity bit packing, world introspection, `register_entity_despawn`, stale index validation, and `EnumChangeSet`/`MutationRef` iteration.

**Architecture:** Server thread owns authoritative World; two operator threads own local client Worlds. Communication via `mpsc` channels. Server runs simulation, records mutations into an `EnumChangeSet` journal, iterates `MutationRef` to build replication packets. Attack commands go through `Optimistic` transactions to demonstrate `Conflict`.

**Tech Stack:** minkowski ECS, std::sync::mpsc, std::thread, rayon (via par_for_each), fastrand

**Design doc:** `docs/plans/2026-03-06-tactical-replication-design.md`

---

### Task 1: Scaffold file with components, constants, and wire types

**Files:**
- Create: `examples/examples/tactical.rs`

**Step 1: Create the file with all type definitions**

Write the module doc comment, imports, constants, component structs, wire types, and spatial index struct. No logic yet — just the data model.

```rust
//! Tactical map replication — multi-operator command & control with server authority.
//!
//! Run: cargo run -p minkowski-examples --example tactical --release
//!
//! Exercises: sparse components (insert_sparse, iter_sparse), par_for_each,
//! Optimistic transactions with Conflict inspection, Entity::to_bits/from_bits,
//! world introspection (archetype_count, component_name), register_entity_despawn,
//! HashIndex with get_valid() stale filtering, EnumChangeSet/MutationRef iteration.
//!
//! Architecture:
//! ```
//!   Operator A  --cmd-->  Server  --replication-->  Operator A
//!   Operator B  --cmd-->  Server  --replication-->  Operator B
//! ```
//!
//! Server thread owns the authoritative World. Operator threads own local
//! replica Worlds. Commands carry entity IDs as u64 via to_bits(); replication
//! packets are built by iterating EnumChangeSet mutations via MutationRef.

use minkowski::{
    Access, CommandBuffer, Entity, EnumChangeSet, HashIndex, MutationRef, Optimistic, QueryMut,
    QueryRef, ReducerRegistry, SpatialIndex, Transact, World,
};
use std::sync::mpsc;
use std::time::Instant;

// ── Constants ───────────────────────────────────────────────────────
const MAP_SIZE: f32 = 1000.0;
const BLUE: u8 = 0;
const RED: u8 = 1;
const BLUE_COUNT: usize = 50;
const RED_COUNT: usize = 50;
const TICK_COUNT: usize = 10;
const COMBAT_RANGE: f32 = 50.0;
const RECON_RANGE: f32 = 150.0;
const ATTACK_DAMAGE: u32 = 25;
const MOVE_SPEED: f32 = 20.0;

// ── Components (archetype-stored) ───────────────────────────────────
#[derive(Clone, Copy, Debug)]
struct Position(f32, f32);

#[derive(Clone, Copy, Debug)]
struct Heading(f32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Faction(u8);

#[derive(Clone, Copy, Debug)]
struct Health(u32);

#[derive(Clone, Copy, Debug)]
struct Speed(f32);

#[derive(Clone, Copy, Debug)]
struct UnitType(u8); // 0=Infantry, 1=Armor, 2=Recon

// ── Sparse components ───────────────────────────────────────────────
#[derive(Clone, Copy, Debug)]
struct IntelReport {
    spotted_tick: u64,
    confidence: f32,
}

#[derive(Clone, Copy, Debug)]
struct MoveOrder {
    target_x: f32,
    target_y: f32,
}

// ── Wire types ──────────────────────────────────────────────────────
#[derive(Clone, Debug)]
enum Command {
    Move { unit: u64, target: (f32, f32) },
    Attack { attacker: u64, target_unit: u64 },
}

#[derive(Clone, Debug)]
enum ReplicationEvent {
    Spawn { entity_bits: u64, pos: (f32, f32), faction: u8, health: u32, unit_type: u8 },
    UpdateHealth { entity_bits: u64, health: u32 },
    Despawn { entity_bits: u64 },
    MoveOrderIssued { entity_bits: u64, target: (f32, f32) },
    Spotted { entity_bits: u64, confidence: f32 },
}

// ── Spatial index ───────────────────────────────────────────────────
struct UnitGrid {
    cell_size: f32,
    grid_w: usize,
    cells: Vec<Vec<usize>>,
    snapshot: Vec<(Entity, f32, f32, u8)>, // entity, x, y, faction
}

fn main() {
    println!("Tactical map replication example — not yet implemented");
}
```

**Step 2: Verify it compiles**

Run: `cargo clippy -p minkowski-examples --example tactical -- -D warnings`
Expected: compiles clean (may warn about unused items — we'll use them soon)

**Step 3: Commit**

```bash
git add examples/examples/tactical.rs
git commit -m "feat(tactical): scaffold components, constants, and wire types"
```

---

### Task 2: Implement spatial index and unit spawning

**Files:**
- Modify: `examples/examples/tactical.rs`

**Step 1: Implement SpatialIndex for UnitGrid**

Add `impl UnitGrid` with `new()` and `neighbors()`, then `impl SpatialIndex for UnitGrid` with `rebuild()`. Same pattern as flatworm.rs FoodGrid.

```rust
impl UnitGrid {
    fn new(cell_size: f32) -> Self {
        let grid_w = (MAP_SIZE / cell_size).ceil() as usize;
        Self {
            cell_size,
            grid_w,
            cells: Vec::new(),
            snapshot: Vec::new(),
        }
    }

    fn neighbors(&self, x: f32, y: f32) -> impl Iterator<Item = &(Entity, f32, f32, u8)> {
        let cx = ((x / self.cell_size) as usize).min(self.grid_w - 1);
        let cy = ((y / self.cell_size) as usize).min(self.grid_w - 1);
        let grid_w = self.grid_w;
        (-1i32..=1).flat_map(move |dy| {
            (-1i32..=1).flat_map(move |dx| {
                let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                self.cells[ny * grid_w + nx].iter().map(|&j| &self.snapshot[j])
            })
        })
    }
}

impl SpatialIndex for UnitGrid {
    fn rebuild(&mut self, world: &mut World) {
        self.snapshot = world
            .query::<(Entity, &Position, &Faction)>()
            .map(|(e, p, f)| (e, p.0, p.1, f.0))
            .collect();
        self.cells.clear();
        self.cells.resize(self.grid_w * self.grid_w, Vec::new());
        for (i, &(_, x, y, _)) in self.snapshot.iter().enumerate() {
            let cx = ((x / self.cell_size) as usize).min(self.grid_w - 1);
            let cy = ((y / self.cell_size) as usize).min(self.grid_w - 1);
            self.cells[cy * self.grid_w + cx].push(i);
        }
    }
}
```

**Step 2: Add spawn_units function and basic main**

```rust
fn spawn_units(world: &mut World) -> Vec<Entity> {
    let mut entities = Vec::new();
    for i in 0..(BLUE_COUNT + RED_COUNT) {
        let faction = if i < BLUE_COUNT { BLUE } else { RED };
        let x = fastrand::f32() * MAP_SIZE;
        let y = fastrand::f32() * MAP_SIZE;
        let heading = fastrand::f32() * std::f32::consts::TAU;
        let unit_type = (i % 3) as u8; // cycle through Infantry, Armor, Recon
        let health = match unit_type {
            0 => 100, // Infantry
            1 => 200, // Armor
            _ => 60,  // Recon
        };
        let e = world.spawn((
            Position(x, y),
            Heading(heading),
            Faction(faction),
            Health(health),
            Speed(MOVE_SPEED),
            UnitType(unit_type),
        ));
        entities.push(e);
    }
    entities
}
```

Update `main()` to spawn units and print a count.

**Step 3: Verify**

Run: `cargo run -p minkowski-examples --example tactical --release`
Expected: prints spawn count, no panics

**Step 4: Commit**

```bash
git commit -am "feat(tactical): spatial index and unit spawning"
```

---

### Task 3: Initial state replication via EnumChangeSet + MutationRef

**Files:**
- Modify: `examples/examples/tactical.rs`

**Step 1: Build initial replication changeset**

After spawning units on the server, record the initial state into an `EnumChangeSet`, iterate `MutationRef` variants to build `ReplicationEvent`s, and send to operator threads. This is the primary demo of `EnumChangeSet::iter_mutations()` and `MutationRef` pattern matching.

```rust
fn build_initial_replication(world: &mut World) -> Vec<ReplicationEvent> {
    let mut events = Vec::new();
    for (entity, pos, faction, health, unit_type) in
        world.query::<(Entity, &Position, &Faction, &Health, &UnitType)>()
    {
        events.push(ReplicationEvent::Spawn {
            entity_bits: entity.to_bits(),
            pos: (pos.0, pos.1),
            faction: faction.0,
            health: health.0,
            unit_type: unit_type.0,
        });
    }
    events
}

fn changeset_to_events(changeset: &EnumChangeSet, world: &World) -> Vec<ReplicationEvent> {
    let mut events = Vec::new();
    for mutation in changeset.iter_mutations() {
        match mutation {
            MutationRef::Despawn { entity } => {
                events.push(ReplicationEvent::Despawn {
                    entity_bits: entity.to_bits(),
                });
            }
            MutationRef::Insert { entity, component_id, data } => {
                // Resolve component type from ID for Health updates
                if Some("tactical::Health") == world.component_name(component_id)
                    || world.component_name(component_id).map_or(false, |n| n.ends_with("Health"))
                {
                    if data.len() == std::mem::size_of::<u32>() {
                        let health = u32::from_ne_bytes(data.try_into().unwrap());
                        events.push(ReplicationEvent::UpdateHealth {
                            entity_bits: entity.to_bits(),
                            health,
                        });
                    }
                }
            }
            MutationRef::Spawn { entity, .. } => {
                events.push(ReplicationEvent::Despawn {
                    entity_bits: entity.to_bits(),
                });
                // Full spawn replication handled by build_initial_replication
            }
            MutationRef::Remove { entity, .. } => {
                // Component removal — log but don't replicate in this example
                let _ = entity;
            }
        }
    }
    events
}
```

**Step 2: Set up channels and spawn operator threads**

```rust
fn operator_thread(
    name: &'static str,
    faction: u8,
    cmd_tx: mpsc::Sender<(u8, Command)>,
    repl_rx: mpsc::Receiver<Vec<ReplicationEvent>>,
    unit_bits: Vec<u64>,
) {
    let mut world = World::new();
    let mut entity_map: std::collections::HashMap<u64, Entity> = std::collections::HashMap::new();

    // Receive initial replication
    if let Ok(events) = repl_rx.recv() {
        for event in &events {
            if let ReplicationEvent::Spawn { entity_bits, pos, faction: f, health, unit_type } = event {
                let e = world.spawn((
                    Position(pos.0, pos.1),
                    Faction(*f),
                    Health(*health),
                    UnitType(*unit_type),
                ));
                entity_map.insert(*entity_bits, e);
            }
        }
        println!("  [{}] received initial state: {} entities", name, entity_map.len());
    }

    // Main loop: send commands, receive replication
    let friendly_units: Vec<u64> = unit_bits.iter()
        .copied()
        .collect();
    let enemy_units: Vec<u64> = unit_bits.iter()
        .copied()
        .collect();

    for tick in 0..TICK_COUNT {
        // Send a command every few ticks
        if tick % 3 == 0 && !friendly_units.is_empty() {
            let unit = friendly_units[tick % friendly_units.len()];
            let target = (fastrand::f32() * MAP_SIZE, fastrand::f32() * MAP_SIZE);
            let _ = cmd_tx.send((faction, Command::Move { unit, target }));
        }
        if tick % 2 == 0 && !enemy_units.is_empty() {
            // Attack a random enemy
            let attacker = friendly_units[tick % friendly_units.len()];
            let target = enemy_units[tick % enemy_units.len()];
            let _ = cmd_tx.send((faction, Command::Attack { attacker, target_unit: target }));
        }

        // Receive replication
        if let Ok(events) = repl_rx.try_recv() {
            for event in &events {
                match event {
                    ReplicationEvent::Despawn { entity_bits } => {
                        if let Some(local_e) = entity_map.remove(entity_bits) {
                            world.despawn(local_e);
                        }
                    }
                    ReplicationEvent::UpdateHealth { entity_bits, health } => {
                        if let Some(&local_e) = entity_map.get(entity_bits) {
                            if let Some(h) = world.get_mut::<Health>(local_e) {
                                h.0 = *health;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Periodically check sparse intel reports
        if tick == TICK_COUNT - 1 {
            if let Some(intel_id) = world.component_id::<IntelReport>() {
                let count = world.iter_sparse::<IntelReport>(intel_id)
                    .map(|iter| iter.count())
                    .unwrap_or(0);
                println!("  [{}] intel reports on spotted enemies: {}", name, count);
            }
        }
    }
}
```

Note: the operator thread signature will be refined when we wire up main(). Entity lists for friendly/enemy selection will be passed based on faction.

**Step 3: Verify it compiles**

Run: `cargo clippy -p minkowski-examples --example tactical -- -D warnings`
Expected: compiles (some unused warnings acceptable at this stage)

**Step 4: Commit**

```bash
git commit -am "feat(tactical): initial replication via EnumChangeSet + operator threads"
```

---

### Task 4: Server tick loop — commands, transactions, and Conflict

**Files:**
- Modify: `examples/examples/tactical.rs`

**Step 1: Implement command processing with Optimistic transactions**

In the server's main loop, drain the command channel and process each command. Attack commands go through `Optimistic::transact()`. Engineer a conflict by having both operators attack the same unit.

Key APIs exercised:
- `Optimistic::new(&world)` — construct strategy
- `Access::of::<(&Health,)>(&world)` — declare access for attack transactions
- `strategy.transact(&mut world, &access, |tx, world| { ... })` — transactional execution
- `Err(Conflict)` — catch and display with `conflict.display_with(&world)`
- `Entity::from_bits(bits)` — deserialize entity IDs from commands
- `world.insert_sparse(entity, MoveOrder { ... })` — sparse insert for move orders

```rust
// Inside server tick loop:
let strategy = Optimistic::new(&world);
let attack_access = Access::of::<(&mut Health,)>(&world);

// Drain commands
let commands: Vec<(u8, Command)> = cmd_rx.try_iter().collect();
let mut frame_changeset = EnumChangeSet::new();

for (operator_faction, cmd) in &commands {
    match cmd {
        Command::Attack { attacker, target_unit } => {
            let atk_entity = Entity::from_bits(*attacker);
            let tgt_entity = Entity::from_bits(*target_unit);

            match strategy.transact(&mut world, &attack_access, |tx, world| {
                // Read target health, buffer damage
                if let Some(h) = world.get::<Health>(tgt_entity) {
                    let new_hp = h.0.saturating_sub(ATTACK_DAMAGE);
                    tx.write(tgt_entity, Health(new_hp));
                }
            }) {
                Ok(()) => {
                    // Record for replication
                    if let Some(h) = world.get::<Health>(tgt_entity) {
                        frame_changeset.insert::<Health>(&mut world, tgt_entity, Health(h.0));
                    }
                    println!("    attack: {} -> {} (damage {})", attacker, target_unit, ATTACK_DAMAGE);
                }
                Err(conflict) => {
                    total_conflicts += 1;
                    println!("    CONFLICT: {}", conflict.display_with(&world));
                }
            }
        }
        Command::Move { unit, target } => {
            let entity = Entity::from_bits(*unit);
            world.insert_sparse(entity, MoveOrder {
                target_x: target.0,
                target_y: target.1,
            });
            println!("    move order: {} -> ({:.0}, {:.0})", unit, target.0, target.1);
        }
    }
}
```

**Step 2: Verify compiles**

Run: `cargo clippy -p minkowski-examples --example tactical -- -D warnings`

**Step 3: Commit**

```bash
git commit -am "feat(tactical): command processing with Optimistic transactions + Conflict"
```

---

### Task 5: Simulation reducers — par_for_each, entity_despawn, QueryRef

**Files:**
- Modify: `examples/examples/tactical.rs`

**Step 1: Register simulation reducers**

Register four reducers with the ReducerRegistry:

1. **movement** — `register_query` with `QueryMut<(&mut Position, &mut Heading, &Speed)>` + `par_for_each` for parallel position update. Check sparse `MoveOrder` and steer toward target.

2. **combat** — `register_entity_despawn` with `EntityMut<(Health,)>`. For each unit, check spatial grid for nearby enemies, apply damage, despawn if dead.

3. **recon** — `register_query_ref` with `QueryRef<(&Position, &Faction, &UnitType)>`. Recon units (UnitType=2) spot nearby enemies, attach sparse `IntelReport`.

4. **census** — `register_query_ref` with `QueryRef<(&Faction, &Health)>`. Count units and total health per faction.

Key APIs exercised:
- `query.par_for_each(|(pos, heading, speed)| { ... })` — rayon parallel iteration
- `registry.register_entity_despawn::<(Health,), _, _>(...)` — despawn-capable EntityMut
- `entity_handle.get::<Health, 0>()` — typed read on EntityMut
- `entity_handle.despawn()` — despawn through EntityMut handle

Note on movement reducer: `par_for_each` cannot access sparse components (it only sees archetype-stored data). Movement toward MoveOrder targets must be handled separately (collect entities with MoveOrder from `iter_sparse`, then apply position changes). The `par_for_each` applies random drift to ALL units in parallel; targeted movement is a sequential pass.

**Step 2: Run the reducers in the server tick loop**

```rust
// Phase: Simulation
unit_grid.rebuild(&mut world);
registry.run(&mut world, movement_id, ());
// Sequential pass for MoveOrder steering (sparse components)
if let Some(mo_id) = world.component_id::<MoveOrder>() {
    let orders: Vec<(Entity, MoveOrder)> = world
        .iter_sparse::<MoveOrder>(mo_id)
        .into_iter()
        .flatten()
        .map(|(e, mo)| (e, *mo))
        .collect();
    for (entity, order) in orders {
        if let Some(pos) = world.get_mut::<Position>(entity) {
            let dx = order.target_x - pos.0;
            let dy = order.target_y - pos.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < MOVE_SPEED {
                pos.0 = order.target_x;
                pos.1 = order.target_y;
                // Arrival — would remove sparse, but no remove_sparse on World
                // Just leave it; it'll be overwritten on next order
            } else {
                pos.0 += dx / dist * MOVE_SPEED;
                pos.1 += dy / dist * MOVE_SPEED;
            }
        }
    }
}
// Combat via entity_despawn reducer (dispatched per-entity through strategy)
// Recon scan + census via QueryRef reducers
registry.run(&mut world, recon_id, ());
registry.run(&mut world, census_id, ());
```

For combat, since `register_entity_despawn` is dispatched per-entity via `registry.call()` with a transaction strategy, we need to iterate units and call it for each:
```rust
let strategy = Optimistic::new(&world);
let targets: Vec<Entity> = world.query::<(Entity, &Health)>()
    .filter(|(_, h)| h.0 == 0)
    .map(|(e, _)| e)
    .collect();
for target in targets {
    let _ = registry.call(&strategy, &mut world, combat_id, (target, ()));
}
```

**Step 3: Verify**

Run: `cargo run -p minkowski-examples --example tactical --release`
Expected: simulation runs, units move, combat events logged

**Step 4: Commit**

```bash
git commit -am "feat(tactical): simulation reducers with par_for_each + entity_despawn"
```

---

### Task 6: HashIndex with stale validation + world introspection

**Files:**
- Modify: `examples/examples/tactical.rs`

**Step 1: Add HashIndex for Faction lookups**

Create a `HashIndex<Faction>` before the tick loop. Rebuild each tick. After combat despawns, demonstrate `get_valid()` to show it filters out stale entries from despawned units.

```rust
let mut faction_index = HashIndex::<Faction>::new(&mut world);

// Inside tick loop, after combat:
faction_index.rebuild(&mut world);
let blue_alive: Vec<Entity> = faction_index
    .get_valid(&Faction(BLUE), &world)
    .collect();
let red_alive: Vec<Entity> = faction_index
    .get_valid(&Faction(RED), &world)
    .collect();
println!("  faction index (stale-filtered): blue={} red={}", blue_alive.len(), red_alive.len());
```

Key APIs exercised:
- `HashIndex::<Faction>::new(&mut world)` — construct index
- `faction_index.rebuild(&mut world)` — full rebuild
- `faction_index.get_valid(&Faction(BLUE), &world)` — stale-filtered lookup

**Step 2: Add world introspection logging**

Every 3 ticks, log archetype and component metadata:

```rust
if tick % 3 == 0 {
    println!("  [introspection] archetypes: {}", world.archetype_count());
    for arch_idx in 0..world.archetype_count() {
        let len = world.archetype_len(arch_idx);
        if len == 0 { continue; }
        let comp_ids = world.archetype_component_ids(arch_idx);
        let names: Vec<&str> = comp_ids.iter()
            .filter_map(|id| world.component_name(*id))
            .collect();
        println!("    arch[{}]: {} entities, components: {:?}", arch_idx, len, names);
    }
}
```

Key APIs exercised:
- `world.archetype_count()` — total archetype count
- `world.archetype_len(idx)` — entities per archetype
- `world.archetype_component_ids(idx)` — component IDs in archetype
- `world.component_name(id)` — resolve ComponentId to type name

**Step 3: Verify**

Run: `cargo run -p minkowski-examples --example tactical --release`
Expected: faction index counts printed, introspection stats printed every 3 ticks

**Step 4: Commit**

```bash
git commit -am "feat(tactical): HashIndex stale validation + world introspection"
```

---

### Task 7: Wire up main(), operator threads, and replication

**Files:**
- Modify: `examples/examples/tactical.rs`

**Step 1: Implement full main() with thread spawning**

Wire everything together:
1. Spawn units, build initial replication events
2. Create channels (2 command channels, 2 replication channels)
3. Spawn 2 operator threads with entity ID lists (split by faction)
4. Send initial replication to both operators
5. Run server tick loop (10 ticks)
6. Join operator threads
7. Print final summary

Pass Blue unit entity bits to operator A, Red unit bits to operator B.

Engineer conflict: on tick 5, both operators will have sent an Attack command targeting the same unit (arrange by sharing the first Red unit's bits with both operators as a "high value target").

**Step 2: Send replication events each tick**

After building `frame_changeset` events (from task 4) and combat despawn events, combine into a `Vec<ReplicationEvent>` and send to both operators.

Also convert the frame `EnumChangeSet` (recording health changes and despawns) to events via `changeset_to_events()`.

**Step 3: Print final summary**

```rust
println!("\n=== Final Summary ===");
println!("  ticks: {}", TICK_COUNT);
println!("  total commands: {}", total_commands);
println!("  total conflicts: {}", total_conflicts);
println!("  total kills: {}", total_kills);
let blue_final = world.query::<(&Faction,)>().filter(|(f,)| f.0 == BLUE).count();
let red_final = world.query::<(&Faction,)>().filter(|(f,)| f.0 == RED).count();
println!("  surviving: blue={} red={}", blue_final, red_final);
```

**Step 4: Verify full example runs**

Run: `cargo run -p minkowski-examples --example tactical --release`
Expected: full output with commands, conflicts, combat, replication, introspection, final summary

Run: `cargo clippy -p minkowski-examples --example tactical -- -D warnings`
Expected: clean

**Step 5: Commit**

```bash
git commit -am "feat(tactical): wire up main, operator threads, and full replication loop"
```

---

### Task 8: Update docs — CLAUDE.md, README.md, skills guide

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `.claude/skills/minkowski-guide.md`

**Step 1: Add run command to CLAUDE.md**

In the Build & Test Commands section, after the `circuit` line, add:
```
cargo run -p minkowski-examples --example tactical --release   # Multi-operator tactical map: sparse components, par_for_each, Optimistic Conflict, entity bit packing, HashIndex stale validation, EnumChangeSet/MutationRef replication (100 units, 10 ticks, 2 threads)
```

**Step 2: Add to README.md examples table**

After the `circuit` row, add a `tactical` row describing the example.

**Step 3: Add to minkowski-guide.md**

Add a row to the examples table and pattern quick-find entries:
```
| `tactical` | Sparse components (insert_sparse, iter_sparse), par_for_each, Optimistic transactions + Conflict inspection, Entity::to_bits/from_bits, world introspection, register_entity_despawn, HashIndex get_valid(), EnumChangeSet/MutationRef iteration | `examples/examples/tactical.rs` |
```

Pattern quick-find:
```
- **Sparse component lifecycle:** `tactical.rs` (insert_sparse for MoveOrder/IntelReport, iter_sparse for intel queries)
- **Entity bit packing for serialization:** `tactical.rs` (to_bits in commands, from_bits in replication)
- **EnumChangeSet as replication journal:** `tactical.rs` (iter_mutations + MutationRef pattern matching)
- **Optimistic Conflict inspection:** `tactical.rs` (catch Err(Conflict), display_with)
- **World introspection (archetype/component metadata):** `tactical.rs`
- **HashIndex stale filtering:** `tactical.rs` (get_valid after despawns)
```

**Step 4: Verify docs don't break anything**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean

**Step 5: Commit**

```bash
git commit -am "docs: add tactical example to CLAUDE.md, README.md, and skills guide"
```

---

### Task 9: Final verification and cleanup

**Files:**
- Possibly: `examples/examples/tactical.rs` (cleanup only)

**Step 1: Run full test suite**

```bash
cargo test -p minkowski
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

**Step 2: Run the example 3 times to verify stability**

```bash
cargo run -p minkowski-examples --example tactical --release
cargo run -p minkowski-examples --example tactical --release
cargo run -p minkowski-examples --example tactical --release
```

Expected: consistent output, no panics, no races. Worm counts and combat outcomes may vary (RNG) but structure is stable.

**Step 3: Verify all 8 API gaps are exercised**

Checklist — grep the file for each:
- [ ] `insert_sparse` — MoveOrder and IntelReport
- [ ] `iter_sparse` — operator thread intel count
- [ ] `par_for_each` — movement reducer
- [ ] `Conflict` + `display_with` — attack transaction failure
- [ ] `to_bits()` — command serialization
- [ ] `from_bits()` — command deserialization
- [ ] `archetype_count()` + `component_name()` — introspection
- [ ] `register_entity_despawn` — combat reducer
- [ ] `HashIndex::get_valid()` — stale filtering
- [ ] `EnumChangeSet::iter_mutations()` + `MutationRef` — replication events

**Step 4: Final commit if any cleanup was needed**

```bash
cargo fmt --all
git commit -am "chore: final cleanup for tactical example"
```
