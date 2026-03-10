//! Tactical map replication -- multi-operator command & control with server authority.
//!
//! Run: cargo run -p minkowski-examples --example tactical --release
//!
//! Exercises: sparse components (insert_sparse, iter_sparse), par_for_each,
//! Optimistic transactions with Conflict inspection, Entity::to_bits/from_bits,
//! world introspection (archetype_count, component_name), register_entity_despawn,
//! HashIndex with get_valid() stale filtering, EnumChangeSet/MutationRef iteration.
//!
//! Architecture:
//! ```text
//!   Operator A  --cmd-->  Server  --replication-->  Operator A
//!   Operator B  --cmd-->  Server  --replication-->  Operator B
//! ```
//!
//! Server thread owns the authoritative World. Operator threads own local
//! replica Worlds. Commands carry entity IDs as u64 via to_bits(); replication
//! packets are built by iterating EnumChangeSet mutations via MutationRef.

use minkowski::{
    Access, CommandBuffer, Entity, EnumChangeSet, HashIndex, MutationRef, Optimistic,
    ReducerRegistry, SpatialIndex, Transact, World,
};
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Instant;

// -- Constants ----------------------------------------------------------------
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

// -- Components (archetype-stored) --------------------------------------------
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

// -- Sparse components --------------------------------------------------------
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct IntelReport {
    spotted_tick: u64,
    confidence: f32,
}

#[derive(Clone, Copy, Debug)]
struct MoveOrder {
    target_x: f32,
    target_y: f32,
}

// -- Wire types ---------------------------------------------------------------
#[derive(Clone, Debug)]
enum Command {
    Move { unit: u64, target: (f32, f32) },
    Attack { attacker: u64, target_unit: u64 },
}

#[derive(Clone, Debug)]
enum ReplicationEvent {
    Spawn {
        entity_bits: u64,
        pos: (f32, f32),
        faction: u8,
        health: u32,
        unit_type: u8,
    },
    UpdateHealth {
        entity_bits: u64,
        health: u32,
    },
    Despawn {
        entity_bits: u64,
    },
}

// -- Spatial index -------------------------------------------------------------
struct UnitGrid {
    cell_size: f32,
    grid_w: usize,
    cells: Vec<Vec<usize>>,
    snapshot: Vec<(Entity, f32, f32, u8)>, // entity, x, y, faction
}

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

    #[allow(clippy::cast_possible_wrap)]
    fn neighbors(&self, x: f32, y: f32, range: f32) -> Vec<(Entity, f32, f32, u8)> {
        let cells_needed = (range / self.cell_size).ceil() as i32;
        let cx = ((x / self.cell_size) as usize).min(self.grid_w.saturating_sub(1));
        let cy = ((y / self.cell_size) as usize).min(self.grid_w.saturating_sub(1));
        let grid_w = self.grid_w;
        let range_sq = range * range;
        let mut result = Vec::new();
        for dy in -cells_needed..=cells_needed {
            for dx in -cells_needed..=cells_needed {
                let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                for &j in &self.cells[ny * grid_w + nx] {
                    let (e, ex, ey, f) = self.snapshot[j];
                    let ddx = (ex - x).abs();
                    let ddy = (ey - y).abs();
                    let ddx = ddx.min(MAP_SIZE - ddx);
                    let ddy = ddy.min(MAP_SIZE - ddy);
                    if ddx * ddx + ddy * ddy <= range_sq {
                        result.push((e, ex, ey, f));
                    }
                }
            }
        }
        result
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
            let cx = ((x / self.cell_size) as usize).min(self.grid_w.saturating_sub(1));
            let cy = ((y / self.cell_size) as usize).min(self.grid_w.saturating_sub(1));
            self.cells[cy * self.grid_w + cx].push(i);
        }
    }
}

// -- Spawn units --------------------------------------------------------------
fn spawn_units(world: &mut World) -> Vec<Entity> {
    let mut entities = Vec::new();
    for i in 0..(BLUE_COUNT + RED_COUNT) {
        let faction = if i < BLUE_COUNT { BLUE } else { RED };
        let x = fastrand::f32() * MAP_SIZE;
        let y = fastrand::f32() * MAP_SIZE;
        let heading = fastrand::f32() * std::f32::consts::TAU;
        let unit_type = (i % 3) as u8;
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

// -- Initial replication ------------------------------------------------------
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

// -- Changeset -> replication events ------------------------------------------
fn changeset_to_events(changeset: &EnumChangeSet, world: &World) -> Vec<ReplicationEvent> {
    let mut events = Vec::new();
    for mutation in changeset.iter_mutations() {
        match mutation {
            MutationRef::Despawn { entity } => {
                events.push(ReplicationEvent::Despawn {
                    entity_bits: entity.to_bits(),
                });
            }
            MutationRef::Insert {
                entity,
                component_id,
                data,
            } => {
                // Check if this is a Health component update
                if world
                    .component_name(component_id)
                    .is_some_and(|n| n.ends_with("Health"))
                    && data.len() == std::mem::size_of::<Health>()
                {
                    let health = u32::from_ne_bytes(data[..4].try_into().unwrap());
                    events.push(ReplicationEvent::UpdateHealth {
                        entity_bits: entity.to_bits(),
                        health,
                    });
                }
            }
            MutationRef::Spawn { entity, .. } => {
                // Spawns during simulation are rare; log as despawn placeholder
                let _ = entity;
            }
            MutationRef::Remove { entity, .. } => {
                let _ = entity;
            }
            MutationRef::SparseInsert { .. } | MutationRef::SparseRemove { .. } => {
                // Sparse components not used in this example — skip replication.
            }
        }
    }
    events
}

// -- Operator thread ----------------------------------------------------------
#[allow(clippy::needless_pass_by_value)]
fn operator_thread(
    name: &'static str,
    faction: u8,
    cmd_tx: mpsc::Sender<(u8, Command)>,
    repl_rx: mpsc::Receiver<Vec<ReplicationEvent>>,
    friendly_bits: Vec<u64>,
    enemy_bits: Vec<u64>,
    high_value_target: u64,
) {
    let mut world = World::new();
    let mut entity_map: HashMap<u64, Entity> = HashMap::new();

    // Receive initial replication
    if let Ok(events) = repl_rx.recv() {
        for event in &events {
            if let ReplicationEvent::Spawn {
                entity_bits,
                pos,
                faction: f,
                health,
                unit_type,
            } = event
            {
                let e = world.spawn((
                    Position(pos.0, pos.1),
                    Faction(*f),
                    Health(*health),
                    UnitType(*unit_type),
                ));
                entity_map.insert(*entity_bits, e);
            }
        }
        println!(
            "  [{}] received initial state: {} entities",
            name,
            entity_map.len()
        );
    }

    // Send all commands upfront so the server processes them in its first drain.
    // Both operators send an attack on the same high-value target to trigger Conflict.
    if !friendly_bits.is_empty() {
        // HVT attack -- both operators target the same enemy
        let _ = cmd_tx.send((
            faction,
            Command::Attack {
                attacker: friendly_bits[0],
                target_unit: high_value_target,
            },
        ));
    }
    for tick in 0..TICK_COUNT {
        if tick % 3 == 0 && !friendly_bits.is_empty() {
            let unit = friendly_bits[tick % friendly_bits.len()];
            let target = (fastrand::f32() * MAP_SIZE, fastrand::f32() * MAP_SIZE);
            let _ = cmd_tx.send((faction, Command::Move { unit, target }));
        }
        if tick % 4 == 0 && !enemy_bits.is_empty() {
            let attacker = friendly_bits[tick % friendly_bits.len()];
            let target = enemy_bits[tick % enemy_bits.len()];
            let _ = cmd_tx.send((
                faction,
                Command::Attack {
                    attacker,
                    target_unit: target,
                },
            ));
        }
    }

    // Receive replication updates via blocking recv — one packet per server tick.
    // The channel closes when the server drops its sender, ending the loop.
    let mut ticks_received = 0u32;
    while let Ok(events) = repl_rx.recv() {
        for event in &events {
            match event {
                ReplicationEvent::Despawn { entity_bits } => {
                    if let Some(local_e) = entity_map.remove(entity_bits) {
                        world.despawn(local_e);
                    }
                }
                ReplicationEvent::UpdateHealth {
                    entity_bits,
                    health,
                } => {
                    if let Some(&local_e) = entity_map.get(entity_bits) {
                        if let Some(h) = world.get_mut::<Health>(local_e) {
                            h.0 = *health;
                        }
                    }
                }
                ReplicationEvent::Spawn { .. } => {
                    // Already handled during initial replication
                }
            }
        }
        ticks_received += 1;
    }

    // Check sparse intel reports after all replication is consumed
    if let Some(intel_id) = world.component_id::<IntelReport>() {
        let count = world
            .iter_sparse::<IntelReport>(intel_id)
            .map_or(0, Iterator::count);
        println!("  [{}] intel reports on spotted enemies: {}", name, count);
    } else {
        println!("  [{}] no intel reports (component never registered)", name);
    }

    println!(
        "  [{}] finished, {} ticks received, {} live entities",
        name,
        ticks_received,
        entity_map.len()
    );
}

// -- Main (server) ------------------------------------------------------------
fn main() {
    let start = Instant::now();
    println!("=== Tactical Map Replication ===\n");

    // -- Spawn units on server world --
    let mut world = World::new();
    let all_entities = spawn_units(&mut world);
    println!(
        "Server spawned {} units ({} blue, {} red)",
        all_entities.len(),
        BLUE_COUNT,
        RED_COUNT
    );

    // Partition entity bits by faction
    let blue_bits: Vec<u64> = all_entities[..BLUE_COUNT]
        .iter()
        .map(|e| e.to_bits())
        .collect();
    let red_bits: Vec<u64> = all_entities[BLUE_COUNT..]
        .iter()
        .map(|e| e.to_bits())
        .collect();

    // High value target: first red unit -- both operators will attack it on tick 5
    let high_value_target = red_bits[0];

    // -- Build initial replication events --
    let initial_events = build_initial_replication(&mut world);
    println!("Built {} initial replication events", initial_events.len());

    // -- Set up channels --
    let (cmd_tx_a, cmd_rx) = mpsc::channel::<(u8, Command)>();
    let cmd_tx_b = cmd_tx_a.clone();
    let (repl_tx_a, repl_rx_a) = mpsc::channel::<Vec<ReplicationEvent>>();
    let (repl_tx_b, repl_rx_b) = mpsc::channel::<Vec<ReplicationEvent>>();

    // Send initial state to both operators
    repl_tx_a.send(initial_events.clone()).unwrap();
    repl_tx_b.send(initial_events).unwrap();

    // -- Register reducers --
    let mut registry = ReducerRegistry::new();

    // Combat cleanup: despawn dead units via register_entity_despawn
    let combat_cleanup_id = registry
        .register_entity_despawn::<(Health,), (), _>(&mut world, "combat_cleanup", {
            |mut entity: minkowski::EntityMut<'_, (Health,)>, ()| {
                let hp = entity.get::<Health, 0>().0;
                if hp == 0 {
                    entity.despawn();
                }
            }
        })
        .unwrap();

    // Movement reducer: parallel position update
    let movement_id = registry
        .register_query::<(&mut Position, &mut Heading, &Speed), (), _>(
            &mut world,
            "movement",
            |mut query: minkowski::QueryMut<'_, (&mut Position, &mut Heading, &Speed)>, ()| {
                query.for_each(|(pos, heading, speed)| {
                    // Random drift
                    let drift = (fastrand::f32() - 0.5) * 0.3;
                    heading.0 += drift;
                    pos.0 += heading.0.cos() * speed.0 * 0.1;
                    pos.1 += heading.0.sin() * speed.0 * 0.1;
                    // Wrap around map
                    pos.0 = pos.0.rem_euclid(MAP_SIZE);
                    pos.1 = pos.1.rem_euclid(MAP_SIZE);
                });
            },
        )
        .unwrap();

    // Recon: read-only query to spot enemies
    let recon_id = registry
        .register_query_ref::<(Entity, &Position, &Faction, &UnitType), (), _>(
            &mut world,
            "recon",
            |mut query: minkowski::QueryRef<'_, (Entity, &Position, &Faction, &UnitType)>, ()| {
                let count = query.count();
                let recon_count = count / 3; // rough estimate, every 3rd is recon
                println!("    [recon] scanning with ~{} recon units", recon_count);
            },
        )
        .unwrap();

    // Census: read-only faction/health tally
    let census_id = registry
        .register_query_ref::<(&Faction, &Health), (), _>(
            &mut world,
            "census",
            |mut query: minkowski::QueryRef<'_, (&Faction, &Health)>, ()| {
                let mut blue_count = 0u32;
                let mut red_count = 0u32;
                let mut blue_hp = 0u32;
                let mut red_hp = 0u32;
                query.for_each(|(f, h)| {
                    if f.0 == BLUE {
                        blue_count += 1;
                        blue_hp += h.0;
                    } else {
                        red_count += 1;
                        red_hp += h.0;
                    }
                });
                println!(
                    "    [census] blue: {} units ({}hp) | red: {} units ({}hp)",
                    blue_count, blue_hp, red_count, red_hp
                );
            },
        )
        .unwrap();

    // -- HashIndex for faction lookups --
    let mut faction_index = HashIndex::<Faction>::new();
    faction_index.rebuild(&mut world);

    // -- Spatial index --
    let mut unit_grid = UnitGrid::new(COMBAT_RANGE);

    // -- Spawn operator threads --
    let thread_a = std::thread::spawn({
        let blue_bits = blue_bits.clone();
        let red_bits = red_bits.clone();
        let hvt = high_value_target;
        move || {
            operator_thread(
                "OpA(Blue)",
                BLUE,
                cmd_tx_a,
                repl_rx_a,
                blue_bits,
                red_bits,
                hvt,
            );
        }
    });

    let thread_b = std::thread::spawn({
        let blue_bits = blue_bits.clone();
        let red_bits = red_bits.clone();
        let hvt = high_value_target;
        move || {
            operator_thread(
                "OpB(Red)", RED, cmd_tx_b, repl_rx_b, red_bits, blue_bits, hvt,
            );
        }
    });

    // -- Server tick loop --
    let mut total_commands = 0u32;
    let mut total_conflicts = 0u32;
    let mut total_kills = 0u32;

    // Use retries=1 for some transactions to demonstrate Conflict
    let strategy = Optimistic::new(&world);
    let conflict_strategy = Optimistic::with_retries(&world, 1);
    let attack_access = Access::of::<(&mut Health,)>(&mut world);

    for tick in 0..TICK_COUNT {
        println!("\n--- Server Tick {} ---", tick);

        // Phase 1: Drain commands
        let commands: Vec<(u8, Command)> = cmd_rx.try_iter().collect();
        total_commands += commands.len() as u32;

        let mut frame_changeset = EnumChangeSet::new();
        let mut frame_events: Vec<ReplicationEvent> = Vec::new();

        // Phase 2: Process commands via Optimistic transactions
        // Collect HVT attacks to demonstrate Conflict
        let hvt_entity = Entity::from_bits(high_value_target);
        let hvt_attacks: Vec<_> = commands
            .iter()
            .filter(|(_, cmd)| matches!(cmd, Command::Attack { target_unit, .. } if *target_unit == high_value_target))
            .collect();

        // If multiple attacks target the HVT, demonstrate conflict by forcing
        // a concurrent modification inside the transaction closure
        if hvt_attacks.len() >= 2 && world.is_alive(hvt_entity) {
            // First attack succeeds normally
            let _ = strategy.transact(&mut world, &attack_access, |tx, world| {
                if let Some(h) = world.get::<Health>(hvt_entity) {
                    let new_hp = h.0.saturating_sub(ATTACK_DAMAGE);
                    tx.write(world, hvt_entity, Health(new_hp));
                }
            });
            let hp_val = world.get::<Health>(hvt_entity).map_or(0, |h| h.0);
            frame_changeset.insert::<Health>(&mut world, hvt_entity, Health(hp_val));
            println!(
                "    attack: HVT {} (damage {}, hp now {})",
                high_value_target, ATTACK_DAMAGE, hp_val
            );

            // Second attack: use retries=1 and mutate the column inside the closure
            // to force a conflict (simulates concurrent modification)
            match conflict_strategy.transact(&mut world, &attack_access, |tx, world| {
                // Read target health
                if let Some(h) = world.get::<Health>(hvt_entity) {
                    let new_hp = h.0.saturating_sub(ATTACK_DAMAGE);
                    // Simulate concurrent modification: directly mutate the column
                    // This advances the column tick, causing validation to fail
                    // Touch mutable to advance column tick (simulates concurrent mod)
                    let _ = world.get_mut::<Health>(hvt_entity);
                    tx.write(world, hvt_entity, Health(new_hp));
                }
            }) {
                Ok(()) => {
                    println!("    attack: HVT {} second hit succeeded", high_value_target);
                }
                Err(conflict) => {
                    total_conflicts += 1;
                    println!("    CONFLICT on HVT: {}", conflict.display_with(&world));
                }
            }
        }

        for (_operator_faction, cmd) in &commands {
            match cmd {
                Command::Attack {
                    attacker,
                    target_unit,
                } => {
                    // Skip HVT attacks -- already handled above
                    if *target_unit == high_value_target {
                        continue;
                    }

                    let _atk_entity = Entity::from_bits(*attacker);
                    let tgt_entity = Entity::from_bits(*target_unit);

                    if !world.is_alive(tgt_entity) {
                        println!("    attack: target {} already dead, skipping", target_unit);
                        continue;
                    }

                    match strategy.transact(&mut world, &attack_access, |tx, world| {
                        if let Some(h) = world.get::<Health>(tgt_entity) {
                            let new_hp = h.0.saturating_sub(ATTACK_DAMAGE);
                            tx.write(world, tgt_entity, Health(new_hp));
                        }
                    }) {
                        Ok(()) => {
                            let hp_val = world.get::<Health>(tgt_entity).map_or(0, |h| h.0);
                            frame_changeset.insert::<Health>(
                                &mut world,
                                tgt_entity,
                                Health(hp_val),
                            );
                            println!(
                                "    attack: {} -> {} (damage {}, hp now {})",
                                attacker, target_unit, ATTACK_DAMAGE, hp_val
                            );
                        }
                        Err(conflict) => {
                            total_conflicts += 1;
                            println!("    CONFLICT: {}", conflict.display_with(&world));
                        }
                    }
                }
                Command::Move { unit, target } => {
                    let entity = Entity::from_bits(*unit);
                    if world.is_alive(entity) {
                        world.insert_sparse(
                            entity,
                            MoveOrder {
                                target_x: target.0,
                                target_y: target.1,
                            },
                        );
                        println!(
                            "    move order: {} -> ({:.0}, {:.0})",
                            unit, target.0, target.1
                        );
                    }
                }
            }
        }

        // Phase 3: Simulation
        // Movement reducer (uses query internally)
        registry.run(&mut world, movement_id, ()).unwrap();

        // par_for_each demo: parallel position clamping pass
        world.query::<(&mut Position,)>().par_for_each(|(pos,)| {
            pos.0 = pos.0.clamp(0.0, MAP_SIZE);
            pos.1 = pos.1.clamp(0.0, MAP_SIZE);
        });

        // Apply MoveOrder steering (sparse components -- sequential pass)
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
                    } else {
                        pos.0 += dx / dist * MOVE_SPEED;
                        pos.1 += dy / dist * MOVE_SPEED;
                    }
                }
            }
        }

        // Rebuild spatial index for combat + recon
        unit_grid.rebuild(&mut world);

        // Combat: find nearby enemies and apply damage
        {
            let mut damage_list: Vec<(Entity, u32)> = Vec::new();
            for &(entity, x, y, faction) in &unit_grid.snapshot {
                let nearby = unit_grid.neighbors(x, y, COMBAT_RANGE);
                for (other_e, _, _, other_f) in nearby {
                    if other_f != faction && entity != other_e {
                        damage_list.push((other_e, ATTACK_DAMAGE / 5)); // light auto-combat
                        break; // only damage one enemy per unit per tick
                    }
                }
            }
            for (target, damage) in &damage_list {
                if let Some(h) = world.get_mut::<Health>(*target) {
                    h.0 = h.0.saturating_sub(*damage);
                }
                let hp_val = world.get::<Health>(*target).map_or(0, |h| h.0);
                frame_changeset.insert::<Health>(&mut world, *target, Health(hp_val));
            }
        }

        // Despawn dead units via register_entity_despawn reducer
        let dead_units: Vec<Entity> = world
            .query::<(Entity, &Health)>()
            .filter(|(_, h)| h.0 == 0)
            .map(|(e, _)| e)
            .collect();
        for target in &dead_units {
            match registry.call(&strategy, &mut world, combat_cleanup_id, (*target, ())) {
                Ok(()) => {
                    total_kills += 1;
                    frame_changeset.record_despawn(*target);
                    println!("    kill: entity {:?} despawned", target);
                }
                Err(err) => {
                    println!("    despawn error: {err}");
                }
            }
        }

        // Fallback via CommandBuffer: catch any dead units the reducer missed
        // (In practice the reducer handles all, but this demonstrates CommandBuffer)
        let mut cmds = CommandBuffer::new();
        let more_dead: Vec<Entity> = world
            .query::<(Entity, &Health)>()
            .filter(|(_, h)| h.0 == 0)
            .map(|(e, _)| e)
            .collect();
        for target in &more_dead {
            cmds.despawn(*target);
            frame_changeset.record_despawn(*target);
            total_kills += 1;
        }
        cmds.apply(&mut world);

        // Recon scan: spot enemies using spatial grid
        {
            let recon_data: Vec<(Entity, f32, f32, u8)> = world
                .query::<(Entity, &Position, &Faction, &UnitType)>()
                .filter(|(_, _, _, ut)| ut.0 == 2)
                .map(|(e, p, f, _)| (e, p.0, p.1, f.0))
                .collect();
            let mut spotted_count = 0u32;
            for (_, x, y, my_faction) in &recon_data {
                let nearby = unit_grid.neighbors(*x, *y, RECON_RANGE);
                for (enemy_e, _, _, enemy_f) in nearby {
                    if enemy_f != *my_faction {
                        world.insert_sparse(
                            enemy_e,
                            IntelReport {
                                spotted_tick: tick as u64,
                                confidence: 0.8 + fastrand::f32() * 0.2,
                            },
                        );
                        spotted_count += 1;
                    }
                }
            }
            if spotted_count > 0 {
                println!("    [recon] spotted {} enemy contacts", spotted_count);
            }
        }

        // Run registered recon + census reducers
        registry.run(&mut world, recon_id, ()).unwrap();
        registry.run(&mut world, census_id, ()).unwrap();

        // Phase 4: HashIndex with stale validation
        faction_index.rebuild(&mut world);
        let blue_alive: Vec<Entity> = faction_index.get_valid(&Faction(BLUE), &world).collect();
        let red_alive: Vec<Entity> = faction_index.get_valid(&Faction(RED), &world).collect();
        println!(
            "  faction index (stale-filtered): blue={} red={}",
            blue_alive.len(),
            red_alive.len()
        );

        // Phase 5: World introspection (every 3 ticks)
        if tick % 3 == 0 {
            println!("  [introspection] archetypes: {}", world.archetype_count());
            for arch_idx in 0..world.archetype_count() {
                let len = world.archetype_len(arch_idx);
                if len == 0 {
                    continue;
                }
                let comp_ids = world.archetype_component_ids(arch_idx);
                let names: Vec<&str> = comp_ids
                    .iter()
                    .filter_map(|id| world.component_name(*id))
                    .collect();
                println!(
                    "    arch[{}]: {} entities, components: {:?}",
                    arch_idx, len, names
                );
            }
        }

        // Phase 6: Build replication from EnumChangeSet + MutationRef
        let changeset_events = changeset_to_events(&frame_changeset, &world);
        frame_events.extend(changeset_events);

        // Demonstration: iterate MutationRef directly for logging
        let mut insert_count = 0u32;
        let mut despawn_count = 0u32;
        for mutation in frame_changeset.iter_mutations() {
            match mutation {
                MutationRef::Insert { .. } => insert_count += 1,
                MutationRef::Despawn { .. } => despawn_count += 1,
                MutationRef::Spawn { .. }
                | MutationRef::Remove { .. }
                | MutationRef::SparseInsert { .. }
                | MutationRef::SparseRemove { .. } => {}
            }
        }
        if insert_count > 0 || despawn_count > 0 {
            println!(
                "  [changeset] {} inserts, {} despawns recorded this tick",
                insert_count, despawn_count
            );
        }

        // Send replication to both operators
        let _ = repl_tx_a.send(frame_events.clone());
        let _ = repl_tx_b.send(frame_events);

        // iter_sparse demo: count intel reports on server
        if let Some(intel_id) = world.component_id::<IntelReport>() {
            let intel_count = world
                .iter_sparse::<IntelReport>(intel_id)
                .map_or(0, Iterator::count);
            if intel_count > 0 {
                println!("  [server] {} active intel reports (sparse)", intel_count);
            }
        }

        // No delay needed — operators block on recv() until we send
    }

    // Drop senders so operator threads will finish
    drop(repl_tx_a);
    drop(repl_tx_b);

    // Wait for operators
    thread_a.join().unwrap();
    thread_b.join().unwrap();

    // -- Final summary --
    println!("\n=== Final Summary ===");
    println!("  ticks: {}", TICK_COUNT);
    println!("  total commands processed: {}", total_commands);
    println!("  total conflicts: {}", total_conflicts);
    println!("  total kills: {}", total_kills);

    let blue_final = world
        .query::<(&Faction,)>()
        .filter(|(f,)| f.0 == BLUE)
        .count();
    let red_final = world
        .query::<(&Faction,)>()
        .filter(|(f,)| f.0 == RED)
        .count();
    println!("  surviving: blue={} red={}", blue_final, red_final);
    println!("  elapsed: {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Final Entity::to_bits / from_bits round-trip verification
    for entity in world.query::<(Entity,)>().map(|(e,)| e).take(3) {
        let bits = entity.to_bits();
        let restored = Entity::from_bits(bits);
        assert_eq!(entity, restored, "entity bit round-trip failed");
        // Demonstrate index() and generation() accessors
        println!(
            "    entity {:?}: index={}, generation={}, bits=0x{:016x}",
            entity,
            entity.index(),
            entity.generation(),
            bits
        );
    }
    println!("  entity bit-packing round-trip: OK");

    println!("\nDone.");
}
