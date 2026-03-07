# Tactical Map Replication Example

**Date:** 2026-03-06
**Status:** Approved

## Purpose

Add a `tactical` example that exercises public API surface gaps not covered by existing examples. The example simulates a multi-operator tactical map where a server thread maintains an authoritative world and replicates state changes to client threads via channels.

## API Gaps Covered

| Gap | API items | How exercised |
|---|---|---|
| Sparse components | `register_sparse`, `insert_sparse`, `iter_sparse` | `IntelReport` on spotted enemies (~5%), `MoveOrder` on units with active orders |
| `par_for_each` | `QueryIter::par_for_each` | Server movement reducer parallelized via rayon |
| Conflict inspection | `Conflict` type, display | Two operators command same unit; server logs conflict details |
| Entity bit packing | `Entity::to_bits()`, `Entity::from_bits()` | Commands and replication packets serialize entity IDs as u64 |
| World introspection | `archetype_count()`, `component_name()` | Server logs archetype/component stats periodically |
| `register_entity_despawn` | `EntityMut` with despawn capability | Combat reducer despawns dead units |
| Stale index validation | `HashIndex::get_valid()` | Faction index filters stale entries after combat despawns |
| `EnumChangeSet` iteration | `MutationRef` pattern matching | Server iterates changeset to build replication packets |

## Architecture

### Threads

- **Server thread** -- owns the authoritative `World`. Runs a tick loop: receive commands, apply via transactions, run simulation, replicate.
- **Operator A thread** -- owns a local client `World`. Sends commands, receives replication packets.
- **Operator B thread** -- owns a local client `World`. Sends commands, receives replication packets.

### Channel Topology

```
Operator A  --cmd-->  Server  --replication-->  Operator A
Operator B  --cmd-->  Server  --replication-->  Operator B
```

Each operator has an `mpsc` command channel (operator->server) and an `mpsc` replication channel (server->operator). Commands and replication packets carry entity IDs as `u64` via `to_bits()`/`from_bits()`.

## Data Model

### Archetype Components

- `Position(f32, f32)` -- map coordinates
- `Heading(f32)` -- facing direction (radians)
- `Faction(u8)` -- 0 = Blue, 1 = Red
- `Health(u32)` -- hit points
- `Speed(f32)` -- movement speed per tick
- `UnitType(u8)` -- Infantry (0), Armor (1), Recon (2)

### Sparse Components

- `IntelReport { spotted_tick: u64, confidence: f32 }` -- attached to enemy units spotted by recon. Genuinely sparse: most enemies are unspotted.
- `MoveOrder { target: (f32, f32) }` -- active move command on a unit. Inserted on command, removed on arrival.

### Wire Types

```rust
enum Command {
    Move { unit: u64, target: (f32, f32) },
    Attack { attacker: u64, target_unit: u64 },
    Spot { unit: u64, confidence: f32 },
}

// Replication packets built by iterating EnumChangeSet via MutationRef
struct ReplicationPacket {
    mutations: Vec<SerializedMutation>,
}

struct SerializedMutation {
    entity_bits: u64,
    kind: MutationKind,
}
```

## Server Tick Loop (10 frames)

### Phase 1: Receive Commands

Drain both operator command channels (non-blocking `try_recv` loop).

### Phase 2: Apply Commands as Transactions

Each command becomes an `Optimistic` transaction:

- `Move` -- `insert_sparse` a `MoveOrder` on the unit
- `Attack` -- read attacker + target `Health`, buffer damage write
- `Spot` -- `insert_sparse` an `IntelReport` on the enemy

On `Conflict`: log with `Conflict` display, send rejection message back to the operator. Engineered scenario: both operators command the same unit on the same tick.

### Phase 3: Simulation Reducers (ReducerRegistry)

1. **`movement`** (`QueryMut` + `par_for_each`) -- units with sparse `MoveOrder` advance toward target. Remove order on arrival.
2. **`combat`** (`register_entity_despawn`) -- units in range of enemies deal damage. Despawn units with Health <= 0.
3. **`recon_scan`** (`QueryRef`) -- recon units auto-spot nearby enemies within scan radius, attaching sparse `IntelReport`.
4. **`census`** (`QueryRef`) -- count units per faction, log status.

### Phase 4: Build Replication Packets

Take the `EnumChangeSet` from the frame. Iterate `MutationRef` variants via pattern matching. Serialize each mutation with `to_bits()` entity IDs. Send packets to both operator channels.

### Phase 5: World Introspection

Every few frames, log `archetype_count()` and `component_name()` stats.

## Operator Threads

- Receive replication packets, reconstruct mutations using `Entity::from_bits()`
- Apply to local world via direct world methods
- Every few frames, send a random command (move a friendly unit, attack a nearby enemy)
- Periodically use `iter_sparse::<IntelReport>()` to list spotted enemies

## Spatial & Column Indexes

- Server maintains a `SpatialIndex` grid (cell size = combat range) for proximity checks in combat and recon phases.
- Server maintains a `HashIndex<Faction>` for fast faction lookups. After `register_entity_despawn` kills units, demonstrate `get_valid()` to filter stale entries from the faction index.

## Scale

- 50 Blue units, 50 Red units (100 total)
- 10 server ticks
- 2 operator threads
- Map size: 1000x1000

## Output

Each frame prints:
- Commands received and applied/rejected (with Conflict details)
- Combat events (damage, kills)
- Census (units per faction, total health)
- Periodic archetype/component introspection stats
- Operator-side: spotted enemy count from sparse iteration

Final summary: total commands, conflicts, kills, surviving units per faction.
