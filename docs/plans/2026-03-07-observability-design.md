# Observability Companion Crate — Design

**Date:** 2026-03-07
**Status:** Approved

## Goal

Add observability to Minkowski without instrumenting hot paths. A companion crate (`minkowski-observe`) polls read-only stats facades, diffs consecutive snapshots, and computes rates. A separate TUI crate (`minkowski-tui`) renders a live dashboard via ratatui. Delivered as two PRs.

## Principles

- **Pure consumer** — observe crate has no write path into the engine.
- **Minimal core changes** — two `stats()` methods returning plain data structs. No counters on hot paths.
- **Facade, not accessors** — `WorldStats` and `WalStats` copy values out. No `pub(crate)` internals leak into the public API.
- **Framework metrics are the framework's problem** — we expose what the engine knows (cursor position, WAL seq). We don't measure network latency, conflict rates across clients, or anything that requires transport awareness.

## PR 1: `minkowski-observe`

### Core crate changes

#### `WorldStats`

Returned by `world.stats()`. All fields copied from internal state.

| Field | Type | Source |
|---|---|---|
| `entity_count` | `usize` | existing `entity_count()` |
| `archetype_count` | `usize` | existing `archetype_count()` |
| `component_count` | `usize` | existing `component_count()` |
| `free_list_len` | `usize` | `entity_allocator_state().1.len()` |
| `query_cache_len` | `usize` | `self.query_cache.len()` |
| `current_tick` | `u64` | `self.current_tick.0` |

Struct is `#[derive(Clone, Copy, Debug, PartialEq)]`.

#### `WalStats`

Returned by `wal.stats()`. Composes existing public accessors plus one promoted field.

| Field | Type | Source |
|---|---|---|
| `next_seq` | `u64` | existing `next_seq()` |
| `segment_count` | `usize` | existing `segment_count()` |
| `oldest_seq` | `Option<u64>` | existing `oldest_seq()` |
| `bytes_since_checkpoint` | `u64` | `self.bytes_since_checkpoint` (currently `pub(crate)`) |
| `last_checkpoint_seq` | `Option<u64>` | existing `last_checkpoint_seq()` |
| `checkpoint_needed` | `bool` | existing `checkpoint_needed()` |

Struct is `#[derive(Clone, Copy, Debug, PartialEq)]`.

#### Safety

Read-only value copies. No references to internal state, no `&mut` paths, no way to influence engine behavior. Promoting a `pub(crate)` field's *value* through a copy is categorically different from promoting the field itself.

### Observe crate

#### `MetricsSnapshot`

Point-in-time capture combining both stats plus per-archetype detail.

```rust
pub struct MetricsSnapshot {
    pub world: WorldStats,
    pub wal: Option<WalStats>,
    pub archetypes: Vec<ArchetypeInfo>,
    pub timestamp: Instant,
}

pub struct ArchetypeInfo {
    pub id: usize,
    pub entity_count: usize,
    pub component_names: Vec<&'static str>,
    pub estimated_bytes: usize,  // entity_count * sum(component layouts)
}
```

`MetricsSnapshot::capture(world: &World, wal: Option<&Wal>)` is the single entry point. Iterates archetypes via existing public methods (`archetype_count`, `archetype_entities`, `archetype_component_ids`, `component_name`, `component_layout`).

#### `MetricsDiff`

Computed from two consecutive snapshots. Pure function, no side effects.

```rust
pub struct MetricsDiff {
    pub elapsed: Duration,
    pub entity_delta: i64,
    pub entity_churn: u64,
    pub tick_delta: u64,
    pub wal_seq_delta: u64,
    pub archetype_delta: i64,
    pub largest_archetypes: Vec<(usize, usize)>,  // (id, entity_count) top N
}
```

`MetricsDiff::compute(before: &MetricsSnapshot, after: &MetricsSnapshot)` computes all deltas.

#### Entity churn estimation

No deep instrumentation. Inferred from facade values:

- `despawns ~ free_list_growth` (free list grows when entities are despawned)
- `spawns ~ entity_delta + despawns`
- `churn = |spawns| + |despawns|`

Not exact (free list recycling muddles it), but accurate enough without adding counters to hot paths.

#### Display

Human-readable table format via `Display` impl on both `MetricsSnapshot` and `MetricsDiff`. Suitable for logging or printing to stderr.

### Dependencies

`minkowski-observe` depends on `minkowski` and `minkowski-persist`. No external dependencies.

## PR 2: `minkowski-tui`

Separate binary crate. Stretch goal.

### Integration model

The TUI cannot hold `&World` across frames. Instead, the application sends snapshots through a channel and the TUI runs on a separate thread.

```rust
// Application side
let (tx, rx) = minkowski_observe::channel();
// In game loop:
tx.send(MetricsSnapshot::capture(&world, Some(&wal)));

// TUI side (separate thread)
minkowski_tui::run(rx)?;
```

### Panels

- **Archetype table**: ID, component names, entity count, estimated bytes — sorted by size
- **Entity stats**: total, free list, churn rate sparkline
- **WAL stats**: seq, segments, checkpoint pressure bar
- **Tick velocity**: sparkline

### Dependencies

`ratatui`, `crossterm`, `minkowski-observe`.

### Non-goals

- No remote/network mode (attach over TCP)
- No persistence of historical metrics
- No configuration file — command line flags only

## Alternatives considered

- **Deep instrumentation** (atomic counters on spawn/despawn/query) — accurate but adds overhead to every hot path. Rejected: the estimation approach is good enough.
- **`world.snapshot_metrics()` monolith** — single method returning everything. Rejected: grows World's API with observe-specific concerns. The facade pattern (`stats()`) is minimal and general.
- **Individual accessor methods** (`query_cache_len()`, `current_tick()`, etc.) — leaks implementation details. Rejected in favor of opaque stats struct.
- **Framework-injected metrics** (slot for sync latency, conflict rates) — overreach. Framework authors can instrument their own code.

## Consequences

- World gains one public method (`stats()`) and one public struct (`WorldStats`)
- Wal gains one public method (`stats()`) and one public struct (`WalStats`)
- Entity churn is estimated, not exact — documented limitation
- TUI requires a channel-based integration pattern — the engine never yields control to the dashboard
