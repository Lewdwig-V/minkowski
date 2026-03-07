# ADR-015: Pull-Based WAL Replication

**Status:** Accepted
**Date:** 2026-03-07

## Context

Distributing ECS state changes to replicas (read replicas, external observers, network peers) requires a replication mechanism. The WAL already contains a complete, ordered record of every committed mutation. The question is how replicas consume it.

## Decision

Replication is pull-based: replicas open a read-only `WalCursor` over the WAL directory and poll for new records via `next_batch(limit)`. No push channel, no subscription, no consumer tracking in the engine. The cursor is a file-level reader that shares no state with the writer — concurrent reads and writes are safe because the writer only appends and the cursor opens its own file handles.

### WalCursor

`WalCursor::open(dir, from_seq)` finds the segment containing `from_seq` and scans forward to position the cursor. `next_batch(limit)` returns a `ReplicationBatch` containing the schema and up to `limit` mutation records. The cursor lazily advances across segment boundaries via `try_advance_segment()` — it scans the directory for the next segment file only when the current one is exhausted.

`CursorBehind` is returned when `from_seq` is behind all remaining segments (i.e., the segments containing that sequence were deleted). This tells the replica it needs to re-bootstrap from a snapshot.

### ReplicationBatch

Every batch carries its own `WalSchema` so receivers can decode without prior handshake. `apply_batch()` builds a component ID remap from the schema and applies each record as its own `EnumChangeSet`. Per-record atomicity, not per-batch — on error, previously applied records are not rolled back.

### Checkpoint and schema entries

`WalEntry::Checkpoint` and `WalEntry::Schema` entries are transparently skipped by the cursor — only `Mutations` records appear in batches. Schema entries are consumed internally for ID remapping.

## Alternatives Considered

- Push-based replication (writer notifies subscribers) — requires consumer tracking in the engine, complicates the writer's hot path, introduces backpressure concerns
- Shared cursor state between reader and writer — risks contention on the write path
- Network-layer replication protocol — too opinionated for a storage engine; the batch format is wire-ready (`to_bytes`/`from_bytes` via rkyv) but transport is the framework's responsibility
- Full-state sync on every change — O(n) per mutation, same problem as full-world serialization

## Consequences

- Replicas poll at their own cadence — no backpressure on the writer
- `CursorBehind` forces snapshot-based re-bootstrap after aggressive WAL truncation — the engine does not retain segments for slow consumers
- Cross-process ID remapping via schema preambles means source and replica can register components in different order
- `ReplicationBatch` is the network-ready unit — frameworks serialize it via `to_bytes()` and transmit over any transport
- Cursor holds one open file handle at a time — no resource accumulation across segments
- No exactly-once delivery guarantee at the engine level — idempotency is the framework's responsibility (sequence numbers enable dedup)
