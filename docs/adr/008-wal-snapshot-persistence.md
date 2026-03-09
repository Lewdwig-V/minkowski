# ADR-008: WAL + Snapshot Persistence

**Status:** Accepted
**Date:** 2026-03-02
**Updated:** 2026-03-09 — CRC32 frame checksums, snapshot integrity, generation validation

## Context

Crash-safe persistence requires durability guarantees without serializing the entire world on every write. The persistence layer must compose with the existing transaction system rather than replacing it, and support arbitrary component types through a codec registry.

## Decision

`Durable<S>` wraps any `Transact` strategy, appending the forward changeset to a write-ahead log on successful commit. `Snapshot` captures full world state for point-in-time recovery. Recovery loads the latest snapshot and replays the WAL from its sequence number. `CodecRegistry` maps `ComponentId` to rkyv-based codecs for serialization (migrated from serde/bincode in ADR-013). WAL write failure panics — the durability invariant is non-negotiable.

**Key insight: persistence composes with any transaction strategy — `Durable::new(strategy, wal, codecs)` adds durability without changing the strategy's conflict detection semantics.**

### Segmented WAL

The WAL is a directory of segment files (`wal-seq{start:06}.seg`), each with a self-describing schema preamble. Segments roll over when they exceed `WalConfig::max_segment_bytes` (default 64 MB). Callers truncate old segments via `Wal::delete_segments_before(seq)` after taking a snapshot — the engine provides the mechanism, not the retention policy.

### Checkpoint markers

`WalEntry::Checkpoint { snapshot_seq }` is a metadata entry written to the WAL stream by `Wal::acknowledge_snapshot()`. It resets the `bytes_since_checkpoint` counter. `Wal::checkpoint_needed()` returns true when accumulated mutation bytes exceed `WalConfig::max_bytes_between_checkpoints`. On `open()`, checkpoint state is recovered from both the active segment and sealed segments.

`CheckpointHandler` is a callback trait invoked by `Durable::with_checkpoint()` after each successful commit when the threshold is exceeded. `on_checkpoint_needed` returns `Result` — checkpoint failure is non-fatal (the transaction is already committed and applied). `AutoCheckpoint` is the batteries-included default: saves a snapshot to `snap_dir/checkpoint-{seq:06}.snap` and acknowledges it.

### Error classification

Every WAL error path is evaluated against "is the mutation durable?" If yes, the error is operational (retry, degrade). If no, the error is fatal (panic). Segment rollover failure after a successful append is non-fatal — the mutation is durable in an oversized segment. Checkpoint handler errors are non-fatal — the transaction is already committed. `roll_segment` is atomic with respect to `Wal`'s internal state — all I/O completes before fields are updated.

### Integrity checking

WAL frames use CRC32 checksums (IEEE via `crc32fast`, hardware-accelerated on SSE 4.2 / AArch64). Frame format: `[len: u32 LE][crc32: u32 LE][payload: len bytes]`. The CRC covers the rkyv payload and catches silent data corruption that rkyv validation alone might miss. On read, checksum mismatches are treated identically to torn writes — the corrupt frame and everything after it is truncated during crash recovery.

Snapshot files use the same CRC32 over the rkyv payload, stored in the v2 envelope header: `[magic: 8B "MK2SNAPK"][crc32: 4B LE][reserved: 4B][len: u64 LE][payload]`. Checksum mismatches return `SnapshotError::Format` — no silent data loss. Legacy v1 snapshots (no magic, no CRC) are accepted for backward compatibility but skip verification.

Segment files use a 4-byte magic (`"MKW2"`) to distinguish v2 format from legacy v1 (no checksums). Legacy segments produce a hard error with a migration message — they are never silently reinterpreted.

After snapshot restore, an entity generation consistency check validates that every entity in every archetype has a generation matching the restored allocator state. A corrupt snapshot where allocator generations diverge from archetype data would silently poison `is_alive()` — this returns `SnapshotError::Format` at load time, consistent with all other corruption detection in the snapshot path.
## Alternatives Considered

- Full-world serialization per write — O(n) cost per mutation, unacceptable for large worlds
- Memory-mapped files (mmap) — crash consistency is hard to guarantee, TLB pressure with many columns
- Custom binary format without WAL — no crash recovery, requires full flush on every commit
- Single-file WAL — no way to truncate old entries without rewriting the file; segmented WAL enables O(1) deletion of old segments
- Engine-managed WAL retention — rejected in favor of caller-driven `delete_segments_before()` (mechanism, not policy)
- Checkpoint as optimization (skip-ahead on replay) — rejected; checkpoint is a safety net against unbounded WAL growth, not a replay optimization

## Consequences

- WAL write failure panics rather than returning an error — partial durability is worse than none
- Schema changes (new component types) require migration since codecs are registered per ComponentId
- Snapshot + WAL sequence number enables point-in-time recovery
- `EnumChangeSet` is the serialization unit — the same type used for transactions and undo/redo
- `sync_reserved()` must be called after snapshot restore to prevent entity ID collisions with `reserve()`
- Every segment is self-describing (schema preamble) — cross-process replay with different component registration order works via ID remapping
- Crash recovery on `open()` truncates torn/corrupt tail of the active segment and rewrites the schema preamble if it was destroyed
- `next_seq` never regresses — falls back to `active_start_seq` when no mutations are found after segment truncation
