# ADR-008: WAL + Snapshot Persistence

**Status:** Accepted
**Date:** 2026-03-02

## Context

Crash-safe persistence requires durability guarantees without serializing the entire world on every write. The persistence layer must compose with the existing transaction system rather than replacing it, and support arbitrary component types through a codec registry.

## Decision

`Durable<S>` wraps any `Transact` strategy, appending the forward changeset to a write-ahead log on successful commit. `Snapshot` captures full world state for point-in-time recovery. Recovery loads the latest snapshot and replays the WAL from its sequence number. `CodecRegistry` maps `ComponentId` to rkyv-based codecs for serialization (migrated from serde/bincode in ADR-013). WAL write failure panics — the durability invariant is non-negotiable.

**Key insight: persistence composes with any transaction strategy — `Durable::new(strategy, wal, codecs)` adds durability without changing the strategy's conflict detection semantics.**

## Alternatives Considered

- Full-world serialization per write — O(n) cost per mutation, unacceptable for large worlds
- Memory-mapped files (mmap) — crash consistency is hard to guarantee, TLB pressure with many columns
- Custom binary format without WAL — no crash recovery, requires full flush on every commit

## Consequences

- WAL write failure panics rather than returning an error — partial durability is worse than none
- Schema changes (new component types) require migration since codecs are registered per ComponentId
- Snapshot + WAL sequence number enables point-in-time recovery
- `EnumChangeSet` is the serialization unit — the same type used for transactions and undo/redo
- `sync_reserved()` must be called after snapshot restore to prevent entity ID collisions with `reserve()
