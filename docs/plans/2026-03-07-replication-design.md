# Replication & Sync Design

## Goal

Pull-based, transport-agnostic replication primitives for read replicas and (future) client mirrors. The engine provides mechanisms; users provide policy (transport, pull frequency, filtering, freshness guarantees).

## Architecture

Three new public types in `minkowski-persist::replication`:

- **`WalCursor`** — Read-only cursor over a WAL file, yielding records from a given seq onward.
- **`ReplicationBatch`** — Self-describing, rkyv-serializable wire type carrying a schema + records.
- **`apply_batch()`** — Standalone function that applies a batch to a target World.

A read replica is: snapshot load -> pull loop -> apply. Filtering (for client mirrors) is a future transform over the batch before apply.

## WalCursor

Read-only view over a WAL file. Opens its own file handle, so it can read concurrently with an active writer.

```rust
pub struct WalCursor {
    file: File,                    // read-only handle
    pos: u64,                      // byte offset into WAL file
    next_seq: u64,                 // next expected seq
    schema: Option<WalSchema>,     // parsed from preamble
}
```

### Construction

`WalCursor::open(path, from_seq)` — Opens the WAL file read-only, reads the schema preamble (if present), scans forward to the first record with `seq >= from_seq`. Returns `Err(WalError::CursorBehind { requested, oldest })` if the WAL has been truncated past the requested seq (future hook for WAL rotation — not triggered today).

### Reading

- `cursor.next_batch(limit) -> Result<ReplicationBatch, WalError>` — Reads up to `limit` records from the current position. Returns a batch with the schema and records. Empty `records` vec means caught up. Advances `pos` and `next_seq`.
- `cursor.schema() -> Option<&WalSchema>` — Schema from the preamble.
- `cursor.next_seq() -> u64` — Next expected seq (for cursor persistence on restart).

### File reading

Reuses the existing `read_next_entry` frame logic from `wal.rs`, extracted into a shared helper. Both `WalCursor` and `Wal::replay_from` use the same code path.

## ReplicationBatch

Self-describing, serializable payload. The unit of exchange between source and sink.

```rust
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ReplicationBatch {
    pub schema: WalSchema,
    pub records: Vec<WalRecord>,
}
```

### Properties

- Every batch carries its own schema — no handshake, no statefulness, sniffable on the wire for debugging.
- `records` is ordered by seq, monotonically increasing.
- Empty `records` is a valid batch (caught up).
- rkyv-serializable. Users can also use their own serialization.

### Convenience methods

- `batch.to_bytes() -> Result<Vec<u8>, WalError>` — wraps `rkyv::to_bytes`.
- `ReplicationBatch::from_bytes(bytes) -> Result<Self, WalError>` — wraps `rkyv::from_bytes`.

These are on the public wire type because every consumer needs them and reimplementing the error mapping is busy work. The fact that rkyv is the underlying format is still an implementation detail.

## apply_batch

Standalone function that applies a `ReplicationBatch` to a target World.

```rust
pub fn apply_batch(
    batch: &ReplicationBatch,
    world: &mut World,
    codecs: &CodecRegistry,
) -> Result<u64, WalError>
```

### Behavior

1. Build remap from `batch.schema` via `codecs.build_remap()`.
2. For each `WalRecord` in `batch.records`, apply each `SerializedMutation`:
   - `Spawn` -> `alloc_entity` + `record_spawn` with sender's entity and deserialized components.
   - `Insert` -> `record_insert` with remapped component ID.
   - `Remove` -> `record_remove` with remapped component ID.
   - `Despawn` -> `record_despawn`.
3. Apply one `EnumChangeSet` per record (preserves per-transaction atomicity).
4. Return the last applied seq.

### Refactoring

The existing `apply_record` helper in `wal.rs` contains this logic. It is extracted as a shared function that both `Wal::replay_from` and `apply_batch` call.

## Error handling

New error variant:

```rust
WalError::CursorBehind { requested: u64, oldest: u64 }
```

Returned when the cursor's starting seq precedes the oldest record in the WAL. Hook for future WAL rotation support — not triggered today since there is no rotation.

## Consistency model

Eventually consistent. The WAL seq is monotonic, so replicas see a consistent prefix of the mutation history. Each `WalRecord` is one committed transaction, applied atomically. Freshness is determined by the caller's pull frequency and apply policy, not by the engine.

## Entity ID allocation

Replicas trust the source's entity IDs. `Spawn` mutations carry the allocated entity bits; the replica places entities at those exact IDs. Snapshot load restores allocator state, and sequential WAL replay maintains it.

## Integration

### Changes to wal.rs

- Extract `read_next_entry` frame logic into a shared helper.
- Extract `apply_record` into a shared function.
- `Wal::replay_from` becomes a thin wrapper: open cursor, read all, apply all. Signature unchanged.

### New file

`crates/minkowski-persist/src/replication.rs` — contains `WalCursor`, `ReplicationBatch`, `apply_batch`.

### Public API additions

```rust
pub mod replication;
pub use replication::{apply_batch, ReplicationBatch, WalCursor};
```

### No changes to

`codec.rs`, `snapshot.rs`, `durable.rs`, `record.rs`, core `minkowski` crate.

### New example

`examples/examples/replicate.rs` — full flow: create world, write mutations via Durable, cursor pulls batches, applies to a second world, verifies convergence. In-process (no transport).

## Future work (not in this PR)

- **Filtered replication** — Transform over `ReplicationBatch` before apply, filtering by component set or entity set. Client mirrors are the use case.
- **WAL rotation** — Snapshot + truncate old WAL records. `CursorBehind` error tells replicas to re-bootstrap from snapshot.
- **Push notification** — Source signals "new data available" so replicas don't have to poll. Thin layer over pull.

## Testing

### Unit tests in replication.rs

- `cursor_reads_from_seq_zero` — read all records from a WAL with 3 mutations.
- `cursor_reads_from_mid_seq` — skip earlier records.
- `cursor_at_end_returns_empty_batch` — caught up yields empty records.
- `cursor_behind_error` — placeholder for the `CursorBehind` variant.
- `batch_round_trip` — `to_bytes` / `from_bytes` survives.
- `apply_batch_spawns_entities` — Spawn mutations create entities in target world.
- `apply_batch_insert_remove` — Insert + Remove applied correctly.
- `apply_batch_cross_process_remap` — different registration order resolved by stable name.
- `apply_batch_preserves_transaction_boundaries` — per-record atomicity.

### Integration

- `full_replication_flow` — Durable source -> snapshot -> cursor pull -> apply to replica -> verify convergence.

### Regression

- All existing `wal.rs` and `snapshot.rs` tests pass unchanged.
