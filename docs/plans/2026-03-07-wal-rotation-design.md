# Segmented WAL Rotation Design

**Goal:** Replace the single-file WAL with a directory of fixed-size segment files, enabling truncation of old entries without rewriting the active file.

**Architecture:** Byte-size-capped segment files with start-seq embedded in filenames. Each segment is self-describing (schema preamble). Truncation is caller-driven `delete_segments_before(seq)` — no consumer tracking or policy in the engine.

**Guiding principle:** Engine provides mechanism (segments, deletion, CursorBehind error), caller provides policy (when to snapshot, when to truncate, how to handle slow consumers).

---

## File Layout

```
my-world.wal/
  wal-seq000000.seg    # schema preamble + records 0..46
  wal-seq000047.seg    # schema preamble + records 47..122
  wal-seq000123.seg    # schema preamble + records 123..  (active)
```

- Each segment starts with a `WalEntry::Schema` preamble (self-describing, no cross-segment dependencies)
- Segments are sealed when `append()` would exceed `max_segment_bytes`
- Naming: `wal-seq{:06}.seg` — zero-padded start-seq, lexicographic sort = chronological order
- Only the last segment is open for writing

## Configuration

```rust
pub struct WalConfig {
    pub max_segment_bytes: usize,  // default 64MB
}
```

## Wal API Changes

```rust
pub struct Wal {
    dir: PathBuf,
    active_file: File,
    active_start_seq: u64,
    active_bytes: u64,
    next_seq: u64,
    config: WalConfig,
    codecs_snapshot: WalSchema,  // cached for writing preambles
}
```

### Methods

- **`create(dir, codecs, config)`** — creates the directory and first segment with schema preamble
- **`open(dir, codecs, config)`** — scans directory for segments, opens last for appending, recovers `next_seq`. Config governs future segment rollover
- **`append(changeset, codecs)`** — writes record, checks byte threshold, rolls to new segment if exceeded
- **`delete_segments_before(seq)`** — deletes all segment files whose entire seq range is below `seq`. For sorted segments `[s0, s1, s2, ...]`, segment `s_i` is safe to delete if `s_{i+1}.start_seq <= seq`. Returns number deleted
- **`segment_count()`** — number of segment files in directory
- **`oldest_seq()`** — start-seq of the oldest remaining segment

## WalCursor Changes

```rust
pub struct WalCursor {
    dir: PathBuf,
    file: File,
    pos: u64,
    next_seq: u64,
    schema: Option<WalSchema>,
    current_segment_start_seq: u64,
}
```

- **`open(dir, from_seq)`** — finds segment containing `from_seq` (largest start_seq <= from_seq). Returns `Err(CursorBehind { requested, oldest })` if all segments start after `from_seq`
- **`next_batch(limit)`** — reads from current segment. On EOF, lazy directory scan for next segment. Opens it, parses schema preamble, continues. No next segment = caught up (empty batch)
- Naturally picks up new segments created after `open()` (lazy scan on transition)

## Error Handling & Edge Cases

- **Crash recovery**: only the active (last) segment needs recovery. Sealed segments are fully written
- **Empty segments**: possible if first append exceeds byte limit. Cursors skip schema-only segments naturally
- **Schema evolution**: each new segment reflects the current `CodecRegistry`. Cursor already handles mid-stream schema updates
- **Concurrent writer + cursor**: separate file handles, no locking. Append-only + read-only
- **Deleted segment with open cursor handle**: on Unix, reads continue (unlinked but open fd). Document Windows limitation
- **Empty directory**: `Wal::open` and `WalCursor::open` return errors

## Breaking Changes

This is a breaking change to the WAL on-disk format. No migration path (pre-1.0, zero users).

**API signature changes:**
- `Wal::create(path, codecs)` → `Wal::create(dir, codecs, config)`
- `Wal::open(path, codecs)` → `Wal::open(dir, codecs, config)`
- `WalCursor::open(path, from_seq)` → `WalCursor::open(dir, from_seq)`

**New API:**
- `WalConfig`
- `Wal::delete_segments_before(seq) -> usize`
- `Wal::segment_count() -> usize`
- `Wal::oldest_seq() -> Option<u64>`

**Unchanged:**
- `WalEntry`, `WalRecord`, `WalSchema`, `SerializedMutation` (wire format)
- `read_next_frame`, `apply_record` (internal helpers)
- `ReplicationBatch`, `apply_batch` (replication API)
- Frame format `[len: u32 LE][payload]` within segments

**Callers to update:** `Durable<S>`, `examples/replicate.rs`, `examples/persist.rs`, all WAL/replication tests.

## What the engine does NOT provide

- Consumer tracking (transport-dependent)
- Snapshot-before-truncate policy (caller uses existing `Snapshot::save` + `delete_segments_before`)
- Backpressure / slow-consumer handling (caller policy)
- Size-based auto-truncation (caller monitors `segment_count` / disk usage)

## Testing Strategy

**Unit tests (wal.rs):**
- Segment rollover on byte threshold
- `delete_segments_before` correctness (files removed, remaining intact)
- `oldest_seq` / `segment_count`
- Crash recovery scoped to active segment
- `open` on empty directory → error
- `next_seq` recovery across multiple segments
- Schema preamble in every segment

**Unit tests (replication.rs):**
- Cursor spanning multiple segments
- `CursorBehind` on deleted segments
- Cursor picks up new segments after `open`

**Integration:**
- Full flow: write → rollover → snapshot → delete old segments → cursor pull → convergence
- Update `replicate` example
