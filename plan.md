# Slotted Pages, Availability Lists & CRC32 WAL Checksums

## Overview

Two related changes to `minkowski-persist`:

1. **CRC32 frame checksums** — Add a per-frame CRC32C checksum to the WAL wire format so bit-flip corruption in complete frames is detected (currently only rkyv deserialization catches structural corruption, but silent data corruption within valid rkyv envelopes goes undetected).

2. **Slotted page layer** — Introduce a fixed-size page abstraction over WAL segments. Each page has a header (magic, slot count, free-space pointer, CRC32) and a slot directory pointing to variable-length records packed within the page. This gives page-aligned I/O, per-page integrity checking, and an availability list for future compaction/reuse.

## Part 1: CRC32 Frame Checksums

### New frame format

Current: `[len: u32 LE][payload: len bytes]`
New: `[len: u32 LE][crc32: u32 LE][payload: len bytes]`

- CRC32C (Castagnoli) covers the payload bytes only (not the length prefix — the length is validated by EOF detection).
- Frame header grows from 4 → 8 bytes.
- On read, compute CRC32C of the payload and compare with stored value. Mismatch → `WalError::ChecksumMismatch` (treated like `Format` — triggers truncation on active segment, hard error on sealed segments).

### New dependency

Add `crc32fast` to `minkowski-persist/Cargo.toml`. This crate is zero-dep, uses hardware CRC32C on x86/ARM, and is widely used (rkyv's own ecosystem uses it).

### Files changed

- **`crates/minkowski-persist/Cargo.toml`** — add `crc32fast = "1"`
- **`crates/minkowski-persist/src/wal.rs`**:
  - `read_next_frame()` — read 8-byte header (len + crc32), validate checksum after reading payload
  - `append()` — compute CRC32 of payload, write 8-byte header
  - `acknowledge_snapshot()` — same write-path change
  - `write_schema_preamble()` — same write-path change
  - `roll_segment()` — same write-path change (schema preamble to new file)
  - `scan_active_segment()` — frame size accounting changes from `4 + len` → `8 + len`
  - All byte-accounting (`active_bytes`, `bytes_since_checkpoint`, frame_bytes) updated for 8-byte header
  - `WalError` — add `ChecksumMismatch` variant
  - Constants: `FRAME_HEADER_SIZE = 8`
- **Tests**: Update corruption tests to account for new format. Add dedicated `checksum_mismatch_detected` test that writes a valid-length frame with wrong CRC32 and verifies it's caught.

### Write helper

Extract a `write_frame(writer, payload) -> Result<u64, WalError>` helper that:
1. Computes `crc32fast::hash(&payload)`
2. Writes `[len: u32 LE][crc32: u32 LE][payload]`
3. Returns bytes written

This eliminates the duplicated write logic across `append`, `acknowledge_snapshot`, `write_schema_preamble`, and `roll_segment`.

## Part 2: Slotted Pages

### Design

A **slotted page** is a fixed-size block (default 4096 bytes, configurable) that packs multiple WAL frames into page-aligned units:

```
┌─────────────────────────────────────────────┐
│ PageHeader (16 bytes)                       │
│   magic: u32    = 0x4D4B5750 ("MKWP")       │
│   page_seq: u32 = monotonic page number      │
│   slot_count: u16                            │
│   free_offset: u16 = end of slot directory    │
│   data_offset: u16 = start of record data     │
│   checksum: u16 = reserved (0 for now)        │
├─────────────────────────────────────────────┤
│ Slot Directory (slot_count × 4 bytes)       │
│   slot[0]: offset: u16, length: u16          │
│   slot[1]: ...                               │
├─────────────────────────────────────────────┤
│ Free Space                                   │
├─────────────────────────────────────────────┤
│ Record Data (packed from end, growing down)  │
│   record[N-1]: [frame bytes]                 │
│   ...                                        │
│   record[0]: [frame bytes]                   │
└─────────────────────────────────────────────┘
```

Each record within a page is a complete WAL frame (with its own CRC32 from Part 1). The page header CRC covers the entire page (header + slots + data) minus the checksum field itself — a second level of integrity on top of per-frame CRC32.

### Availability list

An in-memory `Vec<u32>` of page indices that have free space (i.e., `data_offset - free_offset > MIN_RECORD_SIZE`). For append-only WAL, this is populated only for the current active page. For future compaction (post-snapshot segment rewrite), sealed pages with invalidated records could be added to the availability list for reuse.

The availability list is **not persisted** — it's rebuilt on open by scanning page headers. This is consistent with the existing `scan_active_segment()` approach.

### Module structure

New file: `crates/minkowski-persist/src/slotted_page.rs`

Public types:
- `SlottedPage` — in-memory representation of a single page
- `PageHeader` — the 16-byte header struct
- `SlotEntry` — (offset, length) pair
- `PageConfig` — page size configuration

Methods:
- `SlottedPage::new(page_seq, page_size)` — create empty page
- `SlottedPage::try_insert(payload) -> Option<u16>` — insert record if space available, returns slot index
- `SlottedPage::get(slot_index) -> Option<&[u8]>` — read record by slot
- `SlottedPage::free_space() -> usize` — available bytes
- `SlottedPage::write_to(writer)` — serialize full page (padded to page_size)
- `SlottedPage::read_from(reader) -> Result<Self>` — deserialize from reader
- `SlottedPage::validate_checksum() -> bool` — verify page-level CRC32
- `SlottedPage::iter_slots() -> impl Iterator<Item = &[u8]>` — iterate all records

### Integration with WAL

The slotted page layer is introduced as an **internal abstraction** within the WAL — not exposed in the public API. The WAL's `append()` method buffers frames into the current active page. When the page is full (not enough space for the next frame), it's flushed to disk and a new page is started.

This is a **non-breaking change** — the public `Wal` API (`append`, `replay`, `replay_from`, etc.) is unchanged. The on-disk format changes but this is a new feature, not a migration.

### Overflow records

Records larger than `page_size - HEADER_SIZE - SLOT_ENTRY_SIZE` are written as **overflow pages**: a sequence of continuation pages with a special `OVERFLOW` magic (`0x4D4B574F`). The first page's slot directory has a single entry pointing to the start of the overflow data, and continuation pages are pure data (no slot directory). This handles the existing 256 MB max frame size.

### Files changed

- **`crates/minkowski-persist/src/slotted_page.rs`** — new module with `SlottedPage`, `PageHeader`, etc.
- **`crates/minkowski-persist/src/lib.rs`** — add `mod slotted_page; pub use slotted_page::SlottedPage;`
- **`crates/minkowski-persist/src/wal.rs`** — integrate page-based I/O (internal, behind existing API)

## Implementation Order

1. **CRC32 frame checksums** first — standalone, well-scoped, immediately useful
2. **`write_frame` / `read_frame` helpers** — refactor duplicated write logic
3. **`SlottedPage` module** — self-contained data structure with unit tests
4. **WAL integration** — wire slotted pages into WAL append/read paths
5. **Tests** — corruption tests, overflow records, availability list rebuild

## Testing

- Existing WAL tests updated for new frame format
- New `checksum_mismatch_detected` test
- New `slotted_page_insert_and_read` unit test
- New `slotted_page_overflow` test for large records
- New `slotted_page_availability_list` test
- Fuzz target `fuzz_wal_replay` continues to work (frames are self-describing)
