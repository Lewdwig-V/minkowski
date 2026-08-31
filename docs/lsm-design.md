# LSM Storage Design (Stage 3)

Date: 2026-08-30. Status: **delivered** (Stages 1–3 complete; recovery cutover shipped in v1.4.0). This document records the `minkowski-lsm` architecture as it exists in the code: incremental persistence over archetype pages, where the persistence cost follows the mutation rate, not the world size.

## 1. Architecture summary

```
World  ──dirty pages──▶  FlushWriter ──▶ L0 sorted run (.run file)
                              │                 │
                    manifest_ops::flush_and_record        │
                              ▼                           ▼
                       LsmManifest ◀── ManifestLog (append-only, CRC32 frames)
                              │                           ▲
                              ▼                    CompactionCommit entry
                       Compactor (compact_one)  ──▶ merged run; inputs deleted

recover_world(lsm_dir, manifest, wal, codecs)
  = LsmRecovery::recover (runs sorted by sequence, latest wins per page)
  + materialize_world (bulk column reconstruction)
  + WAL tail replay (records with seq >= replay floor)
```

The public API surface is deliberately small: `flush_and_record`, `compact_one`, `compact_one_observed`, `LsmManifest`, `ManifestLog`, `recover_world`, `AutoCheckpoint`, `CodecRegistry`, `BlockedBloomFilter`, `COMPACTION_TRIGGER`. `FlushWriter`, `CompactionWriter`, and `BloomView` are internal.

## 2. On-disk format

One sorted run = one immutable `.run` file:

```
[Magic "MKLSM01\0"]                — 8-byte run magic, header CRC32 over the first 40 bytes
[Schema section]                    — stable type names per column, storage kind
[Pages]                             — archetype page images, sorted by (arch_id, slot, page_index)
[Sparse index]                      — binary-searchable (page key → file offset)
[Bloom filter]                      — cache-line-blocked, 64-byte aligned
[Footer]                            — section offsets
```

- Page key packing: `(arch_id, slot, page_index)` as u64. `slot` is the component column slot from the schema section, or `ENTITY_SLOT = 0xFFFF` for the entity-row pseudo-column.
- Pages hold 256 rows (`PAGE_SIZE = 256`). Dirty-page bitsets track which pages changed since the last flush.
- Bloom filter: 8 hashes per 64-byte block, ~1% false-positive rate at 10 bits/key. Built from the sparse-index entries during flush and compaction. Read path: `SortedRunReader::get_page` probes `contains_page` before the binary search — definite misses skip the search; false positives fall through. Zero false negatives: bloom keys are the same index entries the search scans.
- Levels: `L0..L3` in practice (`MAX_LEVELS = 32` is the sanity bound). Flushes land at **L0** (`flush_and_record` records `Level::L0`); `compact_one` merges L0 into L1 when L0 reaches `COMPACTION_TRIGGER = 4` runs.

## 3. Invariants

| Invariant | Statement | Enforced by |
|---|---|---|
| Identity by type | on-disk schema stores stable type names; recovery resolves against the recovered world's own table, never by numeric `ComponentId` | `resolve_local_component`; `stable_name_by_type` lookups; codec resolution by `TypeId` (`World::component_type_id`) |
| Codec gate | every dense component needs a registered codec; missing codec = hard error at flush, never silent drop | `CodecRegistry::register`; `FlushWriter` flush gate |
| Hybrid dense storage | `RawCopy` for raw-copyable components (archived size == native size; memcpy round-trip), `Serialized` for heap-backed components (rkyv per row). Kind recorded per column in the run schema | `storage_kind_for_type`; `raw_copy_size` gate |
| Atomic compaction | `CompactionCommit` records inputs + output in one manifest-log entry: either all inputs are replaced by the output, or none are | `ManifestLog` frame append; `execute_compaction_observed` deletes input files only after commit |
| Orphan cleanup | files not tracked by the manifest are crash garbage; `cleanup_orphans` removes them | `manifest_ops::cleanup_orphans` |
| Latest sequence wins | recovery orders runs by sequence, not by level: `LsmRecovery` sorts all runs by `sequence_range().lo()` and overwrites a page when the incoming run's `seq_hi` is greater or equal | `LsmRecovery::recover` sort + `store_page` overwrite rule |
| Replay floor | WAL tail replays records with `seq >= seq_hi` of the newest run — inclusive, so a removal that straddles the flush boundary is not lost | `recover.rs` replay-floor invariant comment + tests |
| Dirty-page discipline | row-level `BlobVec` writes mark their page dirty; batch entry points that hand out raw or mutable row access (`World::query` with `&mut T`, `query_table_mut`, `query_table_raw`, and the planner's `execute_stream_batched`/`execute_stream_join_chunk` for mutable queries) mark whole columns dirty, because per-row attribution is impossible ahead of iteration; flush writes exactly the dirty pages | `storage::dirty_pages` wired into `BlobVec` write paths; `mark_all_pages_dirty` at the batch entry points; `batch_mutation_entry_points_mark_pages_dirty` |
| State-comparison determinism | equal world states compare equal via `world_fingerprint` — but flush **byte output** is not required to be identical across worlds: page keys use per-world numeric `arch_idx`, so archetype creation order changes file layout | `world_fingerprint` (minkowski-persist) keys by type, never numeric id |

### 3.1 The invariant matrix: recovery × storage kind

Recovery reconstruction has one consumer (`materialize_world`) and three page kinds. The matrix is small but every cell is load-bearing:

| Page kind | Normalization | Reassembly | Identity |
|---|---|---|---|
| `RawCopy` column page | stored bytes are native bytes (zero-padded to `PAGE_SIZE * item_size`) | sliced by native stride, one `append_bytes_unchecked` per page | codec by `TypeId` at flush gate |
| `Serialized` column page | decoded row-by-row (`native_column_page`) into a contiguous native buffer | same stride slicing | codec resolved by `TypeId`, never `ComponentId` |
| Sparse component | rkyv round-trip (`serialize_sparse` / `insert_sparse_raw`) | re-inserted per entity | codec by `TypeId` (`serialize_sparse_by_type`) |
| Allocator metadata | dedicated metadata pages | allocator state restored before WAL replay (spawn records need valid allocator state) | n/a |

The unsafe precondition — page bytes are a valid native image of the component type — is discharged by the codec gate plus per-page CRC validation on read, never assumed.

## 4. The manifest

`LsmManifest` holds per-level run lists with sequence ranges and archetype coverage. `ManifestLog` is an append-only log of manifest changes with CRC32 frame integrity; recovery replays it to rebuild the manifest state.

- `flush_and_record` reads dirty pages, calls `FlushWriter`, records the new run — one call, the only sanctioned flush entry point. It takes `&World` and does **not** clear dirty bits; clearing is the caller's job (`World::clear_all_dirty_pages(&mut self)`), and `AutoCheckpoint` performs it after each successful flush so checkpoints stay incremental.
- `CompactionCommit` is the atomic-entry pattern: the log entry names the inputs and the output; a crash mid-compaction leaves either the old manifest (orphan output file, cleaned by `cleanup_orphans`) or the new one.
- `AutoCheckpoint` (minkowski-persist) wires a `CheckpointHandler` onto `Durable<S>`: on checkpoint, flush dirty pages via `flush_and_record` and record. This replaces full-world snapshots (removed in the Stage 3 cutover).

## 5. Identity rules (why "by type" is absolute)

`ComponentId` is a per-world index. Two worlds that register the same types can assign different ids — after recovery, ids are compacted. Therefore:

- The on-disk schema stores each component's stable type name; recovery resolves it against the recovered world (`resolve_local_component`).
- Codec selection for dense flush, sparse flush, and the fingerprint resolves by `TypeId` (`stable_name_by_type`, `serialize_sparse_by_type`), never by numeric id.
- The rkyv remap (`build_remap`) maps sender-local ids to local ids by stable name, and refuses unmapped ids.
- Any new path that touches component identity must resolve by type. A numeric-id lookup in a cross-world path is a bug even when it works in-process.

## 6. Test inventory

| Area | Coverage |
|---|---|
| Format | 246 lib tests and 38 integration tests: frame CRC, magic validation, legacy-format rejection |
| Bloom | probe false-negative tests (`bloom_filter_matches_index`, `bloom_filter_rejects_absent_pages`), read-path wiring (`get_page` prefilter hit/miss, no-filter pass-through) |
| Compaction | end-to-end `compact_one` with input-file deletion asserted; `CompactionCommit` atomicity; orphan cleanup |
| Recovery | `recover_world` tail replay, replay-floor edge (removal straddling the flush), sparse restoration ordering, allocator metadata restoration |
| Soundness | decode-fingerprint gate for unchecked rkyv (`deserialize_unchecked_by_type` requires the per-run layout fingerprint plus per-page `CrcProof`); recovery import-error drop-safety; raw-copy gate to raw-copyable codecs |
| Fuzz | `fuzz_lsm_recovery` (flush + recover round-trip over statistically varied world states, f32 bit-pattern preservation). **Gap**: malformed run/manifest bytes are not fuzzed — `fuzz_lsm_recovery` builds valid inputs only. `fuzz_wal_replay` mode 0 feeds raw malformed bytes as a discoverable WAL segment; no equivalent exists for `.run` files |

## 7. Known constraints

- Sparse components round-trip through rkyv and require a registered codec (skip-with-warning if absent at flush — recorded, not silently dropped).
- Compaction is on-demand. Background scheduling is a later phase. Space amplification stays within 2× during compaction.
- The bloom filter serves page lookups only (`get_page`); `entity_pages` is an existential query over `page_index` with no single key to probe, so it stays unfenced.
- `world_fingerprint` is dense-only (see the RSM substrate doc, section 5); sparse components are visible to recovery but not to the fingerprint.
