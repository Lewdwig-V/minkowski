# LSM Phase 3: Compactor + Recovery Integration

*Parent document: [Stage 3: LSM Tree Storage](./2026-04-03-stage3-lsm-implementation-plan.md). Follow-up to Phase 2 + B-series + PR #167.*
*Status: Design. Prerequisites: Phase 1 (SortedRun format), Phase 2 (LsmManifest, ManifestLog, flush_and_record), PR #167 (ManifestTag enum) all merged.*

---

## Summary

Phase 3 implements LSM compaction, replaces the snapshot-based checkpoint/recovery path with LSM-based equivalents, and reorganizes crate layering so that `minkowski-lsm` owns the zero-copy storage primitives and `minkowski-persist` depends on it.

Scope:
1. **Compactor** — pause-and-compact `World::compact_one()` primitive that merges all of one archetype's runs at L(N) into a new run at L(N+1). Count-based trigger (≥4 runs per archetype per level).
2. **Atomic commit** — new `CompactionCommit` manifest entry fusing the output-add with input-removes into a single crash-safe frame.
3. **Recovery** — `SortedRunReader::validate_page_crc` returns `CrcProof`, enabling the existing `raw_copy_size` fast path in `CodecRegistry::decode`. Zero-copy recovery preserved.
4. **Snapshot cleanup (clean cut)** — remove `Snapshot`, `SnapshotError`, `AutoCheckpoint`, `save_to_bytes`/`load_from_bytes`. Replace with `LsmCheckpoint` impl of `CheckpointHandler`.
5. **Crate layering shift** — move `CrcProof` and `CodecRegistry` from `minkowski-persist::codec` into `minkowski-lsm::codec`; `minkowski-persist` gains a dep on `minkowski-lsm`.
6. **Const-generic level count** — `Manifest<const N: usize = 4>` with `DefaultManifest = Manifest<4>` alias. 4 remains the default; the constant now has a workload model behind it.

Non-goals (deferred to later phases):
- Continuous background compaction (D-mode scheduling). This is a dedicated thread with throttling, back-pressure, and frame-budget awareness. Phase 3 ships the C-mode primitive that D will call in a loop.
- Schema evolution across levels. Sorted runs at different levels must share the same component schema today.
- LSM-based replica bootstrap. The `replicate.rs` example's `save_to_bytes` path disappears in the clean cut; replication bootstrap is revisited in a later phase.
- Concurrency between compaction and flush. C-mode is synchronous; not an issue.

## Motivation

Two shortcomings of the current snapshot-based persistence model:

1. **Full-snapshot cost scales with World size, not delta size.** Every checkpoint rewrites the entire World. For a 10GB world, each checkpoint writes 10GB. The LSM approach writes only dirty pages since the last flush.
2. **Cold recovery replays the full WAL tail from the last snapshot.** A 10-minute snapshot interval + 9 minutes of WAL can mean seconds of replay on boot. LSM recovery reads the most recent compacted state directly and replays only the WAL tail since the last flush.

Phase 3 is the point where those benefits actually land in the recovery path. Phases 1 and 2 built the storage primitives (sorted runs, manifest log). Phase 3 wires them into `Durable<S>` and removes the old snapshot path.

A secondary motivation surfaced during this brainstorm: the current crate layering (`persist` owns `CrcProof` + `CodecRegistry`) was an accident of snapshot having been the first user. Post-cleanup, LSM is the primary producer of page-level proofs, and WAL's single callsite of `CrcProof` is well-served by depending on `minkowski-lsm`. The layering shift reflects the inversion: `lsm` is the load-bearing storage layer; `persist` is operations over it.

## Design

### 1. Level count model and const generic

The current `NUM_LEVELS = 4` constant in `manifest.rs:7` was an arbitrary choice. Phase 3 replaces it with a const generic parameter on `LsmManifest`, keeping 4 as the default.

**Workload regime: data fits in ~100× RAM, biased low (10–30× typical).** The dominant use case is real-time interactive — games, simulations, collaborative tools. The working set is RAM-bounded because the frame budget can only touch what's in RAM. On-disk data is historical — ledger of what's happened, not a larger active dataset.

**Math**: at size ratio T=10 between levels, with L1 being 1–10% of RAM (a typical flush unit):
- 4 levels covers ~0.1× to 100× RAM on disk → matches the regime
- 5 levels extends to ~1000× RAM → overkill for the regime, but harmless to support for archival-style users

**API**:
```rust
pub struct LsmManifest<const N: usize = 4> {
    levels: [Vec<SortedRunMeta>; N],
}

pub type DefaultManifest = LsmManifest<4>;

impl<const N: usize> LsmManifest<N> { /* ... */ }
```

The const generic propagates through `FlushWriter<N>`, `Compactor<N>`, and `LsmCheckpoint<N>`. **Merge logic is level-count-agnostic** — merging archetype X's runs at L(i) into L(i+1) is the same operation regardless of total N. Only bounds checks and manifest serialization care about N.

**Documentation**: module-level comment on `manifest.rs` explains the regime model:
```rust
//! # Level Count
//!
//! Defaults to 4 levels. This fits the expected regime: on-disk data
//! up to ~100× RAM, with reads served from the in-memory World rather
//! than from level traversal (the in-memory World IS the merged view).
//!
//! At T=10 size ratio: 4 levels covers ~0.1× to 100× RAM on disk.
//!
//! For ledger-style workloads (TigerBeetle territory, ever-growing
//! history), construct `LsmManifest<7>` instead. Merge logic is
//! level-count-agnostic; only bounds checks and manifest serialization
//! care about N.
```

### 2. Compactor design

**Granularity: A (full-archetype-level merge).** A compaction job merges all of archetype X's runs at L(N) — plus any overlapping runs at L(N+1) — into a single new run at L(N+1). For the append-only ledger shape, overlaps at L(N+1) are rare (new runs have timestamps beyond all older runs), so the "plus overlapping" branch is usually empty.

**Trigger: count-based, K=4 runs per archetype per level.** When any archetype has ≥4 runs at L(N), that archetype+level becomes a compaction candidate. Simpler than size-based; adequate when flush sizes are roughly constant (which they are — we control the flush unit). Upgrade path to size-based is straightforward for D-phase if skew becomes a problem.

**Picking: stable iteration, first-over-threshold wins.** No priority queue. `compact_one` iterates archetypes in stable order, picks the first one over threshold at any level, performs one merge, returns.

**API**:
```rust
impl World {
    /// True if any archetype has ≥ COMPACTION_TRIGGER runs at any level.
    pub fn needs_compaction(&self) -> bool;

    /// Perform one compaction job. Returns Ok(None) if nothing is over threshold.
    pub fn compact_one(&mut self) -> Result<Option<CompactionReport>, LsmError>;
}

pub struct CompactionReport {
    pub archetype: ArchetypeId,
    pub from_level: Level,
    pub to_level: Level,
    pub input_run_count: usize,
    pub output_bytes: u64,
}

const COMPACTION_TRIGGER: usize = 4;
```

The caller drives the loop:
```rust
while world.needs_compaction() {
    world.compact_one()?;
}
```

C-mode. D-mode wraps this in a background thread with throttling.

### 3. Atomic compaction commit

Compacting archetype X at L(N) → L(N+1) produces one output run and removes multiple input runs. Serializing these as separate `AddRun` + `RemoveRun` manifest entries creates a window where a crash leaves the manifest with both the output AND some/all inputs still listed. Recovery reads would see double-counted data — a correctness bug.

**Solution**: new atomic manifest entry, following PR B2's `AddRunAndSequence` pattern.

```rust
#[repr(u8)]
pub enum ManifestTag {
    AddRun = 0x01,
    RemoveRun = 0x02,
    PromoteRun = 0x03,
    SetSequence = 0x04,
    AddRunAndSequence = 0x05,
    CompactionCommit = 0x06,  // NEW in Phase 3
}

pub enum ManifestEntry {
    // ... existing variants ...
    CompactionCommit {
        output_level: Level,
        output: SortedRunMeta,
        inputs: Vec<(Level, PathBuf)>,  // runs to remove on apply
    },
}
```

**Wire format for `CompactionCommit`**:
```
[tag: u8 = 0x06]
[output_level: u8]
[output: SortedRunMeta encoded as in AddRun]
[input_count: u32 LE]
input_count × {
    [level: u8]
    [path_len: u16 LE][path: bytes]
}
```

`input_count` is `u32` rather than `u16`. The expected value is <100 even under pathological conditions (bounded by per-level count trigger), but the asymmetry of change cost favors the wider field: u16→u32 later requires a manifest-log format-version bump and dual-decode, while u32→u16 later is trivial. A logical bound check sits at the apply site:

```rust
const MAX_COMPACTION_INPUTS: usize = 1024;  // orders of magnitude past expected

debug_assert!(
    inputs.len() <= MAX_COMPACTION_INPUTS,
    "CompactionCommit with {} inputs — check compaction granularity",
    inputs.len()
);
```
That preserves the "bounded everything" property at the semantic layer without baking it into the wire format.

**Apply semantics**: on replay, `apply_entry` atomically adds the output to the manifest and removes all listed inputs. Partial application is impossible — the frame is a single CRC-validated unit; it either applies entirely or gets truncated as tail garbage.

**Input file deletion** happens after the manifest entry is durable, via the existing `cleanup_orphans` mechanism (paths no longer referenced in the manifest get swept on the next clean operation).

### 4. Tombstone elision

**Policy: drop tombstones only when compacting INTO the bottom level (L(N-1) where N is the const-generic level count).** For the default 4-level config, tombstones survive until they reach L3. Rationale: at any level above the bottom, older data may exist at a deeper level that the tombstone is shadowing; dropping the tombstone early would resurrect shadowed data.

This is the standard LSM policy. For the append-only ledger shape, tombstones are rare (history is mostly immutable), so the "tombstone lingers until bottom level" cost is minimal in practice.

**Implementation**: `Level` gains a const-generic-aware accessor:
```rust
impl Level {
    /// The bottom level for an N-level manifest: Level(N - 1).
    pub const fn bottom<const N: usize>() -> Level {
        Level(N as u8 - 1)
    }
}
```
During merge: `if to_level == Level::bottom::<N>() { skip_tombstones() } else { include_tombstones() }`. One branch on the output level at compaction start.

### 5. Filter construction hook (Phase 4 carry-forward)

Phase 3 does not implement the bloom filter, but must leave a clean hook for Phase 4 to slot into. The hook goes on the **output writer side, after merge resolution**. Keys that get dropped during merge (tombstoned, overwritten, deduped across input runs) never reach the hook. This keeps FPR honest — the filter's claimed membership matches exactly what ends up on disk.

**Writer API**:
```rust
impl<const N: usize> FlushWriter<N> {
    /// Install a per-entry observer. Called once per key that passes merge
    /// resolution and is written to the output run.
    pub fn set_entry_observer(&mut self, observer: Box<dyn FnMut(&EntryKey)>);
}
```

For Phase 3, no observer is installed; the hook is a no-op. Phase 4 installs a `BloomFilterBuilder` observer and extracts the finalized filter after the writer closes.

**Size estimation for the filter (Phase 4 detail, noted here)**: blocked bloom needs block count at construction time. Sum input run key counts as an upper bound; accept slight over-provisioning after tombstones/overwrites reduce the actual output. Alternative (buffer entries, finalize on close) is deferred to Phase 4 if the over-provisioning overhead proves material.

### 6. Zero-copy recovery preservation

The existing snapshot path gets zero-copy recovery via `CrcProof` + `CodecRegistry::decode(..., Some(&proof))` which routes to the `raw_copy_size` fast path (direct memcpy, skipping rkyv bytecheck). Phase 3 preserves this for LSM-based recovery with a small API tweak on `SortedRunReader`:

```rust
// crates/minkowski-lsm/src/reader.rs (post-cleanup namespace)
impl SortedRunReader {
    /// Validate the page CRC and return a proof token on success.
    ///
    /// The returned CrcProof is the input to CodecRegistry::decode's
    /// raw_copy_size fast path, which skips rkyv bytecheck for decoded
    /// pages. Identical mechanism to the old snapshot v2 format.
    pub fn validate_page_crc(&self, page: &PageRef<'_>) -> Result<CrcProof, LsmError>;
}
```

This is a return-type change on an existing method. Current callers either discard the success value (`let _ = validate_page_crc(...)?;` or `validate_page_crc(...)?;` — both forms keep working since `CrcProof` is `Drop`-only-no-op) or migrate to the new signature to get the proof.

**Recovery path** (new, replacing `Snapshot::load`):

```rust
pub fn recover_world_from_lsm<const N: usize>(
    run_dir: &Path,
    codecs: &CodecRegistry,
) -> Result<(World, SeqNo), LsmError> {
    let (manifest, _log) = ManifestLog::recover(&run_dir.join("manifest.log"))?;
    let mut world = World::new();
    for level in 0..N {
        for meta in manifest.runs_at_level(Level::new(level as u8).unwrap()) {
            let reader = SortedRunReader::open(meta.path(), codecs)?;
            for page in reader.pages() {
                let proof = reader.validate_page_crc(&page)?;
                let bytes = page.data();
                // raw_copy_size fast path (skips rkyv bytecheck)
                codecs.decode(page.header().slot, bytes, Some(&proof))?
                    .apply_to(&mut world);
            }
        }
    }
    let last_flush_seq = manifest.max_sequence();
    Ok((world, last_flush_seq))
}
```

Then WAL replay from `last_flush_seq` applies any mutations since the last flush, matching the existing `Durable::recover` protocol.

### 7. Snapshot cleanup (clean cut)

Minkowski has no external users. Clean-cut removal is safe.

**Removed**:
- `Snapshot` struct (save, load, save_to_bytes, load_from_bytes)
- `SnapshotError` (folded; any residual semantics merge into `LsmError`)
- `AutoCheckpoint` (replaced by `LsmCheckpoint`)

**Kept (moved)**:
- `CrcProof` — moves to `minkowski-lsm::codec`
- `CodecRegistry`, `CodecError` — move to `minkowski-lsm::codec`

**Caller migration**:
| File | Current | Post |
|---|---|---|
| `crates/minkowski-persist/src/checkpoint.rs:68-69` | Calls `Snapshot::save` | Replaced with `LsmCheckpoint` |
| `crates/minkowski-persist/src/index.rs:535,568` | Tests use `Snapshot::new().save/load` | Rewrite against LSM or delete |
| `crates/minkowski-persist/src/replication.rs:619` | Test uses `Snapshot::new()` | Rewrite against LSM or delete |
| `crates/minkowski-persist/benches/persist.rs` | Benchmarks | Rewrite against LSM |
| `crates/minkowski-bench/benches/serialize.rs` | Benchmarks | Rewrite against LSM |
| `examples/examples/replicate.rs:69,137` | Uses `save_to_bytes`/`load_from_bytes` for replica bootstrap | Remove or stub; full LSM-based bootstrap in a later phase |

### 8. Crate layering shift

**Post-cleanup stack**:
```
minkowski              (ECS core)
     ↑
minkowski-lsm          (storage primitives: sorted runs, manifest, codec, CrcProof)
     ↑
minkowski-persist      (WAL, replication, orchestration over LSM)
```

**Moves**:
- `crates/minkowski-persist/src/codec.rs` → `crates/minkowski-lsm/src/codec.rs`
  - Exports: `CodecRegistry`, `CodecError`, `CrcProof`
- `crates/minkowski-persist/Cargo.toml` gains `minkowski-lsm = { path = "../minkowski-lsm" }`
- `crates/minkowski-lsm/Cargo.toml` does NOT gain a dep on persist (cycle prevention)
- WAL's import of `CrcProof` updates from `crate::codec::CrcProof` to `minkowski_lsm::codec::CrcProof`

### 9. `LsmCheckpoint` API

```rust
// In minkowski-persist/src/checkpoint.rs (replaces AutoCheckpoint)

pub struct LsmCheckpoint<const N: usize = 4> {
    manifest: LsmManifest<N>,
    log: ManifestLog,
    run_dir: PathBuf,
}

impl<const N: usize> LsmCheckpoint<N> {
    pub fn new(run_dir: PathBuf) -> Result<Self, LsmError> {
        let log_path = run_dir.join("manifest.log");
        let (manifest, log) = ManifestLog::recover(&log_path)?;
        Ok(Self { manifest, log, run_dir })
    }
}

impl<const N: usize> CheckpointHandler for LsmCheckpoint<N> {
    fn checkpoint(&mut self, world: &World, seq: SeqNo) -> Result<(), CheckpointError> {
        // Lower bound: sequence immediately after the last durable flush.
        // Upper bound: current WAL sequence (passed in).
        let lo = self.manifest.max_sequence()
            .map(SeqNo::next)
            .unwrap_or(SeqNo::from(0u64));
        let seq_range = SeqRange::new(lo, seq)
            .ok_or(CheckpointError::InvalidSeqRange)?;
        flush_and_record(world, seq_range, &mut self.manifest, &mut self.log, &self.run_dir)
            .map_err(CheckpointError::from)
    }
}
```

`Durable<S>` is unchanged; it depends on the `CheckpointHandler` trait abstractly.

### 10. Error handling

**Reuses existing atomic primitives** — no new error-handling machinery needed.

| Failure point | Existing mechanism | Behavior |
|---|---|---|
| Output run write fails partway | `tmp + rename + fsync dir` atomicity (Phase 1) | Tmp file unlinked on `Drop`; no manifest changes; state unchanged |
| Manifest `CompactionCommit` write fails (crash mid-frame) | PR B1 truncate-on-error replay | Frame becomes tail garbage; output file becomes orphan |
| Orphan output file (no manifest ref) | `cleanup_orphans` sweep | Removed on next clean boot or explicit cleanup call |
| CRC mismatch on recovery page read | `LsmError::Crc { offset, expected, actual }` (Phase 1) | Propagates up; recovery fails hard (not tail-truncate) |

**No silent failures.** Every failure path surfaces as `LsmError::*` with enough context to diagnose.

## Testing strategy

### Unit tests

- **`compact_one` with no candidates**: `Ok(None)`.
- **`compact_one` single archetype over threshold**: K=4 runs at L0 → 1 run at L1, 0 runs at L0.
- **Tombstone elision at bottom level**: compact with a tombstone, confirm it's dropped at L(N-1) compaction but preserved at higher levels.
- **CompactionCommit atomic apply**: write a CompactionCommit frame, apply, verify manifest matches.
- **CompactionCommit torn frame**: truncate the frame mid-payload, replay, verify truncation to prior frame boundary.
- **`validate_page_crc` returns CrcProof**: valid page → proof; corrupted page → `LsmError::Crc`.
- **`LsmCheckpoint::checkpoint` flushes to new sorted run**: call, verify new run appears in manifest at L0.

### Integration tests

- **`recover_world_from_lsm` round-trip**: populate World, flush, drop, recover, verify all entities/components restored.
- **Recovery replays WAL tail after last_flush_seq**: flush at seq=100, append WAL at seq=100..200, recover, verify state includes post-flush mutations.
- **Full Durable<S> integration**: replace AutoCheckpoint with LsmCheckpoint in the existing Durable tests; all pass.
- **`compact_one` idempotence under crash**: mid-compaction crash (simulated by truncating the manifest at various byte offsets), replay, verify correct state or clean "pre-compaction" state, never double-counted.
- **Cross-level tombstone survival**: insert, delete, flush L0, compact L0→L1, compact L1→L2, verify tombstone still present (not yet at bottom).
- **Tombstone dropped at bottom level**: continue above test through L(N-1) compaction, verify tombstone now gone.

### Byte-prefix convergence test (extends existing)

Phase 2's `replay_converges_at_every_truncation_prefix` currently covers frame-level truncation. Extend to include a `CompactionCommit` frame with K=4 inputs, verify every prefix 0..=frame_len either produces (a) an error pre-frame with consistent pre-compaction state, or (b) successful apply with post-compaction state. No intermediate "partial compaction" state should be reachable.

## Rollout

Phase 3 lands as **three sequential squash-merged PRs**, each with its own coherent theme. The split avoids a single blast-radius PR that mixes pure refactor, behavioral replacement, and new feature work.

**Dependency graph:**
```
  PR 1 ──┬─→ PR 2 (LsmCheckpoint + snapshot cut)
         └─→ PR 3 (compactor)
```

### PR 1: Infrastructure (no semantics change)

Pure refactor + type-system extension. Zero behavioral changes to existing callers. Estimated ~400–600 lines.

- Move `CrcProof` + `CodecRegistry` + `CodecError` from `minkowski-persist::codec` to `minkowski-lsm::codec`
- Add `minkowski-lsm` dep to `minkowski-persist`; update all imports
- Convert `LsmManifest` → `LsmManifest<const N: usize = 4>` with `DefaultManifest` alias
- Propagate const-generic through `FlushWriter<N>` and manifest_ops call sites
- Change `SortedRunReader::validate_page_crc` return type to `Result<CrcProof, LsmError>`
- Module-level regime doc on `manifest.rs`

All existing tests pass unchanged. Easy review: if the build and tests pass, it's correct.

**Deferred to PR 2**: `recover_world_from_lsm` helper. The page-to-archetype plumbing (reconstruct archetype schema from slots, spawn entities from entity-ID pages, fill columns from component pages) is non-trivial and has no value without a caller. PR 2 provides both the helper implementation and its first consumer (`LsmCheckpoint`'s recovery path).

### PR 2: LsmCheckpoint + snapshot clean cut

Behavioral replacement. Estimated ~800–1200 lines, heavy deletions + the recovery helper. Depends on PR 1.

- Implement `recover_world_from_lsm<const N: usize>(run_dir, codecs) -> (World, SeqNo)` helper — reconstructs World by iterating sorted-run pages, using `CrcProof` fast path
- Implement `LsmCheckpoint<N>` as new `CheckpointHandler` impl
- Migrate `Durable<S>` default checkpoint handler from `AutoCheckpoint` to `LsmCheckpoint`
- Delete `Snapshot`, `SnapshotError`, `AutoCheckpoint`, `save_to_bytes`, `load_from_bytes`
- Stub `examples/replicate.rs` with a TODO pointing to this design doc
- Migrate benchmarks (`persist.rs`, `serialize.rs`) to use `LsmCheckpoint`

Public `CheckpointHandler` trait is unchanged; the impl swap is invisible to any external user of `Durable<S>`. (No external users exist; this is belt-and-braces.)

### PR 3: Compactor

Pure feature addition. Estimated ~800–1200 lines. Depends on PR 1, independent of PR 2.

- `ManifestTag::CompactionCommit = 0x06` + `ManifestEntry::CompactionCommit` variant
- `CompactionCommit` encode/decode with `input_count: u32 LE`
- `World::compact_one()` + `World::needs_compaction()` API
- K=4 count-based trigger, stable-iteration picking
- Tombstone elision at bottom level only
- `FlushWriter::set_entry_observer` hook (no-op observer for Phase 3; Phase 4 installs a real one)
- `MAX_COMPACTION_INPUTS = 1024` debug_assert at apply site
- **Manifest-log header `max_level` byte** (repurpose one of PR #164's 3 reserved bytes at offset 5–7): write `N as u8` on create; on `recover::<N>()`, reject mismatches as fatal `LsmError::Format` rather than letting per-frame OOR checks silently truncate valid frames. Closes the cross-`N` portability hazard flagged on PR #168 (Codex review + internal type-design review). Since PR 3 already bumps the on-disk format with `CompactionCommit`, the header byte rides along naturally.

PR 2 and PR 3 can land in either order (or in parallel worktrees). PR 1 is the strict prerequisite for both.

**CI gates each PR**: fmt, clippy, test, tsan, loom, claude-review.

### Post-phase memory updates

After PR 3 lands:
- `project_scaling_roadmap.md`: mark Phase 3 complete, update type-rating table (add `Compactor`, `LsmCheckpoint`), update next-step to Phase 4 (bloom filter).
- `project_bloom_compaction_coupling.md`: confirm Phase 3's filter hook API landed as designed; Phase 4 can now start.

## Risks

- **Zero-copy regression risk**: the `validate_page_crc` return-type change has to be picked up at every recovery-path callsite. Missing one means `CodecRegistry::decode` gets `None` for the proof, falls back to full rkyv bytecheck — silently correct but slow. Mitigate with a benchmark in `minkowski-bench` that asserts recovery throughput within an expected range of the old snapshot recovery path.

- **Crate layering move blast radius**: `CrcProof` + `CodecRegistry` move across crates changes every import. Mechanical but wide. Run it as a separate commit within the PR so review can see the import-rewriting and the feature changes independently.

- **`replicate.rs` example in limbo**: clean cut breaks the example. If the example is the only documentation of how to set up replication, removing it without replacement leaves a doc gap. Decision: stub with a `TODO: LSM-based bootstrap pending` comment and a pointer to this design doc, rather than delete outright.

- **Tombstone elision correctness**: the "drop only at bottom level" rule is right for a system where older data exists at deeper levels. If a future design adds shortcuts (e.g., direct L0 → L3 promotion), the rule needs to be re-examined — the tombstone must survive long enough to shadow all older data that could still exist below it.
