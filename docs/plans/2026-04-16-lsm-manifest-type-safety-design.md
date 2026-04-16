# LSM Manifest Type Safety (PR A)

*Parent document: [Stage 3: LSM Tree Storage](./2026-04-03-stage3-lsm-implementation-plan.md), follow-up to Phase 2.*
*Status: Design. Prerequisites: PR #160 (LSM Phase 2) merged.*

---

## Summary

This PR tightens the type design of the `minkowski-lsm` manifest subsystem by introducing newtype primitives (`SeqNo`, `SeqRange`, `Level`), dropping the redundant `SortedRunMeta::level` field, and adding a validated `SortedRunMeta::new` constructor. The work was identified during review of PR #160 and is the first half of a two-PR sequence; PR B (unified `recover` entry point, on-disk magic+version header, parent-dir fsync) is explicitly out of scope.

The refactor is **internal-only in on-disk format terms** — no wire format change, no replay compat story, no migration of existing log files. Type changes are source-level only.

## Motivation

PR #160's review (via `type-design-analyzer`) rated the manifest types 3–5/10 on invariant enforcement. Specific holes:

- `level: u8` fields scattered across `LsmManifest::{add,remove,promote,runs_at}_level` with runtime `assert!` at three call sites. One forgotten assertion produces a panic in the wrong place.
- `sequence_range: (u64, u64)` expresses no invariant. `(5, 3)` is representable. The convention (half-open, `[lo, hi)`) lives in a code comment.
- `SortedRunMeta` has `pub(crate)` fields and three construction sites (`flush_and_record`, `decode_entry`, tests) with no validation. A decoded frame with an unsorted `archetype_coverage` vector silently produces a manifest that breaks downstream binary-search-based lookups.
- `SortedRunMeta::level` duplicates the Vec index it sits in. PR #160's fix kept the field in sync on promotion, but the structural problem — two sources of truth for one fact — remains.

The principle driving this work: move invariants from convention-in-comments to enforced-at-construction. Keep the validation cheap so there's no performance pressure to bypass it.

## Non-goals

Explicitly deferred to **PR B (LSM Manifest Format Hardening)**:

- `ManifestLog::recover(path) -> (LsmManifest, ManifestLog)` unified entry point.
- Magic + version header at the start of the log file.
- Parent-directory fsync after `truncate_at` in replay.

These are deferred because they touch on-disk format and forward/backward compatibility, which are independent design axes from type invariants. Bundling them would double the review surface and couple type-shape decisions to format-version decisions.

## Design

### 1. New types — `crates/minkowski-lsm/src/types.rs` (new module)

```rust
/// A WAL sequence number. Half-open ranges (`[lo, hi)`) are encoded
/// via `SeqRange`; raw `u64` arithmetic on seq numbers is disallowed
/// by the type (no Add/Sub impls) because seqs are identities, not sizes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SeqNo(pub u64);

impl From<u64> for SeqNo { /* ... */ }
impl From<SeqNo> for u64 { /* ... */ }
impl fmt::Display for SeqNo { /* ... */ }

/// A half-open sequence range `[lo, hi)`. Construction enforces
/// `lo <= hi`. `hi == lo` represents an empty range (not currently
/// produced by any code path but syntactically allowed).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SeqRange { pub(crate) lo: SeqNo, pub(crate) hi: SeqNo }

impl SeqRange {
    pub fn new(lo: SeqNo, hi: SeqNo) -> Result<Self, LsmError> {
        if lo > hi {
            return Err(LsmError::Format(
                format!("SeqRange: lo ({lo}) > hi ({hi})")
            ));
        }
        Ok(Self { lo, hi })
    }
}

/// An LSM level index, guaranteed `< NUM_LEVELS` at construction.
/// Bounds check lives once, in `Level::new`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Level(u8);

impl Level {
    pub const L0: Level = Level(0);
    pub const L1: Level = Level(1);
    pub const L2: Level = Level(2);
    pub const L3: Level = Level(3);

    pub fn new(level: u8) -> Option<Self> {
        if (level as usize) < NUM_LEVELS { Some(Self(level)) } else { None }
    }

    pub fn as_u8(self) -> u8 { self.0 }
    pub fn as_index(self) -> usize { self.0 as usize }
}
```

Design notes:

- `SeqNo` is transparent (pub field) because the type's only job is nominal distinction. `SeqRange` fields are `pub(crate)` — the invariant (`lo <= hi`) must flow through `SeqRange::new`; downstream code still reads via field syntax within the crate. `Level` keeps its `u8` private because the bounds invariant is the whole point.
- No arithmetic impls on `SeqNo` — seq numbers aren't scalar quantities. Callers that need "next seq" explicitly go through `SeqNo(x.0 + 1)` or via `SeqRange.hi`.
- `Level` has associated consts `L0..L3` for code that knows the level at compile time, avoiding an `.unwrap()` on `Level::new(0)`.
- `NUM_LEVELS` stays a constant. Bumping to 5 or 6 in the future is a one-line change; no `match Level` to update.

### 2. `SortedRunMeta` refactor

```rust
pub struct SortedRunMeta {
    // All fields private. The public surface is new() + accessors.
    path: PathBuf,
    sequence_range: SeqRange,
    archetype_coverage: Vec<u16>,
    page_count: u64,
    size_bytes: u64,
    // `level: u8` DROPPED. Was never stored on disk — decode_entry
    // previously derived it from the outer AddRun.level byte and
    // populated meta.level as a redundant copy. Array position in
    // LsmManifest::levels[i] is the single source of truth.
}

impl SortedRunMeta {
    pub fn new(
        path: PathBuf,
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,
        page_count: u64,
        size_bytes: u64,
    ) -> Result<Self, LsmError> {
        // Invariants enforced at construction:
        if archetype_coverage.windows(2).any(|w| w[0] >= w[1]) {
            return Err(LsmError::Format(
                "archetype_coverage is not strictly sorted".to_owned()
            ));
        }
        if page_count == 0 {
            return Err(LsmError::Format(
                "page_count must be non-zero".to_owned()
            ));
        }
        // size_bytes > 0 is NOT validated — redundant with page_count
        // given that every sorted run has a non-empty header.
        // SeqRange lo<=hi is already enforced by SeqRange::new.
        Ok(Self { path, sequence_range, archetype_coverage, page_count, size_bytes })
    }

    pub fn path(&self) -> &Path { &self.path }
    pub fn sequence_range(&self) -> SeqRange { self.sequence_range }
    pub fn archetype_coverage(&self) -> &[u16] { &self.archetype_coverage }
    pub fn page_count(&self) -> u64 { self.page_count }
    pub fn size_bytes(&self) -> u64 { self.size_bytes }

    // pub fn level(&self) -> u8 — REMOVED.
    // Callers always know the level from the context they obtained
    // the meta in: `manifest.runs_at_level(l)` or explicit iteration
    // over levels.
}
```

The `windows(2).any(w[0] >= w[1])` check rejects both unsorted and duplicated entries in a single pass. `>=` catches equal adjacent values (dedup), `<` ordering guarantees strict monotonicity.

### 3. `LsmManifest` API updates

Signatures are mechanical — every `u8` level becomes `Level`, every `u64` seq becomes `SeqNo`.

```rust
pub fn add_run(&mut self, level: Level, meta: SortedRunMeta);
pub fn remove_run(&mut self, level: Level, path: &Path) -> Option<SortedRunMeta>;
pub fn promote_run(&mut self, from: Level, to: Level, path: &Path) -> Result<(), LsmError>;
pub fn runs_at_level(&self, level: Level) -> &[SortedRunMeta];
pub fn set_next_sequence(&mut self, seq: SeqNo);
pub fn next_sequence(&self) -> SeqNo;
// all_run_paths, total_runs unchanged.
```

Internal simplifications:

- `add_run` loses its `assert!((level as usize) < NUM_LEVELS, ...)` — `Level::new` already guaranteed it. The array index becomes `level.as_index()`.
- `remove_run` / `runs_at_level` same.
- `promote_run` no longer needs `meta.level = to_level` — the field is gone.
- `levels` array indexing replaces scattered bounds checks.

### 4. Wire format — unchanged

The existing encoding of `ManifestEntry::AddRun`, `AddRunAndSequence`, `RemoveRun`, `PromoteRun`, `SetSequence` is untouched. Per inspection of PR #160's codec:

- `AddRun` and `AddRunAndSequence` already encode `level` exactly once per entry (at the outer position). The inner `meta.level` was never serialized — decode was populating it from the outer byte. Dropping the field changes only memory layout, not bytes.
- `RemoveRun` and `PromoteRun` still encode `level: u8`, decoded via `Level::new(byte).ok_or(LsmError::Format(...))?`.
- `SetSequence` still encodes `next_sequence: u64`, decoded via `SeqNo(u64)`.

### 5. Decode path integration

`decode_entry` gains two new failure modes, both surfaced as `LsmError::Format`:

- Invalid `level` byte (>= `NUM_LEVELS`) → `Format("invalid level N")`.
- `SortedRunMeta::new` rejects the payload (unsorted coverage, zero page_count) → forwards that error.

These integrate automatically with the replay loop fixed in PR #160: decode-stage `LsmError::Format` is treated as tail corruption, triggering `truncate_at` and breaking the loop. No additional replay logic needed.

`apply_entry` is already `Result`-returning from PR #160; promotion with a missing source still propagates via `?`. The validated-constructor errors are caught one layer earlier (decode) so they never reach `apply_entry`.

### 6. Call-site migration inventory

Known construction sites needing updating:

- `crates/minkowski-lsm/src/manifest_ops.rs::flush_and_record` — one `SortedRunMeta` struct literal, becomes `SortedRunMeta::new(...)?`.
- `crates/minkowski-lsm/src/manifest_log.rs::decode_entry` — two construction sites (AddRun and AddRunAndSequence variants), both become `SortedRunMeta::new(...)?`.
- `crates/minkowski-lsm/src/manifest.rs` — `test_meta` helper in the test module, one construction.
- `crates/minkowski-lsm/src/manifest_ops.rs::cleanup_orphans_removes_untracked` test — one struct literal.

Signature-only updates (level as `u8` → `Level`, seq as `u64` → `SeqNo`):

- Public accessors on `LsmManifest`.
- `ManifestEntry` enum variants that carry `level` or `next_sequence`.
- Tests throughout `manifest.rs`, `manifest_log.rs`, `manifest_integration.rs`.

No external callers of `minkowski-lsm` outside the workspace exist today, so the breaking accessor-signature change is internal-only.

### 7. Error-type reuse

All new validation failures surface as existing `LsmError::Format(String)`. No new error variants. This matches the codec's existing convention: `Format` already covers "bytes on disk don't form a valid entry," and validation failures are semantically in the same category.

## Testing strategy

### Unit tests for new types (`types.rs` test module)

- `SeqNo` ordering, equality, `Display` round-trip.
- `SeqRange::new` rejects `lo > hi`, accepts `lo == hi`.
- `Level::new` rejects 4, 5, 255; accepts 0, 1, 2, 3.
- `Level::L0..L3` associated consts equal their `Level::new` counterparts.

### Unit tests for `SortedRunMeta::new`

Table-driven — one valid case, plus one case per rejected-invariant:

- Valid: sorted coverage, non-zero page_count → `Ok`.
- Unsorted coverage (`[3, 1, 2]`) → `Err(Format)`.
- Duplicated coverage (`[1, 2, 2, 3]`) → `Err(Format)`.
- Empty coverage → `Ok`. The constructor does not enforce a cross-field "coverage must be non-empty when page_count > 0" invariant; in practice `flush_and_record` never produces this state (a dirty page implies at least one archetype), but the type alone doesn't know that.
- `page_count == 0` → `Err(Format)`.

### Integration regression test

One new test in `manifest_integration.rs`: inject a handcrafted `ManifestEntry::AddRun` with `archetype_coverage: vec![3, 1]` via `log.append`, then `ManifestLog::replay` and assert:

- Replay returns `Ok` (no error propagated).
- Manifest has only the entries *before* the bad one.
- Log file is truncated to the offset immediately before the bad frame.

This wires the new validation into the existing torn-tail recovery path and confirms the error classification reaches the truncate branch.

### Existing tests

All existing tests in `manifest.rs`, `manifest_log.rs`, `manifest_integration.rs` should continue to pass with mechanical migration. No coverage is lost.

## Rollout

Single PR, squash-merge per project convention. CI gates: fmt, clippy, test, tsan, loom (loom is irrelevant here — no new concurrency), Miri runs on nightly schedule.

Following merge, update the `project_lsm_phase2_type_safety.md` memory entry to reflect PR A completion and adjust PR B scope (magic header + recover + dir fsync).

## Risks

- **Clippy lint drift**: PR #160's post-merge CI revealed three clippy 1.95 lints that my local (1.93) didn't trigger. Before pushing this PR, run `rustup update stable` locally and re-run `cargo clippy --workspace --all-targets -- -D warnings` to catch newer lints locally.
- **Test migration churn**: every test helper that built `SortedRunMeta` via struct literal needs updating. This is mechanical and will be caught by the compiler, but expect ~30–50 lines of test-file edits.
- **`ManifestEntry::RemoveRun` and `PromoteRun` carry `level: u8`** — these decode paths need the new `Level::new(...).ok_or(...)?` treatment. Easy to miss one in the migration; `cargo check` catches it because the variant field type changes.
