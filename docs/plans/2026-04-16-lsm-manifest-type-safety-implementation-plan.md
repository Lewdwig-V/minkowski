# LSM Manifest Type Safety (PR A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `SeqNo` / `SeqRange` / `Level` newtypes, a validated `SortedRunMeta::new` constructor, and drop the redundant `SortedRunMeta::level` field across the `minkowski-lsm` crate.

**Architecture:** Pure Rust type-level refactor. No wire format change, no cross-crate ripple. Adds ~150 lines to a new `types.rs` module; mechanical rewrites across the existing five files. TDD for new type logic; compiler-driven migration for signature changes. Each task ends with a green `cargo test -p minkowski-lsm` + `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`.

**Tech Stack:** Rust 2024 edition, `minkowski-lsm` workspace crate, existing `LsmError` error type, existing `tempfile` test harness.

**Spec:** `docs/plans/2026-04-16-lsm-manifest-type-safety-design.md`

---

## Starting state

- Branch: `docs/lsm-type-safety-design` (one commit ahead of `main` — the spec).
- Continue on this branch. When implementation completes, the PR squash-merges both the spec doc and the implementation as one clean squash commit.

## File structure

**Create:**
- `crates/minkowski-lsm/src/types.rs` — new module holding `SeqNo`, `SeqRange`, `Level`. One responsibility: shared invariant-carrying newtype primitives.

**Modify:**
- `crates/minkowski-lsm/src/lib.rs` — add `pub mod types;` and re-exports.
- `crates/minkowski-lsm/src/manifest.rs` — `SortedRunMeta::new`, accessor return types, `LsmManifest` signatures, drop `meta.level`.
- `crates/minkowski-lsm/src/manifest_log.rs` — `ManifestEntry` variant field types, encode/decode using new primitives, call `SortedRunMeta::new`.
- `crates/minkowski-lsm/src/manifest_ops.rs` — `flush_and_record` uses `SortedRunMeta::new` + `SeqRange::new`.
- `crates/minkowski-lsm/src/reader.rs` — `sequence_range()` accessor return type becomes `SeqRange` (the reader's internal storage stays `(u64, u64)` since that's what's on disk; only the public accessor changes, or it stays `(u64, u64)` and callers build a SeqRange — decide in Task 3).
- `crates/minkowski-lsm/tests/manifest_integration.rs` — test harness migration, add one new regression test.

---

## Task 1: Types module — `SeqNo`, `SeqRange`, `Level`

**Files:**
- Create: `crates/minkowski-lsm/src/types.rs`
- Modify: `crates/minkowski-lsm/src/lib.rs`

- [ ] **Step 1: Write the types module with inline tests**

Create `crates/minkowski-lsm/src/types.rs`:

```rust
//! Invariant-carrying newtype primitives shared across the manifest.

use std::fmt;

use crate::error::LsmError;
use crate::manifest::NUM_LEVELS;

/// A WAL sequence number.
///
/// Raw `u64` arithmetic on sequence numbers is disallowed by the type
/// (no `Add`/`Sub` impls) because sequences are identities, not sizes.
/// Callers that need "next seq" do so explicitly: `SeqNo(x.0 + 1)`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SeqNo(pub u64);

impl From<u64> for SeqNo {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<SeqNo> for u64 {
    fn from(s: SeqNo) -> Self {
        s.0
    }
}

impl fmt::Display for SeqNo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A half-open sequence range `[lo, hi)`.
///
/// Construction enforces `lo <= hi`. `hi == lo` represents an empty
/// range (syntactically allowed, not currently produced by any code path).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SeqRange {
    pub(crate) lo: SeqNo,
    pub(crate) hi: SeqNo,
}

impl SeqRange {
    pub fn new(lo: SeqNo, hi: SeqNo) -> Result<Self, LsmError> {
        if lo > hi {
            return Err(LsmError::Format(format!(
                "SeqRange: lo ({lo}) > hi ({hi})"
            )));
        }
        Ok(Self { lo, hi })
    }
}

/// An LSM level index. Construction enforces `< NUM_LEVELS`.
///
/// The bounds check lives in exactly one place (`Level::new`); all
/// other code sites trust the invariant once they hold a `Level`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Level(u8);

impl Level {
    pub const L0: Level = Level(0);
    pub const L1: Level = Level(1);
    pub const L2: Level = Level(2);
    pub const L3: Level = Level(3);

    pub fn new(level: u8) -> Option<Self> {
        if (level as usize) < NUM_LEVELS {
            Some(Self(level))
        } else {
            None
        }
    }

    pub fn as_u8(self) -> u8 {
        self.0
    }

    pub fn as_index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seqno_display_matches_inner_u64() {
        assert_eq!(SeqNo(42).to_string(), "42");
    }

    #[test]
    fn seqno_ordering_matches_u64() {
        assert!(SeqNo(1) < SeqNo(2));
        assert_eq!(SeqNo(5), SeqNo(5));
    }

    #[test]
    fn seqrange_rejects_lo_greater_than_hi() {
        let result = SeqRange::new(SeqNo(10), SeqNo(5));
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn seqrange_accepts_lo_equal_to_hi() {
        let r = SeqRange::new(SeqNo(5), SeqNo(5)).unwrap();
        assert_eq!(r.lo, SeqNo(5));
        assert_eq!(r.hi, SeqNo(5));
    }

    #[test]
    fn seqrange_accepts_lo_less_than_hi() {
        let r = SeqRange::new(SeqNo(0), SeqNo(10)).unwrap();
        assert_eq!(r.lo, SeqNo(0));
        assert_eq!(r.hi, SeqNo(10));
    }

    #[test]
    fn level_rejects_values_at_or_above_num_levels() {
        assert!(Level::new(NUM_LEVELS as u8).is_none());
        assert!(Level::new(255).is_none());
    }

    #[test]
    fn level_accepts_values_below_num_levels() {
        for i in 0..NUM_LEVELS {
            assert!(Level::new(i as u8).is_some());
        }
    }

    #[test]
    fn level_consts_match_new() {
        assert_eq!(Level::L0, Level::new(0).unwrap());
        assert_eq!(Level::L1, Level::new(1).unwrap());
        assert_eq!(Level::L2, Level::new(2).unwrap());
        assert_eq!(Level::L3, Level::new(3).unwrap());
    }

    #[test]
    fn level_as_u8_and_as_index_agree() {
        let l = Level::L2;
        assert_eq!(l.as_u8(), 2);
        assert_eq!(l.as_index(), 2);
    }
}
```

- [ ] **Step 2: Register the module in lib.rs**

Edit `crates/minkowski-lsm/src/lib.rs` — add to the module list (alphabetical ordering preserved):

```rust
pub mod error;
pub mod format;
pub mod manifest;
pub mod manifest_log;
pub mod manifest_ops;
pub mod reader;
pub mod schema;
pub mod types;      // <-- new
pub mod writer;
```

- [ ] **Step 3: Run the new tests**

Run: `cargo test -p minkowski-lsm --lib types::tests`
Expected: 8 tests pass.

- [ ] **Step 4: Run the full crate test suite to verify nothing else broke**

Run: `cargo test -p minkowski-lsm`
Expected: all existing tests still pass, plus the 8 new `types::tests`.

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski-lsm/src/types.rs crates/minkowski-lsm/src/lib.rs
git commit -m "feat(lsm): add SeqNo, SeqRange, Level newtype primitives

Foundation for the manifest type-safety refactor. Pure addition — no
existing code changed yet. Each newtype centralizes an invariant that
was previously scattered across assert! sites or open structural fields."
```

---

## Task 2: `SortedRunMeta::new` constructor (keep fields pub(crate))

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs`

- [ ] **Step 1: Write failing unit tests for the constructor**

In `crates/minkowski-lsm/src/manifest.rs`, add to the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn sorted_run_meta_new_accepts_valid_input() {
        let meta = SortedRunMeta::new(
            PathBuf::from("0-10.run"),
            0,
            SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
            vec![0, 3, 7],
            1,
            1024,
        )
        .unwrap();
        assert_eq!(meta.sequence_range().lo, SeqNo(0));
        assert_eq!(meta.page_count(), 1);
    }

    #[test]
    fn sorted_run_meta_new_rejects_unsorted_coverage() {
        let result = SortedRunMeta::new(
            PathBuf::from("x.run"),
            0,
            SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
            vec![3, 1, 2],
            1,
            1024,
        );
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn sorted_run_meta_new_rejects_duplicated_coverage() {
        let result = SortedRunMeta::new(
            PathBuf::from("x.run"),
            0,
            SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
            vec![1, 2, 2, 3],
            1,
            1024,
        );
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn sorted_run_meta_new_accepts_empty_coverage() {
        let meta = SortedRunMeta::new(
            PathBuf::from("x.run"),
            0,
            SeqRange::new(SeqNo(0), SeqNo(0)).unwrap(),
            vec![],
            1,
            1024,
        );
        assert!(meta.is_ok());
    }

    #[test]
    fn sorted_run_meta_new_rejects_zero_page_count() {
        let result = SortedRunMeta::new(
            PathBuf::from("x.run"),
            0,
            SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
            vec![0],
            0,
            1024,
        );
        assert!(matches!(result, Err(LsmError::Format(_))));
    }
```

Add to the test module's imports (or `use super::*;` if already present):
```rust
    use crate::types::{SeqNo, SeqRange};
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-lsm --lib manifest::tests::sorted_run_meta_new`
Expected: FAIL — `SortedRunMeta::new` doesn't exist yet AND `sequence_range()` still returns `(u64, u64)` not `SeqRange` — compilation error.

- [ ] **Step 3: Add `new()` and intermediate accessor bridge**

In `crates/minkowski-lsm/src/manifest.rs`, at the top of the file, update imports:

```rust
use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::types::{SeqNo, SeqRange};
```

Then inside `impl SortedRunMeta { ... }`, add the constructor:

```rust
    /// Build a `SortedRunMeta` with enforced invariants.
    ///
    /// Validates:
    /// - `archetype_coverage` is strictly sorted ascending (sorted + deduped).
    /// - `page_count` is non-zero.
    ///
    /// `seq_range` is already validated by `SeqRange::new`. `size_bytes` is
    /// not validated (redundant with `page_count`; a valid run file always
    /// has a non-empty header).
    pub fn new(
        path: PathBuf,
        level: u8,
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,
        page_count: u64,
        size_bytes: u64,
    ) -> Result<Self, LsmError> {
        if archetype_coverage.windows(2).any(|w| w[0] >= w[1]) {
            return Err(LsmError::Format(
                "archetype_coverage is not strictly sorted".to_owned(),
            ));
        }
        if page_count == 0 {
            return Err(LsmError::Format(
                "page_count must be non-zero".to_owned(),
            ));
        }
        Ok(Self {
            path,
            level,
            sequence_range,
            archetype_coverage,
            page_count,
            size_bytes,
        })
    }
```

Change the `sequence_range` field type in the struct from `(u64, u64)` to `SeqRange`:

```rust
pub struct SortedRunMeta {
    pub(crate) path: PathBuf,
    pub(crate) level: u8,
    pub(crate) sequence_range: SeqRange,       // was: (u64, u64)
    pub(crate) archetype_coverage: Vec<u16>,
    pub(crate) page_count: u64,
    pub(crate) size_bytes: u64,
}
```

Update the existing accessor:

```rust
    pub fn sequence_range(&self) -> SeqRange {
        self.sequence_range
    }
```

- [ ] **Step 4: Update existing `test_meta` helper to use `SeqRange`**

In the same file's test module, update `test_meta`:

```rust
    fn test_meta(name: &str, level: u8) -> SortedRunMeta {
        SortedRunMeta {
            path: PathBuf::from(name),
            level,
            sequence_range: SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
            archetype_coverage: vec![0],
            page_count: 1,
            size_bytes: 1024,
        }
    }
```

Note: existing tests use struct literal — this is fine while fields are still `pub(crate)`. Task 5 flips them private.

- [ ] **Step 5: Run tests — expect compilation errors elsewhere now**

Run: `cargo check -p minkowski-lsm`
Expected: FAIL — `manifest_log.rs`, `manifest_ops.rs`, `reader.rs`, and integration tests all construct `SortedRunMeta` with `sequence_range: (u64, u64)` tuple. These are next to migrate.

- [ ] **Step 6: Update `crates/minkowski-lsm/src/reader.rs` sequence_range accessor**

Find `SortedRunReader::sequence_range()`:

```rust
    pub fn sequence_range(&self) -> (u64, u64) {
        (self.header.sequence_lo, self.header.sequence_hi)
    }
```

Change return type to `SeqRange`:

```rust
    pub fn sequence_range(&self) -> SeqRange {
        // On-disk header validated at open time; these u64s already form
        // a valid range there, so new() will succeed. unwrap justified by
        // file-open-time invariant.
        SeqRange::new(
            SeqNo(self.header.sequence_lo),
            SeqNo(self.header.sequence_hi),
        )
        .expect("sorted-run header sequence range must be valid")
    }
```

Add imports at top of `reader.rs`:

```rust
use crate::types::{SeqNo, SeqRange};
```

- [ ] **Step 7: Run cargo check again**

Run: `cargo check -p minkowski-lsm`
Expected: still FAIL — callers of `SortedRunMeta` struct literal with `(u64, u64)` still remain. Will be fixed in next tasks.

Don't commit yet — the codebase isn't compiling. This entire task commits at the end of Task 3 (below), once all callers are migrated.

---

## Task 3: Migrate all `SortedRunMeta` construction sites

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs`
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs`

- [ ] **Step 1: Migrate `flush_and_record` in manifest_ops.rs**

Edit the construction site. Current code:

```rust
    let meta = SortedRunMeta {
        path: path.clone(),
        level: 0,
        sequence_range: reader.sequence_range(),
        archetype_coverage,
        page_count: reader.page_count(),
        size_bytes: file_size,
    };
```

Replace with:

```rust
    let meta = SortedRunMeta::new(
        path.clone(),
        0,
        reader.sequence_range(),
        archetype_coverage,
        reader.page_count(),
        file_size,
    )?;
```

`reader.sequence_range()` already returns `SeqRange` after Task 2 Step 6. The `?` propagates any validation error (shouldn't happen in practice — a just-written run has valid coverage — but the type system now demands handling).

- [ ] **Step 2: Migrate the cleanup test in manifest_ops.rs**

Find `cleanup_orphans_removes_untracked` in the test module. Current:

```rust
        manifest.add_run(
            0,
            SortedRunMeta {
                path: dir.path().join("0-10.run"),
                level: 0,
                sequence_range: (0, 10),
                archetype_coverage: vec![0],
                page_count: 1,
                size_bytes: 4,
            },
        );
```

Replace:

```rust
        manifest.add_run(
            0,
            SortedRunMeta::new(
                dir.path().join("0-10.run"),
                0,
                SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
                vec![0],
                1,
                4,
            )
            .unwrap(),
        );
```

Add import at top of file's test module (or `use` at top of file):

```rust
use crate::types::{SeqNo, SeqRange};
```

- [ ] **Step 3: Migrate decode_entry in manifest_log.rs — two sites**

Find the `TAG_ADD_RUN` decode arm. Current:

```rust
            Ok(ManifestEntry::AddRun {
                level,
                meta: SortedRunMeta {
                    path,
                    level,
                    sequence_range: (seq_lo, seq_hi),
                    archetype_coverage: coverage,
                    page_count,
                    size_bytes,
                },
            })
```

Replace:

```rust
            let meta = SortedRunMeta::new(
                path,
                level,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,
                size_bytes,
            )?;
            Ok(ManifestEntry::AddRun { level, meta })
```

Same pattern for `TAG_ADD_RUN_AND_SEQUENCE`:

```rust
            let meta = SortedRunMeta::new(
                path,
                level,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,
                size_bytes,
            )?;
            Ok(ManifestEntry::AddRunAndSequence {
                level,
                meta,
                next_sequence,
            })
```

Add import at top of `manifest_log.rs`:

```rust
use crate::types::{SeqNo, SeqRange};
```

- [ ] **Step 4: Migrate encode sites in manifest_log.rs**

The encoder reads `meta.sequence_range.0` and `meta.sequence_range.1`. After the field type change, these accesses become `.lo.0` and `.hi.0`. Find and update two sites (AddRun and AddRunAndSequence branches in `encode_entry`):

```rust
            buf.extend_from_slice(&meta.sequence_range.lo.0.to_le_bytes());
            buf.extend_from_slice(&meta.sequence_range.hi.0.to_le_bytes());
```

(Previously: `.sequence_range.0.to_le_bytes()` and `.sequence_range.1.to_le_bytes()`.)

- [ ] **Step 5: Migrate the test_meta helper in manifest_log.rs test module**

Find:

```rust
    fn test_meta(name: &str) -> SortedRunMeta {
        SortedRunMeta {
            path: PathBuf::from(name),
            level: 0,
            sequence_range: (10, 20),
            archetype_coverage: vec![0, 3, 7],
            page_count: 42,
            size_bytes: 8192,
        }
    }
```

Replace:

```rust
    fn test_meta(name: &str) -> SortedRunMeta {
        SortedRunMeta::new(
            PathBuf::from(name),
            0,
            SeqRange::new(SeqNo(10), SeqNo(20)).unwrap(),
            vec![0, 3, 7],
            42,
            8192,
        )
        .unwrap()
    }
```

- [ ] **Step 6: Integration tests — reader.sequence_range() usage**

In `crates/minkowski-lsm/tests/manifest_integration.rs`, `three_flushes_then_replay` currently asserts:

```rust
        assert_eq!(original.sequence_range(), recovered.sequence_range());
```

`SeqRange` has `PartialEq`, so this continues to work unchanged.

If integration tests construct `SortedRunMeta` directly, migrate them using the same pattern. (A quick `grep "SortedRunMeta {" tests/` catches any missed sites.)

Run: `grep -rn "SortedRunMeta {" crates/minkowski-lsm/` — any remaining matches need migrating.

- [ ] **Step 7: Run `cargo check` — expect success now**

Run: `cargo check -p minkowski-lsm`
Expected: clean compile.

- [ ] **Step 8: Run the full test suite**

Run: `cargo test -p minkowski-lsm`
Expected: all tests pass, including the 5 new `sorted_run_meta_new_*` tests.

- [ ] **Step 9: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add crates/minkowski-lsm/src/manifest.rs \
        crates/minkowski-lsm/src/manifest_log.rs \
        crates/minkowski-lsm/src/manifest_ops.rs \
        crates/minkowski-lsm/src/reader.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "feat(lsm): validated SortedRunMeta::new constructor + SeqRange

Introduces the SortedRunMeta::new constructor that validates
archetype_coverage is strictly sorted and page_count is non-zero.
Migrates all in-crate construction sites through it.

Changes sequence_range field type from (u64, u64) to SeqRange across
SortedRunMeta, SortedRunReader, and the wire decoder. Encode path now
accesses .lo.0 / .hi.0 for the u64 bytes. No wire format change.

Fields remain pub(crate) for one more commit — Task 4 flips them fully
private once all construction has been verified to go through new()."
```

---

## Task 4: Make `SortedRunMeta` fields fully private

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs`

- [ ] **Step 1: Flip field visibility**

Change each `pub(crate)` to private (remove the visibility modifier):

```rust
pub struct SortedRunMeta {
    path: PathBuf,
    level: u8,
    sequence_range: SeqRange,
    archetype_coverage: Vec<u16>,
    page_count: u64,
    size_bytes: u64,
}
```

- [ ] **Step 2: Run cargo check — verify no construction site still uses struct literal**

Run: `cargo check -p minkowski-lsm`
Expected: clean compile. If anything fails, it's a missed construction site — migrate it via `SortedRunMeta::new` and re-run.

- [ ] **Step 3: Run full tests**

Run: `cargo test -p minkowski-lsm`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski-lsm/src/manifest.rs
git commit -m "refactor(lsm): SortedRunMeta fields fully private

All construction now goes through SortedRunMeta::new. Flip field
visibility from pub(crate) to private to enforce this at the type
system level — no more struct-literal construction bypassing validation.

Behavior-preserving; all accessors already exist on the impl."
```

---

## Task 5: Drop `SortedRunMeta::level` field

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs`
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`

- [ ] **Step 1: Audit all readers of `meta.level()`**

Run: `grep -rn "\.level()" crates/minkowski-lsm/`

Expected results (as of the current branch):
- `manifest.rs:183` in `promote_run_moves_between_levels` test — `assert_eq!(promoted.level(), 1);`
- Possibly `manifest_log.rs` tests or `manifest_integration.rs`.

Each reader needs migration: the level is always known from surrounding context (the `runs_at_level(l)` call, or the outer `AddRun.level` field, or the loop variable).

- [ ] **Step 2: Remove the field and accessor**

In `crates/minkowski-lsm/src/manifest.rs`:

```rust
pub struct SortedRunMeta {
    path: PathBuf,
    // level: u8,   <-- REMOVED
    sequence_range: SeqRange,
    archetype_coverage: Vec<u16>,
    page_count: u64,
    size_bytes: u64,
}

impl SortedRunMeta {
    // Delete the `pub fn level(&self) -> u8` accessor entirely.
    // ... other accessors unchanged ...
}
```

Update the `new` constructor signature to drop the `level` parameter:

```rust
    pub fn new(
        path: PathBuf,
        // level parameter removed
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,
        page_count: u64,
        size_bytes: u64,
    ) -> Result<Self, LsmError> {
```

Update `promote_run` — the `meta.level = to_level` line is gone (field doesn't exist):

```rust
    pub fn promote_run(
        &mut self,
        from_level: u8,
        to_level: u8,
        path: &Path,
    ) -> Result<(), LsmError> {
        let meta = self.remove_run(from_level, path).ok_or_else(|| {
            LsmError::Format(format!(
                "run {} not found at level {}",
                path.display(),
                from_level
            ))
        })?;
        self.add_run(to_level, meta);
        Ok(())
    }
```

- [ ] **Step 3: Update callers of `SortedRunMeta::new` to drop the level arg**

Expect compile errors in:
- `manifest_ops.rs::flush_and_record`
- `manifest_log.rs::decode_entry` (two sites)
- Test helpers (`test_meta` in manifest.rs, manifest_log.rs; cleanup_orphans test in manifest_ops.rs)
- The 5 `sorted_run_meta_new_*` unit tests added in Task 2 — drop the `0` level argument from each call.

For `flush_and_record`:

```rust
    let meta = SortedRunMeta::new(
        path.clone(),
        reader.sequence_range(),
        archetype_coverage,
        reader.page_count(),
        file_size,
    )?;
```

For `decode_entry` AddRun arm:

```rust
            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,
                size_bytes,
            )?;
            Ok(ManifestEntry::AddRun { level, meta })
```

Same pattern for AddRunAndSequence.

Test helpers drop the `level` arg from the `new` call — note `test_meta` in manifest.rs still takes `level: u8` as a parameter, but only uses it for the test's tracking context, not for `SortedRunMeta` construction. Change signature to no longer require it, OR (simpler) keep the parameter for test readability but don't pass it to `new`. Simpler option:

```rust
    fn test_meta(name: &str, _level: u8) -> SortedRunMeta {
        SortedRunMeta::new(
            PathBuf::from(name),
            SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(),
            vec![0],
            1,
            1024,
        )
        .unwrap()
    }
```

But cleaner: drop the parameter entirely. Callers of `test_meta("foo.sst", 0)` become `test_meta("foo.sst")`.

- [ ] **Step 4: Update the `promote_run_moves_between_levels` test**

Original:

```rust
    #[test]
    fn promote_run_moves_between_levels() {
        let mut m = LsmManifest::new();
        let meta = test_meta("run_x.sst", 0);
        m.add_run(0, meta);
        m.promote_run(0, 1, Path::new("run_x.sst")).unwrap();
        assert!(m.runs_at_level(0).is_empty());
        let promoted = &m.runs_at_level(1)[0];
        assert_eq!(promoted.path(), Path::new("run_x.sst"));
        assert_eq!(promoted.level(), 1);
    }
```

Replace the `promoted.level()` check — it's gone. Level is now implicit from `runs_at_level(1)`:

```rust
    #[test]
    fn promote_run_moves_between_levels() {
        let mut m = LsmManifest::new();
        let meta = test_meta("run_x.sst");
        m.add_run(0, meta);
        m.promote_run(0, 1, Path::new("run_x.sst")).unwrap();
        assert!(m.runs_at_level(0).is_empty());
        assert_eq!(m.runs_at_level(1).len(), 1);
        assert_eq!(m.runs_at_level(1)[0].path(), Path::new("run_x.sst"));
    }
```

- [ ] **Step 5: Run cargo check**

Run: `cargo check -p minkowski-lsm`
Expected: clean. Any remaining error is a missed `meta.level()` reader or `test_meta(_, level)` call site.

- [ ] **Step 6: Run full tests**

Run: `cargo test -p minkowski-lsm`
Expected: all pass.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski-lsm/
git commit -m "refactor(lsm): drop redundant SortedRunMeta::level field

The field was always a derived copy of the outer ManifestEntry::AddRun
level (never independently stored on disk). Its in-memory existence
created two sources of truth for one fact, surfacing as a real bug in
PR #160 where promote_run left the field stale.

The field and its accessor are gone. Callers that need the level know
it from the runs_at_level(l) call or the outer entry variant — no
information lost. No wire format change.

promote_run simplifies: no more meta.level = to_level sync step."
```

---

## Task 6: Migrate `LsmManifest` signatures + `ManifestEntry` variants to `Level`

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs`
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs`
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs`

- [ ] **Step 1: Update LsmManifest public signatures**

In `crates/minkowski-lsm/src/manifest.rs`:

```rust
use crate::types::{Level, SeqNo, SeqRange};

impl LsmManifest {
    pub fn add_run(&mut self, level: Level, meta: SortedRunMeta) {
        self.levels[level.as_index()].push(meta);
    }

    pub fn remove_run(&mut self, level: Level, path: &Path) -> Option<SortedRunMeta> {
        let runs = &mut self.levels[level.as_index()];
        runs.iter()
            .position(|r| r.path() == path)
            .map(|pos| runs.remove(pos))
    }

    pub fn promote_run(
        &mut self,
        from: Level,
        to: Level,
        path: &Path,
    ) -> Result<(), LsmError> {
        let meta = self.remove_run(from, path).ok_or_else(|| {
            LsmError::Format(format!(
                "run {} not found at level {}",
                path.display(),
                from
            ))
        })?;
        self.add_run(to, meta);
        Ok(())
    }

    pub fn runs_at_level(&self, level: Level) -> &[SortedRunMeta] {
        &self.levels[level.as_index()]
    }
}
```

The `assert!((level as usize) < NUM_LEVELS, ...)` in `add_run` goes away — `Level::new` enforced it at construction.

Note: `remove_run` used `r.path` (direct field access). After Task 4 private fields, that field is private. Use the accessor `r.path()` instead.

- [ ] **Step 2: Update ManifestEntry variants**

In `crates/minkowski-lsm/src/manifest_log.rs`:

```rust
use crate::types::{Level, SeqNo, SeqRange};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestEntry {
    AddRun {
        level: Level,
        meta: SortedRunMeta,
    },
    RemoveRun {
        level: Level,
        path: PathBuf,
    },
    PromoteRun {
        from_level: Level,
        to_level: Level,
        path: PathBuf,
    },
    SetSequence {
        next_sequence: SeqNo,
    },
    AddRunAndSequence {
        level: Level,
        meta: SortedRunMeta,
        next_sequence: SeqNo,
    },
}
```

- [ ] **Step 3: Update encode_entry for Level and SeqNo**

In `encode_entry`:

```rust
        ManifestEntry::AddRun { level, meta } => {
            buf.push(TAG_ADD_RUN);
            buf.push(level.as_u8());            // was: *level
            ...
        }
        ManifestEntry::RemoveRun { level, path } => {
            buf.push(TAG_REMOVE_RUN);
            buf.push(level.as_u8());            // was: *level
            ...
        }
        ManifestEntry::PromoteRun { from_level, to_level, path } => {
            buf.push(TAG_PROMOTE_RUN);
            buf.push(from_level.as_u8());       // was: *from_level
            buf.push(to_level.as_u8());         // was: *to_level
            ...
        }
        ManifestEntry::SetSequence { next_sequence } => {
            buf.push(TAG_SET_SEQUENCE);
            buf.extend_from_slice(&next_sequence.0.to_le_bytes());   // was: next_sequence.to_le_bytes()
        }
        ManifestEntry::AddRunAndSequence { level, meta, next_sequence } => {
            buf.push(TAG_ADD_RUN_AND_SEQUENCE);
            buf.push(level.as_u8());
            ...
            buf.extend_from_slice(&next_sequence.0.to_le_bytes());
        }
```

- [ ] **Step 4: Update decode_entry for Level and SeqNo**

Every `data[offset]` that represents a level now needs `Level::new(...)`:

```rust
        TAG_ADD_RUN => {
            if offset >= data.len() {
                return Err(LsmError::Format("truncated AddRun".to_owned()));
            }
            let level_byte = data[offset];
            offset += 1;
            let level = Level::new(level_byte)
                .ok_or_else(|| LsmError::Format(format!("invalid level {level_byte}")))?;
            let path = decode_path(data, &mut offset)?;
            let seq_lo = read_u64_le(data, &mut offset)?;
            let seq_hi = read_u64_le(data, &mut offset)?;
            let count = read_u16_le(data, &mut offset)? as usize;
            // ... (coverage loop unchanged)
            let page_count = read_u64_le(data, &mut offset)?;
            let size_bytes = read_u64_le(data, &mut offset)?;

            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,
                size_bytes,
            )?;
            Ok(ManifestEntry::AddRun { level, meta })
        }
```

Apply the same `level_byte → Level::new(...)?` pattern to `TAG_REMOVE_RUN`, `TAG_PROMOTE_RUN` (two bytes), and `TAG_ADD_RUN_AND_SEQUENCE`.

For `TAG_SET_SEQUENCE`:

```rust
        TAG_SET_SEQUENCE => {
            let next_sequence = SeqNo(read_u64_le(data, &mut offset)?);
            Ok(ManifestEntry::SetSequence { next_sequence })
        }
```

- [ ] **Step 5: Update `apply_entry`**

Inside `apply_entry`:

```rust
        ManifestEntry::AddRun { level, meta } => manifest.add_run(*level, meta.clone()),
        ManifestEntry::RemoveRun { level, path } => {
            manifest.remove_run(*level, path);
        }
        ManifestEntry::PromoteRun { from_level, to_level, path } => {
            manifest.promote_run(*from_level, *to_level, path)?;
        }
        ManifestEntry::SetSequence { next_sequence } => {
            manifest.set_next_sequence(*next_sequence);
        }
        ManifestEntry::AddRunAndSequence { level, meta, next_sequence } => {
            manifest.add_run(*level, meta.clone());
            manifest.set_next_sequence(*next_sequence);
        }
```

`*level` is now `Level` (Copy), `*next_sequence` is `SeqNo` (Copy). Signatures downstream take these types; no conversion needed.

- [ ] **Step 6: Update LsmManifest::set_next_sequence and next_sequence signatures**

In `manifest.rs`:

```rust
    pub fn set_next_sequence(&mut self, seq: SeqNo) {
        self.next_sequence = seq.0;
    }

    pub fn next_sequence(&self) -> SeqNo {
        SeqNo(self.next_sequence)
    }
```

Keep the internal `next_sequence: u64` storage — the public face is `SeqNo`, the internal representation stays the same.

- [ ] **Step 7: Migrate flush_and_record in manifest_ops.rs**

```rust
    log.append(&ManifestEntry::AddRunAndSequence {
        level: Level::L0,                                      // was: 0
        meta: meta.clone(),
        next_sequence: SeqNo(sequence_range.1),                // was: sequence_range.1
    })?;

    manifest.add_run(Level::L0, meta);                         // was: 0
    manifest.set_next_sequence(SeqNo(sequence_range.1));       // was: sequence_range.1
```

Wait — `sequence_range` here is still `(u64, u64)` because it's the function parameter tuple, not the SeqRange on the meta. Keep as is (external callers still pass the tuple for now). Internal conversion to SeqRange happens when building the meta.

Update import at top of file:

```rust
use crate::types::{Level, SeqNo, SeqRange};
```

- [ ] **Step 8: Migrate test call sites**

Throughout `manifest.rs` test module, `manifest_log.rs` test module, `manifest_ops.rs` test module, and `manifest_integration.rs`:

- Change `manifest.add_run(0, meta)` → `manifest.add_run(Level::L0, meta)`.
- Change `manifest.runs_at_level(0)` → `manifest.runs_at_level(Level::L0)`.
- Change `manifest.promote_run(0, 1, path)` → `manifest.promote_run(Level::L0, Level::L1, path)`.
- Change `ManifestEntry::AddRun { level: 0, ... }` → `ManifestEntry::AddRun { level: Level::L0, ... }`.
- Change `ManifestEntry::SetSequence { next_sequence: 10 }` → `ManifestEntry::SetSequence { next_sequence: SeqNo(10) }`.
- For assertion sites comparing `next_sequence()`, compare to `SeqNo(value)`: `assert_eq!(manifest.next_sequence(), SeqNo(10));`.

Import in integration tests:

```rust
use minkowski_lsm::types::{Level, SeqNo, SeqRange};
```

Run `cargo check -p minkowski-lsm` and iterate — the compiler enumerates the remaining sites. Expect ~20-40 line-level edits across all test files.

- [ ] **Step 9: Run cargo check, tests, clippy**

```
cargo check -p minkowski-lsm
cargo test -p minkowski-lsm
cargo clippy -p minkowski-lsm --all-targets -- -D warnings
```

All expected to pass.

- [ ] **Step 10: Commit**

```bash
git add crates/minkowski-lsm/
git commit -m "refactor(lsm): migrate LsmManifest and ManifestEntry to Level/SeqNo

All public APIs taking levels now take Level. All public APIs exposing
sequence numbers use SeqNo. ManifestEntry variants follow suit.

Bounds checking centralizes in Level::new. The scattered assert! sites
in LsmManifest are gone — the type system has already rejected invalid
levels at decode time (Level::new returns None -> LsmError::Format).

Wire format unchanged: Level.as_u8()/Level::new(byte) at codec edges,
SeqNo.0/SeqNo(u64) at seq-field edges. Internal u8/u64 storage is
unchanged; only the API surface shifts."
```

---

## Task 7: Regression test — corrupted archetype_coverage triggers tail truncation

**Files:**
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs`

- [ ] **Step 1: Write the regression test**

Add to `manifest_integration.rs`:

```rust
/// Regression for PR A: a frame whose decoded SortedRunMeta fails
/// validation (unsorted coverage) must be treated as tail garbage.
/// This wires the new constructor's validation into the existing
/// torn-tail recovery path.
#[test]
fn replay_truncates_log_on_unsorted_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // One real flush, produces a valid AddRunAndSequence frame.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    let len_after_first_frame = fs::metadata(&log_path).unwrap().len();

    // Manually craft an AddRun frame with unsorted archetype_coverage.
    // Bypasses SortedRunMeta::new (can't call it — would error) by
    // encoding the bytes directly.
    let mut payload = Vec::new();
    payload.push(0x01); // TAG_ADD_RUN
    payload.push(0);    // level
    // path: "x.run"
    let path_bytes = b"x.run";
    payload.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(path_bytes);
    payload.extend_from_slice(&0u64.to_le_bytes()); // seq_lo
    payload.extend_from_slice(&10u64.to_le_bytes()); // seq_hi
    // archetype_coverage: [3, 1] — intentionally unsorted
    payload.extend_from_slice(&2u16.to_le_bytes()); // count
    payload.extend_from_slice(&3u16.to_le_bytes());
    payload.extend_from_slice(&1u16.to_le_bytes());
    payload.extend_from_slice(&1u64.to_le_bytes()); // page_count
    payload.extend_from_slice(&1024u64.to_le_bytes()); // size_bytes

    // Write frame: [len: u32 LE][crc32: u32 LE][payload]
    let mut f = fs::OpenOptions::new().append(true).open(&log_path).unwrap();
    let len = payload.len() as u32;
    let crc = crc32fast::hash(&payload);
    f.write_all(&len.to_le_bytes()).unwrap();
    f.write_all(&crc.to_le_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.sync_all().unwrap();
    drop(f);

    // Replay must truncate back to the end of the first valid frame.
    let recovered = ManifestLog::replay(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 1, "only the valid first flush survives");

    let len_after_replay = fs::metadata(&log_path).unwrap().len();
    assert_eq!(
        len_after_replay, len_after_first_frame,
        "replay truncated the bad frame"
    );
}
```

Add `std::io::Write` to imports at the top of the integration test file (already present from existing tests; verify).

`crc32fast` is already a main dependency of `minkowski-lsm` (see `crates/minkowski-lsm/Cargo.toml:11`), so it's usable directly from integration tests via `minkowski_lsm`-adjacent imports. Specifically, access it as `use crc32fast;` at the top of `manifest_integration.rs` — integration tests can depend on main dependencies of the crate under test.

If the import fails at compile time (integration tests sometimes need transitive deps declared explicitly in `[dev-dependencies]`), add to `crates/minkowski-lsm/Cargo.toml`:

```toml
[dev-dependencies]
crc32fast = "1"
# existing dev deps unchanged
```

- [ ] **Step 2: Run the new test to verify it passes**

Run: `cargo test -p minkowski-lsm --test manifest_integration replay_truncates_log_on_unsorted_coverage`
Expected: PASS.

- [ ] **Step 3: Run full crate tests**

Run: `cargo test -p minkowski-lsm`
Expected: all pass, including this new test.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski-lsm/
git commit -m "test(lsm): replay truncates on SortedRunMeta validation failure

Regression covering the new SortedRunMeta::new validation: a log frame
whose decoded coverage vec fails the sorted+deduped check must be
treated as tail garbage, not propagated as a fatal error. Wires the
new validation into the existing torn-tail recovery path.

Handcrafts the bad frame with raw bytes + crc32 to bypass the
constructor (which would otherwise reject the payload at construction)."
```

---

## Task 8: Final verification

**No code changes. Green-light gate before PR.**

- [ ] **Step 1: Update local toolchain to match CI**

Run: `rustup update stable`
Expected: toolchain updates (or no-op if already current).

- [ ] **Step 2: Run workspace clippy (matches CI)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

If any lint fires that your local 1.93 missed, fix it and amend the relevant commit (or add a new chore commit).

- [ ] **Step 3: Run workspace test suite**

Run: `cargo test --workspace`
Expected: all pass.

- [ ] **Step 4: Run cargo fmt check**

Run: `cargo fmt --all -- --check`
Expected: clean.

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin docs/lsm-type-safety-design
gh pr create --title "feat(lsm): manifest type-safety refactor (PR A)" --body "$(cat <<'EOF'
## Summary

First half of the type-safety follow-up from PR #160's review. Introduces
`SeqNo` / `SeqRange` / `Level` newtype primitives, a validated
`SortedRunMeta::new` constructor, and drops the redundant
`SortedRunMeta::level` field.

Design: docs/plans/2026-04-16-lsm-manifest-type-safety-design.md

No wire format change — purely internal type refactor.

## Test plan

- [x] cargo test -p minkowski-lsm (all pass, including 13 new tests across types + constructor + regression)
- [x] cargo clippy --workspace --all-targets -- -D warnings (clean)
- [x] cargo test --workspace (all pass)

🤖 Generated with Claude Code
EOF
)"
```

- [ ] **Step 6: Update project memory after merge**

Once the PR merges, update `project_lsm_phase2_type_safety.md` to reflect PR A completion. Trim items 1-4 from the follow-up list; PR B scope remains (items 5-7).

---

## Self-review (done inline before saving)

- **Spec coverage:** Each of the 7 design sections in the spec has a task:
  - Section 1 (new types) → Task 1
  - Section 2 (SortedRunMeta shape) → Tasks 2, 4, 5
  - Section 3 (LsmManifest signatures) → Task 6
  - Section 4 (wire format unchanged) → enforced by Task 6's encode/decode edits
  - Section 5 (decode integration) → Task 6 Step 4
  - Section 6 (migration inventory) → spread across Tasks 2-6
  - Section 7 (error type reuse) → enforced by using `LsmError::Format` throughout
  - Testing strategy → Tasks 1, 2, 7

- **Placeholder scan:** None. Every code block is full code.

- **Type consistency:** `Level::L0` / `Level::new(0).unwrap()` / `Level::new(byte).ok_or(...)?` used consistently. `SeqNo(u64)` / `SeqNo(seq_hi)` / `seq.0` consistently. `SortedRunMeta::new` signature in Task 2 takes 6 args; in Task 5 drops `level` to 5 args. Documented at the Task 5 transition.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-04-16-lsm-manifest-type-safety-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good for an 8-task plan with compiler-driven migrations where the model could get lost in the weeds.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
