# LSM Manifest Format Hardening (PR B1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add magic + version header to the manifest log file, introduce `ManifestLog::recover()` as the sole public entry (closing the `open_or_create` footgun), and fix the `RemoveRun` silent no-op in `apply_entry`.

**Architecture:** Pure Rust file-format hardening. Adds an 8-byte header at offset 0 of `manifest.log`. Refactors `ManifestLog`'s public API from three entries (`create` / `open_or_create` / `replay`) to one (`recover`), with `create` kept as `pub(crate)` for test use. Mostly TDD; compiler-driven migration for the one public-API signature change. Each task ends with green `cargo test -p minkowski-lsm` + `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`.

**Tech Stack:** Rust 2024 edition, `minkowski-lsm` workspace crate, existing `LsmError` error type, existing `tempfile` test harness, `crc32fast` for frame CRC (unchanged).

**Spec:** `docs/plans/2026-04-17-lsm-manifest-format-hardening-design.md`

---

## Starting state

- Branch: `lsm/pr-b-format-hardening` (one commit ahead of `main` — the spec).
- Continue on this branch. When implementation completes, the PR squash-merges both the spec and implementation as one clean squash commit.
- Base commit: `ff1e366` (spec commit).

## File structure

**Modify:**
- `crates/minkowski-lsm/src/manifest_log.rs` — header constants, `write_header` / `validate_header` helpers, `recover()` function, `replay_frames` extracted from `replay`, API deletions, `RemoveRun` propagation fix.
- `crates/minkowski-lsm/src/manifest_ops.rs` — test call sites shift from `ManifestLog::create` to `ManifestLog::recover` where they set up a log.
- `crates/minkowski-lsm/tests/manifest_integration.rs` — migrate existing tests to `recover()`, adjust byte-prefix convergence test for the 8-byte header offset, add four new integration tests.

**No new files.** All changes fit within the existing module layout.

---

## Task 1: Header helpers (TDD)

**Goal:** Add the 8-byte header encoding/decoding helpers with full test coverage. No behavior change yet — these are building blocks for Task 2.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`

- [ ] **Step 1: Write failing unit tests**

In the existing `#[cfg(test)] mod tests` block of `manifest_log.rs`, add:

```rust
    #[test]
    fn write_header_emits_expected_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        let mut file = File::create(&path).unwrap();
        write_header(&mut file).unwrap();
        drop(file);

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], b"MKMF");
        assert_eq!(bytes[4], 0x01);
        assert_eq!(&bytes[5..8], &[0u8; 3]);
    }

    #[test]
    fn validate_header_accepts_valid_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        write_header(&mut file).unwrap();
        validate_header(&mut file).unwrap();
    }

    #[test]
    fn validate_header_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"XXXX\x01\x00\x00\x00").unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("bad magic"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_rejects_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"MKMF\xFF\x00\x00\x00").unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        if let LsmError::Format(msg) = err {
            assert!(msg.contains("unsupported manifest version"), "got: {msg}");
        }
    }

    #[test]
    fn validate_header_rejects_file_too_short() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.log");
        fs::write(&path, b"MKMF").unwrap(); // only 4 bytes
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let err = validate_header(&mut file).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
    }
```

- [ ] **Step 2: Run tests to verify they fail (compilation error expected)**

Run: `cargo test -p minkowski-lsm --lib manifest_log::tests::write_header_emits_expected_bytes`
Expected: FAIL — `write_header` / `validate_header` / constants don't exist yet.

- [ ] **Step 3: Add constants and helper functions**

Near the top of `crates/minkowski-lsm/src/manifest_log.rs`, after the imports and before the existing `ManifestEntry` enum, add:

```rust
// ── File header ─────────────────────────────────────────────────────────────

/// 4-byte magic: "M", "K", "M", "F" — Minkowski Manifest.
const MAGIC_BYTES: [u8; 4] = *b"MKMF";

/// Current manifest log format version.
const CURRENT_VERSION: u8 = 0x01;

/// Total header size in bytes: 4 magic + 1 version + 3 reserved.
const HEADER_SIZE: u64 = 8;

/// Write the manifest log header at offset 0.
///
/// Layout: `[magic: 4][version: 1][reserved: 3]`. Reserved bytes are
/// written as zero; they are ignored on read but reserved for future
/// flags/hints.
fn write_header(file: &mut File) -> Result<(), LsmError> {
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&MAGIC_BYTES)?;
    file.write_all(&[CURRENT_VERSION])?;
    file.write_all(&[0u8; 3])?;
    Ok(())
}

/// Read and validate the manifest log header.
///
/// Returns `LsmError::Format` with a descriptive message on:
/// - File shorter than 8 bytes
/// - Magic bytes don't match `MKMF`
/// - Version byte doesn't match `CURRENT_VERSION`
///
/// Reserved bytes are not validated (forward-compat for future flags).
fn validate_header(file: &mut File) -> Result<(), LsmError> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; 8];
    match file.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(LsmError::Format(
                "not a manifest log: file too short for header".to_owned(),
            ));
        }
        Err(e) => return Err(LsmError::Io(e)),
    }
    if header[0..4] != MAGIC_BYTES {
        return Err(LsmError::Format(
            "not a manifest log: bad magic".to_owned(),
        ));
    }
    let version = header[4];
    if version != CURRENT_VERSION {
        return Err(LsmError::Format(format!(
            "unsupported manifest version {version}"
        )));
    }
    Ok(())
}
```

- [ ] **Step 4: Run the new tests**

Run: `cargo test -p minkowski-lsm --lib manifest_log::tests -- write_header validate_header`
Expected: 5 tests pass.

- [ ] **Step 5: Run full crate test suite to confirm no regressions**

Run: `cargo test -p minkowski-lsm`
Expected: all existing tests pass, plus the 5 new header tests.

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski-lsm/src/manifest_log.rs
git commit -m "feat(lsm): add manifest log header encoding helpers

8-byte header at offset 0: [magic: b\"MKMF\"; 4][version: u8][reserved: 0u8; 3].
Adds write_header() and validate_header() with full test coverage. No
behavior change yet — helpers are wired in by Task 2's recover() function."
```

If fmt hook modifies files, re-stage and re-commit (no amend).

---

## Task 2: Extract `replay_frames` helper + add `recover()` (TDD)

**Goal:** Extract the existing replay loop into a helper that takes a starting offset. Add `recover()` as the new public API, using the header helpers from Task 1. Old `create()`/`open_or_create()`/`replay()` stay public for this commit; Task 3 migrates callers and removes them.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`

- [ ] **Step 1: Write failing unit tests for `recover()`**

Add to the test module in `manifest_log.rs`:

```rust
    #[test]
    fn recover_creates_file_with_header_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.log");
        assert!(!path.exists());

        let (manifest, _log) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(manifest.next_sequence(), SeqNo(0));

        assert!(path.exists());
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], b"MKMF");
        assert_eq!(bytes[4], 0x01);
    }

    #[test]
    fn recover_accepts_valid_header_with_no_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.log");
        // Pre-create with just a header.
        {
            let mut file = File::create(&path).unwrap();
            write_header(&mut file).unwrap();
            file.sync_all().unwrap();
        }
        let (manifest, log) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.total_runs(), 0);
        assert_eq!(log.write_pos, 8);
    }

    #[test]
    fn recover_rejects_file_with_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.log");
        fs::write(&path, b"XXXXv1\x00\x00\x00").unwrap();
        let err = ManifestLog::recover(&path).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn recover_rejects_file_with_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v99.log");
        fs::write(&path, b"MKMF\x63\x00\x00\x00").unwrap();
        let err = ManifestLog::recover(&path).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn recover_replays_existing_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("populated.log");

        // Write a header + one entry using the current API.
        {
            let mut file = File::create(&path).unwrap();
            write_header(&mut file).unwrap();
            file.sync_all().unwrap();
        }

        // Reopen via recover, append, reopen again.
        let (_, mut log) = ManifestLog::recover(&path).unwrap();
        log.append(&ManifestEntry::SetSequence {
            next_sequence: SeqNo(42),
        })
        .unwrap();
        drop(log);

        let (manifest, _log) = ManifestLog::recover(&path).unwrap();
        assert_eq!(manifest.next_sequence(), SeqNo(42));
    }
```

Imports (add at the top of the file or the test module, as appropriate):

```rust
    use crate::types::SeqNo;
```

(`SeqNo` is already imported at the module level for the codec, so add to the test module if it's not already in scope.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-lsm --lib manifest_log::tests -- recover`
Expected: FAIL — `ManifestLog::recover` doesn't exist.

- [ ] **Step 3: Extract `replay_frames` from the existing `replay` function**

Replace the current `pub fn replay(path: &Path) -> Result<LsmManifest, LsmError>` body. Add a private helper that takes a starting offset and returns both the manifest and the final position:

```rust
/// Replay the frame sequence starting at `start` in the given file.
/// Truncates on torn-tail / decode / apply errors, as the existing
/// recovery contract requires. Returns the recovered manifest and the
/// post-truncation position (end of the valid frame region).
fn replay_frames(
    file: &File,
    path: &Path,
    start: u64,
) -> Result<(LsmManifest, u64), LsmError> {
    let mut manifest = LsmManifest::new();
    let mut pos: u64 = start;

    loop {
        let (payload, next_pos) = match read_frame(file, pos) {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(LsmError::Crc { .. } | LsmError::Format(_)) => {
                truncate_at(path, pos)?;
                break;
            }
            Err(e) => return Err(e),
        };

        let Ok(entry) = decode_entry(&payload) else {
            truncate_at(path, pos)?;
            break;
        };
        if apply_entry(&mut manifest, &entry).is_err() {
            truncate_at(path, pos)?;
            break;
        }
        pos = next_pos;
    }

    Ok((manifest, pos))
}
```

Update the existing public `replay` function to use this helper (keeps behavior identical for current callers; Task 3 removes this public version):

```rust
    /// Replay the log to reconstruct a manifest.
    ///
    /// Tolerates corrupt tail frames. Returns an empty manifest if the
    /// file doesn't exist.
    ///
    /// Note: this entry point assumes the file has no header (pre-PR-B1
    /// format). New code should use `recover()` which handles the
    /// v1-header format.
    pub fn replay(path: &Path) -> Result<LsmManifest, LsmError> {
        if !path.exists() {
            return Ok(LsmManifest::new());
        }
        let file = File::open(path)?;
        let (manifest, _) = replay_frames(&file, path, 0)?;
        Ok(manifest)
    }
```

- [ ] **Step 4: Add `recover()` implementation**

Add to the `impl ManifestLog` block:

```rust
    /// Load an existing manifest log or initialize a new empty one.
    ///
    /// If `path` does not exist: creates it, writes the header, fsyncs.
    /// Returns `(LsmManifest::new(), log_handle)` ready to append.
    ///
    /// If `path` exists: reads the 8-byte header and validates magic +
    /// version (rejecting unknown formats with `LsmError::Format`),
    /// replays frames from offset 8 onward (truncating torn tails), and
    /// returns `(recovered_manifest, log_handle)` with `write_pos` at
    /// end of valid data.
    pub fn recover(path: &Path) -> Result<(LsmManifest, Self), LsmError> {
        if !path.exists() {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(false)
                .open(path)?;
            write_header(&mut file)?;
            file.sync_all()?;
            return Ok((
                LsmManifest::new(),
                Self {
                    file,
                    write_pos: HEADER_SIZE,
                },
            ));
        }

        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(path)?;
        validate_header(&mut file)?;
        let (manifest, write_pos) = replay_frames(&file, path, HEADER_SIZE)?;
        Ok((manifest, Self { file, write_pos }))
    }
```

- [ ] **Step 5: Run the new tests**

Run: `cargo test -p minkowski-lsm --lib manifest_log::tests -- recover`
Expected: 5 tests pass.

- [ ] **Step 6: Run full crate test suite**

Run: `cargo test -p minkowski-lsm`
Expected: all existing tests still pass (they use old API), plus the 5 new `recover_*` tests.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski-lsm/src/manifest_log.rs
git commit -m "feat(lsm): add ManifestLog::recover as unified entry point

recover(path) subsumes create/open_or_create/replay into one API:
missing file -> create + write header, existing file -> validate
header + replay frames + return log handle positioned at EOF.

Extracts the replay loop into replay_frames(file, path, start) so
recover() can start reading frames at offset 8 (after header). The
old public replay(path) still exists, calling replay_frames(0, ...)
for compatibility with existing callers; Task 3 migrates them and
removes both replay and open_or_create."
```

---

## Task 3: Migrate callers + byte-prefix test + remove old API

**Goal:** Migrate every caller of `create`/`open_or_create`/`replay` to use `recover()` (or `create` via `pub(crate)` where needed). Remove `open_or_create` and the public `replay`. Update the byte-prefix convergence test to account for the 8-byte header offset.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_log.rs` (API deletions + visibility change)
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs` (test call sites)
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs` (all test call sites + prefix test)

- [ ] **Step 1: Migrate unit tests in `manifest_log.rs`**

Find every `ManifestLog::create(&path)` or `ManifestLog::open_or_create(&path)` call in the `#[cfg(test)] mod tests` block. Replace with `recover` or keep using `create` (which becomes `pub(crate)` in Step 3).

Pattern: a test that writes then replays:

```rust
        let mut log = ManifestLog::create(&path).unwrap();
        log.append(&entry).unwrap();
        let manifest = ManifestLog::replay(&path).unwrap();
```

Becomes:

```rust
        let (_, mut log) = ManifestLog::recover(&path).unwrap();
        log.append(&entry).unwrap();
        drop(log);
        let (manifest, _) = ManifestLog::recover(&path).unwrap();
```

Specific tests to update (based on existing test names after PR A):
- `replay_empty_file` — replaces `ManifestLog::replay(&path)` with `ManifestLog::recover(&path)` returning a tuple.
- `replay_three_add_runs` — use `recover` for both the setup and the final replay.
- `replay_add_then_remove` — same.
- `replay_tolerates_torn_tail` — same; the "append garbage" step still works since it's just writing past the last valid frame.

Run: `cargo check -p minkowski-lsm --tests 2>&1 | head -30` to enumerate remaining call sites after each edit.

- [ ] **Step 2: Migrate unit tests in `manifest_ops.rs`**

In the `#[cfg(test)] mod tests` block of `manifest_ops.rs`:

`flush_and_record_dirty_world` and `flush_and_record_clean_world` both have:

```rust
        let mut log = ManifestLog::create(&log_path).unwrap();
```

Replace with:

```rust
        let (_, mut log) = ManifestLog::recover(&log_path).unwrap();
```

(Ignores the empty manifest returned — the test constructs its own via `LsmManifest::new()`.)

Actually: those tests have `let mut manifest = LsmManifest::new();` on the previous line, so after migration:

```rust
        let mut manifest = LsmManifest::new();
        let (_, mut log) = ManifestLog::recover(&log_path).unwrap();
```

The `_` is redundant with `let mut manifest = LsmManifest::new()`. Alternative: use the returned manifest:

```rust
        let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();
```

Go with the second form — uses what `recover` returns, matches production usage.

- [ ] **Step 3: Migrate `tests/manifest_integration.rs`**

Every integration test currently has a pattern like:

```rust
    let mut manifest = LsmManifest::new();
    let mut log = ManifestLog::create(&log_path).unwrap();
```

Replace with:

```rust
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();
```

And every `ManifestLog::replay(&log_path).unwrap()` becomes:

```rust
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
```

(Keep the variable name `recovered` for readability where it's used.)

Tests to update: `three_flushes_then_replay`, `corrupt_tail_partial_recovery`, `replay_converges_at_every_truncation_prefix`, `replay_recovers_from_torn_set_sequence_after_add_run`, `replay_truncates_log_on_promote_of_missing_run`, `cleanup_removes_orphans_and_tmp`, `flush_and_record_clean_world_no_change`, `replay_truncates_log_on_unsorted_coverage`, `replay_truncates_log_on_invalid_level_byte`, `replay_truncates_log_on_inverted_seq_range`.

- [ ] **Step 4: Adjust the byte-prefix convergence test**

In `tests/manifest_integration.rs`, find `replay_converges_at_every_truncation_prefix`. After migrating to `recover`, the test logic needs updating because prefixes shorter than 8 bytes now return `Err` (truncated header) instead of `Ok(empty_manifest)`.

Current loop (post-migration placeholder):

```rust
    for truncate_len in 0..=full_bytes.len() {
        let truncated_path = dir.path().join(format!("truncated_{truncate_len:05}.log"));
        fs::write(&truncated_path, &full_bytes[..truncate_len]).unwrap();

        let (replayed, _) = ManifestLog::recover(&truncated_path)
            .unwrap_or_else(|e| panic!("replay failed at truncate_len={truncate_len}: {e:?}"));

        // ... assertions ...
    }
```

New loop shape: prefixes `< 8` return `Err` (bad / short header); prefixes `>= 8` return `Ok`:

```rust
    let mut prev_total_runs = 0usize;
    let mut prev_next_seq = SeqNo(0);

    for truncate_len in 0..=full_bytes.len() {
        let truncated_path = dir.path().join(format!("truncated_{truncate_len:05}.log"));
        fs::write(&truncated_path, &full_bytes[..truncate_len]).unwrap();

        if truncate_len < 8 {
            // Header missing or truncated: recover must return a
            // Format error; no manifest is produced.
            let err = ManifestLog::recover(&truncated_path).unwrap_err();
            assert!(
                matches!(err, LsmError::Format(_)),
                "truncate_len={truncate_len}: expected Format, got {err:?}"
            );
            continue;
        }

        let (replayed, _) = ManifestLog::recover(&truncated_path)
            .unwrap_or_else(|e| panic!("recover failed at truncate_len={truncate_len}: {e:?}"));

        assert!(
            replayed.total_runs() <= 3,
            "truncate_len={truncate_len}: total_runs={} > 3",
            replayed.total_runs()
        );

        // Monotonicity: extending the prefix never loses state.
        assert!(
            replayed.total_runs() >= prev_total_runs,
            "truncate_len={truncate_len}: total_runs rewound {prev_total_runs} -> {}",
            replayed.total_runs()
        );
        assert!(
            replayed.next_sequence() >= prev_next_seq,
            "truncate_len={truncate_len}: next_sequence rewound {prev_next_seq:?} -> {:?}",
            replayed.next_sequence()
        );
        prev_total_runs = replayed.total_runs();
        prev_next_seq = replayed.next_sequence();
    }

    assert_eq!(prev_total_runs, 3, "full replay must recover all 3 runs");
    assert_eq!(prev_next_seq, SeqNo(30), "full replay must recover final sequence");
```

Add import at top of `manifest_integration.rs` if missing:

```rust
use minkowski_lsm::error::LsmError;
use minkowski_lsm::types::SeqNo;
```

(`SeqNo` is probably already imported from PR A. Verify.)

- [ ] **Step 5: Remove `open_or_create` + public `replay`; make `create` pub(crate)**

In `crates/minkowski-lsm/src/manifest_log.rs`:

Delete the entire `pub fn open_or_create(path: &Path) -> Result<Self, LsmError>` function.

Delete the entire `pub fn replay(path: &Path) -> Result<LsmManifest, LsmError>` function.

Change `pub fn create` visibility to `pub(crate)`:

```rust
    /// Test-only: create a fresh log, truncating any existing file and
    /// writing a valid header.
    pub(crate) fn create(path: &Path) -> Result<Self, LsmError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(path)?;
        write_header(&mut file)?;
        file.sync_all()?;
        Ok(Self {
            file,
            write_pos: HEADER_SIZE,
        })
    }
```

(Note: `create` must also write a header now — it didn't before. Any test that was calling `create` expected an empty file; with the header, the file now has 8 bytes. Tests that then call `recover` will pick up where `create` left off.)

- [ ] **Step 6: Run `cargo check`**

Run: `cargo check -p minkowski-lsm`
Expected: clean compile. Any remaining error is a call site that wasn't migrated.

- [ ] **Step 7: Run full test suite**

Run: `cargo test -p minkowski-lsm`
Expected: all tests pass. Total test count stays at 85 for now (new tests from Task 2 are in, old tests migrated in place).

- [ ] **Step 8: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add crates/minkowski-lsm/src/manifest_log.rs \
        crates/minkowski-lsm/src/manifest_ops.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "refactor(lsm): migrate callers to recover(); remove old entry points

Every caller of ManifestLog::{create, open_or_create, replay} migrated
to use recover() as the sole public entry. open_or_create and the
public replay are deleted entirely — the former was the footgun this
PR exists to close, the latter folds into recover.

create() remains as pub(crate) for test-only 'truncate + start fresh'
semantics; it now writes the 8-byte header as part of initialization.

Byte-prefix convergence test updated: prefixes 0..=7 now return
LsmError::Format (truncated or missing header) instead of Ok(empty
manifest). Assertions split accordingly."
```

---

## Task 4: `RemoveRun` propagation fix + regression test

**Goal:** Fix the pre-existing silent no-op where `apply_entry` discards a `None` from `manifest.remove_run`. Add a regression test.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs`

- [ ] **Step 1: Write failing regression test**

Add to `tests/manifest_integration.rs`:

```rust
/// Regression: a RemoveRun frame referencing a path the manifest doesn't
/// know is log corruption. apply_entry must propagate the error so replay
/// treats the rest of the log as tail garbage — same policy as PromoteRun.
#[test]
fn replay_truncates_log_on_remove_of_missing_run() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();

    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    // One real flush — produces a valid AddRunAndSequence entry.
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();

    // Inject a RemoveRun referencing a path the manifest doesn't know.
    log.append(&ManifestEntry::RemoveRun {
        level: Level::L0,
        path: PathBuf::from("ghost.run"),
    })
    .unwrap();
    // Anything after the bad entry must be discarded on replay.
    log.append(&ManifestEntry::SetSequence {
        next_sequence: SeqNo(999),
    })
    .unwrap();
    drop(log);

    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(
        recovered.total_runs(),
        1,
        "only the valid first flush should survive"
    );
    // The trailing SetSequence must not have been applied.
    assert!(
        recovered.next_sequence() < SeqNo(999),
        "SetSequence past the bad RemoveRun must not apply"
    );
    assert_eq!(recovered.next_sequence(), SeqNo(10));
}
```

- [ ] **Step 2: Run the test — expect it to fail with manifest showing 0 runs**

Run: `cargo test -p minkowski-lsm --test manifest_integration replay_truncates_log_on_remove_of_missing_run`
Expected: FAIL — current `apply_entry` silently no-ops on the bad RemoveRun, then applies the SetSequence, so `total_runs == 1` (pass) but `next_sequence == 999` (fail).

Actually the bigger issue: the silent no-op means the SetSequence(999) DOES get applied. So the test's `next_sequence() == 10` assertion fails. That's the exact gap this task closes.

- [ ] **Step 3: Fix `apply_entry` in `manifest_log.rs`**

Find the `RemoveRun` arm:

```rust
        ManifestEntry::RemoveRun { level, path } => {
            manifest.remove_run(*level, path);
        }
```

Replace with:

```rust
        ManifestEntry::RemoveRun { level, path } => {
            // A RemoveRun for a path the manifest doesn't know means log
            // corruption — the corresponding AddRun was lost, or entries
            // are out of order. Propagate so replay treats the rest as
            // tail garbage. Same policy as PromoteRun above.
            if manifest.remove_run(*level, path).is_none() {
                return Err(LsmError::Format(format!(
                    "RemoveRun: run {} not found at level {}",
                    path.display(),
                    level
                )));
            }
        }
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p minkowski-lsm --test manifest_integration replay_truncates_log_on_remove_of_missing_run`
Expected: PASS.

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `cargo test -p minkowski-lsm`
Expected: all tests pass, plus the 1 new test = 86 total.

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski-lsm/src/manifest_log.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "fix(lsm): RemoveRun propagates missing-run as LsmError::Format

apply_entry for RemoveRun was discarding the Option<SortedRunMeta>
return from manifest.remove_run, silently no-opping on a log that
references a missing run. Asymmetric with PromoteRun in the same
function, which correctly propagates via ? (per PR A's review-driven
fix).

Fix: mirror PromoteRun. Return LsmError::Format on a None, so replay's
error-handling arm truncates the log. Regression test injects a
RemoveRun-of-missing between a valid flush and a trailing SetSequence,
and asserts the SetSequence is discarded (would have been applied
before this fix)."
```

---

## Task 5: Additional integration regression tests

**Goal:** Cover the three remaining integration scenarios called out in the spec's testing strategy.

**Files:**
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs`

- [ ] **Step 1: Write the three new tests**

Add to `tests/manifest_integration.rs`:

```rust
/// A recover -> flush -> recover round trip reconstructs identical state.
/// Exercises the full lifecycle through the new unified entry point.
#[test]
fn recover_then_flush_then_recover_roundtrips_state() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("manifest.log");

    // Fresh recover creates the file.
    let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(manifest.total_runs(), 0);

    // Flush produces one AddRunAndSequence frame.
    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 0.0 },));
    flush_and_record(&world, (0, 10), &mut manifest, &mut log, dir.path()).unwrap();
    world.clear_all_dirty_pages();
    world.spawn((Pos { x: 2.0, y: 0.0 },));
    flush_and_record(&world, (10, 20), &mut manifest, &mut log, dir.path()).unwrap();
    drop(log);

    // Second recover replays both entries.
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 2);
    assert_eq!(recovered.next_sequence(), SeqNo(20));

    // Metadata round-trips faithfully.
    for (orig, rec) in manifest
        .runs_at_level(Level::L0)
        .iter()
        .zip(recovered.runs_at_level(Level::L0).iter())
    {
        assert_eq!(orig.path(), rec.path());
        assert_eq!(orig.sequence_range(), rec.sequence_range());
    }
}

/// Opening a file that wasn't produced by PR-B1-or-later code (no 8-byte
/// header at offset 0) must fail fast with a Format error. Explicitly
/// documents the strict-reject compatibility decision.
#[test]
fn recover_rejects_file_without_header() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("legacy.log");

    // Write raw bytes starting with what looks like a frame length prefix
    // (not a header). This is what a pre-PR-B1 manifest log would look
    // like byte-for-byte.
    fs::write(&log_path, &[0x20, 0x00, 0x00, 0x00, 0xAB, 0xCD, 0xEF, 0x12]).unwrap();

    let err = ManifestLog::recover(&log_path).unwrap_err();
    assert!(
        matches!(err, LsmError::Format(ref msg) if msg.contains("bad magic")),
        "expected bad-magic Format error, got {err:?}"
    );
}

/// A file with valid magic but an unrecognized version byte must be
/// rejected (forward-compat gate: an older binary reading a newer file
/// must fail loudly, not silently decode garbage).
#[test]
fn recover_rejects_file_with_unsupported_version() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("future.log");

    fs::write(&log_path, b"MKMF\x63\x00\x00\x00").unwrap(); // version 0x63

    let err = ManifestLog::recover(&log_path).unwrap_err();
    assert!(
        matches!(err, LsmError::Format(ref msg) if msg.contains("unsupported manifest version")),
        "expected version-mismatch Format error, got {err:?}"
    );
}
```

Verify imports at top of file include:

```rust
use minkowski_lsm::error::LsmError;
use minkowski_lsm::types::{Level, SeqNo};
```

(`Level` from PR A migration, `SeqNo` same. `LsmError` may need adding.)

- [ ] **Step 2: Run the new tests**

Run: `cargo test -p minkowski-lsm --test manifest_integration -- recover_`
Expected: three tests pass.

- [ ] **Step 3: Run full suite**

Run: `cargo test -p minkowski-lsm`
Expected: all pass. Total should now be 89 (86 + 3).

- [ ] **Step 4: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "test(lsm): recover() lifecycle and rejection regressions

Three integration tests covering recover()'s new behavior:

- recover_then_flush_then_recover_roundtrips_state: full lifecycle
  through the unified entry; two flushes survive a close/reopen cycle
  with identical metadata.
- recover_rejects_file_without_header: pre-PR-B1-format files (no
  8-byte header) fail with LsmError::Format containing 'bad magic'.
  Documents the strict-reject compatibility decision explicitly.
- recover_rejects_file_with_unsupported_version: valid magic + unknown
  version byte fails with a version-mismatch Format error. Guards
  against silent decode of future-format files by older binaries."
```

---

## Task 6: Final verification + push + PR

**No code changes. Green-light gate before opening the PR.**

- [ ] **Step 1: Update local toolchain to match CI (same drill as PR A)**

Run: `rustup update stable`
Expected: toolchain updates or no-op if already current.

- [ ] **Step 2: Run workspace clippy (CI-equivalent)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

If a newer lint fires that wasn't caught by `-p minkowski-lsm`, fix the offending code in the same file where it lives (e.g., a new `sort_by`/`map_unwrap_or` lint could fire in a cross-crate file; follow the PR A pattern — one tiny chore commit per lint, merge into the PR).

- [ ] **Step 3: Run workspace test suite**

Run: `cargo test --workspace`
Expected: the 3 pre-existing `minkowski-observe` failures (`prometheus::tests::new_renders_default_zeros`, `diff::tests::diff_no_pool_omits_delta`, `snapshot::tests::snapshot_without_pool_omits_pool_line`) will still fail — they're pre-existing on main and CI doesn't run them (`cargo test -p minkowski` only). All other tests pass, including the 89 in `minkowski-lsm`.

- [ ] **Step 4: Run cargo fmt check**

Run: `cargo fmt --all -- --check`
Expected: clean.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin lsm/pr-b-format-hardening
gh pr create --title "feat(lsm): manifest format hardening (PR B1)" --body "$(cat <<'EOF'
## Summary

First half of the PR B follow-up from PR A's review. Adds an 8-byte
magic + version header at offset 0 of the manifest log, introduces
`ManifestLog::recover()` as the sole public entry point (closing the
`open_or_create` footgun), and fixes a pre-existing silent no-op in
`apply_entry` for `RemoveRun`.

Spec: `docs/plans/2026-04-17-lsm-manifest-format-hardening-design.md`
Plan: `docs/plans/2026-04-17-lsm-manifest-format-hardening-implementation-plan.md`

## What landed

- **8-byte header** at offset 0: `[magic: b"MKMF"; 4][version: u8; 1][reserved: 0u8; 3]`. Frames now start at byte 8. Version 0x01 is the initial format; unknown versions and bad magic are rejected at open with `LsmError::Format`.
- **`recover(path) -> Result<(LsmManifest, ManifestLog)>`** — one door. Missing path → create + header; existing path → validate header + replay from offset 8.
- **Removed**: `open_or_create` (the footgun) and public `replay`. `create` becomes `pub(crate)` for test use.
- **`RemoveRun` apply propagates** missing-run as `LsmError::Format`, same as `PromoteRun`. Replay's existing error-handling arm truncates on the bad entry.

## Breaking change

Pre-PR-B1 manifest log files (no header) are rejected at open. Recovery: `rm manifest.log` and let WAL replay rebuild. Justified in the spec's Motivation section — LSM shipped in v1.3.0 days before this PR; no production logs to migrate.

## Tests

89 total in `minkowski-lsm` (was 85 at start of PR B1):
- 5 new header encode/decode unit tests
- 5 new `recover()` unit tests (missing file, valid header, bad magic, wrong version, replay existing entries)
- 1 regression for `RemoveRun`-of-missing truncation
- 3 integration tests for the recover() lifecycle and rejection paths
- Existing byte-prefix convergence test adjusted for the 8-byte header offset

## Test plan

- [x] `cargo test -p minkowski-lsm` — 89/89 pass
- [x] `cargo clippy --workspace --all-targets -- -D warnings` — clean
- [x] `cargo fmt --all -- --check` — clean
- [ ] CI pipeline (fmt, clippy, test, tsan, loom, claude-review)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Monitor CI; update memory after merge**

Once the PR merges, update the memory files:
- `project_scaling_roadmap.md` — note PR B1 landed.
- `project_lsm_phase2_type_safety.md` — trim items 1-3 (primary); PR B2 scope is now items 4-10 (secondary).

---

## Self-review (done inline before saving)

- **Spec coverage:**
  - Section 1 (header layout) → Task 1
  - Section 2 (API surface) → Task 2 adds `recover`; Task 3 removes `open_or_create`/public `replay`, makes `create` pub(crate)
  - Section 3 (recover flow) → Task 2
  - Section 4 (RemoveRun fix) → Task 4
  - Section 5 (compat: strict reject) → Task 5's `recover_rejects_file_without_header` test documents the behavior explicitly
  - Section 6 (call-site migration) → Task 3
  - Section 7 (error-type reuse) → enforced by using `LsmError::Format` in every new error path
  - Testing strategy → Tasks 1, 2, 4, 5 each produce the tests listed in the spec

- **Placeholder scan:** None. Every code block is full, every command has an expected outcome.

- **Type consistency:**
  - `HEADER_SIZE: u64 = 8` used consistently across Task 1 (defined), Task 2 (`write_pos: HEADER_SIZE`), and Task 3 (prefix-test boundary).
  - `LsmError::Format(String)` used in every new error path — bad magic, wrong version, truncated header, missing RemoveRun target.
  - `recover()` signature `Result<(LsmManifest, Self), LsmError>` in Task 2, consistent with usage in Tasks 3-5.
  - `create()` signature stays `Result<Self, LsmError>` in Task 3's `pub(crate)` version; callers updated consistently.

No issues found. Ready to hand off to implementation.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-04-17-lsm-manifest-format-hardening-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Good for a 6-task plan with one TDD-heavy task and mechanical migrations.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
