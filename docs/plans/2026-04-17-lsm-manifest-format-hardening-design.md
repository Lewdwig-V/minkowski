# LSM Manifest Format Hardening (PR B1)

*Parent document: [Stage 3: LSM Tree Storage](./2026-04-03-stage3-lsm-implementation-plan.md), follow-up to Phase 2 + PR A.*
*Status: Design. Prerequisites: PR A (manifest type-safety, squash `b0927e8`) merged.*

---

## Summary

This PR hardens the `minkowski-lsm` manifest log by adding a magic + version header at the start of the file, introducing `ManifestLog::recover()` as the sole public entry point (eliminating the `open_or_create()` footgun), and fixing a silent no-op in `apply_entry` where `RemoveRun` for a missing path was being dropped.

Scope is intentionally tight: no parent-directory fsync (the original review item conflated rename atomicity with bare-creation durability — Linux's fsync-parent-inclusion covers the latter), no wire-format changes to frame entries, no API changes outside `ManifestLog` itself.

Existing `manifest.log` files written pre-PR-B1 will be rejected at open. Recovery path: `rm manifest.log`; the WAL replays and rebuilds state. Justified by the LSM feature having shipped in v1.3.0 only days before this PR — no production logs exist to migrate.

## Motivation

PR A's type-design reviewer flagged `ManifestLog` as the weakest remaining type in the manifest subsystem (baseline rating: encap 8, invariant expression 5, usefulness 7, enforcement 5). Two structural holes:

1. **`open_or_create()` lets callers append on a torn tail.** `open_or_create` returns a log handle without running replay. A caller who skips replay and appends writes new data past corrupt bytes; the next recovery then truncates away the new data as tail garbage. The type system provides no guardrail.

2. **No format versioning.** The manifest log has no magic bytes, no version byte — just raw CRC-framed entries from offset 0. Any future entry-layout change (new tag, new field in an existing variant) is undetectable. Old binaries on new format would decode garbage until a CRC failure, then truncate the entire file.

Additionally, the silent-failure review caught a pre-existing (from PR #162) bug in `apply_entry`:

```rust
ManifestEntry::RemoveRun { level, path } => {
    manifest.remove_run(*level, path);   // Option return discarded
}
```

`PromoteRun` correctly propagates a missing-run condition via `?`. `RemoveRun` doesn't, silently no-opping on a log with an internally inconsistent remove. Asymmetric and wrong.

## Non-goals

- **Parent-directory fsync**: dropped from scope. The original review item was "parent-dir fsync after `truncate_at`", but `truncate_at` only changes inode metadata on a file with an already-durable dirent — `fsync(file)` covers it. Bare file creation (via `recover()` on a missing path) has a theoretical dirent-durability gap, but on Linux with ext4 `data=ordered` (default), `fsync(file)` implicitly covers the parent dir through journal-commit coupling. The fsync+rename+dir-fsync combo belongs where atomic replacement is the mechanism (`writer.rs::flush` already has it). For bare creation with no rename, add dir fsync only if a concrete field issue surfaces.
- **Wire format changes to `ManifestEntry`**: unchanged. Only the file-level header is new.
- **PR B2 scope**: deferred to a follow-up PR. See `project_lsm_phase2_type_safety.md` in memory.

## Design

### 1. Header layout at offset 0

```
Byte 0-3: magic = b"MKMF"      ("M", "K", "M", "F" — Minkowski Manifest)
Byte 4:   version = 0x01       (u8; current format version)
Byte 5-7: reserved = [0u8; 3]  (future flags/hints; must be zero today)
```

Total: 8 bytes. No CRC: magic + all-zero reserved bytes gives ~40 bits of structural validity (1 in ~2^40 chance of a random file passing both checks). Adding a 4-byte header CRC would double the header size for marginal benefit.

Frames start at offset 8. The existing `[len: u32 LE][crc32: u32 LE][payload]` frame format is unchanged.

### 2. `ManifestLog` API surface

```rust
impl ManifestLog {
    /// Load an existing manifest log or initialize a new empty one.
    ///
    /// If `path` does not exist: creates the file, writes the header,
    /// fsyncs, and returns `(LsmManifest::new(), log_handle)` ready to
    /// append.
    ///
    /// If `path` exists: reads and validates the header (rejects unknown
    /// magic or version via `LsmError::Format`), replays frames from
    /// offset 8 onward (truncating torn tails, as before), and returns
    /// `(recovered_manifest, log_handle)` with `write_pos` at EOF.
    pub fn recover(path: &Path) -> Result<(LsmManifest, Self), LsmError>;

    /// Append an entry, fsyncing for durability.
    pub fn append(&mut self, entry: &ManifestEntry) -> Result<(), LsmError>;

    /// Explicit fsync.
    pub fn sync(&mut self) -> Result<(), LsmError>;

    // Note: an earlier draft of this design retained `pub(crate) fn create`
    // as a test-only "wipe and start fresh" helper. During implementation
    // (Task 3) it was found to have zero callers — every existing test that
    // used create() did so on a freshly-created tempdir where recover()'s
    // missing-path branch produces identical behavior. Removed entirely
    // instead of kept as dead code. Tests that need truncation semantics
    // can use fs::remove_file(path).ok(); recover(path).
}
```

**Removed entirely:** `open_or_create()` (the footgun — removed, not even `pub(crate)`) and `replay(path)` (folded into `recover()`'s existing-file branch via the `replay_frames` private helper). Also removed: `create()` — all tests migrated to `recover(path)` on non-existing paths.

### 3. `recover()` implementation sketch

```rust
pub fn recover(path: &Path) -> Result<(LsmManifest, Self), LsmError> {
    if !path.exists() {
        // Fresh install path.
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
            Self { file, write_pos: HEADER_SIZE },
        ));
    }

    // Existing-file path: validate header, then replay.
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .open(path)?;
    validate_header(&mut file)?;  // reads bytes 0..8, checks magic + version
    let mut manifest = LsmManifest::new();
    let write_pos = replay_frames(&file, path, &mut manifest, HEADER_SIZE)?;
    Ok((manifest, Self { file, write_pos }))
}
```

`validate_header` reads exactly 8 bytes, compares against magic + version + reserved, returns `LsmError::Format("not a manifest log: bad magic")` / `LsmError::Format("unsupported manifest version N")` on mismatch. The existing replay loop (from PR A) is extracted into `replay_frames` taking a starting offset, so it reads frames from byte 8 onward instead of byte 0.

### 4. `RemoveRun` propagation fix

```rust
ManifestEntry::RemoveRun { level, path } => {
    // A RemoveRun for a path the manifest doesn't know means log
    // corruption (the corresponding AddRun was lost or ordering is
    // wrong). Propagate so replay treats the rest as tail garbage —
    // same policy as PromoteRun.
    if manifest.remove_run(*level, path).is_none() {
        return Err(LsmError::Format(format!(
            "RemoveRun: run {} not found at level {}",
            path.display(),
            level
        )));
    }
}
```

`apply_entry` already returns `Result<(), LsmError>` from PR A, and replay truncates on apply error. This is a one-location change that hooks into the existing recovery mechanism.

### 5. On-disk format compatibility

Existing `manifest.log` files written by PR A-era code lack the 8-byte header — byte 0 is the first frame's length prefix, not `M`. Calling `recover()` on such a file fails at `validate_header` with `LsmError::Format("not a manifest log: bad magic")`.

**User recovery:** delete the file. The WAL replay on the next boot rebuilds manifest state from scratch.

**Why this is acceptable:** the LSM feature shipped as part of v1.3.0 (tagged days before this PR). No production deployment has accumulated manifest log state yet. Any developer who's been testing the feature can re-initialize. The cost of a migration path — detection heuristic, dual-read replay code, once-only write path — exceeds the benefit for zero current users.

Explicit spec note: this is intentional, not an oversight. Future readers of the code should not be surprised by the lack of an auto-migration path.

### 6. Call-site migration

Internal callers of the old API:

- `crates/minkowski-lsm/src/manifest_ops.rs`: `flush_and_record` takes `&mut ManifestLog` as a parameter; unchanged. Upstream initialization sites (tests, any future `Durable` integration in Phase 5) shift from `ManifestLog::create(...)` or `::open_or_create(...)` to `ManifestLog::recover(...)`.
- `crates/minkowski-lsm/src/manifest_log.rs` tests: use `ManifestLog::create(...)` via `pub(crate)` visibility (still works; it's the "wipe and start fresh" test helper).
- `crates/minkowski-lsm/tests/manifest_integration.rs`: all current tests use `ManifestLog::create` + `ManifestLog::replay` separately. Migrate to `ManifestLog::recover(...)` which returns `(LsmManifest, ManifestLog)`. Test bodies become slightly simpler.

No external callers of `minkowski-lsm` exist today, so the public API change (open_or_create + replay removed) is internal-only.

### 7. Error-type reuse

All new validation failures surface as `LsmError::Format(String)`:
- Bad magic: `"not a manifest log: bad magic"`
- Unsupported version: `"unsupported manifest version N"`
- Missing `RemoveRun` target: `"RemoveRun: run {path} not found at level {level}"`

No new error variants. Matches the existing convention where Format covers both on-disk structural errors and semantic validation errors.

## Testing strategy

### Unit tests (manifest_log.rs)

- Header encode/decode round-trip (write 8 bytes, read 8 bytes, fields match).
- `recover(missing_path)` creates a file that starts with the header.
- `recover(file_with_bad_magic)` returns `LsmError::Format` with "bad magic" substring.
- `recover(file_with_wrong_version)` returns `LsmError::Format` with "unsupported manifest version" substring.
- `apply_entry(&mut manifest, &ManifestEntry::RemoveRun { level, path: unknown })` returns `LsmError::Format`.

### Integration tests (manifest_integration.rs)

Migrate existing tests plus add:
- `recover_then_flush_then_recover_roundtrips_state`: call `recover` on empty path, do a flush, drop the log, `recover` again, assert the manifest matches.
- `recover_rejects_file_without_header`: write a raw frame to offset 0 (no header), call `recover`, expect `LsmError::Format`.
- `recover_rejects_file_with_unsupported_version`: write a header with version byte `0xFF`, call `recover`, expect `LsmError::Format`.
- `replay_truncates_log_on_remove_of_missing_run`: handcraft a `RemoveRun` frame referencing a nonexistent path, append after a good frame, call `recover`, expect 1 run and file truncated.

### Byte-prefix convergence test

The existing `replay_converges_at_every_truncation_prefix` test exercises every byte offset `0..=file_len`. With the new 8-byte header, prefixes 0..=7 should all produce `LsmError::Format` (truncated or absent header), not a valid manifest. Update the test's expectations: for `truncate_len < 8`, `recover` returns an error (not Ok with an empty manifest). Adjust the assertion loop accordingly.

## Rollout

Single squash-merged PR per project convention. CI gates: fmt, clippy, test, tsan, loom, claude-review.

Post-merge, update `project_scaling_roadmap.md` and `project_lsm_phase2_type_safety.md` memory entries to reflect PR B1 landed and PR B2 is the remaining scope.

## Risks

- **Callers in hypothetical branches or local work**: anyone with an in-progress branch using `ManifestLog::create` / `open_or_create` / `replay` directly will see a compile error after this lands. The migration is mechanical (use `recover()`), but it's a breaking API change. Worth calling out in the PR body.
- **The byte-prefix convergence test adjustment** is the trickiest test change — the test currently expects every prefix to produce `Ok(manifest)`. New behavior: prefixes 0..=7 return `Err(Format)`. Get the boundary right or the test silently stops exercising its intent.
- **Header reserved bytes policy**: the spec says "must be zero today." If a future PR adds flags, existing v1 files already in the wild would have zeros in those bytes (meaning "no flags set"), which works. But if a future version ever writes non-zero reserved bytes with v1 magic+version, old binaries would accept them as valid v1 — they'd ignore the flag. Document the policy as "reserved bytes must be zero when writing at a given version; readers should not silently accept non-zero reserved bytes at a known version." If we care about forward-compat in reserved, a reader check can be added in PR B2.
