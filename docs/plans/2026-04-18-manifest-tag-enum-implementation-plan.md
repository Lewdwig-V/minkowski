# ManifestTag Enum — Phase 3 Pre-Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `manifest_log.rs`'s five raw-u8 `TAG_*` constants with a single `#[repr(u8)] pub enum ManifestTag`. Decode via `TryFrom<u8>` so the match becomes exhaustive, removing the `_ => Err(LsmError::Format(...))` catchall.

**Architecture:** One file-local refactor in `manifest_log.rs`. The `#[repr(u8)]` guarantees byte-level layout compatibility with the existing wire format — encode casts via `as u8`, decode parses via `TryFrom<u8>` which returns `LsmError::Format` on unknown bytes. Exhaustive match over the new enum removes the wildcard arm; any future tag addition becomes a compile error at every decode site, not a runtime "unknown tag" surprise.

**Tech Stack:** Rust 2024 edition, `minkowski-lsm` workspace crate, existing `LsmError` type.

**Spec:** This plan — scope is narrow enough (~80 lines across 2 files) that no separate design doc is warranted.

---

## Starting state

- Branch: `lsm/phase3-prep-enum-tags` already created off `origin/main` (post-PR-B3 squash `22c4486`).
- 114 tests currently passing in `minkowski-lsm`.

## File structure

**Modify:**
- `crates/minkowski-lsm/src/manifest_log.rs` — replace `pub const TAG_*: u8 = 0x0N` constants with `#[repr(u8)] pub enum ManifestTag`. Add `TryFrom<u8>` impl. Update encode_entry (5 sites) and decode_entry (5 sites). Remove the `_` catchall arm.
- `crates/minkowski-lsm/tests/manifest_integration.rs` — migrate imports from `TAG_ADD_RUN`/`TAG_REMOVE_RUN` to `ManifestTag::AddRun`/`::RemoveRun` via `as u8` casts (4 call sites).

**No new files.**

## Design decisions (locked in)

1. **Enum name**: `ManifestTag` — matches `ManifestEntry` naming.
2. **`#[repr(u8)]`**: required for wire-format compat. Discriminants match existing byte values (0x01–0x05).
3. **`TryFrom<u8>` error type**: `LsmError` directly — integrates with `?` at decode sites without adapter. Error message preserves the existing `"unknown entry tag: {byte:#04x}"` phrasing.
4. **Raw constants `pub const TAG_*`**: **removed**. Tests migrate to `ManifestTag::Variant as u8` at the four call sites. Keeping both would duplicate the naming surface.
5. **Decode match**: becomes exhaustive; no wildcard. Adding a new variant to `ManifestTag` without also adding a match arm becomes a compile error.

---

## Task 1: Replace `TAG_*` constants with `ManifestTag` enum

**Goal:** Introduce the enum, migrate encode/decode, remove old constants, migrate test call sites.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs`

- [ ] **Step 1: Write a failing unit test for `TryFrom<u8>`**

In `crates/minkowski-lsm/src/manifest_log.rs`, inside the `#[cfg(test)] mod tests` block, add:

```rust
    #[test]
    fn manifest_tag_try_from_u8_accepts_known_values() {
        assert_eq!(ManifestTag::try_from(0x01).unwrap(), ManifestTag::AddRun);
        assert_eq!(ManifestTag::try_from(0x02).unwrap(), ManifestTag::RemoveRun);
        assert_eq!(ManifestTag::try_from(0x03).unwrap(), ManifestTag::PromoteRun);
        assert_eq!(ManifestTag::try_from(0x04).unwrap(), ManifestTag::SetSequence);
        assert_eq!(ManifestTag::try_from(0x05).unwrap(), ManifestTag::AddRunAndSequence);
    }

    #[test]
    fn manifest_tag_try_from_u8_rejects_unknown_values() {
        for byte in [0x00u8, 0x06, 0x7F, 0xFF] {
            let err = ManifestTag::try_from(byte).unwrap_err();
            assert!(matches!(err, LsmError::Format(_)));
            if let LsmError::Format(msg) = err {
                assert!(msg.contains("unknown entry tag"), "got: {msg}");
            }
        }
    }

    #[test]
    fn manifest_tag_as_u8_matches_discriminant() {
        assert_eq!(ManifestTag::AddRun as u8, 0x01);
        assert_eq!(ManifestTag::RemoveRun as u8, 0x02);
        assert_eq!(ManifestTag::PromoteRun as u8, 0x03);
        assert_eq!(ManifestTag::SetSequence as u8, 0x04);
        assert_eq!(ManifestTag::AddRunAndSequence as u8, 0x05);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-lsm --lib manifest_log::tests::manifest_tag`
Expected: FAIL — `ManifestTag` doesn't exist yet.

- [ ] **Step 3: Replace the `TAG_*` constants with the enum + `TryFrom<u8>` impl**

In `crates/minkowski-lsm/src/manifest_log.rs`, find the block:

```rust
pub const TAG_ADD_RUN: u8 = 0x01;
pub const TAG_REMOVE_RUN: u8 = 0x02;
pub const TAG_PROMOTE_RUN: u8 = 0x03;
pub const TAG_SET_SEQUENCE: u8 = 0x04;
pub const TAG_ADD_RUN_AND_SEQUENCE: u8 = 0x05;
```

Replace with:

```rust
/// Tag byte identifying a `ManifestEntry` variant on disk.
///
/// `#[repr(u8)]` pins the discriminant values so the enum is layout-compatible
/// with the existing wire format. Encode casts via `as u8`; decode parses via
/// `TryFrom<u8>`, which returns `LsmError::Format` on unknown bytes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ManifestTag {
    AddRun = 0x01,
    RemoveRun = 0x02,
    PromoteRun = 0x03,
    SetSequence = 0x04,
    AddRunAndSequence = 0x05,
}

impl TryFrom<u8> for ManifestTag {
    type Error = LsmError;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        match byte {
            0x01 => Ok(Self::AddRun),
            0x02 => Ok(Self::RemoveRun),
            0x03 => Ok(Self::PromoteRun),
            0x04 => Ok(Self::SetSequence),
            0x05 => Ok(Self::AddRunAndSequence),
            other => Err(LsmError::Format(format!(
                "unknown entry tag: {other:#04x}"
            ))),
        }
    }
}
```

- [ ] **Step 4: Migrate `encode_entry` call sites**

Still in `manifest_log.rs`, find `encode_entry`. Each `buf.push(TAG_*)` becomes `buf.push(ManifestTag::Variant as u8)`. Specifically:

```rust
// Before:
        ManifestEntry::AddRun { level, meta } => {
            buf.push(TAG_ADD_RUN);
            // ...
        }
        ManifestEntry::RemoveRun { level, path } => {
            buf.push(TAG_REMOVE_RUN);
            // ...
        }
        ManifestEntry::PromoteRun { from_level, to_level, path } => {
            buf.push(TAG_PROMOTE_RUN);
            // ...
        }
        ManifestEntry::SetSequence { next_sequence } => {
            buf.push(TAG_SET_SEQUENCE);
            // ...
        }
        ManifestEntry::AddRunAndSequence { level, meta, next_sequence } => {
            buf.push(TAG_ADD_RUN_AND_SEQUENCE);
            // ...
        }
```

```rust
// After:
        ManifestEntry::AddRun { level, meta } => {
            buf.push(ManifestTag::AddRun as u8);
            // ...
        }
        ManifestEntry::RemoveRun { level, path } => {
            buf.push(ManifestTag::RemoveRun as u8);
            // ...
        }
        ManifestEntry::PromoteRun { from_level, to_level, path } => {
            buf.push(ManifestTag::PromoteRun as u8);
            // ...
        }
        ManifestEntry::SetSequence { next_sequence } => {
            buf.push(ManifestTag::SetSequence as u8);
            // ...
        }
        ManifestEntry::AddRunAndSequence { level, meta, next_sequence } => {
            buf.push(ManifestTag::AddRunAndSequence as u8);
            // ...
        }
```

- [ ] **Step 5: Migrate `decode_entry` to exhaustive match via `TryFrom<u8>`**

Still in `manifest_log.rs`, find `decode_entry`. Current shape (simplified):

```rust
fn decode_entry(data: &[u8]) -> Result<ManifestEntry, LsmError> {
    if data.is_empty() {
        return Err(LsmError::Format("empty entry payload".to_owned()));
    }
    let tag = data[0];
    let offset = 1;

    match tag {
        TAG_ADD_RUN => {
            // ...
        }
        TAG_REMOVE_RUN => {
            // ...
        }
        TAG_PROMOTE_RUN => {
            // ...
        }
        TAG_SET_SEQUENCE => {
            // ...
        }
        TAG_ADD_RUN_AND_SEQUENCE => {
            // ...
        }
        other => Err(LsmError::Format(format!(
            "unknown entry tag: {other:#04x}"
        ))),
    }
}
```

Change to:

```rust
fn decode_entry(data: &[u8]) -> Result<ManifestEntry, LsmError> {
    if data.is_empty() {
        return Err(LsmError::Format("empty entry payload".to_owned()));
    }
    let tag = ManifestTag::try_from(data[0])?;
    let offset = 1;

    match tag {
        ManifestTag::AddRun => {
            // ...
        }
        ManifestTag::RemoveRun => {
            // ...
        }
        ManifestTag::PromoteRun => {
            // ...
        }
        ManifestTag::SetSequence => {
            // ...
        }
        ManifestTag::AddRunAndSequence => {
            // ...
        }
    }
}
```

The wildcard arm is GONE. The exhaustive match is structurally guaranteed by the enum — adding a sixth variant to `ManifestTag` without a matching arm becomes a compile error.

Leave the arm bodies (the actual decode logic) unchanged — they compile as-is against the new match patterns.

- [ ] **Step 6: Migrate test call sites in `tests/manifest_integration.rs`**

Find the import:

```rust
use minkowski_lsm::manifest_log::{ManifestEntry, ManifestLog, TAG_ADD_RUN, TAG_REMOVE_RUN};
```

Replace with:

```rust
use minkowski_lsm::manifest_log::{ManifestEntry, ManifestLog, ManifestTag};
```

Find the four `payload.push(TAG_*)` call sites (all inside regression tests):

```rust
// Before:
    payload.push(TAG_ADD_RUN);    // (3 occurrences)
    payload.push(TAG_REMOVE_RUN); // (1 occurrence)

// After:
    payload.push(ManifestTag::AddRun as u8);
    payload.push(ManifestTag::RemoveRun as u8);
```

- [ ] **Step 7: Run cargo check**

Run: `cargo check -p minkowski-lsm --all-targets`
Expected: clean compile. The exhaustive match in `decode_entry` will fail to compile if any variant was forgotten — good signal.

- [ ] **Step 8: Run full tests**

Run: `cargo test -p minkowski-lsm`
Expected: 117 tests pass (114 existing + 3 new `manifest_tag_*`).

- [ ] **Step 9: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean. No new suppressions expected.

- [ ] **Step 10: Commit**

```bash
git add crates/minkowski-lsm/src/manifest_log.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "refactor(lsm): ManifestTag enum replaces raw u8 TAG_* constants

Introduces #[repr(u8)] pub enum ManifestTag with TryFrom<u8>. The
five pub const TAG_*: u8 constants are gone — encode casts via
'as u8', decode parses via TryFrom<u8> and surfaces unknown bytes as
LsmError::Format('unknown entry tag: 0xNN').

The decode match is now exhaustive — the prior '_ => Err(Format(...))'
wildcard arm is removed. A future ManifestEntry variant requires
both a ManifestTag enum entry AND a match arm, caught at compile
time rather than as a runtime 'unknown tag' surprise.

Wire format bit-for-bit unchanged: #[repr(u8)] pins the discriminants
at their prior values (0x01-0x05).

Three new unit tests:
- manifest_tag_try_from_u8_accepts_known_values (all five variants)
- manifest_tag_try_from_u8_rejects_unknown_values (0x00, 0x06, 0x7F, 0xFF)
- manifest_tag_as_u8_matches_discriminant (pins the repr contract)

Tests migrating from the old TAG_* imports use 'ManifestTag::X as u8'
at the four regression-test frame-crafting sites."
```

If the pre-commit fmt hook modifies files, re-stage and re-commit (never amend — TigerStyle rule).

---

## Task 2: Final verification + push + PR

**No code changes. Green-light gate.**

- [ ] **Step 1: Workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 2: Workspace tests**

Run: `cargo test --workspace`
Expected: the 3 pre-existing `minkowski-observe` failures still fail (pre-existing on main; CI doesn't run them). All other tests pass, including 117 in `minkowski-lsm`.

- [ ] **Step 3: Fmt check**

Run: `cargo fmt --all -- --check`
Expected: clean.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin lsm/phase3-prep-enum-tags
gh pr create --title "refactor(lsm): ManifestTag enum (Phase 3 pre-work)" --body "$(cat <<'EOF'
## Summary

Replaces \`manifest_log.rs\`'s five raw-u8 \`TAG_*\` constants with a \`#[repr(u8)] pub enum ManifestTag\`. Decode becomes exhaustive via \`TryFrom<u8>\`, removing the wildcard catchall arm.

Final piece of the LSM type-safety arc (Phase 3 pre-work per \`project_scaling_roadmap.md\`). Addresses the type-design reviewer's last remaining suggestion across PR A/B1/B2/B3: \`ManifestLog\` Invariant Expression 5 → 8.

## What landed

- **\`ManifestTag\` enum** with variants matching the prior constants (0x01–0x05).
- **\`TryFrom<u8> for ManifestTag\`**: unknown bytes surface as \`LsmError::Format("unknown entry tag: 0xNN")\`.
- **Exhaustive match in \`decode_entry\`**: the \`_ => Err(...)\` wildcard is gone. A future variant without a matching arm fails to compile.
- **Tests migrate** from \`TAG_ADD_RUN\` / \`TAG_REMOVE_RUN\` imports to \`ManifestTag::X as u8\` at the four frame-crafting regression-test sites.

## Breaking changes

Internal only — no external consumers of \`minkowski-lsm\` exist.

- \`pub const TAG_*\` constants removed. Callers in tests migrate to \`ManifestTag::X as u8\`.

Wire format bit-for-bit unchanged: \`#[repr(u8)]\` pins discriminants at the prior 0x01-0x05 values.

## Tests

117 total in \`minkowski-lsm\` (up from 114):
- 3 new unit tests for \`ManifestTag::try_from(u8)\` + discriminant-as-u8 contract.

## Test plan

- [x] \`cargo test -p minkowski-lsm\` — 117/117 pass
- [x] \`cargo clippy --workspace --all-targets -- -D warnings\` — clean
- [x] \`cargo fmt --all -- --check\` — clean
- [ ] CI pipeline (fmt, clippy, test, tsan, loom, claude-review)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Monitor CI; update memory after merge**

Once the PR merges, update `project_scaling_roadmap.md` to:
- Mark Phase 3 pre-work complete.
- Bump `ManifestLog`'s Invariant Expression rating from 5 to 8 in the type-ratings table.
- Note that Phase 3 (compactor) or Phase 4 (bloom) is now the unambiguous next step.

---

## Self-review (done inline before saving)

- **Spec coverage**: single deliverable = "replace TAG_* with ManifestTag enum." Covered in Task 1 with all 10 call sites (5 encode + 5 decode) plus 4 test-file migrations.
- **Placeholder scan**: none. Every code block is complete.
- **Type consistency**: `ManifestTag` enum variants match the match arm names. The `TryFrom<u8>` error type is `LsmError` (project convention). `as u8` casts are used consistently.

Self-review complete.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-04-18-manifest-tag-enum-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — two tasks, tight scope, matches the pattern from PR B2/B3.

**2. Inline Execution** — batch here.

Which approach?
