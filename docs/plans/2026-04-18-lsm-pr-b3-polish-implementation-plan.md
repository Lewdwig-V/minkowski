# LSM PR B3 Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the five deferred polish items from PR B2's review queue: `SeqNo` privatization + `.next()`, `SizeBytes(u64)` newtype, `SortedRunMeta::new` taking both newtypes at the boundary, expanded arithmetic tombstones, and a `recover → append → recover` reserved-bytes round-trip test.

**Architecture:** Type-level hardening on the manifest subsystem. No wire format changes, no public API consumer impact outside the crate (no external consumers exist today). Privatizes `SeqNo.0`, introduces `SizeBytes(u64)` as `PageCount`'s companion, and pushes validation/wrapping up to callers of `SortedRunMeta::new` so the constructor takes type-safe arguments that make the swap-at-call-site bug a compile error. Each task ends with green `cargo test -p minkowski-lsm` + `cargo clippy --workspace --all-targets -- -D warnings`.

**Tech Stack:** Rust 2024 edition, `minkowski-lsm` workspace crate, existing `SeqNo`/`SeqRange`/`Level`/`PageCount` newtypes, `static_assertions` dev-dep (from PR B2).

**Spec:** `project_lsm_phase2_type_safety.md` in memory — the "PR B3 candidate queue" section.

---

## Starting state

- Branch: `lsm/pr-b3-polish` already created off `origin/main` (post-PR-B2 squash `d521e51`).
- 107 tests currently passing in `minkowski-lsm`.

## File structure

**Modify (by task):**

- Task 1: `crates/minkowski-lsm/src/types.rs` (privatize `SeqNo.0`, add `.get()` + `.next()`), plus ~40 call sites across `manifest_log.rs`, `manifest_ops.rs`, `writer.rs`, `reader.rs`, `manifest.rs`, and the two integration-test files.
- Task 2: `crates/minkowski-lsm/src/types.rs` (new `SizeBytes(u64)` type with unit tests).
- Task 3: `crates/minkowski-lsm/src/manifest.rs` (constructor signature + `size_bytes` field + accessor), `crates/minkowski-lsm/src/manifest_ops.rs` (construction site wraps), `crates/minkowski-lsm/src/manifest_log.rs` (decode wraps).
- Task 4: `crates/minkowski-lsm/src/types.rs` (expand tombstones), `crates/minkowski-lsm/tests/manifest_integration.rs` (new test).
- Task 5: Final verification + push + PR.

**No new files.**

---

## Task 1: Privatize `SeqNo.0`; add `.get()` and `.next()`

**Goal:** Move `SeqNo` from "transparent newtype with `pub u64`" (Encapsulation 5/10) to "opaque newtype with validated advance" (10/10). Adds `.next()` as the named `succ` operation with panic-on-overflow semantics (an internal-invariant violation, per TigerStyle).

**Files:**
- Modify: `crates/minkowski-lsm/src/types.rs`
- Modify (cascade): every call site constructing `SeqNo(u64)` or reading `seq.0` — primarily `manifest_log.rs`, `manifest_ops.rs`, `writer.rs`, `reader.rs`, `manifest.rs`, and both test files.

- [ ] **Step 1: Write failing unit tests for `.get()` and `.next()`**

In `crates/minkowski-lsm/src/types.rs`, inside the `#[cfg(test)] mod tests` block, add:

```rust
    #[test]
    fn seqno_get_returns_inner_u64() {
        let s = SeqNo::from(42u64);
        assert_eq!(s.get(), 42);
    }

    #[test]
    fn seqno_next_advances_by_one() {
        let s = SeqNo::from(5u64);
        let n = s.next();
        assert_eq!(n.get(), 6);
    }

    #[test]
    #[should_panic(expected = "SeqNo overflow")]
    fn seqno_next_panics_on_overflow() {
        let s = SeqNo::from(u64::MAX);
        let _ = s.next();
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-lsm --lib types::tests -- seqno_get seqno_next`
Expected: FAIL — `.get()` and `.next()` methods don't exist yet. Compilation error.

- [ ] **Step 3: Privatize the field and add methods**

In `crates/minkowski-lsm/src/types.rs`, find the `SeqNo` struct:

```rust
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SeqNo(pub u64);
```

Change to private field and add the two methods (keep the existing `From`/`Display` impls unchanged — they're inside the module so they retain field access):

```rust
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SeqNo(u64);

impl SeqNo {
    /// Extract the underlying `u64`.
    pub fn get(self) -> u64 {
        self.0
    }

    /// The next sequence number. Panics on `u64::MAX + 1` — an internal
    /// invariant violation, since the WAL is the only `SeqNo` producer
    /// and a 64-bit sequence space exhausts long after any realistic
    /// process lifetime.
    pub fn next(self) -> Self {
        Self(self.0.checked_add(1).expect("SeqNo overflow"))
    }
}
```

Verify the existing `impl From<u64> for SeqNo { fn from(v: u64) -> Self { Self(v) } }` and `impl From<SeqNo> for u64 { fn from(s: SeqNo) -> Self { s.0 } }` remain unchanged — they're in the same module so they access the now-private field just fine.

- [ ] **Step 4: Run the new tests**

Run: `cargo test -p minkowski-lsm --lib types::tests -- seqno_get seqno_next`
Expected: 3 tests pass.

- [ ] **Step 5: Run `cargo check` to enumerate external call sites**

Run: `cargo check -p minkowski-lsm 2>&1 | head -60`
Expected: compile errors at every `SeqNo(x)` tuple-construction call site and every `seq.0` field-access site outside `types.rs`.

Known call sites (approximate count ~40):
- `crates/minkowski-lsm/src/manifest_log.rs` — `next_sequence.0.to_le_bytes()` at encode sites (2x), `SeqNo(read_u64_le(...)?)` at decode sites (2x), `SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?` (2x).
- `crates/minkowski-lsm/src/writer.rs` — `sequence_range.lo().0` / `sequence_range.hi().0` (2x).
- `crates/minkowski-lsm/src/reader.rs` — similar header byte-extraction sites.
- `crates/minkowski-lsm/src/manifest.rs` tests — `test_meta` helper, `SeqRange::new(SeqNo(0), SeqNo(10))` patterns.
- `crates/minkowski-lsm/src/manifest_ops.rs` tests — test setup sites.
- `crates/minkowski-lsm/tests/manifest_integration.rs` — ~25 sites constructing `SeqNo(...)` in `SeqRange::new` calls and assertions.
- `crates/minkowski-lsm/tests/integration.rs` — ~5 sites.

- [ ] **Step 6: Migrate `SeqNo(x)` construction sites to `SeqNo::from(x)`**

Every `SeqNo(literal)` or `SeqNo(variable)` pattern becomes `SeqNo::from(literal)` / `SeqNo::from(variable)`.

For the decode path in `manifest_log.rs`:

```rust
// Before:
let next_sequence = SeqNo(read_u64_le(data, &mut offset)?);

// After:
let next_sequence = SeqNo::from(read_u64_le(data, &mut offset)?);
```

For `SeqRange::new(SeqNo(lo), SeqNo(hi))?` in two decode branches:

```rust
// Before:
SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,

// After:
SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi))?,
```

In tests, every `SeqNo(N)` literal becomes `SeqNo::from(N)` with the `u64` type ascription preserved where required. The compiler complains at every site — iterate until clean.

- [ ] **Step 7: Migrate `seq.0` field-access sites to `seq.get()`**

For encode sites in `manifest_log.rs`:

```rust
// Before:
buf.extend_from_slice(&next_sequence.0.to_le_bytes());

// After:
buf.extend_from_slice(&next_sequence.get().to_le_bytes());
```

For `sequence_range.lo().0` / `.hi().0` in `writer.rs`:

```rust
// Before:
header.sequence_lo = sequence_range.lo().0;
header.sequence_hi = sequence_range.hi().0;

// After:
header.sequence_lo = sequence_range.lo().get();
header.sequence_hi = sequence_range.hi().get();
```

Apply `.get()` substitution everywhere the compiler flags `.0` on a `SeqNo` value.

- [ ] **Step 8: Run cargo check**

Run: `cargo check -p minkowski-lsm --all-targets`
Expected: clean compile.

- [ ] **Step 9: Run full tests**

Run: `cargo test -p minkowski-lsm`
Expected: 110 tests pass (107 existing + 3 new `seqno_*`).

- [ ] **Step 10: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add crates/minkowski-lsm/
git commit -m "feat(lsm): privatize SeqNo.0; add .get() and .next()

Moves SeqNo from a transparent newtype (pub u64) to an opaque newtype
with explicit accessor and named 'advance' operation. Closes the
Encapsulation 5/10 → 10/10 gap the type-design reviewer flagged across
PR A / PR B1 / PR B2 reviews.

- .get(self) -> u64 for explicit, grep-friendly extraction.
- .next(self) -> Self for 'next sequence' with checked_add(1).expect()
  semantics — overflow is an internal invariant violation (WAL is the
  only producer; 2^64 seqs exhausts long after any realistic lifetime).

From<u64> and From<SeqNo> for u64 impls retained — callers can still
use .into() or From::from for infallible construction. All ~40
call sites migrated: SeqNo(x) -> SeqNo::from(x), seq.0 -> seq.get().

No wire format change. No public API change visible to external
consumers (there are none today)."
```

If the pre-commit fmt hook modifies files, re-stage and re-commit (never amend — TigerStyle rule).

---

## Task 2: `SizeBytes(u64)` newtype

**Goal:** Add `SizeBytes(u64)` as the companion to `PageCount`, preparing for Task 3 where `SortedRunMeta::new` takes both newtypes. Pure addition; no caller migration yet.

**Files:**
- Modify: `crates/minkowski-lsm/src/types.rs`

- [ ] **Step 1: Write failing unit tests for `SizeBytes`**

Add to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn sizebytes_get_returns_inner_u64() {
        let s = SizeBytes::new(1024);
        assert_eq!(s.get(), 1024);
    }

    #[test]
    fn sizebytes_allows_zero() {
        let s = SizeBytes::new(0);
        assert_eq!(s.get(), 0);
    }

    #[test]
    fn sizebytes_display() {
        assert_eq!(SizeBytes::new(42).to_string(), "42");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-lsm --lib types::tests -- sizebytes`
Expected: FAIL — `SizeBytes` doesn't exist.

- [ ] **Step 3: Add `SizeBytes`**

In `crates/minkowski-lsm/src/types.rs`, after the `PageCount` definition (they're symmetric newtypes; place them together), add:

```rust
/// Size in bytes of an on-disk artifact.
///
/// Infallible newtype — zero is permitted (matches the semantics of
/// `fs::metadata(...).len()`, which returns `0` for empty files).
/// Type-level distinction from `PageCount` prevents arg-swap bugs at
/// `SortedRunMeta::new`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SizeBytes(u64);

impl SizeBytes {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SizeBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
```

Note the difference from `PageCount`: `SizeBytes::new` is infallible (returns `Self` directly, not `Option<Self>`). The `u64` field is private, matching the privatization precedent set by Task 1's `SeqNo`. No `From<u64>` impl — the spec item #2 doesn't mention one, and adding it invites silent arg-swap via `.into()` at `SortedRunMeta::new` call sites. Keep construction explicit via `::new`.

- [ ] **Step 4: Add layout assertions matching `PageCount`**

Find the `assert_eq_size!` block for `PageCount` in the test module. Add matching assertions for `SizeBytes`:

```rust
    assert_eq_size!(SizeBytes, u64);
    assert_eq_size!(Option<SizeBytes>, u64);
```

Wait — the second one is false: `Option<SizeBytes>` is NOT u64-sized because `SizeBytes` has no niche (plain `u64` uses the full range). Use only the first assertion:

```rust
    assert_eq_size!(SizeBytes, u64);
```

- [ ] **Step 5: Run the new tests**

Run: `cargo test -p minkowski-lsm --lib types::tests -- sizebytes`
Expected: 3 tests pass. The `assert_eq_size!` is a compile-time check, not a runtime test.

- [ ] **Step 6: Run full tests + clippy**

```bash
cargo test -p minkowski-lsm
cargo clippy -p minkowski-lsm --all-targets -- -D warnings
```

Expected: 113 tests pass (110 from Task 1 + 3 new `sizebytes_*`). Clippy clean.

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski-lsm/src/types.rs
git commit -m "feat(lsm): add SizeBytes(u64) newtype

Infallible newtype — zero is permitted (matches fs::metadata.len()
semantics). Private u64 field; pub fn new/get/Display only. No From<u64>
impl to prevent arg-swap via .into() at the SortedRunMeta::new
call site (Task 3 uses SizeBytes::new explicitly).

No callers migrated yet. Task 3 threads this and PageCount through
SortedRunMeta::new so the existing arg-swap hazard (two adjacent u64s)
becomes a compile error."
```

---

## Task 3: `SortedRunMeta::new` takes `PageCount` + `SizeBytes`

**Goal:** Push `PageCount::new` validation up to callers; replace raw `u64` constructor args with the type-safe wrappers. Closes the "two adjacent u64s can be swapped" hazard the type-design reviewer flagged.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs` (constructor signature + field type for `size_bytes` + accessor)
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs` (`flush_and_record` construction site)
- Modify: `crates/minkowski-lsm/src/manifest_log.rs` (two decode construction sites)

- [ ] **Step 1: Change the field type and accessor for `size_bytes`**

In `crates/minkowski-lsm/src/manifest.rs`, update imports:

```rust
use crate::types::{PageCount, SeqRange, SizeBytes};
```

Find the `SortedRunMeta` struct definition and change `size_bytes`:

```rust
pub struct SortedRunMeta {
    path: PathBuf,
    sequence_range: SeqRange,
    archetype_coverage: Box<[u16]>,
    page_count: PageCount,
    size_bytes: SizeBytes,       // was: u64
}
```

Update the accessor:

```rust
    pub fn size_bytes(&self) -> SizeBytes {
        self.size_bytes
    }
```

- [ ] **Step 2: Update `SortedRunMeta::new` signature**

Change the signature to take `PageCount` and `SizeBytes` directly:

```rust
    pub fn new(
        path: PathBuf,
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,
        page_count: PageCount,       // was: u64
        size_bytes: SizeBytes,       // was: u64
    ) -> Result<Self, LsmError> {
        if archetype_coverage.windows(2).any(|w| w[0] >= w[1]) {
            return Err(LsmError::Format(
                "archetype_coverage is not strictly sorted".to_owned(),
            ));
        }
        // Note: the PageCount::new fallibility has moved up to callers.
        // SortedRunMeta::new is now only responsible for the coverage
        // sort check.
        Ok(Self {
            path,
            sequence_range,
            archetype_coverage: archetype_coverage.into_boxed_slice(),
            page_count,
            size_bytes,
        })
    }
```

Delete the old `PageCount::new(page_count).ok_or_else(...)?` line — the validation is now at the call sites.

- [ ] **Step 3: Migrate `flush_and_record` in `manifest_ops.rs`**

In `crates/minkowski-lsm/src/manifest_ops.rs`, update imports:

```rust
use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};
```

Find the `SortedRunMeta::new(...)` call. Currently:

```rust
    let meta = SortedRunMeta::new(
        path.clone(),
        reader.sequence_range(),
        archetype_coverage,
        reader.page_count(),            // u64
        file_size,                       // u64
    )?;
```

`reader.page_count()` already returns `u64`; `file_size` is `u64`. Wrap both:

```rust
    let page_count = PageCount::new(reader.page_count()).ok_or_else(|| {
        LsmError::Format("page_count must be non-zero".to_owned())
    })?;
    let meta = SortedRunMeta::new(
        path.clone(),
        reader.sequence_range(),
        archetype_coverage,
        page_count,
        SizeBytes::new(file_size),
    )?;
```

- [ ] **Step 4: Migrate `decode_entry` sites in `manifest_log.rs`**

Find the `TAG_ADD_RUN` decode arm. Currently:

```rust
            let page_count = read_u64_le(data, &mut offset)?;
            let size_bytes = read_u64_le(data, &mut offset)?;
            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi))?,
                coverage,
                page_count,        // u64
                size_bytes,        // u64
            )?;
```

Update to wrap inline:

```rust
            let page_count = read_u64_le(data, &mut offset)?;
            let size_bytes = read_u64_le(data, &mut offset)?;
            let page_count = PageCount::new(page_count).ok_or_else(|| {
                LsmError::Format("page_count must be non-zero".to_owned())
            })?;
            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi))?,
                coverage,
                page_count,
                SizeBytes::new(size_bytes),
            )?;
```

Apply the same pattern to the `TAG_ADD_RUN_AND_SEQUENCE` decode arm. Both sites now call `PageCount::new(...).ok_or_else(...)?` before `SortedRunMeta::new`, surfacing `LsmError::Format` on a zero `page_count` — same behavior as before, just at a slightly earlier layer. The replay loop's error handling (truncate-on-Format) is unchanged.

- [ ] **Step 5: Migrate test construction sites**

In `manifest.rs` test module, find the `test_meta` helper. Update it to use the new signature:

```rust
    fn test_meta(name: &str, coverage: Vec<u16>) -> SortedRunMeta {
        SortedRunMeta::new(
            PathBuf::from(name),
            SeqRange::new(SeqNo::from(0), SeqNo::from(10)).unwrap(),
            coverage,
            PageCount::new(42).unwrap(),
            SizeBytes::new(8192),
        )
        .unwrap()
    }
```

Also update test-module imports to include `PageCount` and `SizeBytes`.

Any direct `SortedRunMeta::new` call in unit tests or integration tests needs the same treatment. The existing 4 `sorted_run_meta_new_*` tests in `manifest.rs` construct with bare `u64`s — each needs updating:

```rust
    #[test]
    fn sorted_run_meta_new_rejects_unsorted_coverage() {
        let result = SortedRunMeta::new(
            PathBuf::from("x.run"),
            SeqRange::new(SeqNo::from(0), SeqNo::from(10)).unwrap(),
            vec![3, 1, 2],
            PageCount::new(1).unwrap(),
            SizeBytes::new(1024),
        );
        assert!(matches!(result, Err(LsmError::Format(_))));
    }
```

Apply to all four constructor tests. The test `sorted_run_meta_new_rejects_zero_page_count` changes semantics — zero page_count is no longer caught by `SortedRunMeta::new`; it's caught by the caller at `PageCount::new(0)`. Rename to `page_count_new_rejects_zero` and move to `types.rs` test module if not already covered there (the existing `pagecount_rejects_zero` test covers this). The redundant test can be deleted.

Similarly, any `.size_bytes()` assertion that compared to a `u64` literal now needs `.get()`:

```rust
// Before:
assert_eq!(meta.size_bytes(), 4);

// After:
assert_eq!(meta.size_bytes().get(), 4);
```

- [ ] **Step 6: Run cargo check**

Run: `cargo check -p minkowski-lsm --all-targets`
Expected: clean compile. If anything fails, it's a missed constructor call site — enumerate via the compile error and fix.

- [ ] **Step 7: Run full tests**

Run: `cargo test -p minkowski-lsm`
Expected: test count may shift depending on whether `sorted_run_meta_new_rejects_zero_page_count` was removed (-1) or kept with new semantics. Target: ~112 tests pass.

- [ ] **Step 8: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add crates/minkowski-lsm/
git commit -m "refactor(lsm): SortedRunMeta::new takes PageCount and SizeBytes

Pushes PageCount::new validation from inside the constructor up to
the two decode sites and flush_and_record. Constructor signature is now
(path, SeqRange, Vec<u16>, PageCount, SizeBytes) — two adjacent
newtypes instead of two adjacent u64s. Swap-at-call-site is now a
compile error.

size_bytes field and accessor return type change from u64 to SizeBytes.
Tests comparing .size_bytes() to a u64 literal migrated to .get().

No wire format change. Decode still reads u64 bytes; PageCount::new
runs at the decode site, surfacing zero as LsmError::Format as before —
replay loop truncation handles it unchanged.

The in-crate test 'sorted_run_meta_new_rejects_zero_page_count' is
removed — the zero-page_count behavior is now covered by
pagecount_rejects_zero (types.rs) plus the in-place decode path
in manifest_log.rs."
```

---

## Task 4: Expand tombstones + reserved-bytes round-trip test

**Goal:** Belt-and-suspenders: add `Mul/Div/Rem/Neg` to `SeqNo` arithmetic tombstones. Add an integration test pinning "reserved bytes survive a `recover → append → recover` cycle."

**Files:**
- Modify: `crates/minkowski-lsm/src/types.rs` (expand tombstone macro calls)
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs` (new test)

- [ ] **Step 1: Expand `SeqNo` arithmetic tombstones**

In `crates/minkowski-lsm/src/types.rs`, find the `assert_not_impl_all!` block for `SeqNo`. Extend the imports and macro calls:

```rust
    use static_assertions::{assert_eq_size, assert_not_impl_all};
    use std::ops::{Add, AddAssign, Div, Mul, Neg, Rem, Sub, SubAssign};

    // Tombstone tests: SeqNo must NOT implement any arithmetic.
    assert_not_impl_all!(SeqNo: Add<SeqNo>, Sub<SeqNo>, AddAssign<SeqNo>, SubAssign<SeqNo>);
    assert_not_impl_all!(SeqNo: Add<u64>, Sub<u64>, AddAssign<u64>, SubAssign<u64>);
    assert_not_impl_all!(SeqNo: Mul<u64>, Div<u64>, Rem<u64>, Neg);
```

The third line is the new one. `Neg` has no type parameter (it's unary). The Mul/Div/Rem against `u64` cover the realistic "sequences divided into chunks" misuse.

- [ ] **Step 2: Run tests to verify the tombstones compile**

Run: `cargo test -p minkowski-lsm --lib`
Expected: clean build. If `SeqNo` accidentally had any of the added traits, compile would fail — none should today.

- [ ] **Step 3: Add the `recover → append → recover` reserved-bytes test**

In `crates/minkowski-lsm/tests/manifest_integration.rs`, near the existing `recover_ignores_nonzero_reserved_bytes` test, add:

```rust
/// Reserved bytes must survive an append round trip. Guards against a
/// future refactor that "normalizes" the header on every recover — which
/// would silently drop forward-compat flags a future version wrote there.
#[test]
fn recover_preserves_reserved_bytes_through_append_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("reserved_append.log");

    // Start with a valid header whose reserved bytes carry a non-zero
    // "flag" pattern.
    fs::write(&log_path, b"MKMF\x01\xFF\xAA\x55").unwrap();

    // Open, append one entry, close.
    let (_, mut log) = ManifestLog::recover(&log_path).unwrap();
    log.append(&ManifestEntry::SetSequence {
        next_sequence: SeqNo::from(42),
    })
    .unwrap();
    drop(log);

    // Reserved bytes at offsets 5..8 must still be intact.
    let bytes = fs::read(&log_path).unwrap();
    assert_eq!(&bytes[0..4], b"MKMF", "magic preserved");
    assert_eq!(bytes[4], 0x01, "version preserved");
    assert_eq!(&bytes[5..8], &[0xFF, 0xAA, 0x55], "reserved bytes preserved");

    // And the entry must replay.
    let (m, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(m.next_sequence(), SeqNo::from(42));
}
```

Imports at the top of the file should already include `SeqNo`, `ManifestEntry`, `ManifestLog` from prior PRs. Verify.

- [ ] **Step 4: Run the new test**

Run: `cargo test -p minkowski-lsm --test manifest_integration recover_preserves_reserved_bytes_through_append_cycle`
Expected: PASS.

- [ ] **Step 5: Run full tests + clippy**

```bash
cargo test -p minkowski-lsm
cargo clippy -p minkowski-lsm --all-targets -- -D warnings
```

Expected: 113 tests pass (Task 3's count + 1 new integration test). Clippy clean.

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski-lsm/src/types.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "test(lsm): expand SeqNo tombstones + reserved-bytes round trip

Two items from the PR B2 review's deferred-polish queue:

- Add Mul/Div/Rem/Neg to SeqNo arithmetic tombstones. The Add/Sub/
  AddAssign/SubAssign set covered the realistic accident surface;
  these additional traits close the 'why only additive?' question
  belt-and-suspenders style.
- Add recover_preserves_reserved_bytes_through_append_cycle. The
  existing recover_ignores_nonzero_reserved_bytes covers the empty-
  body case; this variant verifies that append doesn't rewrite the
  header and that forward-compat reserved-byte flags survive round
  trips through append + reopen."
```

---

## Task 5: Final verification + push + PR

**No code changes. Green-light gate.**

- [ ] **Step 1: Update local toolchain**

Run: `rustup update stable`
Expected: toolchain update or no-op.

- [ ] **Step 2: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace`
Expected: the 3 pre-existing `minkowski-observe` failures will still fail (pre-existing on main; CI doesn't run them). All other tests pass, including the 113 in `minkowski-lsm`.

- [ ] **Step 4: Run cargo fmt check**

Run: `cargo fmt --all -- --check`
Expected: clean.

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin lsm/pr-b3-polish
gh pr create --title "chore(lsm): manifest type-safety polish (PR B3)" --body "$(cat <<'EOF'
## Summary

Five deferred polish items from PR B2's review queue. Type-level hardening on the manifest subsystem; no wire format or external API changes.

Plan: \`docs/plans/2026-04-18-lsm-pr-b3-polish-implementation-plan.md\`

## What landed

- **\`SeqNo\` privatization + \`.get()\` + \`.next()\`**: the inner \`u64\` is now private. Explicit accessor and named \`succ\` operation (\`.next()\` with \`checked_add(1).expect("SeqNo overflow")\` semantics — overflow is an internal invariant violation per TigerStyle).
- **\`SizeBytes(u64)\` newtype**: companion to \`PageCount\`, infallible (zero permitted per \`fs::metadata().len()\` semantics).
- **\`SortedRunMeta::new\` takes \`PageCount\` + \`SizeBytes\`**: pushes \`PageCount::new\` validation up to callers (decode + flush_and_record); closes the arg-swap hazard where two adjacent \`u64\`s could be transposed.
- **Expanded \`SeqNo\` tombstones**: \`Mul/Div/Rem/Neg\` added to the existing \`Add/Sub/AddAssign/SubAssign\` set.
- **\`recover → append → recover\` reserved-bytes test**: pins the forward-compat contract that append doesn't rewrite the header.

## Breaking changes

Internal only — no external consumers of \`minkowski-lsm\` exist.

- \`SeqNo\` inner field is private. Construction via \`SeqNo::from(u64)\` / \`.into()\`, extraction via \`.get()\` or \`u64::from(seq)\`.
- \`SortedRunMeta::new\` signature changes: takes \`PageCount\`, \`SizeBytes\` instead of two \`u64\`s. Callers wrap at construction.
- \`SortedRunMeta::size_bytes()\` returns \`SizeBytes\` instead of \`u64\`. Test assertions against \`u64\` literals need \`.get()\`.

No wire format changes — encode/decode still reads/writes \`u64\` bytes, wrapping at codec boundaries.

## Tests

113 total in \`minkowski-lsm\` (up from 107):
- 3 new \`SeqNo::next/get\` unit tests (including overflow panic)
- 3 new \`SizeBytes\` unit tests
- 1 new \`recover_preserves_reserved_bytes_through_append_cycle\` integration test
- 1 redundant test removed (\`sorted_run_meta_new_rejects_zero_page_count\` — now covered by \`pagecount_rejects_zero\`)

## Test plan

- [x] \`cargo test -p minkowski-lsm\` — 113/113 pass
- [x] \`cargo clippy --workspace --all-targets -- -D warnings\` — clean
- [x] \`cargo fmt --all -- --check\` — clean
- [ ] CI pipeline (fmt, clippy, test, tsan, loom, claude-review)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Monitor CI; update memory after merge**

Once the PR merges, update:
- \`project_scaling_roadmap.md\`: note PR B3 landed.
- \`project_lsm_phase2_type_safety.md\`: items are all consumed — repurpose as a Phase 3 notes file or delete.

---

## Self-review (done inline before saving)

- **Spec coverage:**
  - Item 1 (SortedRunMeta::new takes PageCount) → Task 3
  - Item 2 (SizeBytes newtype) → Task 2 + Task 3
  - Item 3 (SeqNo privatization + .next()) → Task 1
  - Item 4 (Mul/Div/Rem/Neg tombstones) → Task 4 Step 1
  - Item 5 (reserved-bytes round trip) → Task 4 Step 3

- **Placeholder scan:** None. Every code block has complete code.

- **Type consistency:**
  - `SeqNo::get(self) -> u64` method referenced in Task 1 is used in Task 3's encode sites and test assertions. Consistent.
  - `SeqNo::next(self) -> Self` — defined in Task 1, not used in later tasks (no call sites migrate to use it in this PR; it's available for future callers). Correct scope.
  - `SizeBytes::new(u64) -> Self` / `SizeBytes::get(self) -> u64` — defined in Task 2, used consistently in Task 3.
  - `SortedRunMeta::new(path, SeqRange, Vec<u16>, PageCount, SizeBytes) -> Result<Self>` — final signature across Task 3 and Task 4's test. Consistent.
  - `.into_boxed_slice()` at the coverage conversion boundary — preserved from PR B2, unchanged.

Self-review complete. Plan is ready.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-04-18-lsm-pr-b3-polish-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — five tasks, each reasonably scoped. Task 1 has the biggest mechanical cascade (~40 SeqNo migration sites); Task 3 cascades through three files. Matches the pattern that worked for PR B1, PR B2.

**2. Inline Execution** — batch execution with checkpoints here.

Which approach?
