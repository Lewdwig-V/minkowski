# Phase 3 PR 3b: Compactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the compactor proper — the merge kernel that consolidates multiple sorted runs at L(N) into a single run at L(N+1), plus the `World::compact_one()` / `World::needs_compaction()` public API and the `FlushWriter::set_entry_observer` hook for Phase 4 bloom.

**Architecture:** A new `Compactor` module orchestrates per-archetype compaction jobs. A new `CompactionWriter` produces output sorted runs by iterating input readers, deduplicating entities by sequence (highest wins), and copying component data from source rows to output rows. Archetype identity is resolved across runs via `SchemaSection` (component stable-name match, not the run-local `arch_id: u16`). `World` gains `compact_one()` + `needs_compaction()` thin wrappers that plumb through `LsmManifest` + `ManifestLog` + `CodecRegistry` to the Compactor.

**Tech Stack:** Rust edition 2024, `#[repr(C)]` on-disk format from Phase 1, const-generic `LsmManifest<N>` from PR #168, `ManifestEntry::CompactionCommit` from PR #169, `SortedRunReader` + `FlushWriter` from Phase 1.

**Scope boundaries:**
- Ledger-shape merge only — no tombstone handling. On-disk despawn semantics are undefined (absence in newer run ≠ deletion under "dirty pages only" flush). A separate follow-up PR will add tombstone-on-disk support if/when needed.
- C-mode only (`World::compact_one()` is synchronous, user-triggered). D-mode (continuous background) is a later phase.
- `FlushWriter::set_entry_observer` hook has a no-op observer in PR 3b; Phase 4 bloom filter installs the real one.

---

## File Structure

### Files created
- `crates/minkowski-lsm/src/compactor.rs` — `Compactor`, picker, orchestration
- `crates/minkowski-lsm/src/compaction_writer.rs` — `CompactionWriter`: produces output sorted run from input readers
- `crates/minkowski-lsm/src/schema_match.rs` — helper: resolve a schema's archetype across multiple `SortedRunReader`s
- `crates/minkowski-lsm/tests/compaction_integration.rs` — end-to-end: flush multiple times, compact, verify output contents

### Files modified
- `crates/minkowski-lsm/src/lib.rs` — `pub mod compactor`, `pub mod compaction_writer`, re-exports
- `crates/minkowski-lsm/src/writer.rs` — add `FlushWriter::set_entry_observer` hook + `EntryObserver` type
- `crates/minkowski-lsm/src/reader.rs` — may need to expose more internals to compaction_writer (page iteration by (arch_id, slot), entity-slot page access)
- `crates/minkowski/src/world.rs` — `World::compact_one()` + `World::needs_compaction()` API
- `crates/minkowski-lsm/src/manifest_ops.rs` — maybe helper `fn compact_archetype_level<N>(world, manifest, log, archetype, from_level, codecs, run_dir) -> Result<Option<CompactionReport>, LsmError>`

---

## Task 1: `FlushWriter::set_entry_observer` hook (trivial stub)

**Files:**
- Modify: `crates/minkowski-lsm/src/writer.rs`

- [ ] **Step 1: Define the observer type**

At the top of `writer.rs` (after existing imports), add:

```rust
/// Per-entry observer invoked by `FlushWriter` once for each entity
/// written to an entity-slot page. Phase 4 bloom filter uses this to
/// build a per-run filter without re-plumbing the writer. Phase 3
/// installs a no-op observer.
pub type EntryObserver = Box<dyn FnMut(EntityKey) + Send>;

/// The value passed to an `EntryObserver` per successful entity write.
///
/// Currently just the entity ID, since bloom filters are per-archetype
/// and operate on entity identity. If Phase 4 needs archetype context
/// it can be extended — the observer is crate-internal and not a
/// stability boundary yet.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct EntityKey(pub u64);
```

- [ ] **Step 2: Add observer field + setter to `FlushWriter`**

Find the `FlushWriter` struct definition. Add an observer field:

```rust
pub struct FlushWriter<const N: usize = 4> {
    // ...existing fields...
    entry_observer: Option<EntryObserver>,
}
```

Update `FlushWriter::new` (or equivalent constructor) to initialize `entry_observer: None`.

Add a setter method:

```rust
impl<const N: usize> FlushWriter<N> {
    /// Install a per-entry observer. Replaces any previously-installed
    /// observer. Only one observer is supported.
    pub fn set_entry_observer(&mut self, observer: EntryObserver) {
        self.entry_observer = Some(observer);
    }
}
```

- [ ] **Step 3: Invoke the observer in the entity-page write path**

Find the code in `FlushWriter` that writes entity-slot pages (look for `ENTITY_SLOT` or `arch_entities` or similar). Wherever entity IDs are encoded for an entity-slot page, invoke the observer once per entity ID:

```rust
for &entity_bits in entities.iter() {
    // existing: emit entity bytes to page buffer
    if let Some(observer) = self.entry_observer.as_mut() {
        observer(EntityKey(entity_bits));
    }
}
```

Be careful to invoke the observer exactly once per successfully-written entity (not on a dropped page, not twice on retries).

- [ ] **Step 4: Add test**

In `writer.rs` tests, add:

```rust
#[test]
fn entry_observer_fires_once_per_entity_id() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Set up a world with N known entities, flush, count observer calls.
    let dir = tempfile::tempdir().unwrap();
    let mut world = World::new();
    let e1 = world.spawn((/* a known component */,));
    let e2 = world.spawn((/* ditto */,));
    let e3 = world.spawn((/* ditto */,));

    let observed: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));
    let observed_clone = observed.clone();

    let mut writer = FlushWriter::<4>::new(/* ... */);
    writer.set_entry_observer(Box::new(move |key| {
        observed_clone.borrow_mut().push(key.0);
    }));

    // ... flush ...

    let observed = observed.borrow();
    assert_eq!(observed.len(), 3, "observer fired once per entity");
    assert!(observed.contains(&e1.to_bits()));
    assert!(observed.contains(&e2.to_bits()));
    assert!(observed.contains(&e3.to_bits()));
}
```

**NOTE**: `Box<dyn FnMut + Send>` cannot be shared via `Rc<RefCell<_>>` because `Rc` is not `Send`. Use `Arc<Mutex<Vec<u64>>>` instead — or drop the `Send` bound on `EntryObserver` if it's not required. Before writing the test, check whether `FlushWriter` crosses thread boundaries today; if it doesn't, drop `+ Send` from the type alias.

- [ ] **Step 5: Run**

```bash
cargo test -p minkowski-lsm entry_observer
cargo clippy -p minkowski-lsm --all-targets -- -D warnings
cargo fmt --all --check
```

- [ ] **Step 6: Commit**

```bash
git add crates/minkowski-lsm/src/writer.rs
git commit -m "feat(lsm): FlushWriter::set_entry_observer hook

Phase 4 bloom filter will install a real observer; Phase 3 leaves the
hook in place with a no-op default. Invoked once per entity ID written
to an entity-slot page."
```

---

## Task 2: Schema-based archetype match (`schema_match.rs`)

**Files:**
- Create: `crates/minkowski-lsm/src/schema_match.rs`
- Modify: `crates/minkowski-lsm/src/lib.rs` (`pub(crate) mod schema_match;`)

- [ ] **Step 1: Write failing test**

Create the file with just a test first:

```rust
//! Resolving an archetype's identity across multiple sorted runs.
//!
//! Each sorted run assigns its own run-local `arch_id: u16` space. The
//! compactor needs to match archetypes across runs by their component
//! identity (set of stable names), not by arch_id.

use crate::reader::SortedRunReader;

/// Find the `arch_id` within `reader` whose schema matches the given
/// sorted component-name list. Returns `None` if no archetype in the
/// run matches exactly.
pub(crate) fn find_archetype_by_components(
    reader: &SortedRunReader,
    components: &[&str],
) -> Option<u16> {
    // To be implemented.
    let _ = (reader, components);
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_archetype_by_components_returns_none_for_empty_run() {
        // Placeholder until SortedRunReader construction is available in a test harness.
        // This test will be expanded after `find_archetype_by_components` is implemented.
    }
}
```

- [ ] **Step 2: Study the reader's schema section API**

Open `crates/minkowski-lsm/src/schema.rs` and `reader.rs`. Locate:
- How `SortedRunReader` exposes its schema (a `schema()` accessor returning `&SchemaSection`)
- How `SchemaSection` maps slot → component name
- How to enumerate archetypes in a run (likely a `reader.archetype_ids()` method, per the exploration notes)
- How to enumerate which slots a given archetype uses

Write a short note to yourself about the APIs. If any required API is missing (e.g., no way to enumerate an archetype's component slots), implement it in `reader.rs` / `schema.rs` with a minimal addition.

- [ ] **Step 3: Implement `find_archetype_by_components`**

Algorithm:
1. For each `arch_id` in the run:
   - Collect the component stable-names for that archetype (via schema + slot enumeration)
   - Sort the names
   - Compare to the (already-sorted) input `components` slice
   - If equal, return `Some(arch_id)`
2. Return `None`

Concrete implementation depends on what's exposed; the exact code goes in this task. If `SortedRunReader` doesn't expose per-archetype component lists, add an accessor.

- [ ] **Step 4: Write integration test**

In a real test, build two sorted runs from a World with two archetypes (`(Pos,)` and `(Pos, Vel)`). Confirm that `find_archetype_by_components` returns the matching arch_id in each run even when the two runs assign different arch_id values to the same archetype.

This test may require a test helper to build a run from scratch — check if one exists; if not, the existing `flush_and_record` integration tests are a template.

- [ ] **Step 5: Build + commit**

```bash
cargo test -p minkowski-lsm schema_match
cargo clippy -p minkowski-lsm --all-targets -- -D warnings
cargo fmt --all --check
git add crates/minkowski-lsm/src/schema_match.rs crates/minkowski-lsm/src/lib.rs
git commit -m "feat(lsm): schema-based archetype match across sorted runs

Resolves an archetype's identity across multiple sorted runs by matching
on the sorted list of component stable-names in its schema section. Each
run assigns its own run-local arch_id, so arch_id alone is not a stable
identity across runs. Required by the compactor merge kernel."
```

---

## Task 3: `CompactionWriter` — merge kernel

This is the biggest task. Splits into sub-steps. Budget: ~300–500 lines.

**Files:**
- Create: `crates/minkowski-lsm/src/compaction_writer.rs`
- Modify: `crates/minkowski-lsm/src/lib.rs`
- Modify: `crates/minkowski-lsm/src/reader.rs` if internal helpers need to be exposed

**Algorithm (ledger-shape, no tombstones):**

1. **Inputs**: `Vec<&SortedRunReader>` (all at level L, sorted by `sequence_range().hi()` descending — newest first), target archetype's component-name list.
2. **Resolve per-input arch_ids** via `find_archetype_by_components`. Skip inputs that don't have this archetype (rare but possible if some runs flushed before the archetype existed).
3. **Build emit list**: a `Vec<(entity_id: u64, source_input_idx: usize, source_row: usize)>`. Iterate inputs newest-first; for each entity-slot page, iterate entity IDs; emit `(entity_id, input_idx, row)` only if entity_id not already in a `HashSet<u64>` seen set.
4. **Sort emit list** by entity_id (optional — keeps output deterministic; matches TigerBeetle/RocksDB convention of sorted output).
5. **Compute output dimensions**:
   - `row_count = emit_list.len()`
   - `page_count = ceil(row_count / PAGE_SIZE)`
   - `output_seq_range = (min(input_seq_lo), max(input_seq_hi))`
   - `output_size_bytes = sum of component page sizes` (estimated; finalized after write)
6. **Open output file** via `tmp + rename` pattern (reuse `FlushWriter`'s atomicity helpers).
7. **Write header** with computed dimensions.
8. **Write pages** in (arch_id, slot, page_index) order:
   - For each output `page_index`, batch of `PAGE_SIZE` entities:
     - For each component slot (from the schema):
       - For each entity in the batch: `memcpy` the component bytes from source_input's matching page at source_row
     - Compute page CRC, write `PageHeader` + page data
     - Record `IndexEntry`
   - Write entity-slot page for this page_index: entity IDs in row order
9. **Write sparse index** (sorted IndexEntry array).
10. **Write schema section** — copy from any input (all inputs share the schema by definition).
11. **Write footer**.
12. **fsync, rename, dir-fsync** for atomicity.
13. **Invoke entry observer** for each entity written (so Phase 4 bloom gets fed).
14. **Return** `SortedRunMeta` for the output file so the caller can issue the `CompactionCommit`.

**API**:
```rust
pub struct CompactionWriter<'a> {
    inputs: Vec<&'a SortedRunReader>,
    target_components: Vec<String>,  // sorted stable names
    output_path: PathBuf,
    codecs: &'a CodecRegistry,
    entry_observer: Option<EntryObserver>,
}

impl<'a> CompactionWriter<'a> {
    pub fn new(
        inputs: Vec<&'a SortedRunReader>,
        target_components: Vec<String>,
        output_path: PathBuf,
        codecs: &'a CodecRegistry,
    ) -> Self { /* ... */ }

    pub fn set_entry_observer(&mut self, observer: EntryObserver) { /* ... */ }

    pub fn write(self) -> Result<SortedRunMeta, LsmError>;
}
```

### Sub-tasks (commit each one separately):

- [ ] **3a**: Implement step 3 (emit-list construction with dedup via HashSet) as an isolated `fn build_emit_list(...) -> Vec<EmitRow>` with unit tests.
- [ ] **3b**: Implement steps 6–11 (the write loop) — this needs the emit list to work first. Write unit tests against fabricated emit lists.
- [ ] **3c**: Wire sub-tasks 3a + 3b into `CompactionWriter::write`. Add an end-to-end test: create two tiny sorted runs, run CompactionWriter, open the result with `SortedRunReader::open`, verify contents.
- [ ] **3d**: Thread the entry observer through the entity-page write path in CompactionWriter (mirrors Task 1's hook on FlushWriter).

Each sub-task is a commit.

---

## Task 4: `Compactor` module — orchestration + picker

**Files:**
- Create: `crates/minkowski-lsm/src/compactor.rs`
- Modify: `crates/minkowski-lsm/src/lib.rs`

**Responsibilities:**
1. Picker: `fn find_compaction_candidate<const N: usize>(manifest: &LsmManifest<N>) -> Option<CompactionJob>` — iterates (level, archetype_components) pairs, returns the first one with ≥ K runs.
2. Executor: `fn execute<const N: usize>(job: CompactionJob, manifest: &mut LsmManifest<N>, log: &mut ManifestLog, codecs: &CodecRegistry, run_dir: &Path) -> Result<CompactionReport, LsmError>`:
   - Resolve input readers from the manifest + file paths
   - Build output file path (tmp + final, atomic rename)
   - Invoke `CompactionWriter::write`
   - Emit `ManifestEntry::CompactionCommit { output_level, output, inputs }` via `log.append`
   - Apply the entry to the in-memory manifest
   - (File deletion of inputs is handled by `cleanup_orphans`, out of scope here)

**Constant:** `pub(crate) const COMPACTION_TRIGGER: usize = 4;` — the K=4 from the design spec.

- [ ] Implement `CompactionJob`, `CompactionReport` structs
- [ ] Implement `find_compaction_candidate` with stable iteration
- [ ] Implement `execute`
- [ ] Unit tests: picker with 0, 1, K-1, K, K+1 runs at a level
- [ ] Integration test: seed manifest + files, execute, verify output run exists + manifest updated
- [ ] Commit

---

## Task 5: `World::compact_one()` + `needs_compaction()` public API

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Challenge**: `World` lives in the `minkowski` crate, while the compactor lives in `minkowski-lsm`. `minkowski` does NOT depend on `minkowski-lsm`. Adding that dependency reverses the current layering.

**Options:**
- **A**: Add `World::compact_one()` in `minkowski-lsm`, not `World` itself. E.g., a free function `pub fn compact_one<const N: usize>(world: &mut World, manifest: &mut LsmManifest<N>, log: &mut ManifestLog, codecs: &CodecRegistry, run_dir: &Path) -> Result<Option<CompactionReport>, LsmError>`. User code calls this from outside.
- **B**: Add a trait `Compactable` in `minkowski` and an impl in `minkowski-lsm`. Viral.
- **C**: Add the API to `minkowski-lsm`, not `World`. Name it `lsm::compact_one` or similar.

Design spec says `World::compact_one()` but that presupposes `minkowski` knows about `LsmManifest`. Given the current crate layering (`minkowski` is the ECS core, `minkowski-lsm` depends on it), **Option A or C is correct**. The design spec should be amended.

- [ ] **Decision point**: choose A or C. Update the design spec to reflect.
- [ ] Implement as a free function in `minkowski-lsm` (recommend: `minkowski_lsm::compactor::compact_one(...)`).
- [ ] Add `needs_compaction` as a method on `LsmManifest<N>` (it's a manifest query; doesn't need World).
- [ ] Commit

---

## Task 6: Integration tests + PR

**Files:**
- Create: `crates/minkowski-lsm/tests/compaction_integration.rs`
- Tests:
  - `flush_four_times_then_compact_consolidates_l0_to_l1`: seed World, flush 4 times into L0, assert 4 runs, call `compact_one`, assert 1 run at L1 + 0 at L0.
  - `compact_preserves_all_entities`: flush multiple batches of different entities, compact, open output run, verify all entities present.
  - `compact_with_entity_updates_keeps_newest_version`: flush entity E with value A, modify E to value B, flush again, compact, verify value B wins.
  - `compact_emits_compaction_commit_atomic`: verify the manifest log has exactly one CompactionCommit entry after a compact call.
  - `needs_compaction_returns_true_when_any_level_over_threshold`.
  - `compact_one_returns_none_when_nothing_over_threshold`.

- [ ] Write tests
- [ ] Run full test suite + clippy + fmt
- [ ] Push branch, open PR
- [ ] Wait for CI green
- [ ] Run `/pr-review-toolkit:review-pr` — this is a large PR, expect review iterations

---

## Known Risks

- **Tombstone gap**: ledger-shape merge is correct for append-only workloads. Any workload that despawns entities will see resurrections on recovery. This is a pre-existing limitation of the on-disk format, not introduced by this PR. Document in the PR body.

- **Entity update semantics**: "update" in minkowski's ECS is modifying a component's value on an existing entity. The merge takes the newest sequence version. This assumes the dirty-page flush captures each update as a new page write, which is the current behavior per PR #160's `DirtyPageTracker`.

- **Cross-run archetype translation via component names**: relies on `SchemaSection` being populated in every run. Verify this in Task 2 — if some runs have incomplete schema sections, the match will spuriously fail.

- **Output file size estimation**: `output_size_bytes` field in `SortedRunMeta` has to be known before the metadata is finalized. Either compute after write (preferred — exact) or estimate and correct. Task 3b's design should accommodate.

- **`FlushWriter` refactor risk**: Task 1's observer hook threads through the flush path. If the flush path has subtle state that the observer invocation could disrupt (e.g., error rollback after observer fires), catch it in code review.

- **Large PR**: estimated 800–1200 lines across 6 tasks. Review will take multiple iterations. Split into two PRs (3b-writer + 3b-compactor) if reviewers push back.
