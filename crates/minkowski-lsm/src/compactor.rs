//! Compaction orchestration: candidate picker and job executor.
//!
//! # Trigger
//!
//! A compaction job is triggered when any `(level, archetype)` pair
//! accumulates at least [`COMPACTION_TRIGGER`] sorted runs. The picker
//! iterates levels and archetypes in a stable order so the choice is
//! deterministic across repeated calls.
//!
//! # Atomicity
//!
//! [`execute_compaction`] appends a single [`ManifestEntry::CompactionCommit`]
//! frame before touching the in-memory manifest. The CRC-protected frame is
//! the commit point: if the process crashes after append but before the
//! in-memory update, recovery replays the frame and converges to the same
//! state.
//!
//! # Bottom level
//!
//! Runs at level N-1 (the bottom of an N-level manifest) are never compacted
//! upward — there is no L(N) to promote them to. The bottom level accumulates
//! indefinitely in the ledger-shape model. Long-term, in-place merge at the
//! bottom level is out of scope for this PR.

use std::path::{Path, PathBuf};

use crate::compaction_writer::CompactionWriter;
use crate::error::LsmError;
use crate::manifest::{LsmManifest, SortedRunMeta};
use crate::manifest_log::{ManifestEntry, ManifestLog};
use crate::reader::SortedRunReader;
use crate::schema_match::find_archetype_by_components;
use crate::types::{Level, SeqNo, SeqRange};
use crate::writer::EntityKey;

// ── Constants ─────────────────────────────────────────────────────────────────

/// K=4 count trigger: if any (level, archetype) has at least this many runs,
/// compact them.
pub const COMPACTION_TRIGGER: usize = 4;

// ── Public types ──────────────────────────────────────────────────────────────

/// A compaction job picked by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactionJob {
    /// The source level — all inputs live here.
    pub from_level: Level,
    /// The destination level — output lands here.
    pub to_level: Level,
    /// The archetype's component stable-names (sorted) — used to identify the
    /// archetype across input runs whose run-local arch_ids may differ.
    pub target_components: Vec<String>,
    /// The full set of input run paths at `from_level` that compact together.
    /// Passed to `CompactionCommit` as the atomic inputs list.
    pub input_paths: Vec<PathBuf>,
}

/// Outcome of a compaction execution.
#[derive(Debug, Clone)]
pub struct CompactionReport {
    pub from_level: Level,
    pub to_level: Level,
    pub input_run_count: usize,
    pub output_path: PathBuf,
    pub output_bytes: u64,
}

// ── Picker ────────────────────────────────────────────────────────────────────

/// Count trigger: find the first `(level, archetype)` pair with >= K runs.
///
/// Iterates levels 0..N-1 in ascending order. For each level, iterates runs
/// in manifest order (stable). Returns `None` when no pair meets the trigger.
///
/// The bottom level (L(N-1)) is excluded: there is no L(N) to promote runs
/// into.
///
/// # Errors
///
/// Returns `LsmError` if a sorted run file cannot be opened to discover its
/// archetype component names.
pub(crate) fn find_compaction_candidate<const N: usize>(
    manifest: &LsmManifest<N>,
) -> Result<Option<CompactionJob>, LsmError> {
    // We need to iterate levels 0..N-1 (excluding bottom level N-1).
    // For each level, group runs by their sorted component-name signature.
    // Use an ordered map to preserve stable iteration within a level.

    let bottom_level = N.saturating_sub(1);

    for level_idx in 0..N {
        if level_idx == bottom_level {
            // Cannot compact bottom level upward — skip.
            continue;
        }
        let level = Level::new(level_idx as u8).ok_or_else(|| {
            LsmError::Format(format!("level index {level_idx} out of MAX_LEVELS range"))
        })?;

        let runs = manifest.runs_at_level(level);
        if runs.len() < COMPACTION_TRIGGER {
            // Fast path: not enough runs at this level to trigger even if all
            // belong to the same archetype.
            continue;
        }

        // Group runs by their sorted component-name signature.
        // We open each reader to discover the archetype. Use an insertion-order
        // preserving structure: Vec<(signature, Vec<meta>)>.
        let mut groups: Vec<(Vec<String>, Vec<&SortedRunMeta>)> = Vec::new();

        for meta in runs {
            let reader = SortedRunReader::open(meta.path())?;
            // A run may contain multiple archetypes. We pick the first archetype
            // that matches a group already present, or start a new group.
            // For the compactor's purposes, each "job" targets ONE archetype
            // across multiple runs. A run with multiple archetypes contributes
            // to multiple potential jobs — we do one at a time.

            // Collect all archetypes in this run as sorted-name signatures.
            let arch_ids = reader.archetype_ids();
            for arch_id in arch_ids {
                let slots = reader.component_slots_for_arch(arch_id);
                let mut comp_names: Vec<String> = slots
                    .iter()
                    .filter_map(|&slot| {
                        reader
                            .schema()
                            .entry_for_slot(slot)
                            .map(|e| e.name().to_owned())
                    })
                    .collect();
                comp_names.sort_unstable();

                if comp_names.is_empty() {
                    continue;
                }

                // Find or create the group for this signature.
                if let Some(group) = groups.iter_mut().find(|(sig, _)| *sig == comp_names) {
                    group.1.push(meta);
                } else {
                    groups.push((comp_names, vec![meta]));
                }
            }
        }

        // Find the first group with >= COMPACTION_TRIGGER runs.
        for (comp_names, group_runs) in groups {
            if group_runs.len() < COMPACTION_TRIGGER {
                continue;
            }

            // Found a candidate. Build the job.
            let to_level_idx = level_idx + 1;
            let to_level = Level::new(to_level_idx as u8).ok_or_else(|| {
                LsmError::Format(format!(
                    "cannot compact from level {level_idx}: destination L{to_level_idx} \
                     exceeds MAX_LEVELS"
                ))
            })?;

            let input_paths: Vec<PathBuf> =
                group_runs.iter().map(|m| m.path().to_path_buf()).collect();

            return Ok(Some(CompactionJob {
                from_level: level,
                to_level,
                target_components: comp_names,
                input_paths,
            }));
        }
    }

    Ok(None)
}

// ── Executor ──────────────────────────────────────────────────────────────────

/// Execute one compaction job end-to-end.
///
/// Thin wrapper around [`execute_compaction_observed`] with no observer.
/// See that function for the full step-by-step contract.
// Tests call this directly; no production caller yet so the lib-only
// dead_code lint fires. #[expect] would be unfulfilled under --all-targets.
#[allow(dead_code)]
pub(crate) fn execute_compaction<const N: usize>(
    job: &CompactionJob,
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    run_dir: &Path,
) -> Result<CompactionReport, LsmError> {
    execute_compaction_observed(job, manifest, log, run_dir, None)
}

/// Execute one compaction job end-to-end, invoking `observer` once per entity
/// ID written to the output entity-slot pages.
///
/// 1. Opens each input as a [`SortedRunReader`].
/// 2. Resolves per-input arch_ids via [`find_archetype_by_components`].
/// 3. Computes the output sequence range.
/// 4. Generates a unique output path.
/// 5. Runs [`CompactionWriter::write_observed`] to produce the output file.
/// 6. Appends [`ManifestEntry::CompactionCommit`] to the log (atomic commit point).
/// 7. Mirrors the commit on the in-memory manifest.
/// 8. Returns [`CompactionReport`].
///
/// If any step before the log append fails, no manifest state is mutated.
/// If the log append succeeds, the in-memory mutation MUST also succeed
/// (all pre-conditions are checked before the append).
pub(crate) fn execute_compaction_observed<const N: usize>(
    job: &CompactionJob,
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    run_dir: &Path,
    observer: Option<&mut dyn FnMut(EntityKey)>,
) -> Result<CompactionReport, LsmError> {
    // ── 1. Open input readers ─────────────────────────────────────────────────
    let readers: Vec<SortedRunReader> = job
        .input_paths
        .iter()
        .map(|p| SortedRunReader::open(p))
        .collect::<Result<Vec<_>, _>>()?;

    // ── 2. Resolve per-input arch_ids ─────────────────────────────────────────
    let comp_name_strs: Vec<&str> = job.target_components.iter().map(String::as_str).collect();
    let arch_ids_per_input: Vec<Option<u16>> = readers
        .iter()
        .map(|r| find_archetype_by_components(r, &comp_name_strs))
        .collect();

    // Collect refs for CompactionWriter (borrows readers).
    let reader_refs: Vec<&SortedRunReader> = readers.iter().collect();

    // ── 3. Compute output sequence range ──────────────────────────────────────
    // min(lo) across inputs, max(hi) across inputs.
    let seq_lo = readers
        .iter()
        .map(|r| r.sequence_range().lo())
        .min()
        .ok_or_else(|| LsmError::Format("compaction job has no input readers".to_owned()))?;
    let seq_hi = readers
        .iter()
        .map(|r| r.sequence_range().hi())
        .max()
        .ok_or_else(|| LsmError::Format("compaction job has no input readers".to_owned()))?;

    let output_seq_range = SeqRange::new(seq_lo, seq_hi)?;

    // ── 4. Generate output path ───────────────────────────────────────────────
    // Mirror the naming convention from writer.rs: `{lo}-{hi}.run`.
    // Add a `.compact` infix to distinguish from flush-output files, which
    // also use `{lo}-{hi}.run`. Without this, two runs at different levels
    // with the same sequence span would collide on disk.
    let output_path = make_output_path(run_dir, seq_lo, seq_hi);

    // ── 5. Run CompactionWriter ────────────────────────────────────────────────
    // Inputs are passed newest-first (highest hi sequence = newest). Sort
    // descending by hi so the emit-list dedup picks the right version.
    let mut indexed: Vec<(usize, &SortedRunReader)> =
        reader_refs.iter().copied().enumerate().collect();
    indexed.sort_by_key(|(_, r)| std::cmp::Reverse(r.sequence_range().hi()));
    let sorted_reader_refs: Vec<&SortedRunReader> = indexed.iter().map(|(_, r)| *r).collect();
    let sorted_arch_ids: Vec<Option<u16>> = indexed
        .iter()
        .map(|(i, _)| arch_ids_per_input[*i])
        .collect();

    let writer = CompactionWriter::new(
        sorted_reader_refs,
        sorted_arch_ids,
        job.target_components.clone(),
        output_path.clone(),
        output_seq_range,
    )?;

    let output_meta: SortedRunMeta = writer.write_observed(observer)?;

    // ── 6. Pre-validate in-memory state before the atomic commit point ────────
    // Verify the output level is in-range for this manifest *before* the log
    // append. If we appended and then discovered the level was OOR, the log
    // would be ahead of what apply_entry can accept — a log-vs-manifest split.
    if job.to_level.as_index() >= N {
        return Err(LsmError::Format(format!(
            "execute_compaction: to_level {} out of range for {N}-level manifest",
            job.to_level
        )));
    }

    // Build the CompactionCommit entry. All inputs must exist in the manifest.
    // apply_entry pre-validates this too, but we check early so the output
    // file can be cleaned up if the manifest state is wrong.
    for input_path in &job.input_paths {
        let exists = manifest
            .runs_at_level(job.from_level)
            .iter()
            .any(|m| m.path() == input_path.as_path());
        if !exists {
            return Err(LsmError::Format(format!(
                "execute_compaction: input run {} not found at level {} in manifest",
                input_path.display(),
                job.from_level
            )));
        }
    }

    let inputs_for_entry: Vec<(Level, PathBuf)> = job
        .input_paths
        .iter()
        .map(|p| (job.from_level, p.clone()))
        .collect();

    let entry = ManifestEntry::CompactionCommit {
        output_level: job.to_level,
        output: output_meta.clone(),
        inputs: inputs_for_entry,
    };

    // ── 7. Log append — atomic commit point ──────────────────────────────────
    // From this point on, all mutations MUST succeed. All fallible checks are
    // done above. The log append is the commit point (mirrors flush_and_record).
    log.append(&entry)?;

    // ── 8. Mirror on in-memory manifest ───────────────────────────────────────
    // add_run first so the output exists, then remove inputs.
    // This mirrors apply_entry's order in manifest_log.rs.
    manifest
        .add_run(job.to_level, output_meta.clone())
        .expect("to_level pre-validated < N");
    for input_path in &job.input_paths {
        manifest
            .remove_run(job.from_level, input_path)
            .unwrap_or_else(|| {
                panic!(
                    "pre-validated compaction input vanished: {} at level {}",
                    input_path.display(),
                    job.from_level
                )
            });
    }

    Ok(CompactionReport {
        from_level: job.from_level,
        to_level: job.to_level,
        input_run_count: job.input_paths.len(),
        output_path: output_meta.path().to_path_buf(),
        output_bytes: output_meta.size_bytes().get(),
    })
}

// ── Public one-shot API ───────────────────────────────────────────────────────

/// One-shot compaction driver. Picks a candidate via
/// [`find_compaction_candidate`], executes it if found, returns the
/// report or `None`. Callers loop:
///
/// ```ignore
/// while manifest.needs_compaction() {
///     compactor::compact_one(&mut manifest, &mut log, &run_dir)?;
/// }
/// ```
///
/// Returns `Ok(None)` when no `(level, archetype)` pair has
/// `>= COMPACTION_TRIGGER` runs.
pub fn compact_one<const N: usize>(
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    run_dir: &Path,
) -> Result<Option<CompactionReport>, LsmError> {
    compact_one_observed(manifest, log, run_dir, None)
}

/// Observer-accepting variant of [`compact_one`] for Phase 4 bloom integration.
///
/// `observer` is invoked once per entity ID written to the output run's
/// entity-slot pages. Pass `None` for no observation (identical to
/// [`compact_one`]).
pub fn compact_one_observed<const N: usize>(
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    run_dir: &Path,
    observer: Option<&mut dyn FnMut(EntityKey)>,
) -> Result<Option<CompactionReport>, LsmError> {
    let Some(job) = find_compaction_candidate(manifest)? else {
        return Ok(None);
    };
    let report = execute_compaction_observed(&job, manifest, log, run_dir, observer)?;
    Ok(Some(report))
}

// ── File helpers ──────────────────────────────────────────────────────────────

/// Build an output path for a compacted run file.
///
/// Uses a `.compact.run` extension to distinguish compacted outputs from
/// flush outputs (`{lo}-{hi}.run`). Both share the same directory, so
/// the extension difference prevents collisions when the sequence spans overlap.
fn make_output_path(run_dir: &Path, seq_lo: SeqNo, seq_hi: SeqNo) -> PathBuf {
    run_dir.join(format!("{}-{}.compact.run", seq_lo.get(), seq_hi.get()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest_log::ManifestLog;
    use crate::manifest_ops::flush_and_record;
    use crate::types::{SeqNo, SeqRange};
    use minkowski::World;

    // ── Component types ──────────────────────────────────────────────────────

    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Flush a fresh world with N_ENTITIES Pos entities as L0 run `seq_no`.
    /// Returns the path of the written run.
    fn do_flush<const N: usize>(
        manifest: &mut LsmManifest<N>,
        log: &mut ManifestLog,
        run_dir: &Path,
        seq_lo: u64,
        seq_hi: u64,
        n_entities: usize,
    ) -> PathBuf {
        let mut world = World::new();
        for i in 0..n_entities {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }
        let seq_range = SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi)).unwrap();
        flush_and_record(&world, seq_range, manifest, log, run_dir)
            .unwrap()
            .expect("world is dirty, flush must return Some")
    }

    // ── Test 1: empty manifest → None ────────────────────────────────────────

    #[test]
    fn find_compaction_candidate_returns_none_when_no_level_over_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Empty manifest.
        assert!(find_compaction_candidate(&manifest).unwrap().is_none());

        // 3 runs at L0 — below K=4.
        for i in 0..3 {
            let lo = i * 10;
            let hi = lo + 9;
            do_flush(&mut manifest, &mut log, dir.path(), lo, hi, 2);
        }
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 3);
        assert!(find_compaction_candidate(&manifest).unwrap().is_none());
    }

    // ── Test 2: exactly K runs at L0 → Some ──────────────────────────────────

    #[test]
    fn find_compaction_candidate_returns_job_at_trigger_count() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Spawn K=4 runs at L0, each with distinct sequence ranges.
        for i in 0..4u64 {
            let lo = i * 10;
            let hi = lo + 9;
            do_flush(&mut manifest, &mut log, dir.path(), lo, hi, 2);
        }
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 4);

        let job = find_compaction_candidate(&manifest)
            .unwrap()
            .expect("should find a candidate with K=4 runs");

        assert_eq!(job.from_level, Level::L0);
        assert_eq!(job.to_level, Level::L1);
        assert_eq!(job.input_paths.len(), 4);
        assert!(
            !job.target_components.is_empty(),
            "must carry component names"
        );
    }

    // ── Test 3: stable iteration — L0 wins over L1 ───────────────────────────

    #[test]
    fn find_compaction_candidate_stable_iteration() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Seed 4 runs at L0.
        for i in 0..4u64 {
            do_flush(&mut manifest, &mut log, dir.path(), i * 10, i * 10 + 9, 2);
        }

        // Manually promote 4 of those to L1 — but we can't promote paths
        // that were flushed to the run_dir because the manifest stores them.
        // Instead, directly add 4 synthetic L1 meta entries pointing to files
        // that would already exist on disk (or just add metadata without files
        // — the picker doesn't open L1 files in this path because L0 triggers
        // first).
        //
        // Actually: we flush 4 *more* worlds as L0, then promote them to L1
        // via manifest.add_run so both levels have K entries. L0 must still
        // win because it is iterated first.
        let dir2 = tempfile::tempdir().unwrap();
        let log2_path = dir2.path().join("manifest2.log");
        let (mut manifest2, mut log2) = ManifestLog::recover::<4>(&log2_path).unwrap();

        // 4 runs at L0.
        for i in 0..4u64 {
            do_flush(
                &mut manifest2,
                &mut log2,
                dir2.path(),
                i * 10,
                i * 10 + 9,
                2,
            );
        }

        // Copy the L0 run metas to L1 in a second manifest to simulate both
        // levels being over the trigger. The picker must return L0.
        let l0_runs: Vec<_> = manifest2.runs_at_level(Level::L0).to_vec();
        for meta in l0_runs {
            manifest2.add_run(Level::L1, meta).unwrap();
        }
        assert_eq!(manifest2.runs_at_level(Level::L0).len(), 4);
        assert_eq!(manifest2.runs_at_level(Level::L1).len(), 4);

        let job = find_compaction_candidate(&manifest2)
            .unwrap()
            .expect("must find a candidate");
        assert_eq!(job.from_level, Level::L0, "L0 must be chosen over L1");
    }

    // ── Test 4: 4 runs at bottom level (L3 of N=4) → None ────────────────────

    #[test]
    fn find_compaction_candidate_skips_bottom_level() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Flush 4 runs at L0, then move them to L3 (bottom of N=4).
        for i in 0..4u64 {
            do_flush(&mut manifest, &mut log, dir.path(), i * 10, i * 10 + 9, 2);
        }
        let l0_metas: Vec<_> = manifest.runs_at_level(Level::L0).to_vec();
        for meta in &l0_metas {
            manifest.remove_run(Level::L0, meta.path());
            manifest.add_run(Level::L3, meta.clone()).unwrap();
        }

        assert_eq!(manifest.runs_at_level(Level::L3).len(), 4);
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 0);

        let result = find_compaction_candidate(&manifest).unwrap();
        assert!(
            result.is_none(),
            "bottom level (L3 of N=4) must not be compacted upward"
        );
    }

    // ── Test 5: execute_compaction end-to-end ────────────────────────────────

    #[test]
    fn execute_compaction_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Flush 4 runs at L0 with distinct entities (different worlds).
        // Each world seeds 3 entities starting at a distinct index so entity
        // IDs don't overlap across worlds.
        let n_runs = 4usize;
        let entities_per_run = 3usize;

        for run_i in 0..n_runs {
            let mut world = World::new();
            // Advance entity allocator so each world produces distinct IDs.
            let skip = run_i * entities_per_run;
            let placeholders: Vec<_> = (0..skip)
                .map(|_| world.spawn((Pos { x: 0.0, y: 0.0 },)))
                .collect();
            for ph in placeholders {
                world.despawn(ph);
            }
            for j in 0..entities_per_run {
                world.spawn((Pos {
                    x: (run_i * 10 + j) as f32,
                    y: 0.0,
                },));
            }
            let lo = (run_i as u64) * 10;
            let hi = lo + 9;
            let seq_range = SeqRange::new(SeqNo::from(lo), SeqNo::from(hi)).unwrap();
            flush_and_record(&world, seq_range, &mut manifest, &mut log, dir.path())
                .unwrap()
                .expect("world is dirty");
        }

        assert_eq!(manifest.runs_at_level(Level::L0).len(), n_runs);

        // Pick and execute the compaction job.
        let job = find_compaction_candidate(&manifest)
            .unwrap()
            .expect("K=4 runs must trigger compaction");

        let report = execute_compaction(&job, &mut manifest, &mut log, dir.path()).unwrap();

        // ── Verify report ─────────────────────────────────────────────────────
        assert_eq!(report.from_level, Level::L0);
        assert_eq!(report.to_level, Level::L1);
        assert_eq!(report.input_run_count, n_runs);
        assert!(report.output_bytes > 0);

        // ── Verify in-memory manifest state ───────────────────────────────────
        assert!(
            manifest.runs_at_level(Level::L0).is_empty(),
            "all L0 inputs must be removed"
        );
        assert_eq!(
            manifest.runs_at_level(Level::L1).len(),
            1,
            "one L1 output must be added"
        );

        // ── Verify output file exists and is readable ─────────────────────────
        let output_path = &report.output_path;
        assert!(
            output_path.exists(),
            "output run file must exist on disk: {output_path:?}"
        );
        let out_reader = SortedRunReader::open(output_path).unwrap();
        // Total entities = n_runs × entities_per_run.
        let expected_entity_count = n_runs * entities_per_run;

        let mut found: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut page_idx: u16 = 0;
        use crate::format::ENTITY_SLOT;
        while let Ok(Some(page)) = out_reader.get_page(0, ENTITY_SLOT, page_idx) {
            let row_count = page.header().row_count as usize;
            let data = page.data();
            for r in 0..row_count {
                let off = r * 8;
                let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                found.insert(id);
            }
            page_idx += 1;
        }
        assert_eq!(
            found.len(),
            expected_entity_count,
            "output must contain all entities from all input runs"
        );

        // ── Verify CompactionCommit is in the log ──────────────────────────────
        // Re-open the log and verify the recovered manifest matches the current
        // in-memory state: L0 empty, L1 has one run at the output path.
        drop(log); // close the file
        let (recovered_manifest, _) = ManifestLog::recover::<4>(&log_path).unwrap();
        assert!(
            recovered_manifest.runs_at_level(Level::L0).is_empty(),
            "recovered L0 must be empty"
        );
        assert_eq!(
            recovered_manifest.runs_at_level(Level::L1).len(),
            1,
            "recovered manifest must have one L1 run"
        );
        assert_eq!(
            recovered_manifest.runs_at_level(Level::L1)[0].path(),
            output_path.as_path(),
            "recovered L1 run path must match output"
        );
    }

    // ── Test 6: compact_one returns None on empty manifest ───────────────────

    #[test]
    fn compact_one_returns_none_when_nothing_over_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Empty manifest — nothing to compact.
        let result = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
        assert!(result.is_none(), "empty manifest must return None");

        // 3 runs at L0 — below K=4.
        for i in 0..3u64 {
            do_flush(&mut manifest, &mut log, dir.path(), i * 10, i * 10 + 9, 2);
        }
        let result = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
        assert!(result.is_none(), "3 runs below K=4 must return None");

        // Manifest must be unchanged (3 L0 runs, 0 L1 runs).
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 3);
        assert_eq!(manifest.runs_at_level(Level::L1).len(), 0);
    }

    // ── Test 7: compact_one drives one job and returns Some(report) ──────────

    #[test]
    fn compact_one_drives_one_job_and_returns_report() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Seed 4 runs at L0 — exactly the trigger.
        for i in 0..4u64 {
            do_flush(&mut manifest, &mut log, dir.path(), i * 10, i * 10 + 9, 2);
        }

        let result = compact_one(&mut manifest, &mut log, dir.path())
            .unwrap()
            .expect("K=4 runs must produce Some(report)");

        assert_eq!(result.from_level, Level::L0);
        assert_eq!(result.to_level, Level::L1);
        assert_eq!(result.input_run_count, 4);
        assert!(result.output_bytes > 0);

        // L0 must be drained; L1 has one run.
        assert!(manifest.runs_at_level(Level::L0).is_empty());
        assert_eq!(manifest.runs_at_level(Level::L1).len(), 1);
    }

    // ── Test 8: driven loop drains all candidates ─────────────────────────────
    //
    // The picker collects ALL runs in a (level, archetype) group — not just K.
    // With 8 homogeneous L0 runs, the first compact_one call compacts all 8
    // into one L1 run. A second call then finds nothing over the trigger
    // (L1 has 1 run, L0 is empty) and returns None.
    //
    // To get *two* compaction rounds, we use two separate archetype signatures
    // at L0: 4 runs of (Pos,) and 4 runs of (Vel,). The picker triggers on
    // the first group it finds; the second call triggers on the second group.

    #[test]
    fn compact_one_driven_loop_drains_all_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // 4 runs of (Pos,) at L0.
        for i in 0..4u64 {
            do_flush(&mut manifest, &mut log, dir.path(), i * 10, i * 10 + 9, 2);
        }

        // 4 runs of (Vel,) at L0 — different archetype signature.
        for i in 4..8u64 {
            let mut world = World::new();
            for j in 0..2usize {
                world.spawn((Vel {
                    dx: (i * 10 + j as u64) as f32,
                    dy: 0.0,
                },));
            }
            let seq_range = SeqRange::new(SeqNo::from(i * 10), SeqNo::from(i * 10 + 9)).unwrap();
            flush_and_record(&world, seq_range, &mut manifest, &mut log, dir.path())
                .unwrap()
                .expect("world is dirty");
        }

        assert_eq!(manifest.runs_at_level(Level::L0).len(), 8);

        // First call: compacts one archetype group (Pos group, 4 runs) → L1.
        let r1 = compact_one(&mut manifest, &mut log, dir.path())
            .unwrap()
            .expect("first call must produce Some");
        assert_eq!(r1.from_level, Level::L0);
        assert_eq!(r1.input_run_count, 4);

        // After first: 4 L0 remain (Vel group), 1 L1.
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 4);
        assert_eq!(manifest.runs_at_level(Level::L1).len(), 1);

        // Second call: compacts the remaining 4 L0 runs (Vel group) → second L1 run.
        let r2 = compact_one(&mut manifest, &mut log, dir.path())
            .unwrap()
            .expect("second call must produce Some");
        assert_eq!(r2.from_level, Level::L0);
        assert_eq!(r2.input_run_count, 4);

        // After second: 0 L0, 2 L1.
        assert!(manifest.runs_at_level(Level::L0).is_empty());
        assert_eq!(manifest.runs_at_level(Level::L1).len(), 2);

        // Third call: nothing left at L0; L1 has 2 < 4.
        let r3 = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
        assert!(
            r3.is_none(),
            "third call must return None — nothing left to compact"
        );
    }

    // ── Test 9: needs_compaction parity with find_compaction_candidate ────────

    #[test]
    fn needs_compaction_matches_find_compaction_candidate() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Empty — both must agree: no compaction needed.
        let candidate = find_compaction_candidate(&manifest).unwrap();
        assert_eq!(
            manifest.needs_compaction(),
            candidate.is_some(),
            "needs_compaction must match find_compaction_candidate (empty)"
        );

        // Add 3 runs — still below trigger.
        for i in 0..3u64 {
            do_flush(&mut manifest, &mut log, dir.path(), i * 10, i * 10 + 9, 2);
        }
        let candidate = find_compaction_candidate(&manifest).unwrap();
        assert_eq!(
            manifest.needs_compaction(),
            candidate.is_some(),
            "needs_compaction must match find_compaction_candidate (3 runs)"
        );

        // Add a 4th run — now at trigger.
        do_flush(&mut manifest, &mut log, dir.path(), 30, 39, 2);
        let candidate = find_compaction_candidate(&manifest).unwrap();
        assert_eq!(
            manifest.needs_compaction(),
            candidate.is_some(),
            "needs_compaction must match find_compaction_candidate (4 runs)"
        );
        assert!(manifest.needs_compaction(), "must be true at K=4");
    }

    // ── Test 10: compact_one_observed fires for every output entity ───────────

    #[test]
    fn compact_one_observed_fires_for_every_output_entity() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Seed 4 runs, each with 3 distinct entities (advance allocator per run).
        let entities_per_run = 3usize;
        let n_runs = 4usize;
        let mut all_entity_bits: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for run_i in 0..n_runs {
            let mut world = World::new();
            let skip = run_i * entities_per_run;
            let phs: Vec<_> = (0..skip)
                .map(|_| world.spawn((Pos { x: 0.0, y: 0.0 },)))
                .collect();
            for ph in phs {
                world.despawn(ph);
            }
            for j in 0..entities_per_run {
                let e = world.spawn((Pos {
                    x: (run_i * 10 + j) as f32,
                    y: 0.0,
                },));
                all_entity_bits.insert(e.to_bits());
            }
            let lo = (run_i as u64) * 10;
            let hi = lo + 9;
            let seq_range = SeqRange::new(SeqNo::from(lo), SeqNo::from(hi)).unwrap();
            flush_and_record(&world, seq_range, &mut manifest, &mut log, dir.path())
                .unwrap()
                .expect("world is dirty");
        }

        assert_eq!(
            manifest.runs_at_level(Level::L0).len(),
            n_runs,
            "all runs must be at L0"
        );

        // Run compact_one_observed and collect observed entity bits.
        let observed: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));
        let observed_clone = Rc::clone(&observed);

        let report = compact_one_observed(
            &mut manifest,
            &mut log,
            dir.path(),
            Some(&mut |key: EntityKey| {
                observed_clone.borrow_mut().push(key.0);
            }),
        )
        .unwrap()
        .expect("K=4 runs must produce Some(report)");

        assert_eq!(report.from_level, Level::L0);

        let seen = observed.borrow();
        let expected_count = n_runs * entities_per_run;
        assert_eq!(
            seen.len(),
            expected_count,
            "observer must fire exactly once per output entity"
        );

        // Every entity from the input runs must have been observed.
        for bits in &all_entity_bits {
            assert!(
                seen.contains(bits),
                "entity {bits:#x} was not observed by compact_one_observed"
            );
        }
    }
}
