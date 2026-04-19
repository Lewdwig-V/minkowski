//! # LsmManifest
//!
//! In-memory index of sorted runs across LSM levels.
//!
//! ## Level Count
//!
//! Defaults to 4 levels. This fits the expected minkowski regime:
//! on-disk data up to ~100× RAM, with reads served from the in-memory
//! World rather than from level traversal (the in-memory World IS the
//! merged view).
//!
//! At T=10 size ratio: 4 levels covers ~0.1× to 100× RAM on disk.
//!
//! For ledger-style workloads (TigerBeetle territory, ever-growing
//! history), construct [`LsmManifest<7>`] instead. Merge logic is
//! level-count-agnostic; only bounds checks and manifest serialization
//! care about `N`.

use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

/// In-memory manifest tracking all sorted runs across `N` levels.
///
/// `N` is a const generic with default 4. Use [`DefaultManifest`] as
/// the conventional alias.
pub struct LsmManifest<const N: usize = 4> {
    levels: [Vec<SortedRunMeta>; N],
    next_sequence: u64,
}

/// Conventional alias for the default 4-level manifest.
pub type DefaultManifest = LsmManifest<4>;

/// Metadata for a single sorted run file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedRunMeta {
    path: PathBuf,
    sequence_range: SeqRange,
    archetype_coverage: Box<[u16]>,
    page_count: PageCount,
    size_bytes: SizeBytes,
}

impl SortedRunMeta {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sequence_range(&self) -> SeqRange {
        self.sequence_range
    }

    pub fn archetype_coverage(&self) -> &[u16] {
        &self.archetype_coverage
    }

    pub fn page_count(&self) -> PageCount {
        self.page_count
    }

    pub fn size_bytes(&self) -> SizeBytes {
        self.size_bytes
    }

    /// Build a `SortedRunMeta` with enforced invariants.
    ///
    /// Validates:
    /// - `archetype_coverage` is strictly ascending (no duplicates, no reordering).
    ///
    /// `page_count` is already non-zero (enforced by `PageCount::new` at the
    /// call site). `sequence_range` is already validated by `SeqRange::new`.
    /// `size_bytes` is not validated (zero is permitted for this newtype).
    pub fn new(
        path: PathBuf,
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,
        page_count: PageCount,
        size_bytes: SizeBytes,
    ) -> Result<Self, LsmError> {
        if archetype_coverage.windows(2).any(|w| w[0] >= w[1]) {
            return Err(LsmError::Format(
                "archetype_coverage is not strictly sorted".to_owned(),
            ));
        }
        Ok(Self {
            path,
            sequence_range,
            archetype_coverage: archetype_coverage.into_boxed_slice(),
            page_count,
            size_bytes,
        })
    }
}

impl<const N: usize> LsmManifest<N> {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            levels: std::array::from_fn(|_| Vec::new()),
            next_sequence: 0,
        }
    }

    /// Add a sorted run to a level.
    ///
    /// Returns `Err(LsmError::Format)` if `level.as_index() >= N`.
    pub fn add_run(&mut self, level: Level, meta: SortedRunMeta) -> Result<(), LsmError> {
        if level.as_index() >= N {
            return Err(LsmError::Format(format!(
                "level {} out of range for {}-level manifest",
                level, N
            )));
        }
        self.levels[level.as_index()].push(meta);
        Ok(())
    }

    /// Remove a sorted run by path from a level. Returns the removed entry.
    ///
    /// Returns `None` if the level is out of range or the path is not found.
    pub fn remove_run(&mut self, level: Level, path: &Path) -> Option<SortedRunMeta> {
        if level.as_index() >= N {
            return None;
        }
        let runs = &mut self.levels[level.as_index()];
        runs.iter()
            .position(|r| r.path() == path)
            .map(|pos| runs.remove(pos))
    }

    /// Move a run from one level to another.
    pub fn promote_run(
        &mut self,
        from_level: Level,
        to_level: Level,
        path: &Path,
    ) -> Result<(), LsmError> {
        if from_level.as_index() >= N || to_level.as_index() >= N {
            return Err(LsmError::Format(format!(
                "level out of range for {}-level manifest: from={}, to={}",
                N, from_level, to_level
            )));
        }
        let meta = self.remove_run(from_level, path).ok_or_else(|| {
            LsmError::Format(format!(
                "run {} not found at level {}",
                path.display(),
                from_level
            ))
        })?;
        self.add_run(to_level, meta)?;
        Ok(())
    }

    /// Record the next sequence number to assign on flush.
    pub fn set_next_sequence(&mut self, seq: SeqNo) {
        self.next_sequence = seq.get();
    }

    /// The next sequence number to assign on the next flush.
    pub fn next_sequence(&self) -> SeqNo {
        SeqNo::from(self.next_sequence)
    }

    /// All sorted runs currently tracked at the given level.
    ///
    /// Returns `&[]` if `level.as_index() >= N`.
    pub fn runs_at_level(&self, level: Level) -> &[SortedRunMeta] {
        if level.as_index() >= N {
            return &[];
        }
        &self.levels[level.as_index()]
    }

    /// All tracked run file paths across all levels.
    pub fn all_run_paths(&self) -> Vec<&Path> {
        self.levels
            .iter()
            .flat_map(|runs| runs.iter().map(SortedRunMeta::path))
            .collect()
    }

    pub fn total_runs(&self) -> usize {
        self.levels.iter().map(Vec::len).sum()
    }

    /// Returns `true` if any `(level, archetype)` pair has at least
    /// [`COMPACTION_TRIGGER`] runs at a non-bottom level — i.e., there is
    /// work the compactor would pick up.
    ///
    /// Delegates to [`compactor::find_compaction_candidate`] so there is one
    /// source of truth for the trigger policy.
    ///
    /// [`COMPACTION_TRIGGER`]: crate::compactor::COMPACTION_TRIGGER
    pub fn needs_compaction(&self) -> bool {
        crate::compactor::find_compaction_candidate(self).is_ok_and(|opt| opt.is_some())
    }
}

impl<const N: usize> Default for LsmManifest<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

    fn test_meta(name: &str) -> SortedRunMeta {
        SortedRunMeta::new(
            PathBuf::from(name),
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            vec![0],
            PageCount::new(1).unwrap(),
            SizeBytes::new(1024),
        )
        .unwrap()
    }

    #[test]
    fn new_manifest_is_empty() {
        let m: DefaultManifest = LsmManifest::new();
        for lvl in 0..4 {
            assert!(m.runs_at_level(Level::new(lvl as u8).unwrap()).is_empty());
        }
        assert_eq!(m.next_sequence(), SeqNo::from(0u64));
        assert_eq!(m.total_runs(), 0);
    }

    #[test]
    fn add_run_places_at_correct_level() {
        let mut m: DefaultManifest = LsmManifest::new();
        let meta = test_meta("run_l1.sst");
        m.add_run(Level::L1, meta.clone()).unwrap();
        assert_eq!(m.runs_at_level(Level::L1), &[meta]);
        assert!(m.runs_at_level(Level::L0).is_empty());
    }

    #[test]
    fn remove_run_by_path() {
        let mut m: DefaultManifest = LsmManifest::new();
        let meta = test_meta("run_a.sst");
        m.add_run(Level::L0, meta.clone()).unwrap();
        let removed = m.remove_run(Level::L0, Path::new("run_a.sst"));
        assert_eq!(removed, Some(meta));
        assert!(m.runs_at_level(Level::L0).is_empty());
    }

    #[test]
    fn remove_run_missing_returns_none() {
        let mut m: DefaultManifest = LsmManifest::new();
        assert!(
            m.remove_run(Level::L0, Path::new("nonexistent.sst"))
                .is_none()
        );
    }

    #[test]
    fn promote_run_moves_between_levels() {
        let mut m: DefaultManifest = LsmManifest::new();
        let meta = test_meta("run_x.sst");
        m.add_run(Level::L0, meta).unwrap();
        m.promote_run(Level::L0, Level::L1, Path::new("run_x.sst"))
            .unwrap();
        assert!(m.runs_at_level(Level::L0).is_empty());
        assert_eq!(m.runs_at_level(Level::L1).len(), 1);
        assert_eq!(m.runs_at_level(Level::L1)[0].path(), Path::new("run_x.sst"));
    }

    #[test]
    fn promote_run_missing_returns_error() {
        let mut m: DefaultManifest = LsmManifest::new();
        let result = m.promote_run(Level::L0, Level::L1, Path::new("missing.sst"));
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn all_run_paths_collects_all_levels() {
        let mut m: DefaultManifest = LsmManifest::new();
        m.add_run(Level::L0, test_meta("l0.sst")).unwrap();
        m.add_run(Level::L2, test_meta("l2.sst")).unwrap();
        let paths = m.all_run_paths();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&Path::new("l0.sst")));
        assert!(paths.contains(&Path::new("l2.sst")));
    }

    #[test]
    fn total_runs_counts_correctly() {
        let mut m: DefaultManifest = LsmManifest::new();
        m.add_run(Level::L0, test_meta("a.sst")).unwrap();
        m.add_run(Level::L0, test_meta("b.sst")).unwrap();
        m.add_run(Level::L2, test_meta("c.sst")).unwrap();
        assert_eq!(m.total_runs(), 3);
    }

    #[test]
    fn set_and_get_next_sequence() {
        let mut m: DefaultManifest = LsmManifest::new();
        m.set_next_sequence(SeqNo::from(42u64));
        assert_eq!(m.next_sequence(), SeqNo::from(42u64));
    }

    #[test]
    fn sorted_run_meta_new_accepts_valid_input() {
        let meta = SortedRunMeta::new(
            PathBuf::from("0-10.run"),
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            vec![0, 3, 7],
            PageCount::new(1).unwrap(),
            SizeBytes::new(1024),
        )
        .unwrap();
        assert_eq!(meta.sequence_range().lo(), SeqNo::from(0u64));
        assert_eq!(meta.page_count().get(), 1);
        assert_eq!(meta.size_bytes().get(), 1024);
    }

    #[test]
    fn sorted_run_meta_new_rejects_unsorted_coverage() {
        let result = SortedRunMeta::new(
            PathBuf::from("x.run"),
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            vec![3, 1, 2],
            PageCount::new(1).unwrap(),
            SizeBytes::new(1024),
        );
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn sorted_run_meta_new_rejects_duplicated_coverage() {
        let result = SortedRunMeta::new(
            PathBuf::from("x.run"),
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            vec![1, 2, 2, 3],
            PageCount::new(1).unwrap(),
            SizeBytes::new(1024),
        );
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn sorted_run_meta_new_accepts_empty_coverage() {
        let meta = SortedRunMeta::new(
            PathBuf::from("x.run"),
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(0u64)).unwrap(),
            vec![],
            PageCount::new(1).unwrap(),
            SizeBytes::new(1024),
        )
        .unwrap();
        assert_eq!(meta.archetype_coverage().len(), 0);
    }

    #[test]
    fn lsm_manifest_alternate_level_count_compiles_and_works() {
        // Default N=4 path still works.
        let m4: LsmManifest<4> = LsmManifest::new();
        assert_eq!(m4.total_runs(), 0);

        // Alternate N=7 manifest constructs and is distinct at the type level.
        let m7: LsmManifest<7> = LsmManifest::new();
        assert_eq!(m7.total_runs(), 0);

        // DefaultManifest alias resolves to N=4.
        let md: DefaultManifest = LsmManifest::new();
        assert_eq!(md.total_runs(), 0);
    }

    #[test]
    fn lsm_manifest_n7_exercises_level_beyond_default() {
        // Confirm that LsmManifest<7>'s extra levels (4..=6) actually work
        // for add/query. This guards against a hypothetical future
        // regression where the fixed-size [Vec<_>; N] array is indexed by
        // a hardcoded 4.
        let mut m7: LsmManifest<7> = LsmManifest::new();
        let level_6 = Level::new(6).unwrap();

        m7.add_run(level_6, test_meta("l6.run")).unwrap();
        assert_eq!(m7.runs_at_level(level_6).len(), 1);
        assert_eq!(m7.total_runs(), 1);

        // Same level on LsmManifest<4> returns the OOR signals.
        let mut m4: DefaultManifest = LsmManifest::new();
        let err = m4.add_run(level_6, test_meta("l6.run")).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        assert!(m4.runs_at_level(level_6).is_empty());
    }

    #[test]
    fn add_run_rejects_out_of_range_level() {
        let mut m: DefaultManifest = LsmManifest::new();
        let oor = Level::new(4).unwrap(); // valid Level, but OOR for N=4.
        let err = m.add_run(oor, test_meta("oor.run")).unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
        assert_eq!(m.total_runs(), 0, "failed add_run must not mutate state");
    }

    #[test]
    fn remove_run_returns_none_for_out_of_range_level() {
        let mut m: DefaultManifest = LsmManifest::new();
        let oor = Level::new(4).unwrap();
        assert!(m.remove_run(oor, Path::new("anything.run")).is_none());
    }

    #[test]
    fn promote_run_rejects_out_of_range_levels() {
        let mut m: DefaultManifest = LsmManifest::new();
        let oor = Level::new(4).unwrap();

        // OOR as destination.
        let err = m
            .promote_run(Level::L0, oor, Path::new("x.run"))
            .unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));

        // OOR as source.
        let err = m
            .promote_run(oor, Level::L0, Path::new("x.run"))
            .unwrap_err();
        assert!(matches!(err, LsmError::Format(_)));
    }

    #[test]
    fn runs_at_level_returns_empty_for_out_of_range_level() {
        let m: DefaultManifest = LsmManifest::new();
        let oor = Level::new(4).unwrap();
        assert!(m.runs_at_level(oor).is_empty());
    }
}
