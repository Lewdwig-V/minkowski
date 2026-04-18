use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

/// Number of LSM levels (L0 through L3).
pub const NUM_LEVELS: usize = 4;

/// In-memory manifest tracking all sorted runs across levels.
pub struct LsmManifest {
    levels: [Vec<SortedRunMeta>; NUM_LEVELS],
    next_sequence: u64,
}

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

impl LsmManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            levels: std::array::from_fn(|_| Vec::new()),
            next_sequence: 0,
        }
    }

    /// Add a sorted run to a level.
    pub fn add_run(&mut self, level: Level, meta: SortedRunMeta) {
        self.levels[level.as_index()].push(meta);
    }

    /// Remove a sorted run by path from a level. Returns the removed entry.
    pub fn remove_run(&mut self, level: Level, path: &Path) -> Option<SortedRunMeta> {
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

    /// Record the next sequence number to assign on flush.
    pub fn set_next_sequence(&mut self, seq: SeqNo) {
        self.next_sequence = seq.get();
    }

    /// The next sequence number to assign on the next flush.
    pub fn next_sequence(&self) -> SeqNo {
        SeqNo::from(self.next_sequence)
    }

    /// All sorted runs currently tracked at the given level.
    pub fn runs_at_level(&self, level: Level) -> &[SortedRunMeta] {
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
}

impl Default for LsmManifest {
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
        let m = LsmManifest::new();
        for lvl in 0..NUM_LEVELS {
            assert!(m.runs_at_level(Level::new(lvl as u8).unwrap()).is_empty());
        }
        assert_eq!(m.next_sequence(), SeqNo::from(0u64));
        assert_eq!(m.total_runs(), 0);
    }

    #[test]
    fn add_run_places_at_correct_level() {
        let mut m = LsmManifest::new();
        let meta = test_meta("run_l1.sst");
        m.add_run(Level::L1, meta.clone());
        assert_eq!(m.runs_at_level(Level::L1), &[meta]);
        assert!(m.runs_at_level(Level::L0).is_empty());
    }

    #[test]
    fn remove_run_by_path() {
        let mut m = LsmManifest::new();
        let meta = test_meta("run_a.sst");
        m.add_run(Level::L0, meta.clone());
        let removed = m.remove_run(Level::L0, Path::new("run_a.sst"));
        assert_eq!(removed, Some(meta));
        assert!(m.runs_at_level(Level::L0).is_empty());
    }

    #[test]
    fn remove_run_missing_returns_none() {
        let mut m = LsmManifest::new();
        assert!(
            m.remove_run(Level::L0, Path::new("nonexistent.sst"))
                .is_none()
        );
    }

    #[test]
    fn promote_run_moves_between_levels() {
        let mut m = LsmManifest::new();
        let meta = test_meta("run_x.sst");
        m.add_run(Level::L0, meta);
        m.promote_run(Level::L0, Level::L1, Path::new("run_x.sst"))
            .unwrap();
        assert!(m.runs_at_level(Level::L0).is_empty());
        assert_eq!(m.runs_at_level(Level::L1).len(), 1);
        assert_eq!(m.runs_at_level(Level::L1)[0].path(), Path::new("run_x.sst"));
    }

    #[test]
    fn promote_run_missing_returns_error() {
        let mut m = LsmManifest::new();
        let result = m.promote_run(Level::L0, Level::L1, Path::new("missing.sst"));
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn all_run_paths_collects_all_levels() {
        let mut m = LsmManifest::new();
        m.add_run(Level::L0, test_meta("l0.sst"));
        m.add_run(Level::L2, test_meta("l2.sst"));
        let paths = m.all_run_paths();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&Path::new("l0.sst")));
        assert!(paths.contains(&Path::new("l2.sst")));
    }

    #[test]
    fn total_runs_counts_correctly() {
        let mut m = LsmManifest::new();
        m.add_run(Level::L0, test_meta("a.sst"));
        m.add_run(Level::L0, test_meta("b.sst"));
        m.add_run(Level::L2, test_meta("c.sst"));
        assert_eq!(m.total_runs(), 3);
    }

    #[test]
    fn set_and_get_next_sequence() {
        let mut m = LsmManifest::new();
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
}
