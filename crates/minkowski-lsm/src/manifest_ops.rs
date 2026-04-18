//! High-level operations composing the manifest, log, and writer.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use minkowski::World;

use crate::error::LsmError;
use crate::manifest::{LsmManifest, SortedRunMeta};
use crate::manifest_log::{ManifestEntry, ManifestLog};
use crate::reader::SortedRunReader;
use crate::types::{Level, PageCount, SeqRange, SizeBytes};
use crate::writer::flush;

/// Flush dirty pages and record the new sorted run in the manifest.
///
/// Returns `Ok(Some(path))` if a run was written, `Ok(None)` if no dirty pages.
pub fn flush_and_record<const N: usize>(
    world: &World,
    sequence_range: SeqRange,
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    output_dir: &Path,
) -> Result<Option<PathBuf>, LsmError> {
    let Some(path) = flush(world, sequence_range, output_dir)? else {
        return Ok(None);
    };

    // Extract metadata from the written file.
    let reader = SortedRunReader::open(&path)?;
    let file_size = fs::metadata(&path)?.len();
    let archetype_coverage = reader.archetype_ids();

    let page_count = PageCount::new(reader.page_count())
        .ok_or_else(|| LsmError::Format("page_count must be non-zero".to_owned()))?;
    let meta = SortedRunMeta::new(
        path.clone(),
        reader.sequence_range(),
        archetype_coverage,
        page_count,
        SizeBytes::new(file_size),
    )?;

    // Persist to log first, then update in-memory state.
    // A single atomic entry ensures a crash can never leave the manifest with
    // a new run recorded but the sequence pointer still at its old value.
    log.append(&ManifestEntry::AddRunAndSequence {
        level: Level::L0,
        meta: meta.clone(),
        next_sequence: sequence_range.hi(),
    })?;

    manifest.add_run(Level::L0, meta)?;
    manifest.set_next_sequence(sequence_range.hi());

    Ok(Some(path))
}

/// Delete sorted run files not tracked by the manifest.
///
/// Also removes any `.run.tmp` files (incomplete flushes).
/// Returns the number of files deleted.
pub fn cleanup_orphans<const N: usize>(
    dir: &Path,
    manifest: &LsmManifest<N>,
) -> Result<usize, LsmError> {
    let known: HashSet<PathBuf> = manifest
        .all_run_paths()
        .into_iter()
        .map(Path::to_path_buf)
        .collect();

    let mut deleted = 0;

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // `.run.tmp` is a compound suffix; `Path::extension()` only returns
        // the final segment (`"tmp"`), so matching both cases via extension
        // would need two passes. Suffix matching handles it inline.
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        let should_delete = if name.ends_with(".run.tmp") {
            true
        } else if name.ends_with(".run") {
            !known.contains(&path)
        } else {
            false
        };

        if should_delete {
            fs::remove_file(&path)?;
            deleted += 1;
        }
    }

    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[test]
    fn flush_and_record_dirty_world() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        let result = flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            &mut manifest,
            &mut log,
            dir.path(),
        )
        .unwrap();
        assert!(result.is_some());
        assert_eq!(manifest.total_runs(), 1);
        assert_eq!(manifest.next_sequence(), SeqNo::from(10u64));
        assert_eq!(manifest.runs_at_level(Level::L0).len(), 1);
    }

    #[test]
    fn flush_and_record_clean_world() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.clear_all_dirty_pages();

        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        let result = flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            &mut manifest,
            &mut log,
            dir.path(),
        )
        .unwrap();
        assert!(result.is_none());
        assert_eq!(manifest.total_runs(), 0);
    }

    #[test]
    fn cleanup_orphans_removes_untracked() {
        let dir = tempfile::tempdir().unwrap();

        // Create some files.
        fs::write(dir.path().join("0-10.run"), b"fake").unwrap();
        fs::write(dir.path().join("10-20.run"), b"fake").unwrap();
        fs::write(dir.path().join("crash.run.tmp"), b"incomplete").unwrap();
        fs::write(dir.path().join("notes.txt"), b"not a run").unwrap();

        // Manifest only knows about 0-10.run.
        let mut manifest: crate::manifest::DefaultManifest = LsmManifest::new();
        manifest
            .add_run(
                Level::L0,
                SortedRunMeta::new(
                    dir.path().join("0-10.run"),
                    SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
                    vec![0],
                    PageCount::new(1).unwrap(),
                    SizeBytes::new(4),
                )
                .unwrap(),
            )
            .unwrap();

        let deleted = cleanup_orphans(dir.path(), &manifest).unwrap();
        assert_eq!(deleted, 2); // 10-20.run + crash.run.tmp

        assert!(dir.path().join("0-10.run").exists());
        assert!(!dir.path().join("10-20.run").exists());
        assert!(!dir.path().join("crash.run.tmp").exists());
        assert!(dir.path().join("notes.txt").exists());
    }
}
