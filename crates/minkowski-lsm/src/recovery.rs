//! Restore a [`World`] from LSM sorted runs recorded in the manifest.

use std::alloc::Layout;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use minkowski::{ComponentId, Entity, World};

use crate::allocator_meta;
use crate::codec::CodecRegistry;
use crate::error::LsmError;
use crate::format::{ALLOCATOR_SLOT, META_ARCH_ID, SPARSE_ARCH_ID};
use crate::manifest::{LsmManifest, SortedRunMeta};
use crate::manifest_log::ManifestLog;
use crate::manifest_ops::cleanup_orphans;
use crate::reader::SortedRunReader;
use crate::sparse_page;
use crate::types::Level;

/// Result of an LSM recovery operation.
pub struct RecoveryResult {
    /// Reconstructed world state from sorted runs (before WAL replay).
    pub world: World,
    /// WAL sequence number to begin replay from.
    pub flush_seq: u64,
}

/// High-level LSM recovery orchestrator.
pub struct LsmRecovery;

type ArchetypeSig = Vec<String>;

/// Identifies a column within an archetype: the entity pseudo-column or a named
/// component. Keying by NAME (not the run's positional schema slot) is what
/// fixes the multi-archetype reconstruction bug — a component's bytes are
/// reassembled and imported by identity, never by signature position.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ColumnKey {
    Entity,
    Component(String),
}

type PageKey = (ArchetypeSig, ColumnKey, u16);
/// Raw sparse entries for one component: `(entity_bits, rkyv_value_bytes)`.
type SparseEntries = Vec<(u64, Vec<u8>)>;

#[derive(Clone)]
struct StoredPage {
    seq_hi: u64,
    row_count: u16,
    data: Vec<u8>,
}

#[derive(Clone)]
struct StoredAllocator {
    seq_hi: u64,
    generations: Vec<u32>,
    free_list: Vec<u32>,
}

struct StoredSparse {
    seq_hi: u64,
    /// (component stable name, entries). The name is decoded directly from each
    /// self-describing blob — the run's schema is never consulted for sparse.
    components: Vec<(String, SparseEntries)>,
}

impl LsmRecovery {
    /// Recover world state from on-disk LSM artifacts.
    pub fn recover<const N: usize>(
        lsm_dir: &Path,
        manifest_log_path: &Path,
        codecs: &CodecRegistry,
    ) -> Result<(RecoveryResult, LsmManifest<N>, ManifestLog), LsmError> {
        let (manifest, log) = ManifestLog::recover::<N>(manifest_log_path)?;
        let _ = cleanup_orphans(lsm_dir, &manifest)?;

        let mut runs: Vec<&SortedRunMeta> = Vec::new();
        for level_idx in 0..N {
            let Some(level) = Level::new(level_idx as u8) else {
                continue;
            };
            runs.extend(manifest.runs_at_level(level).iter());
        }
        runs.sort_by_key(|meta| meta.sequence_range().lo().get());

        let mut pages: BTreeMap<PageKey, StoredPage> = BTreeMap::new();
        let mut allocator: Option<StoredAllocator> = None;
        let mut sparse: Option<StoredSparse> = None;
        let mut component_layouts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut max_seq_hi = 0u64;

        for meta in runs {
            let reader = SortedRunReader::open(meta.path())?;
            let seq_hi = meta.sequence_range().hi().get();
            max_seq_hi = max_seq_hi.max(seq_hi);

            for entry in reader.schema().entries() {
                component_layouts
                    .entry(entry.name().to_owned())
                    .or_insert((entry.item_size() as usize, entry.item_align() as usize));
            }

            // Allocator metadata may span multiple pages (large free lists);
            // the run with the highest seq_hi wins. Only the winner is read and
            // decoded. slot_pages yields in page_index order, so concatenation
            // reconstructs the original blob.
            if allocator.as_ref().is_none_or(|a| seq_hi >= a.seq_hi) {
                let mut alloc_blob: Vec<u8> = Vec::new();
                let mut saw_allocator = false;
                for result in reader.slot_pages(META_ARCH_ID, ALLOCATOR_SLOT) {
                    let (_page_index, page) = result?;
                    reader.validate_page_crc(&page)?;
                    let payload_len = page.header().row_count as usize;
                    alloc_blob.extend_from_slice(&page.data()[..payload_len]);
                    saw_allocator = true;
                }
                if saw_allocator {
                    let (generations, free_list) = allocator_meta::decode(&alloc_blob)?;
                    allocator = Some(StoredAllocator {
                        seq_hi,
                        generations,
                        free_list,
                    });
                }
            }

            // Sparse baseline: newest run wins wholesale (same rule as allocator).
            if sparse.as_ref().is_none_or(|s| seq_hi >= s.seq_hi) {
                let mut components: Vec<(String, SparseEntries)> = Vec::new();
                for slot in reader.component_slots_for_arch(SPARSE_ARCH_ID) {
                    let mut blob: Vec<u8> = Vec::new();
                    for result in reader.slot_pages(SPARSE_ARCH_ID, slot) {
                        let (_page_index, page) = result?;
                        reader.validate_page_crc(&page)?;
                        let len = page.header().row_count as usize;
                        blob.extend_from_slice(&page.data()[..len]);
                    }
                    if blob.is_empty() {
                        continue;
                    }
                    // The blob is self-describing: it carries the component's
                    // stable name. The page `slot` is only a grouping index.
                    let (name, entries) = sparse_page::decode(&blob)?;
                    components.push((name, entries));
                }
                // Update UNCONDITIONALLY (even when `components` is empty). Every
                // flush writes the COMPLETE current sparse state, so the newest
                // run is authoritative: a run with no sparse pages means sparse
                // is genuinely empty and must supersede an older run's data.
                // Skipping the update on empty would resurrect components removed
                // between flushes (e.g. a live entity whose sparse component was
                // removed before the newer flush).
                sparse = Some(StoredSparse { seq_hi, components });
            }

            for arch_id in reader.archetype_ids() {
                if arch_id == META_ARCH_ID || arch_id == SPARSE_ARCH_ID {
                    continue;
                }
                let sig = archetype_signature(&reader, arch_id)?;

                for slot in reader.component_slots_for_arch(arch_id) {
                    // component_slots_for_arch excludes ENTITY_SLOT, so every slot
                    // here has a named schema entry.
                    let (name, item_size) = {
                        let entry = reader.schema().entry_for_slot(slot).ok_or_else(|| {
                            LsmError::Format(format!("missing schema entry for slot {slot}"))
                        })?;
                        (entry.name().to_owned(), entry.item_size as usize)
                    };
                    for result in reader.slot_pages(arch_id, slot) {
                        let (page_index, page) = result?;
                        reader.validate_page_crc(&page)?;
                        let payload_len = page.header().row_count as usize * item_size;
                        store_page(
                            &mut pages,
                            (sig.clone(), ColumnKey::Component(name.clone()), page_index),
                            seq_hi,
                            page.header().row_count,
                            &page.data()[..payload_len],
                        );
                    }
                }

                for result in reader.entity_pages(arch_id) {
                    let (page_index, page) = result?;
                    reader.validate_page_crc(&page)?;
                    let payload_len = page.header().row_count as usize * 8;
                    store_page(
                        &mut pages,
                        (sig.clone(), ColumnKey::Entity, page_index),
                        seq_hi,
                        page.header().row_count,
                        &page.data()[..payload_len],
                    );
                }
            }
        }

        let flush_seq = if manifest.total_runs() > 0 {
            max_seq_hi
        } else {
            0
        };

        // Release-mode invariant: if we recovered any pages we must have resolved
        // their component layouts, or materialize_world produces a corrupt world.
        assert!(
            pages.is_empty() || !component_layouts.is_empty(),
            "recovery: pages present but no component layouts were resolved"
        );
        let mut world = materialize_world(pages, allocator.as_ref(), &component_layouts, codecs)?;
        apply_sparse(&mut world, sparse.as_ref(), codecs)?;
        Ok((RecoveryResult { world, flush_seq }, manifest, log))
    }
}

fn store_page(
    pages: &mut BTreeMap<PageKey, StoredPage>,
    key: PageKey,
    seq_hi: u64,
    row_count: u16,
    data: &[u8],
) {
    if pages
        .get(&key)
        .is_none_or(|existing| seq_hi >= existing.seq_hi)
    {
        pages.insert(
            key,
            StoredPage {
                seq_hi,
                row_count,
                data: data.to_vec(),
            },
        );
    }
}

fn archetype_signature(reader: &SortedRunReader, arch_id: u16) -> Result<ArchetypeSig, LsmError> {
    let slots = reader.component_slots_for_arch(arch_id);
    let mut names: Vec<String> = Vec::with_capacity(slots.len());
    for slot in slots {
        let entry = reader
            .schema()
            .entry_for_slot(slot)
            .ok_or_else(|| LsmError::Format(format!("missing schema entry for slot {slot}")))?;
        names.push(entry.name().to_owned());
    }
    names.sort();
    Ok(names)
}

fn resolve_schema_component(
    codecs: &CodecRegistry,
    world: &World,
    schema_name: &str,
) -> Option<ComponentId> {
    if let Some(id) = codecs.resolve_name(schema_name) {
        return Some(id);
    }
    codecs
        .registered_ids()
        .into_iter()
        .find(|&id| world.component_name(id).is_some_and(|n| n == schema_name))
}

/// Build the entity-allocator state for recovery.
///
/// When a persisted allocator (`stored`) is available it is AUTHORITATIVE — it
/// was written by the newest run and carries the free list plus the bumped
/// generations of every despawned entity. We return it verbatim. On-disk
/// entity pages may carry stale (older-generation) rows from runs that wrote
/// them when those entities were still alive; if we overlaid those generations
/// unconditionally we would downgrade the persisted allocator's dead generations
/// back to alive, resurrecting despawned entities. The dead-row filter in
/// `materialize_world` drops those stale rows against the authoritative
/// allocator instead.
///
/// When `stored` is `None` (no run has allocator metadata — e.g. a run written
/// before allocator persistence was wired), we rebuild generations from the
/// on-disk entity pages. This path cannot detect despawns that happened before
/// the flush, so it is only a fallback; the normal path always has `stored`.
fn build_allocator_state(
    by_sig: &BTreeMap<ArchetypeSig, BTreeMap<ColumnKey, BTreeMap<u16, StoredPage>>>,
    stored: Option<&StoredAllocator>,
) -> (Vec<u32>, Vec<u32>) {
    let mut generations = stored.map(|a| a.generations.clone()).unwrap_or_default();
    let free_list = stored.map(|a| a.free_list.clone()).unwrap_or_default();

    // Only overlay on-disk generations when there is no persisted allocator.
    // With a persisted allocator this loop is skipped — the allocator is the
    // single source of truth for alive/dead state.
    if stored.is_none() {
        for columns in by_sig.values() {
            if let Some(entity_pages) = columns.get(&ColumnKey::Entity) {
                for page in entity_pages.values() {
                    for chunk in page.data.chunks_exact(8).take(page.row_count as usize) {
                        let entity = Entity::from_bits(u64::from_le_bytes(
                            chunk.try_into().expect("8 bytes"),
                        ));
                        let idx = entity.index() as usize;
                        if generations.len() <= idx {
                            generations.resize(idx + 1, 0);
                        }
                        generations[idx] = entity.generation();
                    }
                }
            }
        }
    }

    (generations, free_list)
}

fn materialize_world(
    pages: BTreeMap<PageKey, StoredPage>,
    allocator: Option<&StoredAllocator>,
    component_layouts: &BTreeMap<String, (usize, usize)>,
    codecs: &CodecRegistry,
) -> Result<World, LsmError> {
    let mut world = World::new();
    for id in codecs.registered_ids() {
        codecs.register_one(id, &mut world);
    }

    // Register any component present on disk but lacking a codec, preserving its
    // layout, and build name -> ComponentId.
    let mut name_to_id: HashMap<String, ComponentId> = HashMap::new();
    for (name, (size, align)) in component_layouts {
        let id = if let Some(local_id) = resolve_schema_component(codecs, &world, name) {
            local_id
        } else {
            let layout = Layout::from_size_align(*size, *align).map_err(|_| {
                LsmError::Format(format!(
                    "invalid layout for component {name}: size={size}, align={align}"
                ))
            })?;
            // Intentional leak: World::register_component_raw requires a
            // &'static str for the component name. We leak one Box<str> per
            // on-disk component that has no registered codec. The volume is
            // bounded by the component type count (not the entity count), and
            // recovery runs once per process lifetime, so this does not grow
            // unbounded. Re-running recovery for the same component would leak
            // a second copy; if that matters, intern names in a static
            // registry keyed by name. For now the leak is the documented cost
            // of recovering a codecless component.
            let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
            world.register_component_raw(leaked, layout)
        };
        name_to_id.insert(name.clone(), id);
    }

    // Group pages: signature -> column -> page_index -> page.
    let mut by_sig: BTreeMap<ArchetypeSig, BTreeMap<ColumnKey, BTreeMap<u16, StoredPage>>> =
        BTreeMap::new();
    for ((sig, col, page_index), page) in pages {
        by_sig
            .entry(sig)
            .or_default()
            .entry(col)
            .or_default()
            .insert(page_index, page);
    }

    // Allocator-first: import_page checks `is_alive`, so generations must be set
    // before any entity is placed.
    let (generations, free_list) = build_allocator_state(&by_sig, allocator);
    world.restore_allocator_state(generations, free_list);

    for (sig, columns) in &by_sig {
        // Resolve (comp_id, name) and sort by comp_id — the canonical archetype
        // key and the order import_page expects its columns in.
        let mut comp_pairs: Vec<(ComponentId, &String)> = sig
            .iter()
            .map(|name| {
                name_to_id
                    .get(name)
                    .copied()
                    .map(|id| (id, name))
                    .ok_or_else(|| LsmError::Format(format!("unregistered component {name}")))
            })
            .collect::<Result<_, _>>()?;
        comp_pairs.sort_by_key(|(id, _)| *id);
        let comp_ids: Vec<ComponentId> = comp_pairs.iter().map(|(id, _)| *id).collect();

        let target = world
            .import_target(&comp_ids)
            .map_err(|e| LsmError::Format(format!("import_target failed for {sig:?}: {e}")))?;

        let entity_pages = columns
            .get(&ColumnKey::Entity)
            .ok_or_else(|| LsmError::Format(format!("archetype {sig:?} has no entity pages")))?;

        // Pre-resolve the per-column item size for this archetype's components,
        // used to slice column bytes when compacting a page down to its live
        // rows. `component_layouts` was built from the on-disk schema entries
        // and is keyed by component name. A missing entry is unreachable today
        // (the signature is derived from the same schema via `entry_for_slot`,
        // which errors first), but return an explicit error rather than
        // silently treating a non-ZST component as a ZST (which would copy no
        // bytes for its rows in the slow-path compaction → silent data loss).
        // This guards against any future change that decouples the signature
        // from the schema.
        let item_sizes: Vec<usize> = comp_pairs
            .iter()
            .map(|(_, name)| {
                component_layouts.get(*name).map_or(
                    Err(LsmError::Format(format!(
                        "component {name} in archetype {sig:?} has no resolved layout"
                    ))),
                    |(size, _)| Ok(*size),
                )
            })
            .collect::<Result<_, _>>()?;

        for (&page_index, entity_page) in entity_pages {
            let row_count = entity_page.row_count as usize;

            // Read the raw entity handles for this page.
            let mut page_entities: Vec<Entity> = Vec::with_capacity(row_count);
            for chunk in entity_page.data.chunks_exact(8).take(row_count) {
                page_entities.push(Entity::from_bits(u64::from_le_bytes(
                    chunk.try_into().expect("8 bytes"),
                )));
            }

            // Dead-row filter: the persisted allocator is authoritative for
            // alive/dead state. A stale page from an older run may carry
            // entities that were alive when that run was written but have since
            // been despawned (their generations were bumped). Importing them
            // would resurrect despawned entities with stale component bytes and
            // collide with survivors already imported from a newer page. Drop
            // any entity whose current generation in the restored allocator
            // does not match the on-disk handle's generation, OR that has
            // already been placed by an earlier page in this recovery (a
            // survivor that appears on both the newer page and a stale
            // higher-index page from an older run — `store_page` keeps both
            // because their page_index keys differ).
            //
            // Page iteration order (BTreeMap ascending by page_index) processes
            // lower-index pages first. Because `swap_remove` only moves a
            // survivor to a lower-or-equal row, the newer run places a survivor
            // at a page_index <= the stale page's index, so the newer page is
            // always seen first and the survivor is imported with current bytes;
            // the stale page then skips it as already-placed.
            let mut live_mask: Vec<bool> = Vec::with_capacity(row_count);
            let mut live_count = 0usize;
            for &e in &page_entities {
                let alive = world.is_alive(e) && !world.is_placed(e);
                live_mask.push(alive);
                if alive {
                    live_count += 1;
                }
            }
            if live_count == 0 {
                continue;
            }

            // Fast path: every entity on the page is alive — use the contiguous
            // column slices directly, no copy.
            let (entities, col_slices): (Vec<Entity>, Vec<Vec<u8>>) = if live_count == row_count {
                let cols: Vec<Vec<u8>> = comp_pairs
                    .iter()
                    .map(|(_, name)| {
                        let col_pages = columns
                            .get(&ColumnKey::Component((*name).clone()))
                            .ok_or_else(|| {
                                LsmError::Format(format!("archetype {sig:?} missing column {name}"))
                            })?;
                        let page = col_pages.get(&page_index).ok_or_else(|| {
                            LsmError::Format(format!(
                                "archetype {sig:?} column {name} missing page {page_index}"
                            ))
                        })?;
                        if page.row_count as usize != row_count {
                            return Err(LsmError::Format(format!(
                                "archetype {sig:?} column {name} page {page_index} has {} rows, \
                                 entity page has {row_count}",
                                page.row_count
                            )));
                        }
                        Ok(page.data.clone())
                    })
                    .collect::<Result<_, _>>()?;
                (page_entities.clone(), cols)
            } else {
                // Slow path: compact the page down to its live rows. Rebuild
                // the entity Vec and each column's byte buffer with only the
                // rows whose entity is alive, so `import_page` sees a
                // consistent (entities, columns) pair with no dead rows.
                let mut live_entities = Vec::with_capacity(live_count);
                for (&e, &alive) in page_entities.iter().zip(live_mask.iter()) {
                    if alive {
                        live_entities.push(e);
                    }
                }
                let mut live_cols: Vec<Vec<u8>> = Vec::with_capacity(comp_pairs.len());
                for (i, (_, name)) in comp_pairs.iter().enumerate() {
                    let col_pages = columns
                        .get(&ColumnKey::Component((*name).clone()))
                        .ok_or_else(|| {
                            LsmError::Format(format!("archetype {sig:?} missing column {name}"))
                        })?;
                    let page = col_pages.get(&page_index).ok_or_else(|| {
                        LsmError::Format(format!(
                            "archetype {sig:?} column {name} missing page {page_index}"
                        ))
                    })?;
                    if page.row_count as usize != row_count {
                        return Err(LsmError::Format(format!(
                            "archetype {sig:?} column {name} page {page_index} has {} rows, \
                             entity page has {row_count}",
                            page.row_count
                        )));
                    }
                    let item_size = item_sizes[i];
                    let mut buf =
                        Vec::with_capacity(live_count.checked_mul(item_size).unwrap_or(0));
                    for (row, &alive) in live_mask.iter().enumerate() {
                        if alive {
                            let start = row * item_size;
                            let end = start + item_size;
                            buf.extend_from_slice(&page.data[start..end]);
                        }
                    }
                    live_cols.push(buf);
                }
                (live_entities, live_cols)
            };

            // Borrow the column bytes back as slices for `ImportPage::page`.
            let col_refs: Vec<&[u8]> = col_slices.iter().map(Vec::as_slice).collect();
            let import_page = target.page(&entities, &col_refs).map_err(|e| {
                LsmError::Format(format!("import page build failed for {sig:?}: {e}"))
            })?;
            // SAFETY: each column slice is the native byte image of a raw-copyable
            // component (enforced at codec registration); the source pages passed
            // per-page CRC validation on read; every entity in `entities` is
            // alive per the allocator state restored above (dead rows were
            // filtered out).
            unsafe {
                world.import_page(&import_page).map_err(|e| {
                    LsmError::Format(format!("import_page failed for {sig:?}: {e}"))
                })?;
            }
        }
    }

    Ok(world)
}

/// Re-inserts the baseline sparse state captured from the newest run. Runs
/// after `materialize_world` so entity generations are already restored. Uses
/// the codec sparse seam, which sets the correct drop function for each set.
fn apply_sparse(
    world: &mut World,
    stored: Option<&StoredSparse>,
    codecs: &CodecRegistry,
) -> Result<(), LsmError> {
    let Some(stored) = stored else {
        return Ok(());
    };
    for (name, entries) in &stored.components {
        let comp_id = resolve_schema_component(codecs, world, name).ok_or_else(|| {
            LsmError::Format(format!(
                "sparse component {name} not registered on recovery"
            ))
        })?;
        for (entity_bits, value) in entries {
            let entity = Entity::from_bits(*entity_bits);
            // Defense-in-depth: only restore sparse for entities that are alive
            // in the materialized world. A blob entry was live at flush time, so
            // co-located allocator state normally keeps it alive here — but
            // guarding against any despawn-cleanup gap or allocator/sparse
            // selection divergence prevents inserting a phantom entry under a
            // dead (index, generation). Generation match is the source of truth.
            if !world.is_alive(entity) {
                continue;
            }
            codecs
                .insert_sparse_raw(comp_id, world, entity, value)
                .map_err(|e| LsmError::Format(format!("sparse restore failed for {name}: {e}")))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::compactor::compact_one;
    use crate::manifest_log::ManifestLog;
    use crate::manifest_ops::flush_and_record;
    use crate::types::{SeqNo, SeqRange};
    use rkyv::{Archive, Deserialize, Serialize};

    /// codex #4 regression: a multi-archetype world where one archetype omits a
    /// component that name-sorts before one it has. The pre-rewrite
    /// materialize_world keyed columns by signature *position*, so the
    /// `{z_comp}`-only archetype read its column at position 0 (a_comp's slot) →
    /// "missing component page slot=0" / corruption. Name-keyed reconstruction
    /// restores both archetypes correctly.
    #[test]
    fn recover_multi_archetype_nonuniform_components() {
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct AComp(u32);
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct ZComp(u64);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<AComp>("a_comp", &mut world).unwrap();
        codecs.register_as::<ZComp>("z_comp", &mut world).unwrap();

        let both = world.spawn((AComp(10), ZComp(20)));
        let zonly = world.spawn((ZComp(99),));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        assert_eq!(recovered.get::<AComp>(both).copied(), Some(AComp(10)));
        assert_eq!(recovered.get::<ZComp>(both).copied(), Some(ZComp(20)));
        assert_eq!(recovered.get::<ZComp>(zonly).copied(), Some(ZComp(99)));
        assert_eq!(recovered.get::<AComp>(zonly), None);
    }

    #[test]
    fn recover_restores_sparse_components() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct SparsePos {
            x: f32,
            y: f32,
        }

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<SparsePos>("sparse_pos", &mut world)
            .unwrap();
        codecs.register_as::<Tag>("tag", &mut world).unwrap();

        let e1 = world.spawn((SparsePos { x: 1.0, y: 2.0 },));
        let e2 = world.spawn((SparsePos { x: 3.0, y: 4.0 },));
        world.insert_sparse::<Tag>(e1, Tag(111));
        world.insert_sparse::<Tag>(e2, Tag(222));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        assert_eq!(recovered.get::<Tag>(e1).copied(), Some(Tag(111)));
        assert_eq!(recovered.get::<Tag>(e2).copied(), Some(Tag(222)));
        // The archetype component must survive alongside the sparse one — the
        // sparse-restore path overlays onto already-materialized entities.
        assert_eq!(recovered.get::<SparsePos>(e1).map(|p| p.x), Some(1.0));
        assert_eq!(recovered.get::<SparsePos>(e2).map(|p| p.x), Some(3.0));
    }

    /// Two distinct sparse component types in a single flush must both round-trip
    /// — they occupy grouping slots 0 and 1 (name-sorted) under SPARSE_ARCH_ID
    /// and decode independently.
    #[test]
    fn recover_restores_two_sparse_component_types() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct PosI {
            x: f32,
            y: f32,
        }
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct TagA(u32);
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct TagB(u64);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<PosI>("pos_i", &mut world).unwrap();
        codecs.register_as::<TagA>("tag_a", &mut world).unwrap();
        codecs.register_as::<TagB>("tag_b", &mut world).unwrap();

        let e = world.spawn((PosI { x: 1.0, y: 1.0 },));
        world.insert_sparse::<TagA>(e, TagA(10));
        world.insert_sparse::<TagB>(e, TagB(20));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        assert_eq!(recovered.get::<TagA>(e).copied(), Some(TagA(10)));
        assert_eq!(recovered.get::<TagB>(e).copied(), Some(TagB(20)));
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    fn flush_world(
        world: &World,
        manifest: &mut crate::manifest::DefaultManifest,
        log: &mut ManifestLog,
        dir: &Path,
        lo: u64,
        hi: u64,
    ) {
        let codecs = CodecRegistry::new();
        flush_and_record(
            world,
            SeqRange::new(SeqNo::from(lo), SeqNo::from(hi)).unwrap(),
            manifest,
            log,
            dir,
            &codecs,
        )
        .unwrap()
        .expect("dirty world must flush");
    }

    #[test]
    fn recover_single_flush() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Vel>(&mut world).unwrap();

        for i in 0..5 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: i as f32,
                },
                Vel { dx: 1.0, dy: 2.0 },
            ));
        }

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_world(&world, &mut manifest, &mut log, dir.path(), 0, 10);

        let (manifest_check, _) = ManifestLog::recover::<4>(&log_path).unwrap();
        assert_eq!(manifest_check.total_runs(), 1);

        let (mut result, _, _) = LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
        assert_eq!(result.flush_seq, 10);
        assert_eq!(result.world.query::<(&Pos,)>().count(), 5);

        let mut xs: Vec<f32> = result.world.query::<(&Pos,)>().map(|(p,)| p.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(xs, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn recover_later_flush_overwrites_page() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_world(&world, &mut manifest, &mut log, dir.path(), 0, 5);

        for (pos,) in world.query::<(&mut Pos,)>() {
            pos.x = 99.0;
        }
        flush_world(&world, &mut manifest, &mut log, dir.path(), 5, 10);

        let (mut result, _, _) = LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
        let x = result.world.query::<(&Pos,)>().next().unwrap().0.x;
        assert_eq!(x, 99.0);
        assert_eq!(result.flush_seq, 10);
    }

    #[test]
    fn recover_empty_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        ManifestLog::recover::<4>(&log_path).unwrap();
        let codecs = CodecRegistry::new();

        let (mut result, _, _) = LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
        assert_eq!(result.flush_seq, 0);
        assert_eq!(result.world.query::<(&Pos,)>().count(), 0);
    }

    /// A despawn-heavy world: the free list and bumped generations must survive
    /// a flush+recover round-trip, or recycled indices collide with stale handles.
    #[test]
    fn recover_allocator_free_list_survives() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let entities: Vec<_> = (0..10)
            .map(|i| {
                world.spawn((Pos {
                    x: i as f32,
                    y: 0.0,
                },))
            })
            .collect();
        // Despawn a few — populates the free list and bumps generations.
        world.despawn(entities[2]);
        world.despawn(entities[5]);
        world.despawn(entities[7]);

        let (gen_before, free_before) = {
            let (g, f) = world.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };
        assert!(
            !free_before.is_empty(),
            "despawns must populate the free list"
        );

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_world(&world, &mut manifest, &mut log, dir.path(), 0, 10);

        let (mut result, _, _) = LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();

        assert_eq!(result.world.query::<(&Pos,)>().count(), 7);
        let (gen_after, free_after) = result.world.entity_allocator_state();
        assert_eq!(
            free_after,
            free_before.as_slice(),
            "free list must survive recovery"
        );
        assert_eq!(
            gen_after,
            gen_before.as_slice(),
            "generations must survive recovery"
        );
    }

    /// More than ~16K entity slots makes the allocator blob exceed `u16::MAX`
    /// bytes, forcing multi-page allocator metadata. Before the fix this failed
    /// to flush entirely.
    #[test]
    fn recover_large_world_allocator_multipage() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let n = 20_000usize;
        for i in 0..n {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }

        let gen_len = {
            let (g, f) = world.entity_allocator_state();
            assert!(
                allocator_meta::encode(g, f).len() > u16::MAX as usize,
                "test must exceed a single allocator page"
            );
            g.len()
        };

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_world(&world, &mut manifest, &mut log, dir.path(), 0, 10);

        let (mut result, _, _) = LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
        assert_eq!(result.world.query::<(&Pos,)>().count(), n);
        let (gen_after, _) = result.world.entity_allocator_state();
        assert_eq!(
            gen_after.len(),
            gen_len,
            "allocator generations must round-trip"
        );
    }

    /// Compaction has no live World, so it must carry the allocator page forward
    /// from the newest input run. Without that, recovery after compaction loses
    /// the free list (rebuilds generations from live entities only). Regression
    /// test for the C1 finding.
    #[test]
    fn recover_allocator_survives_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("manifest.log");
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let entities: Vec<_> = (0..20)
            .map(|i| {
                world.spawn((Pos {
                    x: i as f32,
                    y: 0.0,
                },))
            })
            .collect();
        world.despawn(entities[3]);
        world.despawn(entities[11]);
        world.despawn(entities[16]);

        let (gen_before, free_before) = {
            let (g, f) = world.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };
        assert!(!free_before.is_empty());

        // Accumulate COMPACTION_TRIGGER (4) L0 runs. Mutate between flushes so
        // each has dirty pages; the allocator state stays constant post-despawn.
        for seq in 0..4u64 {
            for (p,) in world.query::<(&mut Pos,)>() {
                p.y += seq as f32 + 1.0;
            }
            flush_world(
                &world,
                &mut manifest,
                &mut log,
                dir.path(),
                seq * 10,
                seq * 10 + 9,
            );
        }

        // Compact the L0 runs away — the allocator-bearing inputs are removed.
        let report = compact_one(&mut manifest, &mut log, dir.path()).unwrap();
        assert!(report.is_some(), "4 L0 runs must trigger compaction");

        let (mut result, _, _) = LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs).unwrap();
        assert_eq!(result.world.query::<(&Pos,)>().count(), 17);

        let (gen_after, free_after) = result.world.entity_allocator_state();
        assert_eq!(
            free_after,
            free_before.as_slice(),
            "free list must survive compaction"
        );
        assert_eq!(
            gen_after,
            gen_before.as_slice(),
            "generations must survive compaction"
        );
    }

    // ── Group-C Task-7 tests ──────────────────────────────────────────────

    /// (a) A despawned entity's sparse component must NOT resurface after
    /// flush+recover. The flush captures net state: despawn erases sparse
    /// storage before the snapshot is written.
    #[test]
    fn sparse_despawn_before_checkpoint_leaves_nothing() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Pos7a {
            x: f32,
            y: f32,
        }

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag7a(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos7a>("pos7a", &mut world).unwrap();
        codecs.register_as::<Tag7a>("tag7a", &mut world).unwrap();

        // Spawn entity, attach sparse, then despawn — all before flush.
        let e = world.spawn((Pos7a { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Tag7a>(e, Tag7a(5));
        world.despawn(e);

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        // Spawn a second entity so the world is non-empty and flush actually runs.
        world.spawn((Pos7a { x: 9.0, y: 9.0 },));
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        // Entity is dead: not alive, sparse must not resurrect.
        assert!(
            !recovered.is_alive(e),
            "despawned entity must not be alive after recovery"
        );
        assert!(
            recovered.get::<Tag7a>(e).is_none(),
            "sparse component of despawned entity must not appear after recovery"
        );
    }

    /// (b) More than u16::MAX bytes of sparse data forces ≥2 pages per
    /// component. Recovery must concatenate them and return every entry.
    #[test]
    fn sparse_multipage_round_trips() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Pos7b {
            x: f32,
            y: f32,
        }

        // Each sparse entry encodes to 8 (entity_bits) + 4 (value_len) + 4 (u32) = 16 bytes.
        // 5000 entries → 4 (header) + 5000 * 16 = 80_004 bytes > u16::MAX (65_535).
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag7b(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos7b>("pos7b", &mut world).unwrap();
        codecs.register_as::<Tag7b>("tag7b", &mut world).unwrap();

        let n: u32 = 5_000;
        let entities: Vec<_> = (0..n)
            .map(|i| {
                let e = world.spawn((Pos7b {
                    x: i as f32,
                    y: 0.0,
                },));
                world.insert_sparse::<Tag7b>(e, Tag7b(i));
                e
            })
            .collect();

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        // Check count via iter_sparse.
        let comp_id = recovered
            .component_id::<Tag7b>()
            .expect("Tag7b must be registered");
        let count = recovered
            .iter_sparse::<Tag7b>(comp_id)
            .map_or(0, std::iter::Iterator::count);
        assert_eq!(count, n as usize, "all {n} sparse entries must survive");

        // Spot-check a sample of values.
        for &i in &[0u32, 1, 99, 999, 2500, 4999] {
            let e = entities[i as usize];
            assert_eq!(
                recovered.get::<Tag7b>(e).copied(),
                Some(Tag7b(i)),
                "entity {i} sparse value mismatch"
            );
        }
    }

    /// (c) An index reused by a new entity after despawn must carry the new
    /// entity's sparse value, not the old one. Recovery must honour the entity
    /// generation embedded in sparse entries.
    #[test]
    fn sparse_generation_reuse_round_trips() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Pos7c {
            x: f32,
            y: f32,
        }

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag7c(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos7c>("pos7c", &mut world).unwrap();
        codecs.register_as::<Tag7c>("tag7c", &mut world).unwrap();

        // Spawn `a`, attach sparse, then despawn to free the index.
        let a = world.spawn((Pos7c { x: 1.0, y: 0.0 },));
        world.insert_sparse::<Tag7c>(a, Tag7c(1));
        world.despawn(a);

        // Spawn enough entities to force reuse of `a`'s index. The allocator
        // returns freed indices from a free-list (LIFO), so the first spawn
        // after a single despawn reuses the same index with a bumped generation.
        let b = world.spawn((Pos7c { x: 2.0, y: 0.0 },));

        // If the allocator didn't reuse immediately, spawn more until reuse.
        let b = if b.index() == a.index() {
            b
        } else {
            // Pad until the recycled slot shows up.
            let mut found = b;
            for _ in 0..32 {
                let e = world.spawn((Pos7c { x: 3.0, y: 0.0 },));
                if e.index() == a.index() {
                    found = e;
                    break;
                }
            }
            found
        };

        // Whatever happened, b must have a different generation than a.
        assert_ne!(
            b.generation(),
            a.generation(),
            "recycled entity must have bumped generation"
        );

        world.insert_sparse::<Tag7c>(b, Tag7c(2));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        assert_eq!(
            recovered.get::<Tag7c>(b).copied(),
            Some(Tag7c(2)),
            "new entity's sparse must survive"
        );
        assert!(
            recovered.get::<Tag7c>(a).is_none(),
            "old (stale generation) entity's sparse must not appear"
        );
    }

    /// (d) When two flushes cover the same entity's sparse component, the
    /// newer run's value must win over the older run's value.
    #[test]
    fn sparse_newest_run_wins() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Pos7d {
            x: f32,
            y: f32,
        }

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag7d(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos7d>("pos7d", &mut world).unwrap();
        codecs.register_as::<Tag7d>("tag7d", &mut world).unwrap();

        // First flush: entity e has Tag7d(1).
        let e = world.spawn((Pos7d { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Tag7d>(e, Tag7d(1));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("first flush");

        // Second flush: overwrite sparse to Tag7d(2). Spawn an extra archetype
        // entity so the second flush is non-trivially dirty.
        world.insert_sparse::<Tag7d>(e, Tag7d(2));
        world.spawn((Pos7d { x: 99.0, y: 99.0 },));
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(1u64), SeqNo::from(2u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("second flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        assert_eq!(
            recovered.get::<Tag7d>(e).copied(),
            Some(Tag7d(2)),
            "newer run's sparse value must win"
        );
    }

    /// (g) A sparse component removed from a still-alive entity between two
    /// flushes must NOT be resurrected: the newer run authoritatively has no
    /// sparse pages, so it supersedes the older run that did. Regression for the
    /// "newest run wins even when empty" rule.
    #[test]
    fn sparse_removed_before_second_flush_does_not_resurrect() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Pos7g {
            x: f32,
            y: f32,
        }

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag7g(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos7g>("pos7g", &mut world).unwrap();
        codecs.register_as::<Tag7g>("tag7g", &mut world).unwrap();

        // First flush: entity e (alive) has Tag7g(7).
        let e = world.spawn((Pos7g { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Tag7g>(e, Tag7g(7));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("first flush");

        // Remove the sparse component while e stays alive, then flush again
        // (spawn another entity so the second flush is dirty). The second run
        // therefore has NO sparse pages.
        let mut cs = minkowski::EnumChangeSet::new();
        cs.remove_sparse::<Tag7g>(&mut world, e);
        cs.apply(&mut world).unwrap();
        world.spawn((Pos7g { x: 9.0, y: 9.0 },));
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(1u64), SeqNo::from(2u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("second flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;

        assert_eq!(
            recovered.get::<Tag7g>(e),
            None,
            "removed sparse component must not be resurrected by recovery"
        );
    }

    /// Sparse components must survive compaction: the compactor carries the
    /// newest run's sparse pages forward verbatim (self-describing blobs).
    #[test]
    fn recover_restores_sparse_after_compaction() {
        #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct SparsePos2 {
            x: f32,
            y: f32,
        }

        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct Tag2(u32);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<SparsePos2>("sparse_pos2", &mut world)
            .unwrap();
        codecs.register_as::<Tag2>("tag2", &mut world).unwrap();

        let e = world.spawn((SparsePos2 { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Tag2>(e, Tag2(999));

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Flush 4 times to trigger compaction (L0 threshold is 4)
        for i in 0..4u64 {
            world.spawn((SparsePos2 {
                x: i as f32,
                y: 0.0,
            },));
            flush_and_record(
                &world,
                SeqRange::new(SeqNo::from(i), SeqNo::from(i + 1)).unwrap(),
                &mut manifest,
                &mut log,
                &lsm_dir,
                &codecs,
            )
            .unwrap()
            .expect("flush");
        }

        compact_one(&mut manifest, &mut log, &lsm_dir)
            .unwrap()
            .expect("compaction ran");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let recovered = result.world;
        assert_eq!(recovered.get::<Tag2>(e).copied(), Some(Tag2(999)));
    }

    /// Multi-page extension of the codex #4 regression: both a two-component
    /// archetype and a one-component (name-sorted-later) archetype span several
    /// pages. Exercises the per-`page_index` column reassembly in the rewritten
    /// materialize_world across differently-shaped archetypes.
    #[test]
    fn recover_multipage_nonuniform_archetypes() {
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct AComp(u32);
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct ZComp(u64);

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<AComp>("a_comp", &mut world).unwrap();
        codecs.register_as::<ZComp>("z_comp", &mut world).unwrap();

        // N = 522 > 2 * PAGE_SIZE (2 * 256 = 512), so each archetype spans 3+ pages.
        let n: u32 = 522;
        let mut both_entities = Vec::new();
        let mut z_entities = Vec::new();
        for i in 0..n {
            both_entities.push(world.spawn((AComp(i), ZComp(i as u64 + 1000))));
        }
        for i in 0..n {
            z_entities.push(world.spawn((ZComp(i as u64 + 5000),)));
        }

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let mut recovered = result.world;

        // Spot-check entities from the first, middle, and last pages of each archetype.
        for &i in &[0u32, 1, n / 2, n - 2, n - 1] {
            let be = both_entities[i as usize];
            assert_eq!(recovered.get::<AComp>(be).copied(), Some(AComp(i)));
            assert_eq!(
                recovered.get::<ZComp>(be).copied(),
                Some(ZComp(i as u64 + 1000))
            );

            let ze = z_entities[i as usize];
            assert_eq!(
                recovered.get::<ZComp>(ze).copied(),
                Some(ZComp(i as u64 + 5000))
            );
            assert_eq!(recovered.get::<AComp>(ze), None);
        }
        assert_eq!(recovered.query::<(&AComp,)>().count(), n as usize);
        assert_eq!(recovered.query::<(&ZComp,)>().count(), (2 * n) as usize);
    }

    /// HIGH finding regression: when an archetype shrinks across flushes (despawns
    /// reduce the row count so the highest page is skipped by `flush` because
    /// `row_count == 0`), the stale higher-index pages from the older run survive
    /// on disk. Without the fix, `build_allocator_state` overwrites the
    /// authoritative persisted allocator's bumped (dead) generations with the
    /// stale on-disk (alive) generations, so `import_page` resurrects the
    /// despawned entities with their pre-despawn component bytes.
    ///
    /// This test must FAIL on the unfixed code (resurrected entities appear alive
    /// and the recovered count is too high) and PASS after the fix (persisted
    /// allocator is the single source of truth; stale rows are dropped).
    #[test]
    fn recover_archetype_shrink_does_not_resurrect_dead_entities() {
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct ShrinkPos {
            x: f32,
            y: f32,
        }

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<ShrinkPos>("shrink_pos", &mut world)
            .unwrap();

        // 600 entities → 3 pages of 256 (pages 0, 1, 2).
        let n_spawn: u32 = 600;
        let mut all = Vec::with_capacity(n_spawn as usize);
        for i in 0..n_spawn {
            all.push(world.spawn((ShrinkPos {
                x: i as f32,
                y: 0.0,
            },)));
        }

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();
        // Flush 1: writes pages 0, 1, 2 (seq_hi = 5).
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(5u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush 1");

        // Despawn 500 entities → archetype shrinks to 100 rows (1 page).
        // swap_remove touches rows on pages 1 and 2 (marking them dirty), but
        // at flush time `arch_len <= start_row` so those pages are skipped
        // (row_count == 0 → continue). Only page 0 is rewritten.
        let to_despawn: Vec<Entity> = all[..500].to_vec();
        for e in &to_despawn {
            world.despawn(*e);
        }
        let survivors: Vec<Entity> = all[500..].to_vec();
        assert_eq!(world.query::<(&ShrinkPos,)>().count(), 100);

        // Capture the authoritative allocator state after despawns.
        let (gen_before, free_before) = {
            let (g, f) = world.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };
        assert!(
            !free_before.is_empty(),
            "despawns must populate the free list"
        );

        // Flush 2: writes only page 0 (seq_hi = 10). Pages 1, 2 from flush 1
        // remain on disk verbatim — the stale entity generations they carry
        // are the resurrection vector.
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(5u64), SeqNo::from(10u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush 2");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let mut recovered = result.world;

        // Exact entity count: 100 survivors, zero resurrected.
        assert_eq!(
            recovered.query::<(&ShrinkPos,)>().count(),
            100,
            "despawned entities must not be resurrected by stale pages"
        );

        // Every despawned entity must be dead and have no component.
        for e in &to_despawn {
            assert!(
                !recovered.is_alive(*e),
                "despawned entity {e:?} resurrected (alive after recovery)"
            );
            assert!(
                recovered.get::<ShrinkPos>(*e).is_none(),
                "despawned entity {e:?} has a component after recovery"
            );
        }

        // Every survivor must retain its original component value.
        for (i, &e) in survivors.iter().enumerate() {
            assert_eq!(
                recovered.get::<ShrinkPos>(e).copied(),
                Some(ShrinkPos {
                    x: (500 + i) as f32,
                    y: 0.0,
                }),
                "survivor {e:?} component corrupted"
            );
        }

        // The free list and generations must match the authoritative allocator
        // state captured before flush 2 — stale on-disk generations must not
        // downgrade the persisted allocator.
        let (gen_after, free_after) = recovered.entity_allocator_state();
        assert_eq!(
            free_after, free_before,
            "free list must match the persisted allocator, not be rebuilt from stale rows"
        );
        assert_eq!(
            gen_after, gen_before,
            "generations must match the persisted allocator, not be downgraded by stale rows"
        );
    }

    /// Variant: shrink the archetype AND compact the runs away, then recover.
    /// Covers the compaction carry-forward interaction with the dead-row filter:
    /// compaction may carry stale entity pages forward verbatim, so the
    /// recovery-side filter is the only safety net.
    #[test]
    fn recover_archetype_shrink_survives_compaction() {
        #[derive(Clone, Copy, PartialEq, Debug, Archive, Serialize, Deserialize)]
        #[repr(C)]
        struct ShrinkPos2 {
            x: f32,
            y: f32,
        }

        let dir = tempfile::tempdir().unwrap();
        let lsm_dir = dir.path().join("lsm");
        let log_path = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<ShrinkPos2>("shrink_pos2", &mut world)
            .unwrap();

        let mut all = Vec::new();
        for i in 0..600u32 {
            all.push(world.spawn((ShrinkPos2 {
                x: i as f32,
                y: 0.0,
            },)));
        }

        let (mut manifest, mut log) = ManifestLog::recover::<4>(&log_path).unwrap();

        // Flush 1: full 600-entity archetype (3 pages).
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush 1");

        // Despawn 500 → shrink to 100 (1 page).
        for e in &all[..500] {
            world.despawn(*e);
        }
        let survivors: Vec<Entity> = all[500..].to_vec();

        // Three more dirty flushes to trigger compaction (L0 threshold = 4).
        // Mutate survivors each time so each flush is non-trivially dirty.
        for seq in 1..4u64 {
            for &e in &survivors {
                let p = world.get_mut::<ShrinkPos2>(e).unwrap();
                p.y += 1.0;
            }
            flush_and_record(
                &world,
                SeqRange::new(SeqNo::from(seq), SeqNo::from(seq + 1)).unwrap(),
                &mut manifest,
                &mut log,
                &lsm_dir,
                &codecs,
            )
            .unwrap()
            .expect("flush");
        }

        // Compact the L0 runs away — stale pages from flush 1 may be carried
        // forward by the compactor.
        let report = compact_one(&mut manifest, &mut log, &lsm_dir).unwrap();
        assert!(report.is_some(), "4 L0 runs must trigger compaction");

        let (result, _, _) = LsmRecovery::recover::<4>(&lsm_dir, &log_path, &codecs).unwrap();
        let mut recovered = result.world;

        assert_eq!(
            recovered.query::<(&ShrinkPos2,)>().count(),
            100,
            "despawned entities must not resurrect after compaction"
        );
        for e in &all[..500] {
            assert!(!recovered.is_alive(*e), "despawned {e:?} resurrected");
        }
        // Survivors retain the last mutation (y = 3.0).
        for &e in &survivors {
            assert_eq!(
                recovered.get::<ShrinkPos2>(e).map(|p| p.y),
                Some(3.0),
                "survivor {e:?} value wrong after compaction"
            );
        }
    }
}
