//! Restore a [`World`] from LSM sorted runs recorded in the manifest.

use std::alloc::Layout;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use minkowski::{ComponentId, Entity, EnumChangeSet, World};

use crate::allocator_meta;
use crate::codec::CodecRegistry;
use crate::error::LsmError;
use crate::format::{ALLOCATOR_SLOT, ENTITY_SLOT, META_ARCH_ID, PAGE_SIZE};
use crate::manifest::{LsmManifest, SortedRunMeta};
use crate::manifest_log::ManifestLog;
use crate::manifest_ops::cleanup_orphans;
use crate::reader::SortedRunReader;
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
type PageKey = (ArchetypeSig, u16, u16);

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

            for arch_id in reader.archetype_ids() {
                if arch_id == META_ARCH_ID {
                    continue;
                }
                let sig = archetype_signature(&reader, arch_id)?;

                for slot in reader.component_slots_for_arch(arch_id) {
                    for result in reader.slot_pages(arch_id, slot) {
                        let (page_index, page) = result?;
                        reader.validate_page_crc(&page)?;
                        let item_size = reader.schema().entry_for_slot(slot).map_or(8, |e| {
                            if slot == ENTITY_SLOT {
                                8
                            } else {
                                e.item_size as usize
                            }
                        });
                        let payload_len = page.header().row_count as usize * item_size;
                        store_page(
                            &mut pages,
                            (sig.clone(), slot, page_index),
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
                        (sig.clone(), ENTITY_SLOT, page_index),
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
        let world = materialize_world(pages, allocator.as_ref(), &component_layouts, codecs)?;
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
            let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
            world.register_component_raw(leaked, layout)
        };
        name_to_id.insert(name.clone(), id);
    }

    let mut by_sig: BTreeMap<ArchetypeSig, BTreeMap<(u16, u16), StoredPage>> = BTreeMap::new();
    for (key, page) in pages {
        let (sig, slot, page_index) = key;
        by_sig
            .entry(sig)
            .or_default()
            .insert((slot, page_index), page);
    }

    let mut changeset = EnumChangeSet::new();

    if let Some(alloc) = &allocator {
        world.restore_allocator_state(alloc.generations.clone(), alloc.free_list.clone());
    }

    for (sig, sig_pages) in by_sig {
        let comp_ids: Vec<ComponentId> = sig
            .iter()
            .map(|name| {
                name_to_id
                    .get(name)
                    .copied()
                    .ok_or_else(|| LsmError::Format(format!("unregistered component {name}")))
            })
            .collect::<Result<_, _>>()?;

        let max_row = sig_pages
            .iter()
            .filter(|((slot, _), _)| *slot == ENTITY_SLOT)
            .map(|((_, page_index), stored)| {
                *page_index as usize * PAGE_SIZE + stored.row_count as usize
            })
            .max()
            .unwrap_or(0);

        for row in 0..max_row {
            let page_index = (row / PAGE_SIZE) as u16;
            let row_in_page = row % PAGE_SIZE;

            let entity_page = sig_pages.get(&(ENTITY_SLOT, page_index)).ok_or_else(|| {
                LsmError::Format("missing entity page during recovery".to_owned())
            })?;
            if row_in_page >= entity_page.row_count as usize {
                continue;
            }
            let entity_offset = row_in_page * 8;
            let entity_bits = u64::from_le_bytes(
                entity_page.data[entity_offset..entity_offset + 8]
                    .try_into()
                    .expect("8 bytes"),
            );
            let entity = Entity::from_bits(entity_bits);

            let mut raw_components: Vec<(ComponentId, Vec<u8>, Layout)> = Vec::new();
            for (slot_idx, comp_name) in sig.iter().enumerate() {
                let slot = slot_idx as u16;
                let comp_id = comp_ids[slot_idx];
                let (size, align) = component_layouts[comp_name];
                let layout = Layout::from_size_align(size, align).map_err(|_| {
                    LsmError::Format(format!("invalid layout for component {comp_name}"))
                })?;
                let col_page_index = (row / PAGE_SIZE) as u16;
                let col_row_in_page = row % PAGE_SIZE;
                let col_page = sig_pages.get(&(slot, col_page_index)).ok_or_else(|| {
                    LsmError::Format(format!(
                        "missing component page slot={slot} page_index={col_page_index}"
                    ))
                })?;
                if col_row_in_page >= col_page.row_count as usize {
                    return Err(LsmError::Format(
                        "component page shorter than entity page".to_owned(),
                    ));
                }
                let item_size = layout.size();
                let offset = col_row_in_page * item_size;
                let end = offset + item_size;
                raw_components.push((comp_id, col_page.data[offset..end].to_vec(), layout));
            }

            let ptrs: Vec<_> = raw_components
                .iter()
                .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                .collect();
            changeset.record_spawn(entity, &ptrs);
        }
    }

    changeset
        .apply(&mut world)
        .map_err(|e| LsmError::Format(format!("recovery changeset apply failed: {e}")))?;

    reconcile_allocator(&mut world, allocator);

    Ok(world)
}

fn reconcile_allocator(world: &mut World, stored: Option<&StoredAllocator>) {
    let mut max_index = 0u32;
    for arch_idx in 0..world.archetype_count() {
        for &entity in world.archetype_entities(arch_idx) {
            max_index = max_index.max(entity.index());
        }
    }

    let mut generations = stored.map(|a| a.generations.clone()).unwrap_or_default();
    if generations.len() <= max_index as usize {
        generations.resize(max_index as usize + 1, 0);
    }

    for arch_idx in 0..world.archetype_count() {
        for &entity in world.archetype_entities(arch_idx) {
            generations[entity.index() as usize] = entity.generation();
        }
    }

    let free_list = stored.map(|a| a.free_list.clone()).unwrap_or_default();
    world.restore_allocator_state(generations, free_list);
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
        flush_and_record(
            world,
            SeqRange::new(SeqNo::from(lo), SeqNo::from(hi)).unwrap(),
            manifest,
            log,
            dir,
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
}
