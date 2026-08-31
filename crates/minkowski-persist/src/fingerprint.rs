//! World fingerprint: a deterministic, order-independent hash of world state.
//!
//! Stage 4.0 substrate deliverable (spec §3). Two worlds are equal if and only
//! if their fingerprints match. Used by the convergence test; stage 4.1 reuses
//! it for replay-equals-transfer proofs and divergence refusal.
//!
//! Properties:
//! - Archetype creation order does not matter (archetypes are keyed by their
//!   sorted set of stable component names).
//! - Entity row order within an archetype does not matter (entities are keyed
//!   by packed `(index, generation)` bits).
//! - Component values hash through the codec path (`serialize_by_type`), so
//!   heap-backed components (`String`, `Vec`, `BlobRef`) hash by content, not
//!   by pointer bytes.
//! - Sparse components are not fingerprinted (dense-only for 4.0-a).
//!
//! The hash is deterministic within one binary. Cross-version comparison is
//! not a goal.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use minkowski::{ComponentId, World};
use minkowski_lsm::codec::CodecRegistry;

/// Per-archetype state, keyed by the archetype's sorted stable-name set so
/// archetype creation order cannot influence the result:
/// `arch_key -> entity bits -> component name -> value hash`.
type ArchetypeMap = BTreeMap<String, BTreeMap<u64, BTreeMap<String, u64>>>;

/// Resolve a component's stable name via its `TypeId` so the fingerprint is
/// independent of per-world numeric component ids (recovered worlds re-register
/// types and may compact ids).
fn name_by_type(
    world: &World,
    codecs: &CodecRegistry,
    comp_id: ComponentId,
) -> Result<String, String> {
    let type_id = world
        .component_type_id(comp_id)
        .ok_or_else(|| format!("unregistered component {comp_id:?}"))?;
    codecs
        .stable_name_by_type(type_id)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("no codec for component {comp_id:?}"))
}

/// Compute the world fingerprint. Every dense component in the world must
/// have a registered codec (the same gate flush enforces).
pub fn world_fingerprint(world: &World, codecs: &CodecRegistry) -> Result<u64, String> {
    let mut archetypes: ArchetypeMap = BTreeMap::new();

    for arch_idx in 0..world.archetype_count() {
        let comp_ids: Vec<ComponentId> = world.archetype_component_ids(arch_idx).to_vec();
        let mut names: Vec<String> = Vec::with_capacity(comp_ids.len());
        for id in &comp_ids {
            names.push(name_by_type(world, codecs, *id)?);
        }
        names.sort();
        let arch_key = names.join("\u{1}");
        let entities = world.archetype_entities(arch_idx).to_vec();
        let entry = archetypes.entry(arch_key).or_default();

        for (row, entity) in entities.iter().enumerate() {
            let values = entry.entry(entity.to_bits()).or_default();
            for &comp_id in &comp_ids {
                let name = name_by_type(world, codecs, comp_id)?;
                let type_id = world
                    .component_type_id(comp_id)
                    .ok_or_else(|| format!("unregistered component {comp_id:?}"))?;
                let page = world
                    .column_page_bytes(arch_idx, comp_id, row, 1)
                    .ok_or_else(|| format!("column bytes unavailable at row {row}"))?;
                let mut hasher = std::hash::DefaultHasher::new();
                // SAFETY: the page bytes are a valid native value of this
                // component type at this row, from a live column — the same
                // guarantee `column_page_bytes` gives the flush path.
                let mut buf = Vec::new();
                match unsafe { codecs.serialize_by_type(type_id, page.as_ptr(), &mut buf) } {
                    Some(Ok(())) => buf.hash(&mut hasher),
                    Some(Err(e)) => return Err(format!("codec serialize failed: {e}")),
                    None => return Err("no codec for fingerprinted component".to_owned()),
                }
                values.insert(name.clone(), hasher.finish());
            }
        }
    }

    let mut hasher = std::hash::DefaultHasher::new();
    for (arch_key, entities) in &archetypes {
        arch_key.hash(&mut hasher);
        for (entity_bits, values) in entities {
            entity_bits.hash(&mut hasher);
            for (name, value_hash) in values {
                name.hash(&mut hasher);
                value_hash.hash(&mut hasher);
            }
        }
    }
    Ok(hasher.finish())
}
