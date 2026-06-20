//! Bulk restoration of a [`World`] from pre-serialized column bytes.
//!
//! Recovery reconstructs committed state at *column* granularity instead of
//! row-by-row: one memcpy per (archetype, component) page rather than ~3 heap
//! allocations per entity. The API is resolve-once / push-many, designed so most
//! illegal states are unrepresentable:
//!
//! 1. [`World::import_target`] validates the component set (strictly sorted,
//!    every component registered) and resolves or creates the archetype,
//!    returning an owned [`ImportTarget`] token whose existence proves those
//!    invariants.
//! 2. [`ImportTarget::page`] validates a page's column count and per-column byte
//!    length, deriving `row_count` from `entities.len()` (so an
//!    entities-vs-row-count mismatch is unrepresentable, not checked).
//! 3. [`World::import_page`] (the only `unsafe` step) appends the entities and
//!    bulk-copies the column bytes.
//!
//! # Ordering contract
//! The caller MUST register components, then
//! [`World::restore_allocator_state`] (so each imported entity's generation is
//! live), and only then call [`World::import_page`], which checks `is_alive`.
//! `import_page` does NOT advance ticks or mark pages dirty — recovery
//! reconstructs already-committed state, it is not a mutation.
//!
//! Note: the up-front entity checks are all-or-nothing, but a panic *during*
//! the column-append block (e.g. allocation failure) leaves the archetype torn;
//! recovery treats any failure as fatal and discards the partially-built World.

use std::fmt;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::storage::archetype::ArchetypeId;
use crate::world::{EntityLocation, World};

/// Errors from the bulk-import API.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportError {
    /// `component_ids` was not strictly ascending (the canonical archetype key).
    NotSorted,
    /// A component id has no registered layout in this world.
    UnregisteredComponent(ComponentId),
    /// The page supplied a different number of columns than the target has
    /// components.
    ColumnCountMismatch { expected: usize, got: usize },
    /// A column's byte length is not `row_count * item_size`.
    ColumnLengthMismatch {
        component: ComponentId,
        expected: usize,
        got: usize,
    },
    /// An imported entity is already placed in an archetype.
    AlreadyPlaced(Entity),
    /// An imported entity is not alive (generation mismatch / freed slot).
    DeadEntity(Entity),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::NotSorted => {
                write!(
                    f,
                    "component_ids must be strictly ascending (canonical archetype key)"
                )
            }
            ImportError::UnregisteredComponent(id) => {
                write!(f, "component id {id} is not registered in this world")
            }
            ImportError::ColumnCountMismatch { expected, got } => {
                write!(
                    f,
                    "import page has {got} columns but target archetype has {expected} components"
                )
            }
            ImportError::ColumnLengthMismatch {
                component,
                expected,
                got,
            } => {
                write!(
                    f,
                    "column for component {component} has {got} bytes, expected {expected} (row_count * item_size)"
                )
            }
            ImportError::AlreadyPlaced(e) => {
                write!(f, "entity {e:?} is already placed in an archetype")
            }
            ImportError::DeadEntity(e) => {
                write!(
                    f,
                    "entity {e:?} is not alive — restore allocator state before importing"
                )
            }
        }
    }
}

impl std::error::Error for ImportError {}

/// A resolved import destination: a specific archetype plus the sorted component
/// set and per-component item sizes. Owned — holds no borrow of the `World`, so
/// it can outlive intermediate `&mut World` calls and feed many pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportTarget {
    pub(crate) arch_id: ArchetypeId,
    component_ids: Vec<ComponentId>,
    item_sizes: Vec<usize>,
}

impl ImportTarget {
    /// The component ids this target writes, in canonical (ascending) order.
    /// `ImportPage` columns must be supplied in this same order.
    pub fn component_ids(&self) -> &[ComponentId] {
        &self.component_ids
    }

    /// Validate and build one page. `row_count` is derived from `entities.len()`.
    /// Checks the column count and each column's byte length.
    pub fn page<'a>(
        &'a self,
        entities: &'a [Entity],
        columns: &'a [&'a [u8]],
    ) -> Result<ImportPage<'a>, ImportError> {
        if columns.len() != self.component_ids.len() {
            return Err(ImportError::ColumnCountMismatch {
                expected: self.component_ids.len(),
                got: columns.len(),
            });
        }
        let row_count = entities.len();
        for (i, col) in columns.iter().enumerate() {
            let expected = row_count * self.item_sizes[i];
            if col.len() != expected {
                return Err(ImportError::ColumnLengthMismatch {
                    component: self.component_ids[i],
                    expected,
                    got: col.len(),
                });
            }
        }
        Ok(ImportPage {
            target: self,
            entities,
            columns,
        })
    }
}

/// A validated page of rows for one archetype: the entity handles plus one
/// native-byte column slice per component, in the target's component order.
pub struct ImportPage<'a> {
    target: &'a ImportTarget,
    entities: &'a [Entity],
    columns: &'a [&'a [u8]],
}

impl World {
    /// Resolve (or create) the archetype for `component_ids` and return an owned
    /// import token. `component_ids` must be strictly ascending and every id must
    /// be registered.
    pub fn import_target(
        &mut self,
        component_ids: &[ComponentId],
    ) -> Result<ImportTarget, ImportError> {
        self.drain_orphans();

        // Strictly ascending (also rejects duplicates).
        for w in component_ids.windows(2) {
            if w[0] >= w[1] {
                return Err(ImportError::NotSorted);
            }
        }
        // Every component registered; capture item sizes.
        let mut item_sizes = Vec::with_capacity(component_ids.len());
        for &id in component_ids {
            if id >= self.components.len() {
                return Err(ImportError::UnregisteredComponent(id));
            }
            item_sizes.push(self.components.info(id).layout.size());
        }

        let arch_id = self
            .archetypes
            .get_or_create(component_ids, &self.components, &self.pool);

        Ok(ImportTarget {
            arch_id,
            component_ids: component_ids.to_vec(),
            item_sizes,
        })
    }

    /// Append a validated page of committed rows into the target archetype.
    ///
    /// # Safety
    /// Each column slice must be the native (in-memory) byte image of
    /// `entities.len()` consecutive values of the corresponding component type,
    /// and ownership of those values is moved into the archetype column (which
    /// holds the component's `drop_fn`). This holds for POD columns (no drop,
    /// trivial ownership) and for heap columns reconstructed by decoding rkyv
    /// bytes into native values whose ownership is transferred into the buffer.
    /// LSM recovery guarantees the column kind matches the codec (RawCopy vs
    /// Serialized) and validates each source page's CRC on read.
    ///
    /// - The entities in `page.entities` must be unique and must not already be
    ///   placed in any archetype (across this and prior imported pages). The LSM
    ///   recovery driver guarantees this — each entity appears in exactly one
    ///   archetype page. A duplicate would create an archetype row with no matching
    ///   `entity_locations` entry (a dangling row).
    pub unsafe fn import_page(&mut self, page: &ImportPage<'_>) -> Result<(), ImportError> {
        let arch_id = page.target.arch_id;

        debug_assert!(
            {
                let mut seen = std::collections::HashSet::with_capacity(page.entities.len());
                page.entities.iter().all(|e| seen.insert(*e))
            },
            "import_page: page.entities must be unique (caller precondition)"
        );

        // Validate all entities up front so the append is all-or-nothing.
        for &e in page.entities {
            if !self.is_alive(e) {
                return Err(ImportError::DeadEntity(e));
            }
            if self.is_placed(e) {
                return Err(ImportError::AlreadyPlaced(e));
            }
        }

        let count = page.entities.len();
        let base_row = {
            let archetype = &mut self.archetypes.archetypes[arch_id.0];
            let base_row = archetype.entities.len();
            for (col_idx, &comp_id) in page.target.component_ids.iter().enumerate() {
                let col = archetype
                    .column_index(comp_id)
                    .expect("import_target proved this component is in the archetype");
                // SAFETY: `ImportTarget::page` checked `columns[col_idx].len() ==
                // count * item_size`; the bytes are a native image of a
                // raw-copyable type (caller's safety contract).
                unsafe {
                    archetype.columns[col].append_bytes_unchecked(page.columns[col_idx], count);
                }
            }
            archetype.entities.extend_from_slice(page.entities);
            archetype.debug_assert_consistent();
            base_row
        };

        for (i, &e) in page.entities.iter().enumerate() {
            let idx = e.index() as usize;
            if idx >= self.entity_locations.len() {
                self.entity_locations.resize(idx + 1, None);
            }
            self.entity_locations[idx] = Some(EntityLocation {
                archetype_id: arch_id,
                row: base_row + i,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct A(u32);
    #[derive(Clone, Copy, PartialEq, Debug)]
    struct B(u64);

    /// Extract a source world's archetype-0 columns and re-import them into a
    /// fresh world; component values and entity handles must round-trip.
    #[test]
    fn import_round_trips_single_archetype() {
        let mut src = World::new();
        let e0 = src.spawn((A(1), B(10)));
        let e1 = src.spawn((A(2), B(20)));
        let e2 = src.spawn((A(3), B(30)));
        let a_id = src.component_id::<A>().unwrap();
        let b_id = src.component_id::<B>().unwrap();

        let entities = [e0, e1, e2];
        let a_bytes = src.column_page_bytes(0, a_id, 0, 3).unwrap().to_vec();
        let b_bytes = src.column_page_bytes(0, b_id, 0, 3).unwrap().to_vec();
        let (gens, free) = {
            let (g, f) = src.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };

        let mut dst = World::new();
        let da = dst.register_component::<A>();
        let db = dst.register_component::<B>();
        dst.restore_allocator_state(gens, free);

        let mut ids = [da, db];
        ids.sort_unstable();
        let target = dst.import_target(&ids).unwrap();
        let cols: [&[u8]; 2] = [&a_bytes, &b_bytes];
        let page = target.page(&entities, &cols).unwrap();
        unsafe { dst.import_page(&page).unwrap() };

        assert_eq!(dst.get::<A>(e0).copied(), Some(A(1)));
        assert_eq!(dst.get::<A>(e2).copied(), Some(A(3)));
        assert_eq!(dst.get::<B>(e2).copied(), Some(B(30)));
        assert_eq!(dst.query::<(&A,)>().count(), 3);
    }

    #[test]
    fn import_target_rejects_unsorted() {
        let mut w = World::new();
        let a = w.register_component::<A>();
        let b = w.register_component::<B>();
        let mut ids = [a, b];
        ids.sort_unstable();
        ids.reverse();
        assert_eq!(w.import_target(&ids), Err(ImportError::NotSorted));
    }

    #[test]
    fn import_target_rejects_unregistered() {
        let mut w = World::new();
        let _ = w.register_component::<A>();
        assert_eq!(
            w.import_target(&[999]),
            Err(ImportError::UnregisteredComponent(999))
        );
    }

    #[test]
    fn page_rejects_column_count_mismatch() {
        let mut w = World::new();
        let a = w.register_component::<A>();
        let b = w.register_component::<B>();
        let mut ids = [a, b];
        ids.sort_unstable();
        let target = w.import_target(&ids).unwrap();
        let entities: [Entity; 0] = [];
        let cols: [&[u8]; 1] = [&[]];
        assert_eq!(
            target.page(&entities, &cols).map(|_| ()),
            Err(ImportError::ColumnCountMismatch {
                expected: 2,
                got: 1
            })
        );
    }

    #[test]
    fn page_rejects_column_length_mismatch() {
        let mut src = World::new();
        let e = src.spawn((A(0),));
        let (g, f) = {
            let (g, f) = src.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };
        let mut w2 = World::new();
        let a2 = w2.register_component::<A>();
        w2.restore_allocator_state(g, f);
        let target2 = w2.import_target(&[a2]).unwrap();
        let entities = [e];
        let bad: [&[u8]; 1] = [&[0u8, 0, 0]]; // 3 bytes, expected 4
        assert_eq!(
            target2.page(&entities, &bad).map(|_| ()),
            Err(ImportError::ColumnLengthMismatch {
                component: a2,
                expected: 4,
                got: 3
            })
        );
    }

    #[test]
    fn import_page_rejects_dead_entity() {
        let mut src = World::new();
        let e = src.spawn((A(5),));
        let a_id = src.component_id::<A>().unwrap();
        let bytes = src.column_page_bytes(0, a_id, 0, 1).unwrap().to_vec();

        let mut dst = World::new();
        let a = dst.register_component::<A>();
        let target = dst.import_target(&[a]).unwrap();
        let entities = [e];
        let cols: [&[u8]; 1] = [&bytes];
        let page = target.page(&entities, &cols).unwrap();
        assert_eq!(
            unsafe { dst.import_page(&page) },
            Err(ImportError::DeadEntity(e))
        );
    }

    #[test]
    fn import_page_rejects_double_import() {
        let mut src = World::new();
        let e = src.spawn((A(9),));
        let a_id = src.component_id::<A>().unwrap();
        let bytes = src.column_page_bytes(0, a_id, 0, 1).unwrap().to_vec();
        let (g, f) = {
            let (g, f) = src.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };

        let mut dst = World::new();
        let a = dst.register_component::<A>();
        dst.restore_allocator_state(g, f);
        let target = dst.import_target(&[a]).unwrap();
        let entities = [e];
        let cols: [&[u8]; 1] = [&bytes];

        let page = target.page(&entities, &cols).unwrap();
        unsafe { dst.import_page(&page).unwrap() };
        let page2 = target.page(&entities, &cols).unwrap();
        assert_eq!(
            unsafe { dst.import_page(&page2) },
            Err(ImportError::AlreadyPlaced(e))
        );
    }

    /// Two import_page calls into the SAME archetype must append correctly with a
    /// non-zero base row — the case the single-page round-trip doesn't exercise
    /// (and the one the recovery driver relies on for multi-page archetypes).
    #[test]
    fn import_two_pages_same_archetype() {
        let mut src = World::new();
        let e0 = src.spawn((A(1),));
        let e1 = src.spawn((A(2),));
        let e2 = src.spawn((A(3),));
        let a_id = src.component_id::<A>().unwrap();
        let (g, f) = {
            let (g, f) = src.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };
        // Per-entity native bytes.
        let b0 = src.column_page_bytes(0, a_id, 0, 1).unwrap().to_vec();
        let b1 = src.column_page_bytes(0, a_id, 1, 1).unwrap().to_vec();
        let b2 = src.column_page_bytes(0, a_id, 2, 1).unwrap().to_vec();

        let mut dst = World::new();
        let a = dst.register_component::<A>();
        dst.restore_allocator_state(g, f);
        let target = dst.import_target(&[a]).unwrap();

        // Page 1: e0, e1.
        let mut first = b0.clone();
        first.extend_from_slice(&b1);
        let ents1 = [e0, e1];
        let cols1 = [first.as_slice()];
        let p1 = target.page(&ents1, &cols1).unwrap();
        unsafe { dst.import_page(&p1).unwrap() };
        // Page 2: e2 (base_row = 2).
        let ents2 = [e2];
        let cols2 = [b2.as_slice()];
        let p2 = target.page(&ents2, &cols2).unwrap();
        unsafe { dst.import_page(&p2).unwrap() };

        assert_eq!(dst.get::<A>(e0).copied(), Some(A(1)));
        assert_eq!(dst.get::<A>(e1).copied(), Some(A(2)));
        assert_eq!(dst.get::<A>(e2).copied(), Some(A(3)));
        assert_eq!(dst.query::<(&A,)>().count(), 3);
    }

    /// A zero-sized component column imports through the seam: page() computes
    /// expected length 0, append_bytes_unchecked only advances len.
    #[test]
    fn import_zst_column() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct Marker;

        let mut src = World::new();
        let e = src.spawn((Marker,));
        let (g, f) = {
            let (g, f) = src.entity_allocator_state();
            (g.to_vec(), f.to_vec())
        };

        let mut dst = World::new();
        let m = dst.register_component::<Marker>();
        dst.restore_allocator_state(g, f);
        let target = dst.import_target(&[m]).unwrap();
        let empty: &[u8] = &[];
        let ents = [e];
        let cols = [empty];
        let page = target.page(&ents, &cols).unwrap();
        unsafe { dst.import_page(&page).unwrap() };

        assert!(dst.get::<Marker>(e).is_some());
        assert_eq!(dst.query::<(&Marker,)>().count(), 1);
    }

    /// An empty page (zero entities) is a clean no-op.
    #[test]
    fn import_empty_page_is_noop() {
        let mut w = World::new();
        let a = w.register_component::<A>();
        let target = w.import_target(&[a]).unwrap();
        let entities: [Entity; 0] = [];
        let empty: &[u8] = &[];
        let cols = [empty];
        let page = target.page(&entities, &cols).unwrap();
        unsafe { w.import_page(&page).unwrap() };
        assert_eq!(w.query::<(&A,)>().count(), 0);
    }
}
