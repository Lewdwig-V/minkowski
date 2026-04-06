use std::alloc::Layout;

use minkowski::World;
use minkowski_lsm::error::LsmError;
use minkowski_lsm::format::{ENTITY_SLOT, PAGE_SIZE};
use minkowski_lsm::reader::SortedRunReader;
use minkowski_lsm::schema::SchemaEntry;
use minkowski_lsm::writer::flush;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vel {
    dx: f32,
    dy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Health(u32);

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Flush the world and return (tempdir, path, reader).
/// The tempdir must be kept alive for the duration of the test.
fn flush_and_open(world: &World) -> (tempfile::TempDir, SortedRunReader) {
    let dir = tempfile::tempdir().unwrap();
    let path = flush(world, (0, 100), dir.path())
        .unwrap()
        .expect("flush should produce a file");
    let reader = SortedRunReader::open(&path).unwrap();
    (dir, reader)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn round_trip_single_archetype() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((
            Pos {
                x: i as f32,
                y: i as f32 * 2.0,
            },
            Vel {
                dx: i as f32 * 0.1,
                dy: -(i as f32),
            },
        ));
    }

    let (_dir, reader) = flush_and_open(&world);

    // There is one archetype (arch_id = 0).
    let schema = reader.schema();
    assert_eq!(schema.len(), 2);

    // Check each component page.
    for entry in schema.entries() {
        let page = reader
            .get_page(0, entry.slot(), 0)
            .expect("get_page should not error")
            .expect("page must exist for slot");
        assert_eq!(page.header().row_count, 5);

        // Reconstruct expected bytes from world.
        let comp_id = world
            .archetype_component_ids(0)
            .iter()
            .find(|&&cid| world.component_name(cid).unwrap() == entry.name())
            .copied()
            .expect("component must exist in archetype");

        let expected = world.column_page_bytes(0, comp_id, 0, 5).unwrap();
        assert_eq!(
            &page.data()[..expected.len()],
            expected,
            "data mismatch for component {}",
            entry.name()
        );

        reader.validate_page_crc(&page).unwrap();
    }

    // Check entity page.
    let entity_page = reader
        .get_page(0, ENTITY_SLOT, 0)
        .expect("get_page should not error")
        .expect("entity page must exist");
    assert_eq!(entity_page.header().row_count, 5);

    let entities = world.archetype_entities(0);
    for (i, &e) in entities.iter().enumerate() {
        let offset = i * 8;
        let stored = u64::from_le_bytes(entity_page.data()[offset..offset + 8].try_into().unwrap());
        assert_eq!(stored, e.to_bits(), "entity mismatch at index {i}");
    }
}

#[test]
fn round_trip_partial_page() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }

    let (_dir, reader) = flush_and_open(&world);
    let schema = reader.schema();
    assert_eq!(schema.len(), 1);

    let slot = schema.slot_for(std::any::type_name::<Pos>()).unwrap();
    let page = reader
        .get_page(0, slot, 0)
        .expect("get_page should not error")
        .expect("page must exist");

    // Verify row count.
    assert_eq!(page.header().row_count, 100);

    // Verify data bytes for valid rows match.
    let item_size = Layout::new::<Pos>().size();
    let valid_len = 100 * item_size;
    let expected = world
        .column_page_bytes(0, world.archetype_component_ids(0)[0], 0, 100)
        .unwrap();
    assert_eq!(&page.data()[..valid_len], expected);

    // Verify padding beyond valid rows is zero.
    let full_len = PAGE_SIZE * item_size;
    assert!(
        page.data()[valid_len..full_len].iter().all(|&b| b == 0),
        "padding bytes beyond row_count must be zero"
    );

    reader.validate_page_crc(&page).unwrap();
}

#[test]
fn multi_archetype_flush() {
    let mut world = World::new();

    // Archetype A: (Pos, Vel)
    for i in 0..3 {
        world.spawn((
            Pos {
                x: i as f32,
                y: 0.0,
            },
            Vel {
                dx: 1.0,
                dy: i as f32,
            },
        ));
    }
    // Archetype B: (Pos, Health)
    for i in 0..4 {
        world.spawn((
            Pos {
                x: 10.0 + i as f32,
                y: 0.0,
            },
            Health(100 + i),
        ));
    }

    assert!(
        world.archetype_count() >= 2,
        "should have at least 2 archetypes"
    );

    let (_dir, reader) = flush_and_open(&world);
    let schema = reader.schema();

    // Schema must contain all three component types.
    let pos_name = std::any::type_name::<Pos>();
    let vel_name = std::any::type_name::<Vel>();
    let health_name = std::any::type_name::<Health>();

    assert!(
        schema.slot_for(pos_name).is_some(),
        "schema must contain Pos"
    );
    assert!(
        schema.slot_for(vel_name).is_some(),
        "schema must contain Vel"
    );
    assert!(
        schema.slot_for(health_name).is_some(),
        "schema must contain Health"
    );

    // Verify pages exist for both archetypes.
    // Scan all possible arch_ids for entity pages.
    let mut found_arch_ids: Vec<u16> = Vec::new();
    for arch_id in 0..world.archetype_count() as u16 {
        if reader.get_page(arch_id, ENTITY_SLOT, 0).unwrap().is_some() {
            found_arch_ids.push(arch_id);
        }
    }
    assert_eq!(
        found_arch_ids.len(),
        2,
        "must have entity pages for exactly 2 archetypes, found: {found_arch_ids:?}"
    );
}

#[test]
fn no_dirty_pages_no_file() {
    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 },));
    world.clear_all_dirty_pages();

    let dir = tempfile::tempdir().unwrap();
    let result = flush(&world, (0, 0), dir.path()).unwrap();
    assert!(
        result.is_none(),
        "flush should return None when no dirty pages"
    );

    // Verify no .run file was created.
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "run" || ext == "tmp")
        })
        .collect();
    assert!(entries.is_empty(), "no run files should exist in temp dir");
}

#[test]
fn crc_corruption_detected() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }

    let dir = tempfile::tempdir().unwrap();
    let path = flush(&world, (0, 50), dir.path()).unwrap().unwrap();

    // Open the reader first to find a valid page's file_offset, then corrupt a
    // byte well inside that page's data region.  We use offset 256, which is
    // past the 64-byte file header, past any realistic schema section, and past
    // the 16-byte page header — i.e. guaranteed to be inside page data for
    // files with 10 entities.
    let corrupt_offset: usize = 256;

    let mut data = std::fs::read(&path).unwrap();
    assert!(
        corrupt_offset < data.len(),
        "file must be large enough to have page data at offset {corrupt_offset}"
    );
    data[corrupt_offset] ^= 0xFF; // flip all bits
    std::fs::write(&path, &data).unwrap();

    // Open may succeed (header CRC region ends at byte 40) or fail with a
    // CRC/Format error if the corruption propagated to a covered region.
    match SortedRunReader::open(&path) {
        Ok(reader) => {
            // File opened — validate all page CRCs; at least one must fail.
            let mut found_corruption = false;
            for entry in reader.schema().entries() {
                match reader.get_page(0, entry.slot(), 0) {
                    Ok(Some(page)) if reader.validate_page_crc(&page).is_err() => {
                        found_corruption = true;
                    }
                    Err(_) => {
                        found_corruption = true;
                    }
                    _ => {}
                }
            }
            match reader.get_page(0, ENTITY_SLOT, 0) {
                Ok(Some(entity_page)) if reader.validate_page_crc(&entity_page).is_err() => {
                    found_corruption = true;
                }
                Err(_) => {
                    found_corruption = true;
                }
                _ => {}
            }
            assert!(
                found_corruption,
                "at least one page CRC should fail after corruption"
            );
        }
        Err(LsmError::Crc { .. } | LsmError::Format(_)) => {
            // Header or total-file CRC corruption detected at open — acceptable.
        }
        Err(e) => panic!("unexpected error type: {e}"),
    }
}

#[test]
fn header_crc_corruption_detected() {
    let mut world = World::new();
    for i in 0..5 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }

    let dir = tempfile::tempdir().unwrap();
    let path = flush(&world, (0, 10), dir.path()).unwrap().unwrap();

    // Corrupt byte 20, which falls in the `page_count` field of the file
    // header (bytes 8-11: version, 12-15: schema_count, 16-23: page_count).
    // This is within the 40-byte CRC-protected region but after the magic
    // bytes (0-7), so the magic check passes but the header CRC fails.
    let mut data = std::fs::read(&path).unwrap();
    data[20] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let result = SortedRunReader::open(&path);
    assert!(
        matches!(result, Err(LsmError::Crc { .. })),
        "expected CRC error for header byte corruption, got: {:?}",
        result.err()
    );
}

#[test]
fn empty_file_returns_format_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.run");
    std::fs::write(&path, b"").unwrap();

    let result = SortedRunReader::open(&path);
    assert!(
        matches!(result, Err(LsmError::Format(_))),
        "expected Format error for empty file, got: {:?}",
        result.err()
    );
}

#[test]
fn truncated_file_returns_format_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.run");
    // 64 bytes — enough for a header but not the minimum 128 (header + footer).
    std::fs::write(&path, [0u8; 64]).unwrap();

    let result = SortedRunReader::open(&path);
    assert!(
        matches!(result, Err(LsmError::Format(_))),
        "expected Format error for truncated file, got: {:?}",
        result.err()
    );
}

#[test]
fn schema_contains_correct_entries() {
    let mut world = World::new();
    world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.5, dy: -0.5 }));

    let (_dir, reader) = flush_and_open(&world);
    let schema = reader.schema();

    assert_eq!(schema.len(), 2, "schema should have exactly 2 entries");

    let pos_name = std::any::type_name::<Pos>();
    let vel_name = std::any::type_name::<Vel>();

    let names: Vec<&str> = schema.entries().iter().map(SchemaEntry::name).collect();
    assert!(names.contains(&pos_name), "schema must contain Pos name");
    assert!(names.contains(&vel_name), "schema must contain Vel name");

    // Verify item sizes.
    let pos_entry = schema
        .entries()
        .iter()
        .find(|e| e.name() == pos_name)
        .unwrap();
    let vel_entry = schema
        .entries()
        .iter()
        .find(|e| e.name() == vel_name)
        .unwrap();

    assert_eq!(pos_entry.item_size() as usize, Layout::new::<Pos>().size());
    assert_eq!(vel_entry.item_size() as usize, Layout::new::<Vel>().size());
}

#[test]
fn sparse_index_finds_all_pages() {
    let mut world = World::new();
    // Spawn > 256 entities to fill 2+ pages.
    let count = PAGE_SIZE + 50; // 306 entities
    for i in 0..count {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }

    let (_dir, reader) = flush_and_open(&world);

    let schema = reader.schema();
    let pos_slot = schema
        .slot_for(std::any::type_name::<Pos>())
        .expect("Pos must be in schema");

    // With 306 entities and PAGE_SIZE=256, we expect 2 pages (page 0 and page 1).
    let expected_pages = count.div_ceil(PAGE_SIZE);
    assert_eq!(expected_pages, 2);

    // Verify component pages.
    for page_index in 0..expected_pages {
        let page = reader
            .get_page(0, pos_slot, page_index as u16)
            .unwrap_or_else(|e| panic!("get_page error: {e}"))
            .unwrap_or_else(|| panic!("component page {page_index} must exist"));

        let expected_rows = if page_index < expected_pages - 1 {
            PAGE_SIZE
        } else {
            count - page_index * PAGE_SIZE
        };
        assert_eq!(
            page.header().row_count,
            expected_rows as u16,
            "page {page_index} row count mismatch"
        );

        reader.validate_page_crc(&page).unwrap();
    }

    // Verify entity pages.
    for page_index in 0..expected_pages {
        let page = reader
            .get_page(0, ENTITY_SLOT, page_index as u16)
            .unwrap_or_else(|e| panic!("get_page error: {e}"))
            .unwrap_or_else(|| panic!("entity page {page_index} must exist"));
        reader.validate_page_crc(&page).unwrap();
    }

    // Verify no extra pages beyond expected.
    assert!(
        reader
            .get_page(0, pos_slot, expected_pages as u16)
            .unwrap()
            .is_none(),
        "no page beyond the expected count"
    );
}

#[test]
fn entity_pages_match_world() {
    let mut world = World::new();
    for i in 0..20 {
        world.spawn((Pos {
            x: i as f32,
            y: 0.0,
        },));
    }

    let (_dir, reader) = flush_and_open(&world);

    let entity_page = reader
        .get_page(0, ENTITY_SLOT, 0)
        .expect("get_page should not error")
        .expect("entity page must exist");
    assert_eq!(entity_page.header().row_count, 20);

    let entities = world.archetype_entities(0);
    assert_eq!(entities.len(), 20);

    for (i, &e) in entities.iter().enumerate() {
        let offset = i * 8;
        let stored_bits =
            u64::from_le_bytes(entity_page.data()[offset..offset + 8].try_into().unwrap());
        assert_eq!(
            stored_bits,
            e.to_bits(),
            "entity bits mismatch at index {i}"
        );
    }
}

#[test]
fn zst_component_round_trip() {
    #[derive(Clone, Copy)]
    struct Marker;

    let mut world = World::new();
    for _ in 0..5 {
        world.spawn((Marker,));
    }
    let dir = tempfile::tempdir().unwrap();
    let path = flush(&world, (0, 0), dir.path()).unwrap().unwrap();
    let reader = SortedRunReader::open(&path).unwrap();

    // Schema should have one entry for the ZST component.
    assert_eq!(reader.schema().len(), 1);
    let entry = &reader.schema().entries()[0];
    assert_eq!(
        entry.item_size(),
        0,
        "ZST component should have item_size 0"
    );

    // ZST page should exist with row_count=5 but empty data region.
    let page = reader
        .get_page(0, entry.slot(), 0)
        .expect("get_page should not error")
        .expect("ZST page must exist");
    assert_eq!(page.header().row_count, 5);
    assert!(
        page.data().is_empty(),
        "ZST page data should be empty (0 * PAGE_SIZE = 0 bytes)"
    );

    // Entity page should also exist.
    let entity_page = reader
        .get_page(0, ENTITY_SLOT, 0)
        .expect("get_page should not error")
        .expect("entity page must exist");
    assert_eq!(entity_page.header().row_count, 5);

    reader.validate_page_crc(&page).unwrap();
    reader.validate_page_crc(&entity_page).unwrap();
}
