use std::alloc::Layout;
use std::collections::HashMap;
use std::io::Write;

use crate::error::LsmError;
use crate::format::ENTITY_SLOT;

/// One entry in the schema section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaEntry {
    pub(crate) slot: u16,
    pub(crate) name: String,
    pub(crate) item_size: u32,
    pub(crate) item_align: u32,
}

impl SchemaEntry {
    /// Component slot index within this sorted run.
    pub fn slot(&self) -> u16 {
        self.slot
    }

    /// Component type name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Size in bytes of each item.
    pub fn item_size(&self) -> u32 {
        self.item_size
    }

    /// Alignment in bytes of each item.
    pub fn item_align(&self) -> u32 {
        self.item_align
    }
}

/// The complete schema section of a sorted run.
#[derive(Debug, Clone)]
pub struct SchemaSection {
    entries: Vec<SchemaEntry>,
    /// name → slot lookup for the writer
    name_to_slot: HashMap<String, u16>,
}

impl SchemaSection {
    /// Build a schema from a set of component `(name, layout)` pairs.
    ///
    /// Sorts by name lexicographically and assigns sequential slot indices
    /// starting from 0.  The entity pseudo-column (`ENTITY_SLOT = 0xFFFF`) is
    /// NOT included — callers must handle it separately.
    pub fn from_components(components: &[(String, Layout)]) -> Result<Self, LsmError> {
        // Deduplicate by name (take the first layout seen for a given name).
        let mut seen: Vec<(String, Layout)> = Vec::with_capacity(components.len());
        for (name, layout) in components {
            if !seen.iter().any(|(n, _)| n == name) {
                seen.push((name.clone(), *layout));
            }
        }

        // Sort lexicographically for deterministic slot assignment.
        seen.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut entries = Vec::with_capacity(seen.len());
        let mut name_to_slot = HashMap::with_capacity(seen.len());

        for (slot_idx, (name, layout)) in seen.into_iter().enumerate() {
            let slot = u16::try_from(slot_idx).map_err(|_| {
                LsmError::Format(format!(
                    "too many components: slot index {slot_idx} exceeds u16"
                ))
            })?;
            // Slots must not collide with the reserved entity pseudo-column.
            if slot == ENTITY_SLOT {
                return Err(LsmError::Format(
                    "too many components: slot index overflowed into ENTITY_SLOT (0xFFFF)"
                        .to_owned(),
                ));
            }
            name_to_slot.insert(name.clone(), slot);
            entries.push(SchemaEntry {
                slot,
                name,
                item_size: layout.size() as u32,
                item_align: layout.align() as u32,
            });
        }

        Ok(Self {
            entries,
            name_to_slot,
        })
    }

    /// Look up slot index by component name.
    pub fn slot_for(&self, name: &str) -> Option<u16> {
        self.name_to_slot.get(name).copied()
    }

    /// Look up entry by slot.
    pub fn entry_for_slot(&self, slot: u16) -> Option<&SchemaEntry> {
        // Entries are stored in slot order (sequential from 0), so slot == index.
        self.entries.get(slot as usize)
    }

    /// Number of schema entries (not counting the entity pseudo-column).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no schema entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate all entries in slot order.
    pub fn entries(&self) -> &[SchemaEntry] {
        &self.entries
    }

    /// Serialize the schema section to `writer`.
    ///
    /// Wire format per entry:
    /// ```text
    /// slot:       u16 LE
    /// name_len:   u16 LE
    /// name:       [u8; name_len]  (UTF-8)
    /// item_size:  u32 LE
    /// item_align: u32 LE
    /// ```
    /// The entire section is zero-padded to the next 64-byte boundary.
    ///
    /// Returns the total number of bytes written (including padding).
    pub fn write_to(&self, writer: &mut impl Write) -> Result<usize, LsmError> {
        let mut written: usize = 0;

        for entry in &self.entries {
            let name_bytes = entry.name.as_bytes();
            let name_len = name_bytes.len();
            if u16::try_from(name_len).is_err() {
                return Err(LsmError::Format(format!(
                    "component name too long: {name_len} bytes exceeds u16"
                )));
            }

            writer.write_all(&entry.slot.to_le_bytes())?;
            writer.write_all(&(name_len as u16).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            writer.write_all(&entry.item_size.to_le_bytes())?;
            writer.write_all(&entry.item_align.to_le_bytes())?;

            // 2 (slot) + 2 (name_len) + name_len + 4 (item_size) + 4 (item_align)
            written += 2 + 2 + name_len + 4 + 4;
        }

        // Pad to 64-byte boundary with zeros.
        let remainder = written % 64;
        if remainder != 0 {
            let pad_len = 64 - remainder;
            let zeros = [0u8; 64];
            writer.write_all(&zeros[..pad_len])?;
            written += pad_len;
        }

        debug_assert_eq!(written % 64, 0);
        Ok(written)
    }

    /// Deserialize from a byte slice starting at the schema section.
    ///
    /// Reads exactly `schema_count` entries and validates UTF-8 names.
    /// Returns `LsmError::Format` on malformed data.
    pub fn read_from(data: &[u8], schema_count: u32) -> Result<Self, LsmError> {
        let count = schema_count as usize;
        let mut entries = Vec::with_capacity(count);
        let mut name_to_slot = HashMap::with_capacity(count);
        let mut cursor = 0usize;

        for i in 0..count {
            // slot: u16
            if cursor + 2 > data.len() {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: unexpected end of data reading slot"
                )));
            }
            let slot = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
            cursor += 2;

            // name_len: u16
            if cursor + 2 > data.len() {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: unexpected end of data reading name_len"
                )));
            }
            let name_len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
            cursor += 2;

            // name: [u8; name_len]
            if cursor + name_len > data.len() {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: unexpected end of data reading name ({name_len} bytes)"
                )));
            }
            let name = std::str::from_utf8(&data[cursor..cursor + name_len])
                .map_err(|e| {
                    LsmError::Format(format!("schema entry {i}: invalid UTF-8 in name: {e}"))
                })?
                .to_owned();
            cursor += name_len;

            // item_size: u32
            if cursor + 4 > data.len() {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: unexpected end of data reading item_size"
                )));
            }
            let item_size = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;

            // item_align: u32
            if cursor + 4 > data.len() {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: unexpected end of data reading item_align"
                )));
            }
            let item_align = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;

            name_to_slot.insert(name.clone(), slot);
            entries.push(SchemaEntry {
                slot,
                name,
                item_size,
                item_align,
            });
        }

        // Validate invariants on parsed entries.
        for (i, entry) in entries.iter().enumerate() {
            // Slot values must match their index position.
            if entry.slot != i as u16 {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: slot {} does not match index position",
                    entry.slot
                )));
            }
            // item_align must be a power of two (or zero for ZST).
            if entry.item_align != 0 && !entry.item_align.is_power_of_two() {
                return Err(LsmError::Format(format!(
                    "schema entry {i}: item_align {} is not a power of two",
                    entry.item_align
                )));
            }
        }

        // Check for duplicate names.
        {
            let mut seen_names: std::collections::HashSet<&str> =
                std::collections::HashSet::with_capacity(entries.len());
            for (i, entry) in entries.iter().enumerate() {
                if !seen_names.insert(&entry.name) {
                    return Err(LsmError::Format(format!(
                        "schema entry {i}: duplicate name {:?}",
                        entry.name
                    )));
                }
            }
        }

        Ok(Self {
            entries,
            name_to_slot,
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_of<T>() -> Layout {
        Layout::new::<T>()
    }

    // Helper: build a SchemaSection from simple (name, size, align) triples.
    fn make_schema(specs: &[(&str, usize, usize)]) -> SchemaSection {
        let components: Vec<(String, Layout)> = specs
            .iter()
            .map(|(name, size, align)| {
                (
                    (*name).to_owned(),
                    Layout::from_size_align(*size, *align).unwrap(),
                )
            })
            .collect();
        SchemaSection::from_components(&components).unwrap()
    }

    #[test]
    fn round_trip() {
        let schema = make_schema(&[("Health", 4, 4), ("Position", 8, 4), ("Velocity", 8, 4)]);

        let mut buf = Vec::new();
        let written = schema.write_to(&mut buf).unwrap();
        assert_eq!(written, buf.len());

        let recovered = SchemaSection::read_from(&buf, schema.len() as u32).unwrap();
        assert_eq!(recovered.entries(), schema.entries());
    }

    #[test]
    fn slot_assignment_is_sorted() {
        // Input in unsorted order; slots must be assigned in lexicographic order.
        let schema = make_schema(&[("Vel", 8, 4), ("Pos", 8, 4), ("Health", 4, 4)]);

        assert_eq!(schema.slot_for("Health"), Some(0));
        assert_eq!(schema.slot_for("Pos"), Some(1));
        assert_eq!(schema.slot_for("Vel"), Some(2));

        assert_eq!(schema.entries()[0].name, "Health");
        assert_eq!(schema.entries()[1].name, "Pos");
        assert_eq!(schema.entries()[2].name, "Vel");
    }

    #[test]
    fn lookup() {
        let schema = make_schema(&[("Pos", 8, 4), ("Vel", 8, 4)]);

        // "Pos" sorts before "Vel", so it gets slot 0.
        assert_eq!(schema.slot_for("Pos"), Some(0));
        assert_eq!(schema.slot_for("Vel"), Some(1));
        assert_eq!(schema.slot_for("Unknown"), None);

        assert_eq!(
            schema.entry_for_slot(0).map(|e| e.name.as_str()),
            Some("Pos")
        );
        assert_eq!(
            schema.entry_for_slot(1).map(|e| e.name.as_str()),
            Some("Vel")
        );
        assert!(schema.entry_for_slot(2).is_none());
    }

    #[test]
    fn empty_schema() {
        let schema = SchemaSection::from_components(&[]).unwrap();
        assert!(schema.is_empty());
        assert_eq!(schema.len(), 0);

        let mut buf = Vec::new();
        let written = schema.write_to(&mut buf).unwrap();

        // No entries, so no content bytes; but the section is still padded to 64.
        assert_eq!(
            written, 0,
            "empty schema writes 0 bytes (no content, no padding needed)"
        );
        assert_eq!(buf.len(), 0);

        let recovered = SchemaSection::read_from(&buf, 0).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn padding_is_multiple_of_64() {
        for n in 0usize..=8 {
            // Vary name lengths to hit different raw sizes.
            let specs: Vec<(String, Layout)> = (0..n)
                .map(|i| {
                    let name = format!("Comp{i:0>3}");
                    (name, layout_of::<u64>())
                })
                .collect();
            let schema = SchemaSection::from_components(&specs).unwrap();

            let mut buf = Vec::new();
            let written = schema.write_to(&mut buf).unwrap();

            assert_eq!(
                written % 64,
                0,
                "n={n}: written={written} is not a multiple of 64"
            );
            assert_eq!(written, buf.len(), "n={n}: returned size matches buffer");
        }
    }

    #[test]
    fn read_from_rejects_truncated_data() {
        let schema = make_schema(&[("X", 4, 4)]);
        let mut buf = Vec::new();
        schema.write_to(&mut buf).unwrap();

        // Truncate to 3 bytes — not enough to read even the slot field fully.
        let result = SchemaSection::read_from(&buf[..3], 1);
        assert!(
            matches!(result, Err(LsmError::Format(_))),
            "expected Format error on truncated input"
        );
    }

    #[test]
    fn entity_slot_is_not_in_schema() {
        // Verify ENTITY_SLOT (0xFFFF) is reserved and not assigned by from_components.
        let schema = make_schema(&[("A", 4, 4), ("B", 4, 4)]);
        for entry in schema.entries() {
            assert_ne!(
                entry.slot, ENTITY_SLOT,
                "component slot must not equal ENTITY_SLOT"
            );
        }
    }
}
