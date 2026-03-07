# Stable Component Identity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make WAL records and snapshots interpretable across processes with different component registration orders, by adding stable string-based component names to `CodecRegistry` and a schema preamble to the WAL format.

**Architecture:** `CodecRegistry` gains `by_name: HashMap<String, ComponentId>` and `register_as()`. WAL files gain a schema preamble as the first record. A shared `build_remap()` function maps sender IDs to receiver IDs for both WAL replay and snapshot load. Layout validation catches size/align mismatches as hard errors. Engine core (`ComponentRegistry`, `World`, `ComponentId`) stays untouched.

**Tech Stack:** Rust, rkyv (serialization), minkowski-persist crate

**Design doc:** `docs/plans/2026-03-07-stable-component-identity-design.md`

---

### Task 1: CodecRegistry — `by_name` field and `register_as()`

Add name-based lookup to `CodecRegistry`. This is the foundation everything else builds on.

**Files:**
- Modify: `crates/minkowski-persist/src/codec.rs`

**Step 1: Write the failing tests**

Add to the existing `mod tests` in `codec.rs`:

```rust
#[test]
fn register_as_assigns_stable_name() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let id = world.component_id::<Pos>().unwrap();
    assert_eq!(codecs.stable_name(id), Some("pos"));
}

#[test]
fn register_defaults_to_type_name() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);

    let id = world.component_id::<Pos>().unwrap();
    let name = codecs.stable_name(id).unwrap();
    assert!(name.contains("Pos"), "default name should contain type name, got: {name}");
}

#[test]
fn resolve_name_returns_component_id() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let id = world.component_id::<Pos>().unwrap();
    assert_eq!(codecs.resolve_name("pos"), Some(id));
    assert_eq!(codecs.resolve_name("nonexistent"), None);
}

#[test]
#[should_panic(expected = "duplicate stable name")]
fn duplicate_name_panics() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("collision", &mut world);
    codecs.register_as::<Vel>("collision", &mut world);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- register_as`
Expected: FAIL — `register_as` method does not exist.

**Step 3: Implement**

In `codec.rs`, modify the `CodecRegistry` struct:

```rust
pub struct CodecRegistry {
    codecs: HashMap<ComponentId, ComponentCodec>,
    by_name: HashMap<String, ComponentId>,
}
```

Change `ComponentCodec.name` from `&'static str` to `String`:

```rust
struct ComponentCodec {
    name: String,   // was &'static str
    // ... rest unchanged
}
```

Update `new()`:
```rust
pub fn new() -> Self {
    Self {
        codecs: HashMap::new(),
        by_name: HashMap::new(),
    }
}
```

Add `register_as`:
```rust
/// Register a component type with an explicit stable name.
/// The name is used for cross-process identity in WAL and snapshot formats.
/// Panics if the name is already registered to a different component.
pub fn register_as<T>(&mut self, stable_name: &str, world: &mut World)
where
    T: Component
        + Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<Vec<u8>, ArenaHandle<'a>, rancor::Error>,
        > + Clone,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    self.register_with_name::<T>(stable_name.to_string(), world);
}
```

Refactor the existing `register` to call a shared internal method:
```rust
pub fn register<T>(&mut self, world: &mut World)
where
    T: Component + Archive + /* same bounds */ Clone,
    T::Archived: /* same bounds */,
{
    let name = std::any::type_name::<T>().to_string();
    self.register_with_name::<T>(name, world);
}

fn register_with_name<T>(&mut self, stable_name: String, world: &mut World)
where
    T: Component + Archive + /* same bounds */ Clone,
    T::Archived: /* same bounds */,
{
    let comp_id = world.register_component::<T>();

    // Check for duplicate names
    if let Some(&existing_id) = self.by_name.get(&stable_name) {
        if existing_id != comp_id {
            panic!(
                "duplicate stable name '{}': already registered to ComponentId {}, \
                 cannot register for ComponentId {}",
                stable_name, existing_id, comp_id
            );
        }
        return; // idempotent re-registration of same type+name
    }

    let layout = Layout::new::<T>();
    // ... (all existing closure construction code from current register())

    self.by_name.insert(stable_name.clone(), comp_id);
    self.codecs.insert(comp_id, ComponentCodec {
        name: stable_name,
        // ... rest unchanged
    });
}
```

Add accessors:
```rust
/// The stable name for a registered component.
pub fn stable_name(&self, id: ComponentId) -> Option<&str> {
    self.codecs.get(&id).map(|c| c.name.as_str())
}

/// Resolve a stable name to the local ComponentId.
pub fn resolve_name(&self, name: &str) -> Option<ComponentId> {
    self.by_name.get(name).copied()
}
```

Update `name()` to return `Option<&str>` instead of `Option<&'static str>`:
```rust
pub fn name(&self, id: ComponentId) -> Option<&str> {
    self.codecs.get(&id).map(|c| c.name.as_str())
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist`
Expected: All tests pass (including existing ones — `name()` return type change is compatible).

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/codec.rs
git commit -m "feat(persist): CodecRegistry::register_as + by_name lookup"
```

---

### Task 2: WAL record types — `WalEntry` and `WalSchema`

Add the schema preamble data structures to `record.rs` and serialization to `format.rs`.

**Files:**
- Modify: `crates/minkowski-persist/src/record.rs`
- Modify: `crates/minkowski-persist/src/format.rs`

**Step 1: Write the failing tests**

Add to `format.rs` tests:

```rust
#[test]
fn wal_schema_round_trip() {
    let schema = WalSchema {
        components: vec![
            WalComponentDef { id: 0, name: "pos".into(), size: 8, align: 4 },
            WalComponentDef { id: 1, name: "vel".into(), size: 8, align: 4 },
        ],
    };
    let entry = WalEntry::Schema(schema);
    let bytes = serialize_wal_entry(&entry).unwrap();
    let restored = deserialize_wal_entry(&bytes).unwrap();
    match restored {
        WalEntry::Schema(s) => {
            assert_eq!(s.components.len(), 2);
            assert_eq!(s.components[0].name, "pos");
            assert_eq!(s.components[1].id, 1);
        }
        _ => panic!("expected Schema"),
    }
}

#[test]
fn wal_entry_mutations_round_trip() {
    let record = WalRecord {
        seq: 7,
        mutations: vec![SerializedMutation::Despawn { entity: 42 }],
    };
    let entry = WalEntry::Mutations(record);
    let bytes = serialize_wal_entry(&entry).unwrap();
    let restored = deserialize_wal_entry(&bytes).unwrap();
    match restored {
        WalEntry::Mutations(r) => {
            assert_eq!(r.seq, 7);
            assert_eq!(r.mutations.len(), 1);
        }
        _ => panic!("expected Mutations"),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- wal_schema_round_trip wal_entry_mutations`
Expected: FAIL — types don't exist.

**Step 3: Implement**

In `record.rs`, add:

```rust
/// A single component definition in a WAL schema preamble.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct WalComponentDef {
    pub id: ComponentId,
    pub name: String,
    pub size: usize,
    pub align: usize,
}

/// Schema preamble: maps sender-local IDs to stable names.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct WalSchema {
    pub components: Vec<WalComponentDef>,
}

/// A WAL file entry: either a schema preamble (first record) or mutation data.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub enum WalEntry {
    Schema(WalSchema),
    Mutations(WalRecord),
}
```

In `format.rs`, add:

```rust
pub fn serialize_wal_entry(entry: &WalEntry) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(entry)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn deserialize_wal_entry(bytes: &[u8]) -> Result<WalEntry, FormatError> {
    rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(bytes)
        .map_err(|e| FormatError(e.to_string()))
}
```

Update `lib.rs` — `pub use record::*` already exports the new types.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist`
Expected: All pass.

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/record.rs crates/minkowski-persist/src/format.rs
git commit -m "feat(persist): WalEntry, WalSchema, WalComponentDef record types"
```

---

### Task 3: `build_remap()` — shared ID remapping function

Create the shared function that maps sender-local IDs to receiver-local IDs using stable names and validates layout.

**Files:**
- Modify: `crates/minkowski-persist/src/codec.rs` (or a new `remap.rs` — preference is to keep it in `codec.rs` since it uses `CodecRegistry` internals)

**Step 1: Write the failing tests**

Add to `codec.rs` tests:

```rust
use crate::record::WalComponentDef;

#[test]
fn build_remap_same_order() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Vel>("vel", &mut world);

    let schema = vec![
        WalComponentDef { id: 0, name: "pos".into(), size: 8, align: 4 },
        WalComponentDef { id: 1, name: "vel".into(), size: 8, align: 4 },
    ];

    let remap = codecs.build_remap(&schema).unwrap();
    // Same order → identity mapping
    assert_eq!(remap.get(&0), Some(&0));
    assert_eq!(remap.get(&1), Some(&1));
}

#[test]
fn build_remap_different_order() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Vel>("vel", &mut world);  // id=0 locally
    codecs.register_as::<Pos>("pos", &mut world);  // id=1 locally

    // Sender had Pos=0, Vel=1 (opposite order)
    let schema = vec![
        WalComponentDef { id: 0, name: "pos".into(), size: 8, align: 4 },
        WalComponentDef { id: 1, name: "vel".into(), size: 8, align: 4 },
    ];

    let remap = codecs.build_remap(&schema).unwrap();
    assert_eq!(remap.get(&0), Some(&1)); // sender's 0 (pos) → receiver's 1
    assert_eq!(remap.get(&1), Some(&0)); // sender's 1 (vel) → receiver's 0
}

#[test]
fn build_remap_size_mismatch_is_error() {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let schema = vec![
        WalComponentDef { id: 0, name: "pos".into(), size: 999, align: 4 },
    ];

    let result = codecs.build_remap(&schema);
    assert!(result.is_err());
}

#[test]
fn build_remap_unknown_name_is_error() {
    let codecs = CodecRegistry::new();

    let schema = vec![
        WalComponentDef { id: 0, name: "nonexistent".into(), size: 8, align: 4 },
    ];

    let result = codecs.build_remap(&schema);
    assert!(result.is_err());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- build_remap`
Expected: FAIL — `build_remap` does not exist.

**Step 3: Implement**

In `codec.rs`, add a new error variant to `CodecError`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    // ... existing variants ...
    #[error("schema mismatch for component '{name}': sender has size={sender_size} align={sender_align}, receiver has size={receiver_size} align={receiver_align}")]
    SchemaMismatch {
        name: String,
        sender_size: usize,
        sender_align: usize,
        receiver_size: usize,
        receiver_align: usize,
    },
    #[error("unknown component name in schema: '{0}'")]
    UnknownComponentName(String),
}
```

Add the method to `CodecRegistry`:

```rust
/// Build a remap table from a sender's schema to the receiver's local IDs.
///
/// For each entry in the sender's schema, resolves the stable name to a
/// local ComponentId and validates that size and align match. Returns a
/// mapping from sender ComponentId → receiver ComponentId.
///
/// Errors if any name is unknown or any layout mismatches.
pub fn build_remap(
    &self,
    schema: &[crate::record::WalComponentDef],
) -> Result<HashMap<ComponentId, ComponentId>, CodecError> {
    let mut remap = HashMap::new();
    for def in schema {
        let local_id = self.resolve_name(&def.name)
            .ok_or_else(|| CodecError::UnknownComponentName(def.name.clone()))?;
        let local_layout = self.layout(local_id)
            .ok_or(CodecError::UnregisteredComponent(local_id))?;

        if def.size != local_layout.size() || def.align != local_layout.align() {
            return Err(CodecError::SchemaMismatch {
                name: def.name.clone(),
                sender_size: def.size,
                sender_align: def.align,
                receiver_size: local_layout.size(),
                receiver_align: local_layout.align(),
            });
        }

        remap.insert(def.id, local_id);
    }
    Ok(remap)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist`
Expected: All pass.

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/codec.rs
git commit -m "feat(persist): build_remap for cross-process ComponentId resolution"
```

---

### Task 4: WAL — write schema preamble on create

Update `Wal::create` to accept `&CodecRegistry` and write a schema preamble as the first record. Update `Wal::append` to use `WalEntry::Mutations` wrapper.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Step 1: Write the failing test**

Add to `wal.rs` tests:

```rust
#[test]
fn create_writes_schema_preamble() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("schema.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);
    codecs.register_as::<Health>("health", &mut world);

    let _wal = Wal::create(&wal_path, &codecs).unwrap();

    // Re-open and verify schema is readable
    let wal2 = Wal::open(&wal_path, &codecs).unwrap();
    assert_eq!(wal2.next_seq(), 0); // no mutation records yet
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- create_writes_schema`
Expected: FAIL — `Wal::create` doesn't accept `&CodecRegistry`.

**Step 3: Implement**

Update `Wal::create` signature and implementation:

```rust
/// Create a new WAL file with a schema preamble. Fails if the file already exists.
pub fn create(path: &Path, codecs: &CodecRegistry) -> Result<Self, WalError> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(path)?;
    let mut wal = Self { file, next_seq: 0 };
    wal.write_schema_preamble(codecs)?;
    Ok(wal)
}
```

Add the internal helper to build and write the schema:

```rust
fn write_schema_preamble(&mut self, codecs: &CodecRegistry) -> Result<(), WalError> {
    let components: Vec<WalComponentDef> = codecs.registered_ids().iter().map(|&id| {
        let name = codecs.stable_name(id).unwrap().to_string();
        let layout = codecs.layout(id).unwrap();
        WalComponentDef {
            id,
            name,
            size: layout.size(),
            align: layout.align(),
        }
    }).collect();

    let entry = WalEntry::Schema(WalSchema { components });
    let payload = format::serialize_wal_entry(&entry)
        .map_err(|e| WalError::Format(e.to_string()))?;

    let mut writer = BufWriter::new(&self.file);
    let len: u32 = payload.len().try_into().map_err(|_| {
        WalError::Format("schema preamble too large".to_string())
    })?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}
```

Update `Wal::append` to wrap records in `WalEntry::Mutations`:

```rust
pub fn append(
    &mut self,
    changeset: &EnumChangeSet,
    codecs: &CodecRegistry,
) -> Result<u64, WalError> {
    let seq = self.next_seq;
    let record = Self::changeset_to_record(seq, changeset, codecs)?;
    let entry = WalEntry::Mutations(record);
    let payload = format::serialize_wal_entry(&entry)
        .map_err(|e| WalError::Format(e.to_string()))?;

    let mut writer = BufWriter::new(&self.file);
    let len: u32 = payload.len().try_into().map_err(|_| {
        WalError::Format(format!(
            "WAL record too large: {} bytes exceeds u32 max",
            payload.len()
        ))
    })?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;

    self.next_seq += 1;
    Ok(seq)
}
```

Update `read_next_record` to deserialize `WalEntry` instead of `WalRecord`:

```rust
fn read_next_record(
    &mut self,
    pos: u64,
) -> Result<Option<(WalEntry, u64)>, WalError> {
    // ... same I/O logic for len_buf and payload ...
    match format::deserialize_wal_entry(&payload) {
        Ok(entry) => Ok(Some((entry, pos + 4 + len as u64))),
        Err(_) => {
            self.file.set_len(pos)?;
            Ok(None)
        }
    }
}
```

Update `Wal::open` to parse the schema preamble and skip it:

```rust
/// Open an existing WAL file. Parses schema preamble if present.
pub fn open(path: &Path, codecs: &CodecRegistry) -> Result<Self, WalError> {
    let file = OpenOptions::new().read(true).append(true).open(path)?;
    let mut wal = Self { file, next_seq: 0 };
    let last = wal.scan_last_seq()?;
    wal.next_seq = if last > 0 || wal.has_records()? {
        last + 1
    } else {
        0
    };
    Ok(wal)
}
```

Update `scan_last_seq` to handle `WalEntry`:

```rust
fn scan_last_seq(&mut self) -> Result<u64, WalError> {
    self.file.seek(SeekFrom::Start(0))?;
    let mut last_seq = 0u64;
    let mut pos: u64 = 0;

    while let Some((entry, next_pos)) = self.read_next_record(pos)? {
        if let WalEntry::Mutations(record) = entry {
            last_seq = record.seq;
        }
        pos = next_pos;
    }

    Ok(last_seq)
}
```

Update `replay_from` to handle `WalEntry`, extract schema, build remap, and apply it:

```rust
pub fn replay_from(
    &mut self,
    from_seq: u64,
    world: &mut World,
    codecs: &CodecRegistry,
) -> Result<u64, WalError> {
    let mut pos: u64 = 0;
    let mut last_seq = if from_seq > 0 { from_seq - 1 } else { 0 };
    let mut remap: Option<HashMap<ComponentId, ComponentId>> = None;

    while let Some((entry, next_pos)) = self.read_next_record(pos)? {
        match entry {
            WalEntry::Schema(schema) => {
                remap = Some(codecs.build_remap(&schema.components)?);
            }
            WalEntry::Mutations(record) => {
                if record.seq >= from_seq {
                    Self::apply_record(&record, world, codecs, remap.as_ref())?;
                    last_seq = record.seq;
                }
            }
        }
        pos = next_pos;
    }

    Ok(last_seq)
}
```

Update `replay` to delegate:

```rust
pub fn replay(&mut self, world: &mut World, codecs: &CodecRegistry) -> Result<u64, WalError> {
    self.replay_from(0, world, codecs)
}
```

Update `apply_record` to accept an optional remap table and apply it to ComponentIds:

```rust
fn apply_record(
    record: &crate::record::WalRecord,
    world: &mut World,
    codecs: &CodecRegistry,
    remap: Option<&HashMap<ComponentId, ComponentId>>,
) -> Result<(), WalError> {
    let remap_id = |id: ComponentId| -> ComponentId {
        remap.map_or(id, |r| r.get(&id).copied().unwrap_or(id))
    };

    let mut changeset = EnumChangeSet::new();
    for mutation in &record.mutations {
        match mutation {
            SerializedMutation::Spawn { entity, components } => {
                let entity = Entity::from_bits(*entity);
                world.alloc_entity();
                let mut raw_components: Vec<(ComponentId, Vec<u8>, std::alloc::Layout)> = Vec::new();
                for (comp_id, data) in components {
                    let local_id = remap_id(*comp_id);
                    let raw = codecs.deserialize(local_id, data)?;
                    let layout = codecs.layout(local_id)
                        .ok_or(CodecError::UnregisteredComponent(local_id))?;
                    raw_components.push((local_id, raw, layout));
                }
                let ptrs: Vec<_> = raw_components.iter()
                    .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                    .collect();
                changeset.record_spawn(entity, &ptrs);
            }
            SerializedMutation::Despawn { entity } => {
                changeset.record_despawn(Entity::from_bits(*entity));
            }
            SerializedMutation::Insert { entity, component_id, data } => {
                let local_id = remap_id(*component_id);
                let raw = codecs.deserialize(local_id, data)?;
                let layout = codecs.layout(local_id)
                    .ok_or(CodecError::UnregisteredComponent(local_id))?;
                changeset.record_insert(Entity::from_bits(*entity), local_id, raw.as_ptr(), layout);
            }
            SerializedMutation::Remove { entity, component_id } => {
                changeset.record_remove(Entity::from_bits(*entity), remap_id(*component_id));
            }
        }
    }
    changeset.apply(world);
    Ok(())
}
```

**Step 4: Update all existing WAL tests**

All existing tests that call `Wal::create` need to pass `&codecs`:
- `create_append_and_replay`: `Wal::create(&wal_path, &codecs)`
- `open_existing_wal`: `Wal::create(&wal_path, &codecs)` and `Wal::open(&wal_path, &codecs)`
- `replay_from_skips_earlier_records`: both create and open
- `empty_wal_replay`: `Wal::create(&wal_path, &codecs)`
- `torn_entry_truncated_on_open`: both
- `torn_entry_truncated_on_replay`: both
- `corrupted_payload_truncated_on_open`: both
- `corrupted_payload_truncated_on_replay`: both
- Snapshot tests in `snapshot.rs` that use `Wal::create`/`Wal::open` (3 tests)

**Step 5: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist`
Expected: All pass.

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "feat(persist): WAL schema preamble + ID remapping on replay"
```

---

### Task 5: Snapshot — use stable names for save and remap on load

Update snapshot save to use `CodecRegistry` stable names. Update load paths to use `build_remap` for ID resolution.

**Files:**
- Modify: `crates/minkowski-persist/src/snapshot.rs`

**Step 1: Write the failing test**

Add to `snapshot.rs` tests:

```rust
#[test]
fn snapshot_cross_process_different_registration_order() {
    let dir = tempfile::tempdir().unwrap();
    let snap_path = dir.path().join("cross.snap");

    // Process A: registers Pos first, then Vel
    let mut world_a = World::new();
    let mut codecs_a = CodecRegistry::new();
    codecs_a.register_as::<Pos>("pos", &mut world_a);
    codecs_a.register_as::<Vel>("vel", &mut world_a);

    world_a.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

    let snap = Snapshot::new();
    snap.save(&snap_path, &world_a, &codecs_a, 0).unwrap();

    // Process B: registers in opposite order
    let mut world_b_tmp = World::new();
    let mut codecs_b = CodecRegistry::new();
    codecs_b.register_as::<Vel>("vel", &mut world_b_tmp);  // id=0
    codecs_b.register_as::<Pos>("pos", &mut world_b_tmp);  // id=1

    let (mut world_b, _) = snap.load(&snap_path, &codecs_b).unwrap();

    // Verify data is correct despite different registration order
    let positions: Vec<(f32, f32)> = world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
    assert_eq!(positions, vec![(1.0, 2.0)]);

    let velocities: Vec<(f32, f32)> = world_b.query::<(&Vel,)>().map(|v| (v.0.dx, v.0.dy)).collect();
    assert_eq!(velocities, vec![(3.0, 4.0)]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- cross_process`
Expected: FAIL — data is corrupted (Pos and Vel swapped) or registration panics.

**Step 3: Implement**

Update `build_snapshot_data` to use stable names from `CodecRegistry`:

```rust
fn build_snapshot_data(
    &self,
    world: &World,
    codecs: &CodecRegistry,
    wal_seq: u64,
) -> Result<SnapshotData, SnapshotError> {
    let schema: Vec<ComponentSchema> = (0..world.component_count())
        .map(|id| {
            // Use stable name from codecs if available, fall back to World's type_name
            let name = codecs.stable_name(id)
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    world.component_name(id).unwrap_or("unknown").to_string()
                });
            ComponentSchema {
                id,
                name,
                size: world.component_layout(id).map(|l| l.size()).unwrap_or(0),
                align: world.component_layout(id).map(|l| l.align()).unwrap_or(1),
            }
        })
        .collect();
    // ... rest unchanged
}
```

Update `restore_world` to use name-based resolution with remapping:

```rust
fn restore_world(
    &self,
    data: &SnapshotData,
    codecs: &CodecRegistry,
) -> Result<World, SnapshotError> {
    let mut world = World::new();

    // Build remap from schema names
    let schema_defs: Vec<WalComponentDef> = data.schema.iter()
        .filter(|entry| codecs.resolve_name(&entry.name).is_some())
        .map(|entry| WalComponentDef {
            id: entry.id,
            name: entry.name.clone(),
            size: entry.size,
            align: entry.align,
        })
        .collect();

    let remap = if !schema_defs.is_empty() {
        codecs.build_remap(&schema_defs)
            .map_err(|e| SnapshotError::Format(e.to_string()))?
    } else {
        HashMap::new()
    };

    let remap_id = |id: ComponentId| -> ComponentId {
        remap.get(&id).copied().unwrap_or(id)
    };

    // Register components: iterate schema in order, register by resolved local ID.
    // We need to ensure the World has enough component slots. Register each
    // component the receiver knows about; fill gaps for unknown ones.
    // The trick: we register ALL codecs' components into the new World first,
    // then fill remaining gaps.
    for def in &schema_defs {
        let local_id = remap_id(def.id);
        codecs.register_one(local_id, &mut world);
    }

    // Fill gaps for non-persisted components
    for entry in &data.schema {
        if world.component_count() <= entry.id {
            // Need to fill gaps up to this ID
            while world.component_count() <= entry.id {
                let layout = std::alloc::Layout::from_size_align(
                    entry.size, entry.align
                ).map_err(|_| {
                    SnapshotError::Format(format!(
                        "invalid layout for component '{}': size={}, align={}",
                        entry.name, entry.size, entry.align
                    ))
                })?;
                let name: &'static str = Box::leak(entry.name.clone().into_boxed_str());
                world.register_component_raw(name, layout);
            }
        }
    }

    // Restore archetypes via EnumChangeSet with remapped IDs
    for arch_data in &data.archetypes {
        let mut changeset = EnumChangeSet::new();
        for (row, &entity_bits) in arch_data.entities.iter().enumerate() {
            let entity = Entity::from_bits(entity_bits);
            world.alloc_entity();

            let mut raw_components: Vec<(ComponentId, Vec<u8>, std::alloc::Layout)> = Vec::new();
            for col in &arch_data.columns {
                let local_id = remap_id(col.component_id);
                let raw = codecs.deserialize(local_id, &col.values[row])?;
                let layout = codecs.layout(local_id)
                    .ok_or(CodecError::UnregisteredComponent(local_id))?;
                raw_components.push((local_id, raw, layout));
            }

            let ptrs: Vec<_> = raw_components.iter()
                .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                .collect();
            changeset.record_spawn(entity, &ptrs);
        }
        changeset.apply(&mut world);
    }

    // Restore sparse with remapping
    for sparse_data in &data.sparse {
        let local_id = remap_id(sparse_data.component_id);
        for (entity_bits, bytes) in &sparse_data.entries {
            let entity = Entity::from_bits(*entity_bits);
            codecs.insert_sparse_raw(local_id, &mut world, entity, bytes)?;
        }
    }

    // Restore allocator state
    world.restore_allocator_state(
        data.allocator.generations.clone(),
        data.allocator.free_list.clone(),
    );

    Ok(world)
}
```

Similarly update `restore_world_zero_copy` with the same remapping pattern.

**Important:** This task involves significant refactoring of the restore paths. The restore logic currently assumes sender IDs = receiver IDs (same registration order). The remap table breaks this assumption. Take care to apply `remap_id()` to every `component_id` reference: `ColumnData.component_id`, `ArchetypeData.component_ids`, `SparseComponentData.component_id`, and the `codecs.deserialize()`/`codecs.layout()` calls.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski-persist`
Expected: All pass, including the new cross-process test.

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/snapshot.rs
git commit -m "feat(persist): snapshot save/load with stable name resolution + ID remapping"
```

---

### Task 6: Update callers — Durable, examples, and existing tests

Update `Durable` and all call sites that construct `Wal::create`/`Wal::open`.

**Files:**
- Modify: `crates/minkowski-persist/src/durable.rs`
- Modify: `examples/examples/persist.rs`
- Modify: any other files using `Wal::create`/`Wal::open`

**Step 1: Check all callers**

Run: `grep -rn 'Wal::create\|Wal::open' crates/ examples/` to find every call site.

**Step 2: Update `Durable`**

Read `durable.rs` and update `Wal::create`/`Wal::open` calls to pass `&codecs`.

**Step 3: Update the persist example**

Change `codecs.register::<Pos>` to `codecs.register_as::<Pos>("pos", ...)` (etc.) to demonstrate explicit naming. Update `Wal::create` and `Wal::open` calls.

**Step 4: Run all tests and examples**

Run:
```bash
cargo test -p minkowski-persist
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p minkowski-examples --example persist --release
```
Expected: All pass.

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/durable.rs examples/examples/persist.rs
git commit -m "feat(persist): update Durable + persist example for schema preamble API"
```

---

### Task 7: Backwards compatibility — old WALs without schema preamble

Ensure WALs created before this change (no schema preamble) still replay correctly.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs` (test only)

**Step 1: Write the test**

```rust
#[test]
fn legacy_wal_without_schema_replays() {
    // Simulate a legacy WAL: write WalEntry::Mutations records directly
    // without a Schema preamble (as old code would have written WalRecords).
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("legacy.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    // Manually write a legacy-format WAL (WalEntry::Mutations, no Schema first)
    {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&wal_path)
            .unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 42.0, y: 99.0 },));

        let record = Wal::changeset_to_record_pub(0, &cs, &codecs).unwrap();
        let entry = WalEntry::Mutations(record);
        let payload = format::serialize_wal_entry(&entry).unwrap();

        let mut writer = BufWriter::new(&file);
        writer.write_all(&(payload.len() as u32).to_le_bytes()).unwrap();
        writer.write_all(&payload).unwrap();
        writer.flush().unwrap();
    }

    // Open and replay — should work without schema (no remapping)
    let mut wal = Wal::open(&wal_path, &codecs).unwrap();
    let mut world2 = World::new();
    codecs.register_one(
        world.component_id::<Pos>().unwrap(),
        &mut world2,
    );

    let last = wal.replay(&mut world2, &codecs).unwrap();
    assert_eq!(last, 0);
    assert_eq!(world2.query::<(&Pos,)>().count(), 1);
}
```

**Note:** This test may require exposing `changeset_to_record` as a test helper or constructing the raw WAL bytes manually. Adapt as needed — the key is that a WAL file whose first record is `WalEntry::Mutations` (not `WalEntry::Schema`) replays without error using identity mapping (no remap).

**Step 2: Verify the test passes**

The implementation in Task 4 already handles this: `remap` starts as `None`, and `apply_record` with `remap: None` uses identity mapping. This test should pass without additional code.

Run: `cargo test -p minkowski-persist -- legacy_wal`
Expected: PASS.

**Step 3: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "test(persist): verify backwards compatibility for WALs without schema preamble"
```

---

### Task 8: WAL cross-process replay test

The end-to-end test: write a WAL with one registration order, replay with a different order.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs` (test)

**Step 1: Write the test**

```rust
#[test]
fn wal_cross_process_different_registration_order() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("cross.wal");

    // "Process A": Pos=0, Health=1
    let mut world_a = World::new();
    let mut codecs_a = CodecRegistry::new();
    codecs_a.register_as::<Pos>("pos", &mut world_a);
    codecs_a.register_as::<Health>("health", &mut world_a);

    let mut wal = Wal::create(&wal_path, &codecs_a).unwrap();

    let e = world_a.alloc_entity();
    let mut cs = EnumChangeSet::new();
    cs.spawn_bundle(&mut world_a, e, (Pos { x: 1.0, y: 2.0 }, Health(100)));
    wal.append(&cs, &codecs_a).unwrap();
    cs.apply(&mut world_a);

    drop(wal);

    // "Process B": Health=0, Pos=1 (opposite order)
    let mut world_b = World::new();
    let mut codecs_b = CodecRegistry::new();
    codecs_b.register_as::<Health>("health", &mut world_b);
    codecs_b.register_as::<Pos>("pos", &mut world_b);

    let mut wal_b = Wal::open(&wal_path, &codecs_b).unwrap();
    wal_b.replay(&mut world_b, &codecs_b).unwrap();

    // Verify: data is correct despite different registration order
    let positions: Vec<(f32, f32)> = world_b.query::<(&Pos,)>()
        .map(|p| (p.0.x, p.0.y)).collect();
    assert_eq!(positions, vec![(1.0, 2.0)]);

    let health: Vec<u32> = world_b.query::<(&Health,)>()
        .map(|h| h.0.0).collect();
    assert_eq!(health, vec![100]);
}
```

**Step 2: Run test**

Run: `cargo test -p minkowski-persist -- wal_cross_process`
Expected: PASS (all infrastructure from Tasks 1–4 supports this).

**Step 3: Final full validation**

Run:
```bash
cargo test -p minkowski-persist
cargo test -p minkowski --lib
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p minkowski-examples --example persist --release
```
Expected: All pass.

**Step 4: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "test(persist): end-to-end cross-process WAL replay with ID remapping"
```
