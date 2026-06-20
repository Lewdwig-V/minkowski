use std::alloc::Layout;
use std::collections::HashMap;

use minkowski::component::Component;
use minkowski::{ComponentId, Entity, World};
use rkyv::api::high::HighValidator;
use rkyv::bytecheck::CheckBytes;
use rkyv::de::Pool;
use rkyv::rancor;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Schema entry describing a component type. Used in both snapshot schemas
/// and WAL preambles. Fields are sender-local: `id` is meaningful only in
/// the originating World's ID space.
///
/// Defined here (in `minkowski-lsm::codec`) so that [`CodecRegistry::build_remap`]
/// can reference it without a dependency on `minkowski-persist`. Re-exported by
/// `minkowski-persist::record` for WAL and snapshot consumers.
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
pub struct ComponentSchema {
    pub id: ComponentId,
    pub name: String,
    pub size: usize,
    pub align: usize,
}

/// Type-erased serialize: reads T from raw pointer, serializes to output buffer.
type SerializeFn = unsafe fn(*const u8, &mut Vec<u8>) -> Result<(), CodecError>;

/// Type-erased deserialize: reads T from bytes, returns raw memory bytes.
type DeserializeFn = fn(&[u8]) -> Result<Vec<u8>, CodecError>;

/// Type-erased component registration: registers the concrete type into a World.
type RegisterFn = fn(&mut World) -> ComponentId;

/// Type-erased sparse serialization: iterates World's sparse storage for a component,
/// returning `(entity_bits, serialized_bytes)` pairs.
type SerializeSparseFn =
    fn(&World, ComponentId, &CodecRegistry) -> Result<Vec<(u64, Vec<u8>)>, CodecError>;

/// Type-erased sparse insertion: deserializes bytes and inserts into World's sparse storage.
type InsertSparseFn = fn(&mut World, Entity, &[u8]) -> Result<(), CodecError>;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("failed to serialize component: {0}")]
    Serialize(String),
    #[error("failed to deserialize component: {0}")]
    Deserialize(String),
    #[error(
        "no codec registered for component id {0} — \
         call `codecs.register::<T>(&mut world)` for each component type before persisting"
    )]
    UnregisteredComponent(ComponentId),
    #[error(
        "schema mismatch for component '{name}': sender has size={sender_size} align={sender_align}, receiver has size={receiver_size} align={receiver_align}"
    )]
    SchemaMismatch {
        name: String,
        sender_size: usize,
        sender_align: usize,
        receiver_size: usize,
        receiver_align: usize,
    },
    #[error("unknown component name in schema: '{0}'")]
    UnknownComponentName(String),
    #[error(
        "component already registered with name {existing_name:?}, cannot re-register as {new_name:?}"
    )]
    DuplicateComponentName {
        existing_name: String,
        new_name: String,
    },
    #[error(
        "duplicate stable name {name:?}: already registered for ComponentId {existing_id}, cannot register for ComponentId {new_id}"
    )]
    DuplicateStableName {
        name: String,
        existing_id: ComponentId,
        new_id: ComponentId,
    },
    #[error(
        "component {name:?} is not raw-copyable: native size is {native_size}, \
         rkyv archived size is {archived_size} (it likely contains heap-allocated \
         data such as String or Vec). LSM persistence requires plain-old-data \
         components whose bytes are position-independent."
    )]
    NotRawCopyable {
        name: String,
        native_size: usize,
        archived_size: usize,
    },
}

struct ComponentCodec {
    name: String,
    layout: Layout,
    serialize_fn: SerializeFn,
    deserialize_fn: DeserializeFn,
    register_fn: RegisterFn,
    serialize_sparse_fn: SerializeSparseFn,
    insert_sparse_fn: InsertSparseFn,
    /// Byte size for the direct-memcpy decode fast path. Always `Some(size)` for
    /// accepted non-ZST components (raw-copyability is enforced at registration,
    /// so the archived layout matches native); `None` for ZSTs (nothing to copy).
    raw_copy_size: Option<usize>,
    /// `TypeId` of the concrete component type this codec was registered for.
    /// `ComponentId` is a per-world index, so the flush gate compares this to the
    /// flushed world's `component_type_id` to confirm the codec actually
    /// describes the type whose native bytes are being persisted.
    type_id: std::any::TypeId,
}

/// Maps ComponentId to rkyv codecs. Separate from core's ComponentRegistry —
/// different concerns, different crates, different lifetimes.
pub struct CodecRegistry {
    codecs: HashMap<ComponentId, ComponentCodec>,
    by_name: HashMap<String, ComponentId>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        Self {
            codecs: HashMap::new(),
            by_name: HashMap::new(),
        }
    }

    /// `TypeId` of the component type registered under `id`, or `None` if no
    /// codec is registered for that id. `ComponentId` is a per-world index, so
    /// the flush gate compares this to the flushed world's `component_type_id`
    /// to confirm the codec describes the type actually being persisted.
    pub fn type_id(&self, id: ComponentId) -> Option<std::any::TypeId> {
        self.codecs.get(&id).map(|c| c.type_id)
    }

    /// Register a component type for persistence.
    /// Requires rkyv Archive + Serialize + Deserialize bounds.
    /// Uses `std::any::type_name::<T>()` as the default stable name.
    pub fn register<T>(&mut self, world: &mut World) -> Result<(), CodecError>
    where
        T: Component
            + Archive
            + for<'a> RkyvSerialize<
                rkyv::api::high::HighSerializer<Vec<u8>, ArenaHandle<'a>, rancor::Error>,
            > + Clone,
        T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
    {
        let name = std::any::type_name::<T>().to_owned();
        self.register_with_name::<T>(name, world)
    }

    /// Register a component type for persistence with an explicit stable name.
    /// The name must be unique across all registered components — duplicate
    /// names mapped to different ComponentIds return an error.
    pub fn register_as<T>(&mut self, stable_name: &str, world: &mut World) -> Result<(), CodecError>
    where
        T: Component
            + Archive
            + for<'a> RkyvSerialize<
                rkyv::api::high::HighSerializer<Vec<u8>, ArenaHandle<'a>, rancor::Error>,
            > + Clone,
        T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
    {
        self.register_with_name::<T>(stable_name.to_owned(), world)
    }

    fn register_with_name<T>(
        &mut self,
        stable_name: String,
        world: &mut World,
    ) -> Result<(), CodecError>
    where
        T: Component
            + Archive
            + for<'a> RkyvSerialize<
                rkyv::api::high::HighSerializer<Vec<u8>, ArenaHandle<'a>, rancor::Error>,
            > + Clone,
        T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
    {
        let comp_id = world.register_component::<T>();

        // Idempotent: same type re-registered with same name is a no-op.
        if let Some(existing) = self.codecs.get(&comp_id) {
            if existing.name != stable_name {
                return Err(CodecError::DuplicateComponentName {
                    existing_name: existing.name.clone(),
                    new_name: stable_name,
                });
            }
            return Ok(());
        }

        // Duplicate name to a different ComponentId is a hard error.
        if let Some(&existing_id) = self.by_name.get(&stable_name)
            && existing_id != comp_id
        {
            return Err(CodecError::DuplicateStableName {
                name: stable_name,
                existing_id,
                new_id: comp_id,
            });
        }

        let layout = Layout::new::<T>();

        // Raw-copyability invariant: a component is persistable only if its rkyv
        // archived representation has the same size as the native type. This is
        // a proxy for "plain-old-data, no heap indirection" — such bytes are
        // position-independent and safe to memcpy back into a column on recovery.
        // Types with heap data (String, Vec, ...) have a differently-sized
        // archived form and are rejected here, at the single registration source,
        // so nothing un-raw-copyable can ever reach a flush, run, or import.
        // ZSTs satisfy `0 == 0` and are accepted (trivially raw-copyable).
        let raw_copyable = std::mem::size_of::<T>() == std::mem::size_of::<T::Archived>();
        if !raw_copyable {
            return Err(CodecError::NotRawCopyable {
                name: stable_name,
                native_size: std::mem::size_of::<T>(),
                archived_size: std::mem::size_of::<T::Archived>(),
            });
        }

        // The decode fast-path still needs a positive byte count to memcpy; ZSTs
        // have nothing to copy, so their `raw_copy_size` stays `None`.
        let raw_copy_size = if layout.size() > 0 {
            Some(layout.size())
        } else {
            None
        };

        let serialize_fn: SerializeFn = |ptr, out| {
            let value = unsafe { &*ptr.cast::<T>() };
            // Write directly into the caller's Vec — no intermediate AlignedVec.
            *out = rkyv::api::high::to_bytes_in::<_, rancor::Error>(value, std::mem::take(out))
                .map_err(|e| CodecError::Serialize(e.to_string()))?;
            Ok(())
        };

        let deserialize_fn: DeserializeFn = |bytes| {
            let value: T = rkyv::from_bytes::<T, rancor::Error>(bytes)
                .map_err(|e| CodecError::Deserialize(e.to_string()))?;
            let layout = Layout::new::<T>();
            let mut buf = vec![0u8; layout.size()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &value as *const T as *const u8,
                    buf.as_mut_ptr(),
                    layout.size(),
                );
            }
            std::mem::forget(value); // ownership transferred to buf
            Ok(buf)
        };

        let register_fn: RegisterFn = |world| world.register_component::<T>();

        let serialize_sparse_fn: SerializeSparseFn = |world, comp_id, codecs| {
            let mut entries = Vec::new();
            if let Some(iter) = world.iter_sparse::<T>(comp_id) {
                for (entity, value) in iter {
                    let mut buf = Vec::new();
                    // SAFETY: value is a valid &T from the sparse HashMap.
                    unsafe {
                        codecs.serialize(comp_id, value as *const T as *const u8, &mut buf)?;
                    }
                    entries.push((entity.to_bits(), buf));
                }
            }
            Ok(entries)
        };

        let insert_sparse_fn: InsertSparseFn = |world, entity, data| {
            let value: T = rkyv::from_bytes::<T, rancor::Error>(data)
                .map_err(|e| CodecError::Deserialize(e.to_string()))?;
            world.insert_sparse::<T>(entity, value);
            Ok(())
        };

        self.by_name.insert(stable_name.clone(), comp_id);
        self.codecs.insert(
            comp_id,
            ComponentCodec {
                name: stable_name,
                layout,
                serialize_fn,
                deserialize_fn,
                register_fn,
                serialize_sparse_fn,
                insert_sparse_fn,
                raw_copy_size,
                type_id: std::any::TypeId::of::<T>(),
            },
        );
        Ok(())
    }

    /// Serialize a component value from a raw pointer to bytes.
    ///
    /// # Safety
    /// `ptr` must point to a valid, aligned instance of the component type
    /// registered under `id`. The pointer must be valid for reads of
    /// `layout.size()` bytes.
    pub unsafe fn serialize(
        &self,
        id: ComponentId,
        ptr: *const u8,
        out: &mut Vec<u8>,
    ) -> Result<(), CodecError> {
        unsafe {
            let codec = self
                .codecs
                .get(&id)
                .ok_or(CodecError::UnregisteredComponent(id))?;
            (codec.serialize_fn)(ptr, out)
        }
    }

    /// Deserialize component bytes into a raw byte buffer.
    pub fn deserialize(&self, id: ComponentId, bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
        let codec = self
            .codecs
            .get(&id)
            .ok_or(CodecError::UnregisteredComponent(id))?;
        (codec.deserialize_fn)(bytes)
    }

    /// Get the layout for a registered component.
    pub fn layout(&self, id: ComponentId) -> Option<Layout> {
        self.codecs.get(&id).map(|c| c.layout)
    }

    /// Get the stable name for a registered component (explicit name from
    /// `register_as`, or `type_name` default from `register`).
    pub fn stable_name(&self, id: ComponentId) -> Option<&str> {
        self.codecs.get(&id).map(|c| c.name.as_str())
    }

    /// Resolve a stable name to its ComponentId.
    pub fn resolve_name(&self, name: &str) -> Option<ComponentId> {
        self.by_name.get(name).copied()
    }

    /// Check if a component has a registered codec.
    pub fn has_codec(&self, id: ComponentId) -> bool {
        self.codecs.contains_key(&id)
    }

    /// If the archived representation matches the native layout (same size),
    /// returns `Some(size)` — the zero-copy load path can copy archived bytes
    /// directly into BlobVec without typed deserialization.
    pub fn raw_copy_size(&self, id: ComponentId) -> Option<usize> {
        self.codecs.get(&id).and_then(|c| c.raw_copy_size)
    }

    /// Deserialize component bytes, using the `raw_copy_size` fast path (direct
    /// memcpy, no rkyv bytecheck) when a [`CrcProof`] is provided and the
    /// component's archived layout matches its native layout.
    ///
    /// Without a proof, falls through to [`deserialize`](Self::deserialize)
    /// which runs full rkyv validation — safe for untrusted bytes.
    pub fn decode(
        &self,
        id: ComponentId,
        bytes: &[u8],
        proof: Option<&CrcProof>,
    ) -> Result<Vec<u8>, CodecError> {
        if proof.is_some()
            && let Some(size) = self.raw_copy_size(id)
            && bytes.len() == size
        {
            return Ok(bytes.to_vec());
        }
        self.deserialize(id, bytes)
    }

    /// All registered ComponentIds.
    pub fn registered_ids(&self) -> Vec<ComponentId> {
        let mut ids: Vec<_> = self.codecs.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Serialize all entries for a sparse component.
    pub fn serialize_sparse(
        &self,
        id: ComponentId,
        world: &World,
    ) -> Result<Vec<(u64, Vec<u8>)>, CodecError> {
        let codec = self
            .codecs
            .get(&id)
            .ok_or(CodecError::UnregisteredComponent(id))?;
        (codec.serialize_sparse_fn)(world, id, self)
    }

    /// Insert a sparse component value from serialized bytes.
    pub fn insert_sparse_raw(
        &self,
        id: ComponentId,
        world: &mut World,
        entity: Entity,
        data: &[u8],
    ) -> Result<(), CodecError> {
        let codec = self
            .codecs
            .get(&id)
            .ok_or(CodecError::UnregisteredComponent(id))?;
        (codec.insert_sparse_fn)(world, entity, data)
    }

    /// Register a single component type by its ComponentId into the given World.
    /// Used by snapshot restore to register persisted components (with drop fns)
    /// while filling non-persisted gaps with raw placeholders.
    pub fn register_one(&self, id: ComponentId, world: &mut World) {
        if let Some(codec) = self.codecs.get(&id) {
            (codec.register_fn)(world);
        }
    }

    /// Build a remap table from a sender's schema to the receiver's local IDs.
    ///
    /// For each entry in the sender's schema, resolves the stable name to a
    /// local ComponentId and validates that size and align match. Returns a
    /// mapping from sender ComponentId → receiver ComponentId.
    pub fn build_remap(
        &self,
        schema: &[ComponentSchema],
    ) -> Result<HashMap<ComponentId, ComponentId>, CodecError> {
        let mut remap = HashMap::new();
        for def in schema {
            let local_id = self
                .resolve_name(&def.name)
                .ok_or_else(|| CodecError::UnknownComponentName(def.name.clone()))?;
            let local_layout = self
                .layout(local_id)
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
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A proof token returned by [`CrcProof::verify`] after successful CRC32
/// validation of a byte payload. Unforgeable: the only public constructor
/// is [`CrcProof::verify`], which runs the actual checksum.
///
/// Used by [`CodecRegistry::decode`] to gate the `raw_copy_size` fast path
/// (direct memcpy, skipping rkyv bytecheck). Producers: WAL frame reader
/// ([`minkowski_persist::wal::read_next_frame`]), LSM page validator
/// ([`SortedRunReader::validate_page_crc`]).
pub struct CrcProof(());

impl CrcProof {
    /// Verify a payload's CRC32 checksum. Returns proof on success, `None` on mismatch.
    pub fn verify(payload: &[u8], expected_crc: u32) -> Option<Self> {
        if crc32fast::hash(payload) == expected_crc {
            Some(Self(()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn register_and_serialize_round_trip() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let pos = Pos { x: 1.0, y: 2.0 };
        let mut buf = Vec::new();
        unsafe {
            codecs
                .serialize(
                    world.component_id::<Pos>().unwrap(),
                    &pos as *const Pos as *const u8,
                    &mut buf,
                )
                .unwrap();
        }

        let raw = codecs
            .deserialize(world.component_id::<Pos>().unwrap(), &buf)
            .unwrap();

        let restored = unsafe { *(raw.as_ptr() as *const Pos) };
        assert_eq!(restored, pos);
    }

    #[test]
    fn unregistered_component_returns_error() {
        let codecs = CodecRegistry::new();
        let mut buf = Vec::new();
        let result = unsafe { codecs.serialize(999, std::ptr::null(), &mut buf) };
        assert!(matches!(
            result,
            Err(CodecError::UnregisteredComponent(999))
        ));
    }

    #[test]
    fn multiple_components() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        codecs.register::<Vel>(&mut world).unwrap();

        assert!(codecs.has_codec(world.component_id::<Pos>().unwrap()));
        assert!(codecs.has_codec(world.component_id::<Vel>().unwrap()));
        assert_eq!(codecs.registered_ids().len(), 2);
    }

    #[test]
    fn layout_and_name() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();

        let id = world.component_id::<Pos>().unwrap();
        assert_eq!(
            codecs.layout(id).unwrap().size(),
            std::mem::size_of::<Pos>()
        );
        assert!(codecs.stable_name(id).unwrap().contains("Pos"));
    }

    #[test]
    fn register_as_assigns_stable_name() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let id = world.component_id::<Pos>().unwrap();
        assert_eq!(codecs.stable_name(id), Some("pos"));
    }

    #[test]
    fn register_defaults_to_type_name() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world).unwrap();
        let id = world.component_id::<Pos>().unwrap();
        let name = codecs.stable_name(id).unwrap();
        assert!(
            name.contains("Pos"),
            "default name should contain type name, got: {name}"
        );
    }

    #[test]
    fn resolve_name_returns_component_id() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let id = world.component_id::<Pos>().unwrap();
        assert_eq!(codecs.resolve_name("pos"), Some(id));
        assert_eq!(codecs.resolve_name("nonexistent"), None);
    }

    #[test]
    fn duplicate_name_returns_error() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("collision", &mut world).unwrap();
        let result = codecs.register_as::<Vel>("collision", &mut world);
        assert!(matches!(
            result,
            Err(CodecError::DuplicateStableName { .. })
        ));
    }

    #[test]
    fn register_as_idempotent_same_name() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Pos>("pos", &mut world).unwrap(); // no-op
        assert_eq!(codecs.registered_ids().len(), 1);
    }

    #[test]
    fn register_as_same_type_different_name_returns_error() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        let result = codecs.register_as::<Pos>("position", &mut world);
        assert!(matches!(
            result,
            Err(CodecError::DuplicateComponentName { .. })
        ));
    }

    use super::ComponentSchema;

    #[test]
    fn build_remap_same_order() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Vel>("vel", &mut world).unwrap();

        let schema = vec![
            ComponentSchema {
                id: 0,
                name: "pos".into(),
                size: std::mem::size_of::<Pos>(),
                align: std::mem::align_of::<Pos>(),
            },
            ComponentSchema {
                id: 1,
                name: "vel".into(),
                size: std::mem::size_of::<Vel>(),
                align: std::mem::align_of::<Vel>(),
            },
        ];
        let remap = codecs.build_remap(&schema).unwrap();
        assert_eq!(remap.get(&0), Some(&0));
        assert_eq!(remap.get(&1), Some(&1));
    }

    #[test]
    fn build_remap_different_order() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Vel>("vel", &mut world).unwrap(); // id=0 locally
        codecs.register_as::<Pos>("pos", &mut world).unwrap(); // id=1 locally

        // Sender had Pos=0, Vel=1
        let schema = vec![
            ComponentSchema {
                id: 0,
                name: "pos".into(),
                size: std::mem::size_of::<Pos>(),
                align: std::mem::align_of::<Pos>(),
            },
            ComponentSchema {
                id: 1,
                name: "vel".into(),
                size: std::mem::size_of::<Vel>(),
                align: std::mem::align_of::<Vel>(),
            },
        ];
        let remap = codecs.build_remap(&schema).unwrap();
        assert_eq!(remap.get(&0), Some(&1)); // sender 0 (pos) → receiver 1
        assert_eq!(remap.get(&1), Some(&0)); // sender 1 (vel) → receiver 0
    }

    #[test]
    fn build_remap_size_mismatch_is_error() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let schema = vec![ComponentSchema {
            id: 0,
            name: "pos".into(),
            size: 999,
            align: 4,
        }];
        let result = codecs.build_remap(&schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("schema mismatch"));
    }

    #[test]
    fn build_remap_align_mismatch_is_error() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let schema = vec![ComponentSchema {
            id: 0,
            name: "pos".into(),
            size: std::mem::size_of::<Pos>(),
            align: 16, // wrong alignment
        }];
        let result = codecs.build_remap(&schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("schema mismatch"));
    }

    #[test]
    fn build_remap_unknown_name_is_error() {
        let codecs = CodecRegistry::new();
        let schema = vec![ComponentSchema {
            id: 0,
            name: "nonexistent".into(),
            size: 8,
            align: 4,
        }];
        let result = codecs.build_remap(&schema);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unknown component name")
        );
    }
}

#[cfg(test)]
mod raw_copy_tests {
    use super::*;
    use minkowski::World;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    #[repr(C)]
    struct Plain {
        x: f32,
        y: u32,
    }

    #[derive(Clone, Archive, Serialize, Deserialize)]
    struct WithHeap {
        label: String,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    #[repr(C)]
    struct ZstTag;

    #[test]
    fn raw_copyable_type_registers_ok() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        assert!(codecs.register_as::<Plain>("plain", &mut world).is_ok());
    }

    #[test]
    fn non_raw_copyable_type_is_rejected() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        let err = codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap_err();
        assert!(
            matches!(err, CodecError::NotRawCopyable { ref name, .. } if name == "with_heap"),
            "expected NotRawCopyable, got {err:?}"
        );
    }

    #[test]
    fn zst_component_registers_ok() {
        // Zero-sized tag components have archived size == native size (both 0),
        // so they are trivially raw-copyable and must NOT be rejected.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        assert!(codecs.register_as::<ZstTag>("zst_tag", &mut world).is_ok());
    }

    #[test]
    fn zst_decode_round_trip() {
        // The decode() path for ZSTs takes the deserialize() fallback because
        // raw_copy_size is None (nothing to memcpy for a ZST). Verify it
        // succeeds without error and returns an empty buffer.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<ZstTag>("zst_tag", &mut world).unwrap();
        let id = world.component_id::<ZstTag>().unwrap();

        // Serialize a ZstTag to get canonical rkyv bytes.
        let tag = ZstTag;
        let mut buf = Vec::new();
        unsafe {
            codecs
                .serialize(id, &tag as *const ZstTag as *const u8, &mut buf)
                .unwrap();
        }

        // decode() with no proof falls through to deserialize() (full rkyv validation).
        let raw = codecs.decode(id, &buf, None).unwrap();
        // ZST has no bytes; the returned buffer is empty.
        assert_eq!(raw.len(), 0);
    }
}
