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
}

struct ComponentCodec {
    name: String,
    layout: Layout,
    serialize_fn: SerializeFn,
    deserialize_fn: DeserializeFn,
    register_fn: RegisterFn,
    serialize_sparse_fn: SerializeSparseFn,
    insert_sparse_fn: InsertSparseFn,
    /// When `Some(size)`, the archived representation has the same size as the
    /// native type — archived bytes can be copied directly into BlobVec without
    /// typed deserialization. True for `#[repr(C)]` types of primitives on
    /// little-endian platforms where rkyv's archived layout matches native.
    raw_copy_size: Option<usize>,
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

        // If the archived type has the same size as the native type, archived
        // bytes can be copied directly into BlobVec without rkyv::from_bytes.
        // This is true for #[repr(C)] structs of primitives on LE platforms.
        let raw_copy_size = if std::mem::size_of::<T>() == std::mem::size_of::<T::Archived>()
            && layout.size() > 0
        {
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
    pub(crate) fn decode(
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
        schema: &[crate::record::ComponentSchema],
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

/// Zero-sized proof that a byte payload has been CRC32-verified.
///
/// Unforgeable: the only constructor runs the actual CRC computation
/// and compares against the expected checksum. The `raw_copy_size` fast
/// path in [`CodecRegistry::decode`] requires this proof — unverified
/// bytes go through full rkyv bytecheck validation instead.
pub(crate) struct CrcProof(());

impl CrcProof {
    /// Verify a payload's CRC32 checksum. Returns proof on success, `None` on mismatch.
    pub(crate) fn verify(payload: &[u8], expected_crc: u32) -> Option<Self> {
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

    use crate::record::ComponentSchema;

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
