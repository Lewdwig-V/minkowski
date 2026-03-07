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
    pub fn register<T>(&mut self, world: &mut World)
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
        self.register_with_name::<T>(name, world);
    }

    /// Register a component type for persistence with an explicit stable name.
    /// The name must be unique across all registered components — duplicate
    /// names mapped to different ComponentIds will panic.
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
        self.register_with_name::<T>(stable_name.to_owned(), world);
    }

    fn register_with_name<T>(&mut self, stable_name: String, world: &mut World)
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
            assert!(
                existing.name == stable_name,
                "component already registered with name {:?}, cannot re-register as {:?}",
                existing.name,
                stable_name
            );
            return;
        }

        // Duplicate name to a different ComponentId is a hard error.
        if let Some(&existing_id) = self.by_name.get(&stable_name) {
            if existing_id != comp_id {
                panic!(
                    "duplicate stable name {:?}: already registered for ComponentId {}, \
                     cannot register for ComponentId {}",
                    stable_name, existing_id, comp_id
                );
            }
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
        let codec = self
            .codecs
            .get(&id)
            .ok_or(CodecError::UnregisteredComponent(id))?;
        (codec.serialize_fn)(ptr, out)
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

    /// Get the type name for a registered component.
    pub fn name(&self, id: ComponentId) -> Option<&str> {
        self.codecs.get(&id).map(|c| c.name.as_str())
    }

    /// Get the stable name for a registered component.
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

    /// All registered ComponentIds.
    pub fn registered_ids(&self) -> Vec<ComponentId> {
        self.codecs.keys().copied().collect()
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
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
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
        codecs.register::<Pos>(&mut world);

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
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        assert!(codecs.has_codec(world.component_id::<Pos>().unwrap()));
        assert!(codecs.has_codec(world.component_id::<Vel>().unwrap()));
        assert_eq!(codecs.registered_ids().len(), 2);
    }

    #[test]
    fn layout_and_name() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let id = world.component_id::<Pos>().unwrap();
        assert_eq!(
            codecs.layout(id).unwrap().size(),
            std::mem::size_of::<Pos>()
        );
        assert!(codecs.name(id).unwrap().contains("Pos"));
    }

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
        assert!(
            name.contains("Pos"),
            "default name should contain type name, got: {name}"
        );
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
}
