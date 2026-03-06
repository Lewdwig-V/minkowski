use std::alloc::Layout;
use std::collections::HashMap;

use minkowski::component::Component;
use minkowski::{ComponentId, Entity, World};
use rkyv::api::high::HighValidator;
use rkyv::bytecheck::CheckBytes;
use rkyv::de::Pool;
use rkyv::rancor;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
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

#[derive(Debug)]
pub enum CodecError {
    Serialize(String),
    Deserialize(String),
    UnregisteredComponent(ComponentId),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialize(msg) => write!(f, "serialize: {msg}"),
            Self::Deserialize(msg) => write!(f, "deserialize: {msg}"),
            Self::UnregisteredComponent(id) => write!(f, "no codec for component {id}"),
        }
    }
}

impl std::error::Error for CodecError {}

struct ComponentCodec {
    name: &'static str,
    layout: Layout,
    serialize_fn: SerializeFn,
    deserialize_fn: DeserializeFn,
    register_fn: RegisterFn,
    serialize_sparse_fn: SerializeSparseFn,
    insert_sparse_fn: InsertSparseFn,
}

/// Maps ComponentId to rkyv codecs. Separate from core's ComponentRegistry —
/// different concerns, different crates, different lifetimes.
pub struct CodecRegistry {
    codecs: HashMap<ComponentId, ComponentCodec>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        Self {
            codecs: HashMap::new(),
        }
    }

    /// Register a component type for persistence.
    /// Requires rkyv Archive + Serialize + Deserialize bounds.
    pub fn register<T>(&mut self, world: &mut World)
    where
        T: Component
            + Archive
            + for<'a> RkyvSerialize<
                rkyv::api::high::HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>,
            > + Clone,
        T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
    {
        let comp_id = world.register_component::<T>();
        let layout = Layout::new::<T>();
        let name = std::any::type_name::<T>();

        let serialize_fn: SerializeFn = |ptr, out| {
            let value = unsafe { &*ptr.cast::<T>() };
            let bytes = rkyv::to_bytes::<rancor::Error>(value)
                .map_err(|e| CodecError::Serialize(e.to_string()))?;
            out.extend_from_slice(&bytes);
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

        self.codecs.insert(
            comp_id,
            ComponentCodec {
                name,
                layout,
                serialize_fn,
                deserialize_fn,
                register_fn,
                serialize_sparse_fn,
                insert_sparse_fn,
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
    pub fn name(&self, id: ComponentId) -> Option<&'static str> {
        self.codecs.get(&id).map(|c| c.name)
    }

    /// Check if a component has a registered codec.
    pub fn has_codec(&self, id: ComponentId) -> bool {
        self.codecs.contains_key(&id)
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
}
