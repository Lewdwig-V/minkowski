use std::alloc::Layout;
use std::collections::HashMap;

use crate::schema::StorageKind;
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

/// Serialized sparse entries for one component: `(entity_bits, value_bytes)`.
pub type SparseEntries = Vec<(u64, Vec<u8>)>;

/// Type-erased sparse serialization: iterates World's sparse storage for a component,
/// returning `(entity_bits, serialized_bytes)` pairs.
type SerializeSparseFn =
    fn(&World, ComponentId, &CodecRegistry) -> Result<SparseEntries, CodecError>;

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
    /// Like `deserialize_fn` but skips rkyv bytecheck via `access_unchecked`.
    /// Only reachable through `deserialize_unchecked_by_type`, which requires a
    /// `CrcProof`. Sound ONLY when the recovery fingerprint gate (spec §2.1) has
    /// also confirmed the on-disk layout matches this binary's.
    deserialize_unchecked_fn: DeserializeFn,
    register_fn: RegisterFn,
    serialize_sparse_fn: SerializeSparseFn,
    insert_sparse_fn: InsertSparseFn,
    /// Native byte size gating the direct-memcpy decode fast path (return the
    /// on-disk payload verbatim as a native image, skipping rkyv). `Some(size)`
    /// ONLY for raw-copyable (POD) non-ZST components, whose rkyv payload IS a
    /// native image; `None` for ZSTs (nothing to copy) AND for Serialized
    /// (heap-backed) components. A heap payload is never a native image even when
    /// its length coincidentally equals the native layout size, so memcpy'ing it
    /// would install archived bytes as native pointer fields → corruption /
    /// double-free. **Invariant: `Some(_) ⟹ raw_copyable`** — the fast-path gate
    /// must not be able to fire for a non-raw-copyable type.
    raw_copy_size: Option<usize>,
    /// Whether this type's rkyv archived size equals its native size (POD, no
    /// heap indirection). RawCopy columns memcpy on recovery; Serialized columns
    /// decode per row. Derived once at registration.
    raw_copyable: bool,
    /// `size_of::<T::Archived>()` — the rkyv archived layout size, captured at
    /// registration. Feeds the decode fingerprint (spec §2.1): a change here
    /// across a binary upgrade fails the gate and forces checked decode.
    archived_size: usize,
    /// `TypeId` of the concrete component type this codec was registered for.
    /// `ComponentId` is a per-world index, so the flush gate compares this to the
    /// flushed world's `component_type_id` to confirm the codec actually
    /// describes the type whose native bytes are being persisted.
    type_id: std::any::TypeId,
    /// `std::any::type_name::<T>()` — the dense on-disk schema stores this
    /// (via `World::component_name`), NOT the codec stable name (which may be a
    /// `register_as` alias). Fingerprint layout lookups resolve by THIS so an
    /// aliased codec still resolves.
    type_name: &'static str,
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

    /// Whether a codec is registered for the given component *type*, regardless
    /// of which numeric `ComponentId` it was filed under. The flush gate uses
    /// this to certify a dense column's type is raw-copyable: `ComponentId` is a
    /// per-world index, so resolving by type (not id) is correct across a
    /// registry and world that assigned the type different ids (e.g. after
    /// recovery re-registers components into a fresh world).
    pub fn has_codec_for_type(&self, type_id: std::any::TypeId) -> bool {
        self.codecs.values().any(|c| c.type_id == type_id)
    }

    /// The on-disk storage kind for a registered component *type*, or `None` if
    /// no codec is registered for it. Resolve by type, never by ComponentId.
    pub fn storage_kind_for_type(&self, type_id: std::any::TypeId) -> Option<StorageKind> {
        let codec = self.codecs.values().find(|c| c.type_id == type_id)?;
        Some(if codec.raw_copyable {
            StorageKind::RawCopy
        } else {
            StorageKind::Serialized
        })
    }

    /// Serialize one component value (rkyv) by its *type*. Resolves the codec by
    /// `TypeId` so it is correct across worlds whose ComponentIds diverge from
    /// this registry's. Returns `None` if no codec is registered for the type.
    ///
    /// # Safety
    /// `ptr` must point to a valid, aligned instance of the type identified by
    /// `type_id`, valid for reads of that type's size.
    pub unsafe fn serialize_by_type(
        &self,
        type_id: std::any::TypeId,
        ptr: *const u8,
        out: &mut Vec<u8>,
    ) -> Option<Result<(), CodecError>> {
        let codec = self.codecs.values().find(|c| c.type_id == type_id)?;
        Some(unsafe { (codec.serialize_fn)(ptr, out) })
    }

    /// Deserialize component bytes into a native-byte buffer by *type* (full
    /// rkyv validation; no raw-copy fast path — Serialized columns are never
    /// raw-copyable). The returned buffer owns a reconstructed `T`; its drop
    /// responsibility passes to whatever takes ownership of the bytes (on
    /// recovery, the archetype column, which holds T's drop_fn). Returns `None`
    /// if no codec is registered for the type.
    pub fn deserialize_by_type(
        &self,
        type_id: std::any::TypeId,
        bytes: &[u8],
    ) -> Option<Result<Vec<u8>, CodecError>> {
        let codec = self.codecs.values().find(|c| c.type_id == type_id)?;
        Some((codec.deserialize_fn)(bytes))
    }

    /// Deserialize component bytes into a native-byte buffer by *type*, skipping
    /// rkyv bytecheck (`access_unchecked` + `deserialize`). Returns `None` if no
    /// codec is registered for the type.
    ///
    /// # Safety
    /// `bytes` MUST be a valid rkyv archive of the type identified by `type_id` —
    /// i.e. produced by this binary's codec for that type and not since mutated.
    /// `CrcProof` proves only byte INTEGRITY (the bytes are unchanged since the
    /// writer emitted them); it does NOT prove the writer emitted a structurally
    /// valid archive of THIS binary's type. The caller must independently establish
    /// well-formedness — recovery does so via the per-run decode-fingerprint gate
    /// (`fingerprint::run_fingerprint` match) before reaching this. Calling with a
    /// CRC-valid-but-malformed archive is undefined behavior.
    pub unsafe fn deserialize_unchecked_by_type(
        &self,
        type_id: std::any::TypeId,
        bytes: &[u8],
        _proof: &CrcProof,
    ) -> Option<Result<Vec<u8>, CodecError>> {
        let codec = self.codecs.values().find(|c| c.type_id == type_id)?;
        Some((codec.deserialize_unchecked_fn)(bytes))
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
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>
            + rkyv::Portable,
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
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>
            + rkyv::Portable,
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
            + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>
            + rkyv::Portable,
    {
        // Mechanical enforcement (audit N4): both decode paths realign rows into
        // `AlignedVec::<16>`, which is only sufficient when the archived type's
        // alignment is <= 16. Enforce per-monomorphization at COMPILE time.
        const {
            assert!(
                std::mem::align_of::<T::Archived>() <= 16,
                "T::Archived alignment exceeds the AlignedVec<16> used by decode"
            );
        }

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

        // Raw-copyability classification: a type whose rkyv archived size equals
        // its native size is plain-old-data (no heap indirection) — its native
        // bytes are position-independent and may be flushed/memcpy'd verbatim
        // (RawCopy). Heap-backed types (String, Vec, …) have a differently-sized
        // archived form; they persist via rkyv (Serialized) and decode per row on
        // recovery. ZSTs satisfy 0 == 0 and are RawCopy. No type is rejected here.
        let raw_copyable = std::mem::size_of::<T>() == std::mem::size_of::<T::Archived>();

        // `raw_copy_size` gates the decode() memcpy fast path, which is sound
        // ONLY for raw-copyable types (their rkyv payload is a native image).
        // Gate it on `raw_copyable` so the invariant `Some(_) ⟹ raw_copyable`
        // holds: a Serialized (heap) component must never expose a size, or a
        // payload whose length coincidentally equals the native layout size
        // would be memcpy'd as native pointer fields. ZSTs have nothing to copy.
        let raw_copy_size = if raw_copyable && layout.size() > 0 {
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
            // `rkyv::from_bytes` accesses the archive in place and requires the
            // buffer aligned to the archived type's alignment. Dense recovery
            // hands us row slices at arbitrary offsets within a page body
            // (`[offsets:(n+1)×u32][values]`), so the slice is essentially never
            // aligned — an unaligned `access` is UB (and Miri-rejected). Realign
            // into an AlignedVec (16-byte aligned, covers all archived alignments).
            let mut aligned = rkyv::util::AlignedVec::<16>::new();
            aligned.extend_from_slice(bytes);
            let value: T = rkyv::from_bytes::<T, rancor::Error>(&aligned)
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

        let deserialize_unchecked_fn: DeserializeFn = |bytes| {
            // Same realignment as `deserialize_fn`: `access_unchecked` reads the
            // archive in place and the row slice is essentially never aligned.
            let mut aligned = rkyv::util::AlignedVec::<16>::new();
            aligned.extend_from_slice(bytes);
            // Defense-in-depth (audit N3): `access_unchecked` roots at the buffer
            // tail and reads `size_of::<T::Archived>()` bytes there. The framing
            // layer (`serialized_page::decode`) already bounds-checks each row
            // slice against the CRC-validated page body, so a sub-root-size row is
            // only reachable via a same-binary writer bug — but turn that into a
            // clean error rather than an OOB read. (Bad INTERNAL structure of a
            // correctly-sized archive is still trusted — that is the bytecheck we
            // intentionally skip.)
            if aligned.len() < std::mem::size_of::<T::Archived>() {
                return Err(CodecError::Deserialize(format!(
                    "unchecked decode: row is {} bytes but archived root needs {}",
                    aligned.len(),
                    std::mem::size_of::<T::Archived>()
                )));
            }
            // SAFETY: the only caller, `deserialize_unchecked_by_type`, requires a
            // `CrcProof` (unforgeable; minted only by a real CRC check), so `bytes`
            // are integrity-verified, writer-authored archive bytes. Combined with
            // the recovery-side fingerprint gate (only runs whose Serialized layout
            // matches this binary reach the unchecked path), these are a valid
            // archive of `T::Archived`; skipping bytecheck is sound. See spec §2/§2.1.
            // The page framing (offset table → row slices) is validated by
            // `serialized_page::decode` on BOTH the checked and unchecked paths
            // before this runs, so the row slice is already guaranteed in-bounds of
            // the CRC-validated page body; the ONLY validation skipped here is
            // per-row rkyv bytecheck (internal relative-pointer / length / UTF-8
            // checks).
            let archived = unsafe { rkyv::access_unchecked::<T::Archived>(&aligned) };
            let value: T = rkyv::deserialize::<T, rancor::Error>(archived)
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

        let serialize_sparse_fn: SerializeSparseFn = |world, comp_id, _codecs| {
            let mut entries = Vec::new();
            if let Some(iter) = world.iter_sparse::<T>(comp_id) {
                for (entity, value) in iter {
                    // Serialize `T` directly — this closure is monomorphized for
                    // `T`, so it does not need to dispatch through the registry by
                    // `comp_id`. `comp_id` is the WORLD's sparse id, used only to
                    // iterate `iter_sparse`; it need not equal this codec's
                    // registry id (they diverge after recovery re-registers into a
                    // fresh world). Routing the serialize through `comp_id` would
                    // pick the wrong codec across worlds.
                    let buf = rkyv::api::high::to_bytes_in::<_, rancor::Error>(value, Vec::new())
                        .map_err(|e| CodecError::Serialize(e.to_string()))?;
                    entries.push((entity.to_bits(), buf));
                }
            }
            Ok(entries)
        };

        let insert_sparse_fn: InsertSparseFn = |world, entity, data| {
            // Same realignment requirement as `deserialize_fn`: `from_bytes`
            // accesses the archive in place and the caller-supplied slice is not
            // guaranteed aligned. Copy into an AlignedVec before decoding.
            let mut aligned = rkyv::util::AlignedVec::<16>::new();
            aligned.extend_from_slice(data);
            let value: T = rkyv::from_bytes::<T, rancor::Error>(&aligned)
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
                deserialize_unchecked_fn,
                register_fn,
                serialize_sparse_fn,
                insert_sparse_fn,
                raw_copy_size,
                raw_copyable,
                archived_size: std::mem::size_of::<T::Archived>(),
                type_id: std::any::TypeId::of::<T>(),
                type_name: std::any::type_name::<T>(),
            },
        );
        Ok(())
    }

    /// Serialize a component value from a raw pointer to bytes.
    ///
    /// Resolves the codec by per-world `ComponentId`. This is ONLY valid when the
    /// registry and the world share an id space — e.g. WAL write/replay against
    /// the same world the registry was built for. For any CROSS-WORLD path
    /// (recovery, flush), the recovered world's local ids diverge from the
    /// registry's; use [`serialize_by_type`](Self::serialize_by_type) /
    /// [`deserialize_by_type`](Self::deserialize_by_type) instead, which key by
    /// component `TypeId`.
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

    /// Deserialize component bytes into a raw byte buffer. Internal to the
    /// crate: callers outside use [`decode`](Self::decode), which gates the
    /// raw-copy fast path on a [`CrcProof`].
    ///
    /// Resolves the codec by per-world `ComponentId`, so it is only valid when the
    /// registry and the world share an id space (e.g. WAL replay against the same
    /// world). For cross-world paths (recovery, flush) use
    /// [`deserialize_by_type`](Self::deserialize_by_type), which keys by `TypeId`.
    pub(crate) fn deserialize(&self, id: ComponentId, bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
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

    /// Archived (rkyv) size for the component whose Rust `type_name` is `type_name`
    /// (the dense schema key). Resolves by `type_name`, NOT the codec stable name.
    pub fn archived_size_by_type_name(&self, type_name: &str) -> Option<usize> {
        self.codecs
            .values()
            .find(|c| c.type_name == type_name)
            .map(|c| c.archived_size)
    }

    /// Native (in-memory) size and align for the component whose Rust `type_name`
    /// is `type_name` (the dense schema key). Resolves by `type_name`.
    pub fn native_layout_by_type_name(&self, type_name: &str) -> Option<(usize, usize)> {
        self.codecs
            .values()
            .find(|c| c.type_name == type_name)
            .map(|c| (c.layout.size(), c.layout.align()))
    }

    /// Resolve a stable name to its ComponentId.
    pub fn resolve_name(&self, name: &str) -> Option<ComponentId> {
        self.by_name.get(name).copied()
    }

    /// If the archived representation matches the native layout (same size),
    /// returns `Some(size)` — the zero-copy load path can copy archived bytes
    /// directly into BlobVec without typed deserialization. Internal to the
    /// crate; gated by [`decode`](Self::decode).
    pub(crate) fn raw_copy_size(&self, id: ComponentId) -> Option<usize> {
        self.codecs.get(&id).and_then(|c| c.raw_copy_size)
    }

    /// Deserialize component bytes, using the `raw_copy_size` fast path (direct
    /// memcpy, no rkyv bytecheck) when a [`CrcProof`] is provided and the
    /// component's archived layout matches its native layout.
    ///
    /// Without a proof, falls through to full rkyv validation — safe for
    /// untrusted bytes.
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

    /// Serialize all entries for a world's sparse component, resolving the codec
    /// by the world's component **type** at `world_comp_id` — never by matching
    /// the registry's id. `ComponentId` is a per-world index, so a registry whose
    /// ids diverge from the flushed world's (e.g. after recovery re-registers
    /// into a fresh world) would otherwise select the wrong codec or none. On
    /// success returns the codec's stable name (for the on-disk blob; recovery
    /// resolves it back via `resolve_name`) and the serialized entries. Returns
    /// `None` if no codec is registered for that type.
    pub fn serialize_sparse_by_type(
        &self,
        world: &World,
        world_comp_id: ComponentId,
    ) -> Option<Result<(String, SparseEntries), CodecError>> {
        let ty = world.component_type_id(world_comp_id)?;
        let codec = self.codecs.values().find(|c| c.type_id == ty)?;
        match (codec.serialize_sparse_fn)(world, world_comp_id, self) {
            Ok(entries) => Some(Ok((codec.name.clone(), entries))),
            Err(e) => Some(Err(e)),
        }
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

        assert!(codecs.has_codec_for_type(std::any::TypeId::of::<Pos>()));
        assert!(codecs.has_codec_for_type(std::any::TypeId::of::<Vel>()));
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
    fn non_raw_copyable_type_registers_and_classifies_serialized() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .expect("heap-backed components are now persistable");
        let ty = std::any::TypeId::of::<WithHeap>();
        assert_eq!(
            codecs.storage_kind_for_type(ty),
            Some(crate::schema::StorageKind::Serialized)
        );
    }

    #[test]
    fn raw_copyable_and_zst_classify_raw_copy() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Plain>("plain", &mut world).unwrap();
        codecs.register_as::<ZstTag>("zst", &mut world).unwrap();
        assert_eq!(
            codecs.storage_kind_for_type(std::any::TypeId::of::<Plain>()),
            Some(crate::schema::StorageKind::RawCopy)
        );
        assert_eq!(
            codecs.storage_kind_for_type(std::any::TypeId::of::<ZstTag>()),
            Some(crate::schema::StorageKind::RawCopy)
        );
        assert_eq!(
            codecs.storage_kind_for_type(std::any::TypeId::of::<WithHeap>()),
            None,
            "unregistered type has no kind"
        );
    }

    #[test]
    fn serialize_then_deserialize_by_type_round_trips_heap() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap();
        let ty = std::any::TypeId::of::<WithHeap>();

        let value = WithHeap {
            label: "hello heap".to_owned(),
        };
        let mut bytes = Vec::new();
        unsafe {
            codecs
                .serialize_by_type(ty, &value as *const WithHeap as *const u8, &mut bytes)
                .expect("codec exists for type")
                .expect("serialize ok");
        }

        let native = codecs
            .deserialize_by_type(ty, &bytes)
            .expect("codec exists for type")
            .expect("deserialize ok");
        // SAFETY: `native` byte-owns a valid `WithHeap` (deserialize_by_type
        // reconstructs it and transfers ownership into the buffer). Read a
        // bitwise copy out to assert on, then forget it below so `native`
        // remains the sole owner of the heap `String` (no double free).
        let restored = unsafe { std::ptr::read(native.as_ptr() as *const WithHeap) };
        assert_eq!(restored.label, "hello heap");
        std::mem::forget(restored); // bytes still own the String; avoid double free
    }

    #[test]
    fn deserialize_by_type_handles_unaligned_input() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap();
        let ty = std::any::TypeId::of::<WithHeap>();

        let value = WithHeap {
            label: "alignment matters".to_owned(),
        };
        let mut bytes = Vec::new();
        unsafe {
            codecs
                .serialize_by_type(ty, &value as *const WithHeap as *const u8, &mut bytes)
                .unwrap()
                .unwrap();
        }

        // Force a misaligned view: prepend one byte, then deserialize from the
        // offset-1 sub-slice. `access` on this unaligned slice is UB without the
        // AlignedVec realign — Miri (and strict-alignment targets) catch it.
        let mut shifted = Vec::with_capacity(bytes.len() + 1);
        shifted.push(0u8);
        shifted.extend_from_slice(&bytes);
        let unaligned = &shifted[1..];

        let native = codecs.deserialize_by_type(ty, unaligned).unwrap().unwrap();
        // `native` is a `Vec<u8>` byte-owning a `WithHeap`; its buffer is not
        // guaranteed aligned to `WithHeap`, so read it out unaligned. (Real
        // recovery memcpys these bytes into an aligned archetype column.) Take
        // ownership of the reconstructed value, then drop the raw byte buffer.
        // `Vec<u8>` drop only frees bytes (u8 has no Drop), so the inner `String`
        // is freed exactly once — by `restored` — with no double free or leak.
        let restored = unsafe { std::ptr::read_unaligned(native.as_ptr() as *const WithHeap) };
        drop(native);
        assert_eq!(restored.label, "alignment matters");
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

    #[test]
    fn serialized_codec_has_no_raw_copy_size() {
        // `raw_copy_size` gates decode()'s memcpy fast path (return the payload
        // verbatim as a native image). That is sound ONLY for raw-copyable (POD)
        // types. A heap-backed (Serialized) codec must expose `None`, or a
        // WAL/replication payload whose length coincidentally equals the native
        // layout size would be installed as native pointer fields → corruption +
        // double-free. Invariant: `Some(_) ⟹ raw_copyable`.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap();
        let heap_id = world.component_id::<WithHeap>().unwrap();
        assert_eq!(
            codecs.raw_copy_size(heap_id),
            None,
            "Serialized (heap) codec must not expose a raw_copy_size"
        );

        // POD types keep their raw_copy_size — the fast path stays enabled.
        codecs.register_as::<Plain>("plain", &mut world).unwrap();
        let pod_id = world.component_id::<Plain>().unwrap();
        assert!(
            codecs.raw_copy_size(pod_id).is_some(),
            "raw-copyable (POD) codec keeps its raw_copy_size"
        );
    }

    #[test]
    fn unchecked_decode_matches_checked_for_heap() {
        // Given CRC-proven bytes, the unchecked path must reconstruct the SAME
        // native value as the checked path. This is the core correctness guard:
        // access_unchecked + deserialize == from_bytes for valid bytes.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap();
        let ty = std::any::TypeId::of::<WithHeap>();

        let value = WithHeap {
            label: "round trip me".to_owned(),
        };
        let mut bytes = Vec::new();
        unsafe {
            codecs
                .serialize_by_type(ty, &value as *const WithHeap as *const u8, &mut bytes)
                .unwrap()
                .unwrap();
        }

        let checked = codecs.deserialize_by_type(ty, &bytes).unwrap().unwrap();
        let proof = CrcProof::verify(&bytes, crc32fast::hash(&bytes)).unwrap();
        // SAFETY: `bytes` was just serialized by this binary's codec for
        // `WithHeap`, so it is a valid archive of that type.
        let unchecked = unsafe { codecs.deserialize_unchecked_by_type(ty, &bytes, &proof) }
            .expect("codec exists")
            .expect("unchecked decode ok");

        // Each buffer byte-owns a reconstructed WithHeap. Read both out
        // unaligned (Vec<u8> is byte-aligned), compare, then forget the copies
        // so the owning buffers free each String exactly once.
        let a = unsafe { std::ptr::read_unaligned(checked.as_ptr() as *const WithHeap) };
        let b = unsafe { std::ptr::read_unaligned(unchecked.as_ptr() as *const WithHeap) };
        assert_eq!(a.label, "round trip me");
        assert_eq!(a.label, b.label);
        std::mem::forget(a);
        std::mem::forget(b);
    }

    #[test]
    fn archived_size_by_name_reports_heap_and_pod() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Plain>("plain", &mut world).unwrap();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap();
        // POD: archived size == native size. Resolve by Rust type_name (the dense
        // schema key), NOT the register_as alias.
        assert_eq!(
            codecs.archived_size_by_type_name(std::any::type_name::<Plain>()),
            Some(std::mem::size_of::<<Plain as rkyv::Archive>::Archived>())
        );
        // Heap: archived size is rkyv's ArchivedString layout (differs from native).
        assert_eq!(
            codecs.archived_size_by_type_name(std::any::type_name::<WithHeap>()),
            Some(std::mem::size_of::<<WithHeap as rkyv::Archive>::Archived>())
        );
        assert_eq!(codecs.archived_size_by_type_name("nope"), None);
    }

    #[test]
    fn decode_with_proof_never_memcpys_heap_payload() {
        // Regression (the raw_copy_size soundness hole): a heap codec must never
        // take decode()'s proof/memcpy fast path, even when the payload length
        // equals the native layout size. Hand it bytes of exactly that length
        // that are NOT a valid native image; decode-with-proof must REJECT them
        // via full rkyv validation, not return them verbatim as a garbage-pointer
        // native value. Before the fix (`raw_copy_size = Some(native_size)` for
        // heap types) this returned `Ok(evil)` — the corruption vector.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_as::<WithHeap>("with_heap", &mut world)
            .unwrap();
        let id = world.component_id::<WithHeap>().unwrap();

        let evil = vec![0xABu8; std::mem::size_of::<WithHeap>()];
        let proof = CrcProof::verify(&evil, crc32fast::hash(&evil))
            .expect("crc matches the bytes we just hashed");
        let result = codecs.decode(id, &evil, Some(&proof));
        assert!(
            result.is_err(),
            "heap decode must rkyv-validate, not memcpy garbage as a native image; got {result:?}"
        );
    }
}
