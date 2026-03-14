//! External blob reference component and lifecycle trait.
//!
//! `BlobRef` holds a key/URL to data in an external object store (S3, MinIO,
//! local filesystem). The ECS stores only the reference — blob bytes never
//! enter the World. Same external composition pattern as
//! [`SpatialIndex`](minkowski::SpatialIndex).
//!
//! `BlobRef` derives rkyv `Archive`/`Serialize`/`Deserialize` and can be
//! registered with [`CodecRegistry`](crate::CodecRegistry) like any other
//! persistable component.

/// Reference to an externally-stored blob.
///
/// The ECS stores only this key string — the actual blob bytes live in an
/// external object store. Persistence serializes the key, not the remote blob.
/// On snapshot restore, keys are restored but remote blobs must still exist.
#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct BlobRef(String);

impl BlobRef {
    /// Create a new blob reference from a key string.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `key` is non-empty. Empty keys are never meaningful
    /// in any object store backend.
    pub fn new(key: impl Into<String>) -> Self {
        let key = key.into();
        debug_assert!(!key.is_empty(), "BlobRef key must not be empty");
        Self(key)
    }

    /// The key/URL string.
    pub fn key(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BlobRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lifecycle hook for external blob storage.
///
/// The engine does **not** call these methods automatically. Users wire them
/// into cleanup reducers or framework-level hooks. Same responsibility model
/// as [`SpatialIndex::rebuild`](minkowski::SpatialIndex::rebuild).
///
/// # Shared-key safety
///
/// If multiple entities share the same blob key, the caller must verify no
/// remaining entity holds the key before including it in the orphaned set.
/// Passing refs that are still attached to live entities will cause data loss.
///
/// # Example
///
/// ```ignore
/// struct S3Store { client: S3Client }
///
/// impl BlobStore for S3Store {
///     fn on_orphaned(&mut self, refs: &[&BlobRef]) {
///         for r in refs {
///             self.client.delete_object(r.key());
///         }
///     }
/// }
/// ```
pub trait BlobStore {
    /// Called with blob references no longer attached to any live entity.
    /// The implementor is responsible for deleting the external blobs.
    fn on_orphaned(&mut self, refs: &[&BlobRef]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::World;

    #[test]
    fn blob_ref_is_a_component() {
        let mut world = World::new();
        let e = world.spawn((BlobRef::new("s3://bucket/key.bin"),));
        let got = world.get::<BlobRef>(e).unwrap();
        assert_eq!(got.key(), "s3://bucket/key.bin");
    }

    #[test]
    fn blob_ref_survives_archetype_migration() {
        let mut world = World::new();
        let e = world.spawn((BlobRef::new("s3://a"),));
        world.insert(e, (42u32,)).unwrap();
        assert_eq!(world.get::<BlobRef>(e).unwrap().key(), "s3://a");
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
    }

    #[test]
    fn blob_store_receives_orphaned_refs() {
        struct TestStore {
            deleted: Vec<String>,
        }
        impl BlobStore for TestStore {
            fn on_orphaned(&mut self, refs: &[&BlobRef]) {
                for r in refs {
                    self.deleted.push(r.key().to_owned());
                }
            }
        }

        let refs = [BlobRef::new("s3://a"), BlobRef::new("s3://b")];
        let borrowed: Vec<&BlobRef> = refs.iter().collect();
        let mut store = TestStore { deleted: vec![] };
        store.on_orphaned(&borrowed);

        assert_eq!(store.deleted, vec!["s3://a", "s3://b"]);
    }

    #[test]
    fn blob_ref_query_iteration() {
        let mut world = World::new();
        world.spawn((BlobRef::new("s3://1"),));
        world.spawn((BlobRef::new("s3://2"),));
        world.spawn((42u32,)); // different archetype, no BlobRef

        let mut keys: Vec<String> = Vec::new();
        world.query::<(&BlobRef,)>().for_each(|(r,)| {
            keys.push(r.key().to_owned());
        });
        keys.sort();
        assert_eq!(keys, vec!["s3://1", "s3://2"]);
    }

    #[test]
    fn blob_ref_display() {
        let r = BlobRef::new("s3://bucket/key.bin");
        assert_eq!(r.to_string(), "s3://bucket/key.bin");
        assert_eq!(format!("{r}"), "s3://bucket/key.bin");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "BlobRef key must not be empty")]
    fn blob_ref_empty_key_panics_in_debug() {
        BlobRef::new("");
    }

    #[test]
    fn blob_ref_codec_registration() {
        let mut world = World::new();
        let mut codecs = crate::CodecRegistry::new();
        codecs.register::<BlobRef>(&mut world).unwrap();

        let id = world.component_id::<BlobRef>().unwrap();
        assert!(codecs.stable_name(id).is_some());
    }
}
