//! Blob offloading — store large data references in the ECS, actual bytes
//! in an external store.
//!
//! Demonstrates the `BlobRef` component + `BlobStore` cleanup trait pattern:
//! entities hold lightweight key strings, while bulk data lives in a
//! user-managed object store. On despawn the user collects orphaned refs
//! and calls the store's cleanup hook.
//!
//! Run: cargo run -p minkowski-examples --example blob --release

use minkowski::{Entity, World};
use minkowski_persist::{BlobRef, BlobStore};
use std::collections::HashMap;

/// Simulated object store (in-memory HashMap standing in for S3/MinIO).
struct MemoryBlobStore {
    objects: HashMap<String, Vec<u8>>,
}

impl MemoryBlobStore {
    fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    fn put(&mut self, key: &str, data: Vec<u8>) -> BlobRef {
        self.objects.insert(key.to_owned(), data);
        BlobRef::new(key)
    }

    fn get(&self, key: &str) -> Option<&[u8]> {
        self.objects.get(key).map(Vec::as_slice)
    }
}

impl BlobStore for MemoryBlobStore {
    fn on_orphaned(&mut self, refs: &[&BlobRef]) {
        for r in refs {
            println!("  Deleting blob: {}", r.key());
            self.objects.remove(r.key());
        }
    }
}

fn main() {
    let mut world = World::new();
    let mut store = MemoryBlobStore::new();

    // Spawn entities with blob references
    println!("--- Spawning entities with blob refs ---");
    let mut entities: Vec<Entity> = Vec::new();
    for i in 0..5 {
        let key = format!("s3://bucket/object_{i}.bin");
        let data = vec![i; 1024]; // 1 KB "large" blob
        let blob_ref = store.put(&key, data);
        let e = world.spawn((blob_ref,));
        entities.push(e);
        println!("  Entity {i}: {key}");
    }
    println!("Store has {} objects", store.objects.len());

    // Despawn some entities -- collect orphaned BlobRefs for cleanup
    println!("\n--- Despawning entities 1 and 3 ---");
    let mut orphaned = Vec::new();
    for &i in &[1_usize, 3] {
        let e = entities[i];
        if let Some(blob_ref) = world.get::<BlobRef>(e) {
            orphaned.push(blob_ref.clone());
        }
        world.despawn(e);
    }

    // User calls cleanup -- engine doesn't do this automatically
    println!("\n--- Running BlobStore cleanup ---");
    let borrowed: Vec<&BlobRef> = orphaned.iter().collect();
    store.on_orphaned(&borrowed);
    println!("Store now has {} objects", store.objects.len());

    // Verify remaining blobs are accessible
    println!("\n--- Remaining blob refs ---");
    world.query::<(&BlobRef,)>().for_each(|(r,)| {
        let size = store.get(r.key()).map_or(0, <[u8]>::len);
        println!("  {r}: {size} bytes"); // BlobRef implements Display
    });

    println!("\nDone.");
}
