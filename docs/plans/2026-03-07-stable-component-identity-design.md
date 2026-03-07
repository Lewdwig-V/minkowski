# Stable Component Identity Design

**Goal:** Make WAL records and snapshots interpretable across process boundaries by replacing bare `ComponentId` (registration-order-dependent usize) with stable string names resolved through `CodecRegistry`.

**Architecture:** `CodecRegistry` gains a `by_name: HashMap<String, ComponentId>` reverse lookup. WAL files gain a schema preamble mapping sender-local IDs to stable names. Receivers build a remap table from preamble names to their own local IDs. Snapshots align to the same name-based resolution. Engine core (`ComponentRegistry`, `World`) stays untouched.

## The Problem

`ComponentId` is a sequential usize assigned by registration order. Process A registering `(Pos, Vel)` and process B registering `(Vel, Pos)` produce different IDs. Today's WAL and snapshot formats serialize bare `ComponentId` usizes — they only work when the reader has identical registration order (same binary, crash recovery). Cross-process replication requires a stable identity that survives different registration orders.

## Design Decisions

### Stable identity key: user-assigned string with `type_name` default

```rust
// Explicit stable name
codecs.register_as::<Pos>("pos", &mut world);

// Unchanged — defaults to std::any::type_name::<Pos>()
codecs.register::<Pos>(&mut world);
```

**Why not `type_name` only:** Renaming or moving a Rust type changes `type_name`, breaking the wire format. Explicit names decouple wire identity from module paths.

**Why not content hash:** A u64 hash is compact but loses human readability for WAL debugging, and hash collisions cause silent corruption. String names fail loudly on mismatch.

### Name mapping lives in `CodecRegistry`, not `ComponentRegistry`

Stable identity is a persistence/replication concern. `ComponentRegistry` is `pub(crate)` engine internals. `CodecRegistry` already stores per-component metadata (name, layout, serialize/deserialize closures). Adding `by_name` there keeps the concern in `minkowski-persist` and follows the "mechanisms not policy" principle.

### Schema preamble + local IDs on the wire

WAL mutations continue to carry compact sender-local `ComponentId` usizes. A schema preamble at the start of each WAL file maps those IDs to stable names:

```
WAL file: [Schema record] [Mutation record 0] [Mutation record 1] ...
```

The receiver builds a remap table once from the preamble, then applies it to each mutation's `ComponentId` fields during replay. Per-mutation overhead stays at one usize.

**Alternative rejected:** Replacing `ComponentId` with the name string in every mutation. Self-describing but repeats `"position"` across thousands of mutations.

### Strict layout validation

Size or align mismatch between sender and receiver is a hard error. No lenient mode. Schema evolution with field migration is a future feature that needs explicit support, not something to paper over by being permissive.

## Data Structures

### CodecRegistry additions

```rust
pub struct CodecRegistry {
    codecs: HashMap<ComponentId, ComponentCodec>,
    by_name: HashMap<String, ComponentId>,     // NEW
}

impl CodecRegistry {
    /// Register with explicit stable name.
    pub fn register_as<T>(&mut self, name: &str, world: &mut World)
    where T: Component + Archive + ... { ... }

    /// Resolve a stable name to the local ComponentId.
    pub fn resolve_name(&self, name: &str) -> Option<ComponentId> { ... }
}
```

`ComponentCodec.name` changes from `&'static str` (type_name) to `String` (explicit or defaulted). Duplicate name registration panics.

### WAL format changes

```rust
pub enum WalEntry {
    Schema(WalSchema),
    Mutations(WalRecord),       // existing WalRecord unchanged
}

pub struct WalSchema {
    pub components: Vec<WalComponentDef>,
}

pub struct WalComponentDef {
    pub id: ComponentId,        // sender's local ID
    pub name: String,           // stable name
    pub size: usize,            // layout validation
    pub align: usize,           // layout validation
}
```

On-disk format unchanged: `[len: u32 LE][payload: len bytes]` repeated. The first payload is a `WalEntry::Schema`, subsequent payloads are `WalEntry::Mutations`. rkyv serialization handles the enum discriminant.

### Shared remap function

```rust
fn build_remap(
    schema: &[WalComponentDef],
    codecs: &CodecRegistry,
) -> Result<HashMap<ComponentId, ComponentId>, SchemaError>
```

For each entry: `codecs.resolve_name(&entry.name)` to find receiver's local ID, validate size/align match, build sender-ID → receiver-ID mapping. Used by both WAL replay and snapshot load.

## Write Path

1. `Wal::create` writes a `WalEntry::Schema` as the first record, built from all registered codecs in `CodecRegistry`
2. `Wal::append` writes `WalEntry::Mutations` as before — mutations carry sender's local `ComponentId`
3. `Snapshot::save` populates `ComponentSchema.name` from `CodecRegistry`'s stable name (not `World::component_name`)

## Read Path

1. `Wal::open` / `replay_from` parses the first record as schema preamble
2. Calls `build_remap(schema, codecs)` — validates layout, builds ID mapping
3. Each `SerializedMutation`'s `ComponentId` fields are remapped before applying
4. `Snapshot::restore_world` uses same `build_remap` for snapshot schema entries

## Backwards Compatibility

Old WAL files without a schema preamble: the first record parses as `WalEntry::Mutations` (no schema). Replay works as today — same-process, no remapping. The absence of schema means "trust that IDs match."

## Scope

**In scope:**
- `CodecRegistry::register_as` and `resolve_name`
- `by_name` HashMap on `CodecRegistry`
- `WalEntry`, `WalSchema`, `WalComponentDef` types
- Schema preamble on WAL create/open
- `build_remap` shared function
- Remap applied during WAL replay and snapshot load
- Size/align validation with hard errors
- Backwards compatibility for old WALs
- Update `persist` example to use `register_as`

**Not in scope:**
- Entity ID partitioning/authority (replication design)
- Schema evolution / field migration
- Replication transport protocol
- WAL retention policy (replication design — noted as dependency: replicas need WAL entries kept until acknowledged)
- Changes to `ComponentRegistry`, `World`, or `ComponentId` type
