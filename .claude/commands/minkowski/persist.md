---
description: Help with Minkowski ECS persistence — WAL, snapshots, Durable strategy, codec registration, recovery
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user implement persistence for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing persistence patterns:

- `minkowski_persist` imports (`Wal`, `Snapshot`, `CodecRegistry`, `Durable`)
- `CodecRegistry` construction and `register::<T>()` calls
- `Durable::new()` construction wrapping a transaction strategy
- `Snapshot::save()` / `Snapshot::load()` calls
- `Wal::open()` / `Wal::replay()` calls
- `sync_reserved()` calls after snapshot restore
- Components with `Serialize`/`Deserialize` derives
- Which components need to survive restarts vs which are transient

## Step 2: Recommend

**Strong defaults:**
- **WAL for crash safety**: `Durable<S, W>` wraps any transaction strategy. On successful commit, the forward changeset is written to WAL before being applied to World. Failed attempts (retries) are not logged.
- **Snapshots for fast restore**: Save periodically (not every frame). On recovery: load snapshot, then replay WAL entries since the snapshot.
- **Register codecs for every persistent component**: `codec_registry.register::<Pos>()` for each component type that needs serialization. Components without codecs are silently skipped in WAL/snapshot serialization.
- **`sync_reserved()` after snapshot restore**: Prevents entity ID overlap. Without this, `EntityAllocator::reserve()` hands out indices starting from 0, which overlap with restored entities.
- **Use `QueryWriter` for durable reducers**: `QueryWriter` buffers writes into `EnumChangeSet`, which is what `Durable` logs to the WAL. `QueryMut` writes directly and cannot be WAL-logged.

**Recovery sequence:**
1. Load snapshot: `Snapshot::load(&mut world, &codec_registry, path)`
2. Sync entity allocator: `world.sync_reserved()` (or equivalent)
3. Open WAL: `Wal::open(path)`
4. Replay entries: `wal.replay(&mut world, &codec_registry)`
5. World is now recovered to the last committed transaction

**Ask if unclear:**
- "What's your recovery time budget?" — Determines snapshot frequency. More frequent snapshots = faster recovery (fewer WAL entries to replay) but more I/O during normal operation.
- "Which components need to survive restarts?" — Only register codecs for those. Transient components (e.g., render state, debug labels) can be omitted.
- "Do you need point-in-time recovery or just crash recovery?" — WAL + snapshots give crash recovery. For point-in-time, keep old snapshots and truncate WAL selectively.

## Step 3: Implement

Help write persistence code. Point to relevant examples:

- **Complete persistence flow**: See `examples/examples/persist.rs` — WAL + snapshots + recovery with QueryWriter reducer
- **Durable strategy construction**: `Durable::new(Optimistic::new(&world), wal, codec_registry)`
- **Snapshot save/load**: `Snapshot::save(&world, &codec_registry, path)` / `Snapshot::load(&mut world, &codec_registry, path)`
- **Component codec registration**: Components must derive `Serialize + Deserialize`. Register: `codecs.register::<Pos>(); codecs.register::<Vel>();`

**Pitfall alerts:**
- Forgetting `sync_reserved()` after snapshot restore causes entity ID overlap — new `reserve()` calls hand out indices that collide with restored entities.
- WAL write failure panics — the durability invariant is non-negotiable. Ensure the WAL path is writable.
- `QueryMut` for durable reducers is wrong — direct mutations bypass the changeset that `Durable` logs. Use `QueryWriter` instead.

For architecture details, see CLAUDE.md § "Transaction Semantics" (Durable tier) and the `minkowski-persist` crate.
