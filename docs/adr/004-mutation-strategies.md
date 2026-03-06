# ADR-004: Mutation Strategies

**Status:** Accepted
**Date:** 2026-02-28

## Context

Two distinct mutation patterns arise in practice: deferred structural changes during query iteration (spawn/despawn/insert/remove), and data-driven mutations that need serialization for persistence and reversibility for undo/redo. A single mechanism cannot serve both without over-constraining one use case.

## Decision

Two complementary systems: `CommandBuffer` stores `Vec<Box<dyn FnOnce(&mut World) + Send>>` for deferred structural changes during iteration. `EnumChangeSet` records mutations as a `Vec<Mutation>` enum with component bytes in a contiguous `Arena`, where `apply()` returns a reverse changeset automatically.

**Key insight: EnumChangeSet's `apply()` returns a reverse changeset automatically — undo is just `reverse.apply(&mut world)`.**

## Alternatives Considered

- Single unified mutation system — over-constrains either the deferred-closure or the serializable-data use case
- Immediate mutation during iteration — unsound, invalidates iterators
- Event sourcing only (log all mutations as events) — high overhead for simple deferred spawns

## Consequences

- `CommandBuffer` is the simple path for iteration-time structural changes — closures capture context naturally
- `EnumChangeSet` enables WAL serialization (persistence), undo/redo (Game of Life example), and transaction buffering
- Typed safe helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) auto-register component types and handle `ManuallyDrop`
- Two systems to learn, but each is optimized for its use case with no impedance mismatch
- `EnumChangeSet` is the foundation for both the transaction system and the persistence layer
