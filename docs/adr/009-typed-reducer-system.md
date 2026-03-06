# ADR-009: Typed Reducer System

**Status:** Accepted
**Date:** 2026-03-04

## Context

Raw transaction closures accept `&mut World` or `(&Tx, &World)` but provide no compile-time proof of what they access. A scheduler cannot determine conflict freedom without runtime metadata. The system needs typed handles that make the access contract visible in the function signature.

## Decision

Three execution models with typed handles: Transactional (`EntityMut<C>`, `Spawner<B>`, `QueryWriter<Q>` — buffered writes via EnumChangeSet), Scheduled (`QueryMut<Q>`, `QueryRef<Q>` — direct `&mut World`), and Dynamic (`DynamicCtx` — runtime-validated access with builder-declared upper bounds). `ReducerRegistry` type-erases closures with `Access` metadata and pre-resolved `ComponentId`s. `Contains<T, INDEX>` uses a const generic to solve coherence conflicts in tuple impls.

**Key insight: typed handles hide World behind a facade exposing exactly the declared operations — the type signature IS the access contract.**

## Alternatives Considered

- Raw `&mut World` in all closures — unsound for concurrent use, no conflict analysis possible
- Capability tokens without handles (pass ComponentId sets, check at call sites) — verbose, error-prone, no compile-time guarantees
- Single execution model for all reducer types — over-constrains either the buffered or direct mutation path

## Consequences

- Type signatures prove conflict freedom: a scheduler can determine parallelism from `Access` metadata alone
- Registry enables both compile-time scheduling (via `access()`) and runtime dispatch by name (via `id_by_name()`)
- Handles hold `&mut EnumChangeSet` not `&mut Tx` — clean borrow splitting avoids lifetime entanglement
- `QueryWriter` must capture its tick after commit (not during read) to support `Changed<T>` correctly
- `EntityAllocator::reserve(&self)` uses `AtomicU32` for lock-free entity ID allocation inside transactional closures
- Dynamic reducers use `assert!` (not `debug_assert!`) for access boundary checks — the scheduler trusts these bounds in release builds
