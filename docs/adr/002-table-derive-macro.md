# ADR-002: Table Derive Macro

**Status:** Accepted
**Date:** 2026-03-01

## Context

Dynamic queries match archetypes via bitset checks on every call. For schemas that are known at compile time and always queried together, this repeated matching is unnecessary overhead. Users also want named field access (e.g., `row.position`) rather than positional tuple destructuring.

## Decision

`#[derive(Table)]` generates a pre-registered archetype with cached column offsets. The generated code produces `Ref<'w>` and `Mut<'w>` associated types for typed row access with named fields. `query_table` and `query_table_mut` bypass archetype matching entirely, going straight to the cached archetype's column pointers.

**Key insight: two-tier access — static table queries compile to direct pointer arithmetic, dynamic queries handle arbitrary combinations.**

## Alternatives Considered

- Manual schema registration with explicit column indices — verbose, error-prone, no named field access
- Query specialization via compiler monomorphization — dependent on compiler optimizations, not guaranteed
- Code generation without a derive macro (build script) — harder to maintain, no access to type information

## Consequences

- Zero-overhead typed row access for known schemas — `query_table` skips the `HashMap<TypeId, QueryCacheEntry>` lookup entirely
- Dynamic queries remain fully available for ad-hoc component combinations
- Derive macro must generate `pub` code visible from external crates — `pub(crate)` in generated code silently breaks downstream users
- `TableDescriptor` caches `archetype_id` and field-to-column index mapping, allocated once at registration
- The `extern crate self as minkowski;` workaround is required for in-crate tests where generated code references `::minkowski::*`
