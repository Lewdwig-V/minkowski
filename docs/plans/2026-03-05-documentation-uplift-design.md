# Documentation Uplift Design

**Goal:** Comprehensive documentation rewrite for hackathon submission — README showcase, ADRs for delivered decisions, tiered rustdocs for the public API.

**Audience:** Technical evaluators + potential users. Lead with novelty and scale, go deep on design, close with AI-assisted development story.

---

## 1. README.md Rewrite

Replace the current architecture-doc-style README with a hackathon showcase:

1. **Hero** — one-paragraph hook: column-oriented database engine built from scratch, typed reducers, transactions, persistence, AI tooling.
2. **What makes this interesting** — 4-5 bullets on novel decisions.
3. **Quick start** — ~30 lines: spawn, query, reducer, transaction.
4. **Feature tour** — concise subsections (Storage, Queries, Reducers, Transactions, Persistence, Spatial Indexing, Change Detection). 3-5 sentences + code snippet each.
5. **Examples** — table of 8 examples with one-line descriptions and run commands.
6. **AI-Assisted Development** — skills, commands, the hackathon differentiator.
7. **ADRs** — link to `docs/adr/`.
8. **Building & Testing** — essential commands.
9. **Roadmap** — stretch goals table (query planning, B-tree/hash indexes, rkyv, replication).
10. **License**

**Cut:** Phase 5 roadmap promises, build roadmap checkboxes, serialization table, detailed architecture section (moved to rustdocs).

## 2. ADRs

**Location:** `docs/adr/`, numbered `001-` through `010-`.

**Format:**
```
# ADR-NNN: Title
Status: Accepted | Date: YYYY-MM-DD
## Context — 2-3 sentences
## Decision — what + why, key insight in bold
## Alternatives Considered — bullet list
## Consequences — enables, constrains, trade-offs
```

| # | Title | Key decision |
|---|---|---|
| 001 | Column-Oriented Archetype Storage | BlobVec + archetype = database columns with ECS flexibility |
| 002 | Table Derive Macro | Compile-time schema with typed row access bypassing archetype matching |
| 003 | Change Detection via Tick Tracking | Per-column ticks, archetype-level skip, no manual tick call |
| 004 | Mutation Strategies | CommandBuffer for deferred, EnumChangeSet for data-driven + undo |
| 005 | Spatial Index Trait | External composition, not World integration |
| 006 | Query Conflict Detection | Access bitsets as scheduling primitive, policy left to framework |
| 007 | Split-Phase Transactions | Tx doesn't hold &mut World; ReadOnlyWorldQuery prevents aliased &mut T |
| 008 | WAL + Snapshot Persistence | Durable wrapper composes with any transaction strategy |
| 009 | Typed Reducer System | Three execution models, handles prove conflict freedom from signatures |
| 010 | AI-Assisted Developer Tooling | Auto-triggering skill + 8 commands make unfamiliar paradigm accessible |

Delete all 43 files in `docs/plans/`. Git history preserves them.

## 3. Rustdocs

**Crate-level (`lib.rs`):** Rich `//!` — what Minkowski is, mental model, "where to start" guide.

**Module-level:** `//!` paragraph per `pub mod`.

**Tier 1 — full treatment** (doc + example + cross-links):
- World, Entity, ReducerRegistry
- EntityRef, EntityMut, QueryRef, QueryMut, QueryWriter, Spawner
- DynamicCtx, DynamicReducerBuilder
- Transact, Tx, Sequential, Optimistic, Pessimistic
- EnumChangeSet, SpatialIndex, CommandBuffer

**Tier 2 — one-liner + link:**
- Access, ComponentId, Changed, ReadOnlyWorldQuery
- WritableRef, WriterQuery, ComponentSet, Contains
- Conflict, SequentialTx
- ReducerId, QueryReducerId, DynamicReducerId

**Tier 3 — skip:** MutationRef, ComponentRegistry (doc-hidden).

**Fix 4 existing warnings** (private item links, unclosed HTML tag).

## 4. Execution

Three parallel subagents in worktrees (README, ADRs, Rustdocs). Cherry-pick into single PR. Verify: `cargo doc --no-deps`, `cargo test`, `cargo clippy`.
