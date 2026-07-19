---
description: Audit a design or implementation for soundness holes before finalizing
args:
  - name: target
    description: Feature name, file path, or branch diff to audit
    required: true
allowed-tools: Bash, Read, Glob, Grep, Agent
---

Run a soundness audit on: $ARGUMENTS

This catches the class of bugs that compile and pass tests but corrupt state under concurrent load or edge cases. It is also the gate that **confirms and enforces** three TigerStyle concepts the project commits to:

- **Go slow to go fast** — investing extra time in design and formal protocols avoids years of costly rework in production.
- **Constrained & explicit** — allocate memory at startup, add zero dependencies, and use explicit limits.
- **Time Travel** — a time-machine approach: build, break, and converge on bug-proof systems.

The audit runs in four phases. Phase 1 scopes the work, Phase 2 is the orchestrator's own soundness review (the original checklist, retained as the team's baseline), Phase 3 dispatches **three TigerStyle enforcement agents in parallel**, and Phase 4 synthesizes everything into a single PASS/FAIL gate.

## Phase 1 — Scope

1. **Identify the scope**: Read the relevant code. If a branch name or PR, use `git diff main...HEAD` to find all changes (`--name-only` for the file list, full diff for content). If a file path or feature name, read those files and their callers.
2. Record whether this is a diff-scoped audit (feature branch) or a whole-target audit (named file/module). Pass the resolved file list and the full diff text to every agent in Phase 3 — they should not re-derive scope.

## Phase 2 — Core Soundness Review (orchestrator)

Run these checks yourself before dispatching the team. These are the non-delegable soundness invariants.

1. **API existence check**: grep for every external API, method, or type the code depends on. List each one with its actual signature. Flag any that are assumed but don't exist.

2. **Mutable aliasing audit**: For every path that obtains `&mut T`:
   - Can two references to the same data exist simultaneously?
   - Is `&mut World` ever handed to code that could alias it?
   - Are `ReadOnlyWorldQuery` bounds enforced on all `&World` query paths?

3. **Semantic review checklist** (from AGENTS.md):
   1. Can this be called with the wrong World?
   2. Can Drop observe inconsistent state?
   3. Can two threads reach this through `&self`?
   4. Does dedup/merge/collapse preserve the strongest invariant?
   5. What happens if this is abandoned halfway through?
   6. Can a type bound be violated by a legal generic instantiation?
   7. Does the API surface of this handle permit any operation not covered by the Access bitset?

4. **Bypass-path check**: Does any new code path skip the normal pipeline? If so, verify change detection ticks are maintained, query cache invalidation still works, Access bitsets accurately reflect actual access, and entity lifecycle tracking is preserved.

5. **Assert boundary check**: For every assert/debug_assert in the diff:
   - If violating it would make the scheduler's Access bitset disagree with reality → must be `assert!`.
   - If it's within an already-correct access boundary → `debug_assert!` is fine.

## Phase 3 — TigerStyle Enforcement Team

Dispatch **all three agents in a single message** (parallel execution) using the Agent tool. Give each agent the resolved file list from Phase 1, the full diff text, and the relevant grounding below.

Every agent must return findings using this shared rating scheme so Phase 4 can gate on them:
- `VIOLATION-CRITICAL` — the concept is broken in a way that must block merge.
- `VIOLATION-IMPORTANT` — a real regression of the concept that should be fixed but is not strictly unsound.
- `NOTE` — defense-in-depth or style observation.
- `CONFIRMED` — the concept is upheld on a path that was checked (brief note, so the report shows positive coverage, not just absence).

Every finding includes `file:line` references and, for violations, a concrete remediation.

### Agent 1: go-slow-to-go-fast

Use the Agent tool with this prompt:

> You are enforcing the TigerStyle concept **"Go slow to go fast"** on a Minkowski ECS change: investing extra time in design and formal protocols up front avoids years of costly rework. Read these files: [SCOPED FILE LIST]. Here is the diff: [DIFF]. You are confirming that design rigor *preceded* the implementation, not reviewing performance.
>
> Check:
> 1. **Verify-before-design**: Every external API, method, or type the diff calls must actually exist. grep the codebase and list each dependency with its real signature. Flag any that are assumed but absent — this is the exact class of past bug (assuming `EntityAllocator::reserve()` had atomics when it didn't exist; proposing `&mut World` in reducer APIs). The type system does not catch "assumed this method exists" — only verification does.
> 2. **Semantic review applied**: For every new primitive that touches concurrency, entity lifecycle, or cross-system state, the 7-question checklist (wrong World? Drop sees inconsistent state? two threads via `&self`? dedup preserves strongest invariant? abandoned halfway? type bound violable by legal generic? handle permits access outside its Access bitset?) must be answerable. Flag any new primitive where a question has no clear answer in the code or comments.
> 3. **Invariant documentation**: Every new `unsafe` block must carry a `// SAFETY:` comment stating the discharged precondition. Every new `pub` handle/type that carries an invariant (an ID-as-proof, a lock-privilege rule, a Drop-as-abort contract) must document it. Flag undocumented `unsafe` and undocumented invariants — undocumented invariants are future rework.
> 4. **Formal-protocol coverage**: Any new code that is concurrent (reachable through `&self` from multiple threads), uses raw-pointer `unsafe`, or is a transaction/commit path must be reachable by at least one formal protocol: loom (`--features loom`, `loom_tests`), Miri (`cargo +nightly miri nextest`), TSan, or a fuzz target (`fuzz_world_ops`, `fuzz_reducers`, `fuzz_wal_replay`, `fuzz_lsm_recovery`). Flag new concurrent/unsafe paths with NO formal-protocol reaching them, and name which protocol should cover them.
> 5. **Design artifact**: For a substantial new feature, there should be a design doc under `docs/plans/` (or the change should reference one). Note its absence as a rework risk, not a hard violation.
>
> Rate each finding `VIOLATION-CRITICAL` (assumed-missing API; `unsafe` with no SAFETY rationale; new concurrency path with zero formal coverage), `VIOLATION-IMPORTANT`, `NOTE`, or `CONFIRMED`. Include `file:line` and a remediation for every violation.

### Agent 2: constrained-and-explicit

Use the Agent tool with this prompt:

> You are enforcing the TigerStyle concept **"Constrained & explicit"** on a Minkowski ECS change: allocate memory at startup, add zero dependencies, and use explicit limits. Read these files: [SCOPED FILE LIST]. Here is the diff: [DIFF].
>
> Check:
> 1. **Allocate at startup, not at runtime**: Flag heap allocation introduced on a per-tick or per-entity path or that grows unboundedly at runtime — `Vec::new()`/`Vec::push` growth, `Box::new`, `HashMap` inserts, `.collect()` inside `for_each`/`for_each_chunk`/`par_for_each`/reducer loops. The project pre-allocates: `WorldBuilder::memory_budget(bytes)` sizes and pre-faults the pool at startup, `EnumChangeSet::with_capacity` / `reserve` reserve up front, and `QueryWriter::for_each` pre-scans entity count and reserves (capped at `MAX_PREALLOC_MUTATIONS` = 64K). New code on a hot path should reserve once or reuse an arena, not grow incrementally.
> 2. **Zero dependencies**: Run `git diff main...HEAD -- '*/Cargo.toml'` (or diff the scoped `Cargo.toml`s). Flag ANY newly added dependency. The dependency set is curated and justified in AGENTS.md's dependency tables (crate, version, exact reason). A new dependency is `VIOLATION-CRITICAL` unless the diff also adds a justification row to that table AND the crate is small/audited. Prefer std, an existing dep, or a small in-tree implementation.
> 3. **Explicit limits**: Every buffer, queue, retry loop, and recursion must have a named, enforced upper bound. Flag: unbounded growth driven by user input; magic-number limits not extracted into named constants (the codebase uses `MAX_PREALLOC_MUTATIONS`, `TCACHE_CAPACITY`, `NUM_SIZE_CLASSES`, `COMPACTION_TRIGGER`, `MaxLevels`); recursion with no depth cap; retry loops with no `max_retries`.
> 4. **Explicit failure on exhaustion**: When a limit is hit, the code must return a typed `Result` so the *caller* decides whether to panic (error philosophy: the decision to panic lies with the user). Resource exhaustion uses `PoolExhausted` / `InsertError` / `TransactError`, never silent truncation, `.unwrap()` deep in the engine, or an unchecked `as` cast that wraps. Internal-invariant `assert!` (boundary-protecting) is fine and is covered by Phase 2.
>
> Rate each finding `VIOLATION-CRITICAL` (new dependency; unbounded runtime allocation or growth on a core path; missing limit on a user-driven input), `VIOLATION-IMPORTANT`, `NOTE`, or `CONFIRMED`. Include `file:line` and a remediation for every violation.

### Agent 3: time-travel

Use the Agent tool with this prompt:

> You are enforcing the TigerStyle concept **"Time Travel"** on a Minkowski ECS change: a time-machine approach to *build, break, and converge* on bug-proof systems. The system must be deterministic enough to replay, hammered hard enough to break, and convergent so that replaying the same inputs reconstructs the same state. Read these files: [SCOPED FILE LIST]. Here is the diff: [DIFF].
>
> Check:
> 1. **Build — determinism**: Any new mutation, reducer, or persistence code that affects committed state must be a deterministic function of its inputs. Per the reducer-determinism rule, flag any RNG, system time (`Instant`/`SystemTime`), `HashMap` *iteration* order, thread-id, or global mutable state on a state-affecting path. Suggest deterministic alternatives the codebase already uses: `BTreeMap`, seeded `SplitMix64`, `fastrand` with explicit seeds, or args-provided seeds. (The identity hasher `TypeIdHasher` keyed by `TypeId` is deterministic and fine.)
> 2. **Break — fuzz / exhaustive coverage**: New World operations, query/reducer iteration, WAL/snapshot/LSM codecs, or concurrency must be reachable by a breaking harness: `fuzz_world_ops`, `fuzz_reducers`, `fuzz_wal_replay`, `fuzz_lsm_recovery`, the loom `loom_tests`, Miri, or TSan. Flag new surface that no harness reaches, and name the specific target to extend or the new one to add.
> 3. **Converge — replay & recovery**: Any new *persisted* mutation or state transition must round-trip so that replaying the same op sequence converges to identical state. Verify: the component has a registered codec (`CodecRegistry::register`), `StorageKind` is correct (`RawCopy` when `size_of::<T>() == size_of::<T::Archived>()`, else `Serialized`), identity is resolved by **type name, never by numeric `ComponentId`** across worlds, and WAL replay remaps recorded ids via `build_remap`. Flag state that is mutated in memory but never captured by WAL/flush (it will silently diverge on recovery), and any recovery path that keys columns by `ComponentId` position instead of `type_name`.
> 4. **Reproducibility**: A seed or input that drives a simulation must be logged or derivable, so a failing run can be replayed exactly. Flag simulations whose driving seed is not recoverable.
>
> Rate each finding `VIOLATION-CRITICAL` (nondeterminism on a committed-state path; persisted mutation with no recovery coverage or divergent replay; identity keyed by numeric id across worlds), `VIOLATION-IMPORTANT`, `NOTE`, or `CONFIRMED`. Include `file:line` and a remediation for every violation.

## Phase 4 — Synthesis & Gate

After all three agents return, combine their findings with the Phase 2 core-soundness results and produce the report below. The gate is hard: **any `VIOLATION-CRITICAL` (from Phase 2 or any agent) = audit FAILS**. List the suggested verification commands (e.g. `cargo +nightly fuzz run fuzz_world_ops`, the loom/Miri/TSan commands from AGENTS.md) for any path an agent flagged as lacking formal/break coverage.

```
## Soundness Audit Report — <target>

### Verdict: PASS | FAIL
[FAIL if any VIOLATION-CRITICAL exists. One-line rationale.]

### Scope
[Files audited; diff-scoped or whole-target.]

### Core Soundness (Phase 2)
[Findings from the orchestrator's checklist: CRITICAL (unsound) / IMPORTANT (correctness risk) / NOTE, with file:line.]

### TigerStyle: Go slow to go fast
[Agent 1 findings, verbatim ratings + file:line.]

### TigerStyle: Constrained & explicit
[Agent 2 findings.]

### TigerStyle: Time Travel
[Agent 3 findings.]

### Required Before Merge
[Numbered list of every VIOLATION-CRITICAL with its remediation.]

### Suggested Verification Commands
[fuzz / loom / Miri / TSan commands covering any flagged gap.]
```

If no violations are found, say so explicitly and report `Verdict: PASS`.
