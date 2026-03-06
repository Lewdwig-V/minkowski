# ADR-010: AI-Assisted Developer Tooling

**Status:** Accepted
**Date:** 2026-03-05

## Context

Minkowski combines ECS patterns with database semantics — an unusual combination. Users face non-obvious decisions: which reducer type to use, which concurrency model fits their workload, how to optimize query iteration for SIMD. Traditional documentation describes what exists but does not guide decisions in context.

## Decision

One auto-triggering skill (`minkowski-guide.md`) provides passive expertise as users write code, activating on relevant keywords. Eight domain-specific slash commands guide paradigm decisions: data modeling, query patterns, reducer selection, concurrency model, mutation strategy, persistence setup, spatial indexing, and performance optimization. Five additional utility commands handle design docs, soundness audits, API validation, macro validation, and PR creation. Skills reference `CLAUDE.md` architecture but teach the paradigm through decision flowcharts and anti-pattern warnings.

**Key insight: meet users where they are — skills auto-trigger on relevant keywords, commands provide Socratic guidance for key decisions.**

## Alternatives Considered

- Traditional documentation only (rustdoc, markdown guides) — not interactive, cannot read the user's code context
- Built-in help system (CLI flags, error messages) — not adaptive, cannot assess trade-offs for the user's specific situation
- Tutorial-style walkthroughs — static, cannot adjust to the user's existing codebase or goals

## Consequences

- Claude Code users get contextual guidance that adapts to what they are currently working on
- Skill content must stay synchronized with the evolving API — stale guidance is worse than none
- Commands follow an assess-then-recommend-then-implement pattern, not just information dumps
- The tooling layer is entirely external to the engine — no runtime cost, no API surface growth
- Users without Claude Code are unaffected — the skills directory has no impact on compilation or functionality
