# ADR-007: Split-Phase Transactions

**Status:** Accepted
**Date:** 2026-03-02

## Context

Concurrent access to ECS data requires isolation and conflict detection. A naive approach of holding `&mut World` in the transaction handle prevents parallel reads entirely. The system must support multiple transactions reading concurrently while maintaining soundness — no aliased `&mut T`.

## Decision

`Tx` does not hold `&mut World`. Methods take world as a parameter, splitting execution into three phases: begin (`&mut World`, sequential) -> execute (`&World`, parallel) -> commit (`&mut World`, sequential). `tx.query(&world)` is bounded by `ReadOnlyWorldQuery`, preventing `&mut T` through a shared reference. Three strategies: `Sequential` (zero-cost passthrough), `Optimistic` (tick-based validation, 3 retries), `Pessimistic` (cooperative column locks, 64 retries with spin+yield). Entity lifecycle is closed through `OrphanQueue` shared via `Arc<Mutex<Vec<Entity>>>`.

**Key insight: the `ReadOnlyWorldQuery` bound on `tx.query(&world)` is the soundness linchpin — without it, two transactions could obtain aliased `&mut T` from the same `&World`.**

## Alternatives Considered

- MVCC with version chains — too complex for ECS workloads, high memory overhead
- Lock-per-entity granularity — too fine-grained, lock table size proportional to entity count
- `&mut World` held by Tx — prevents parallel reads entirely, serializes all transaction execution

## Consequences

- Safe parallel execution: multiple transactions can read concurrently during the execute phase
- Zero-cost sequential path: `Sequential` strategy delegates directly to World with no buffering
- Entity IDs never leak: `OrphanQueue` drains automatically at the top of every `&mut World` method
- `WorldId` prevents cross-world corruption (strategy from world A used with world B)
- Lock privilege can only escalate, never downgrade — `dedup_by` keeps highest privilege, not first-seen
- Drop is the abort path: cleanup must be reachable from `&self` via `Arc`-shared handles
