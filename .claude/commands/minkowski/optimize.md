---
description: Help with Minkowski ECS performance optimization — SIMD alignment, vectorization, iteration strategies, migration costs
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user optimize performance in their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for performance-relevant patterns:

- `#[repr(align(16))]` or `[f32; 4]` component types (SIMD alignment)
- `for_each` vs `for_each_chunk` vs `par_for_each` usage
- `world.insert()` / `world.remove()` in loops (archetype migration cost)
- `world.spawn()` call frequency and batch sizes
- Number of distinct archetypes (many archetypes = more query matching work)
- Component sizes and layouts (oversized components hurt cache performance)
- `Changed<T>` usage for skipping unchanged data
- Hot loops that could benefit from vectorization
- Per-entity `world.get()` in loops (candidate for `get_batch`)
- `BTreeIndex` / `HashIndex` for narrowing before component fetch

## Step 2: Recommend

**Strong defaults:**
- **`#[repr(align(16))]` for SIMD-friendly components.** Or use naturally aligned types like `[f32; 4]`. BlobVec columns are 64-byte aligned (cache line), but component layout determines whether LLVM can pack SIMD operations.
- **`for_each_chunk` for numeric tight loops.** Yields typed `&[T]`/`&mut [T]` slices per archetype — LLVM auto-vectorizes loops over these contiguous slices. This is the fastest iteration path for numeric data.
- **`par_for_each` for CPU-heavy per-entity work.** Uses rayon. Only worth it when per-entity work is expensive enough to amortize thread overhead (roughly >1us per entity).
- **Build with `-C target-cpu=native`** (already configured in `.cargo/config.toml`) to enable platform-specific SIMD instructions (AVX2, etc.).
- **Minimize archetype migrations.** `world.insert()` and `world.remove()` move entities between archetypes — O(components) per entity. Design archetypes upfront. If you need a "has/doesn't have" flag, use a `bool` component spawned with the entity rather than adding/removing a marker component.
- **Query cache is automatic.** Repeat `world.query()` calls skip the archetype scan when no new archetypes have been created. No manual cache management needed.
- **Keep components small.** Large components (>64 bytes) hurt cache utilization during iteration. Split rarely-accessed fields into separate components.
- **Use `Changed<T>` to skip unchanged archetypes.** If only a fraction of entities change each frame, `Changed<T>` skips entire archetype columns that haven't been mutably accessed.

**Ask if unclear:**
- "Where's the bottleneck? Iteration, migration, or matching?" — Profile first. Iteration: optimize component layout + use `for_each_chunk`. Migration: reduce `insert/remove` calls. Matching: reduce archetype count or use Table queries.
- "How many archetypes do you have?" — Many archetypes (>100) means more work during query matching. Consider if you can consolidate component sets.
- "What's the entity count?" — Under 10K: micro-optimizations rarely matter. 10K-1M: alignment and vectorization pay off. 1M+: everything matters, consider parallel iteration.

## Step 3: Implement

Help write optimized code. Point to relevant examples:

- **SIMD-aligned components**:
  ```
  #[repr(C, align(16))]
  #[derive(Clone, Copy)]
  struct Vec4(f32, f32, f32, f32);
  ```
- **Chunk iteration for vectorization**: See `examples/examples/boids.rs` — `for_each_chunk` with aligned position/velocity
- **Parallel iteration**: `world.query::<(&Pos, &Mass)>().par_for_each(|(pos, mass)| { /* heavy */ })`
- **Batch spawning**: Spawn all entities in one loop at startup rather than scattered through the frame
- **Changed filter**: `world.query::<(&mut Render, Changed<Transform>)>().for_each(...)` — skips archetypes where Transform column hasn't been touched
- **Index-driven access**: Use `BTreeIndex`/`HashIndex` to narrow the entity set, then `get_batch` instead of per-entity `get()`:
  ```
  let candidates = score_index.range(Score(90)..).flat_map(|(_, e)| e).copied().collect::<Vec<_>>();
  let positions = world.get_batch::<Pos>(&candidates);
  ```
  `get_batch` groups by archetype internally — one ComponentId resolution, sequential memory access per archetype. Worth it for ~10+ entities.
- **Benchmarking**: `cargo bench -p minkowski` runs criterion benchmarks. `cargo bench -p minkowski -- spawn` for a specific benchmark.

**Pitfall alerts:**
- `world.insert()` in a hot loop: Each call is an archetype migration. If adding the same component to many entities, consider redesigning the archetype.
- `for_each` where `for_each_chunk` would work: You lose SIMD vectorization. Use `for_each_chunk` for numeric tight loops.
- Oversized components in hot queries: If a 256-byte component is queried every frame but only 8 bytes are read, split it.
- Per-entity `world.get()` in a loop over index results: Each call independently resolves ComponentId and jumps to a random archetype. Use `get_batch` to amortize the resolution and group by archetype for cache locality.

For architecture details, see CLAUDE.md § "Column Alignment & Vectorization" and § "Query Caching".
