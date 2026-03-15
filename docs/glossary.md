# Glossary

| Term | Definition |
|---|---|
| [Archetype][archetype] | A unique combination of component types. Entities with the same components share an archetype, enabling contiguous column storage. |
| [Barnes-Hut][barnes-hut] | An O(N log N) approximation algorithm for N-body force computation using a quadtree to aggregate distant particles. |
| [Bitset][bitset] | A compact array of bits used here for fast archetype matching — query matching is a bitwise subset check. |
| [Column store][column-store] | A storage layout where each field is stored in its own contiguous array, enabling cache-friendly sequential access and SIMD vectorization. |
| [Component][component] | A data type attached to an entity. Any `'static + Send + Sync` Rust type qualifies. Stored in BlobVec columns within archetypes. |
| [ECS][ecs] | Entity-Component System — an architectural pattern where entities are IDs, components are data, and systems are logic. Decouples data layout from behavior. |
| [Generational index][generational-index] | An ID scheme pairing an array index with a generation counter. Reusing an index bumps the generation, so stale handles are detected without a free-list scan. |
| [Miri][miri] | An interpreter for Rust's Mid-level IR that detects undefined behavior (aliasing violations, use-after-free, data races) at runtime. |
| [OCC][occ] | Optimistic concurrency control — transactions execute without locks, then validate at commit that no conflicting writes occurred. |
| [PCC][pcc] | Pessimistic concurrency control — transactions acquire locks before accessing data, preventing conflicts at the cost of potential contention. |
| [Query planner](queries.md#query-planner) | A compiled push-based optimizer that compiles queries into execution plans with automatic index selection. Based on the [Neumann compilation model][push-compiled]. |
| [Rayon][rayon] | A Rust library for data parallelism. Used here for `par_for_each` parallel query iteration. |
| [Reducer](../README.md#typed-reducers) | A registered closure whose type signature declares its data access. The registry extracts conflict metadata at registration time. |
| [SIMD][simd] | Single Instruction, Multiple Data — CPU instructions that process multiple values in parallel. Minkowski's 64-byte column alignment enables auto-vectorization. |
| [SoA][soa] | Struct of Arrays — storing each field in a separate array rather than interleaving fields per record. The storage layout archetypes use. |
| [Tree Borrows][tree-borrows] | An experimental Rust aliasing model (stricter than Stacked Borrows) that Miri can check. Minkowski passes under this model. |
| [Uniform grid][uniform-grid] | A spatial index dividing space into fixed-size cells. O(1) cell lookup, O(k) neighbor iteration where k is the number of occupied neighbor cells. |
| [WAL][wal] | Write-ahead log — an append-only file where every mutation is recorded before being applied. Enables crash recovery by replaying the log. |

<!-- Link definitions -->
[archetype]: https://ajmmertens.medium.com/building-an-ecs-2-archetypes-and-vectorization-fe21690f6d51
[barnes-hut]: https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
[rkyv]: https://github.com/rkyv/rkyv
[bitset]: https://en.wikipedia.org/wiki/Bit_array
[column-store]: https://en.wikipedia.org/wiki/Column-oriented_DBMS
[component]: https://en.wikipedia.org/wiki/Entity_component_system#Components
[ecs]: https://en.wikipedia.org/wiki/Entity_component_system
[generational-index]: https://lucassardois.medium.com/generational-indices-guide-8e3c5f7fd594
[miri]: https://github.com/rust-lang/miri
[occ]: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
[pcc]: https://en.wikipedia.org/wiki/Lock_(database)
[rayon]: https://github.com/rayon-rs/rayon
[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[soa]: https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays
[tree-borrows]: https://perso.crans.org/vanille/treebor/
[uniform-grid]: https://en.wikipedia.org/wiki/Grid_(spatial_index)
[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
[tsan]: https://clang.llvm.org/docs/ThreadSanitizer.html
[loom]: https://github.com/tokio-rs/loom
[push-compiled]: https://www.vldb.org/pvldb/vol4/p539-neumann.pdf
[wal]: https://en.wikipedia.org/wiki/Write-ahead_logging
