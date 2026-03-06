---
description: Audit implementation for mutation path gaps, visibility issues, and edge cases
allowed-tools: Bash, Read, Glob, Grep
---

Run a self-audit on the current branch's implementation changes. This is complementary to `/soundness-audit` (which covers design soundness, mutable aliasing, and semantic review). This command focuses on implementation gaps that compile and pass tests but break under real usage.

## Steps

1. **Scope the diff**: Run `git diff main...HEAD --name-only` to find all changed files. Read each changed file.

2. **Mutation path completeness**: For every new or modified function that hands out `&mut T`, writes through raw pointers, or calls `BlobVec` methods:
   - Verify it goes through `get_ptr_mut(row, tick)` or a World method that marks columns changed
   - Check it does NOT use `get_ptr` (read path) for writes — this silently bypasses `Changed<T>`
   - Enumerate ALL mutation paths that reach the new code: spawn, get_mut, insert, remove, query `&mut T`, query_table_mut, query_table_raw, changeset apply
   - Flag any path that is missing change detection

3. **Visibility correctness**: For every new `pub` item:
   - Check that its signature does not reference `pub(crate)` types
   - If derive macro output is involved, verify it would compile from `minkowski-examples` (external crate)
   - Flag any generated code that assumes in-crate access

4. **Edge cases**: For every new query path, iterator, or data structure:
   - Same-tick interleaving: can two mutations in the same tick both be visible to a subsequent query?
   - Empty archetypes: does the code handle archetypes with zero entities?
   - Despawn-during-iteration: are structural changes properly deferred via CommandBuffer or EnumChangeSet?
   - Component removal: do indexes, caches, or lookups handle entities that had a component removed?

5. **Report**: List findings with severity and file:line references:
   - ISSUE: must fix before merging (missed change detection, visibility bug, unsound edge case)
   - NOTE: worth checking but may be intentional (unusual pattern, potential edge case)

If no issues are found, say so explicitly.
