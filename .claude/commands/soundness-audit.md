---
description: Audit a design or implementation for soundness holes before finalizing
args:
  - name: target
    description: Feature name, file path, or branch diff to audit
    required: true
allowed-tools: Bash, Read, Glob, Grep
---

Run a soundness audit on: $ARGUMENTS

This catches the class of bugs that compile and pass tests but corrupt state under concurrent load or edge cases. Follow these steps:

1. **Identify the scope**: Read the relevant code. If a branch name or PR, use `git diff main...HEAD` to find all changes.

2. **API existence check**: grep for every external API, method, or type the code depends on. List each one with its actual signature. Flag any that are assumed but don't exist.

3. **Mutable aliasing audit**: For every path that obtains `&mut T`:
   - Can two references to the same data exist simultaneously?
   - Is `&mut World` ever handed to code that could alias it?
   - Are `ReadOnlyWorldQuery` bounds enforced on all `&World` query paths?

4. **Semantic review checklist** (from CLAUDE.md):
   1. Can this be called with the wrong World?
   2. Can Drop observe inconsistent state?
   3. Can two threads reach this through `&self`?
   4. Does dedup/merge/collapse preserve the strongest invariant?
   5. What happens if this is abandoned halfway through?
   6. Can a type bound be violated by a legal generic instantiation?
   7. Does the API surface of this handle permit any operation not covered by the Access bitset?

5. **Bypass-path check**: Does any new code path skip the normal pipeline? If so, verify:
   - Change detection ticks are maintained
   - Query cache invalidation still works
   - Access bitsets accurately reflect actual access
   - Entity lifecycle tracking is preserved

6. **Assert boundary check**: For every assert/debug_assert in the diff:
   - If violating it would make the scheduler's Access bitset disagree with reality → must be `assert!`
   - If it's within an already-correct access boundary → `debug_assert!` is fine

7. **Report**: List findings as CRITICAL (unsound), IMPORTANT (correctness risk), or NOTE (style/defense-in-depth). Include file:line references.
