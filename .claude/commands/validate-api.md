---
description: Sketch API designs in a scratch file and validate soundness via cargo check
args:
  - name: feature
    description: The feature or API surface to design and validate
    required: true
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Validate API design soundness for: $ARGUMENTS

This workflow uses the Rust compiler as an automated design reviewer. Follow these steps:

1. **Understand the context**: Search the codebase for related types, traits, and patterns that the new API will interact with.

2. **Create scratch file**: Create `src/scratch_api.rs` (or in the appropriate crate's src/) with:
   - Necessary imports from the crate
   - Add `mod scratch_api;` to the crate's lib.rs (temporarily)

3. **Sketch 2-3 alternative API designs** in the scratch file:
   - For each design, write: type signatures, trait impls, and a usage example function
   - Label each design clearly with comments (// Design A: ..., // Design B: ...)
   - Focus on the borrow checker implications — lifetimes, mutability, Send/Sync bounds

4. **Iterate with cargo check**: Run `cargo check -p <crate> 2>&1` after each design sketch.
   - If it compiles: document WHY it's sound (what invariants does the type system enforce?)
   - If it fails: analyze the error. Is this a fundamental soundness issue or a fixable syntax problem?
   - Iterate each design until it either compiles or you can explain why it's fundamentally unsound

5. **Semantic review**: For each compiling design, answer:
   - Can this be called with the wrong World?
   - Can two threads reach this through `&self`?
   - What happens if this is abandoned halfway through?
   - Does the type system actually prevent the misuse, or does it just happen to compile?

6. **Report findings**: Summarize tradeoffs of each approach. Recommend which to use in the design doc.

7. **Clean up**: Remove the scratch file and the `mod scratch_api;` line from lib.rs. Do NOT commit scratch files.

IMPORTANT: The goal is to find designs that are REJECTED by the compiler when misused. A design that compiles when used correctly is necessary but not sufficient — it must also FAIL to compile when used incorrectly (e.g., trying to get &mut T through &World).
