---
description: Validate derive macro output compiles from an external crate
args:
  - name: macro_name
    description: The derive macro to validate (e.g., Table)
    required: true
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Validate that #[derive($ARGUMENTS)] generates code visible from external crates.

Derive macros generate code that references crate internals. In-crate tests can't catch `pub(crate)` leaking into generated code — only external consumers can. This command validates from the examples crate.

1. **Find existing usage**: Search `examples/examples/` for existing uses of `#[derive($ARGUMENTS)]`. If found, run `cargo check -p minkowski-examples` to verify they still compile.

2. **Create test case** (if no existing usage covers the change):
   - Create a temporary file `examples/examples/macro_test.rs`
   - Add a minimal struct using `#[derive($ARGUMENTS)]` with representative field types
   - Add a `fn main()` that exercises the generated API (constructors, accessors, query methods)
   - Add the example to `examples/Cargo.toml` if needed

3. **Compile from external perspective**:
   ```
   cargo check -p minkowski-examples 2>&1
   ```
   - If it compiles: the macro output is correctly visible
   - If it fails with visibility errors: list each inaccessible type/method and its location in the generated code

4. **Check generated code** (if errors found):
   - Use `cargo expand` (if available) or reason about macro output
   - For each visibility error: trace whether the type should be `pub` or if the macro is incorrectly referencing a `pub(crate)` internal

5. **Clean up**: Remove any temporary test files. Do NOT commit them.

6. **Report**: Pass/fail with specific visibility issues found and recommended fixes.
