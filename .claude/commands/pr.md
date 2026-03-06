---
description: Create a PR with full test/lint validation
allowed-tools: Bash, Read, Glob, Grep
---

Follow these steps to create a pull request:

1. **Self-audit**: Run the self-audit checklist (mutation path completeness, visibility correctness, edge cases) on all changed files. Report any ISSUE or NOTE findings to the user. Ask whether to fix the issues or proceed with the PR as-is. Wait for the user's response before continuing.
2. Run `cargo fmt --all -- --check` — fix any formatting issues
3. Run `cargo clippy --workspace --all-targets -- -D warnings` — fix any lint issues
4. Run `cargo test -p minkowski` — all tests must pass
5. Run `git diff main...HEAD` to understand the full scope of changes
6. Run `git log main..HEAD --oneline` to see all commits
7. Summarize the changes: what was added/changed/fixed and why
8. Create the PR with `gh pr create` using a concise title and structured body with Summary and Test Plan sections
9. Return the PR URL
