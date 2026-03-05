---
description: Create a PR with full test/lint validation
allowed-tools: Bash, Read, Glob, Grep
---

Follow these steps to create a pull request:

1. Run `cargo fmt --all -- --check` — fix any formatting issues
2. Run `cargo clippy --workspace --all-targets -- -D warnings` — fix any lint issues
3. Run `cargo test -p minkowski` — all tests must pass
4. Run `git diff main...HEAD` to understand the full scope of changes
5. Run `git log main..HEAD --oneline` to see all commits
6. Summarize the changes: what was added/changed/fixed and why
7. Create the PR with `gh pr create` using a concise title and structured body with Summary and Test Plan sections
8. Return the PR URL
