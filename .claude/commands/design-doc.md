---
description: Create a structured design doc for a new feature
args:
  - name: topic
    description: Feature or system being designed
    required: true
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Create a design document for: $ARGUMENTS

Follow these steps:

1. Search the codebase to understand existing related code and patterns
2. **Verify-before-design**: grep for any APIs, methods, or types that the design depends on. Confirm what exists vs what needs to be created. List findings explicitly.
3. Create the doc at `docs/plans/<topic>.md` with this structure:

```
# <Topic> Design

## Problem
What problem does this solve? Why now?

## Current State
What exists today that's relevant? (with file paths)

## Proposed Design
### API Surface
### Internal Architecture
### Data Flow

## Alternatives Considered
At least 2 alternatives with tradeoffs

## Semantic Review
Answer the 7 checklist questions from CLAUDE.md Key Conventions:
1. Can this be called with the wrong World?
2. Can Drop observe inconsistent state?
3. Can two threads reach this through `&self`?
4. Does dedup/merge/collapse preserve the strongest invariant?
5. What happens if this is abandoned halfway through?
6. Can a type bound be violated by a legal generic instantiation?
7. Does the API surface of this handle permit any operation not covered by the Access bitset?

## Implementation Plan
Ordered steps with file paths
```

4. Commit the design doc
