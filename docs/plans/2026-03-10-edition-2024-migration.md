# Edition 2024 Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the entire workspace from Rust edition 2021 to 2024, gaining `unsafe_op_in_unsafe_fn` enforcement and fixing `tail_expr_drop_order` changes.

**Architecture:** Mechanical migration — wrap unsafe operations in explicit `unsafe { }` blocks within `unsafe fn` bodies, fix 3 drop-order issues in transaction code, update all `Cargo.toml` edition fields. No behavioral changes.

**Tech Stack:** Rust 1.93+, `cargo fix --edition` for automated fixes where possible, manual annotation for quality `// SAFETY:` comments.

---

## Scope

| Crate | `unsafe_op_in_unsafe_fn` | `tail_expr_drop_order` | Total |
|---|---|---|---|
| `minkowski` | 106 (4 files) | 2 | 108 |
| `minkowski-persist` | 0 | 1 | 1 |
| `minkowski-derive` | 0 | 0 | 0 |
| `minkowski-observe` | 0 | 0 | 0 |
| `minkowski-bench` | 0 | 0 | 0 |
| `minkowski-py` | TBD | TBD | TBD |
| `minkowski-examples` | 0 | 0 | 0 |

### `unsafe_op_in_unsafe_fn` breakdown by file

| File | Warnings | Distinct operations | Notes |
|---|---|---|---|
| `query/fetch.rs` | 90 | 6 unique (×12 macro expansions + 5 direct) | `ptr.add`, `from_raw_parts`, `WorldQuery::fetch` delegation |
| `reducer.rs` | 42 | 3 unique (×12 macro expansions + 6 direct) | `fetch_writer` delegation, `ptr.add` deref |
| `storage/blob_vec.rs` | 15 | 15 | `copy_nonoverlapping`, `drop_fn`, `ptr_at`, `BlobVec::push/get_ptr` |
| `storage/sparse.rs` | 15 | 15 | `copy_nonoverlapping`, `drop_fn`, `BlobVec::push/get_ptr/insert` |
| `component.rs` | 4 | 4 | `drop_in_place` × 2 fns, pointer cast |

### `tail_expr_drop_order` breakdown

| File | Line | Issue |
|---|---|---|
| `transaction.rs` | 634 | `value` vs `tx` drop order in Optimistic::transact |
| `transaction.rs` | 732 | `value` vs `tx` drop order in Pessimistic::transact |
| `durable.rs` | 82 | `value` vs `tx` drop order in Durable::transact |

---

## Strategy

### `unsafe_op_in_unsafe_fn`: minimal wrapping with SAFETY comments

Wrap each unsafe operation in the smallest `unsafe { }` block that covers it. Add a `// SAFETY:` comment only when the invariant is non-obvious (not for simple delegations to other unsafe fns that have the same preconditions).

```rust
// BEFORE (edition 2021)
unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w T {
    &*fetch.ptr.add(row)
}

// AFTER (edition 2024)
unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w T {
    // SAFETY: caller guarantees row < archetype.len(); ptr was initialized by init_fetch
    unsafe { &*fetch.ptr.add(row) }
}
```

For macro-generated code (`impl_world_query_tuple!`, `impl_writer_query_tuple!`), the `unsafe { }` blocks go inside the macro body — the 12 expansions are fixed automatically.

### `tail_expr_drop_order`: explicit binding

The fix is to bind the match result to a variable before the match arms run, ensuring `tx` drop order is deterministic:

```rust
// BEFORE: tx might drop after value in edition 2024
let value = f(&mut tx, world);
match self.try_commit(&mut tx, world) {
    Ok(forward) => { ... return Ok(value); }
    Err(conflict) => { ... }
}

// AFTER: explicit drop order via semicolon
let value = f(&mut tx, world);
let result = self.try_commit(&mut tx, world);
match result { ... }
```

### `minkowski-derive` proc macro

The derive macro generates `unsafe fn put()`, `unsafe fn from_row()` — these need `unsafe { }` blocks in the generated `quote!` output. Must test from external crate (`minkowski-examples`).

---

## Tasks

### Task 1: component.rs (trivial — 4 warnings)

**Files:**
- Modify: `crates/minkowski/src/component.rs:130-131,142-143`

**Step 1: Wrap unsafe operations**

```rust
// Line 130-131: ComponentRegistry::drop_ptr
unsafe fn drop_ptr<T>(ptr: *mut u8) {
    // SAFETY: caller provides a valid ptr to a T that is ready to drop
    unsafe { std::ptr::drop_in_place(ptr as *mut T) };
}

// Line 142-143: standalone drop_ptr
pub(crate) unsafe fn drop_ptr<T>(ptr: *mut u8) {
    // SAFETY: caller provides a valid ptr to a T that is ready to drop
    unsafe { std::ptr::drop_in_place(ptr as *mut T) };
}
```

**Step 2: Verify**

Run: `RUSTFLAGS="-W unsafe-op-in-unsafe-fn" cargo check -p minkowski 2>&1 | grep component.rs`
Expected: no warnings from component.rs

**Step 3: Commit**

```bash
git add crates/minkowski/src/component.rs
git commit -m "edition 2024 prep: wrap unsafe ops in component.rs"
```

---

### Task 2: blob_vec.rs (15 warnings)

**Files:**
- Modify: `crates/minkowski/src/storage/blob_vec.rs`

**Step 1: Wrap all unsafe operations in unsafe fns**

Functions to modify (line numbers from current source):

| Function | Lines | Operations to wrap |
|---|---|---|
| `push` | 79-89 | `self.ptr_at(self.len)` (safe — ptr_at is not unsafe fn), `copy_nonoverlapping` |
| `get_ptr` | 105-108 | `self.ptr_at(row)` (safe) — **no wrapping needed** |
| `get_ptr_mut` | 120-124 | `self.ptr_at(row)` (safe) — **no wrapping needed** |
| `swap_remove` | 131-146 | `self.ptr_at` (safe), `drop_fn(ptr)` ×2, `copy_nonoverlapping` |
| `swap_remove_unchecked` | 153-168 | `self.ptr_at` (safe), `copy_nonoverlapping` ×2 |
| `swap_remove_no_drop` | 176-186 | `self.ptr_at` (safe), `copy_nonoverlapping` |
| `drop_in_place` | 195-200 | `drop_fn(ptr)` |
| `copy_unchecked` | 209-218 | `self.ptr_at` (safe), `copy_nonoverlapping` |
| `set_len` | 226-229 | **no unsafe ops** — just assignment |
| Test: `push_val` | 322-326 | `bv.push(ptr)` |
| Test: `read_val` | 329-331 | `bv.get_ptr(row)`, deref |
| Test: `drop_ptr` | 334-336 | `drop_in_place` |

Key insight: `ptr_at()` is a safe method (not `unsafe fn`) that internally uses `unsafe { self.data.as_ptr().add(...) }`. Its callers inside `unsafe fn` bodies don't need `unsafe { }` for the `ptr_at` call — they need it for `copy_nonoverlapping`, `drop_fn`, etc.

Wait — `ptr_at` calls `.add()` in its body, which is currently inside an existing `unsafe { }` block (line 243). `ptr_at` is a **safe fn** whose body already has an `unsafe { }` block. So calling `ptr_at` from an `unsafe fn` body doesn't require wrapping.

The actual unsafe ops that need wrapping:
- `std::ptr::copy_nonoverlapping(...)` — 5 call sites
- `drop_fn(ptr)` — 3 call sites (calling function pointer `unsafe fn(*mut u8)`)
- `bv.push(ptr)` — 1 call site in test helper (calling unsafe method)
- `bv.get_ptr(row)` — 1 call site in test helper
- `*(ptr as *const T)` — 1 deref in test helper
- `std::ptr::drop_in_place(...)` — 1 in test helper
- `std::mem::forget(val)` — 1 in test helper (NOT unsafe in recent Rust — safe since 1.0)

Total: ~13 operations need wrapping.

**Step 2: Verify**

Run: `RUSTFLAGS="-W unsafe-op-in-unsafe-fn" cargo check -p minkowski 2>&1 | grep blob_vec.rs`
Expected: no warnings from blob_vec.rs

**Step 3: Run blob_vec tests**

Run: `cargo test -p minkowski --lib blob_vec`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/minkowski/src/storage/blob_vec.rs
git commit -m "edition 2024 prep: wrap unsafe ops in blob_vec.rs"
```

---

### Task 3: sparse.rs (15 warnings)

**Files:**
- Modify: `crates/minkowski/src/storage/sparse.rs`

**Step 1: Wrap unsafe operations**

Functions to modify:

| Function | Operations |
|---|---|
| `insert` (73-107) | `get_ptr`, `drop_fn(dst)`, `copy_nonoverlapping`, `push(ptr as *mut u8)` |
| `insert_no_drop` (116-147) | `get_ptr`, `copy_nonoverlapping`, `push(ptr as *mut u8)` |
| Test: `drop_ptr` (236) | `drop_in_place` |
| `insert_raw` (344-366) | delegates to `PagedSparseSet::insert` |
| `insert_raw_no_drop` (369-387) | delegates to `PagedSparseSet::insert_no_drop` |
| Test: `insert_val` (442-446) | `set.insert(entity, &mut val as *mut T as *mut u8)` |
| Test: `read_ptr` (448-450) | `*(ptr as *const T)` |

**Step 2: Verify**

Run: `RUSTFLAGS="-W unsafe-op-in-unsafe-fn" cargo check -p minkowski 2>&1 | grep sparse.rs`
Expected: no warnings

**Step 3: Run sparse tests**

Run: `cargo test -p minkowski --lib sparse`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/minkowski/src/storage/sparse.rs
git commit -m "edition 2024 prep: wrap unsafe ops in sparse.rs"
```

---

### Task 4: query/fetch.rs (90 warnings — 6 unique sites in macros + 5 direct)

**Files:**
- Modify: `crates/minkowski/src/query/fetch.rs`

**Step 1: Wrap direct impl unsafe operations**

| Impl | Function | Operation |
|---|---|---|
| `&T` | `fetch` (142-143) | `&*fetch.ptr.add(row)` |
| `&T` | `as_slice` (146-148) | `from_raw_parts(fetch.ptr as *const T, len)` |
| `&mut T` | `fetch` (173-174) | `&mut *fetch.ptr.add(row)` |
| `&mut T` | `as_slice` (177-178) | `from_raw_parts_mut(fetch.ptr, len)` |
| `Entity` | `fetch` (196-198) | `*fetch.ptr.add(row)` |
| `Entity` | `as_slice` (200-202) | `from_raw_parts(fetch.ptr as *const Entity, len)` |
| `Option<&T>` | `fetch` (231-232) | `&*f.ptr.add(row)` inside `.map()` |
| `Option<&T>` | `as_slice` (235-238) | `from_raw_parts(f.ptr as *const T, len)` inside `.map()` |
| `Changed<T>` | `fetch` (279) | **no-op** (empty body) |
| `Changed<T>` | `as_slice` (281) | **no-op** (empty body) |

**Step 2: Wrap macro-generated tuple impl unsafe operations**

In `impl_world_query_tuple!` macro (lines 355-363):
```rust
unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
    let ($($name,)*) = fetch;
    // SAFETY: caller guarantees row < archetype.len()
    ($(unsafe { <$name as WorldQuery>::fetch($name, row) },)*)
}

unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
    let ($($name,)*) = fetch;
    // SAFETY: caller guarantees len == archetype.len()
    ($(unsafe { <$name as WorldQuery>::as_slice($name, len) },)*)
}
```

This single macro change fixes 24 of the 90 warnings (12 expansions × 2 methods).

**Step 3: Verify**

Run: `RUSTFLAGS="-W unsafe-op-in-unsafe-fn" cargo check -p minkowski 2>&1 | grep fetch.rs`
Expected: no warnings

**Step 4: Run fetch tests**

Run: `cargo test -p minkowski --lib fetch`
Expected: all pass

**Step 5: Commit**

```bash
git add crates/minkowski/src/query/fetch.rs
git commit -m "edition 2024 prep: wrap unsafe ops in query/fetch.rs"
```

---

### Task 5: reducer.rs (42 warnings — 3 unique sites in macros + 6 direct)

**Files:**
- Modify: `crates/minkowski/src/reducer.rs`

**Step 1: Wrap direct impl unsafe operations**

| Impl | Function | Operation |
|---|---|---|
| `&T` WriterQuery | `fetch_writer` (852-859) | delegates to `WorldQuery::fetch` |
| `&mut T` WriterQuery | `fetch_writer` (878-887) | `&*ptr.ptr.add(row)` |
| `Entity` WriterQuery | `fetch_writer` (902-909) | **no unsafe ops** (returns entity by value) |
| `Option<&T>` WriterQuery | `fetch_writer` (925-932) | delegates to `WorldQuery::fetch` |
| `Changed<T>` WriterQuery | `fetch_writer` (947-953) | **no-op** (empty body) |

**Step 2: Wrap macro-generated tuple impl**

In `impl_writer_query_tuple!` macro (lines 972-980):
```rust
unsafe fn fetch_writer<'w>(
    fetch: &Self::WriterFetch<'w>,
    row: usize,
    entity: Entity,
    changeset: *mut EnumChangeSet,
) -> Self::WriterItem<'w> {
    let ($($name,)*) = fetch;
    ($(unsafe { <$name as WriterQuery>::fetch_writer($name, row, entity, changeset) },)*)
}
```

**Step 3: Verify**

Run: `RUSTFLAGS="-W unsafe-op-in-unsafe-fn" cargo check -p minkowski 2>&1 | grep reducer.rs`
Expected: no warnings

**Step 4: Run reducer tests**

Run: `cargo test -p minkowski --lib reducer`
Expected: all pass

**Step 5: Commit**

```bash
git add crates/minkowski/src/reducer.rs
git commit -m "edition 2024 prep: wrap unsafe ops in reducer.rs"
```

---

### Task 6: Fix tail_expr_drop_order in transaction.rs and durable.rs (3 warnings)

**Files:**
- Modify: `crates/minkowski/src/transaction.rs:628-647,726-747`
- Modify: `crates/minkowski-persist/src/durable.rs:76-95`

**Step 1: Bind match subject to a variable**

In all three locations, the pattern is identical:
```rust
// BEFORE
let value = f(&mut tx, world);
match self.try_commit(&mut tx, world) {
    Ok(forward) => { ... }
    Err(conflict) => { ... }
}

// AFTER — ensures try_commit result drops before tx
let value = f(&mut tx, world);
let commit_result = self.try_commit(&mut tx, world);
match commit_result {
    Ok(forward) => { ... }
    Err(conflict) => { ... }
}
```

**Step 2: Verify**

Run: `RUSTFLAGS="-W tail-expr-drop-order" cargo check -p minkowski -p minkowski-persist 2>&1 | grep "tail_expr_drop_order\|relative drop order"`
Expected: no warnings

**Step 3: Run transaction tests**

Run: `cargo test -p minkowski --lib transaction`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/minkowski/src/transaction.rs crates/minkowski-persist/src/durable.rs
git commit -m "edition 2024 prep: fix tail_expr_drop_order in transaction code"
```

---

### Task 7: Update minkowski-derive proc macro for edition 2024

**Files:**
- Modify: `crates/minkowski-derive/src/lib.rs`

**Step 1: Add unsafe blocks in generated code**

The derive macro generates two `unsafe fn` impls:

1. `Bundle::put` — current generated code (line 54-63):
```rust
unsafe fn put(self, registry, func) {
    let #field_names = ManuallyDrop::new(self.#field_names);
    func(
        registry.id::<#field_types>().unwrap(),
        &#field_names as *const ManuallyDrop<#field_types> as *const #field_types as *const u8,
        Layout::new::<#field_types>(),
    );
}
```
The `func` call is NOT unsafe (it's `&mut dyn FnMut`). The pointer cast is NOT unsafe.
So `put` may have **zero** unsafe ops that need wrapping. Check by running `cargo check`.

2. `TableRow::from_row` (lines 104-128) — contains raw pointer arithmetic + deref:
```rust
unsafe fn from_row(col_ptrs: &[(*mut u8, usize)], row: usize) -> Self {
    Self {
        #field_names: &*(col_ptrs[#field_indices].0
            .add(row * col_ptrs[#field_indices].1)
            as *const #field_types),
    }
}
```
This needs `unsafe { &*(col_ptrs[...].0.add(...) as *const T) }` wrapping.

Update the `quote!` blocks to emit `unsafe { }` around pointer dereferences.

**Step 2: Verify from external crate**

Run: `cargo check -p minkowski-examples`
Expected: no errors (derive macro generates valid edition 2024 code)

**Step 3: Run table tests**

Run: `cargo test -p minkowski --lib table`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/minkowski-derive/src/lib.rs
git commit -m "edition 2024 prep: wrap unsafe ops in derive macro output"
```

---

### Task 8: Flip all Cargo.toml editions to 2024

**Files:**
- Modify: All 7 `Cargo.toml` files (`edition = "2021"` → `edition = "2024"`)
  - `crates/minkowski/Cargo.toml`
  - `crates/minkowski-derive/Cargo.toml`
  - `crates/minkowski-persist/Cargo.toml`
  - `crates/minkowski-observe/Cargo.toml`
  - `crates/minkowski-py/Cargo.toml`
  - `crates/minkowski-bench/Cargo.toml`
  - `examples/Cargo.toml`

**Step 1: Update all editions**

Change `edition = "2021"` to `edition = "2024"` in each file.

**Step 2: Full workspace check**

Run: `cargo clippy --workspace --all-targets --exclude minkowski-py -- -D warnings`
Expected: clean

**Step 3: Full test suite**

Run: `cargo test -p minkowski`
Expected: 445+ tests pass

**Step 4: Commit**

```bash
git add */Cargo.toml crates/*/Cargo.toml
git commit -m "chore: migrate workspace to Rust edition 2024"
```

---

### Task 9: Handle minkowski-py (if applicable)

**Files:**
- Modify: `crates/minkowski-py/src/*.rs` (if unsafe ops exist)

Check with: `RUSTFLAGS="-W rust-2024-compatibility" cargo check -p minkowski-py`

Fix any warnings, then commit separately.

---

### Task 10: Final verification

**Step 1: Full CI-equivalent checks**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --exclude minkowski-py -- -D warnings
cargo test -p minkowski
cargo test -p minkowski-persist
```

**Step 2: Verify no 2024-compatibility warnings remain**

```bash
RUSTFLAGS="-W rust-2024-compatibility" cargo check --workspace --exclude minkowski-py
```
Expected: clean

**Step 3: Squash or create PR**
