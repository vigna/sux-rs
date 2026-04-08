# Hollow Trie Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `hollow_trie.rs` so that both byte-sequence and integer paths follow the same distributor + MMPHF-wrapper pattern, with all inner types as type parameters to enable epserde derives and generic `TryIntoUnaligned`.

**Architecture:** Extract `HtDistInt<K>` as a new integer distributor parallel to `HtDist`. Make `HtDistMmphfInt` wrap `HtDistInt` by composition (like `HtDistMmphf` wraps `HtDist`). Parameterize all VFunc fields as full type parameters `E`, `F`, `O` instead of constructing them from a data-store parameter `D`.

**Tech Stack:** Rust, sux crate (succinct data structures), VFunc, epserde, ambassador

**Spec:** `docs/superpowers/specs/2026-04-08-hollow-trie-refactoring-design.md`

---

## Pre-Flight

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `cargo test --features epserde,rayon --lib hollow_trie -- --nocapture 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 2: Commit any pending changes**

---

### Task 1: Refactor HtDist Type Parameters

Change `HtDist<D, B, S>` to `HtDist<E, F, B, S>` where `E` and `F` are the
full VFunc types for external behavior and false-follows detector.

**Files:**
- Modify: `src/func/hollow_trie.rs`

- [ ] **Step 1: Update HtDist struct definition**

Change the struct from:

```rust
pub struct HtDist<
    D = BitFieldVec<Box<[usize]>>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    bal_paren: B,
    skips: S,
    num_nodes: usize,
    num_delimiters: usize,
    false_follows_detector: VFunc<[u8], D, [u64; 1], FuseLge3NoShards>,
    external_behaviour: VFunc<[u8], D, [u64; 1], FuseLge3NoShards>,
}
```

To:

```rust
pub struct HtDist<
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = CompIntList,
> {
    bal_paren: B,
    skips: S,
    num_nodes: usize,
    num_delimiters: usize,
    external_behaviour: E,
    false_follows_detector: F,
}
```

- [ ] **Step 2: Update HtDist MemSize impl**

Change the impl header from `impl<D: ..., B: ...> MemSize for HtDist<D, B>`
to `impl<E: MemSize + ..., F: MemSize + ..., B: MemSize + ..., S: MemSize + ...> MemSize for HtDist<E, F, B, S>`.
Adjust the `mem_size_rec` body accordingly — it already calls `.mem_size_rec()`
on each field, just the type bounds change.

- [ ] **Step 3: Update HtDist MemDbgImpl impl**

Same type parameter update as MemSize.

- [ ] **Step 4: Update HtDist::try_new constructor**

The constructor currently has impl header:
```rust
impl<B: BalParen + ..., S: ...> HtDist<BitFieldVec<Box<[usize]>>, B, S>
```

Change to fix the E and F parameters to the concrete types the constructor builds:
```rust
impl HtDist
```

This works because `try_new` constructs the VFuncs with concrete types that
match the defaults. The constructor body stays the same — it already creates
`VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>` values
and assigns them to `false_follows_detector` and `external_behaviour`.

Note: Check whether the existing `try_new` impl block uses the old `D`
parameter in its bounds or body and update accordingly. The `VFunc::try_new`
calls use concrete types, so no change needed there.

- [ ] **Step 5: Update HtDist::get impl block**

The `get()` method's impl block currently has bounds on `D`. Change from:
```rust
impl<D: SliceByValue<Value = usize> + MemSize, B: BalParen + AsRef<[usize]>> HtDist<D, B>
```
To bounds on E and F:
```rust
impl<
    E: SliceByValue<Value = usize>,
    F: SliceByValue<Value = usize>,
    B: BalParen + AsRef<[usize]>,
    S: SliceByValue<Value = usize>,
> HtDist<E, F, B, S>
```

Inside `get()`, references to `self.false_follows_detector.get(...)` and
`self.external_behaviour.get(...)` already use the `SliceByValue` trait, so the
body should compile unchanged. The `self.skips.get_value(...)` call needs `S: SliceByValue`.

Check that all trait bounds match what the method body actually calls. Minimize
bounds per guidelines.

- [ ] **Step 6: Run tests**

Run: `cargo test --features epserde,rayon --lib hollow_trie`
Expected: all tests pass

- [ ] **Step 7: Commit**

```
git commit -am "refactor: change HtDist type params from <D, B, S> to <E, F, B, S>"
```

---

### Task 2: Refactor HtDistMmphf Type Parameters

Change `HtDistMmphf<K, D, B>` to `HtDistMmphf<K, E, F, O, B, S>`.

**Files:**
- Modify: `src/func/hollow_trie.rs`

- [ ] **Step 1: Update HtDistMmphf struct definition**

Change from:
```rust
pub struct HtDistMmphf<K: ?Sized, D = BitFieldVec<Box<[usize]>>, B: BalParen = JacobsonBalParen> {
    distributor: HtDist<D, B>,
    offset: VFunc<K, D>,
    log2_bucket_size: usize,
    n: usize,
}
```

To:
```rust
pub struct HtDistMmphf<
    K: ?Sized,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    O = VFunc<K, BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = CompIntList,
> {
    distributor: HtDist<E, F, B, S>,
    offset: O,
    log2_bucket_size: usize,
    n: usize,
}
```

- [ ] **Step 2: Update HtDistMmphf MemSize impl**

Update type params and bounds. The body calls `self.distributor.mem_size_rec()`
and `self.offset.mem_size_rec()`, so E, F, O, B, S all need `MemSize` bounds.

- [ ] **Step 3: Update HtDistMmphf MemDbgImpl impl**

Same type parameter update.

- [ ] **Step 4: Update type aliases**

Change:
```rust
pub type HtDistMmphfStr<D = BitFieldVec<Box<[usize]>>> = HtDistMmphf<str, D>;
pub type HtDistMmphfSliceU8<D = BitFieldVec<Box<[usize]>>> = HtDistMmphf<[u8], D>;
```

To:
```rust
pub type HtDistMmphfStr = HtDistMmphf<str>;
pub type HtDistMmphfSliceU8 = HtDistMmphf<[u8]>;
```

Type aliases cannot have default type parameters, but since the underlying
struct has defaults, users can write `HtDistMmphfStr` and get all defaults.

- [ ] **Step 5: Update HtDistMmphf::try_new constructor**

The impl block currently uses `D` in bounds. Change to fix E, F, O to concrete
defaults (the constructor builds those exact types). Update the `Ok(HtDistMmphf { ... })`
at the end — the `distributor` field is now built via `HtDist::try_new()`
(which returns `HtDist` with default E, F types), and `offset` is the VFunc
built in the constructor.

- [ ] **Step 6: Update HtDistMmphf::get impl block**

The `get()` method delegates to `self.distributor.get()` and
`self.offset.get()`. Update bounds on E, F, O, B, S to match what the method
body needs (primarily `SliceByValue<Value = usize>` on the VFunc types and
`BalParen + AsRef<[usize]>` on B).

- [ ] **Step 7: Update HtDistMmphf accessor methods**

Update `len()`, `is_empty()`, and any other inherent methods to use the new
type params.

- [ ] **Step 8: Run tests**

Run: `cargo test --features epserde,rayon --lib hollow_trie`
Expected: all tests pass

- [ ] **Step 9: Commit**

```
git commit -am "refactor: change HtDistMmphf type params to <K, E, F, O, B, S>"
```

---

### Task 3: Extract HtDistInt from HtDistMmphfInt

Create the new `HtDistInt<K, E, F, B, S>` struct by extracting the distributor
fields and trie navigation logic from `HtDistMmphfInt`.

**Files:**
- Modify: `src/func/hollow_trie.rs`

- [ ] **Step 1: Define HtDistInt struct**

Add after `HtDist` and its impl blocks (before the `HtDistMmphf` section):

```rust
pub struct HtDistInt<
    K,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = CompIntList,
> {
    bal_paren: B,
    skips: S,
    num_nodes: usize,
    num_delimiters: usize,
    external_behaviour: E,
    false_follows_detector: F,
    _marker: PhantomData<K>,
}
```

- [ ] **Step 2: Implement HtDistInt::try_new**

Extract the distributor construction logic from `HtDistMmphfInt::try_new`.
The current `HtDistMmphfInt::try_new` builds the trie, computes behavior keys,
creates the two VFuncs, AND builds the offset VFunc. The distributor constructor
should do everything EXCEPT the offset VFunc.

The constructor should take:
- `delimiters: &[K]` — the bucket delimiter keys
- `keys_for_behaviour: &[K]` — all keys (needed for behavior key computation)
- `log2_bucket_size: usize`
- `pl: &mut impl ProgressLog`

Look at the current `HtDistMmphfInt::try_new` (lines ~1754-2071) and extract:
- Trie building with `HollowTrieBuilderInt` (the delimiter push loop)
- Balanced-parentheses construction
- Skip list construction
- Behavior key computation (the second pass over all keys)
- VFunc construction for `false_follows_detector` and `external_behaviour`

Return `Result<Self>`.

- [ ] **Step 3: Implement HtDistInt::get**

Extract the trie navigation from `HtDistMmphfInt::get` (lines ~2084-2168).
The current method computes `bucket_index` (the distributor part) and then
`offset` (the MMPHF part). Extract only the distributor part.

The method signature:
```rust
pub fn get(&self, key: K) -> usize
```

This contains the integer-specific bit extraction using `get_int_bit` and the
XOR mapping with `K::MIN`. The trie navigation structure mirrors `HtDist::get`
but uses `get_int_bit` instead of `get_key_bit`.

- [ ] **Step 4: Implement MemSize and MemDbgImpl for HtDistInt**

Follow the same pattern as `HtDist`'s implementations.

- [ ] **Step 5: Run tests (tests won't change yet — HtDistMmphfInt still uses old code)**

Run: `cargo test --features epserde,rayon --lib hollow_trie`
Expected: all tests pass (HtDistInt is defined but not yet used)

- [ ] **Step 6: Commit**

```
git commit -am "feat: add HtDistInt<K, E, F, B, S> integer distributor"
```

---

### Task 4: Refactor HtDistMmphfInt to Wrap HtDistInt

Change `HtDistMmphfInt` from a flat struct to a wrapper around `HtDistInt`.

**Files:**
- Modify: `src/func/hollow_trie.rs`

- [ ] **Step 1: Update HtDistMmphfInt struct definition**

Change from the current flat struct (with `bal_paren`, `skips`,
`false_follows_detector`, `external_behaviour`, `offset`, etc.) to:

```rust
pub struct HtDistMmphfInt<
    K,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    O = VFunc<K, BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = CompIntList,
> {
    distributor: HtDistInt<K, E, F, B, S>,
    offset: O,
    log2_bucket_size: usize,
    n: usize,
}
```

- [ ] **Step 2: Update HtDistMmphfInt::try_new**

Refactor to call `HtDistInt::try_new` for the distributor part, then build the
offset VFunc separately. The multi-pass orchestration stays here:

1. First pass: compute bucket size, collect delimiters
2. Call `HtDistInt::try_new(delimiters, all_keys, log2_bucket_size, pl)`
3. Second pass: build offset VFunc using `distributor.get()` to compute
   bucket assignments

Return `Ok(HtDistMmphfInt { distributor, offset, log2_bucket_size, n })`.

- [ ] **Step 3: Update HtDistMmphfInt::get**

Simplify from the current monolithic implementation to delegation:

```rust
pub fn get(&self, key: K) -> usize {
    let bucket = self.distributor.get(key);
    (bucket << self.log2_bucket_size) + self.offset.get(key)
}
```

This should match the pattern of `HtDistMmphf::get`.

- [ ] **Step 4: Update HtDistMmphfInt MemSize and MemDbgImpl**

Update to use the new field structure (`distributor` + `offset` instead of all
the flat fields).

- [ ] **Step 5: Update HtDistMmphfInt accessor methods**

Update `len()`, `is_empty()` to work with the new structure.

- [ ] **Step 6: Run ALL tests**

Run: `cargo test --features epserde,rayon --lib hollow_trie`
Expected: all tests pass — this is the critical regression check

- [ ] **Step 7: Commit**

```
git commit -am "refactor: HtDistMmphfInt now wraps HtDistInt by composition"
```

---

### Task 5: Update TryIntoUnaligned and From Implementations

Update all `TryIntoUnaligned` and `From<Unaligned<...>>` implementations to
work generically with the new type parameters.

**Files:**
- Modify: `src/func/hollow_trie.rs`

- [ ] **Step 1: Update HtDist TryIntoUnaligned**

The current impl is `impl TryIntoUnaligned for HtDist` (uses all defaults).
Update to be generic over the type parameters:

```rust
impl<E: TryIntoUnaligned, F: TryIntoUnaligned, B: BalParen + TryIntoUnaligned, S>
    TryIntoUnaligned for HtDist<E, F, B, S>
{
    type Unaligned = HtDist<Unaligned<E>, Unaligned<F>, Unaligned<B>, S>;
    ...
}
```

Note: `S` (skip storage) is NOT converted — it stays as-is. Check whether
this matches the current behavior (current impl passes `self.skips` through
unchanged).

- [ ] **Step 2: Update HtDist From<Unaligned<...>>**

Make generic to match the new TryIntoUnaligned. The `From` impl converts
unaligned VFuncs and bal_paren back to aligned versions.

- [ ] **Step 3: Add HtDistInt TryIntoUnaligned and From**

Follow the same pattern as HtDist. The `PhantomData<K>` field passes through
unchanged.

- [ ] **Step 4: Update HtDistMmphf TryIntoUnaligned**

Currently: `impl<K: ?Sized> TryIntoUnaligned for HtDistMmphf<K>`.
Update to be generic over E, F, O, B, S:

```rust
impl<K: ?Sized, E: TryIntoUnaligned, F: TryIntoUnaligned, O: TryIntoUnaligned, B: BalParen + TryIntoUnaligned, S>
    TryIntoUnaligned for HtDistMmphf<K, E, F, O, B, S>
{
    type Unaligned = HtDistMmphf<K, Unaligned<E>, Unaligned<F>, Unaligned<O>, Unaligned<B>, S>;
    ...
}
```

The body calls `self.distributor.try_into_unaligned()` and
`self.offset.try_into_unaligned()`.

- [ ] **Step 5: Update HtDistMmphf From<Unaligned<...>>**

Make generic.

- [ ] **Step 6: Update HtDistMmphfInt TryIntoUnaligned and From**

Follow the same pattern as HtDistMmphf, using `HtDistInt` as the distributor.

- [ ] **Step 7: Run tests**

Run: `cargo test --features epserde,rayon --lib hollow_trie`
Expected: all tests pass

- [ ] **Step 8: Commit**

```
git commit -am "refactor: generic TryIntoUnaligned for all hollow trie structs"
```

---

### Task 6: Clean Up and Remove Dead Code

Remove duplicated code that is no longer needed after the refactoring.

**Files:**
- Modify: `src/func/hollow_trie.rs`

- [ ] **Step 1: Remove duplicated trie navigation code**

After Task 4, `HtDistMmphfInt::get` delegates to `HtDistInt::get`. Verify
that the old monolithic `get` implementation in `HtDistMmphfInt` is fully
removed and no dead code remains.

- [ ] **Step 2: Remove duplicated constructor code**

Verify that `HtDistMmphfInt::try_new` delegates trie construction to
`HtDistInt::try_new` and doesn't duplicate the behavior-key computation.

- [ ] **Step 3: Verify doc comments are accurate**

Check that struct-level and method-level documentation references are correct:
- `HtDistMmphfInt` docs should mention it wraps `HtDistInt`
- `HtDistInt` docs should parallel `HtDist` docs
- Type parameter docs on each struct should list `E`, `F`, `O`, `B`, `S`

- [ ] **Step 4: Verify impl block ordering follows guidelines**

For each struct, check order:
1. Declaration + derives
2. Inherent impls (constructors, accessors, `map_*`)
3. Crate traits (`TryIntoUnaligned`)
4. External crate traits (`MemSize`, `MemDbgImpl`)
5. Std traits (`From`)

- [ ] **Step 5: Run full test suite**

Run: `cargo test --features epserde,rayon`
Expected: all tests pass (not just hollow_trie — full suite to catch breakage
in downstream users)

- [ ] **Step 6: Run clippy**

Run: `cargo clippy --features epserde,rayon`
Expected: no new warnings

- [ ] **Step 7: Commit**

```
git commit -am "chore: clean up hollow trie after refactoring"
```

---

### Task 7: Update Doc Tests and Public API Surface

Fix doc examples and verify the public API is consistent.

**Files:**
- Modify: `src/func/hollow_trie.rs`
- Modify: `src/func/mod.rs` (if re-exports need updating)

- [ ] **Step 1: Update HtDist doc examples**

Check all `/// ```rust` blocks in HtDist. Update any that reference old type
params `D`.

- [ ] **Step 2: Update HtDistMmphf doc examples**

Same — update references to old type params.

- [ ] **Step 3: Add HtDistInt doc example**

Add a basic usage example showing construction and query:

```rust
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// use sux::func::HtDistInt;
/// use dsi_progress_logger::no_logging;
/// // ... example ...
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
```

- [ ] **Step 4: Update HtDistMmphfInt doc examples**

Update to reflect the new wrapper structure.

- [ ] **Step 5: Update re-exports in src/func/mod.rs**

Add `HtDistInt` to the public exports if not automatically covered by a glob.

- [ ] **Step 6: Run doc tests**

Run: `cargo test --features epserde,rayon --doc hollow_trie`
Expected: all doc tests pass

- [ ] **Step 7: Commit**

```
git commit -am "docs: update hollow trie documentation for new API"
```
