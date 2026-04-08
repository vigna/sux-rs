# Hollow Trie Refactoring Design

## Problem

The hollow trie module (`src/func/hollow_trie.rs`, ~2475 lines) has two
implementations that diverged:

- **Byte-sequence path:** `HtDist` (distributor) + `HtDistMmphf` (MMPHF wrapper)
- **Integer path:** `HtDistMmphfInt` (flat struct duplicating all distributor
  fields inline)

This causes several problems:

1. `HtDistMmphfInt` duplicates all of `HtDist`'s fields and trie navigation
   logic instead of wrapping a distributor.
2. Type parameters are incomplete: `HtDistMmphf` forwards `<D, B>` but loses
   `S` (skip storage); `HtDistMmphfInt` has `<D, B, S>` but no distributor
   wrapper.
3. The VFunc fields are hardcoded constructed types
   (`VFunc<[u8], D, [u64; 1], FuseLge3NoShards>`), making epserde derives
   impossible.
4. `TryIntoUnaligned` is manually implemented only for default types, not
   generically.
5. No epserde derives on any struct.

## Design

### Architecture

```
HtDist<E, F, B, S>              — byte-sequence distributor
HtDistInt<K, E, F, B, S>        — integer distributor (new)
HtDistMmphf<K, E, F, O, B, S>   — byte-sequence MMPHF, wraps HtDist
HtDistMmphfInt<K, E, F, O, B, S> — integer MMPHF, wraps HtDistInt
```

Both MMPHF structs compose their respective distributor plus an offset VFunc.

### Type Parameters

All structs share the same parameter convention:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `K` | Key type (`?Sized` for byte-seq, `PrimitiveInteger` for int) | — |
| `E` | External behavior VFunc | `VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>` |
| `F` | False-follows detector VFunc | same as `E` |
| `O` | Offset VFunc (MMPHF only) | `VFunc<K, BitFieldVec<Box<[usize]>>>` |
| `B` | Balanced parentheses structure | `JacobsonBalParen` (with `BalParen` bound) |
| `S` | Skip storage | `PrefixSumIntList` |

Both behavior VFuncs use `[u8]` as their key type because the behavior key
encodes a composite `(node_pos: usize, path_len: usize, path_bits)` that must
be serialized to bytes.

### Struct Definitions

**HtDist:**

```rust
pub struct HtDist<
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    bal_paren: B,
    skips: S,
    num_nodes: usize,
    num_delimiters: usize,
    external_behaviour: E,
    false_follows_detector: F,
}
```

**HtDistInt:**

```rust
pub struct HtDistInt<
    K,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
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

`HtDistInt` needs `PhantomData<K>` because `K` determines how bits are
extracted from keys during `get()` but does not appear in any field type.

**HtDistMmphf:**

```rust
pub struct HtDistMmphf<
    K: ?Sized,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    O = VFunc<K, BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    distributor: HtDist<E, F, B, S>,
    offset: O,
    log2_bucket_size: usize,
    n: usize,
}
```

**HtDistMmphfInt:**

```rust
pub struct HtDistMmphfInt<
    K,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>, [u64; 1], FuseLge3NoShards>,
    O = VFunc<K, BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    distributor: HtDistInt<K, E, F, B, S>,
    offset: O,
    log2_bucket_size: usize,
    n: usize,
}
```

### Construction

**Distributors** (`HtDist::try_new`, `HtDistInt::try_new`):

- Take delimiters (as a slice) + bucket size + progress logger
- Build the trie, compute behavior keys, construct behavior VFuncs
- Return `Result<Self>`

**MMPHF constructors** (`HtDistMmphf::try_new`, `HtDistMmphfInt::try_new`):

- Take keys (as `FallibleRewindableLender`) + count + progress logger
- Multi-pass orchestration:
  1. Compute bucket size, collect delimiters
  2. Call distributor constructor
  3. Build offset VFunc
- Return `Result<Self>`

### Builders

`HollowTrieBuilder` (byte-sequence) and `HollowTrieBuilderInt<K>` (integer)
remain as-is. They are internal construction helpers that produce
`(BitVec, Vec<usize>, usize)`.

### Serialization and Unaligned Conversion

With all inner types as type parameters, the structs can:

1. Derive `epserde::Epserde` via `#[cfg_attr(feature = "epserde", derive(...))]`
2. Implement `TryIntoUnaligned` generically (not just for default types)
3. Implement `From<Unaligned<Self>>` generically

### Type Aliases

Preserved:

```rust
pub type HtDistMmphfStr<E, F, O, B, S> = HtDistMmphf<str, E, F, O, B, S>;
pub type HtDistMmphfSliceU8<E, F, O, B, S> = HtDistMmphf<[u8], E, F, O, B, S>;
```

### Impl Block Ordering

Per project guidelines, each struct's impl blocks appear in this order:

1. Declaration + derives
2. Inherent implementations (constructors, `map_*` methods, accessors)
3. Crate trait implementations (`TryIntoUnaligned`, etc.)
4. External crate trait implementations (`MemSize`, `MemDbgImpl`, etc.)
5. Standard library trait implementations (`From`, etc.)

### File Organization

Everything stays in `src/func/hollow_trie.rs` for now. The refactoring should
reduce code duplication (the integer distributor's `get()` logic is extracted
from `HtDistMmphfInt` rather than duplicated). If the file remains large after
deduplication, it can be split in a later pass.

## Non-Goals

- Changing the trie construction algorithm
- Changing the behavior key encoding (stays as `[u8]` serialization for both
  variants)
- Splitting into multiple files (deferred)
- Making the trie navigation generic over a bit-access trait (the byte and
  integer `get()` methods remain separate implementations)
