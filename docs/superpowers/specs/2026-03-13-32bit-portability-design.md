# 32-Bit Portability Design for sux-rs

**Date:** 2026-03-13
**Status:** Draft
**Goal:** Make sux-rs compile and run on 32-bit targets (WASM, embedded, IoT) while preserving full performance on 64-bit.

---

## Motivation

sux-rs currently enforces `target_pointer_width = "64"` at compile time. The most interesting near-term target is WASM (wasm32), which has 32-bit addressing but native 64-bit integer arithmetic. Embedded and IoT 32-bit targets are also desirable. The design should enable all three use cases without sacrificing 64-bit performance.

## Key Design Principles

1. **`usize` is an index type, not a word type.** Indices (positions, ranks, lengths) remain `usize` on all platforms.
2. **Word type is generic.** Bit-storage structures (`BitVec`, `BitFieldVec`) are generic over the backing word type `W: Word`, inferred from the backing slice type `B: AsRef<[W]>`.
3. **Platform word as default.** A `PlatformWord` alias (`u64` on 64-bit, `u32` on 32-bit) provides the default word type.
4. **64-bit-native algorithms stay 64-bit.** Rank9 and Select9 are designed around 512-bit (8x64) blocks. They always use `u64` words and are available on all platforms.
5. **Values vs. storage.** EliasFano stores `u64` values (universe not platform-limited) but uses platform-word storage for its internal bit vectors.

## Detailed Design

### 1. Foundation: Word Type and PlatformWord

The existing `Word` trait in `bit_field_slice.rs` serves as the foundation. It requires `PrimitiveUnsigned + PrimitiveNumberAs<u128>` and is implemented for `u8, u16, u32, u64, u128, usize`. Additional trait bounds may be needed:

- `SelectInWord` for rank/select inner loops
- Standard bit-manipulation methods (`count_ones`, `trailing_zeros`, `leading_zeros`) are available via primitive types

A platform word alias:

```rust
#[cfg(target_pointer_width = "64")]
pub type PlatformWord = u64;
#[cfg(target_pointer_width = "32")]
pub type PlatformWord = u32;
```

### 2. BitVec

Currently `BitVec<B = Vec<usize>>`. The word type is inferred from `B`:

```rust
pub struct BitVec<B = Vec<PlatformWord>> { ... }
```

No explicit `W` parameter. `B: AsRef<[u64]>` means word type is `u64`; `B: AsRef<[u32]>` means `u32`. The default changes from `Vec<usize>` to `Vec<PlatformWord>` (identical on 64-bit).

### 3. BitVecOps / BitVecOpsMut / AtomicBitVecOps

Currently hardcoded to `AsRef<[usize]>` and `AsRef<[AtomicUsize]>`:

```rust
pub trait BitVecOps: AsRef<[usize]> + BitLength { ... }
pub trait AtomicBitVecOps: AsRef<[AtomicUsize]> + BitLength { ... }
```

Both become generic over `W: Word`:

```rust
pub trait BitVecOps<W: Word>: AsRef<[W]> + BitLength { ... }
pub trait AtomicBitVecOps<W: Word>: AsRef<[W::Atomic]> + BitLength { ... }
```

where `W::Atomic` comes from the `AtomicPrimitive` trait.

The module-level constant `pub const BITS: usize = usize::BITS as usize` is replaced by `W::BITS` inside methods.

Iterators (`BitIter`, `OnesIter`, `ZerosIter`, `AtomicBitIter`) become generic over `W`. Their stored `word` fields change from `usize` to `W`. They use `W::trailing_zeros()`, `W::count_ones()`, etc.

### 4. BitFieldVec

Stays `BitFieldVec<V, B>` (no explicit word-type parameter). The word type `V` is inferred from `B: AsRef<[V]>`, and `bit_width` is enforced to be `<= V::BITS`.

**Constraint:** `BitFieldVec<V, B>` couples the value type and the storage word type — `V` serves as both. A `BitFieldVec<u64>` always uses `u64` words; a `BitFieldVec<u32>` always uses `u32` words. Values wider than `V::BITS` cannot be stored. This means structures that need wide values (like EliasFano's low bits, up to 63 bits) must use `BitFieldVec<u64>` with u64-word backing, even on 32-bit platforms.

### 5. AsRef<[W]> Propagation

Every structure exposes `AsRef<[W]>` matching its underlying word type. A `Rank9<BitVec<Box<[u64]>>>` exposes `AsRef<[u64]>`. Delegation via `ambassador` propagates the concrete type through composition stacks.

**Risk:** Ambassador `#[delegate(AsRef<[usize]>)]` currently generates code for a concrete type. Making this generic requires structures to be generic over `W`, so the delegation resolves at monomorphization. Structures like `SelectAdapt<B, I>` do not currently have a `W` parameter; the word type must be inferred from `B`'s `AsRef` impl. If ambassador cannot handle this, we fall back to a custom `AsWordSlice` trait with a blanket impl for `AsRef<[W]>`.

### 6. RankHinted\<N\>

Stays as a const generic, tied to the concrete word type:

- `BitVec<Vec<u64>>` implements `RankHinted<64>`
- `BitVec<Vec<u32>>` implements `RankHinted<32>`

The implementation body is identical across word sizes: iterate words, `count_ones()`, mask the last word.

**Delegation cascade:** All ambassador `#[delegate(RankHinted<64>)]` macros throughout the rank/select structures are currently hardcoded to `64`. These must become `RankHinted<32>` on 32-bit, or be parameterized. For structures that are generic over word type (like RankSmall), this requires either:
- A const generic `N` on the structure itself (propagated from the word type)
- Two `cfg`-gated delegation lines (`RankHinted<64>` on 64-bit, `RankHinted<32>` on 32-bit)
- A macro that expands the delegation based on the word type

For structures pinned to `u64` (Rank9, Select9), the delegation stays `RankHinted<64>` unconditionally.

### 7. Rank9

Always `u64`. Default type parameter changes:

```rust
pub struct Rank9<B = BitVec<Box<[u64]>>> { ... }
```

Requires `B: AsRef<[u64]> + RankHinted<64>`. `WORDS_PER_BLOCK = 8` (512-bit blocks).

**`BlockCounters` must change:** Currently the fields `absolute: usize` and `relative: usize` are `usize`, not `u64`. Since Rank9 is pinned to `u64`, these must become `u64`:

```rust
pub struct BlockCounters {
    pub(super) absolute: u64,
    pub(super) relative: u64,
}
```

The `rel()` and `set_rel()` methods pack 9-bit counters into the `relative` field — this requires 63 bits (7 counters × 9 bits), which only fits in `u64`. The `#[epserde_zero_copy]` and `#[repr(C)]` annotations on `BlockCounters` remain valid since the layout is now explicitly `u64` regardless of platform (identical to current 64-bit layout).

All uses of `usize::BITS` in Rank9's `new()` and `rank_unchecked` (e.g., `pos / usize::BITS as usize`) must become `u64::BITS as usize` since the structure is pinned to u64.

Available on all platforms; on 32-bit, you must provide a `u64`-backed bit vector.

### 8. RankSmall

The portable rank structure. Works with any word size natively.

**Block-size formula adapts:** `WORDS_PER_BLOCK = 1 << (COUNTER_WIDTH - usize::BITS.ilog2())` keeps block *bit*-sizes constant across platforms (e.g., 512 bits for `<2, 9>`), adjusting the word count (8 × 64-bit words or 16 × 32-bit words).

**Per-word arithmetic does NOT adapt and must be changed:** 5 occurrences of hardcoded `64` as word size (lines 362, 364, 419, 432, 446) must become `W::BITS` equivalents (e.g., `pos / 64` → `pos / W::BITS`).

**Superblock index:** `word_pos / (1usize << 26)` generalizes to `word_pos / (1usize << (32 - W::BITS.ilog2()))`. On 64-bit: `1 << 26` (unchanged). On 32-bit: `1 << 27`, giving one superblock covering the entire 2³² address space.

**Upper counts cfg-gated on 32-bit:** On 32-bit, the entire bit vector is <= 2^32 bits, so there is always exactly one superblock with `upper_count = 0`. Both the `upper_counts` field and the `C1` generic parameter are removed on 32-bit via two cfg-gated struct definitions:

```rust
#[cfg(target_pointer_width = "64")]
pub struct RankSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B, C1, C2,
> {
    bits: B,
    upper_counts: C1,
    counts: C2,
    num_ones: usize,
}

#[cfg(not(target_pointer_width = "64"))]
pub struct RankSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B, C2,
> {
    bits: B,
    counts: C2,
    num_ones: usize,
}
```

This means all downstream code that names `RankSmall` with explicit type parameters (the `impl_rank_small!` macro, `SelectSmall`, `SelectZeroSmall`, etc.) must also be cfg-gated. Platform-specific type aliases can reduce the blast radius:

```rust
#[cfg(target_pointer_width = "64")]
pub type DefaultRankSmall<const N: usize, const C: usize, B> =
    RankSmall<N, C, B, Box<[usize]>, Box<[Block32Counters<N, C>]>>;

#[cfg(not(target_pointer_width = "64"))]
pub type DefaultRankSmall<const N: usize, const C: usize, B> =
    RankSmall<N, C, B, Box<[Block32Counters<N, C>]>>;
```

In `rank_unchecked`:

```rust
#[cfg(target_pointer_width = "64")]
let upper_count = *self.upper_counts.as_ref().get_unchecked(...);
#[cfg(not(target_pointer_width = "64"))]
let upper_count = 0;
```

`Block32Counters` with `absolute: u32` already covers the full range on 32-bit. No change needed.

### 9. Select9

Always `u64`, like Rank9. Built on top of Rank9; shares 512-bit block structure.

**Broadword constants must change type:** `ONES_STEP_9` and `ONES_STEP_16` are currently `usize` (e.g., `1usize << 54` would overflow on 32-bit). Since Select9 is pinned to `u64`, these must become `u64` constants:

```rust
const ONES_STEP_9: u64 = (1u64 << 0) | (1u64 << 9) | ... | (1u64 << 54);
const ONES_STEP_16: u64 = (1u64 << 0) | (1u64 << 16) | (1u64 << 32) | (1u64 << 48);
```

### 10. SelectAdapt / SelectZeroAdapt / SelectAdaptConst / SelectZeroAdaptConst

All four adapt variants share the `Inventory` trait impl and `SpanType` enum.

- **Inventory:** `usize` entries with span type in top 2 bits
- **Bit-packing adapts to platform:** `>> 62` becomes `>> (usize::BITS - 2)`, `<< 62` becomes `<< (usize::BITS - 2)`, `<< 63` becomes `<< (usize::BITS - 1)`
- **Position limit on 32-bit:** 30 bits (~1 billion bit positions). Accepted as sufficient; on 32-bit platforms, data being indexed limits `n` well below this.
- **Word iteration:** `select_in_word`, `count_ones` follow the backing word type via `W: Word + SelectInWord`

**`SpanType::from_span` must be cfg-gated:** The match arm `0x10001..=0x100000000` uses a literal that exceeds `usize::MAX` on 32-bit. On 32-bit, only two span types are reachable (U16 and U32); the U64 span type is dead code since `usize` cannot represent spans > 2³². The match must be cfg-gated:

```rust
#[cfg(target_pointer_width = "64")]
0x10001..=0x100000000 => SpanType::U32,
#[cfg(not(target_pointer_width = "64"))]
0x10001.. => SpanType::U32,
```

The const parameter `LOG2_U64S_PER_SUBINVENTORY` in SelectAdaptConst/SelectZeroAdaptConst is renamed to `LOG2_WORDS_PER_SUBINVENTORY` since subinventory entries are `usize`, not necessarily `u64`.

### 11. SelectSmall / SelectZeroSmall

Built on top of RankSmall; follow its word type. The broadword constants for `complete_select` variants (`ONES_STEP_13`, `POS_STEP_13`) are already typed as `u64` or `u128` (not `usize`), so they compile unchanged on 32-bit. However, the interaction between these fixed-width broadword operations and the variable-width backing word type needs review: the broadword select operates on values composed from multiple backing words, and the composition logic may assume 64-bit word access.

### 12. EliasFano

This is a substantial refactoring. EliasFano currently uses `usize` for all values (`u`, `Types::Output`, `Types::Input`, `SliceByValue<Value = usize>`). The change to `u64` values affects:

- The struct field `u: usize` → `u: u64`
- `Types::Output<'a> = usize` → `u64`
- `Types::Input = usize` → `u64`
- All `SliceByValue<Value = usize>` bounds → `SliceByValue<Value = u64>`
- Builder methods (`push`, `set`) take `u64` values
- Iterator return types change from `usize` to `u64`
- All succ/pred return types and parameters change
- Low bits default: `BitFieldVec<usize, Box<[usize]>>` → `BitFieldVec<u64, Box<[u64]>>`
- All downstream consumers that pattern-match on or use EliasFano values

The struct becomes:

```rust
pub struct EliasFano<H = BitVec<Box<[usize]>>, L = BitFieldVec<u64, Box<[u64]>>> {
    n: usize,       // number of elements (index-sized)
    u: u64,         // upper bound (always 64-bit)
    l: usize,       // number of low bits
    low_bits: L,    // u64 values in u64-word backing (always)
    high_bits: H,   // platform-word bit vector
}
```

- **High bits:** `BitVec<Box<[usize]>>` with platform-word backing. SelectAdaptConst works natively on top. The high-bits length is ~2n bits, bounded by `usize`.
- **Low bits:** `BitFieldVec<u64, Box<[u64]>>`. Always u64-backed since low-bit widths can reach 63 bits, exceeding what a 32-bit word can hold. This is consistent with EliasFano being a u64-value structure.
- **Values:** `u: u64`, `Types::Output = u64`, `Types::Input = u64`.
- **Indices:** `n: usize`, all position parameters `usize`.
- **Type aliases** (`EfSeq`, `EfDict`, `EfSeqDict`) use `Box<[usize]>` backing.

### 13. RearCodedList / MappedRearCodedList

String compression structures. Pointers are offsets into byte arrays, bounded by `usize`. Follow platform word naturally. No `u64` universe needed.

### 14. VFunc / VBuilder / SignedVFunc / VFilter

- Hash/signature machinery stays `u64` always (128-bit hashes, fixed-point arithmetic, GF(2) solving in `mod2_sys.rs`)
- Stored values in `BitFieldVec<W>` follow the generic word design
- `sig_store.rs` uses `u64` hashes internally — stays `u64`
- Indices `usize`
- Minimal changes beyond what falls out from `BitFieldVec` becoming word-generic

### 15. SelectInWord

Already has `impl_usize!` macro with `cfg(target_pointer_width)` for 16, 32, 64. Ready for 32-bit. Just needs the `compile_error!` gate in `lib.rs` removed.

### 16. AtomicBitVec / AtomicBitFieldVec

Used by `EliasFanoConcurrentBuilder`. Currently backed by `AtomicUsize`. Become generic over the word type, using `AtomicPrimitive` (from the `atomic-primitive` subcrate) to get the corresponding atomic type (`AtomicU32` on 32-bit, `AtomicU64` on 64-bit).

### 17. AddNumBits

The `AddNumBits<B>` wrapper caches the one-count of a bit vector. It delegates `AsRef<[usize]>` to its inner `B`. This delegation must become `AsRef<[W]>` following the same pattern as other structures. The `number_of_ones: usize` field stays `usize` (it's a count, bounded by bit-vector length).

### 18. PartialArray

`PartialArray` uses concrete type aliases like `Rank9<BitVec<Box<[usize]>>>` and `SelectZeroAdaptConst<BitVec<D>, D, 12, 3>`. These must be updated: the Rank9 variant uses `Box<[u64]>`, and the SelectZeroAdaptConst variant uses platform-word backing. The aliases may need cfg-gating on the Rank9 side.

### 19. CLI Binaries

The `cli` feature builds binaries (`rcl`, `mrcl`, `vfunc`, `vfilter`, `mem_usage`). These are primarily useful on 64-bit workstations. They should compile on 32-bit if possible but are not a priority target. Any remaining 64-bit assumptions in CLI code are acceptable.

## Summary Table

| Component | Word Type | Rationale |
|---|---|---|
| `BitVec<B>` | Inferred from `B`, default `PlatformWord` | Platform-native bit vector |
| `BitFieldVec<V, B>` | Inferred from `B` | Same as BitVec |
| `BitVecOps<W>` | Generic `W: Word` | Unlocks all word sizes |
| `AtomicBitVecOps<W>` | Generic `W: Word` | Matches non-atomic counterpart |
| `RankHinted<N>` | Const generic, N = word bit-width | Stable Rust, no nightly |
| `Rank9` | Always `u64` | 512-bit block algorithm |
| `Select9` | Always `u64`, broadword consts → `u64` | Built on Rank9 |
| `RankSmall` | Follows backing word type | Portable rank structure |
| `SelectAdapt` (all 4 variants) | `usize` inventory, packing adapts | 30-bit limit on 32-bit OK |
| `SelectSmall` | Follows RankSmall word type | Review broadword interaction |
| `EliasFano` | `u64` values, `usize` storage | Universe not platform-limited |
| `VFunc` / `VBuilder` | Hash `u64`, values generic | Hashing is algorithm-specific |
| `AddNumBits` | Follows inner type | Wrapper, delegates AsRef |
| `PartialArray` | Update type aliases | Uses Rank9 and SelectZeroAdaptConst |
| Indices | Always `usize` | Natural Rust convention |

## Migration Order

Incremental, bottom-up. Each step is testable on 64-bit where behavior is unchanged:

1. `BitVec` + `BitVecOps` + `AtomicBitVecOps` — make generic over `W: Word`, default `PlatformWord`. Update iterators (`BitIter`, `OnesIter`, `ZerosIter`, `AtomicBitIter`) to be generic with `word: W` fields.
2. `BitFieldVec` — infer word type from backing. Verify that `BitFieldVec<u32>` works correctly on 32-bit. No cross-word-type packing is needed since `V` is both the value type and the storage word type.
3. `RankHinted<N>` — parameterize by word-bit-width. Address delegation cascade in all rank/select structures.
4. `RankSmall` — replace `64` literals with `W::BITS`, cfg-gate `upper_counts` field and `C1` parameter via two struct definitions. Introduce `DefaultRankSmall` type alias to reduce downstream cfg blast radius.
5. `Rank9` / `Select9` — pin to `u64`. Change `BlockCounters` fields to `u64`. Change Select9 broadword constants to `u64`.
6. `SelectAdapt` (all 4 variants) — generalize bit-packing to `usize::BITS`, cfg-gate `SpanType::from_span`, rename `LOG2_U64S_PER_SUBINVENTORY`.
7. `SelectSmall` / `SelectZeroSmall` — review broadword-to-backing-word interaction.
8. `AddNumBits` — update `AsRef` delegation to be word-generic.
9. `EliasFano` — `u: u64`, platform-word high bits, `BitFieldVec<u64>` low bits, update all `Types`/`SliceByValue` bounds. This is the largest single step.
10. `VFunc` / `VBuilder` / `SignedVFunc` / `VFilter` — minimal changes, verify `BitFieldVec` usage.
11. `PartialArray` — update type aliases for Rank9 and SelectZeroAdaptConst.
12. Remove `compile_error!` in `lib.rs` — cross-compile test for `wasm32-unknown-unknown`. Run test suite under `wasm32-wasip1` (or similar WASI target) to verify correctness, not just compilation.

## Risks and Open Questions

- **Ambassador delegation with `AsRef<[W]>`:** Need to verify that `ambassador`'s `#[delegate]` macro handles generic `AsRef<[W]>` propagation. Structures like `SelectAdapt<B, I>` do not currently have a `W` parameter; the word type must be inferred from `B`. If ambassador or orphan rules prevent this, fall back to a custom `AsWordSlice` trait.
- **`RankHinted<N>` delegation cascade:** Every ambassador delegation of `RankHinted<64>` must become parameterized. For word-generic structures this requires a mechanism to derive `N` from the word type without `generic_const_exprs` (nightly). Options: cfg-gated delegation lines, or a macro that expands based on word type.
- **`BitFieldVec` value/word coupling:** `BitFieldVec<V, B>` requires `B: AsRef<[V]>` and enforces `bit_width <= V::BITS`. Values wider than `V` cannot be stored. Structures needing wide values (EliasFano low bits) must use `BitFieldVec<u64, Box<[u64]>>` always.
- **cfg-gated RankSmall struct:** Two struct definitions with different generic parameter counts (5 on 64-bit, 4 on 32-bit). All downstream code naming `RankSmall` with explicit parameters (`impl_rank_small!` macro, `SelectSmall`, `SelectZeroSmall`) must be cfg-gated. Platform-specific type aliases (`DefaultRankSmall`) reduce the blast radius.
- **epserde cross-platform compatibility:** `BlockCounters` has `#[epserde_zero_copy]` and `#[repr(C)]`. Changing fields from `usize` to `u64` preserves layout on 64-bit but means files serialized after the change cannot be read by old code (semver break). Cross-platform deserialization (32-bit reading 64-bit files) requires `epserde` to handle the layout differences. `Block32Counters` also has `epserde_zero_copy` and needs similar analysis.
- **EliasFano scope:** Changing value types from `usize` to `u64` is the largest single refactoring. It touches `Types`, `SliceByValue`, all iterator return types, builders, succ/pred, and downstream consumers. This should be carefully staged.
- **Backward compatibility:** This is a breaking change (semver major bump). The `AsRef<[usize]>` → `AsRef<[W]>` change affects all public APIs. Users who name concrete types with `usize` backing will need to update their code.
- **Testing strategy:** Beyond compilation checks with `wasm32-unknown-unknown`, the full test suite must run on a 32-bit target. `wasm32-wasip1` with `wasmtime` is the recommended approach. Performance benchmarks should be run on WASM to verify that platform-word operations are efficient.
