# 32-bit Portability: Complete Change Review

Branch `32` vs `main` — 21 commits, 38 files, +2198/−1573 lines.

## 1. Design Overview

The branch separates two concerns that were previously conflated through `usize`:

- **`u64`** for structures that inherently require 64-bit words (Rank9, Select9,
  BlockCounters, EliasFano values, broadword constants).
- **`PlatformWord`** (`u64` on 64-bit, `u32` on 32-bit) for structures that
  should adapt to the native word width (bit vector backing, selection
  inventories, SelectSmall/SelectZeroSmall).

The `#[cfg(not(target_pointer_width = "64"))] compile_error!(...)` guard in
`lib.rs` is removed, enabling compilation on 32-bit and WASM targets.

## 2. Trait Changes (Breaking)

### `BitCount` gains type parameter `W`

`BitCount: BitLength` becomes `BitCount<W>: BitLength`. All implementors and
callers must specify the word type.

### `RankHinted` changes from const generic to type parameter

`RankHinted<const HINT_BIT_SIZE: usize>` becomes `RankHinted<W>`. The hint
bit size is now derived from `W::BITS`. This unifies the hint granularity with
the word type (e.g., 32-bit hints for u32 backends).

### `SelectHinted` and `SelectZeroHinted` gain type parameter `W`

`SelectHinted` becomes `SelectHinted<W>`, `SelectZeroHinted` becomes
`SelectZeroHinted<W>`. The impl for `BitVec` now also requires `W: SelectInWord`.

### `BitVecOps` and `BitVecOpsMut` generalized

`BitVecOps: AsRef<[usize]>` becomes `BitVecOps<W: Word>: AsRef<[W]>`. Same for
`BitVecOpsMut`. The blanket impls are correspondingly generalized.

### `AtomicBitFieldSlice` supertrait changed

Supertrait changes from `BitWidth<W::Atomic>` to the new `AtomicBitWidth` trait,
which has method `atomic_bit_width()` instead of `bit_width()`. This avoids a
blanket-impl conflict between `BitWidth<W>` for `[W]` and `BitWidth<A>` for
`[A]`.

### `AddNumBits<B>` gains type parameter `W`

`AddNumBits<B>` becomes `AddNumBits<B, W = PlatformWord>`. The `From<B>` impl
requires `B: BitCount<W>` and disambiguates with
`BitCount::<W>::count_ones(&bits)`. A `PhantomData<W>` field is added.

### Iterator types gain `W` parameter

`BitIter<'a, B>`, `OnesIter<'a, B>`, `ZerosIter<'a, B>` all become
`...<'a, W: Word, B>`.

### Removed constant

`pub const BITS: usize` removed from `bit_vec_ops`. Replaced by local
`W::BITS as usize` computations.

## 3. New Types and Traits

| Item             | Location                  | Purpose                                                         |
| ---------------- | ------------------------- | --------------------------------------------------------------- |
| `PlatformWord`   | `traits::bit_field_slice` | Type alias: `u64` on 64-bit, `u32` on 32-bit                    |
| `AtomicBitWidth` | `traits::bit_field_slice` | Trait for atomic slice bit width (avoids blanket-impl conflict) |

## 4. Core Type Changes

### `BitVec` and `AtomicBitVec` defaults

`BitVec<B = Vec<usize>>` becomes `BitVec<B = Vec<PlatformWord>>`.
`AtomicBitVec<B = Box<[AtomicUsize]>>` becomes
`AtomicBitVec<B = Box<[Atomic<PlatformWord>]>>`.

The `impl BitVec<Vec<usize>>` block (with `new`, `with_value`, `push`, `pop`,
`resize`) is generalized to `impl<W: Word> BitVec<Vec<W>>`, making these
operations available for any word type.

`PartialEq`, `Eq`, `Index<usize>`, `IntoIterator`, `Display` remain
`PlatformWord`-specific (they are for the default backend).

### `BitFieldVec` and `AtomicBitFieldVec` defaults

`BitFieldVec<W = usize>` becomes `BitFieldVec<W = PlatformWord>`. Same for
`AtomicBitFieldVec`.

### `bit_vec!` macro

New arms with explicit word type via colon syntax: `bit_vec![u64: 0, 1, 0, 1]`,
`bit_vec![u64: false; 10]`, `bit_vec![u64]` (empty). The default arms still use
`PlatformWord`.

### `bit_field_vec!` macro

Default word type changed from `usize` to `PlatformWord`.

## 5. Trait Implementation Refactoring

### Macros replaced with generics (`bit_field_slice.rs`)

The per-type macros `impl_bit_width!`, `impl_slice!`, `impl_slice_mut!`,
`impl_bit_width_atomic!`, `impl_atomic!` (which enumerated `u8, u16, u32, u64,
u128, usize`) are all replaced with generic implementations bounded on `W: Word`
or `A: PrimitiveAtomicInteger`. Delegation macros are replaced with
`#[autoimpl]` attributes from `impl_tools`.

Zero performance impact — monomorphization produces the same code.

### RankHinted implemented via macro

`impl_rank_hinted_for_bitvec!($W, $BITS)` generates `RankHinted<$W>` for
`BitVec<B: AsRef<[$W]>>`. Called for `(usize, 64)` and `(u32, 32)`. The mask
computation now uses `wrapping_sub` for correctness at word boundaries.

## 6. Rank/Select Structure Changes

### Rank9 (`rank9.rs`)

- `BlockCounters` fields pinned to `u64` (was `usize`). No layout change on 64-bit.
- All trait bounds use `AsRef<[u64]>` explicitly — Rank9 inherently requires
  64-bit word backing.
- Internal arithmetic uses `u64` with `as usize` casts for indexing.

### RankSmall (`rank_small.rs`)

- Default upper-counts type: `Box<[usize]>` → `Box<[u64]>`.
- `SmallCounters::upper_counts()` returns `&[u64]`.
- New associated constant `WORD_BIT_LOG2`: derives log2 of word bit width from
  `COUNTER_WIDTH` (widths 7/8 → 5 for 32-bit, otherwise 6 for 64-bit).
- `impl_rank_small!` macro gains `$W` parameter for word type.
- Superblock boundary: `words_per_superblock = 1 << (32 - W::BITS.ilog2())`.
- **New u32 variants**: `RankSmall<1,7>` (32-bit words, 7-bit counters, 12.5%
  overhead) and `RankSmall<1,8>` (32-bit words, 8-bit counters, 6.25% overhead).
- New `Block32Counters<1,7>` and `Block32Counters<1,8>` with broadword
  accessors.
- **Bug fix**: `Block32Counters<3,13>::all_rel()` big-endian branch had operator
  precedence bug: `& (1 << 96) - 1` → `& ((1 << 96) - 1)`.
- `rank_small!` macro extended: `rank_small![0, u32; bits]` syntax.

### Select9 (`select9.rs`)

- Inventory default: `Box<[usize]>` → `Box<[u64]>`.
- Broadword constants pinned to `u64` (was `usize`).
- All bounds use `AsRef<[u64]>` — inherently 64-bit.
- `usize::BITS` → literal `64`.

### SelectSmall (`select_small.rs`)

- Inventory default: `Box<[usize]>` → `Box<[PlatformWord]>`.
- Constants made word-independent: `BLOCK_BIT_SIZE = 1 << COUNTER_WIDTH`,
  `SUBBLOCK_BIT_SIZE = BLOCK_BIT_SIZE / (WORDS_PER_BLOCK / WORDS_PER_SUBBLOCK)`.
- `SUPERBLOCK_BIT_SIZE`: `1 << 32` on 64-bit, `usize::MAX` on 32-bit.
- `impl_rank_small_sel!` macro gains `$W` parameter.
- **New u32 `complete_select` implementations**:
  - `SelectSmall<1,7>`: broadword ULEQ with 7-bit step constants
    (`ONES_STEP_7 = (1<<0)|(1<<7)|(1<<14)`), single `select_in_word` on u32.
  - `SelectSmall<1,8>`: broadword ULEQ with 8-bit step constants
    (`ONES_STEP_8 = (1<<0)|(1<<8)|(1<<16)`), delegates to `select_hinted`.
- Existing `complete_select` methods: `PlatformWord` → explicit `u64`.

### SelectAdapt family (`select_adapt.rs`, `select_adapt_const.rs`)

- Inventory entries: `usize` → `PlatformWord`.
- Field rename: `log2_u64_per_subinventory` → `log2_words_per_subinventory`.
- Inventory trait span detection adapts to `PlatformWord::BITS`.
- `SpanType::U64` variant gated `#[cfg(target_pointer_width = "64")]`.
- Length assertion added: `bits.len() <= (PlatformWord::MAX >> 2) as usize`.
- Tests gated `#[cfg(target_pointer_width = "64")]`.

### SelectZeroAdapt family (`select_zero_adapt.rs`, `select_zero_adapt_const.rs`)

Mirror of SelectAdapt changes.

### SelectZeroSmall (`select_zero_small.rs`)

- Inventory default: `Box<[usize]>` → `Box<[PlatformWord]>`.
- `SelectZeroSmall<3,13>` gated `#[cfg(target_pointer_width = "64")]`.
- Superblock upper-count search: overflow-safe `i * SUPERBLOCK_BIT_SIZE - x`
  replaces `(i << 32) - x`.

## 7. EliasFano Changes

The most impactful single-file change (477 lines modified in `elias_fano.rs`).

- **Values**: `usize` → `u64` throughout. Fields `u`, `last_value` in builders
  are `u64`. `upper_bound()` returns `u64`.
- **Low bits backing**: Default `BitFieldVec<usize, Box<[usize]>>` →
  `BitFieldVec<u64, Box<[u64]>>`.
- **High bits backing**: `Box<[usize]>` → `Box<[PlatformWord]>`.
- **Iterator windows**: `usize` → `PlatformWord`.
- **Public API**: `IndexedSeq::get_unchecked()` returns `u64`. Iterator `Item`
  types are `u64`. `Extend<usize>` → `Extend<u64>`.
  `From<A: AsRef<[usize]>>` → `From<A: AsRef<[u64]>>`.

### Cascading changes from EliasFano

| File                        | Change                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `fair_chunks.rs`            | `target_weight`, `max_weight`: `usize` → `u64`. Trait bounds: `SuccUnchecked<Input = u64>`. |
| `partial_array.rs`          | Dense index: `Box<[usize]>` → `Box<[u64]>`. `EliasFanoBuilder::new(n, len as u64)`.         |
| `mapped_rear_coded_list.rs` | `BitFieldVec` → `BitFieldVec<usize>` (explicit).                                            |
| `shard_edge.rs`             | Overflow-safe assertions: `MAX + 1` → `result - 1 <= MAX`.                                  |
| `signed_vfunc.rs`           | `BitFieldVec` → `BitFieldVec<usize>` (explicit).                                            |

## 8. Dependency and Build Changes

| Change                                                            | File                 | Purpose                                      |
| ----------------------------------------------------------------- | -------------------- | -------------------------------------------- |
| `thread-priority` made optional, gated by `rayon`                 | `Cargo.toml`         | Reduce mandatory deps                        |
| `epserde` default-features disabled, explicit `["std", "derive"]` | `Cargo.toml`         | Reduce feature surface                       |
| `criterion` default-features disabled                             | `Cargo.toml`         | Avoid deps that don't compile everywhere     |
| WASM runner configured                                            | `.cargo/config.toml` | `[target.wasm32-wasip1] runner = "wasmtime"` |

## 9. Test Changes

| File                             | Added | Removed | Summary                                                                           |
| -------------------------------- | ----- | ------- | --------------------------------------------------------------------------------- |
| `test_bit_vec.rs`                | 5     | 0       | Word-type parametric tests (u8/u16/u32/u64/u128); `PlatformWord` replaces `usize` |
| `test_bit_field_vec.rs`          | 0     | 0       | `bit_width()` → `atomic_bit_width()`; iterator yields `PlatformWord`              |
| `test_bit_field_slice.rs`        | 0     | 0       | `BitWidth::bit_width` → `AtomicBitWidth::atomic_bit_width` for atomics            |
| `test_elias_fano.rs`             | 0     | 0       | All values `usize` → `u64`                                                        |
| `test_fair_chunks.rs`            | 0     | 0       | All weights `usize` → `u64`                                                       |
| `test_indexed_dict.rs`           | 0     | 0       | EF values `usize` → `u64`                                                         |
| `test_mapped_rear_coded_list.rs` | 0     | 0       | Explicit `BitFieldVec::<usize>`                                                   |
| `test_rank_sel.rs`               | 0     | 0       | `bit_vec![u64: ...]` syntax                                                       |
| `test_rank_small.rs`             | 2     | 0       | u32 variant tests                                                                 |
| `test_select9.rs`                | 0     | 0       | Explicit `BitVec<Vec<u64>>`                                                       |
| `test_select_small.rs`           | 2     | 0       | u32 variant tests                                                                 |
| `test_signed_vfunc.rs`           | 0     | 0       | Explicit `BitFieldVec<usize>`                                                     |
| **Total**                        | **9** | **0**   |                                                                                   |

## 10. Performance Assessment

### On 64-bit platforms

**No performance regression expected.** `PlatformWord = u64`, so all
`as PlatformWord` / `as u64` / `as usize` casts are no-ops after
monomorphization. The generic trait implementations produce the same machine
code as the previous per-type macro expansions.

One minor change worth benchmarking: `BitVec::push()` changed from
`(b as usize) << bit_index` (branchless) to `if b { ... |= W::ONE << ... }`
(conditional). The compiler likely optimizes both to the same code, but this
should be verified.

The `wrapping_sub` in `rank_hinted` mask computation is a correctness
improvement (edge case when `pos % bits == 0`).

### On 32-bit platforms

New functionality — no baseline to compare. The key design decisions:

- Rank9/Select9 retain `u64` backing (wider-than-native memory accesses).
- SelectAdapt family uses 32-bit inventory entries (halves inventory memory).
- EliasFano values are `u64` (architecture-independent).
- SelectAdapt length limited to `PlatformWord::MAX >> 2` (~1 billion bits on
  32-bit).

### New u32 RankSmall/SelectSmall variants

Entirely new hot-path code. The broadword ULEQ operations in `complete_select`
for `<1,7>` and `<1,8>` use the K−1 field convention (3 fields for 4 subblocks):

- `ONES_STEP_7 = (1<<0)|(1<<7)|(1<<14)` — the 4th subblock uses the implicit
  zero-extension field.
- `ONES_STEP_8 = (1<<0)|(1<<8)|(1<<16)` — same convention.

## 11. Potential Concerns

1. **`push()` branchless → conditional**: Verify via benchmark that the compiler
   still produces branchless code for the generic version.

2. **SelectAdapt length limit on 32-bit**: `PlatformWord::MAX >> 2` = ~1 billion
   bits on 32-bit. May be too restrictive for some use cases. Documented via
   runtime assertion.

3. **`usize` vs `u64` type identity on 64-bit**: Code that explicitly used
   `BitVec<Vec<usize>>` in type annotations will fail to compile since the
   default is now `Vec<PlatformWord>` (= `Vec<u64>`, not `Vec<usize>`). The
   two are distinct types in Rust even when they have the same representation.

4. **`AtomicBitVecOps` not generalized**: Remains hardcoded to `PlatformWord`.
   This is intentional (atomic operations always use the platform word), but
   limits testing atomic bit vectors with non-native word types.

WARNING: hash truncation in VFunc!
