# WordType Associated Type Refactor

## Problem

Many traits in sux-rs carry a phantom type parameter `W` (the word type) that
never appears in method signatures:

- `BitCount<W>` — `count_ones(&self) -> usize`
- `RankHinted<W>` — `rank_hinted(&self, pos, hint_pos, hint_rank) -> usize`
- `SelectHinted<W>` — `select_hinted(&self, rank, hint_pos, hint_rank) -> usize`
- `SelectZeroHinted<W>` — same pattern

`W` exists to link the word type through the composition chain (e.g.,
`SelectAdapt<W, Rank9<BitVec<Vec<u64>>>, I>` ensures the word type is
consistent). But this causes problems:

1. **Structs must carry `W`**: `SelectAdapt<W, B, I>`, `SelectAdaptConst<W, B, I, ...>`,
   `SelectZeroAdapt<W, B, I>`, etc. — `W` is stored only in `PhantomData`.
2. **Consumer complexity**: Users must write `SelectAdapt::<u64, _>` or
   `SelectAdapt::<PlatformWord, _>` explicitly.
3. **Conflicting impls**: On a generic struct like `SelectAdapt<W, B, I>`, you
   cannot add `usize` delegations alongside `W` delegations — when `W = usize`
   they conflict.
4. **Hardcoded duplications**: `AddNumBits` delegates `RankHinted<u32>` and
   `RankHinted<u64>` separately. `RankSmall` does the same. Each new word type
   requires more delegation lines.
5. **No code sharing**: Rank9 (64-bit) and a hypothetical Rank8 (32-bit) cannot
   share code because the word type is baked in, not parameterized through the
   type being ranked.

## Solution: `WordType` Associated Type

Replace `W` parameters on traits with a single associated type trait:

```rust
pub trait WordType {
    type Word;
}
```

### Design principles

- **Only backends implement `WordType` concretely.** `BitVec<B>` determines
  `Word` from its backing store. Every other struct (`Rank9`, `SelectAdapt`,
  `AddNumBits`, etc.) delegates `WordType` to its inner field.

- **Traits lose the `W` parameter.** `BitCount<W>` becomes `BitCount`,
  `RankHinted<W>` becomes `RankHinted`, etc. Method signatures are unchanged
  (they already use `usize` everywhere).

- **`AsRef<[W]>` remains separate.** We considered bundling
  `WordType: AsRef<[Self::Word]>` but it doesn't save delegation work — `AsRef`
  must be implemented/delegated at each layer anyway. Keep them orthogonal.

- **`W` is recovered via `B::Word`.** Where bounds currently say
  `B: AsRef<[W]> + BitCount<W> + SelectHinted<W>`, they become
  `B: WordType + AsRef<[B::Word]> + BitCount + SelectHinted`.

### Example: before and after

**Before:**
```rust
#[delegate(AsRef<[W]>, target = "bits")]
#[delegate(crate::traits::rank_sel::BitCount<W>, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted<W>, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted<W>, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted<W>, target = "bits")]
pub struct SelectAdapt<W, B, I = Box<[usize]>> {
    bits: B,
    inventory: I,
    spill: I,
    _phantom: PhantomData<W>,
    // ...
}
```

**After:**
```rust
#[delegate(WordType, target = "bits")]
#[delegate(crate::traits::rank_sel::BitCount, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
pub struct SelectAdapt<B, I = Box<[usize]>> {
    bits: B,
    inventory: I,
    spill: I,
    // ... (no PhantomData<W>)
}

// AsRef must be forwarded manually (ambassador can't resolve B::Word)
impl<B: WordType, I> AsRef<[B::Word]> for SelectAdapt<B, I>
where B: AsRef<[B::Word]> {
    fn as_ref(&self) -> &[B::Word] { self.bits.as_ref() }
}

// Impl uses B::Word where W was before
impl<B, I> SelectUnchecked for SelectAdapt<B, I>
where
    B: WordType + AsRef<[B::Word]> + SelectHinted + NumBits,
    B::Word: Word + SelectInWord,
    I: AsRef<[usize]>,
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let words: &[B::Word] = self.bits.as_ref();
        // ... same algorithm, B::Word replaces W
    }
}
```

### Why this enables code sharing

Structures like Rank9 are currently hardwired to u64. With `WordType`, a unified
`Rank` struct (parameterized by const generics like `RankSmall`) could work with
any word type:

- `Rank<BitVec<Vec<u64>>>` → `WordType::Word = u64` → 64-bit block layout
- `Rank<BitVec<Vec<u32>>>` → `WordType::Word = u32` → 32-bit block layout

The const parameters control counter sizes/geometry; `B::Word` controls the word
type. The two dimensions are orthogonal. This eliminates the need for separate
Rank9/Rank8 implementations.

## Current inventory

### Traits with `W` parameter (to be modified)

| Trait | File | `W` in signatures? |
|---|---|---|
| `BitCount<W>` | `src/traits/rank_sel.rs:49` | No |
| `RankHinted<W>` | `src/traits/rank_sel.rs:201` | No |
| `SelectHinted<W>` | `src/traits/rank_sel.rs:287` | No |
| `SelectZeroHinted<W>` | `src/traits/rank_sel.rs:312` | No |

Note: `BitVecOps<W>` and `BitFieldSlice<W>` also have `W`, but `W` appears in
their method signatures (they return/accept `W` values), so they are unaffected
by this refactor.

### Structs with `W` type parameter (to lose `W`)

| Struct | File | `W` stored? |
|---|---|---|
| `SelectAdapt<W, B, I>` | `src/rank_sel/select_adapt.rs:244` | PhantomData |
| `SelectAdaptConst<W, B, I, ...>` | `src/rank_sel/select_adapt_const.rs:177` | PhantomData |
| `SelectZeroAdapt<W, B, I>` | `src/rank_sel/select_zero_adapt.rs:160` | PhantomData |
| `SelectZeroAdaptConst<W, B, I, ...>` | `src/rank_sel/select_zero_adapt_const.rs:159` | PhantomData |
| `SelectSmall<..., W, C, ...>` | `src/rank_sel/select_small.rs:125` | PhantomData |
| `SelectZeroSmall<..., W, C, ...>` | `src/rank_sel/select_zero_small.rs:91` | PhantomData |
| `AddNumBits<B, W>` | `src/traits/rank_sel.rs:354` | PhantomData |

### Structs with hardcoded u64/u32 delegations (to use WordType delegation)

| Struct | File | Hardcoded types |
|---|---|---|
| `Rank9<B, C>` | `src/rank_sel/rank9.rs:94` | u64 |
| `Select9<R, I>` | `src/rank_sel/select9.rs:135` | u64 |
| `RankSmall<..., B, C>` | `src/rank_sel/rank_small.rs` | u64, u32 |

### Leaf implementation (source of truth)

```rust
// BitVec: the only concrete WordType implementation
impl<W: Word, B: AsRef<[W]>> WordType for BitVec<B> {
    type Word = W;
}
```

All other structs delegate `WordType` to their inner field.

## Implementation plan

### Phase 1: Add `WordType` trait and implement on backends

1. Define `WordType` trait in `src/traits/rank_sel.rs` (or a new
   `src/traits/word_type.rs`).
2. Make it `#[delegatable_trait]` for ambassador.
3. Implement on `BitVec<B>` and `AtomicBitVec<B>`.
4. **No breaking changes** — everything else is additive.

### Phase 2: Remove `W` from the four phantom-only traits

Modify in `src/traits/rank_sel.rs`:
- `BitCount<W>` → `BitCount`
- `RankHinted<W>` → `RankHinted`
- `SelectHinted<W>` → `SelectHinted`
- `SelectZeroHinted<W>` → `SelectZeroHinted`

Update all implementations:
- `BitVec`: `impl<W: Word, B: AsRef<[W]>> BitCount<W>` → `impl<W: Word, B: AsRef<[W]>> BitCount`
  (the impl still needs `W` for internal computation, just not on the trait)
- All delegations change from `#[delegate(BitCount<W>)]` to `#[delegate(BitCount)]`
- Hardcoded delegations like `#[delegate(RankHinted<u64>)]` become just
  `#[delegate(RankHinted)]`

**This is the big breaking change** — every file that references these traits
must be updated. But it's mechanical: remove the type parameter everywhere.

### Phase 3: Remove `W` from structs

For each struct in the "Structs with W" table:
1. Remove `W` type parameter and `PhantomData<W>`.
2. Add `#[delegate(WordType, target = "...")]`.
3. Add manual `AsRef<[B::Word]>` forwarding impl.
4. Replace `W` in `where` bounds with `B::Word` (or `<B as WordType>::Word`).
5. Update all consumer code (tests, benches, bins, dict/, etc.).

### Phase 4: Simplify `AddNumBits`

`AddNumBits<B, W>` becomes `AddNumBits<B>`:
- Remove `W` parameter and `PhantomData<W>`.
- Delegate `WordType` to `bits`.
- The current hardcoded u32/u64 delegations for `RankHinted`, `SelectHinted`,
  etc. become a single unparameterized delegation each.
- `BitCount` impl uses `B::Word` instead of `W`.

### Phase 5: Unify Rank9/Rank8 (future)

With `WordType` in place, Rank9 can be generalized:
- Replace `BlockCounters` (64-bit specific) with a const-parameterized counter
  type (like `RankSmall` already does with `Block32Counters`).
- The struct delegates `WordType` and uses `B::Word` in its algorithms.
- Rank9 (u64) and Rank8 (u32) become the same struct with different const
  parameters and `B::Word` constraints.

## Consumer impact

- `SelectAdapt::<u64, _>::new(bits, 3)` → `SelectAdapt::new(bits, 3)` (W inferred from bits)
- `SelectAdapt::<PlatformWord, _>` → `SelectAdapt` (no W needed)
- `AddNumBits<_, u64>` → `AddNumBits<_>` (W inferred from inner)
- Trait bounds like `B: RankHinted<u64>` → `B: RankHinted`

Overall: simpler types, less explicit parameterization, no conflicting-impl
issues, and a path to code sharing between word-size variants.
