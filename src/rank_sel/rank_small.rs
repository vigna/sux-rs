/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use ambassador::{Delegate, delegatable_trait};
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use std::{
    ops::Deref,
    ptr::{addr_of, addr_of_mut, read_unaligned, write_unaligned},
};

use crate::{
    prelude::{BitLength, BitVec, Rank, RankHinted, RankUnchecked, RankZero},
    traits::{
        Backend, BitCount, NumBits, Select, SelectHinted, SelectUnchecked, SelectZero,
        SelectZeroHinted, SelectZeroUnchecked, Word,
    },
};

use crate::ambassador_impl_Index;
use crate::traits::ambassador_impl_Backend;
use crate::traits::bal_paren::{BalParen, ambassador_impl_BalParen};
use crate::traits::bit_vec_ops::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_RankHinted;
use crate::traits::rank_sel::ambassador_impl_Select;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectUnchecked;
use crate::traits::rank_sel::ambassador_impl_SelectZero;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZeroUnchecked;
use std::ops::Index;

/// A trait abstracting the access to the internal counters of a [`RankSmall`]
/// structure.
///
/// This trait is implemented by [`RankSmall`], but it is propagated by
/// [`SelectSmall`] and [`SelectZeroSmall`], making it possible to combine
/// selection structures arbitrarily.
///
/// [`SelectSmall`]: crate::rank_sel::SelectSmall
/// [`SelectZeroSmall`]: crate::rank_sel::SelectZeroSmall
#[delegatable_trait]
pub trait SmallCounters<const NUM_U32S: usize, const COUNTER_WIDTH: usize> {
    fn upper_counts(&self) -> &[u64];
    fn counts(&self) -> &[Block32Counters<NUM_U32S, COUNTER_WIDTH>];
}

/// A family of ranking structures using very little additional space but with
/// slower operations than [`Rank9`].
///
/// [`RankSmall`] structures combine two ideas from [`Rank9`], that is,
/// the interleaving of absolute and relative counters and the storage of
/// implicit counters using zero extension, and a design trick from
/// [`poppy`], that is, that the structures are actually designed around
/// bit vectors of at most 2³² bits. This allows the use of 32-bit
/// counters, which use less space, at the expense of a high-level
/// additional list of 64-bit counters that contain the actual absolute
/// cumulative counts for each block of 2³² bits. Since in most applications
/// these counters will be very few, their additional space is negligible,
/// and they will usually be accessed without cache misses.
///
/// An associated family of selection structures is provided by
/// [`SelectSmall`] and [`SelectZeroSmall`]
///
/// [`Rank9`]: super::Rank9
/// [`SelectSmall`]: crate::rank_sel::SelectSmall
/// [`SelectZeroSmall`]: crate::rank_sel::SelectZeroSmall
///
/// The [`RankSmall`] variants are parameterized by the number of 32-bit words
/// per block and by the size of the relative counters. Only certain
/// combinations are possible, and to simplify construction we provide a
/// [`rank_small`] macro that selects the correct combination.
///
/// [`rank_small`]: crate::rank_small
///
/// The first const generic parameter `WORD_BITS` (32 or 64) specifies the
/// word size; the remaining parameters `NUM_U32S` and `COUNTER_WIDTH` define
/// the counter layout. The same `(NUM_U32S, COUNTER_WIDTH)` pair can be used
/// with both word sizes (with the exception of `(2, 8)` and `(2, 9)`, which
/// are word-size-specific).
///
/// The type parameter `B` is a bit-based [backend]; the remaining
/// type parameter are internal and should always have their default values.
///
/// Presently we support the following combinations for `u64` words:
///
/// - `rank_small![u64: 0; -]` (builds `RankSmall<64, 2, 9>`): 18.75%
///   additional space, speed slightly slower than [`Rank9`].
/// - `rank_small![u64: 1; -]` (builds `RankSmall<64, 1, 9>`): 12.5%
///   additional space.
/// - `rank_small![u64: 2; -]` (builds `RankSmall<64, 1, 10>`): 6.25%
///   additional space.
/// - `rank_small![u64: 3; -]` (builds `RankSmall<64, 1, 11>`): 3.125%
///   additional space.
/// - `rank_small![u64: 4; -]` (builds `RankSmall<64, 3, 13>`): 1.56%
///   additional space.
///
/// And the following for `u32` words:
///
/// - `rank_small![u32: 0; -]` (builds `RankSmall<32, 2, 8>`): 37.5%
///   additional space.
/// - `rank_small![u32: 1; -]` (builds `RankSmall<32, 1, 8>`): 25%
///   additional space.
/// - `rank_small![u32: 2; -]` (builds `RankSmall<32, 1, 9>`): 12.5%
///   additional space; same counter layout as `RankSmall<64, 1, 9>`.
/// - `rank_small![u32: 3; -]` (builds `RankSmall<32, 1, 10>`): 6.25%
///   additional space; same counter layout as `RankSmall<64, 1, 10>`.
/// - `rank_small![u32: 4; -]` (builds `RankSmall<32, 1, 11>`): 3.125%
///   additional space; same counter layout as `RankSmall<64, 1, 11>`.
/// - `rank_small![u32: 5; -]` (builds `RankSmall<32, 3, 13>`): 1.56%
///   additional space; same counter layout as `RankSmall<64, 3, 13>`.
///
/// The word type can be omitted, in which case it defaults to `usize`;
/// the numbering mirrors the `u64` variants (see the [`rank_small`] macro
/// documentation).
///
/// `RankSmall<64, 2, 9>` and `RankSmall<32, 2, 8>` (selector 0) work like
/// [`Rank9`], while the other variants provide increasingly less space
/// usage at the expense of slower operations.
///
/// `RankSmall<64, 1, 11>` and `RankSmall<32, 1, 11>` are similar to
/// [`poppy`], but instead of storing counters and rebuilding cumulative
/// counters on the fly it stores the cumulative counters directly using
/// implicit zero extension, as in [`Rank9`].
///
/// This structure forwards several traits and [`Deref`]'s to its backend.
///
/// # Examples
///
/// ```rust
/// # use sux::{bit_vec,rank_small};
/// # use sux::traits::Rank;
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![0; bits];
///
/// assert_eq!(rank_small.rank(0), 0);
/// assert_eq!(rank_small.rank(1), 1);
/// assert_eq!(rank_small.rank(2), 1);
/// assert_eq!(rank_small.rank(3), 2);
/// assert_eq!(rank_small.rank(4), 3);
/// assert_eq!(rank_small.rank(5), 3);
/// assert_eq!(rank_small.rank(6), 4);
/// assert_eq!(rank_small.rank(7), 4);
/// assert_eq!(rank_small.rank(8), 5);
///
/// // Access to the underlying bit vector is forwarded
/// assert_eq!(rank_small[0], true);
/// assert_eq!(rank_small[1], false);
/// assert_eq!(rank_small[2], true);
/// assert_eq!(rank_small[3], true);
/// assert_eq!(rank_small[4], false);
/// assert_eq!(rank_small[5], true);
/// assert_eq!(rank_small[6], false);
/// assert_eq!(rank_small[7], true);
/// ```
///
/// [`poppy`]: https://link.springer.com/chapter/10.1007/978-3-642-38527-8_15
/// [backend]: Backend
#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::Backend, target = "bits")]
#[delegate(crate::traits::bit_vec_ops::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::Select, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
#[delegate(crate::bal_paren::BalParen, target = "bits")]
pub struct RankSmall<
    const WORD_BITS: usize,
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B = BitVec,
    C1 = Box<[u64]>,
    C2 = Box<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
> {
    pub(super) bits: B,
    pub(super) upper_counts: C1,
    pub(super) counts: C2,
    pub(super) num_ones: usize,
}

impl<
    const WORD_BITS: usize,
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B: Backend + AsRef<[B::Word]>,
    C1,
    C2,
> AsRef<[B::Word]> for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn as_ref(&self) -> &[B::Word] {
        self.bits.as_ref()
    }
}

impl<const WORD_BITS: usize, const NUM_U32S: usize, const COUNTER_WIDTH: usize, B> Deref
    for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B>
{
    type Target = B;

    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

/// A convenient macro to build a [`RankSmall`] structure with the correct
/// parameters.
///
/// The syntax is `rank_small![W: n; bits]`, where `W` is the word type
/// (`u64`, `u32`, or `usize`) and `n` is the variant index. If `W:` is
/// omitted, it defaults to `usize`.
///
/// `u64` variants (0–4):
///
/// - `rank_small![u64: 0; -]` → `RankSmall<64, 2, 9>` (18.75%)
/// - `rank_small![u64: 1; -]` → `RankSmall<64, 1, 9>` (12.5%)
/// - `rank_small![u64: 2; -]` → `RankSmall<64, 1, 10>` (6.25%)
/// - `rank_small![u64: 3; -]` → `RankSmall<64, 1, 11>` (3.125%)
/// - `rank_small![u64: 4; -]` → `RankSmall<64, 3, 13>` (1.56%)
///
/// `u32` variants (0–5):
///
/// - `rank_small![u32: 0; -]` → `RankSmall<32, 2, 8>` (37.5%)
/// - `rank_small![u32: 1; -]` → `RankSmall<32, 1, 8>` (25%)
/// - `rank_small![u32: 2; -]` → `RankSmall<32, 1, 9>` (12.5%)
/// - `rank_small![u32: 3; -]` → `RankSmall<32, 1, 10>` (6.25%)
/// - `rank_small![u32: 4; -]` → `RankSmall<32, 1, 11>` (3.125%)
/// - `rank_small![u32: 5; -]` → `RankSmall<32, 3, 13>` (1.56%)
///
/// Default / `usize` variants — the index represents the rank in the
/// space-usage ordering for the platform's word size (0 = fastest, single
/// popcount). `WORD_BITS` is `usize::BITS`.
///
/// | Index | 64-bit | 32-bit |
/// |-------|--------|--------|
/// | 0 | `<64, 2, 9>` (18.75%) | `<32, 2, 8>` (37.5%) |
/// | 1 | `<64, 1, 9>` (12.5%) | `<32, 1, 8>` (25%) |
/// | 2 | `<64, 1, 10>` (6.25%) | `<32, 1, 9>` (12.5%) |
/// | 3 | `<64, 1, 11>` (3.125%) | `<32, 1, 10>` (6.25%) |
/// | 4 | `<64, 3, 13>` (1.56%) | `<32, 1, 11>` (3.125%) |
/// | 5 | — | `<32, 3, 13>` (1.56%) |
///
/// # Examples
///
/// ```rust
/// # use sux::{prelude::Rank,bit_vec,rank_small};
/// // Explicit word size
/// let bits = bit_vec![u32: 1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![u32: 0; bits];
/// assert_eq!(rank_small.rank(0), 0);
///
/// // Platform word size (works on both 32-bit and 64-bit)
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![0; bits];
/// assert_eq!(rank_small.rank(0), 0);
/// ```
#[macro_export]
macro_rules! rank_small {
    // Explicit u64 variants
    (u64 : 0 ; $bits:expr) => {
        $crate::prelude::RankSmall::<64, 2, 9, _, _, _>::new($bits)
    };
    (u64 : 1 ; $bits:expr) => {
        $crate::prelude::RankSmall::<64, 1, 9, _, _, _>::new($bits)
    };
    (u64 : 2 ; $bits:expr) => {
        $crate::prelude::RankSmall::<64, 1, 10, _, _, _>::new($bits)
    };
    (u64 : 3 ; $bits:expr) => {
        $crate::prelude::RankSmall::<64, 1, 11, _, _, _>::new($bits)
    };
    (u64 : 4 ; $bits:expr) => {
        $crate::prelude::RankSmall::<64, 3, 13, _, _, _>::new($bits)
    };
    // Explicit u32 variants (ordered by decreasing overhead)
    (u32 : 0 ; $bits:expr) => {
        $crate::prelude::RankSmall::<32, 2, 8, _, _, _>::new($bits)
    };
    (u32 : 1 ; $bits:expr) => {
        $crate::prelude::RankSmall::<32, 1, 8, _, _, _>::new($bits)
    };
    (u32 : 2 ; $bits:expr) => {
        $crate::prelude::RankSmall::<32, 1, 9, _, _, _>::new($bits)
    };
    (u32 : 3 ; $bits:expr) => {
        $crate::prelude::RankSmall::<32, 1, 10, _, _, _>::new($bits)
    };
    (u32 : 4 ; $bits:expr) => {
        $crate::prelude::RankSmall::<32, 1, 11, _, _, _>::new($bits)
    };
    (u32 : 5 ; $bits:expr) => {
        $crate::prelude::RankSmall::<32, 3, 13, _, _, _>::new($bits)
    };
    // Default / usize: the index represents the rank in the space-usage
    // ordering for the platform's word size. Index 0 is always the fastest
    // variant (single popcount). WORD_BITS is { usize::BITS as usize }.
    (0 ; $bits:expr) => {
        {
            #[cfg(target_pointer_width = "64")]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 2, 9, _, _, _>::new($bits) }
            #[cfg(not(target_pointer_width = "64"))]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 2, 8, _, _, _>::new($bits) }
        }
    };
    (1 ; $bits:expr) => {
        {
            #[cfg(target_pointer_width = "64")]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 9, _, _, _>::new($bits) }
            #[cfg(not(target_pointer_width = "64"))]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 8, _, _, _>::new($bits) }
        }
    };
    (2 ; $bits:expr) => {
        {
            #[cfg(target_pointer_width = "64")]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 10, _, _, _>::new($bits) }
            #[cfg(not(target_pointer_width = "64"))]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 9, _, _, _>::new($bits) }
        }
    };
    (3 ; $bits:expr) => {
        {
            #[cfg(target_pointer_width = "64")]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 11, _, _, _>::new($bits) }
            #[cfg(not(target_pointer_width = "64"))]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 10, _, _, _>::new($bits) }
        }
    };
    (4 ; $bits:expr) => {
        {
            #[cfg(target_pointer_width = "64")]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 3, 13, _, _, _>::new($bits) }
            #[cfg(not(target_pointer_width = "64"))]
            { $crate::prelude::RankSmall::<{ usize::BITS as usize }, 1, 11, _, _, _>::new($bits) }
        }
    };
    // 32-bit only: one more variant than 64-bit
    (5 ; $bits:expr) => {
        $crate::prelude::RankSmall::<{ usize::BITS as usize }, 3, 13, _, _, _>::new($bits)
    };
    // Explicit usize prefix: forward to the bare-number arms
    (usize : $n:tt ; $bits:expr) => {
        $crate::rank_small![$n ; $bits]
    };
}

#[doc(hidden)]
#[derive(Copy, Debug, Clone, MemSize, MemDbg)]
#[mem_size(flat)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    repr(C),
    epserde(zero_copy)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Block32Counters<const NUM_U32S: usize, const COUNTER_WIDTH: usize> {
    pub(super) absolute: u32,
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub(super) relative: [u32; NUM_U32S],
}

impl Block32Counters<2, 9> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        unsafe { read_unaligned(addr_of!(self.relative) as *const u64) }
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        ((self.all_rel() >> (9 * (word ^ 7))) & ((1 << 9) - 1)) as usize
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        let mut packed = unsafe { read_unaligned(addr_of!(self.relative) as *const u64) };
        packed |= (counter as u64) << (9 * (word ^ 7));
        unsafe { write_unaligned(addr_of_mut!(self.relative) as *mut u64, packed) };
    }
}

impl Block32Counters<1, 9> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.relative[0] as usize >> (9 * (word ^ 3))) & ((1 << 9) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (9 * (word ^ 3));
    }
}

impl Block32Counters<1, 10> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.relative[0] as u64 >> (10 * (word ^ 3))) as usize & ((1 << 10) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (10 * (word ^ 3));
    }
}

impl Block32Counters<1, 11> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.relative[0] as u64 >> (11 * (word ^ 3))) as usize & ((1 << 11) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (11 * (word ^ 3));
    }
}

impl Block32Counters<2, 8> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        unsafe { read_unaligned(addr_of!(self.relative) as *const u64) }
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        ((self.all_rel() >> (8 * (word ^ 7))) & ((1 << 8) - 1)) as usize
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        let mut packed = unsafe { read_unaligned(addr_of!(self.relative) as *const u64) };
        packed |= (counter as u64) << (8 * (word ^ 7));
        unsafe { write_unaligned(addr_of_mut!(self.relative) as *mut u64, packed) };
    }
}

impl Block32Counters<1, 8> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.relative[0] as usize >> (8 * (word ^ 3))) & ((1 << 8) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (8 * (word ^ 3));
    }
}

impl Block32Counters<3, 13> {
    #[inline(always)]
    pub fn all_rel(&self) -> u128 {
        #[cfg(target_endian = "little")]
        unsafe {
            read_unaligned(addr_of!(*self) as *const u128) >> 32
        }
        #[cfg(target_endian = "big")]
        unsafe {
            read_unaligned(addr_of!(*self) as *const u128) & ((1 << 96) - 1)
        }
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        ((self.all_rel() >> (13 * (word ^ 7))) & ((1 << 13) - 1)) as usize
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        let mut packed = self.all_rel();
        packed |= (counter as u128) << (13 * (word ^ 7));

        #[cfg(target_endian = "little")]
        unsafe {
            write_unaligned(
                addr_of_mut!(*self) as *mut u128,
                (packed << 32) | self.absolute as u128,
            )
        };
        #[cfg(target_endian = "big")]
        unsafe {
            write_unaligned(
                addr_of_mut!(*self) as *mut u128,
                packed | (self.absolute as u128) << 96,
            );
        };
    }
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize> Default
    for Block32Counters<NUM_U32S, COUNTER_WIDTH>
{
    fn default() -> Self {
        Self {
            absolute: 0,
            relative: [0; NUM_U32S],
        }
    }
}

impl<const WORD_BITS: usize, const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2>
    RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    /// Log2 of the word bit width for this variant.
    pub(super) const WORD_BIT_LOG2: usize = match WORD_BITS {
        32 => 5,
        64 => 6,
        _ => panic!("Unsupported word size"),
    };
    pub(super) const WORDS_PER_BLOCK: usize = 1 << (COUNTER_WIDTH - Self::WORD_BIT_LOG2);
    pub(super) const WORDS_PER_SUBBLOCK: usize = match NUM_U32S {
        1 => Self::WORDS_PER_BLOCK / 4, // poppy has 4 subblocks
        2 => Self::WORDS_PER_BLOCK / 8, // small rank9 has 8 subblocks
        3 => Self::WORDS_PER_BLOCK / 8, // rank13 has 8 subblocks
        _ => panic!("Unsupported number of u32s"),
    };
}

macro_rules! impl_rank_small {
    ($WORD_BITS: literal; $NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        impl<B: Backend<Word: Word> + AsRef<[B::Word]> + BitLength + RankHinted>
            RankSmall<
                $WORD_BITS,
                $NUM_U32S,
                $COUNTER_WIDTH,
                B,
                Box<[u64]>,
                Box<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            >
        {
            /// Creates a new RankSmall structure from a given bit vector.
            ///
            /// Compile-time panic if `B::Word` does not match the expected word size.
            pub fn new(bits: B) -> Self {
                const {
                    assert!(
                        size_of::<B::Word>() * 8 == $WORD_BITS,
                        concat!(
                            "RankSmall<",
                            stringify!($WORD_BITS),
                            ", ",
                            stringify!($NUM_U32S),
                            ", ",
                            stringify!($COUNTER_WIDTH),
                            "> requires ",
                            stringify!($WORD_BITS),
                            "-bit words"
                        )
                    )
                }
                let bits_per_word = B::Word::BITS as usize;
                let num_bits = bits.len();
                let num_words = num_bits.div_ceil(bits_per_word);
                let num_upper_counts = (num_bits as u64).div_ceil(1u64 << 32) as usize;
                let num_counts = num_bits.div_ceil(bits_per_word * Self::WORDS_PER_BLOCK);

                let mut upper_counts: Vec<u64> = Vec::with_capacity(num_upper_counts);
                let mut counts = Vec::with_capacity(num_counts);

                let mut past_ones: usize = 0;
                let mut upper_count: usize = 0;

                // Superblock boundary: number of words per 2^32 bits
                let words_per_superblock = 1usize << (32 - Self::WORD_BIT_LOG2);

                for i in (0..num_words).step_by(Self::WORDS_PER_BLOCK) {
                    if i % words_per_superblock == 0 {
                        upper_count = past_ones;
                        upper_counts.push(upper_count as u64);
                    }
                    let mut count = Block32Counters::<$NUM_U32S, $COUNTER_WIDTH>::default();
                    count.absolute = (past_ones - upper_count) as u32;
                    past_ones += bits.as_ref()[i].count_ones() as usize;

                    for j in 1..Self::WORDS_PER_BLOCK {
                        #[allow(clippy::modulo_one)]
                        if j % Self::WORDS_PER_SUBBLOCK == 0 {
                            let rel_count = past_ones - upper_count - count.absolute as usize;
                            count.set_rel(j / Self::WORDS_PER_SUBBLOCK, rel_count);
                        }
                        if i + j < num_words {
                            past_ones += bits.as_ref()[i + j].count_ones() as usize;
                        }
                    }

                    counts.push(count);
                }

                assert_eq!(upper_counts.len(), num_upper_counts);
                assert_eq!(counts.len(), num_counts);

                let upper_counts = upper_counts.into_boxed_slice();
                let counts = counts.into_boxed_slice();

                Self {
                    bits,
                    upper_counts,
                    counts,
                    num_ones: past_ones,
                }
            }
        }
        impl<
            B: Backend<Word: Word> + AsRef<[B::Word]> + BitLength + RankHinted,
            C1: AsRef<[u64]>,
            C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
        > RankUnchecked for RankSmall<$WORD_BITS, $NUM_U32S, $COUNTER_WIDTH, B, C1, C2>
        {
            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> usize {
                let bits_per_word = B::Word::BITS as usize;
                debug_assert!(pos < self.bits.len());
                unsafe {
                    let word_pos = pos / bits_per_word;
                    // Prefetch the bit-vector cache line containing word_pos
                    // BEFORE loading counts, so both DRAM fetches can proceed
                    // in parallel (counts and bit vector are independent once
                    // word_pos is known).
                    crate::utils::prefetch_index(self.bits.as_ref(), word_pos);
                    let block = word_pos / Self::WORDS_PER_BLOCK;
                    let offset = (word_pos % Self::WORDS_PER_BLOCK) / Self::WORDS_PER_SUBBLOCK;
                    let counts = self.counts.as_ref().get_unchecked(block);
                    let words_per_superblock = 1usize << (32 - Self::WORD_BIT_LOG2);
                    let upper_count = self
                        .upper_counts
                        .as_ref()
                        .get_unchecked(word_pos / words_per_superblock);

                    let hint_rank =
                        *upper_count as usize + counts.absolute as usize + counts.rel(offset);
                    if Self::WORDS_PER_SUBBLOCK == 1 {
                        // Single-word subblocks: rank directly from the word.
                        let word = *self.bits.as_ref().get_unchecked(word_pos);
                        hint_rank
                            + (word
                                & ((B::Word::ONE << (pos % bits_per_word) as u32) - B::Word::ONE))
                                .count_ones() as usize
                    } else {
                        // Multi-word subblocks: use RankHinted with a compile-time
                        // bound so LLVM can fully unroll the inner popcount loop.
                        // We compute WORDS_PER_SUBBLOCK from the macro literal
                        // parameters to avoid the self-in-anonymous-const limitation
                        // (Self::WORDS_PER_SUBBLOCK cannot be used in turbofish).
                        // TODO: replace with Self::WORDS_PER_SUBBLOCK once Rust
                        // allows associated consts in const generic arguments.
                        const WPS: usize = {
                            let word_bit_log2: usize = match $WORD_BITS {
                                32 => 5,
                                64 => 6,
                                _ => panic!(""),
                            };
                            let words_per_block: usize = 1 << ($COUNTER_WIDTH - word_bit_log2);
                            match $NUM_U32S {
                                1 => words_per_block / 4,
                                2 => words_per_block / 8,
                                3 => words_per_block / 8,
                                _ => panic!(""),
                            }
                        };
                        #[allow(clippy::modulo_one)]
                        let hint_pos = word_pos
                            - ((word_pos % Self::WORDS_PER_BLOCK) % Self::WORDS_PER_SUBBLOCK);

                        RankHinted::rank_hinted::<WPS>(&self.bits, pos, hint_pos, hint_rank)
                    }
                }
            }
        }
    };
}

// 64-bit word variants
impl_rank_small!(64; 2; 9);
impl_rank_small!(64; 1; 9);
impl_rank_small!(64; 1; 10);
impl_rank_small!(64; 1; 11);
impl_rank_small!(64; 3; 13);

// 32-bit word variants
impl_rank_small!(32; 2; 8);
impl_rank_small!(32; 1; 8);
impl_rank_small!(32; 1; 9);
impl_rank_small!(32; 1; 10);
impl_rank_small!(32; 1; 11);
impl_rank_small!(32; 3; 13);

impl<const WORD_BITS: usize, const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2> Rank
    for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
where
    Self: BitLength + NumBits + RankUnchecked,
{
}

impl<const WORD_BITS: usize, const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2> RankZero
    for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
where
    Self: Rank,
{
}

impl<
    const WORD_BITS: usize,
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B: BitLength,
    C1,
    C2,
> RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    /// Returns the number of bits in the underlying bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<const WORD_BITS: usize, const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2>
    RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    /// Returns the underlying bit vector, consuming this structure.
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Replaces the backend with a new one.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new backend is identical to the old one as a bit vector.
    pub unsafe fn map<B1: BitLength>(
        self,
        f: impl FnOnce(B) -> B1,
    ) -> RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B1, C1, C2> {
        RankSmall {
            bits: f(self.bits),
            upper_counts: self.upper_counts,
            counts: self.counts,
            num_ones: self.num_ones,
        }
    }
}

impl<
    const WORD_BITS: usize,
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B: BitLength,
    C1,
    C2,
> NumBits for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn num_ones(&self) -> usize {
        self.num_ones
    }
}

impl<
    const WORD_BITS: usize,
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B: BitLength,
    C1,
    C2,
> BitCount for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones
    }
}

impl<
    const WORD_BITS: usize,
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B,
    C1: AsRef<[u64]>,
    C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
> SmallCounters<NUM_U32S, COUNTER_WIDTH>
    for RankSmall<WORD_BITS, NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn upper_counts(&self) -> &[u64] {
        self.upper_counts.as_ref()
    }

    #[inline(always)]
    fn counts(&self) -> &[Block32Counters<NUM_U32S, COUNTER_WIDTH>] {
        self.counts.as_ref()
    }
}
