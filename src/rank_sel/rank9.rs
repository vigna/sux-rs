/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use ambassador::Delegate;
use epserde::*;
use mem_dbg::*;

use crate::ambassador_impl_AsRef;
use crate::ambassador_impl_Index;
use crate::traits::rank_sel::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_RankHinted;
use crate::traits::rank_sel::ambassador_impl_Select;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectUnchecked;
use crate::traits::rank_sel::ambassador_impl_SelectZero;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZeroUnchecked;
use std::ops::Index;

/// A ranking structure using 25% of additional space and providing the fastest
/// available rank operations.
///
/// `Rank9` stores 64-bit absolute cumulative counters for 512-bit blocks, and
/// relative cumulative 9-bit counters for each 64-bit word in a block. The
/// first relative counter is stored implicitly using zero extension, so eight
/// 9-bit counters can be stored in just 64 bits. Moreover, absolute and
/// relative counters are interleaved. These two ideas make it possible to rank
/// using a most two cache misses and no tests or loops.
///
/// This structure has been described by Sebastiano Vigna in “[Broadword
/// Implementation of Rank/Select
/// Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
/// _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
/// 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
/// Springer, 2008.
///
/// # Examples
///
/// ```rust
/// use sux::bit_vec;
/// use sux::prelude::{Rank, Rank9};
///
/// let rank9 = Rank9::new(bit_vec![1, 0, 1, 1, 0, 1, 0, 1]);
/// assert_eq!(rank9.rank(0), 0);
/// assert_eq!(rank9.rank(1), 1);
/// assert_eq!(rank9.rank(2), 1);
/// assert_eq!(rank9.rank(3), 2);
/// assert_eq!(rank9.rank(4), 3);
/// assert_eq!(rank9.rank(5), 3);
/// assert_eq!(rank9.rank(6), 4);
/// assert_eq!(rank9.rank(7), 4);
/// assert_eq!(rank9.rank(8), 5);
///
/// // Access to the underlying bit vector is forwarded
/// assert_eq!(rank9[0], true);
/// assert_eq!(rank9[1], false);
/// assert_eq!(rank9[2], true);
/// assert_eq!(rank9[3], true);
/// assert_eq!(rank9[4], false);
/// assert_eq!(rank9[5], true);
/// assert_eq!(rank9[6], false);
/// assert_eq!(rank9[7], true);
/// ```

#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Delegate)]
#[delegate(AsRef<[usize]>, target = "bits")]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::rank_sel::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted<64>, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "bits")]
#[delegate(
    crate::traits::rank_sel::Select,
    target = "bits",
    where = "C: AsRef<[BlockCounters]>"
)]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "bits")]
#[delegate(
    crate::traits::rank_sel::SelectZero,
    target = "bits",
    where = "C: AsRef<[BlockCounters]>"
)]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
pub struct Rank9<B = BitVec, C = Box<[BlockCounters]>> {
    pub(super) bits: B,
    pub(super) counts: C,
}

#[doc(hidden)]
#[derive(Epserde, Copy, Debug, Clone, MemDbg, MemSize, Default)]
#[repr(C)]
#[zero_copy]
pub struct BlockCounters {
    pub(super) absolute: usize,
    pub(super) relative: usize,
}

impl BlockCounters {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.relative >> (9 * (word ^ 7))) & 0x1FF
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative |= counter << (9 * (word ^ 7));
    }
}

impl<B, C> Rank9<B, C> {
    pub(super) const WORDS_PER_BLOCK: usize = 8;

    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Replaces the backend with a new one.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new backend is identical to the old one as a bit vector.
    pub unsafe fn map<B1>(self, f: impl FnOnce(B) -> B1) -> Rank9<B1, C>
    where
        B1: AsRef<[usize]> + BitLength,
    {
        Rank9 {
            bits: f(self.bits),
            counts: self.counts,
        }
    }
}

impl<B: BitLength, C> Rank9<B, C> {
    /// Returns the number of bits in the underlying bit vector.
    ///
    /// This method is equivalent to
    /// [`BitLength::len`](crate::traits::BitLength::len), but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<B: AsRef<[usize]> + BitLength> Rank9<B, Box<[BlockCounters]>> {
    /// Creates a new Rank9 structure from a given bit vector.
    pub fn new(bits: B) -> Self {
        let num_bits = bits.len();
        let num_words = num_bits.div_ceil(usize::BITS as usize);
        let num_counts = num_bits.div_ceil(usize::BITS as usize * Self::WORDS_PER_BLOCK);

        // We use the last counter to store the total number of ones
        let mut counts = Vec::with_capacity(num_counts + 1);

        let mut num_ones = 0;

        for i in (0..num_words).step_by(Self::WORDS_PER_BLOCK) {
            let mut count = BlockCounters {
                absolute: num_ones,
                relative: 0,
            };
            num_ones += bits.as_ref()[i].count_ones() as usize;

            for j in 1..8 {
                let rel_count = num_ones - count.absolute;
                count.set_rel(j, rel_count);
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as usize;
                }
            }

            counts.push(count);
        }

        counts.push(BlockCounters {
            absolute: num_ones,
            relative: 0,
        });

        Self {
            bits,
            counts: counts.into(),
        }
    }
}

impl<B: BitLength, C: AsRef<[BlockCounters]>> NumBits for Rank9<B, C> {
    #[inline(always)]
    fn num_ones(&self) -> usize {
        // SAFETY: The last counter is always present
        unsafe { self.counts.as_ref().last().unwrap_unchecked().absolute }
    }
}

impl<B: BitLength, C: AsRef<[BlockCounters]>> BitCount for Rank9<B, C> {
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones()
    }
}

impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>> RankUnchecked for Rank9<B, C> {
    /// # Safety
    ///
    /// The implementation of [`RankUnchecked`] for [`Rank9`] has an weakened
    /// safety requirement: it is possible to call this method with `pos` equal
    /// to [the length of te underlying bit
    /// vector](crate::traits::BitLength::len) provided there is at least one
    /// unused bit. This is always true if the length is not a multiple of the
    /// number of bits in a word, but requires allocating an extra word
    /// otherwise. The content of the extra word is irrelevant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sux::prelude::*;
    ///
    /// let rank9 = Rank9::new(bit_vec![1, 0, 1, 1, 0, 1, 0, 1]);
    /// // This is safe as there are several unused bits
    /// assert_eq!(unsafe { rank9.rank_unchecked(8) }, rank9.num_ones());
    ///
    /// // The same call would not be legal where
    /// let rank9 = Rank9::new(BitVec::new(usize::BITS as usize));
    /// // But we can ensure that an unused bit is present
    /// let mut bv = BitVec::new(usize::BITS as usize);
    /// bv.push(false);
    /// bv.pop();
    /// let rank9 = Rank9::new(bv);
    /// // This is safe as there is at least one unused bit
    /// assert_eq!(unsafe { rank9.rank_unchecked(8) }, rank9.num_ones());
    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word_pos = pos / usize::BITS as usize;
        let bit_pos = pos % usize::BITS as usize;
        let block = word_pos / Self::WORDS_PER_BLOCK;
        let offset = word_pos % Self::WORDS_PER_BLOCK;
        // When pos is equal to the length of the underlying bit vector and
        // there is at least one unused bit, this access is safe as there
        // is a word of index word_pos.
        let word = self.bits.as_ref().get_unchecked(word_pos);
        let counts = self.counts.as_ref().get_unchecked(block);

        counts.absolute + counts.rel(offset) + (word & ((1 << bit_pos) - 1)).count_ones() as usize
    }
}

impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>> Rank for Rank9<B, C> {}
impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>> RankZero for Rank9<B, C> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::traits::BitCount;
    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 10], (1 << 10) * 64) };

        let rank9: Rank9 = Rank9::new(bits);

        assert_eq!(rank9.rank(rank9.len()), rank9.bits.count_ones());
    }
}
