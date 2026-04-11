/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! An implementation of Jacobson's balanced parentheses data structure.
//!
//! See [`JacobsonBalParen`] for details.
//!
//! The in-word [`find_near_close`] and far-match [`find_far_close`] functions
//! are also available as standalone public functions for use in other contexts.
//!
//! # References
//!
//! Guy Jacobson. [Space-efficient static trees and graphs]. In *30th annual
//! symposium on foundations of computer science (FOCS '89)*, pp. 549−554. IEEE,
//! 1989.
//!
//! [Space-efficient static trees and graphs]: https://ieeexplore.ieee.org/abstract/document/63533

use std::ops::Index;

use crate::ambassador_impl_Index;
use crate::traits::ambassador_impl_Backend;
use crate::traits::bit_vec_ops::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_BitCount;
use crate::traits::rank_sel::ambassador_impl_NumBits;
use crate::traits::rank_sel::ambassador_impl_Rank;
use crate::traits::rank_sel::ambassador_impl_RankHinted;
use crate::traits::rank_sel::ambassador_impl_RankUnchecked;
use crate::traits::rank_sel::ambassador_impl_RankZero;
use crate::traits::rank_sel::ambassador_impl_Select;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectUnchecked;
use crate::traits::rank_sel::ambassador_impl_SelectZero;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZeroUnchecked;

use crate::bits::{BitFieldVec, BitVec};
use crate::dict::{EfDict, EliasFanoBuilder};
use crate::list::comp_int_list::CompIntList;
use crate::list::prefix_sum_int_list::PrefixSumIntList;
use crate::prelude::{
    BitCount, NumBits, Rank, RankHinted, RankUnchecked, RankZero, Select, SelectHinted,
    SelectUnchecked, SelectZero, SelectZeroHinted, SelectZeroUnchecked,
};
use crate::traits::bal_paren::BalParen;
use crate::traits::indexed_dict::PredUnchecked;
use ambassador::Delegate;
use mem_dbg::*;
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// Number of bits in a `usize` word.
const WORD_BITS: usize = usize::BITS as usize;

/// For each byte value *b* (interpreted as 8 balanced-parentheses bits, LSB
/// first, 1 = open, 0 = close), the minimum running excess encountered when
/// scanning the 8 bits with initial excess 0.
static BYTE_MIN_EXCESS: [i8; 256] = {
    let mut table = [0i8; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut excess = 0i8;
        let mut min_e = 0i8;
        let mut i = 0;
        while i < 8 {
            if b & (1 << i) != 0 {
                excess += 1;
            } else {
                excess -= 1;
            }
            if excess < min_e {
                min_e = excess;
            }
            i += 1;
        }
        table[b] = min_e;
        b += 1;
    }
    table
};

/// For each byte value *b* and target excess *t* (0 . . 8), the bit position (0
/// . . 7) within the byte where the running excess first reaches −(*t* + 1), or
/// 8 if the target is not reached within the byte.
static BYTE_FIND_CLOSE: [[u8; 8]; 256] = {
    let mut table = [[8u8; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut excess = 0i8;
        let mut i = 0u8;
        while i < 8 {
            if b & (1 << i) != 0 {
                excess += 1;
            } else {
                excess -= 1;
            }
            // excess is -(target+1) when target = -excess - 1
            if excess < 0 {
                let target = (-excess - 1) as usize;
                if target < 8 && table[b][target] == 8 {
                    table[b][target] = i;
                }
            }
            i += 1;
        }
        b += 1;
    }
    table
};

/// Count the number of far open parentheses scanning `l` bits of `word`
/// from MSB to LSB.
///
/// A "far open" parenthesis is an open parenthesis (1-bit) that, when scanning
/// from MSB to LSB, causes the excess to become positive.
fn count_far_open(word: usize, l: usize) -> usize {
    let mut c = 0usize;
    let mut e = 0i32;
    for i in (0..l).rev() {
        if word & (1usize << i) != 0 {
            e += 1;
            if e > 0 {
                c += 1;
            }
        } else if e > 0 {
            e = -1;
        } else {
            e -= 1;
        }
    }
    c
}

/// Count the number of far close parentheses scanning `l` bits of `word`
/// from LSB to MSB.
fn count_far_close(word: usize, l: usize) -> usize {
    let mut c = 0usize;
    let mut e = 0i32;
    for i in 0..l {
        if word & (1usize << i) != 0 {
            if e > 0 {
                e = -1;
            } else {
                e -= 1;
            }
        } else {
            e += 1;
            if e > 0 {
                c += 1;
            }
        }
    }
    c
}

/// Finds the position of the matching close parenthesis within a single
/// `usize` word, using byte-level lookup tables.
///
/// Bit 0 of `word` must be the open parenthesis (1-bit) whose match is
/// sought. The function shifts right by 1 internally and scans byte by
/// byte with an initial excess of 1.
///
/// Returns the bit position of the matching close, or a value ≥ `usize::BITS`
/// if the match is not in this word.
#[inline]
pub fn find_near_close(word: usize) -> usize {
    // The input word has the open paren at bit 0. We shift right by 1 to
    // skip it, so bit 0 of `scan_word` is the first bit after the open.
    // The initial excess is 1 (from the consumed open paren).
    let scan_word = word >> 1;
    let bytes = scan_word.to_le_bytes();
    let mut excess: i8 = 1; // starting excess (the open paren we consumed)
    for (i, &b) in bytes.iter().enumerate() {
        let min_e = BYTE_MIN_EXCESS[b as usize];
        // Check if excess can drop to 0 in this byte:
        // excess + min_e <= 0
        if excess + min_e <= 0 {
            // The match is in this byte. We want the first position where
            // byte running excess = -excess, i.e., drops by `excess`.
            // Table index: excess - 1 (0-based: target 0 means drop by 1).
            let target = (excess - 1) as usize;
            debug_assert!(target < 8);
            let bit_in_byte = BYTE_FIND_CLOSE[b as usize][target] as usize;
            return i * 8 + bit_in_byte + 1; // +1 for the original shift
        }
        // Update excess: byte excess delta = 2*popcount(b) - 8
        excess += 2 * b.count_ones() as i8 - 8;
    }
    // Not found in this word
    WORD_BITS
}

/// Finds the *k*-th (0-based) far close parenthesis in a `usize` word using
/// byte-level lookup tables.
///
/// A *far close* parenthesis is an unmatched close parenthesis (0-bit)
/// when scanning from LSB to MSB. The *k*-th far close is at the first
/// position where the running excess (open = +1, close = −1, starting
/// at 0) reaches −(*k* + 1).
///
/// This uses the same `BYTE_MIN_EXCESS` and `BYTE_FIND_CLOSE` tables
/// as [`find_near_close`].
#[inline]
pub fn find_far_close(word: usize, k: i64) -> usize {
    let bytes = word.to_le_bytes();
    let target = -(k + 1) as i32; // target global excess
    let mut excess: i32 = 0; // cumulative excess at byte boundary

    for (i, &b) in bytes.iter().enumerate() {
        let min_e = BYTE_MIN_EXCESS[b as usize] as i32;
        // Check if the target excess is reachable in this byte
        if excess + min_e <= target {
            // The table index: how many more excess units we need to drop
            // within this byte. Since excess + local_excess = target, and
            // BYTE_FIND_CLOSE[b][t] gives where local excess = -(t+1),
            // we need t = k + excess (guaranteed to be in 0..7).
            let t = (k as i32 + excess) as usize;
            debug_assert!(t < 8);
            return i * 8 + BYTE_FIND_CLOSE[b as usize][t] as usize;
        }
        // Update cumulative excess: 2*popcount - 8
        excess += 2 * b.count_ones() as i32 - 8;
    }
    panic!("find_far_close: k={k} not found in word {word:#x}")
}

/// A balanced parentheses structure supporting
/// [`find_close`] queries, based on Jacobson's pioneer technique.
///
/// [`find_close`]: Self::find_close
///
/// Open parentheses are represented as 1-bits and close parentheses as
/// 0-bits, with bit 0 being the LSB.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access. Due to
/// genericity of all the type parameter involved, it is not possible to provide
/// a [`From`] implementation for the unaligned version, but the conversion can
/// be done using the map methods.
///
/// # Implementation details
///
/// This implementation uses the pioneer technique from Jacobson: an opening
/// parenthesis whose match falls in a different `usize` word is called *far*
/// (the original paper uses blocks that are logarithmic in the number of
/// parentheses).
///
/// Among the far opening parentheses, a subset called *pioneers* is selected: a
/// far opening parenthesis is a pioneer if it is the first far opening
/// parenthesis in its word, or if its match falls in a different word than
/// the previous far opening parenthesis's match. Pioneer positions are
/// stored in a structure supporting [predecessor queries], and the offsets
/// from each pioneer to its matching close parenthesis are stored in a
/// [`SliceByValue`] structure. Three offset storage variants are available:
///
/// [predecessor queries]: crate::traits::indexed_dict::PredUnchecked
///
/// - [`CompIntList`] (default, best space);
/// - [`PrefixSumIntList`] (faster queries, more space);
/// - [`BitFieldVec`] (fastest queries, largest space).
///
/// Both structures can be replaced with custom implementations as long as they
/// return the same values, using [`map_pioneer_positions`] and
/// [`map_pioneer_match_offsets`].
///
/// [`map_pioneer_positions`]: Self::map_pioneer_positions
/// [`map_pioneer_match_offsets`]: Self::map_pioneer_match_offsets
///
/// Queries work in two stages:
/// 1. **In-word**: byte-level lookup tables are used to find the matching close
///    parenthesis within the same `usize` word (the common case).
/// 2. **Far match**: if the match is in a different word, a predecessor query
///    on the pioneer positions locates the relevant pioneer, whose stored match
///    offset is then adjusted using lookup tables.
///
/// # Type Parameters
///
/// - `B`: The balanced parentheses bit vector. Must implement `AsRef<[usize]>` and
///   `BitLength`.
///
/// - `P`: The predecessor structure for pioneer positions. Must implement
///   [`PredUnchecked<Input = usize, Output<'_> = usize>`].
///   Defaults to [`EfDict<usize>`].
///
/// - `O`: The storage for pioneer match offsets. Must implement
///   [`SliceByValue<Value = usize>`]. Defaults to [`CompIntList`]
///   for compact variable-length encoding. See the
///   [`new_with_bit_field_vec`] and [`new_with_prefix_sum`] constructors
///   for alternative configurations.
///
/// [`PredUnchecked<Input = usize, Output<'_> = usize>`]: PredUnchecked
/// [`SliceByValue<Value = usize>`]: SliceByValue
/// [`new_with_bit_field_vec`]: Self::new_with_bit_field_vec
/// [`new_with_prefix_sum`]: Self::new_with_prefix_sum
///
/// # Examples
///
/// ```rust
/// # use sux::bal_paren::JacobsonBalParen;
/// # use sux::bit_vec;
/// # use sux::prelude::BalParen;
/// // Sequence "(()())" = bits 1,1,0,1,0,0
/// let bp = JacobsonBalParen::new(bit_vec![1, 1, 0, 1, 0, 0]);
/// assert_eq!(bp.find_close(0), Some(5)); // outermost pair
/// assert_eq!(bp.find_close(1), Some(2)); // first inner pair
/// assert_eq!(bp.find_close(3), Some(4)); // second inner pair
/// ```
///
/// # References
///
/// Guy Jacobson. [Space-efficient static trees and graphs]. In *30th annual
/// symposium on foundations of computer science (FOCS '89)*, pp. 549−554. IEEE,
/// 1989.
///
/// [Space-efficient static trees and graphs]: https://ieeexplore.ieee.org/abstract/document/63533
#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(Index<usize>, target = "paren")]
#[delegate(crate::traits::Backend, target = "paren")]
#[delegate(crate::traits::bit_vec_ops::BitLength, target = "paren")]
#[delegate(crate::traits::rank_sel::BitCount, target = "paren")]
#[delegate(crate::traits::rank_sel::NumBits, target = "paren")]
#[delegate(crate::traits::rank_sel::RankHinted, target = "paren")]
#[delegate(crate::traits::rank_sel::RankUnchecked, target = "paren")]
#[delegate(crate::traits::rank_sel::Rank, target = "paren")]
#[delegate(crate::traits::rank_sel::RankZero, target = "paren")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "paren")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "paren")]
#[delegate(crate::traits::rank_sel::Select, target = "paren")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "paren")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "paren")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "paren")]
pub struct JacobsonBalParen<B = BitVec<Box<[usize]>>, P = EfDict<usize>, O = CompIntList> {
    /// The balanced parentheses bit vector.
    paren: B,
    /// Positions of opening pioneers in a predecessor-capable structure.
    pioneer_positions: P,
    /// Offset from each pioneer to its matching close parenthesis.
    pioneer_match_offsets: O,
}

/// Identifies pioneers and builds the Elias–Fano position index.
///
/// Returns `(ef_positions, pioneer_match_offsets)` where the offsets are
/// raw `Vec<usize>` values to be stored in the chosen offset structure.
fn build_pioneers(words: impl AsRef<[usize]> + BitLength) -> (EfDict<usize>, Vec<usize>) {
    let len = words.len();
    let words = words.as_ref();
    let num_words = len.div_ceil(WORD_BITS);
    debug_assert!(words.len() >= num_words);

    let mut count = vec![0i32; num_words];
    let mut residual = vec![0usize; num_words];

    let mut opening_pioneers = Vec::new();
    let mut opening_pioneer_matches = Vec::new();

    // Scan words right-to-left to identify far opening parentheses
    // and select pioneers among them.
    for block in (0..num_words).rev() {
        let l = std::cmp::min(WORD_BITS, len - block * WORD_BITS);

        // The last word has no words to its right, so it cannot
        // contain far opening parentheses.
        if block != num_words - 1 {
            let mut excess = 0i32;
            let mut count_far_opening = count_far_open(words[block], l) as i32;

            for j in (0..l).rev() {
                if words[block] & (1usize << j) == 0 {
                    if excess > 0 {
                        excess = -1;
                    } else {
                        excess -= 1;
                    }
                } else {
                    excess += 1;
                    if excess > 0 {
                        let mut matching_block = block;
                        loop {
                            matching_block += 1;
                            if count[matching_block] != 0 {
                                break;
                            }
                        }
                        count_far_opening -= 1;

                        if {
                            count[matching_block] -= 1;
                            count[matching_block]
                        } == 0
                            || count_far_opening == 0
                        {
                            let pos = block * WORD_BITS + j;
                            let match_pos = matching_block * WORD_BITS
                                + find_far_close(
                                    words[matching_block],
                                    residual[matching_block] as i64,
                                );
                            opening_pioneers.push(pos);
                            opening_pioneer_matches.push(match_pos - pos);
                        }
                        residual[matching_block] += 1;
                    }
                }
            }
        }
        count[block] = count_far_close(words[block], l) as i32;
    }

    for &c in &count {
        assert!(c == 0, "Unbalanced parentheses");
    }

    // Reverse to get increasing order.
    opening_pioneers.reverse();
    opening_pioneer_matches.reverse();

    // Build Elias–Fano for pioneer positions. The upper bound must be
    // the maximum possible query position (len - 1), not just the last
    // pioneer, because pred_unchecked may be called with any far-open
    // position in the parentheses.
    let num_pioneers = opening_pioneers.len();
    let upper_bound = if num_pioneers > 0 { len - 1 } else { 0 };
    let mut builder = EliasFanoBuilder::new(num_pioneers, upper_bound);
    for &p in &opening_pioneers {
        builder.push(p);
    }
    let ef_positions = builder.build_with_dict();

    (ef_positions, opening_pioneer_matches)
}

impl<B: AsRef<[usize]> + BitLength> JacobsonBalParen<B> {
    /// Builds a new [`JacobsonBalParen`] using a [`CompIntList`] for pioneer
    /// match offsets.
    ///
    /// Pioneer match offsets are stored in a [`CompIntList`] for compact
    /// variable-length encoding. See also [`new_with_prefix_sum`].
    /// and [`new_with_bit_field_vec`].
    ///
    /// [`new_with_prefix_sum`]: Self::new_with_prefix_sum
    /// [`new_with_bit_field_vec`]: Self::new_with_bit_field_vec
    ///
    /// # Panics
    ///
    /// Panics if the parentheses are not balanced.
    pub fn new(paren: B) -> Self {
        let (ef_positions, matches) = build_pioneers(&paren);
        let min_offset = matches.iter().copied().min().unwrap_or(0);
        let offsets = CompIntList::new(min_offset, &matches);

        JacobsonBalParen {
            paren,
            pioneer_positions: ef_positions,
            pioneer_match_offsets: offsets,
        }
    }
}

impl<B: AsRef<[usize]> + BitLength> JacobsonBalParen<B, EfDict<usize>, BitFieldVec<Box<[usize]>>> {
    /// Builds a new [`JacobsonBalParen`] using a [`BitFieldVec`] for pioneer
    /// match offsets.
    ///
    /// This variant uses fixed-width encoding for the offsets, which is faster
    /// to query but uses more space than the default [`CompIntList`]-based
    /// [`new`] or the [`PrefixSumIntList`]-based [`new_with_prefix_sum`].
    ///
    /// [`new`]: JacobsonBalParen::new
    /// [`new_with_prefix_sum`]: Self::new_with_prefix_sum
    ///
    /// # Panics
    ///
    /// Panics if the parentheses are not balanced.
    pub fn new_with_bit_field_vec(paren: B) -> Self {
        let (ef_positions, opening_pioneer_matches) = build_pioneers(&paren);

        let max_offset = opening_pioneer_matches.iter().copied().max().unwrap_or(0);
        let bit_width = if max_offset == 0 {
            1
        } else {
            usize::BITS as usize - max_offset.leading_zeros() as usize
        };
        let num_pioneers = opening_pioneer_matches.len();
        let mut offsets = BitFieldVec::new(bit_width, num_pioneers);
        for (i, &off) in opening_pioneer_matches.iter().enumerate() {
            unsafe { offsets.set_value_unchecked(i, off) };
        }

        JacobsonBalParen {
            paren,
            pioneer_positions: ef_positions,
            pioneer_match_offsets: offsets.into(),
        }
    }
}

impl<B: AsRef<[usize]> + BitLength> JacobsonBalParen<B, EfDict<usize>, PrefixSumIntList> {
    /// Builds a new [`JacobsonBalParen`] using a [`PrefixSumIntList`] for
    /// pioneer match offsets.
    ///
    /// This variant stores the offsets as prefix-sum differences over
    /// Elias–Fano. The space usage is between the default [`CompIntList`]-based
    /// returned by [`new`] and the [`BitFieldVec`]-based returned by
    /// [`new_with_bit_field_vec`].
    ///
    /// [`new`]: JacobsonBalParen::new
    /// [`new_with_bit_field_vec`]: Self::new_with_bit_field_vec
    ///
    /// # Panics
    ///
    /// Panics if the parentheses are not balanced.
    pub fn new_with_prefix_sum(paren: B) -> Self {
        let (ef_positions, matches) = build_pioneers(&paren);
        let offsets = PrefixSumIntList::new(&matches);

        JacobsonBalParen {
            paren,
            pioneer_positions: ef_positions,
            pioneer_match_offsets: offsets,
        }
    }
}

impl<B, P, O> JacobsonBalParen<B, P, O> {
    /// Replaces the pioneer position structure with a new one obtained by
    /// applying `func` to the current one.
    ///
    /// # Safety
    ///
    /// The new structure must return the same values as the old one.
    pub unsafe fn map_pioneer_positions<P2>(
        self,
        func: impl FnOnce(P) -> P2,
    ) -> JacobsonBalParen<B, P2, O> {
        JacobsonBalParen {
            paren: self.paren,
            pioneer_positions: func(self.pioneer_positions),
            pioneer_match_offsets: self.pioneer_match_offsets,
        }
    }

    /// Replaces the pioneer match offset structure with a new one obtained by
    /// applying `func` to the current one.
    ///
    /// # Safety
    ///
    /// The new structure must return the same values as the old one.
    pub unsafe fn map_pioneer_match_offsets<O2>(
        self,
        func: impl FnOnce(O) -> O2,
    ) -> JacobsonBalParen<B, P, O2> {
        JacobsonBalParen {
            paren: self.paren,
            pioneer_positions: self.pioneer_positions,
            pioneer_match_offsets: func(self.pioneer_match_offsets),
        }
    }
}

// ── Queries ────────────────────────────────────────────────────────────

impl<
    B: AsRef<[usize]> + BitLength,
    P: for<'a> PredUnchecked<Input = usize, Output<'a> = usize>,
    O: SliceByValue<Value = usize>,
> BalParen for JacobsonBalParen<B, P, O>
{
    /// Returns the position of the matching close parenthesis for the open
    /// parenthesis at bit position `pos`, or `None` if `pos` is not an
    /// open parenthesis.
    ///
    /// # Panics
    ///
    /// Panics if `pos` is out of bounds.
    fn find_close(&self, pos: usize) -> Option<usize> {
        assert!(
            pos < self.paren.len(),
            "find_close: pos {} out of bounds for length {}",
            pos,
            self.paren.len()
        );
        let word_idx = pos / WORD_BITS;
        let bit_idx = pos % WORD_BITS;

        // Check that it's an open paren
        let words = self.paren.as_ref();
        if words[word_idx] & (1usize << bit_idx) == 0 {
            return None;
        }

        // Try to find the close within the same word
        let result = find_near_close(words[word_idx] >> bit_idx);
        if result < WORD_BITS - bit_idx {
            return Some(word_idx * WORD_BITS + bit_idx + result);
        }

        // Far match: look up the pioneer using predecessor query
        let (pioneer_index, pioneer) =
            unsafe { self.pioneer_positions.pred_unchecked::<false>(pos) };

        let match_pos = pioneer
            + unsafe {
                self.pioneer_match_offsets
                    .get_value_unchecked(pioneer_index)
            };

        if pos == pioneer {
            return Some(match_pos);
        }

        debug_assert_eq!(word_idx, pioneer / WORD_BITS);

        let pioneer_bit = pioneer % WORD_BITS;
        let dist = pos - pioneer;

        // Compute excess difference between pioneer and pos:
        // 2 * popcount(bits from pioneer to pos-1) - dist.
        // dist < WORD_BITS is guaranteed because both are in the same word.
        debug_assert!(dist < WORD_BITS);
        let mask_bits = (words[word_idx] >> pioneer_bit) & ((1usize << dist) - 1);
        let e = 2 * mask_bits.count_ones() as i64 - dist as i64;

        let match_word = match_pos / WORD_BITS;
        let match_bit = match_pos % WORD_BITS;

        // Number of far close parens before match_bit in match_word:
        // match_bit - 2 * popcount(match_word_bits & ((1 << match_bit) - 1))
        let match_word_bits = words[match_word];
        let mask_before_match = if match_bit == 0 {
            0
        } else {
            (1usize << match_bit) - 1
        };
        let num_far_close =
            match_bit as i64 - ((match_word_bits & mask_before_match).count_ones() as i64) * 2;

        Some(match_word * WORD_BITS + find_far_close(match_word_bits, num_far_close - e))
    }
}

// ── Aligned ↔ Unaligned conversion ──────────────────────────────────

use crate::traits::{Backend, BitLength, TryIntoUnaligned};

impl<B, P: TryIntoUnaligned, O: TryIntoUnaligned> TryIntoUnaligned for JacobsonBalParen<B, P, O> {
    type Unaligned = JacobsonBalParen<B, P::Unaligned, O::Unaligned>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(JacobsonBalParen {
            paren: self.paren,
            pioneer_positions: self.pioneer_positions.try_into_unaligned()?,
            pioneer_match_offsets: self.pioneer_match_offsets.try_into_unaligned()?,
        })
    }
}

impl<W, B: AsRef<[W]>, P, O> AsRef<[W]> for JacobsonBalParen<B, P, O> {
    fn as_ref(&self) -> &[W] {
        self.paren.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::BitVecOpsMut;

    use super::*;

    /// Naive find_near_close for testing: scans bit by bit starting after the
    /// open paren (at bit 0 of the shifted word), returns position or
    /// `WORD_BITS`+ if not found.
    fn find_near_close_naive(word: usize) -> usize {
        let mut c = 1i32;
        for i in 1..WORD_BITS {
            if word & (1usize << i) != 0 {
                c += 1;
            } else {
                c -= 1;
            }
            if c == 0 {
                return i;
            }
        }
        WORD_BITS
    }

    /// Naive find_far_close for testing: returns the position of the k-th
    /// (0-based) unmatched close.
    fn find_far_close_reference(word: usize, k: i64) -> usize {
        let mut e = 0i32;
        let mut remaining = k;
        for i in 0..WORD_BITS {
            if word & (1usize << i) != 0 {
                if e > 0 {
                    e = -1;
                } else {
                    e -= 1;
                }
            } else {
                e += 1;
                if e > 0 {
                    if remaining == 0 {
                        return i;
                    }
                    remaining -= 1;
                }
            }
        }
        panic!("Not enough far close parens")
    }

    /// Naive find_close by scanning the whole bit vector.
    fn find_close_naive(words: &[usize], pos: usize, len: usize) -> usize {
        assert!(pos < len);
        assert!(
            words[pos / WORD_BITS] & (1usize << (pos % WORD_BITS)) != 0,
            "Not an open paren"
        );
        let mut c = 1i32;
        for i in (pos + 1)..len {
            let w = words[i / WORD_BITS];
            if w & (1usize << (i % WORD_BITS)) != 0 {
                c += 1;
            } else {
                c -= 1;
            }
            if c == 0 {
                return i;
            }
        }
        panic!("Unmatched open paren at position {pos}")
    }

    #[test]
    fn test_find_near_close_simple() {
        assert_eq!(find_near_close(0b01), 1);
        assert_eq!(find_near_close_naive(0b01), 1);
    }

    #[test]
    fn test_find_near_close_nested() {
        // "(())" = 1 1 0 0 in bit order: bits 0,1 set, bits 2,3 clear
        // word = 0b0011 = 3
        // find_near_close(3) should return 3 (matching close for the outer open)
        assert_eq!(find_near_close(0b0011), 3);
        assert_eq!(find_near_close_naive(0b0011), 3);

        // For the inner open at bit 1: find_near_close(3 >> 1) = find_near_close(1)
        // = 1 (the close at bit 0 of the shifted word = bit 2 of original)
        assert_eq!(find_near_close(0b0011 >> 1), 1);
    }

    #[test]
    fn test_find_near_close_exhaustive_16bit() {
        for w in (1usize..65536).step_by(2) {
            let naive = find_near_close_naive(w);
            let fast = find_near_close(w);
            assert_eq!(
                fast.min(WORD_BITS),
                naive.min(WORD_BITS),
                "find_near_close mismatch for word 0x{w:04x}"
            );
        }
    }

    #[test]
    fn test_find_near_close_large_words() {
        let test_words: [usize; 3] = [
            usize::MAX / 3, // ()()()...
            1,              // single open, no close in word
            usize::MAX,     // all opens, no close in word
        ];
        for w in test_words {
            let naive = find_near_close_naive(w);
            let fast = find_near_close(w);
            assert_eq!(
                fast.min(WORD_BITS),
                naive.min(WORD_BITS),
                "find_near_close mismatch for word 0x{w:016x}"
            );
        }
    }

    #[test]
    fn test_find_far_close_simple() {
        // Word: all zeros (all close parens). Far close 0 is at position 0.
        assert_eq!(find_far_close(0, 0), 0);
        assert_eq!(find_far_close(0, 1), 1);
        assert_eq!(find_far_close(0, 2), 2);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_find_far_close_against_naive() {
        let test_words: Vec<usize> = vec![
            0,
            0b0010, // open at 1, close at 0 and 2,3
            0b1010, // opens at 1,3; close at 0,2
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0x0123_4567_89AB_CDEF,
            0xFEDC_BA98_7654_3210,
            0x0F0F_0F0F_0F0F_0F0F,
            0xF0F0_F0F0_F0F0_F0F0,
            0xDEAD_BEEF_CAFE_BABE,
        ];
        for &w in &test_words {
            let n_far = count_far_close(w, WORD_BITS);
            for k in 0..n_far {
                let naive = find_far_close_reference(w, k as i64);
                let fast = find_far_close(w, k as i64);
                assert_eq!(
                    fast, naive,
                    "find_far_close mismatch for word 0x{:016x}, k={}: fast={}, naive={}",
                    w, k, fast, naive
                );
            }
        }
    }

    #[test]
    fn test_find_far_close_exhaustive_8bit() {
        for w in 0usize..256 {
            let n_far = count_far_close(w, WORD_BITS);
            for k in 0..n_far {
                let naive = find_far_close_reference(w, k as i64);
                let fast = find_far_close(w, k as i64);
                assert_eq!(
                    fast, naive,
                    "find_far_close mismatch for word 0b{:08b}, k={}: fast={}, naive={}",
                    w, k, fast, naive
                );
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_find_close_far_match() {
        // Create a pattern where we have some opens in word 0 matching in word 1.
        // Pattern: "((((" at bits 0-3, then "()" pairs at bits 4-63,
        // then matching "))))" at bits 0-3 of word 1.
        // Bits 0-3: 1111
        // Bits 4-63: 0101...01 (30 pairs)
        // Total word 0: 0x5555_5555_5555_555F
        // Word 1: bits 0-3 = 0000 (closes), bits 4-63 = don't care for balance
        // Actually we need it balanced. Let's be more careful.

        // Simpler: 4 opens, then 30 "()" pairs to fill word 0, then 4 closes.
        // Word 0: bits 0-3 = 1111 (4 opens), bits 4-63 = 01 repeated 30 times
        //   = 0x5555_5555_5555_555F
        // Word 1: bits 0-3 = 0000 (4 closes). Length = 68.
        let bp = JacobsonBalParen::new(unsafe {
            BitVec::from_raw_parts(
                vec![0x5555_5555_5555_555Fusize, 0x0000_0000_0000_0000usize].into_boxed_slice(),
                68,
            )
        });
        let words = bp.as_ref();
        let len = bp.len();

        // Check all opens against naive
        for pos in 0..len {
            if words[pos / WORD_BITS] & (1usize << (pos % WORD_BITS)) != 0 {
                let expected = find_close_naive(words, pos, len);
                assert_eq!(
                    bp.find_close(pos),
                    Some(expected),
                    "find_close mismatch at pos={pos}"
                );
            }
        }
    }

    /// Generate a random balanced parentheses sequence of given length using a
    /// simple stack-based approach.
    fn random_balanced(n: usize) -> BitVec<Box<[usize]>> {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        assert!(n % 2 == 0, "Need even length for balanced parens");

        let mut rng = SmallRng::seed_from_u64(n as u64);

        let half = n / 2;
        let mut bv: BitVec = BitVec::new(n);
        // We need to place n/2 opens and n/2 closes in a balanced way.
        // Use the ballot problem approach: at each position, we can place an
        // open if opens_remaining > 0, and a close if excess > 0.
        let mut opens_remaining = half;
        let mut excess = 0usize;
        for i in 0..n {
            let place_open = if opens_remaining == 0 {
                false
            } else if excess == 0 {
                true
            } else {
                rng.random_bool(opens_remaining as f64 / (opens_remaining + excess) as f64)
            };

            if place_open {
                bv.set(i, true);
                opens_remaining -= 1;
                excess += 1;
            } else {
                excess -= 1;
            }
        }
        assert_eq!(excess, 0);
        assert_eq!(opens_remaining, 0);
        bv.into()
    }

    #[test]
    fn test_find_close_random_small() {
        for size in (2..=40).step_by(2) {
            let bp = JacobsonBalParen::new(random_balanced(size));
            let words = bp.as_ref();
            let len = bp.len();
            for pos in 0..len {
                if words[pos / WORD_BITS] & (1usize << (pos % WORD_BITS)) != 0 {
                    let expected = find_close_naive(words, pos, len);
                    assert_eq!(bp.find_close(pos), Some(expected), "size={size}, pos={pos}");
                }
            }
        }
    }

    #[test]
    fn test_find_close_random_large() {
        for &size in &[128, 256, 512, 1024, 2048] {
            let bp = JacobsonBalParen::new(random_balanced(size));
            let words = bp.as_ref();
            let len = bp.len();
            for pos in 0..len {
                if words[pos / WORD_BITS] & (1usize << (pos % WORD_BITS)) != 0 {
                    let expected = find_close_naive(words, pos, len);
                    assert_eq!(bp.find_close(pos), Some(expected), "size={size}, pos={pos}");
                }
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_find_close_all_nested() {
        // ((((...))))  - 32 opens then 32 closes, fits in one word
        let word = [0x0000_0000_FFFF_FFFFusize];
        let bp = JacobsonBalParen::new(unsafe { BitVec::from_raw_parts(&word, WORD_BITS) });
        for i in 0..32 {
            assert_eq!(bp.find_close(i), Some(63 - i), "pos={i}");
        }
    }

    #[cfg(not(target_pointer_width = "64"))]
    #[test]
    fn test_find_close_all_nested() {
        // ((((...))))  - 16 opens then 16 closes, fits in one word
        let word = [0x0000_FFFFusize];
        let bp = JacobsonBalParen::new(unsafe { BitVec::from_raw_parts(&word, WORD_BITS) });
        for i in 0..16 {
            assert_eq!(bp.find_close(i), Some(31 - i), "pos={i}");
        }
    }

    #[test]
    fn test_find_close_3_words() {
        let bp = JacobsonBalParen::new(random_balanced(192));
        let words = bp.as_ref();
        let len = bp.len();
        for pos in 0..len {
            if words[pos / WORD_BITS] & (1usize << (pos % WORD_BITS)) != 0 {
                let expected = find_close_naive(words, pos, len);
                assert_eq!(bp.find_close(pos), Some(expected), "3-word test, pos={pos}");
            }
        }
    }

    #[test]
    fn test_count_far_open_close() {
        // All opens: count_far_open should count all of them
        assert_eq!(count_far_open(usize::MAX, WORD_BITS), WORD_BITS);
        // All closes: count_far_close should count all of them
        assert_eq!(count_far_close(0, WORD_BITS), WORD_BITS);
        // "()" pairs: no far parens
        assert_eq!(count_far_open(usize::MAX / 3, WORD_BITS), 0);
        assert_eq!(count_far_close(usize::MAX / 3, WORD_BITS), 0);
    }
}
