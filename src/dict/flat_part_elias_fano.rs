/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Flat-packed partitioned Elias–Fano representation.
//!
//! All chunk data (select inventories, high/low bits for EF, rank counters and
//! bitvectors for dense) is stored in a single contiguous `Box<[usize]>`,
//! eliminating per-chunk allocation overhead.

use crate::bits::{BitFieldVec, BitVec};
use crate::dict::elias_fano::{EfDict, EliasFanoBuilder};
use crate::traits::indexed_dict::Types;
use crate::traits::iter::{BidiIterator, SwappedIter};
use crate::traits::{
    BitVecOpsMut, IndexedSeq, Pred, PredBidiIter, PredBidiIterUnchecked, PredIter, PredIterBack,
    PredIterBackUnchecked, PredIterUnchecked, PredUnchecked, Succ, SuccBidiIter,
    SuccBidiIterUnchecked, SuccIter, SuccIterBack, SuccIterBackUnchecked, SuccIterUnchecked,
    SuccUnchecked,
};
use crate::utils::SelectInWord;
use mem_dbg::{MemDbg, MemSize};
use std::borrow::Borrow;
use value_traits::slices::SliceByValue;

const DEFAULT_LOG2_SAMPLING1: usize = 8;

// --- Constants for ds2i-style rank blocks ---

const LOG2_RANK_SAMPLING: usize = 9;
const RANK_SAMPLING: usize = 1 << LOG2_RANK_SAMPLING;

// --- Main structure ---

#[derive(MemSize, MemDbg)]
pub struct FlatPartEliasFano {
    n: usize,
    u: usize,
    log2_sampling1: usize,
    /// Cumulative element counts per partition (for index → partition lookup).
    endpoints: BitFieldVec<Box<[usize]>>,
    /// Upper bound values per partition (for value → partition lookup via succ).
    upper_bounds: EfDict<usize>,
    /// Upper bound values per partition (random access by index).
    upper_bound_values: BitFieldVec<Box<[usize]>>,
    /// One bit per chunk: 0 = sparse (EF), 1 = dense (bitvector).
    chunk_kinds: BitVec<Box<[usize]>>,
    /// Offset into `data` for each chunk.
    chunk_offsets: Box<[usize]>,
    /// Flat backing store for all chunk data.
    data: Box<[usize]>,
}

// --- Inline select/rank helpers operating on raw slices ---

/// Forward scan selecting the `rank`-th one in `words` starting from
/// bit position `hint_pos` with `hint_rank` ones already counted.
#[inline(always)]
unsafe fn select_hinted_raw(
    words: &[usize],
    rank: usize,
    hint_pos: usize,
    hint_rank: usize,
) -> usize {
    let mut word_index = hint_pos / 64;
    let bit_index = hint_pos % 64;
    let mut residual = rank - hint_rank;
    let mut word = (unsafe { *words.get_unchecked(word_index) } >> bit_index) << bit_index;
    loop {
        let bit_count = word.count_ones() as usize;
        if residual < bit_count {
            return word_index * 64 + word.select_in_word(residual);
        }
        residual -= bit_count;
        word_index += 1;
        word = unsafe { *words.get_unchecked(word_index) };
    }
}

/// Forward scan selecting the `rank`-th zero in `words` starting from
/// bit position `hint_pos` with `hint_rank` zeros already counted.
#[inline(always)]
unsafe fn select_zero_hinted_raw(
    words: &[usize],
    rank: usize,
    hint_pos: usize,
    hint_rank: usize,
) -> usize {
    let mut word_index = hint_pos / 64;
    let bit_index = hint_pos % 64;
    let mut residual = rank - hint_rank;
    let mut word = (!unsafe { *words.get_unchecked(word_index) } >> bit_index) << bit_index;
    loop {
        let bit_count = word.count_ones() as usize;
        if residual < bit_count {
            return word_index * 64 + word.select_in_word(residual);
        }
        residual -= bit_count;
        word_index += 1;
        word = !unsafe { *words.get_unchecked(word_index) };
    }
}

/// Select the `rank`-th one using ds2i-style sampled pointers.
#[inline]
unsafe fn inline_select(
    pointers1: &[usize],
    high_bits: &[usize],
    rank: usize,
    ptr_width: usize,
    log2_sampling: usize,
) -> usize {
    let block = rank >> log2_sampling;
    let (hint_pos, hint_rank) = if block == 0 {
        (0, 0)
    } else {
        let pos = unsafe { get_low_bits(pointers1, block - 1, ptr_width) };
        (pos, block << log2_sampling)
    };
    unsafe { select_hinted_raw(high_bits, rank, hint_pos, hint_rank) }
}

/// Select the `rank`-th zero using ds2i-style sampled pointers.
#[inline]
unsafe fn inline_select_zero(
    pointers0: &[usize],
    high_bits: &[usize],
    rank: usize,
    ptr_width: usize,
    log2_sampling: usize,
) -> usize {
    let block = rank >> log2_sampling;
    let (hint_pos, hint_rank) = if block == 0 {
        (0, 0)
    } else {
        let pos = unsafe { get_low_bits(pointers0, block - 1, ptr_width) };
        (pos, block << log2_sampling)
    };
    unsafe { select_zero_hinted_raw(high_bits, rank, hint_pos, hint_rank) }
}

/// Read a value of `bit_width` bits at position `index` from packed low bits.
#[inline(always)]
unsafe fn get_low_bits(low_bits: &[usize], index: usize, bit_width: usize) -> usize {
    if bit_width == 0 {
        return 0;
    }
    let pos = index * bit_width;
    let word_index = pos / 64;
    let bit_index = pos % 64;
    let mask = (1usize << bit_width) - 1;
    let lo = unsafe { *low_bits.get_unchecked(word_index) } >> bit_index;
    if bit_index + bit_width <= 64 {
        lo & mask
    } else {
        (lo | (unsafe { *low_bits.get_unchecked(word_index + 1) } << (64 - bit_index))) & mask
    }
}

/// Rank (count of ones up to but not including `pos`) using ds2i-style
/// packed rank samples (one per 512 bits).
#[inline]
unsafe fn inline_rank(
    rank_samples: &[usize],
    bv: &[usize],
    pos: usize,
    rank_sample_width: usize,
) -> usize {
    let block = pos >> LOG2_RANK_SAMPLING;
    let (base_rank, start_word) = if block == 0 {
        (0, 0)
    } else {
        (
            unsafe { get_low_bits(rank_samples, block - 1, rank_sample_width) },
            (block << LOG2_RANK_SAMPLING) / 64,
        )
    };
    let target_word = pos / 64;
    let mut rank = base_rank;
    let mut wp = start_word;
    while wp < target_word {
        rank += unsafe { *bv.get_unchecked(wp) }.count_ones() as usize;
        wp += 1;
    }
    rank + (unsafe { *bv.get_unchecked(wp) } & ((1usize << (pos % 64)) - 1)).count_ones() as usize
}

/// Select the `rank`-th one in a dense bitvector using ds2i-style sampled
/// select pointers.
#[inline]
unsafe fn inline_dense_select(
    select_pointers: &[usize],
    bv: &[usize],
    rank: usize,
    select_ptr_width: usize,
    log2_sampling1: usize,
) -> usize {
    let block = rank >> log2_sampling1;
    let (hint_pos, hint_rank) = if block == 0 {
        (0, 0)
    } else {
        (
            unsafe { get_low_bits(select_pointers, block - 1, select_ptr_width) },
            block << log2_sampling1,
        )
    };
    unsafe { select_hinted_raw(bv, rank, hint_pos, hint_rank) }
}

// --- Chunk layout helpers ---

/// Compute EF parameters for a chunk.
#[inline]
fn ef_params(n: usize, universe: usize) -> (usize, usize, usize) {
    let l = if universe > n && n > 0 {
        (universe / n).ilog2() as usize
    } else {
        0
    };
    let high_bits_len = n + (universe >> l) + 2;
    (l, high_bits_len, high_bits_len - n)
}

/// Number of bits needed to represent values in [0, len).
#[inline]
fn pointer_width(len: usize) -> usize {
    if len <= 1 {
        0
    } else {
        usize::BITS as usize - (len - 1).leading_zeros() as usize
    }
}

/// Number of words for sampled pointers (ones and zeros combined).
#[inline]
fn sampled_pointers_words(
    n: usize,
    num_zeros: usize,
    high_bits_len: usize,
    log2_sampling1: usize,
) -> (usize, usize) {
    let ptr_width = pointer_width(high_bits_len);
    let sampling1 = 1usize << log2_sampling1;
    let sampling0 = 1usize << (log2_sampling1 + 1);
    let num_ptr1 = if n == 0 { 0 } else { (n - 1) / sampling1 };
    let num_ptr0 = if num_zeros == 0 {
        0
    } else {
        (num_zeros - 1) / sampling0
    };
    let ptr1_words = (num_ptr1 * ptr_width).div_ceil(64);
    let ptr0_words = (num_ptr0 * ptr_width).div_ceil(64);
    (ptr1_words, ptr0_words)
}

// --- Chunk views for accessing flat data ---

struct SparseView<'a> {
    pointers1: &'a [usize],
    pointers0: &'a [usize],
    high_bits: &'a [usize],
    low_bits: &'a [usize],
    n: usize,
    l: usize,
    ptr_width: usize,
    log2_sampling1: usize,
}

struct DenseView<'a> {
    rank_samples: &'a [usize],
    select_pointers: &'a [usize],
    bv: &'a [usize],
    rank_sample_width: usize,
    select_ptr_width: usize,
    log2_sampling1: usize,
}

impl FlatPartEliasFano {
    pub fn num_partitions(&self) -> usize {
        (*self.chunk_offsets).len()
    }

    #[inline]
    fn sparse_view(&self, p: usize, n: usize, universe: usize) -> SparseView<'_> {
        let offset = self.chunk_offsets[p];
        let (l, high_bits_len, num_zeros) = ef_params(n, universe + 1);
        let ptr_width = pointer_width(high_bits_len);

        let (ptr1_words, ptr0_words) =
            sampled_pointers_words(n, num_zeros, high_bits_len, self.log2_sampling1);
        let high_words = high_bits_len.div_ceil(64);

        let pointers1 = &self.data[offset..offset + ptr1_words];
        let pointers0 = &self.data[offset + ptr1_words..offset + ptr1_words + ptr0_words];
        let high_start = offset + ptr1_words + ptr0_words;
        let high_bits = &self.data[high_start..high_start + high_words];
        let low_start = high_start + high_words;
        let low_words = (n * l).div_ceil(64);
        let low_bits = &self.data[low_start..low_start + low_words];

        SparseView {
            pointers1,
            pointers0,
            high_bits,
            low_bits,
            n,
            l,
            ptr_width,
            log2_sampling1: self.log2_sampling1,
        }
    }

    #[inline]
    fn dense_view(&self, p: usize, n: usize, universe: usize) -> DenseView<'_> {
        let offset = self.chunk_offsets[p];

        let rank_sample_width = pointer_width(n + 1);
        let select_ptr_width = pointer_width(universe);
        let num_rank_samples = universe >> LOG2_RANK_SAMPLING;
        let sampling1 = 1usize << self.log2_sampling1;
        let num_select_pointers = if n == 0 { 0 } else { (n - 1) / sampling1 };

        let rank_words = (num_rank_samples * rank_sample_width).div_ceil(64);
        let select_words = (num_select_pointers * select_ptr_width).div_ceil(64);
        let bv_words = universe.div_ceil(64);

        let rank_samples = &self.data[offset..offset + rank_words];
        let select_pointers = &self.data[offset + rank_words..offset + rank_words + select_words];
        let bv_start = offset + rank_words + select_words;
        let bv = &self.data[bv_start..bv_start + bv_words];

        DenseView {
            rank_samples,
            select_pointers,
            bv,
            rank_sample_width,
            select_ptr_width,
            log2_sampling1: self.log2_sampling1,
        }
    }

    #[inline]
    fn is_dense(&self, p: usize) -> bool {
        self.chunk_kinds[p]
    }
}

// --- IndexedSeq ---

impl Types for FlatPartEliasFano {
    type Output<'a> = usize;
    type Input = usize;
}

impl IndexedSeq for FlatPartEliasFano {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        let num_partitions = (*self.chunk_offsets).len();
        let mut lo = 0usize;
        let mut hi = num_partitions;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if unsafe { self.endpoints.get_value_unchecked(mid) } <= index {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let partition_idx = lo;
        let endpoint = unsafe { self.endpoints.get_value_unchecked(partition_idx) };
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_value_unchecked(partition_idx - 1) }
        };
        let local_index = index - partition_start;
        let n = endpoint - partition_start;

        let upper_bound =
            unsafe { self.upper_bound_values.get_value_unchecked(partition_idx) };
        let base = if partition_idx == 0 {
            0
        } else {
            unsafe { self.upper_bound_values.get_value_unchecked(partition_idx - 1) }
        };
        let universe = upper_bound - base;

        (unsafe { self.chunk_get_unchecked(partition_idx, local_index, n, universe) }) + base
    }

    fn len(&self) -> usize {
        self.n
    }
}

// --- SuccUnchecked ---

pub struct FlatPefIter<'a> {
    pef: &'a FlatPartEliasFano,
    pos: usize,
}

impl Iterator for FlatPefIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.pos >= self.pef.n {
            return None;
        }
        let val = unsafe { self.pef.get_unchecked(self.pos) };
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.pef.n - self.pos;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for FlatPefIter<'_> {}

pub struct FlatPefBidiIter<'a> {
    pef: &'a FlatPartEliasFano,
    pos: usize,
}

impl Iterator for FlatPefBidiIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.pos >= self.pef.n {
            return None;
        }
        let val = unsafe { self.pef.get_unchecked(self.pos) };
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.pef.n - self.pos;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for FlatPefBidiIter<'_> {}

impl BidiIterator for FlatPefBidiIter<'_> {
    type SwappedIter = SwappedIter<Self>;

    #[inline(always)]
    fn swap(self) -> SwappedIter<Self> {
        SwappedIter(self)
    }

    #[inline]
    fn prev(&mut self) -> Option<usize> {
        if self.pos == 0 {
            return None;
        }
        self.pos -= 1;
        Some(unsafe { self.pef.get_unchecked(self.pos) })
    }

    fn prev_size_hint(&self) -> (usize, Option<usize>) {
        (self.pos, Some(self.pos))
    }
}

impl SuccUnchecked for FlatPartEliasFano {
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, usize) {
        let value = *value.borrow();
        let (partition_idx, mut iter) =
            unsafe { self.upper_bounds.iter_back_from_succ_unchecked::<false>(value) };
        let upper_bound = iter.next().unwrap();
        let base = if partition_idx == 0 {
            0
        } else {
            iter.next().unwrap()
        };
        let universe = upper_bound - base;
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_value_unchecked(partition_idx - 1) }
        };
        let endpoint = unsafe { self.endpoints.get_value_unchecked(partition_idx) };
        let n = endpoint - partition_start;

        let relative = value - base;
        let (local_idx, local_val) = if STRICT {
            unsafe { self.chunk_succ_strict(partition_idx, relative, n, universe) }
        } else {
            unsafe { self.chunk_succ(partition_idx, relative, n, universe) }
        };
        (partition_start + local_idx, local_val + base)
    }
}

impl SuccIterUnchecked for FlatPartEliasFano {
    type Iter<'a> = FlatPefIter<'a>;

    unsafe fn iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::Iter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (idx, FlatPefIter { pef: self, pos: idx })
    }
}

impl SuccBidiIterUnchecked for FlatPartEliasFano {
    type BidiIter<'a> = FlatPefBidiIter<'a>;

    unsafe fn iter_bidi_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BidiIter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (idx, FlatPefBidiIter { pef: self, pos: idx })
    }
}

impl SuccIterBackUnchecked for FlatPartEliasFano {
    type BackIter<'a> = SwappedIter<FlatPefBidiIter<'a>>;

    unsafe fn iter_back_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BackIter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (
            idx,
            SwappedIter(FlatPefBidiIter {
                pef: self,
                pos: idx + 1,
            }),
        )
    }
}

impl FlatPartEliasFano {
    #[inline]
    unsafe fn chunk_get_unchecked(
        &self,
        p: usize,
        index: usize,
        n: usize,
        universe: usize,
    ) -> usize {
        if self.is_dense(p) {
            let view = self.dense_view(p, n, universe + 1);
            unsafe { inline_dense_select(view.select_pointers, view.bv, index, view.select_ptr_width, view.log2_sampling1) }
        } else {
            let view = self.sparse_view(p, n, universe);
            let high = unsafe {
                inline_select(view.pointers1, view.high_bits, index, view.ptr_width, view.log2_sampling1)
            } - index;
            let low = unsafe { get_low_bits(view.low_bits, index, view.l) };
            (high << view.l) | low
        }
    }

    #[inline]
    unsafe fn chunk_succ(
        &self,
        p: usize,
        value: usize,
        n: usize,
        universe: usize,
    ) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p, n, universe + 1);
            let rank = unsafe { inline_rank(view.rank_samples, view.bv, value, view.rank_sample_width) };
            let pos = unsafe { inline_dense_select(view.select_pointers, view.bv, rank, view.select_ptr_width, view.log2_sampling1) };
            if pos >= value {
                (rank, pos)
            } else {
                (rank + 1, unsafe { inline_dense_select(view.select_pointers, view.bv, rank + 1, view.select_ptr_width, view.log2_sampling1) })
            }
        } else {
            let view = self.sparse_view(p, n, universe);
            unsafe { ef_succ_raw::<false>(&view, value) }
        }
    }

    #[inline]
    unsafe fn chunk_succ_strict(
        &self,
        p: usize,
        value: usize,
        n: usize,
        universe: usize,
    ) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p, n, universe + 1);
            let bv_len = universe + 1;
            let rank = if value >= bv_len {
                (unsafe { inline_rank(view.rank_samples, view.bv, bv_len, view.rank_sample_width) }) + 1
            } else {
                unsafe { inline_rank(view.rank_samples, view.bv, value + 1, view.rank_sample_width) }
            };
            (rank, unsafe { inline_dense_select(view.select_pointers, view.bv, rank, view.select_ptr_width, view.log2_sampling1) })
        } else {
            let view = self.sparse_view(p, n, universe);
            unsafe { ef_succ_raw::<true>(&view, value) }
        }
    }

    #[inline]
    unsafe fn chunk_pred(
        &self,
        p: usize,
        value: usize,
        n: usize,
        universe: usize,
    ) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p, n, universe + 1);
            let bv_len = universe + 1;
            let rank = if value >= bv_len {
                n
            } else {
                unsafe { inline_rank(view.rank_samples, view.bv, value + 1, view.rank_sample_width) }
            };
            (rank - 1, unsafe { inline_dense_select(view.select_pointers, view.bv, rank - 1, view.select_ptr_width, view.log2_sampling1) })
        } else {
            let view = self.sparse_view(p, n, universe);
            unsafe { ef_pred_raw::<false>(&view, value) }
        }
    }

    #[inline]
    unsafe fn chunk_pred_strict(
        &self,
        p: usize,
        value: usize,
        n: usize,
        universe: usize,
    ) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p, n, universe + 1);
            let rank = unsafe { inline_rank(view.rank_samples, view.bv, value, view.rank_sample_width) };
            (rank - 1, unsafe { inline_dense_select(view.select_pointers, view.bv, rank - 1, view.select_ptr_width, view.log2_sampling1) })
        } else {
            let view = self.sparse_view(p, n, universe);
            unsafe { ef_pred_raw::<true>(&view, value) }
        }
    }
}

/// Successor search within a sparse (EF) chunk.
#[inline]
unsafe fn ef_succ_raw<const STRICT: bool>(view: &SparseView<'_>, value: usize) -> (usize, usize) {
    let zeros_to_skip = value >> view.l;
    let bit_pos = if zeros_to_skip == 0 {
        0
    } else {
        (unsafe {
            inline_select_zero(view.pointers0, view.high_bits, zeros_to_skip - 1, view.ptr_width, view.log2_sampling1 + 1)
        }) + 1
    };

    let mut rank = bit_pos - zeros_to_skip;
    let mut word_idx = bit_pos / 64;
    let bits_to_clean = bit_pos % 64;
    let mut window =
        unsafe { *view.high_bits.get_unchecked(word_idx) } & (usize::MAX << bits_to_clean);

    loop {
        while window == 0 {
            word_idx += 1;
            window = unsafe { *view.high_bits.get_unchecked(word_idx) };
        }
        let bit_idx = window.trailing_zeros() as usize;
        let high_bits_val = word_idx * 64 + bit_idx - rank;
        let low = unsafe { get_low_bits(view.low_bits, rank, view.l) };
        let res = (high_bits_val << view.l) | low;

        let found = if STRICT { res > value } else { res >= value };
        if found {
            return (rank, res);
        }
        rank += 1;
        window &= window - 1; // clear lowest set bit
    }
}

/// Predecessor search within a sparse (EF) chunk.
#[inline]
unsafe fn ef_pred_raw<const STRICT: bool>(view: &SparseView<'_>, value: usize) -> (usize, usize) {
    let zeros_to_skip = (value >> view.l) + 1;
    let bit_pos = if zeros_to_skip == 0 {
        0
    } else {
        let num_zeros_in_high = view.high_bits.len() * 64 - view.n;
        if zeros_to_skip > num_zeros_in_high {
            let last_idx = view.n - 1;
            let high = unsafe {
                inline_select(view.pointers1, view.high_bits, last_idx, view.ptr_width, view.log2_sampling1)
            } - last_idx;
            let low = unsafe { get_low_bits(view.low_bits, last_idx, view.l) };
            let last_val = (high << view.l) | low;
            if STRICT {
                if last_val < value {
                    return (last_idx, last_val);
                }
            } else if last_val <= value {
                return (last_idx, last_val);
            }
        }
        (unsafe {
            inline_select_zero(view.pointers0, view.high_bits, zeros_to_skip - 1, view.ptr_width, view.log2_sampling1 + 1)
        }) + 1
    };

    let rank = bit_pos - zeros_to_skip;

    if STRICT {
        if rank == 0 {
            let high = unsafe {
                inline_select(view.pointers1, view.high_bits, 0, view.ptr_width, view.log2_sampling1)
            };
            let low = unsafe { get_low_bits(view.low_bits, 0, view.l) };
            let val = (high << view.l) | low;
            return (0, val);
        }
        let idx = rank - 1;
        let high = unsafe {
            inline_select(view.pointers1, view.high_bits, idx, view.ptr_width, view.log2_sampling1)
        } - idx;
        let low = unsafe { get_low_bits(view.low_bits, idx, view.l) };
        let val = (high << view.l) | low;
        if val < value {
            return (idx, val);
        }
        if idx == 0 {
            panic!("pred_strict called but no predecessor exists");
        }
        let idx2 = idx - 1;
        let high2 = unsafe {
            inline_select(view.pointers1, view.high_bits, idx2, view.ptr_width, view.log2_sampling1)
        } - idx2;
        let low2 = unsafe { get_low_bits(view.low_bits, idx2, view.l) };
        (idx2, (high2 << view.l) | low2)
    } else {
        let mut idx = rank;
        loop {
            if idx == 0 {
                let high = unsafe {
                    inline_select(view.pointers1, view.high_bits, 0, view.ptr_width, view.log2_sampling1)
                };
                let low = unsafe { get_low_bits(view.low_bits, 0, view.l) };
                return (0, (high << view.l) | low);
            }
            idx -= 1;
            let high = unsafe {
                inline_select(view.pointers1, view.high_bits, idx, view.ptr_width, view.log2_sampling1)
            } - idx;
            let low = unsafe { get_low_bits(view.low_bits, idx, view.l) };
            let val = (high << view.l) | low;
            if val <= value {
                return (idx, val);
            }
        }
    }
}

// --- Succ/Pred trait implementations ---

impl Succ for FlatPartEliasFano {
    fn succ(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<false>(value) })
        }
    }

    fn succ_strict(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<true>(value) })
        }
    }
}

impl SuccIter for FlatPartEliasFano {
    fn iter_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccIterUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<false>(value) })
        }
    }

    fn iter_from_succ_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccIterUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<true>(value) })
        }
    }
}

impl SuccBidiIter for FlatPartEliasFano {
    fn iter_bidi_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccBidiIterUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<false>(value) })
        }
    }

    fn iter_bidi_from_succ_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccBidiIterUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<true>(value) })
        }
    }
}

impl SuccIterBack for FlatPartEliasFano {
    fn iter_back_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccIterBackUnchecked>::BackIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_succ_unchecked::<false>(value) })
        }
    }

    fn iter_back_from_succ_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccIterBackUnchecked>::BackIter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_succ_unchecked::<true>(value) })
        }
    }
}

impl PredUnchecked for FlatPartEliasFano {
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, usize) {
        let value = *value.borrow();
        let (partition_idx, mut iter) =
            unsafe { self.upper_bounds.iter_back_from_succ_unchecked::<false>(value) };
        let upper_bound = iter.next().unwrap();
        let base = if partition_idx == 0 {
            0
        } else {
            iter.next().unwrap()
        };
        let universe = upper_bound - base;
        let endpoint = unsafe { self.endpoints.get_value_unchecked(partition_idx) };
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_value_unchecked(partition_idx - 1) }
        };
        let n = endpoint - partition_start;

        let relative = value - base;
        let first_elem =
            unsafe { self.chunk_get_unchecked(partition_idx, 0, n, universe) };
        if relative < first_elem || (STRICT && relative == first_elem) {
            let prev_part = partition_idx - 1;
            let prev_base = if prev_part == 0 {
                0
            } else {
                unsafe { self.upper_bound_values.get_value_unchecked(prev_part - 1) }
            };
            let prev_upper = base;
            let prev_universe = prev_upper - prev_base;
            let prev_start = if prev_part == 0 {
                0
            } else {
                unsafe { self.endpoints.get_value_unchecked(prev_part - 1) }
            };
            let prev_n = partition_start - prev_start;
            let local_last = prev_n - 1;
            let local_val =
                unsafe { self.chunk_get_unchecked(prev_part, local_last, prev_n, prev_universe) };
            return (prev_start + local_last, local_val + prev_base);
        }

        let (local_idx, local_val) = if STRICT {
            unsafe { self.chunk_pred_strict(partition_idx, relative, n, universe) }
        } else {
            unsafe { self.chunk_pred(partition_idx, relative, n, universe) }
        };
        (partition_start + local_idx, local_val + base)
    }
}

impl PredIterUnchecked for FlatPartEliasFano {
    type Iter<'a> = FlatPefIter<'a>;

    unsafe fn iter_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::Iter<'_>) {
        let (idx, _) = unsafe { self.pred_unchecked::<STRICT>(value) };
        (idx, FlatPefIter { pef: self, pos: idx })
    }
}

impl PredIterBackUnchecked for FlatPartEliasFano {
    type BackIter<'a> = SwappedIter<FlatPefBidiIter<'a>>;

    unsafe fn iter_back_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BackIter<'_>) {
        let (idx, _) = unsafe { self.pred_unchecked::<STRICT>(value) };
        (
            idx,
            SwappedIter(FlatPefBidiIter {
                pef: self,
                pos: idx + 1,
            }),
        )
    }
}

impl PredBidiIterUnchecked for FlatPartEliasFano {
    type BidiIter<'a> = FlatPefBidiIter<'a>;

    unsafe fn iter_bidi_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BidiIter<'_>) {
        let (idx, _) = unsafe { self.pred_unchecked::<STRICT>(value) };
        (
            idx,
            FlatPefBidiIter {
                pef: self,
                pos: idx + 1,
            },
        )
    }
}

impl Pred for FlatPartEliasFano {
    fn pred(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<false>(value) })
        }
    }

    fn pred_strict(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<true>(value) })
        }
    }
}

impl PredIter for FlatPartEliasFano {
    fn iter_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as PredIterUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_from_pred_unchecked::<false>(value) })
        }
    }

    fn iter_from_pred_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as PredIterUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_from_pred_unchecked::<true>(value) })
        }
    }
}

impl PredIterBack for FlatPartEliasFano {
    fn iter_back_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as PredIterBackUnchecked>::BackIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<false>(value) })
        }
    }

    fn iter_back_from_pred_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as PredIterBackUnchecked>::BackIter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<true>(value) })
        }
    }
}

impl PredBidiIter for FlatPartEliasFano {
    fn iter_bidi_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as PredBidiIterUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<false>(value) })
        }
    }

    fn iter_bidi_from_pred_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as PredBidiIterUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<true>(value) })
        }
    }
}

// --- Builder ---

/// Cost in bits of a sparse (EF) chunk in the flat layout, including sampled
/// pointer overhead.
fn flat_ef_cost(universe: usize, n: usize, log2_sampling1: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let l = if universe > n {
        (universe / n).ilog2() as usize
    } else {
        0
    };
    let high_bits_len = n + (universe >> l) + 2;
    let num_zeros = high_bits_len - n;
    let ptr_width = pointer_width(high_bits_len);
    let sampling1 = 1usize << log2_sampling1;
    let sampling0 = 1usize << (log2_sampling1 + 1);
    let num_ptr1 = (n - 1) / sampling1;
    let num_ptr0 = if num_zeros == 0 {
        0
    } else {
        (num_zeros - 1) / sampling0
    };
    let ptr_bits = (num_ptr1 + num_ptr0) * ptr_width;
    high_bits_len + n * l + ptr_bits
}

/// Cost in bits of a dense chunk in the flat layout (ds2i compact_ranked_bitvector).
fn flat_dense_cost(universe: usize, n: usize, log2_sampling1: usize) -> usize {
    let num_rank_samples = universe >> LOG2_RANK_SAMPLING;
    let rank_sample_width = pointer_width(n + 1);
    let sampling1 = 1usize << log2_sampling1;
    let num_select_pointers = if n == 0 { 0 } else { (n - 1) / sampling1 };
    let select_ptr_width = pointer_width(universe);
    universe + num_rank_samples * rank_sample_width + num_select_pointers * select_ptr_width
}

fn flat_chunk_cost(universe: usize, n: usize, log2_sampling1: usize) -> usize {
    if n > universe {
        flat_ef_cost(universe, n, log2_sampling1)
    } else {
        flat_ef_cost(universe, n, log2_sampling1)
            .min(flat_dense_cost(universe, n, log2_sampling1))
    }
}

struct ChunkInfo {
    start: usize,
    end: usize,
    base: usize,
    universe: usize,
    is_dense: bool,
    data_words: usize,
}

pub struct FlatPartEliasFanoBuilder {
    n: usize,
    u: usize,
    eps1: f64,
    eps2: f64,
    fix_cost: usize,
    log2_sampling1: usize,
    values: Vec<usize>,
}

impl FlatPartEliasFanoBuilder {
    pub fn new(n: usize, u: usize) -> Self {
        Self {
            n,
            u,
            eps1: 0.03,
            eps2: 0.3,
            fix_cost: 64,
            log2_sampling1: DEFAULT_LOG2_SAMPLING1,
            values: Vec::with_capacity(n),
        }
    }

    pub fn eps1(mut self, eps1: f64) -> Self {
        self.eps1 = eps1;
        self
    }

    pub fn eps2(mut self, eps2: f64) -> Self {
        self.eps2 = eps2;
        self
    }

    pub fn fix_cost(mut self, fix_cost: usize) -> Self {
        self.fix_cost = fix_cost;
        self
    }

    pub fn log2_sampling1(mut self, log2_sampling1: usize) -> Self {
        self.log2_sampling1 = log2_sampling1;
        self
    }

    pub fn push(&mut self, value: usize) {
        debug_assert!(
            self.values.is_empty() || value >= *self.values.last().unwrap(),
            "values must be monotone non-decreasing"
        );
        self.values.push(value);
    }

    pub fn build(self) -> FlatPartEliasFano {
        assert_eq!(self.values.len(), self.n);
        if self.n == 0 {
            return FlatPartEliasFano {
                n: 0,
                u: self.u,
                log2_sampling1: self.log2_sampling1,
                endpoints: BitFieldVec::<Vec<usize>>::new(1, 0).into_padded(),
                upper_bounds: EliasFanoBuilder::new(0, 0usize).build_with_dict(),
                upper_bound_values: BitFieldVec::<Vec<usize>>::new(1, 0).into_padded(),
                chunk_kinds: BitVec::new(0).into(),
                chunk_offsets: Box::new([]),
                data: Box::new([]),
            };
        }

        let fix_cost = self.fix_cost;
        let log2_sampling1 = self.log2_sampling1;
        let partition_points = super::part_elias_fano::optimal_partition_with(
            &self.values,
            self.eps1,
            self.eps2,
            |universe, n| flat_chunk_cost(universe, n, log2_sampling1) + fix_cost,
        );

        let num_partitions = partition_points.len() - 1;
        let mut chunk_kinds_vec = Vec::with_capacity(num_partitions);
        let mut cumulative_sizes = Vec::with_capacity(num_partitions);
        let mut upper_bound_values = Vec::with_capacity(num_partitions);

        // First pass: determine chunk types and compute total data size
        let mut chunk_infos = Vec::with_capacity(num_partitions);
        let mut total_data_words = 0usize;
        let mut cumulative = 0usize;

        for p in 0..num_partitions {
            let start = partition_points[p];
            let end = partition_points[p + 1];
            let chunk_n = end - start;
            cumulative += chunk_n;
            cumulative_sizes.push(cumulative);

            let upper = self.values[end - 1];
            // Base is upper_bound of previous partition (0 for first)
            let base = if p == 0 { 0 } else { upper_bound_values[p - 1] };
            upper_bound_values.push(upper);

            let universe = upper - base; // range is [0, universe]
            let has_duplicates = self.values[start..end].windows(2).any(|w| w[0] == w[1]);
            let use_dense = !has_duplicates
                && chunk_n <= universe + 1
                && flat_dense_cost(universe + 1, chunk_n, log2_sampling1)
                    < flat_ef_cost(universe + 1, chunk_n, log2_sampling1);
            chunk_kinds_vec.push(use_dense);

            let data_words = if use_dense {
                let bv_bits = universe + 1;
                let num_rank_samples = bv_bits >> LOG2_RANK_SAMPLING;
                let rank_sample_width = pointer_width(chunk_n + 1);
                let sampling1 = 1usize << log2_sampling1;
                let num_select_pointers = if chunk_n == 0 { 0 } else { (chunk_n - 1) / sampling1 };
                let select_ptr_width = pointer_width(bv_bits);
                let rank_words = (num_rank_samples * rank_sample_width).div_ceil(64);
                let select_words = (num_select_pointers * select_ptr_width).div_ceil(64);
                let bv_words = bv_bits.div_ceil(64);
                rank_words + select_words + bv_words
            } else {
                let (l, high_bits_len, num_zeros) = ef_params(chunk_n, universe + 1);
                let (ptr1_words, ptr0_words) =
                    sampled_pointers_words(chunk_n, num_zeros, high_bits_len, log2_sampling1);
                let high_words = high_bits_len.div_ceil(64);
                let low_words = (chunk_n * l).div_ceil(64);
                ptr1_words + ptr0_words + high_words + low_words
            };

            chunk_infos.push(ChunkInfo {
                start,
                end,
                base,
                universe,
                is_dense: use_dense,
                data_words,
            });
            total_data_words += data_words;
        }

        // Allocate the flat data store
        let mut data = vec![0usize; total_data_words];
        let mut chunk_offsets = Vec::with_capacity(num_partitions);
        let mut offset = 0usize;

        // Second pass: populate data
        for info in chunk_infos.iter() {
            chunk_offsets.push(offset);

            if info.is_dense {
                self.build_dense_chunk(&mut data, offset, info);
            } else {
                self.build_sparse_chunk(&mut data, offset, info);
            }

            offset += info.data_words;
        }
        debug_assert_eq!(offset, total_data_words);

        // Build chunk_kinds bitvec
        let mut chunk_kinds = BitVec::new(num_partitions);
        for (i, &is_dense) in chunk_kinds_vec.iter().enumerate() {
            if is_dense {
                chunk_kinds.set(i, true);
            }
        }

        // Build endpoints as BitFieldVec
        let ep_width = pointer_width(self.n + 1);
        let mut endpoints = BitFieldVec::<Vec<usize>>::new(ep_width, 0);
        for &c in &cumulative_sizes {
            endpoints.push(c);
        }
        let endpoints = endpoints.into_padded();

        // Build upper_bounds as EfDict (for succ/pred by value)
        let mut ub_builder = EliasFanoBuilder::new(num_partitions, self.u);
        for &ub in &upper_bound_values {
            unsafe { ub_builder.push_unchecked(ub) };
        }
        let upper_bounds = ub_builder.build_with_dict();

        // Build upper_bound_values as BitFieldVec (for random access by index)
        let ubv_width = pointer_width(self.u + 1);
        let mut ubv = BitFieldVec::<Vec<usize>>::new(ubv_width, 0);
        for &ub in &upper_bound_values {
            ubv.push(ub);
        }
        let upper_bound_values_bfv = ubv.into_padded();

        FlatPartEliasFano {
            n: self.n,
            u: self.u,
            log2_sampling1,
            endpoints,
            upper_bounds,
            upper_bound_values: upper_bound_values_bfv,
            chunk_kinds: chunk_kinds.into(),
            chunk_offsets: chunk_offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
        }
    }

    fn build_dense_chunk(&self, data: &mut [usize], offset: usize, info: &ChunkInfo) {
        let chunk_n = info.end - info.start;
        let bv_bits = info.universe + 1;
        let num_rank_samples = bv_bits >> LOG2_RANK_SAMPLING;
        let rank_sample_width = pointer_width(chunk_n + 1);
        let sampling1 = 1usize << self.log2_sampling1;
        let num_select_pointers = if chunk_n == 0 { 0 } else { (chunk_n - 1) / sampling1 };
        let select_ptr_width = pointer_width(bv_bits);

        let rank_words = (num_rank_samples * rank_sample_width).div_ceil(64);
        let select_words = (num_select_pointers * select_ptr_width).div_ceil(64);
        let bv_words = bv_bits.div_ceil(64);

        let rank_start = offset;
        let select_start = offset + rank_words;
        let bv_start = select_start + select_words;

        // Fill bitvector
        for &v in &self.values[info.start..info.end] {
            let bit = v - info.base;
            data[bv_start + bit / 64] |= 1usize << (bit % 64);
        }

        // Build rank samples: rank_sample[k] = number of ones in [0, (k+1)*512)
        if rank_sample_width > 0 {
            let mut past_ones = 0usize;
            let rank_block_words = RANK_SAMPLING / 64;
            for k in 0..num_rank_samples {
                let block_word_start = k * rank_block_words;
                for j in 0..rank_block_words {
                    let word_idx = block_word_start + j;
                    if word_idx < bv_words {
                        past_ones += data[bv_start + word_idx].count_ones() as usize;
                    }
                }
                // Write packed rank sample
                let pos = k * rank_sample_width;
                let w_idx = pos / 64;
                let b_idx = pos % 64;
                data[rank_start + w_idx] |= past_ones << b_idx;
                if b_idx + rank_sample_width > 64 {
                    data[rank_start + w_idx + 1] |= past_ones >> (64 - b_idx);
                }
            }
        }

        // Build select pointers using the same mechanism as sparse chunks
        Self::build_sampled_pointers(
            data, select_start, bv_start, bv_words, chunk_n, sampling1, select_ptr_width, true,
        );
    }

    fn build_sparse_chunk(&self, data: &mut [usize], offset: usize, info: &ChunkInfo) {
        let chunk_n = info.end - info.start;
        let (l, high_bits_len, num_zeros) = ef_params(chunk_n, info.universe + 1);
        let ptr_width = pointer_width(high_bits_len);

        let (ptr1_words, ptr0_words) =
            sampled_pointers_words(chunk_n, num_zeros, high_bits_len, self.log2_sampling1);
        let high_words = high_bits_len.div_ceil(64);

        let ptr1_start = offset;
        let ptr0_start = offset + ptr1_words;
        let high_start = ptr0_start + ptr0_words;
        let low_start = high_start + high_words;

        // Fill high bits and low bits
        for (i, &v) in self.values[info.start..info.end].iter().enumerate() {
            let relative = v - info.base;
            let high = relative >> l;
            let low = relative & ((1 << l) - 1);

            let high_pos = i + high;
            data[high_start + high_pos / 64] |= 1usize << (high_pos % 64);

            if l > 0 {
                let bit_pos = i * l;
                let word_idx = bit_pos / 64;
                let bit_idx = bit_pos % 64;
                data[low_start + word_idx] |= low << bit_idx;
                if bit_idx + l > 64 {
                    data[low_start + word_idx + 1] |= low >> (64 - bit_idx);
                }
            }
        }

        let sampling1 = 1usize << self.log2_sampling1;
        let sampling0 = 1usize << (self.log2_sampling1 + 1);

        Self::build_sampled_pointers(
            data, ptr1_start, high_start, high_words, chunk_n, sampling1, ptr_width, true,
        );

        Self::build_sampled_pointers(
            data, ptr0_start, high_start, high_words, num_zeros, sampling0, ptr_width, false,
        );
    }

    /// Build ds2i-style sampled pointers for ones or zeros in the high bits.
    /// Writes packed pointers (each `ptr_width` bits) to `data[ptr_start..]`.
    fn build_sampled_pointers(
        data: &mut [usize],
        ptr_start: usize,
        bits_offset: usize,
        bits_words: usize,
        num_targets: usize,
        sampling: usize,
        ptr_width: usize,
        ones: bool,
    ) {
        if ptr_width == 0 || num_targets == 0 {
            return;
        }
        let num_pointers = (num_targets - 1) / sampling;
        if num_pointers == 0 {
            return;
        }

        let mut past_targets = 0usize;
        let mut next_record_rank = sampling;
        let mut ptr_idx = 0usize;

        for word_i in 0..bits_words {
            let raw_word = data[bits_offset + word_i];
            let effective_word = if ones { raw_word } else { !raw_word };
            let targets_in_word = effective_word.count_ones() as usize;

            while past_targets + targets_in_word > next_record_rank && ptr_idx < num_pointers {
                let in_word_pos =
                    effective_word.select_in_word(next_record_rank - past_targets);
                let bit_position = word_i * 64 + in_word_pos;

                // Write packed pointer
                let pos = ptr_idx * ptr_width;
                let w_idx = pos / 64;
                let b_idx = pos % 64;
                data[ptr_start + w_idx] |= bit_position << b_idx;
                if b_idx + ptr_width > 64 {
                    data[ptr_start + w_idx + 1] |= bit_position >> (64 - b_idx);
                }

                ptr_idx += 1;
                next_record_rank += sampling;
            }
            past_targets += targets_in_word;
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let values: Vec<usize> = vec![0, 3, 7, 15, 20, 30, 50, 100, 200, 500];
        let n = values.len();
        let u = 500;

        let mut builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }
    }

    #[test]
    fn test_succ() {
        let values: Vec<usize> = vec![10, 20, 30, 40, 50, 100, 200, 300];
        let n = values.len();
        let u = 300;

        let mut builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        assert_eq!(pef.succ(0), Some((0, 10)));
        assert_eq!(pef.succ(10), Some((0, 10)));
        assert_eq!(pef.succ(11), Some((1, 20)));
        assert_eq!(pef.succ(25), Some((2, 30)));
        assert_eq!(pef.succ(300), Some((7, 300)));
        assert_eq!(pef.succ(301), None);
    }

    #[test]
    fn test_pred() {
        let values: Vec<usize> = vec![10, 20, 30, 40, 50, 100, 200, 300];
        let n = values.len();
        let u = 300;

        let mut builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        assert_eq!(pef.pred(300), Some((7, 300)));
        assert_eq!(pef.pred(299), Some((6, 200)));
        assert_eq!(pef.pred(50), Some((4, 50)));
        assert_eq!(pef.pred(10), Some((0, 10)));
        assert_eq!(pef.pred(9), None);
    }

    #[test]
    fn test_empty() {
        let builder = FlatPartEliasFanoBuilder::new(0, 0);
        let pef = builder.build();
        assert_eq!(pef.len(), 0);
        assert_eq!(pef.succ(0), None);
        assert_eq!(pef.pred(0), None);
    }

    #[test]
    fn test_single() {
        let mut builder = FlatPartEliasFanoBuilder::new(1, 42);
        builder.push(42);
        let pef = builder.build();
        assert_eq!(pef.get(0), 42);
        assert_eq!(pef.succ(42), Some((0, 42)));
        assert_eq!(pef.succ(43), None);
        assert_eq!(pef.pred(42), Some((0, 42)));
        assert_eq!(pef.pred(41), None);
    }

    #[test]
    fn test_large_sequence() {
        let values: Vec<usize> = (0..10000).map(|i| i * 3).collect();
        let n = values.len();
        let u = values[n - 1];

        let mut builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }

        assert_eq!(pef.succ(0), Some((0, 0)));
        assert_eq!(pef.succ(1), Some((1, 3)));
        assert_eq!(pef.succ(29997), Some((9999, 29997)));
        assert_eq!(pef.succ(29998), None);

        assert_eq!(pef.pred(29997), Some((9999, 29997)));
        assert_eq!(pef.pred(29996), Some((9998, 29994)));
        assert_eq!(pef.pred(0), Some((0, 0)));
    }

    #[test]
    fn test_dense_chunks() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let universe = 2000;
        let mut rng = SmallRng::seed_from_u64(42);
        let mut values: Vec<usize> = (0..universe)
            .filter(|_| rng.random_bool(0.5))
            .take(1000)
            .collect();
        values.sort_unstable();
        values.dedup();

        let n = values.len();
        let u = *values.last().unwrap();

        let mut builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.succ(v), Some((i, v)), "succ({v})");
        }

        for i in 0..values.len() - 1 {
            if values[i] + 1 < values[i + 1] {
                assert_eq!(
                    pef.succ(values[i] + 1),
                    Some((i + 1, values[i + 1])),
                    "succ({} + 1)",
                    values[i]
                );
            }
        }
        assert_eq!(pef.succ(u + 1), None);

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.pred(v), Some((i, v)), "pred({v})");
        }

        for i in 1..values.len() {
            if values[i] - 1 > values[i - 1] {
                assert_eq!(
                    pef.pred(values[i] - 1),
                    Some((i - 1, values[i - 1])),
                    "pred({} - 1)",
                    values[i]
                );
            }
        }
    }

    #[test]
    fn test_space_improvement() {
        use mem_dbg::MemSize;
        use crate::dict::part_elias_fano::{PartEliasFano, PartEliasFanoBuilder};

        let values: Vec<usize> = (0..10000).map(|i| i * 3).collect();
        let n = values.len();
        let u = values[n - 1];

        // Build old PEF
        let mut old_builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            old_builder.push(v);
        }
        let old_pef: PartEliasFano = old_builder.build();
        let old_size = old_pef.mem_size(mem_dbg::SizeFlags::default());

        // Build flat PEF
        let mut flat_builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            flat_builder.push(v);
        }
        let flat_pef = flat_builder.build();
        let flat_size = flat_pef.mem_size(mem_dbg::SizeFlags::default());

        // Flat should be smaller (or at least not much larger)
        println!(
            "Old PEF: {} bytes, Flat PEF: {} bytes, ratio: {:.3}",
            old_size,
            flat_size,
            flat_size as f64 / old_size as f64
        );
    }

    #[test]
    fn test_skewed_distribution() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let n = 100_000;
        let mut rng = SmallRng::seed_from_u64(123);
        let mut values: Vec<usize> = Vec::with_capacity(n);
        // 25% in [0, 100), 25% in [100, 10000), 50% in [10000, 1000000)
        for _ in 0..n / 4 {
            values.push(rng.random_range(0..100));
        }
        for _ in 0..n / 4 {
            values.push(rng.random_range(100..10000));
        }
        for _ in 0..n / 2 {
            values.push(rng.random_range(10000..1_000_000));
        }
        values.sort_unstable();

        let u = *values.last().unwrap();
        let mut builder = FlatPartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        // Verify all gets
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }

        // Verify succ at random positions
        for _ in 0..1000 {
            let v = rng.random_range(0..u + 1);
            let expected = values.partition_point(|&x| x < v);
            if expected < n {
                assert_eq!(
                    pef.succ(v),
                    Some((expected, values[expected])),
                    "succ({v})"
                );
            } else {
                assert_eq!(pef.succ(v), None, "succ({v}) should be None");
            }
        }

        // Verify pred at random positions
        for _ in 0..1000 {
            let v = rng.random_range(0..u + 1);
            let idx = values.partition_point(|&x| x <= v);
            if idx > 0 {
                assert_eq!(
                    pef.pred(v),
                    Some((idx - 1, values[idx - 1])),
                    "pred({v})"
                );
            } else {
                assert_eq!(pef.pred(v), None, "pred({v}) should be None");
            }
        }

        println!(
            "Skewed test: n={}, partitions={}, bpe={:.2}",
            n,
            pef.num_partitions(),
            (pef.mem_size(mem_dbg::SizeFlags::default()) * 8) as f64 / n as f64
        );
    }
}
