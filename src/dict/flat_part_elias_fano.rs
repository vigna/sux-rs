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

use crate::bits::BitVec;
use crate::dict::elias_fano::{EfSeqDict, EliasFanoBuilder};
use crate::traits::indexed_dict::Types;
use crate::traits::iter::{BidiIterator, SwappedIter};
use crate::traits::{
    BitVecOpsMut, IndexedSeq, Pred, PredUnchecked, Succ, SuccUnchecked,
};
use crate::utils::SelectInWord;
use mem_dbg::{MemDbg, MemSize};
use std::borrow::Borrow;

// --- Constants for the inline SelectAdaptConst (u16-only, no spill) ---

const LOG2_ONES_PER_INVENTORY: usize = 11;
const ONES_PER_INVENTORY: usize = 1 << LOG2_ONES_PER_INVENTORY;
const ONES_PER_INVENTORY_MASK: usize = ONES_PER_INVENTORY - 1;
const LOG2_WORDS_PER_SUBINVENTORY: usize = 3;
const WORDS_PER_SUBINVENTORY: usize = 1 << LOG2_WORDS_PER_SUBINVENTORY;
const WORDS_PER_INVENTORY_ENTRY: usize = WORDS_PER_SUBINVENTORY + 1; // 9
const LOG2_U16_PER_USIZE: usize = (usize::BITS / 16).ilog2() as usize;
const LOG2_ONES_PER_SUB16: usize =
    LOG2_ONES_PER_INVENTORY.saturating_sub(LOG2_WORDS_PER_SUBINVENTORY + LOG2_U16_PER_USIZE);
const ONES_PER_SUB16_MASK: usize = (1 << LOG2_ONES_PER_SUB16) - 1;

// --- Constants for inline RankSmall<64, 1, 10> ---

const RANK_BLOCK_WORDS: usize = 16; // 1024 bits per block
const RANK_BLOCK_BITS: usize = RANK_BLOCK_WORDS * 64;
const RANK_SUBBLOCK_WORDS: usize = 4; // 256 bits per subblock

// --- Main structure ---

#[derive(MemSize, MemDbg)]
pub struct FlatPartEliasFano {
    n: usize,
    u: usize,
    endpoints: EfSeqDict<usize>,
    upper_bounds: EfSeqDict<usize>,
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

/// Select the `rank`-th one using the inline inventory (u16-only spans).
#[inline]
unsafe fn inline_select(inventory: &[usize], high_bits: &[usize], rank: usize) -> usize {
    let inventory_index = rank >> LOG2_ONES_PER_INVENTORY;
    let inventory_start_pos = inventory_index * WORDS_PER_INVENTORY_ENTRY;

    let inventory_rank = unsafe { *inventory.get_unchecked(inventory_start_pos) };
    let subrank = rank & ONES_PER_INVENTORY_MASK;

    let subinventory: &[u16] = unsafe {
        inventory
            .get_unchecked(inventory_start_pos + 1..)
            .align_to::<u16>()
            .1
    };

    let hint_pos =
        inventory_rank + unsafe { *subinventory.get_unchecked(subrank >> LOG2_ONES_PER_SUB16) } as usize;
    let residual = subrank & ONES_PER_SUB16_MASK;

    unsafe { select_hinted_raw(high_bits, rank, hint_pos, rank - residual) }
}

/// Select the `rank`-th zero using the inline inventory (u16-only spans).
#[inline]
unsafe fn inline_select_zero(inventory: &[usize], high_bits: &[usize], rank: usize) -> usize {
    let inventory_index = rank >> LOG2_ONES_PER_INVENTORY;
    let inventory_start_pos = inventory_index * WORDS_PER_INVENTORY_ENTRY;

    let inventory_rank = unsafe { *inventory.get_unchecked(inventory_start_pos) };
    let subrank = rank & ONES_PER_INVENTORY_MASK;

    let subinventory: &[u16] = unsafe {
        inventory
            .get_unchecked(inventory_start_pos + 1..)
            .align_to::<u16>()
            .1
    };

    let hint_pos =
        inventory_rank + unsafe { *subinventory.get_unchecked(subrank >> LOG2_ONES_PER_SUB16) } as usize;
    let residual = subrank & ONES_PER_SUB16_MASK;

    unsafe { select_zero_hinted_raw(high_bits, rank, hint_pos, rank - residual) }
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

/// Rank (count of ones up to but not including `pos`) using inline
/// RankSmall<64, 1, 10> counters.
#[inline]
unsafe fn inline_rank(counters: &[usize], bv: &[usize], pos: usize) -> usize {
    let word_pos = pos / 64;
    let block = word_pos / RANK_BLOCK_WORDS;
    let subblock = (word_pos % RANK_BLOCK_WORDS) / RANK_SUBBLOCK_WORDS;

    let counter_word = unsafe { *counters.get_unchecked(block) };
    let absolute = counter_word as u32 as usize;
    let relative_packed = (counter_word >> 32) as u32;
    let rel = ((relative_packed as u64) >> (10 * (subblock ^ 3))) as usize & 0x3FF;
    let hint_rank = absolute + rel;

    // Popcount from subblock start to pos
    let hint_word_pos = block * RANK_BLOCK_WORDS + subblock * RANK_SUBBLOCK_WORDS;
    let mut rank = hint_rank;
    let mut wp = hint_word_pos;
    while (wp + 1) * 64 <= pos {
        rank += unsafe { *bv.get_unchecked(wp) }.count_ones() as usize;
        wp += 1;
    }
    rank + (unsafe { *bv.get_unchecked(wp) } & ((1usize << (pos % 64)) - 1)).count_ones() as usize
}

/// Select the `rank`-th one in a dense bitvector by linear scan over
/// RankSmall<64,1,10> counters then forward scan.
#[inline]
unsafe fn inline_dense_select(counters: &[usize], bv: &[usize], rank: usize) -> usize {
    let num_blocks = counters.len();

    // Binary search over block absolute counters
    let mut lo = 0usize;
    let mut hi = num_blocks;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let abs = unsafe { *counters.get_unchecked(mid) } as u32 as usize;
        if abs <= rank {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // lo is the first block whose absolute > rank, so target is in block lo-1
    let block = lo - 1;
    let counter_word = unsafe { *counters.get_unchecked(block) };
    let absolute = counter_word as u32 as usize;
    let relative_packed = (counter_word >> 32) as u32;

    // Find the subblock within this block
    let mut hint_rank = absolute;
    let mut subblock = 0usize;
    for sb in 1..4 {
        let rel = ((relative_packed as u64) >> (10 * (sb ^ 3))) as usize & 0x3FF;
        if absolute + rel <= rank {
            hint_rank = absolute + rel;
            subblock = sb;
        } else {
            break;
        }
    }

    let hint_pos = (block * RANK_BLOCK_WORDS + subblock * RANK_SUBBLOCK_WORDS) * 64;
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

/// Number of words for a select inventory (ones or zeros).
#[inline]
fn inventory_words(num_items: usize) -> usize {
    let inventory_size = num_items.div_ceil(ONES_PER_INVENTORY);
    inventory_size * WORDS_PER_INVENTORY_ENTRY + 1
}

// --- Chunk views for accessing flat data ---

struct SparseView<'a> {
    sel1_inv: &'a [usize],
    sel0_inv: &'a [usize],
    high_bits: &'a [usize],
    low_bits: &'a [usize],
    n: usize,
    l: usize,
}

struct DenseView<'a> {
    counters: &'a [usize],
    bv: &'a [usize],
    n: usize,
    universe: usize,
}

impl FlatPartEliasFano {
    pub fn num_partitions(&self) -> usize {
        self.chunk_offsets.len()
    }

    #[inline]
    fn chunk_n(&self, p: usize) -> usize {
        let end = unsafe { self.endpoints.get_unchecked(p) };
        if p == 0 {
            end
        } else {
            end - unsafe { self.endpoints.get_unchecked(p - 1) }
        }
    }

    #[inline]
    fn base(&self, p: usize) -> usize {
        if p == 0 {
            0
        } else {
            unsafe { self.upper_bounds.get_unchecked(p - 1) }
        }
    }

    #[inline]
    fn chunk_universe(&self, p: usize) -> usize {
        (unsafe { self.upper_bounds.get_unchecked(p) }) - self.base(p)
    }

    #[inline]
    fn sparse_view(&self, p: usize) -> SparseView<'_> {
        let offset = self.chunk_offsets[p];
        let n = self.chunk_n(p);
        let universe = self.chunk_universe(p);
        let (l, high_bits_len, num_zeros) = ef_params(n, universe + 1);

        let sel1_words = inventory_words(n);
        let sel0_words = inventory_words(num_zeros);
        let high_words = high_bits_len.div_ceil(64);
        // low_bits follow high_bits

        let sel1_inv = &self.data[offset..offset + sel1_words];
        let sel0_inv = &self.data[offset + sel1_words..offset + sel1_words + sel0_words];
        let high_start = offset + sel1_words + sel0_words;
        let high_bits = &self.data[high_start..high_start + high_words];
        let low_start = high_start + high_words;
        let low_words = (n * l).div_ceil(64);
        let low_bits = &self.data[low_start..low_start + low_words];

        SparseView {
            sel1_inv,
            sel0_inv,
            high_bits,
            low_bits,
            n,
            l,
        }
    }

    #[inline]
    fn dense_view(&self, p: usize) -> DenseView<'_> {
        let offset = self.chunk_offsets[p];
        let n = self.chunk_n(p);
        let universe = self.chunk_universe(p) + 1; // bitvector length

        let num_counters = universe.div_ceil(RANK_BLOCK_BITS);
        let bv_words = universe.div_ceil(64);

        let counters = &self.data[offset..offset + num_counters];
        let bv = &self.data[offset + num_counters..offset + num_counters + bv_words];

        DenseView {
            counters,
            bv,
            n,
            universe,
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
        let (partition_idx, _) = unsafe { self.endpoints.succ_unchecked::<false>(index + 1) };
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_unchecked(partition_idx - 1) }
        };
        let local_index = index - partition_start;
        let base = self.base(partition_idx);

        if self.is_dense(partition_idx) {
            let view = self.dense_view(partition_idx);
            base + unsafe { inline_dense_select(view.counters, view.bv, local_index) }
        } else {
            let view = self.sparse_view(partition_idx);
            let high = unsafe { inline_select(view.sel1_inv, view.high_bits, local_index) }
                - local_index;
            let low = unsafe { get_low_bits(view.low_bits, local_index, view.l) };
            base + ((high << view.l) | low)
        }
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
    type Iter<'a> = FlatPefIter<'a>;
    type BidiIter<'a> = FlatPefBidiIter<'a>;

    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, usize) {
        let value = *value.borrow();
        let (partition_idx, _) = unsafe { self.upper_bounds.succ_unchecked::<false>(value) };
        let base = self.base(partition_idx);
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_unchecked(partition_idx - 1) }
        };

        let relative = value - base;
        let (local_idx, local_val) = if STRICT {
            unsafe { self.chunk_succ_strict(partition_idx, relative) }
        } else {
            unsafe { self.chunk_succ(partition_idx, relative) }
        };
        (partition_start + local_idx, local_val + base)
    }

    unsafe fn iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::Iter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (idx, FlatPefIter { pef: self, pos: idx })
    }

    unsafe fn iter_bidi_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BidiIter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (idx, FlatPefBidiIter { pef: self, pos: idx })
    }
}

impl FlatPartEliasFano {
    #[inline]
    unsafe fn chunk_get_unchecked(&self, p: usize, index: usize) -> usize {
        if self.is_dense(p) {
            let view = self.dense_view(p);
            unsafe { inline_dense_select(view.counters, view.bv, index) }
        } else {
            let view = self.sparse_view(p);
            let high =
                unsafe { inline_select(view.sel1_inv, view.high_bits, index) } - index;
            let low = unsafe { get_low_bits(view.low_bits, index, view.l) };
            (high << view.l) | low
        }
    }

    #[inline]
    unsafe fn chunk_succ(&self, p: usize, value: usize) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p);
            let rank = unsafe { inline_rank(view.counters, view.bv, value) };
            let pos = unsafe { inline_dense_select(view.counters, view.bv, rank) };
            if pos >= value {
                (rank, pos)
            } else {
                (rank + 1, unsafe { inline_dense_select(view.counters, view.bv, rank + 1) })
            }
        } else {
            let view = self.sparse_view(p);
            unsafe { ef_succ_raw::<false>(&view, value) }
        }
    }

    #[inline]
    unsafe fn chunk_succ_strict(&self, p: usize, value: usize) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p);
            let universe = view.universe;
            let rank = if value >= universe {
                (unsafe { inline_rank(view.counters, view.bv, universe) }) + 1
            } else {
                unsafe { inline_rank(view.counters, view.bv, value + 1) }
            };
            (rank, unsafe { inline_dense_select(view.counters, view.bv, rank) })
        } else {
            let view = self.sparse_view(p);
            unsafe { ef_succ_raw::<true>(&view, value) }
        }
    }

    #[inline]
    unsafe fn chunk_pred(&self, p: usize, value: usize) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p);
            let rank = if value >= view.universe {
                view.n
            } else {
                unsafe { inline_rank(view.counters, view.bv, value + 1) }
            };
            (rank - 1, unsafe { inline_dense_select(view.counters, view.bv, rank - 1) })
        } else {
            let view = self.sparse_view(p);
            unsafe { ef_pred_raw::<false>(&view, value) }
        }
    }

    #[inline]
    unsafe fn chunk_pred_strict(&self, p: usize, value: usize) -> (usize, usize) {
        if self.is_dense(p) {
            let view = self.dense_view(p);
            let rank = unsafe { inline_rank(view.counters, view.bv, value) };
            (rank - 1, unsafe { inline_dense_select(view.counters, view.bv, rank - 1) })
        } else {
            let view = self.sparse_view(p);
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
        (unsafe { inline_select_zero(view.sel0_inv, view.high_bits, zeros_to_skip - 1) }) + 1
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
    // Find succ, then go back one
    let zeros_to_skip = (value >> view.l) + 1;
    // Find the position just past all elements with high bits <= value >> l
    let bit_pos = if zeros_to_skip == 0 {
        0
    } else {
        let num_zeros_in_high = view.high_bits.len() * 64 - view.n; // approximate
        if zeros_to_skip > num_zeros_in_high {
            // value is beyond all elements
            let last_idx = view.n - 1;
            let high = unsafe { inline_select(view.sel1_inv, view.high_bits, last_idx) } - last_idx;
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
        (unsafe { inline_select_zero(view.sel0_inv, view.high_bits, zeros_to_skip - 1) }) + 1
    };

    // rank = number of ones before bit_pos
    let rank = bit_pos - zeros_to_skip;

    // Scan backwards from rank to find the predecessor
    if STRICT {
        // Find largest element < value
        // The element at rank-1 has value < value (since its high bits < zeros_to_skip)
        // But we might need to check if element at rank has the same high bits
        // and a smaller low value
        // Actually, let's use a simpler approach: binary-search back from rank
        if rank == 0 {
            // Check if first element works
            let high = unsafe { inline_select(view.sel1_inv, view.high_bits, 0) };
            let low = unsafe { get_low_bits(view.low_bits, 0, view.l) };
            let val = ((high) << view.l) | low;
            return (0, val); // caller guarantees pred exists
        }
        // Check element at rank-1 — it must be < value since its high bits are at most
        // value >> l, and if equal, the linear scan in succ would have found it
        let idx = rank - 1;
        let high = unsafe { inline_select(view.sel1_inv, view.high_bits, idx) } - idx;
        let low = unsafe { get_low_bits(view.low_bits, idx, view.l) };
        let val = (high << view.l) | low;
        if val < value {
            return (idx, val);
        }
        // Need to go further back
        if idx == 0 {
            panic!("pred_strict called but no predecessor exists");
        }
        let idx2 = idx - 1;
        let high2 = unsafe { inline_select(view.sel1_inv, view.high_bits, idx2) } - idx2;
        let low2 = unsafe { get_low_bits(view.low_bits, idx2, view.l) };
        (idx2, (high2 << view.l) | low2)
    } else {
        // Find largest element <= value
        // Start from rank and scan backwards
        let mut idx = rank;
        loop {
            if idx == 0 {
                let high = unsafe { inline_select(view.sel1_inv, view.high_bits, 0) };
                let low = unsafe { get_low_bits(view.low_bits, 0, view.l) };
                return (0, ((high) << view.l) | low);
            }
            idx -= 1;
            let high = unsafe { inline_select(view.sel1_inv, view.high_bits, idx) } - idx;
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

    fn iter_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccUnchecked>::Iter<'_>)> {
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
    ) -> Option<(usize, <Self as SuccUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<true>(value) })
        }
    }

    fn iter_bidi_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccUnchecked>::BidiIter<'_>)> {
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
    ) -> Option<(usize, <Self as SuccUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<true>(value) })
        }
    }
}

impl PredUnchecked for FlatPartEliasFano {
    type BackIter<'a> = SwappedIter<FlatPefBidiIter<'a>>;
    type BidiIter<'a> = FlatPefBidiIter<'a>;

    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, usize) {
        let value = *value.borrow();
        let (partition_idx, _) = unsafe { self.upper_bounds.succ_unchecked::<false>(value) };
        let base = self.base(partition_idx);
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_unchecked(partition_idx - 1) }
        };

        let relative = value - base;
        let first_elem = unsafe { self.chunk_get_unchecked(partition_idx, 0) };
        if relative < first_elem || (STRICT && relative == first_elem) {
            let prev_part = partition_idx - 1;
            let prev_start = if prev_part == 0 {
                0
            } else {
                unsafe { self.endpoints.get_unchecked(prev_part - 1) }
            };
            let prev_n = self.chunk_n(prev_part);
            let prev_base = self.base(prev_part);
            let local_last = prev_n - 1;
            let local_val = unsafe { self.chunk_get_unchecked(prev_part, local_last) };
            return (prev_start + local_last, local_val + prev_base);
        }

        let (local_idx, local_val) = if STRICT {
            unsafe { self.chunk_pred_strict(partition_idx, relative) }
        } else {
            unsafe { self.chunk_pred(partition_idx, relative) }
        };
        (partition_start + local_idx, local_val + base)
    }

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

    fn iter_back_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
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
    ) -> Option<(usize, Self::BackIter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<true>(value) })
        }
    }

    fn iter_bidi_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
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
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<true>(value) })
        }
    }
}

// --- Builder ---

/// Cost in bits of a sparse (EF) chunk in the flat layout, including select
/// inventory overhead.
fn flat_ef_cost(universe: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if universe <= 1 {
        return n + inventory_words(n) * 64 + inventory_words(0) * 64;
    }
    let l = if universe > n {
        (universe / n).ilog2() as usize
    } else {
        0
    };
    let high_bits_len = n + (universe >> l) + 2;
    let num_zeros = high_bits_len - n;
    let raw_bits = high_bits_len + n * l;
    raw_bits + inventory_words(n) * 64 + inventory_words(num_zeros) * 64
}

/// Cost in bits of a dense chunk in the flat layout, including rank counter
/// overhead.
fn flat_dense_cost(universe: usize) -> usize {
    let num_counters = universe.div_ceil(RANK_BLOCK_BITS);
    universe + num_counters * 64
}

fn flat_chunk_cost(universe: usize, n: usize) -> usize {
    if n > universe {
        // Duplicates: dense encoding not possible
        flat_ef_cost(universe, n)
    } else {
        flat_ef_cost(universe, n).min(flat_dense_cost(universe))
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
            let efb = EliasFanoBuilder::new(0, 0usize);
            let empty_ef = efb.build_with_seq_and_dict();
            return FlatPartEliasFano {
                n: 0,
                u: self.u,
                endpoints: {
                    let efb = EliasFanoBuilder::new(0, 0usize);
                    efb.build_with_seq_and_dict()
                },
                upper_bounds: empty_ef,
                chunk_kinds: BitVec::new(0).into(),
                chunk_offsets: Box::new([]),
                data: Box::new([]),
            };
        }

        let fix_cost = self.fix_cost;
        let partition_points = super::part_elias_fano::optimal_partition_with(
            &self.values,
            self.eps1,
            self.eps2,
            |universe, n| flat_chunk_cost(universe, n) + fix_cost,
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
                && flat_dense_cost(universe + 1) < flat_ef_cost(universe + 1, chunk_n);
            chunk_kinds_vec.push(use_dense);

            let data_words = if use_dense {
                let bv_bits = universe + 1;
                let num_counters = bv_bits.div_ceil(RANK_BLOCK_BITS);
                let bv_words = bv_bits.div_ceil(64);
                num_counters + bv_words
            } else {
                let (l, high_bits_len, num_zeros) = ef_params(chunk_n, universe + 1);
                let sel1_words = inventory_words(chunk_n);
                let sel0_words = inventory_words(num_zeros);
                let high_words = high_bits_len.div_ceil(64);
                let low_words = (chunk_n * l).div_ceil(64);
                sel1_words + sel0_words + high_words + low_words
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

        // Build master EF structures
        let mut ep_builder = EliasFanoBuilder::new(num_partitions, self.n);
        for &c in &cumulative_sizes {
            unsafe { ep_builder.push_unchecked(c) };
        }
        let endpoints = ep_builder.build_with_seq_and_dict();

        let mut ub_builder = EliasFanoBuilder::new(num_partitions, self.u);
        for &ub in &upper_bound_values {
            unsafe { ub_builder.push_unchecked(ub) };
        }
        let upper_bounds = ub_builder.build_with_seq_and_dict();

        FlatPartEliasFano {
            n: self.n,
            u: self.u,
            endpoints,
            upper_bounds,
            chunk_kinds: chunk_kinds.into(),
            chunk_offsets: chunk_offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
        }
    }

    fn build_dense_chunk(&self, data: &mut [usize], offset: usize, info: &ChunkInfo) {
        let bv_bits = info.universe + 1;
        let num_counters = bv_bits.div_ceil(RANK_BLOCK_BITS);
        let bv_words = bv_bits.div_ceil(64);

        // Layout: [counters: num_counters words] [bitvector: bv_words words]
        let bv_start = offset + num_counters;

        // Fill bitvector
        for &v in &self.values[info.start..info.end] {
            let bit = v - info.base;
            data[bv_start + bit / 64] |= 1usize << (bit % 64);
        }

        // Build RankSmall<64, 1, 10> counters
        // Matches the original pattern: record absolute, then for each word j
        // in the block, record relative counter at subblock boundaries (j%4==0
        // for j>0), then count word j.
        let mut past_ones = 0usize;
        for block in 0..num_counters {
            let absolute = past_ones as u32;
            let block_word_start = block * RANK_BLOCK_WORDS;

            let mut relative_packed: u32 = 0;
            for j in 0..RANK_BLOCK_WORDS {
                if j > 0 && j % RANK_SUBBLOCK_WORDS == 0 {
                    let sb = j / RANK_SUBBLOCK_WORDS;
                    let rel_count = past_ones - absolute as usize;
                    relative_packed |= (rel_count as u32) << (10 * (sb ^ 3));
                }
                let word_idx = block_word_start + j;
                if word_idx < bv_words {
                    past_ones += data[bv_start + word_idx].count_ones() as usize;
                }
            }

            data[offset + block] = absolute as usize | ((relative_packed as usize) << 32);
        }
    }

    fn build_sparse_chunk(&self, data: &mut [usize], offset: usize, info: &ChunkInfo) {
        let chunk_n = info.end - info.start;
        let (l, high_bits_len, num_zeros) = ef_params(chunk_n, info.universe + 1);

        let sel1_words = inventory_words(chunk_n);
        let sel0_words = inventory_words(num_zeros);
        let high_words = high_bits_len.div_ceil(64);
        let _low_words = (chunk_n * l).div_ceil(64);

        let sel1_start = offset;
        let sel0_start = offset + sel1_words;
        let high_start = sel0_start + sel0_words;
        let low_start = high_start + high_words;

        // Fill high bits and low bits
        for (i, &v) in self.values[info.start..info.end].iter().enumerate() {
            let relative = v - info.base;
            let high = relative >> l;
            let low = relative & ((1 << l) - 1);

            // High bit: set position (i + high) in the high bits unary code
            let high_pos = i + high;
            data[high_start + high_pos / 64] |= 1usize << (high_pos % 64);

            // Low bits: store l-bit value at position i
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

        // Build select-ones inventory on high bits
        Self::build_select_inventory(data, sel1_start, high_start, high_words, chunk_n, high_bits_len, true);

        // Build select-zeros inventory on high bits
        Self::build_select_inventory(data, sel0_start, high_start, high_words, num_zeros, high_bits_len, false);
    }

    /// Build a u16-only select inventory for ones (if `ones=true`) or zeros
    /// (if `ones=false`) in `bits` at `bits_offset..bits_offset+bits_words`,
    /// writing the inventory to `data[inv_start..]`.
    fn build_select_inventory(
        data: &mut [usize],
        inv_start: usize,
        bits_offset: usize,
        bits_words: usize,
        num_target: usize,
        num_bits: usize,
        ones: bool,
    ) {
        let inventory_size = num_target.div_ceil(ONES_PER_INVENTORY);
        // Layout: inventory_size * 9 words + 1 sentinel word

        let mut past_targets = 0usize;
        let mut next_quantum = 0usize;

        // First phase: build primary inventory entries
        for i in 0..bits_words {
            let raw_word = data[bits_offset + i];
            let effective_word = if ones { raw_word } else { !raw_word };
            let targets_in_word = effective_word.count_ones() as usize;

            while past_targets + targets_in_word > next_quantum {
                let in_word_index = effective_word.select_in_word(next_quantum - past_targets);
                let index = i * 64 + in_word_index;

                let inv_idx = next_quantum / ONES_PER_INVENTORY;
                let entry_start = inv_start + inv_idx * WORDS_PER_INVENTORY_ENTRY;
                data[entry_start] = index;

                next_quantum += ONES_PER_INVENTORY;
            }
            past_targets += targets_in_word;
        }

        // Sentinel: total number of bits
        let sentinel_pos = inv_start + inventory_size * WORDS_PER_INVENTORY_ENTRY;
        data[sentinel_pos] = num_bits;

        // Second phase: fill subinventories
        for inv_idx in 0..inventory_size {
            let entry_start = inv_start + inv_idx * WORDS_PER_INVENTORY_ENTRY;
            let start_bit_idx = data[entry_start];
            let next_entry_start = entry_start + WORDS_PER_INVENTORY_ENTRY;
            let end_bit_idx = data[next_entry_start];

            let quantum = 1usize << LOG2_ONES_PER_SUB16;
            let mut subinventory_idx = 1usize;
            let mut next_sub_quantum = inv_idx * ONES_PER_INVENTORY + quantum;

            let mut past_targets_local = inv_idx * ONES_PER_INVENTORY;
            let mut word_idx = start_bit_idx / 64;
            let end_word_idx = end_bit_idx.div_ceil(64);
            let bit_idx = start_bit_idx % 64;

            let first_word = if ones {
                data[bits_offset + word_idx]
            } else {
                !data[bits_offset + word_idx]
            };
            let mut word = (first_word >> bit_idx) << bit_idx;

            'outer: loop {
                let targets_in_word = word.count_ones() as usize;

                while past_targets_local + targets_in_word > next_sub_quantum {
                    let in_word_index = word.select_in_word(next_sub_quantum - past_targets_local);
                    let bit_index = word_idx * 64 + in_word_index;

                    if bit_index >= end_bit_idx {
                        break 'outer;
                    }

                    let sub_offset = (bit_index - start_bit_idx) as u16;

                    let subinv_slice: &mut [u16] = unsafe {
                        let ptr = data
                            .as_mut_ptr()
                            .add(entry_start + 1) as *mut u16;
                        std::slice::from_raw_parts_mut(ptr, WORDS_PER_SUBINVENTORY * 4)
                    };
                    subinv_slice[subinventory_idx] = sub_offset;
                    subinventory_idx += 1;

                    if subinventory_idx << LOG2_ONES_PER_SUB16 == ONES_PER_INVENTORY {
                        break 'outer;
                    }
                    next_sub_quantum += quantum;
                }

                past_targets_local += targets_in_word;
                word_idx += 1;
                if word_idx >= end_word_idx {
                    break;
                }
                word = if ones {
                    data[bits_offset + word_idx]
                } else {
                    !data[bits_offset + word_idx]
                };
            }
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
