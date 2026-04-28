/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Prefix-free codes for [`CompVFunc`].
//!
//! Symbols are parametric over a symbol type `W` (any [`PrimitiveInteger`]). A
//! [`Coder<W>`] turns a symbol into a codeword (or flags it as escaped); a
//! [`Decoder<W>`] turns the high bits of a [`max_codeword_len`]-bit window
//! back into a symbol (or returns `None` for the escape codeword).
//!
//! [`max_codeword_len`] is the depth of the prefix code tree (at most
//! `usize::BITS` - 2 bits). Escaped symbols carry an additional literal field
//! whose width is [`escaped_symbols_len`].
//!
//! [`CompVFunc`]: crate::func::CompVFunc
//! [`PrimitiveInteger`]: num_primitive::PrimitiveInteger
//! [`max_codeword_len`]: Decoder::max_codeword_len
//! [`escaped_symbols_len`]: Decoder::escaped_symbols_len

use mem_dbg::{MemDbg, MemSize};
use num_primitive::PrimitiveInteger;
use std::collections::HashMap;
use std::hash::Hash;

/// A factory for a [`Coder<W>`], given a frequency map.
///
/// Implementations represent a *family* of codes (e.g. Huffman); the
/// concrete code is built per data distribution.
pub trait Codec<W> {
    type Coder: Coder<W>;
    /// Builds a coder for the given symbol → frequency map.
    ///
    /// All frequencies must be strictly positive. The empty map is
    /// allowed and yields a degenerate coder.
    fn build_coder(&self, frequencies: &HashMap<W, usize>) -> Self::Coder;
}

/// A prefix-free encoder for `W` symbols.
pub trait Coder<W> {
    type Decoder: Decoder<W>;

    /// Returns the codeword for `symbol`, or `None` if `symbol` must be
    /// escaped.
    ///
    /// The returned codeword is [`codeword_len(symbol)`] bits wide and
    /// is laid out so that bit *l* of the returned `usize` is the *l*-th
    /// bit appended to the data array. With the layout used by
    /// [`CompVFunc`], that means bit 0 is the most significant bit of
    /// the canonical (MSB-first) codeword.
    ///
    /// [`CompVFunc`]: crate::func::CompVFunc
    /// [`codeword_len(symbol)`]: Self::codeword_len
    fn encode(&self, symbol: W) -> Option<usize>;

    /// Returns the prefix code length in bits for `symbol`.
    ///
    /// For escaped symbols this is [`max_codeword_len`](Self::max_codeword_len).
    fn codeword_len(&self, symbol: W) -> u32;

    /// Returns the total encoded length in bits for `symbol`.
    ///
    /// For non-escaped symbols this equals [`codeword_len`]. For escaped
    /// symbols this is [`max_codeword_len`] + [`escaped_symbols_len`].
    ///
    /// [`max_codeword_len`]: Self::max_codeword_len
    /// [`escaped_symbols_len`]: Self::escaped_symbols_len
    /// [`codeword_len`]: Self::codeword_len
    fn encoded_len(&self, symbol: W) -> u32;

    /// The maximum prefix code length across all symbols.
    fn max_codeword_len(&self) -> u32;

    /// The escape codeword (length is [`max_codeword_len`](Self::max_codeword_len)).
    fn escape_codeword(&self) -> usize;

    /// The length in bits of escaped symbols, or zero if the code has
    /// no escape.
    fn escaped_symbols_len(&self) -> u32;

    /// Builds the matching decoder.
    fn into_decoder(self) -> Self::Decoder;
}

/// A prefix-free decoder for `W` symbols.
pub trait Decoder<W> {
    /// Decodes the prefix codeword found in the high bits of a
    /// [`max_codeword_len`](Self::max_codeword_len)-bit window.
    ///
    /// The first bit of the codeword is at bit position
    /// `max_codeword_len - 1` of `value`; lower bits hold whatever
    /// follows in the data and are masked off by the canonical-Huffman
    /// algorithm.
    ///
    /// Returns `Some(symbol)` for a normal codeword, or `None` if
    /// the codeword is the escape codeword (the caller must then
    /// read the literal value separately).
    fn decode(&self, value: usize) -> Option<W>;

    /// The maximum prefix code length in bits. This is the width of
    /// the read window expected by [`decode`](Self::decode).
    fn max_codeword_len(&self) -> u32;

    /// The length in bits of escaped symbols, or zero if the code has
    /// no escape.
    fn escaped_symbols_len(&self) -> u32;
}

// ── ZeroCodec ───────────────────────────────────────────────────────

/// Degenerate codec that always emits length-0 codewords.
///
/// Used when the value distribution has a single distinct value (or
/// none): every key resolves to that value (or zero), so no bits need
/// to be stored.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroCodec;

#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroCoder;

#[derive(Debug, Clone, Copy, Default, MemSize, MemDbg)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ZeroDecoder;

impl<W: PrimitiveInteger + Hash> Codec<W> for ZeroCodec {
    type Coder = ZeroCoder;
    fn build_coder(&self, _frequencies: &HashMap<W, usize>) -> ZeroCoder {
        ZeroCoder
    }
}

impl<W: PrimitiveInteger> Coder<W> for ZeroCoder {
    type Decoder = ZeroDecoder;
    fn encode(&self, _symbol: W) -> Option<usize> {
        Some(0)
    }
    fn codeword_len(&self, _symbol: W) -> u32 {
        0
    }
    fn encoded_len(&self, _symbol: W) -> u32 {
        0
    }
    fn max_codeword_len(&self) -> u32 {
        0
    }
    fn escape_codeword(&self) -> usize {
        0
    }
    fn escaped_symbols_len(&self) -> u32 {
        0
    }
    fn into_decoder(self) -> ZeroDecoder {
        ZeroDecoder
    }
}

impl<W: PrimitiveInteger> Decoder<W> for ZeroDecoder {
    fn decode(&self, _value: usize) -> Option<W> {
        Some(W::default())
    }
    fn max_codeword_len(&self) -> u32 {
        0
    }
    fn escaped_symbols_len(&self) -> u32 {
        0
    }
}

/// A length-limited canonical Huffman codec.
///
/// Lengths are computed with the in-place Moffat–Katajainen algorithm, then
/// optionally truncated according to
/// [`max_decoding_table_len`](Self::max_decoding_table_len) and
/// [`entropy_threshold`](Self::entropy_threshold) using the techniques from
/// “[Fast scalable construction of (compressed static | minimal perfect hash)
/// functions]”. Symbols beyond the cutoff are *escaped*: they share a single
/// dedicated escape codeword followed by a literal `escaped_symbol_length`-bit
/// field. The decoder uses [canonical codes].
///
/// # References
///
/// Alistair Moffat and Jyrki Katajainen. [In-place calculation of
/// minimum-redundancy codes]. In *Workshop on Algorithms and Data Structures*,
/// pp. 393–402. Berlin, Heidelberg: Springer Berlin Heidelberg, 1995.
///
/// Eugene S. Schwartz and Bruce Kallick. [Generating a canonical prefix
/// encoding]. *Communications of the ACM* 7(3), pp. 166-169, 1964.
///
/// Marco Genuzio, Giuseppe Ottaviano, and Sebastiano Vigna. [Fast scalable
/// construction of (\[compressed\] static | minimal perfect hash)
/// functions](https://doi.org/10.1016/j.ic.2020.104517). Information and
/// Computation, 273:104517, 2020.
///
/// [In-place calculation of minimum-redundancy codes]: https://dl.acm.org/doi/10.5555/645930.672864
/// [Generating a canonical prefix encoding]: https://doi.org/10.1145/363958.363991
/// [canonical codes]: https://doi.org/10.1145/363958.363991
#[derive(Debug, Clone, Copy)]
pub struct Huffman {
    /// Hard cap on the number of distinct codeword lengths kept in
    /// the decoding table. Symbols whose codeword length exceeds
    /// this limit are encoded via the escape codeword followed by
    /// a literal. Default: 20.
    pub max_decoding_table_len: usize,
    /// Cumulative-entropy fraction threshold: the table is cut once
    /// the cumulative bit length of the codewords kept exceeds this
    /// fraction of the overall bit length. Symbols beyond the cut
    /// are diverted to the escape codeword. Default: 0.9.
    pub entropy_threshold: f64,
}

impl Default for Huffman {
    fn default() -> Self {
        Self {
            max_decoding_table_len: 20,
            entropy_threshold: 0.9,
        }
    }
}

impl Huffman {
    /// New Huffman codec with sensible defaults (table length 20,
    /// entropy threshold 0.9).
    pub const fn new() -> Self {
        Self {
            max_decoding_table_len: 20,
            entropy_threshold: 0.9,
        }
    }

    /// Unlimited Huffman codec: no cap on codeword lengths, no
    /// entropy cut. All symbols get a codeword; no escape mechanism
    /// is used. This can produce very long codewords for skewed
    /// distributions.
    pub const fn unlimited() -> Self {
        Self {
            max_decoding_table_len: usize::MAX,
            entropy_threshold: 1.0,
        }
    }

    /// Huffman codec with custom limits.
    ///
    /// `max_decoding_table_len` caps the number of distinct
    /// codeword lengths in the canonical decoding table;
    /// `entropy_threshold` is the cumulative-entropy fraction
    /// beyond which infrequent symbols are diverted to the escape
    /// codeword.
    pub const fn length_limited(max_decoding_table_len: usize, entropy_threshold: f64) -> Self {
        Self {
            max_decoding_table_len,
            entropy_threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HuffmanCoder<W> {
    /// Codeword for each non-escaped symbol, plus the escape codeword at the
    /// end. Bit *l* of an entry is the *l*-th bit appended to the data, that
    /// is, bit 0 is the MSB of the canonical (MSB-first) codeword.
    codeword: Box<[usize]>,
    /// Length of each entry in [`Self::codeword`]. The last entry is
    /// the escape length.
    codeword_len: Box<[u32]>,
    /// The symbols, in order of decreasing frequency, kept up to the
    /// cutpoint. The last entry is the escape sentinel (`W::MAX`).
    symbol: Box<[W]>,
    /// Inverse map: symbol → rank position in [`Self::symbol`]. Symbols
    /// not in the map are escaped.
    symbol_to_rank: HashMap<W, usize>,
    /// The length in bits of the literal field for escaped symbols.
    escaped_symbols_len: u32,
    /// The length in bits of the escape codeword.
    escape_codeword_len: u32,
}

impl<W: PrimitiveInteger + Hash> Codec<W> for Huffman {
    type Coder = HuffmanCoder<W>;
    fn build_coder(&self, frequencies: &HashMap<W, usize>) -> HuffmanCoder<W> {
        build_huffman_coder(
            frequencies,
            self.max_decoding_table_len,
            self.entropy_threshold,
        )
    }
}

impl<W: PrimitiveInteger + Hash> Coder<W> for HuffmanCoder<W> {
    type Decoder = HuffmanDecoder<W>;

    fn encode(&self, symbol: W) -> Option<usize> {
        self.symbol_to_rank.get(&symbol).map(|&r| self.codeword[r])
    }

    fn codeword_len(&self, symbol: W) -> u32 {
        match self.symbol_to_rank.get(&symbol) {
            Some(&r) => self.codeword_len[r],
            None => self.escape_codeword_len,
        }
    }

    fn encoded_len(&self, symbol: W) -> u32 {
        match self.symbol_to_rank.get(&symbol) {
            Some(&r) => self.codeword_len[r],
            None => self.escape_codeword_len + self.escaped_symbols_len,
        }
    }

    #[inline(always)]
    fn max_codeword_len(&self) -> u32 {
        *self.codeword_len.last().unwrap_or(&0)
    }

    #[inline(always)]
    fn escape_codeword(&self) -> usize {
        *self.codeword.last().unwrap_or(&0)
    }

    #[inline(always)]
    fn escaped_symbols_len(&self) -> u32 {
        self.escaped_symbols_len
    }

    fn into_decoder(self) -> HuffmanDecoder<W> {
        let has_escape = self.escape_codeword_len > 0;
        let size = self.codeword.len();

        if size == 0 {
            return HuffmanDecoder {
                last_codeword_plus_one: Box::new([]),
                how_many_up_to_block: Box::new([]),
                shift: Box::new([]),
                symbol: Box::new([]),
                num_real_symbols: 0,
                max_codeword_len: 0,
                escaped_symbol_length: 0,
                branchless: false,
            };
        }

        let w = self.max_codeword_len();
        assert!(
            w <= usize::BITS - 2,
            "Codeword length must not exceed {}",
            usize::BITS - 2
        );

        // Number of distinct length blocks (plus one sentinel when
        // the code has an escape — see below).
        let mut decoding_table_length: usize = 1;
        if size > 1 {
            for i in (0..size - 1).rev() {
                debug_assert!(
                    self.codeword_len[i] <= self.codeword_len[i + 1],
                    "lengths must be non-decreasing"
                );
                if self.codeword_len[i] != self.codeword_len[i + 1] {
                    decoding_table_length += 1;
                }
            }
        }
        if has_escape {
            decoding_table_length += 1;
        }

        let mut shift: Vec<u8> = vec![0; decoding_table_length];
        let mut how_many_up_to_block: Vec<u32> = vec![0; decoding_table_length];
        let mut last_codeword_plus_one: Vec<usize> = vec![0; decoding_table_length];

        // p is the current block index; word tracks the canonical codeword
        // counter (MSB-first as an integer); prev_l is the length of the
        // previous block.
        let mut p: i32 = -1;
        let mut prev_l: u32 = 0;
        let mut last_l: u32 = 0;
        let mut word: usize = 0;

        for i in 0..size {
            let l = self.codeword_len[i];
            last_l = l;
            if l != prev_l {
                if i != 0 {
                    last_codeword_plus_one[p as usize] = word << (w - prev_l);
                    how_many_up_to_block[p as usize] = i as u32;
                }
                p += 1;
                shift[p as usize] = (w - l) as u8;
                word <<= l - prev_l;
                prev_l = l;
            }
            word += 1;
        }

        // Close the last block.
        last_codeword_plus_one[p as usize] = word << (w - prev_l);
        how_many_up_to_block[p as usize] = size as u32;
        p += 1;

        let num_real_symbols = if has_escape {
            // Sentinel block. The escape codeword is stored as all-ones
            // at the max kept length, but the canonical assignment above
            // processes it as the next sequential value. Values between
            // the sequential count and all-ones are gaps that no real
            // block covers; the sentinel catches them so the decoder
            // never falls through.
            let last_p = p as usize;
            last_codeword_plus_one[last_p] = usize::MAX >> 1;
            shift[last_p] = (usize::BITS - 1) as u8;
            how_many_up_to_block[last_p] = (size as u32).saturating_sub(1);
            (size as u32).saturating_sub(1)
        } else {
            size as u32
        };

        // Default heuristic for branchy vs branchless: `> 3` distinct length
        // blocks ⇒ branchless, otherwise branchy.
        //
        // The threshold is empirical: with one or two length classes the
        // branchy decoder is the obvious winner (perfect branch prediction);
        // with three classes it is still a wash; with four or more, we start
        // losing time to mispredictions on any non-trivial frequency skew.
        // Callers that know the codeword distribution shape can override the
        // choice with `HuffmanDecoder::branchless`.
        let branchless = decoding_table_length > 3;

        HuffmanDecoder {
            last_codeword_plus_one: last_codeword_plus_one.into_boxed_slice(),
            how_many_up_to_block: how_many_up_to_block.into_boxed_slice(),
            shift: shift.into_boxed_slice(),
            symbol: self.symbol,
            num_real_symbols,
            max_codeword_len: last_l,
            escaped_symbol_length: self.escaped_symbols_len,
            branchless,
        }
    }
}

/// Canonical Huffman decoder built from a [`HuffmanCoder`].
///
/// The [`Decoder::decode`] implementation has two strategies, selected
/// at runtime by [`Self::is_branchless`]:
///
/// * **Branchy** (the textbook canonical decoder): walk the length
///   blocks, returning at the first whose upper bound exceeds `value`.
///   Fewer iterations on average but each iteration carries a
///   data-dependent branch. Wins when codeword frequencies are very
///   skewed (the first few blocks match almost always so the branch
///   predictor is right almost always).
///
/// * **Branchless**: count how many block upper bounds are `<= value`
///   and use the resulting index. Always touches every block, but
///   contains no data-dependent branch — the compiler lowers it to a
///   tight `cset/add` (AArch64) or `setcc/add` (x86) chain. Wins when
///   the codeword distribution is broad enough to defeat branch
///   prediction.
///
/// The dispatch itself reads a single `bool` field that is constant
/// for the lifetime of the decoder, so the surrounding `if` is
/// trivially predictable.
///
/// The default is chosen at construction time based on the number of
/// length blocks (`> 3` ⇒ branchless). Use [`Self::branchless`]
/// to override.
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HuffmanDecoder<W> {
    /// For each length block, one past the last canonical codeword in
    /// the block, left-aligned to `max_codeword_len` bits. The
    /// decoder finds the matching block by comparing `value` against
    /// these upper bounds. When the code has an escape, a final
    /// sentinel entry (`usize::MAX >> 1`) catches gap values between
    /// the last sequential canonical codeword and the all-ones escape.
    last_codeword_plus_one: Box<[usize]>,
    /// Cumulative number of symbols up to and including each length
    /// block. After computing the within-block offset, the decoder adds
    /// this value to obtain the global index into [`Self::symbol`].
    how_many_up_to_block: Box<[u32]>,
    /// Right-shift amount for each length block, equal to
    /// `max_codeword_len − block_length`. Applied to both `value`
    /// and the block's upper bound before computing the within-block
    /// offset.
    shift: Box<[u8]>,
    /// Symbols in canonical order: sorted by codeword length (shortest
    /// first), then by decreasing frequency within each length group.
    /// If the code has an escape, the last entry is a placeholder
    /// sentinel (not a real symbol).
    symbol: Box<[W]>,
    /// Number of non-escape symbols. An index `≥ num_real_symbols`
    /// from the canonical decoder means the escape codeword was hit.
    num_real_symbols: u32,
    /// The maximum codeword length in bits. This is the width of the read
    /// window expected by [`Decoder::decode`].
    max_codeword_len: u32,
    /// The length in bits of the literal field for escaped symbols, or
    /// zero if the code has no escape.
    escaped_symbol_length: u32,
    /// Whether [`Decoder::decode`] uses the branchless strategy.
    branchless: bool,
}

impl<W> HuffmanDecoder<W> {
    /// Returns the current branchy/branchless dispatch flag.
    pub const fn is_branchless(&self) -> bool {
        self.branchless
    }

    /// Sets the branchy/branchless dispatch flag.
    ///
    /// Returns `&mut self` for chaining.
    pub fn branchless(&mut self, branchless: bool) -> &mut Self {
        self.branchless = branchless;
        self
    }
}

impl<W: PrimitiveInteger> Decoder<W> for HuffmanDecoder<W> {
    #[inline(always)]
    fn decode(&self, value: usize) -> Option<W> {
        // Read a single field that is constant for the life of the
        // decoder. The branch is trivially predictable (one direction
        // for every query of a given function) and the per-call cost
        // is below the noise floor.
        if self.branchless {
            self.decode_branchless(value)
        } else {
            self.decode_branchy(value)
        }
    }

    #[inline(always)]
    fn max_codeword_len(&self) -> u32 {
        self.max_codeword_len
    }

    #[inline(always)]
    fn escaped_symbols_len(&self) -> u32 {
        self.escaped_symbol_length
    }
}

impl<W: PrimitiveInteger> HuffmanDecoder<W> {
    /// Branchy decode: walks the length blocks until one matches.
    ///
    /// Wins for very skewed codeword distributions (the first one or
    /// two blocks catch almost every query).
    #[inline(always)]
    fn decode_branchy(&self, value: usize) -> Option<W> {
        let nrs = self.num_real_symbols as usize;
        for curr in 0..self.last_codeword_plus_one.len() {
            // SAFETY: `curr` is bounded by the loop range.
            unsafe {
                let lcp1 = *self.last_codeword_plus_one.get_unchecked(curr);
                if value < lcp1 {
                    let s = *self.shift.get_unchecked(curr) as u32;
                    let off = (value >> s).wrapping_sub(lcp1 >> s);
                    let idx =
                        off.wrapping_add(*self.how_many_up_to_block.get_unchecked(curr) as usize);
                    return if idx < nrs {
                        Some(*self.symbol.get_unchecked(idx))
                    } else {
                        None
                    };
                }
            }
        }
        // Without escape the code is complete, so the last block
        // covers all w-bit values. With escape, the sentinel
        // (lcp1 = usize::MAX >> 1, w <= BITS - 2) catches gap values.
        unreachable!("decoder fell through all blocks")
    }

    /// Branchless decode: counts blocks whose upper bound is `<=
    /// value` to derive the matching block index. No data-dependent
    /// branches inside the loop, at the cost of always touching every
    /// block.
    #[inline(always)]
    fn decode_branchless(&self, value: usize) -> Option<W> {
        let nrs = self.num_real_symbols as usize;
        let n = self.last_codeword_plus_one.len();
        let mut idx: usize = 0;
        for curr in 0..n {
            // SAFETY: `curr` is bounded by the loop range.
            let lcp1 = unsafe { *self.last_codeword_plus_one.get_unchecked(curr) };
            idx += (lcp1 <= value) as usize;
        }
        // Without escape the code is complete (last block's lcp1 =
        // 1 << w covers all values). With escape the sentinel
        // guarantees idx < n.
        debug_assert!(idx < n);
        unsafe {
            let lcp1 = *self.last_codeword_plus_one.get_unchecked(idx);
            let s = *self.shift.get_unchecked(idx) as u32;
            let off = (value >> s).wrapping_sub(lcp1 >> s);
            let sym_idx = off.wrapping_add(*self.how_many_up_to_block.get_unchecked(idx) as usize);
            if sym_idx < nrs {
                Some(*self.symbol.get_unchecked(sym_idx))
            } else {
                None
            }
        }
    }
}

// ── Huffman code construction ──────────────────────────────────────

fn build_huffman_coder<W: PrimitiveInteger + Hash>(
    frequencies: &HashMap<W, usize>,
    max_decoding_table_len: usize,
    entropy_threshold: f64,
) -> HuffmanCoder<W> {
    let size = frequencies.len();
    if size == 0 {
        return HuffmanCoder {
            codeword: Box::new([]),
            codeword_len: Box::new([]),
            symbol: Box::new([]),
            symbol_to_rank: HashMap::new(),
            escaped_symbols_len: 0,
            escape_codeword_len: 0,
        };
    }

    // Sort symbols by frequency (most frequent first).
    let mut symbol: Vec<W> = frequencies.keys().copied().collect();
    symbol.sort_unstable_by(|a, b| frequencies[b].cmp(&frequencies[a]).then(a.cmp(b)));

    // Moffat–Katajainen builds depths in place on a frequency array sorted
    // *ascending*. The array is reused to store parent-pointers, so it needs to
    // be `u64`-wide regardless of W.
    let mut a: Vec<u64> = vec![0; size];
    for i in 0..size {
        a[size - 1 - i] = frequencies[&symbol[i]] as u64;
    }

    let mut overall_length: u64 = 0;
    if size > 1 {
        // First pass, left to right: build sibling pointers.
        a[0] = a[0].wrapping_add(a[1]);
        let mut root: usize = 0;
        let mut leaf: usize = 2;
        for next in 1..size - 1 {
            // Select first item for a pairing.
            if leaf >= size || a[root] < a[leaf] {
                a[next] = a[root];
                a[root] = next as u64;
                root += 1;
            } else {
                a[next] = a[leaf];
                leaf += 1;
            }
            // Add on the second item.
            if leaf >= size || (root < next && a[root] < a[leaf]) {
                a[next] = a[next].wrapping_add(a[root]);
                a[root] = next as u64;
                root += 1;
            } else {
                a[next] = a[next].wrapping_add(a[leaf]);
                leaf += 1;
            }
        }

        // Second pass, right to left: internal node depths.
        // When size == 2 the tree has no internal nodes — skip.
        a[size - 2] = 0;
        for next in (0..size.saturating_sub(2)).rev() {
            a[next] = a[a[next] as usize] + 1;
        }

        // Third pass, right to left: leaf depths.
        let mut available: i64 = 1;
        let mut used: i64 = 0;
        let mut depth: u64 = 0;
        let mut root_i: i64 = (size as i64) - 2;
        let mut next_i: i64 = (size as i64) - 1;

        while available > 0 {
            while root_i >= 0 && a[root_i as usize] == depth {
                used += 1;
                root_i -= 1;
            }
            while available > used {
                let symbol_index = (size as i64 - next_i - 1) as usize;
                overall_length += depth * frequencies[&symbol[symbol_index]] as u64;
                a[next_i as usize] = depth;
                next_i -= 1;
                available -= 1;
            }
            available = 2 * used;
            depth += 1;
            used = 0;
        }
    } else {
        a[0] = 1;
    }

    // Reverse depths into a per-rank length array. We allocate one
    // extra slot for the escape codeword.
    let mut length: Vec<u32> = vec![0; size + 1];
    for i in 0..size {
        length[size - 1 - i] = a[i] as u32;
    }

    // Truncate the table at the cutpoint where adding the next length
    // class would either exceed the table size cap or push us past the
    // cumulative-entropy threshold.
    let mut accumulated: u64 = 0;
    let mut current_length = length[0];
    let mut d = 1usize;
    let mut cutpoint = 0usize;
    while cutpoint < size {
        if current_length != length[cutpoint] {
            d += 1;
            if d >= max_decoding_table_len {
                break;
            }
            if overall_length != 0
                && (accumulated as f64) / (overall_length as f64) > entropy_threshold
            {
                break;
            }
            current_length = length[cutpoint];
        }
        accumulated += length[cutpoint] as u64 * frequencies[&symbol[cutpoint]] as u64;
        cutpoint += 1;
    }

    let has_escape = cutpoint < size;

    // Assign canonical codewords for the kept symbols.
    let codeword_len = if has_escape { cutpoint + 1 } else { cutpoint };
    let mut codeword: Vec<usize> = vec![0; codeword_len];
    let mut value: usize = 0;
    let mut current_length = length[0];
    codeword[0] = 0; // Length stays 0 only when cutpoint == 0.

    for i in 1..cutpoint {
        if length[i] == current_length {
            value += 1;
        } else {
            value += 1;
            value <<= length[i] - current_length;
            current_length = length[i];
        }
        // Store the codeword in append-order: bit 0 is the MSB of the
        // canonical codeword.
        codeword[i] = value.reverse_bits() >> (usize::BITS - current_length);
    }

    if has_escape {
        // Escape codeword: all-ones in `current_length` bits.
        if current_length == 0 {
            codeword[cutpoint] = 0;
        } else {
            codeword[cutpoint] = usize::MAX >> (usize::BITS - current_length);
        }
        length[cutpoint] = current_length;
    }

    // Maximum literal width across the escaped symbols.
    let mut max_length_escaped: u32 = 0;
    for &s in &symbol[cutpoint..] {
        let bits = if s == W::default() {
            0
        } else {
            W::BITS - s.leading_zeros()
        };
        max_length_escaped = max_length_escaped.max(bits);
    }

    // Symbol → rank for the kept symbols.
    let mut symbol_to_rank: HashMap<W, usize> = HashMap::with_capacity(cutpoint);
    for (i, &s) in symbol.iter().take(cutpoint).enumerate() {
        symbol_to_rank.insert(s, i);
    }

    let mut symbol_kept: Vec<W> = Vec::with_capacity(codeword_len);
    symbol_kept.extend_from_slice(&symbol[..cutpoint]);
    if has_escape {
        symbol_kept.push(W::default());
    }

    let escape_codeword_len = if has_escape { length[cutpoint] } else { 0 };

    HuffmanCoder {
        codeword: codeword.into_boxed_slice(),
        codeword_len: length[..codeword_len].to_vec().into_boxed_slice(),
        symbol: symbol_kept.into_boxed_slice(),
        symbol_to_rank,
        escaped_symbols_len: max_length_escaped,
        escape_codeword_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_escape_codeword_count() {
        let f: HashMap<u64, usize> = [(10, 5), (20, 3), (30, 1)].into_iter().collect();
        let coder: HuffmanCoder<u64> = Huffman::new().build_coder(&f);
        assert_eq!(coder.codeword.len(), 3);
        assert_eq!(coder.codeword_len.len(), 3);
        assert_eq!(coder.symbol.len(), 3);
        assert_eq!(coder.escape_codeword_len, 0);
        assert_eq!(coder.escaped_symbols_len, 0);
    }
}
