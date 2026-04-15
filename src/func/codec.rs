/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Prefix-free codes for [`CompVFunc`].
//!
//! Symbols are parametric over a symbol type `W` (any
//! [`PrimitiveUnsigned`] type: `u8`, `u16`, `u32`, `u64`, `u128`,
//! `usize`). A [`Coder<W>`] turns a symbol into a codeword (or flags
//! it as escaped); a [`Decoder<W>`] turns the high bits of a
//! `max_codeword_length`-bit window back into a symbol (or returns
//! the escape sentinel `W::MAX`).
//!
//! Codewords themselves are always `u64`-packed: the asserted
//! max-codeword-length bound is 62 bits, so they comfortably fit.
//!
//! The decoder is the canonical-Huffman fast decoder of `csf3.c`:
//! parallel arrays `last_codeword_plus_one`, `how_many_up_to_block`,
//! `shift`, `symbol`, indexed by codeword length block.
//!
//! [`CompVFunc`]: crate::func::CompVFunc
//! [`PrimitiveUnsigned`]: num_primitive::PrimitiveUnsigned

use mem_dbg::{MemDbg, MemSize};
use num_primitive::PrimitiveUnsigned;
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
    /// The returned codeword is `codeword_length(symbol)` bits wide and
    /// is laid out so that bit *l* of the returned `u64` is the *l*-th
    /// bit appended to the data array. With the layout used by
    /// [`CompVFunc`], that means bit 0 is the *most significant* bit of
    /// the canonical (MSB-first) codeword.
    ///
    /// [`CompVFunc`]: crate::func::CompVFunc
    fn encode(&self, symbol: W) -> Option<u64>;

    /// Returns the length in bits of the codeword for `symbol`.
    ///
    /// For escaped symbols this is `escape_length() +
    /// escaped_symbol_length()`.
    fn codeword_length(&self, symbol: W) -> u32;

    /// The maximum codeword length, including escaped symbols.
    fn max_codeword_length(&self) -> u32;

    /// The escape codeword (length is `escape_length()`).
    fn escape(&self) -> u64;

    /// The length in bits of the escape codeword.
    fn escape_length(&self) -> u32;

    /// The length in bits of an escaped symbol, or zero if the code has
    /// no escape.
    fn escaped_symbol_length(&self) -> u32;

    /// Builds the matching decoder.
    fn into_decoder(self) -> Self::Decoder;
}

/// A prefix-free decoder for `W` symbols.
pub trait Decoder<W> {
    /// Decodes the codeword found in the high bits of a
    /// `max_codeword_length`-bit window.
    ///
    /// The first bit of the codeword is at bit position
    /// `max_codeword_length - 1` of `value`; lower bits hold whatever
    /// follows in the data and are masked off by the canonical-Huffman
    /// algorithm.
    ///
    /// Returns the decoded symbol, or `W::MAX` if the leading codeword
    /// is the escape codeword (the caller must then read
    /// [`escaped_symbol_length`](Self::escaped_symbol_length) further
    /// bits as the literal value). This reserves `W::MAX` as a
    /// non-storable sentinel.
    fn decode(&self, value: u64) -> W;

    fn escape_length(&self) -> u32;
    fn escaped_symbol_length(&self) -> u32;
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

impl<W: PrimitiveUnsigned + Hash> Codec<W> for ZeroCodec {
    type Coder = ZeroCoder;
    fn build_coder(&self, _frequencies: &HashMap<W, usize>) -> ZeroCoder {
        ZeroCoder
    }
}

impl<W: PrimitiveUnsigned> Coder<W> for ZeroCoder {
    type Decoder = ZeroDecoder;
    fn encode(&self, _symbol: W) -> Option<u64> {
        Some(0)
    }
    fn codeword_length(&self, _symbol: W) -> u32 {
        0
    }
    fn max_codeword_length(&self) -> u32 {
        0
    }
    fn escape(&self) -> u64 {
        0
    }
    fn escape_length(&self) -> u32 {
        0
    }
    fn escaped_symbol_length(&self) -> u32 {
        0
    }
    fn into_decoder(self) -> ZeroDecoder {
        ZeroDecoder
    }
}

impl<W: PrimitiveUnsigned> Decoder<W> for ZeroDecoder {
    fn decode(&self, _value: u64) -> W {
        W::from(0u8)
    }
    fn escape_length(&self) -> u32 {
        0
    }
    fn escaped_symbol_length(&self) -> u32 {
        0
    }
}

// ── Huffman ─────────────────────────────────────────────────────────

/// A length-limited canonical Huffman codec.
///
/// Lengths are computed with the in-place Moffat–Katajainen algorithm,
/// then optionally truncated according to
/// [`max_decoding_table_length`](Self::max_decoding_table_length) and
/// [`entropy_threshold`](Self::entropy_threshold). Symbols beyond the
/// cutoff are *escaped*: they share a single dedicated escape codeword
/// followed by a literal `escaped_symbol_length`-bit field.
#[derive(Debug, Clone, Copy)]
pub struct Huffman {
    /// Hard cap on the number of distinct codeword lengths kept in
    /// the decoding table. The default is `usize::MAX` (no cap).
    pub max_decoding_table_length: usize,
    /// Cumulative-entropy fraction threshold: the table is cut once
    /// the cumulative bit length of the codewords kept exceeds this
    /// fraction of the overall bit length. The default is `1.0`
    /// (no cut).
    pub entropy_threshold: f64,
}

impl Default for Huffman {
    fn default() -> Self {
        Self {
            max_decoding_table_length: usize::MAX,
            entropy_threshold: 1.0,
        }
    }
}

impl Huffman {
    /// New unlimited Huffman codec (table grows as needed, no escapes).
    pub const fn new() -> Self {
        Self {
            max_decoding_table_length: usize::MAX,
            entropy_threshold: 1.0,
        }
    }

    /// Length-limited Huffman codec.
    ///
    /// `max_decoding_table_length` caps the number of distinct codeword
    /// lengths in the canonical decoding table; `entropy_threshold` is
    /// the cumulative-entropy fraction beyond which infrequent symbols
    /// are diverted to the escape codeword.
    pub const fn length_limited(max_decoding_table_length: usize, entropy_threshold: f64) -> Self {
        Self {
            max_decoding_table_length,
            entropy_threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HuffmanCoder<W> {
    /// Codeword for each rank position 0 . . cutpoint − 1, plus the
    /// escape codeword at index `cutpoint`. Bit *l* of an entry is the
    /// *l*-th bit appended to the data — i.e. bit 0 is the MSB of the
    /// canonical (MSB-first) codeword.
    codeword: Box<[u64]>,
    /// Length of each entry in [`Self::codeword`]. The last entry is
    /// the escape length.
    codeword_length: Box<[u32]>,
    /// The symbols, in order of decreasing frequency, kept up to the
    /// cutpoint. The last entry is the escape sentinel (`W::MAX`).
    symbol: Box<[W]>,
    /// Inverse map: symbol → rank position in [`Self::symbol`]. Symbols
    /// not in the map are escaped.
    symbol_to_rank: HashMap<W, usize>,
    escaped_symbol_length: u32,
    escape_length: u32,
}

impl<W: PrimitiveUnsigned + Hash> Codec<W> for Huffman {
    type Coder = HuffmanCoder<W>;
    fn build_coder(&self, frequencies: &HashMap<W, usize>) -> HuffmanCoder<W> {
        build_huffman_coder(
            frequencies,
            self.max_decoding_table_length,
            self.entropy_threshold,
        )
    }
}

impl<W: PrimitiveUnsigned + Hash> Coder<W> for HuffmanCoder<W> {
    type Decoder = HuffmanDecoder<W>;

    fn encode(&self, symbol: W) -> Option<u64> {
        self.symbol_to_rank.get(&symbol).map(|&r| self.codeword[r])
    }

    fn codeword_length(&self, symbol: W) -> u32 {
        match self.symbol_to_rank.get(&symbol) {
            Some(&r) => self.codeword_length[r],
            None => self.escape_length + self.escaped_symbol_length,
        }
    }

    fn max_codeword_length(&self) -> u32 {
        if self.codeword_length.is_empty() {
            0
        } else {
            self.codeword_length[self.codeword_length.len() - 1] + self.escaped_symbol_length
        }
    }

    fn escape(&self) -> u64 {
        *self.codeword.last().unwrap_or(&0)
    }

    fn escape_length(&self) -> u32 {
        self.escape_length
    }

    fn escaped_symbol_length(&self) -> u32 {
        self.escaped_symbol_length
    }

    fn into_decoder(mut self) -> HuffmanDecoder<W> {
        // The codeword array (and length array) always contains a
        // trailing escape entry, even when no symbol is escaped. We
        // build the decoder from `self.codeword_length` and the
        // canonical sequence; we then place the [`ESCAPE`] sentinel at
        // the end of `self.symbol` (mirroring the Java side effect).
        let size = self.codeword.len();
        let w = self.max_codeword_length();
        assert!(w <= 62, "Codeword length must not exceed 62");

        // Number of distinct length blocks, plus one extra trailing
        // block that catches the escape codeword if it shares a length
        // with real codewords.
        let mut decoding_table_length = if size < 2 { 0 } else { 1 };
        if size > 1 {
            for i in (0..size - 1).rev() {
                debug_assert!(
                    self.codeword_length[i] <= self.codeword_length[i + 1],
                    "lengths must be non-decreasing"
                );
                if self.codeword_length[i] != self.codeword_length[i + 1] {
                    decoding_table_length += 1;
                }
            }
        }
        decoding_table_length += 1; // For the escape codeword

        let mut shift: Vec<u8> = vec![0; decoding_table_length];
        let mut how_many_up_to_block: Vec<u32> = vec![0; decoding_table_length];
        let mut last_codeword_plus_one: Vec<u64> = vec![0; decoding_table_length];

        // p is the current block index; word tracks the canonical
        // codeword counter (MSB-first as an integer); prev_l is the
        // length of the previous block.
        let mut p: i32 = -1;
        let mut prev_l: u32 = 0;
        let mut last_l: u32 = 0;
        let mut word: u64 = 0;

        for i in 0..size {
            let l = self.codeword_length[i];
            last_l = l;
            if l != prev_l || i == size - 1 {
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

        // Final escape catch-all block. Even when the escape shares a
        // length with the last real block, this entry catches the
        // canonical "11…1" pattern that the previous block leaves
        // unmatched (because the previous block's
        // last_codeword_plus_one is the escape's own canonical value).
        let last_p = p as usize;
        last_codeword_plus_one[last_p] = u64::MAX >> 1;
        shift[last_p] = 63;
        if !self.symbol.is_empty() {
            let last_symbol = self.symbol.len() - 1;
            self.symbol[last_symbol] = W::MAX;
        }
        how_many_up_to_block[last_p] = (size as u32).saturating_sub(1);

        // Default heuristic for branchy vs branchless: `> 3` distinct
        // length blocks ⇒ branchless, otherwise branchy.
        //
        // The threshold is empirical (Apple-Silicon arm64): with one
        // or two length classes the branchy decoder is the obvious
        // winner (perfect branch prediction); with three classes it
        // is still a wash; with four or more, we start losing time
        // to mispredictions on any non-trivial frequency skew.
        // Callers that know the codeword distribution shape can
        // override the choice with [`HuffmanDecoder::set_branchless`].
        let branchless = decoding_table_length > 3;

        HuffmanDecoder {
            last_codeword_plus_one: last_codeword_plus_one.into_boxed_slice(),
            how_many_up_to_block: how_many_up_to_block.into_boxed_slice(),
            shift: shift.into_boxed_slice(),
            symbol: self.symbol,
            escape_length: last_l,
            escaped_symbol_length: self.escaped_symbol_length,
            branchless,
        }
    }
}

/// Canonical-Huffman decoder built from a [`HuffmanCoder`].
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
/// trivially predictable and the overhead is below the noise floor.
///
/// The default is chosen at construction time based on the number of
/// length blocks (`> 3` ⇒ branchless). Use [`Self::set_branchless`]
/// to override.
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HuffmanDecoder<W> {
    last_codeword_plus_one: Box<[u64]>,
    how_many_up_to_block: Box<[u32]>,
    shift: Box<[u8]>,
    symbol: Box<[W]>,
    escape_length: u32,
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

impl<W: PrimitiveUnsigned> Decoder<W> for HuffmanDecoder<W> {
    #[inline(always)]
    fn decode(&self, value: u64) -> W {
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

    fn escape_length(&self) -> u32 {
        self.escape_length
    }

    fn escaped_symbol_length(&self) -> u32 {
        self.escaped_symbol_length
    }
}

impl<W: PrimitiveUnsigned> HuffmanDecoder<W> {
    /// Branchy decode: walks the length blocks until one matches.
    ///
    /// Wins for very skewed codeword distributions (the first one or
    /// two blocks catch almost every query).
    #[inline(always)]
    fn decode_branchy(&self, value: u64) -> W {
        for curr in 0..self.last_codeword_plus_one.len() {
            // SAFETY: `curr` is bounded by the loop range.
            unsafe {
                let lcp1 = *self.last_codeword_plus_one.get_unchecked(curr);
                if value < lcp1 {
                    let s = *self.shift.get_unchecked(curr) as u32;
                    let off = (value >> s).wrapping_sub(lcp1 >> s);
                    let idx = off
                        .wrapping_add(*self.how_many_up_to_block.get_unchecked(curr) as u64)
                        as usize;
                    return *self.symbol.get_unchecked(idx);
                }
            }
        }
        // The final block has `last_codeword_plus_one = u64::MAX >> 1`
        // and `w <= 62`, so `value` is always strictly less than the
        // final upper bound — we always return inside the loop.
        unreachable!("decoder fell through all blocks")
    }

    /// Branchless decode: counts blocks whose upper bound is `<=
    /// value` to derive the matching block index. No data-dependent
    /// branches inside the loop, at the cost of always touching every
    /// block.
    #[inline(always)]
    fn decode_branchless(&self, value: u64) -> W {
        let n = self.last_codeword_plus_one.len();
        let mut idx: usize = 0;
        for curr in 0..n {
            // SAFETY: `curr` is bounded by the loop range.
            let lcp1 = unsafe { *self.last_codeword_plus_one.get_unchecked(curr) };
            idx += (lcp1 <= value) as usize;
        }
        // SAFETY: see `decode_branchy` for why `idx < n`.
        unsafe {
            let lcp1 = *self.last_codeword_plus_one.get_unchecked(idx);
            let s = *self.shift.get_unchecked(idx) as u32;
            let off = (value >> s).wrapping_sub(lcp1 >> s);
            let sym_idx =
                off.wrapping_add(*self.how_many_up_to_block.get_unchecked(idx) as u64) as usize;
            *self.symbol.get_unchecked(sym_idx)
        }
    }
}

// ── Huffman code construction ──────────────────────────────────────

fn build_huffman_coder<W: PrimitiveUnsigned + Hash>(
    frequencies: &HashMap<W, usize>,
    max_decoding_table_length: usize,
    entropy_threshold: f64,
) -> HuffmanCoder<W> {
    let size = frequencies.len();
    if size == 0 {
        // No symbols: codeword[0] is the (unused) escape, length 0.
        return HuffmanCoder {
            codeword: Box::new([0]),
            codeword_length: Box::new([0]),
            symbol: Box::new([]),
            symbol_to_rank: HashMap::new(),
            escaped_symbol_length: 0,
            escape_length: 0,
        };
    }

    // Sort symbols by frequency descending (most frequent first).
    let mut symbol: Vec<W> = frequencies.keys().copied().collect();
    symbol.sort_unstable_by(|a, b| frequencies[b].cmp(&frequencies[a]).then(a.cmp(b)));

    // Moffat–Katajainen builds depths in place on a frequency array
    // sorted *ascending*. We mirror the Java buffer reuse so the
    // numerical behaviour matches. The array is reused to store
    // parent-pointers, so it needs to be `u64`-wide regardless of W.
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
        a[size - 2] = 0;
        for next in (0..=size - 3).rev() {
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
            if d >= max_decoding_table_length {
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

    // Assign canonical codewords for the kept symbols.
    let mut codeword: Vec<u64> = vec![0; cutpoint + 1];
    let mut value: u64 = 0;
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
        codeword[i] = value.reverse_bits() >> (64 - current_length);
    }

    // Escape codeword: all-ones in `current_length` bits, length =
    // current_length. This may collide canonically with the last real
    // codeword when `cutpoint == size`, but in that case nothing is
    // ever encoded as escape, so the collision is harmless.
    if current_length == 0 {
        codeword[cutpoint] = 0;
    } else {
        codeword[cutpoint] = u64::MAX >> (64 - current_length);
    }
    length[cutpoint] = current_length;

    // Maximum literal width across the escaped symbols.
    let mut max_length_escaped: u32 = 0;
    for &s in &symbol[cutpoint..] {
        let bits = if s == W::from(0u8) {
            0
        } else {
            W::BITS - s.leading_zeros()
        };
        max_length_escaped = max_length_escaped.max(bits);
    }

    // Symbol → rank for the kept symbols, with one trailing slot for
    // the escape sentinel that the decoder will write.
    let mut symbol_to_rank: HashMap<W, usize> = HashMap::with_capacity(cutpoint);
    for (i, &s) in symbol.iter().take(cutpoint).enumerate() {
        symbol_to_rank.insert(s, i);
    }

    let mut symbol_kept: Vec<W> = Vec::with_capacity(cutpoint + 1);
    symbol_kept.extend_from_slice(&symbol[..cutpoint]);
    symbol_kept.push(W::from(0u8)); // Decoder will overwrite with W::MAX sentinel.

    let escape_length = length[cutpoint];

    HuffmanCoder {
        codeword: codeword.into_boxed_slice(),
        codeword_length: length[..cutpoint + 1].to_vec().into_boxed_slice(),
        symbol: symbol_kept.into_boxed_slice(),
        symbol_to_rank,
        escaped_symbol_length: max_length_escaped,
        escape_length,
    }
}
