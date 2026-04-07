/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! An implementation of Jacobson's balanced parentheses data structure.
//!
//! This is a standalone implementation supporting only `find_close`, intended
//! for use in the hollow trie. Open parentheses are represented as 1 bits and
//! close parentheses as 0 bits, with bit 0 being the LSB.
//!
//! The structure uses the *pioneer* technique: an opening parenthesis whose
//! match falls in a different 64-bit word is called "far". A far opening
//! parenthesis is a "pioneer" if it is the first far opening parenthesis in its
//! word, or if its match falls in a different word than the previous far
//! opening parenthesis's match. Pioneer positions are stored in an Elias–Fano
//! structure supporting predecessor queries, and match offsets in a compact
//! [`BitFieldVec`].

use crate::bits::BitFieldVec;
use crate::dict::{EfDict, EliasFanoBuilder};
use crate::traits::indexed_dict::Pred;
use mem_dbg::*;

/// For each byte value b (interpreted as 8 balanced-parenthesis bits, LSB
/// first, open=1 close=0), stores `(min_excess, match_position)` where:
///  - `min_excess` (i8): the minimum running excess encountered scanning
///    the 8 bits, assuming initial excess 0. This is stored negated for the
///    broadword ≤₈ comparison.
///  - `match_position` (u8): for each possible target excess (-1..=-8),
///    the bit position within the byte where that excess is first reached.
///    Encoded as: for target excess `e` (starting at -1), the position is
///    `BYTE_MATCH[byte][(-e - 1) as usize]`. Value 8 means not reached.
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

/// For each byte value b and target excess (0..8, meaning the first position
/// where running excess reaches -(target+1)), the bit position (0..7) or 8 if
/// not reached.
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

const ONES_STEP_4: u64 = 0x1111_1111_1111_1111;
const MSBS_STEP_4: u64 = 0x8 * ONES_STEP_4;
const ONES_STEP_8: u64 = 0x0101_0101_0101_0101;
const MSBS_STEP_8: u64 = 0x80 * ONES_STEP_8;
const ONES_STEP_16: u64 = 0x0001_0001_0001_0001;
const MSBS_STEP_16: u64 = 0x8000_8000_8000_8000;
const ONES_STEP_32: u64 = 0x0000_0001_0000_0001;
const MSBS_STEP_32: u64 = 0x8000_0000_8000_0000;

/// Count the number of far open parentheses scanning `l` bits of `word`
/// from MSB to LSB.
///
/// A "far open" parenthesis is an open parenthesis (1-bit) that, when scanning
/// from MSB to LSB, causes the excess to become positive.
fn count_far_open(word: u64, l: usize) -> usize {
    let mut c = 0usize;
    let mut e = 0i32;
    for i in (0..l).rev() {
        if word & (1u64 << i) != 0 {
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
fn count_far_close(word: u64, l: usize) -> usize {
    let mut c = 0usize;
    let mut e = 0i32;
    for i in 0..l {
        if word & (1u64 << i) != 0 {
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

/// Find the k-th (0-based) far close parenthesis in a word using a simple loop.
///
/// Used during construction to locate pioneer match positions.
fn find_far_close_naive(word: u64, k: usize) -> usize {
    let mut e = 0i32;
    let mut remaining = k;
    for i in 0..64 {
        if word & (1u64 << i) != 0 {
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
    panic!("find_far_close_naive: not enough far close parentheses (k={k})")
}

/// Find the k-th (0-based) far open parenthesis scanning `l` bits from MSB to
/// LSB.
///
/// This is currently unused but kept for potential `find_open` support.
#[allow(dead_code)]
fn find_far_open(word: u64, l: usize, k: usize) -> usize {
    let mut e = 0i32;
    let mut remaining = k;
    for i in (0..l).rev() {
        if word & (1u64 << i) != 0 {
            e += 1;
            if e > 0 {
                if remaining == 0 {
                    return i;
                }
                remaining -= 1;
            }
        } else if e > 0 {
            e = -1;
        } else {
            e -= 1;
        }
    }
    panic!("find_far_open: not enough far open parentheses (k={k})")
}

/// Find the position of the matching close parenthesis within a single word,
/// using byte-level lookup tables.
///
/// The word is assumed to be shifted so that bit 0 is the first bit *after* the
/// opening parenthesis (i.e., the initial excess is 1). Returns the bit
/// position of the matching close within the word (0..63), or a value >= 64 if
/// the match is not in this word.
///
/// This uses precomputed per-byte min-excess and match-position tables,
/// requiring only ~15 operations to identify the target byte plus a single
/// table lookup.
pub fn find_near_close(word: u64) -> usize {
    // The word is shifted so bit 0 is the open paren that we already consumed
    // (initial excess = 1). We scan from bit 1 onward looking for excess = 0.
    //
    // Shift right by 1 so that bit 0 is the first bit to scan. The initial
    // excess from the open paren at the original bit 0 is 1.
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
    127
}

/// Find the k-th (0-based) far close parenthesis in a 64-bit word using
/// broadword techniques.
///
/// A "far close" parenthesis is an unmatched close parenthesis (0-bit) when
/// scanning from LSB to MSB with open = 1, close = 0.
pub fn find_far_close(word: u64, k: i64) -> usize {
    // Decompose into 2-bit groups: extract high and low bits
    let b1 = (word & (0xA * ONES_STEP_4)) >> 1;
    let b0 = word & (0x5 * ONES_STEP_4);
    let lsb = (b1 ^ b0) & b1;

    // Open parens in 2-bit groups: 00->00, 01->01, 10->00, 11->10
    let open2 = (b1 & b0) << 1 | lsb;
    // Close parens in 2-bit groups: 00->10, 01->01, 10->00, 11->00
    let closed2 = (((b1 | b0) ^ (0x5 * ONES_STEP_4)) << 1) | lsb;

    // 4-bit level
    let open4eccess = open2 & (0x3 * ONES_STEP_4);
    let closed4eccess = (closed2 & (0xC * ONES_STEP_4)) >> 2;

    let mut open4 =
        ((open4eccess | MSBS_STEP_4).wrapping_sub(closed4eccess)) ^ MSBS_STEP_4;
    let open4mask =
        ((((open4 & MSBS_STEP_4) >> 3) | MSBS_STEP_4).wrapping_sub(ONES_STEP_4)) ^ MSBS_STEP_4;

    let mut closed4 =
        ((closed4eccess | MSBS_STEP_4).wrapping_sub(open4eccess)) ^ MSBS_STEP_4;
    let closed4mask =
        ((((closed4 & MSBS_STEP_4) >> 3) | MSBS_STEP_4).wrapping_sub(ONES_STEP_4)) ^ MSBS_STEP_4;

    open4 = ((open2 & (0xC * ONES_STEP_4)) >> 2) + (open4mask & open4);
    closed4 = (closed2 & (0x3 * ONES_STEP_4)) + (closed4mask & closed4);

    // 8-bit level
    let open8eccess = open4 & (0xF * ONES_STEP_8);
    let closed8eccess = (closed4 & (0xF0 * ONES_STEP_8)) >> 4;

    let mut open8 =
        ((open8eccess | MSBS_STEP_8).wrapping_sub(closed8eccess)) ^ MSBS_STEP_8;
    let open8mask =
        ((((open8 & MSBS_STEP_8) >> 7) | MSBS_STEP_8).wrapping_sub(ONES_STEP_8)) ^ MSBS_STEP_8;

    let mut closed8 =
        ((closed8eccess | MSBS_STEP_8).wrapping_sub(open8eccess)) ^ MSBS_STEP_8;
    let closed8mask =
        ((((closed8 & MSBS_STEP_8) >> 7) | MSBS_STEP_8).wrapping_sub(ONES_STEP_8)) ^ MSBS_STEP_8;

    open8 = ((open4 & (0xF0 * ONES_STEP_8)) >> 4) + (open8mask & open8);
    closed8 = (closed4 & (0xF * ONES_STEP_8)) + (closed8mask & closed8);

    // 16-bit level
    let open16eccess = open8 & (0xFF * ONES_STEP_16);
    let closed16eccess = (closed8 & (0xFF00 * ONES_STEP_16)) >> 8;

    let mut open16 =
        ((open16eccess | MSBS_STEP_16).wrapping_sub(closed16eccess)) ^ MSBS_STEP_16;
    let open16mask = ((((open16 & MSBS_STEP_16) >> 15) | MSBS_STEP_16).wrapping_sub(ONES_STEP_16))
        ^ MSBS_STEP_16;

    let mut closed16 =
        ((closed16eccess | MSBS_STEP_16).wrapping_sub(open16eccess)) ^ MSBS_STEP_16;
    let closed16mask =
        ((((closed16 & MSBS_STEP_16) >> 15) | MSBS_STEP_16).wrapping_sub(ONES_STEP_16))
            ^ MSBS_STEP_16;

    open16 = ((open8 & (0xFF00 * ONES_STEP_16)) >> 8) + (open16mask & open16);
    closed16 = (closed8 & (0xFF * ONES_STEP_16)) + (closed16mask & closed16);

    // 32-bit level
    let open32eccess = open16 & (0xFFFF * ONES_STEP_32);
    let closed32eccess = (closed16 & (0xFFFF0000_u64.wrapping_mul(ONES_STEP_32))) >> 16;

    let mut open32 =
        ((open32eccess | MSBS_STEP_32).wrapping_sub(closed32eccess)) ^ MSBS_STEP_32;
    let open32mask = ((((open32 & MSBS_STEP_32) >> 31) | MSBS_STEP_32).wrapping_sub(ONES_STEP_32))
        ^ MSBS_STEP_32;

    let mut closed32 =
        ((closed32eccess | MSBS_STEP_32).wrapping_sub(open32eccess)) ^ MSBS_STEP_32;
    let closed32mask =
        ((((closed32 & MSBS_STEP_32) >> 31) | MSBS_STEP_32).wrapping_sub(ONES_STEP_32))
            ^ MSBS_STEP_32;

    open32 = ((open16 & (0xFFFF0000_u64.wrapping_mul(ONES_STEP_32))) >> 16) + (open32mask & open32);
    closed32 = (closed16 & (0xFFFF * ONES_STEP_32)) + (closed32mask & closed32);

    // Selection phase: walk down from 32-bit to 2-bit granularity.
    // k is i64 to allow the signed arithmetic to work correctly.
    let mut k = k;

    // In Java: ((k - (closed32 & 0xFFFFFFFFL)) >>> Long.SIZE - 1) - 1
    // k is sign-extended to i64; we subtract, then check the sign.
    let check32 = ((k.wrapping_sub((closed32 & 0xFFFF_FFFF) as i64)) as u64 >> 63).wrapping_sub(1);
    // check32 is 0 if k < closed32_low (need to go to low half), or all-ones otherwise
    let mut mask = check32 & 0xFFFF_FFFF;
    k -= (closed32 & mask) as i64;
    k += (open32 & mask) as i64;
    let mut shift = (32 & check32) as u32;

    let check16 =
        ((k.wrapping_sub((closed16 >> shift & 0xFFFF) as i64)) as u64 >> 63).wrapping_sub(1);
    mask = check16 & 0xFFFF;
    k -= (closed16 >> shift & mask) as i64;
    k += (open16 >> shift & mask) as i64;
    shift += (16 & check16) as u32;

    let check8 =
        ((k.wrapping_sub((closed8 >> shift & 0xFF) as i64)) as u64 >> 63).wrapping_sub(1);
    mask = check8 & 0xFF;
    k -= (closed8 >> shift & mask) as i64;
    k += (open8 >> shift & mask) as i64;
    shift += (8 & check8) as u32;

    let check4 =
        ((k.wrapping_sub((closed4 >> shift & 0xF) as i64)) as u64 >> 63).wrapping_sub(1);
    mask = check4 & 0xF;
    k -= (closed4 >> shift & mask) as i64;
    k += (open4 >> shift & mask) as i64;
    shift += (4 & check4) as u32;

    let check2 =
        ((k.wrapping_sub((closed2 >> shift & 0x3) as i64)) as u64 >> 63).wrapping_sub(1);
    mask = check2 & 0x3;
    k -= (closed2 >> shift & mask) as i64;
    k += (open2 >> shift & mask) as i64;
    shift += (2 & check2) as u32;

    // Final bit-level resolution
    (shift as usize)
        + (k as usize)
        + (((word >> shift) & (((k << 1) as u64) | 1)) << 1) as usize
}

/// Jacobson's balanced parentheses structure supporting `find_close` queries.
///
/// Open parentheses are represented as 1-bits and close parentheses as 0-bits.
/// Bit 0 is the LSB.
#[derive(Debug, MemDbg, MemSize)]
pub struct JacobsonBP {
    /// The balanced parentheses bit vector.
    words: Vec<u64>,
    /// Total number of valid bits.
    len: usize,
    /// Pioneer positions stored in an Elias–Fano structure supporting
    /// predecessor queries.
    pioneer_positions: EfDict<u64>,
    /// For each pioneer, the offset from the pioneer position to its matching
    /// close position, stored in a compact bit-field vector.
    pioneer_match_offsets: BitFieldVec<Box<[u64]>>,
}

impl JacobsonBP {
    /// Construct from a word array and the number of valid bits.
    ///
    /// # Panics
    ///
    /// Panics if the parentheses are not balanced.
    pub fn new(words: Vec<u64>, len: usize) -> Self {
        let num_words = len.div_ceil(64);
        debug_assert!(words.len() >= num_words);

        let bits = &words;

        // count[block] will hold the number of far close parentheses in block
        // (populated as we scan right-to-left, so it stores the far-close count
        // of blocks to the right of the current one).
        let mut count = vec![0i32; num_words];
        let mut residual = vec![0usize; num_words];

        let mut opening_pioneers = Vec::new();
        let mut opening_pioneer_matches = Vec::new();

        // Scan words right-to-left (Java lines 529-577)
        for block in (0..num_words).rev() {
            let l = std::cmp::min(64, len - block * 64);

            if block != num_words - 1 {
                let mut excess = 0i32;
                let mut count_far_opening = count_far_open(bits[block], l) as i32;

                for j in (0..l).rev() {
                    if bits[block] & (1u64 << j) == 0 {
                        // Close paren
                        if excess > 0 {
                            excess = -1;
                        } else {
                            excess -= 1;
                        }
                    } else {
                        // Open paren
                        excess += 1;
                        if excess > 0 {
                            // This is a far opening parenthesis; find the block
                            // containing its matching far close.
                            let mut matching_block = block;
                            loop {
                                matching_block += 1;
                                if count[matching_block] != 0 {
                                    break;
                                }
                            }
                            count_far_opening -= 1;

                            if { count[matching_block] -= 1; count[matching_block] } == 0
                                || count_far_opening == 0
                            {
                                // This is an opening pioneer
                                let pos = block * 64 + j;
                                let match_pos = matching_block * 64
                                    + find_far_close_naive(
                                        bits[matching_block],
                                        residual[matching_block],
                                    );
                                opening_pioneers.push(pos as u64);
                                opening_pioneer_matches.push((match_pos - pos) as u64);
                            }
                            residual[matching_block] += 1;
                        }
                    }
                }
            }
            count[block] = count_far_close(bits[block], l) as i32;
        }

        // Verify balanced
        for &c in &count {
            assert!(c == 0, "Unbalanced parentheses");
        }

        // The pioneers were collected in reverse order (scanning right-to-left),
        // so reverse them to get increasing order.
        opening_pioneers.reverse();
        opening_pioneer_matches.reverse();

        // Build Elias-Fano for pioneer positions
        let num_pioneers = opening_pioneers.len();
        let ef_positions = if num_pioneers > 0 {
            let max_pos = *opening_pioneers.last().unwrap();
            let mut builder = EliasFanoBuilder::new(num_pioneers, max_pos);
            for &p in &opening_pioneers {
                builder.push(p);
            }
            builder.build_with_dict()
        } else {
            EliasFanoBuilder::new(0, 0u64).build_with_dict()
        };

        // Build BitFieldVec for match offsets
        let max_offset = opening_pioneer_matches.iter().copied().max().unwrap_or(0);
        let bit_width = if max_offset == 0 { 1 } else { 64 - max_offset.leading_zeros() as usize };
        let mut offsets = BitFieldVec::new(bit_width, num_pioneers);
        for (i, &off) in opening_pioneer_matches.iter().enumerate() {
            use value_traits::slices::SliceByValueMut;
            unsafe { offsets.set_value_unchecked(i, off) };
        }

        JacobsonBP {
            words,
            len,
            pioneer_positions: ef_positions,
            pioneer_match_offsets: offsets.into(),
        }
    }

    /// Find the matching close parenthesis for the open parenthesis at position
    /// `pos`.
    ///
    /// Returns `None` if `pos` is out of bounds or is not an open parenthesis.
    pub fn find_close(&self, pos: usize) -> Option<usize> {
        if pos >= self.len {
            return None;
        }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;

        // Check that it's an open paren
        if self.words[word_idx] & (1u64 << bit_idx) == 0 {
            return None;
        }

        // Try to find the close within the same word
        let result = find_near_close(self.words[word_idx] >> bit_idx);
        if result < 64 - bit_idx {
            return Some(word_idx * 64 + bit_idx + result);
        }

        // Far match: look up the pioneer using Elias-Fano predecessor query
        let (pioneer_index, pioneer_val) = self
            .pioneer_positions
            .pred(pos as u64)
            .expect("No pioneer found for far open paren");

        let pioneer = pioneer_val as usize;
        let match_pos = pioneer + unsafe {
            use value_traits::slices::SliceByValue;
            self.pioneer_match_offsets.get_value_unchecked(pioneer_index)
        } as usize;

        if pos == pioneer {
            return Some(match_pos);
        }

        debug_assert_eq!(word_idx, pioneer / 64);

        let pioneer_bit = pioneer % 64;
        let dist = pos - pioneer;

        // Compute excess difference between pioneer and pos
        // This is: 2 * popcount(bits from pioneer to pos-1) - dist
        // where we count only open parens (1-bits)
        let mask_bits = if dist < 64 {
            (self.words[word_idx] >> pioneer_bit) & ((1u64 << dist) - 1)
        } else {
            self.words[word_idx] >> pioneer_bit
        };
        let e = 2 * mask_bits.count_ones() as i64 - dist as i64;

        let match_word = match_pos / 64;
        let match_bit = match_pos % 64;

        // Number of far close parens before match_bit in match_word:
        // match_bit - 2 * popcount(match_word_bits & ((1 << match_bit) - 1))
        let match_word_bits = self.words[match_word];
        let mask_before_match = if match_bit == 0 {
            0
        } else {
            (1u64 << match_bit) - 1
        };
        let num_far_close =
            match_bit as i64 - ((match_word_bits & mask_before_match).count_ones() as i64) * 2;

        Some(match_word * 64 + find_far_close(match_word_bits, num_far_close - e))
    }

    /// Get the underlying words.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Get the total bit length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bit vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive find_near_close for testing: scans bit by bit starting after the
    /// open paren (at bit 0 of the shifted word), returns position or 64+ if
    /// not found.
    fn find_near_close_naive(word: u64) -> usize {
        let mut c = 1i32;
        for i in 1..64 {
            if word & (1u64 << i) != 0 {
                c += 1;
            } else {
                c -= 1;
            }
            if c == 0 {
                return i;
            }
        }
        64
    }

    /// Naive find_far_close for testing: returns the position of the k-th
    /// (0-based) unmatched close.
    fn find_far_close_reference(word: u64, k: i64) -> usize {
        let mut e = 0i32;
        let mut remaining = k;
        for i in 0..64 {
            if word & (1u64 << i) != 0 {
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
    fn find_close_naive(words: &[u64], pos: usize, len: usize) -> usize {
        assert!(pos < len);
        assert!(words[pos / 64] & (1u64 << (pos % 64)) != 0, "Not an open paren");
        let mut c = 1i32;
        for i in (pos + 1)..len {
            let w = words[i / 64];
            if w & (1u64 << (i % 64)) != 0 {
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
        // "()" = bits: 1 at pos 0 (open), 0 at pos 1 (close) => word = 0b01 = 1
        // After shifting past the open paren: word >> 0 gives the same word,
        // but find_near_close receives the word shifted by 1 already:
        // find_near_close(1 >> 0) should look at the word starting after bit 0.
        // Actually, find_near_close is called with word >> bit, and bit = 0,
        // so word = 0b01. In find_near_close, bit 0 is already the bit after
        // the open. Actually, let me reconsider: the caller does
        // find_near_close(words[word] >> bit). For pos=0, bit=0, so it's
        // find_near_close(word). The "initial excess" for find_near_close is 1,
        // meaning the function tracks excess starting from 1 (the opening
        // paren just before bit 0 of the input).
        //
        // For "()" the word is 0b01 and find_near_close(0b01) should return 1
        // (bit 1 is position of the matching close after shifting).
        // But wait: after >> 0, bit 0 of the result represents position 0 which
        // is the open paren itself. The find_near_close function should look
        // starting from bit 1...
        //
        // Looking at the Java code more carefully:
        // findClose shifts by `bit` not `bit+1`. So find_near_close receives
        // a word where bit 0 is the open paren itself. But find_near_close
        // internally starts with excess 1 (meaning: we already consumed the
        // open paren). The naive Java function findNearClose2 starts from i=1.
        // So it skips bit 0 (the open paren) and checks bits 1..63.

        // Simple test: "()" = word has bit 0 set, bit 1 clear
        // find_near_close(0b01) should return 1
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
    fn test_find_near_close_exhaustive_byte() {
        // Test all possible 8-bit words where bit 0 is set (valid input:
        // bit 0 is the open paren we are matching).
        for w in (1u64..256).step_by(2) {
            let naive = find_near_close_naive(w);
            let fast = find_near_close(w);
            assert_eq!(
                fast, naive,
                "find_near_close mismatch for word 0b{:08b}: fast={}, naive={}",
                w, fast, naive
            );
        }
    }

    #[test]
    fn test_find_near_close_exhaustive_16bit() {
        // Test all 16-bit odd words (bit 0 set)
        for w in (1u64..65536).step_by(2) {
            let naive = find_near_close_naive(w);
            let fast = find_near_close(w);
            if naive < 64 {
                assert_eq!(
                    fast, naive,
                    "find_near_close mismatch for word 0x{:04x}: fast={}, naive={}",
                    w, fast, naive
                );
            } else {
                assert!(fast >= 64);
            }
        }
    }

    #[test]
    fn test_find_near_close_against_naive() {
        // Test a variety of words. Bit 0 must be set (it's the open paren).
        let test_words: Vec<u64> = vec![
            0b01,                           // ()
            0b0011,                         // (())
            0b001011,                       // ()(())
            0b0101,                         // ()()
            0b00001111,                     // (((())))
            0b00110011,                     // (())(())
            0x5555_5555_5555_5555,          // ()()()...
            0x0000_0000_FFFF_FFFF,          // 32 opens then 32 closes
            1,                              // single open, close at bit 1
            0xFFFF_FFFF_FFFF_FFFF,          // all opens - no close in word
        ];
        for &w in &test_words {
            let naive = find_near_close_naive(w);
            let fast = find_near_close(w);
            if naive < 64 {
                assert_eq!(
                    fast, naive,
                    "find_near_close mismatch for word 0x{:016x}: fast={}, naive={}",
                    w, fast, naive
                );
            } else {
                // Both should indicate "not found in this word" (>= 64)
                assert!(
                    fast >= 64,
                    "find_near_close false positive for word 0x{:016x}: fast={}, naive={}",
                    w, fast, naive
                );
            }
        }
    }

    #[test]
    fn test_find_near_close_vs_naive_16bit() {
        for w in (0u64..=0xFFFF).filter(|w| w & 1 == 1) {
            let fast = find_near_close(w);
            let naive = find_near_close_naive(w);
            assert_eq!(
                fast.min(64),
                naive.min(64),
                "mismatch for word {w:#018b}: fast={fast}, naive={naive}"
            );
        }
    }

    /// Helper: count total far closes in a word
    fn count_far_closes(word: u64) -> usize {
        let mut e = 0i32;
        let mut c = 0usize;
        for i in 0..64 {
            if word & (1u64 << i) != 0 { if e > 0 { e = -1; } else { e -= 1; } }
            else { e += 1; if e > 0 { c += 1; } }
        }
        c
    }

    #[test]
    fn test_find_far_close_simple() {
        // Word: all zeros (all close parens). Far close 0 is at position 0.
        assert_eq!(find_far_close(0, 0), 0);
        assert_eq!(find_far_close(0, 1), 1);
        assert_eq!(find_far_close(0, 2), 2);
    }

    #[test]
    fn test_find_far_close_against_naive() {
        let test_words: Vec<u64> = vec![
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
            let n_far = count_far_close(w, 64);
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
        for w in 0u64..256 {
            let n_far = count_far_close(w, 64);
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

    #[test]
    fn test_find_close_simple() {
        // "()" = bit 0 open, bit 1 close: word = 0b01
        let bp = JacobsonBP::new(vec![0b01], 2);
        assert_eq!(bp.find_close(0), Some(1));
    }

    #[test]
    fn test_find_close_nested() {
        // "(())" = bits 0,1 open, bits 2,3 close: word = 0b0011
        let bp = JacobsonBP::new(vec![0b0011], 4);
        assert_eq!(bp.find_close(0), Some(3));
        assert_eq!(bp.find_close(1), Some(2));
    }

    #[test]
    fn test_find_close_sequential() {
        // "()()" = bits: open, close, open, close = 1,0,1,0 = 0b0101
        let bp = JacobsonBP::new(vec![0b0101], 4);
        assert_eq!(bp.find_close(0), Some(1));
        assert_eq!(bp.find_close(2), Some(3));
    }

    #[test]
    fn test_find_close_not_open() {
        let bp = JacobsonBP::new(vec![0b0011], 4);
        // Position 2 is a close paren
        assert_eq!(bp.find_close(2), None);
        // Out of bounds
        assert_eq!(bp.find_close(4), None);
    }

    #[test]
    fn test_find_close_across_words() {
        // Create a sequence where open parens in the first word have their
        // matching close in the second word.
        // 64 opens (all 1s) followed by 64 closes (all 0s)
        let bp = JacobsonBP::new(vec![0xFFFF_FFFF_FFFF_FFFF, 0], 128);
        // The outermost open (bit 0) should close at bit 127
        assert_eq!(bp.find_close(0), Some(127));
        // The innermost open (bit 63) should close at bit 64
        assert_eq!(bp.find_close(63), Some(64));
    }

    #[test]
    fn test_find_close_medium() {
        // "((())) (())" across a word boundary.
        // Build a balanced sequence of known structure.
        // Let's build: 32 pairs of "()" = ()()()...() repeated 32 times = 64 bits
        // Then another 32 pairs in the second word.
        // Word 0: 0101...01 (32 times) = 0x5555_5555_5555_5555
        // Word 1: 0101...01 (32 times) = 0x5555_5555_5555_5555
        let bp = JacobsonBP::new(
            vec![0x5555_5555_5555_5555, 0x5555_5555_5555_5555],
            128,
        );
        for i in 0..64 {
            let pos = i * 2;
            assert_eq!(bp.find_close(pos), Some(pos + 1), "Failed for pos={pos}");
        }
    }

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
        let words = vec![0x5555_5555_5555_555F, 0x0000_0000_0000_0000];
        let len = 68;
        let bp = JacobsonBP::new(words.clone(), len);

        // Check all opens against naive
        for pos in 0..len {
            if words[pos / 64] & (1u64 << (pos % 64)) != 0 {
                let expected = find_close_naive(&words, pos, len);
                let actual = bp.find_close(pos).unwrap();
                assert_eq!(
                    actual, expected,
                    "find_close mismatch at pos={pos}: got {actual}, expected {expected}"
                );
            }
        }
    }

    /// Generate a random balanced parentheses sequence of given length using a
    /// simple stack-based approach.
    fn random_balanced(n: usize) -> (Vec<u64>, usize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        assert!(n % 2 == 0, "Need even length for balanced parens");

        // Simple deterministic PRNG seeded by n
        let mut state = {
            let mut h = DefaultHasher::new();
            n.hash(&mut h);
            h.finish()
        };
        let mut next_rand = move || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let half = n / 2;
        let mut bits = vec![0u64; (n + 63) / 64];
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
            } else if opens_remaining == n - i - excess {
                // Must place open (otherwise we can't close all remaining)
                // Actually: remaining positions = n - i, must place opens_remaining opens
                // and (n - i - opens_remaining) closes. Need excess + opens_remaining - closes_remaining >= 0
                // at all times. If we place a close here: excess - 1 >= 0 (already checked).
                // Use random choice weighted by remaining.
                next_rand() % ((opens_remaining + excess) as u64) < opens_remaining as u64
            } else {
                next_rand() % ((opens_remaining + excess) as u64) < opens_remaining as u64
            };

            if place_open {
                bits[i / 64] |= 1u64 << (i % 64);
                opens_remaining -= 1;
                excess += 1;
            } else {
                excess -= 1;
            }
        }
        assert_eq!(excess, 0);
        assert_eq!(opens_remaining, 0);
        (bits, n)
    }

    #[test]
    fn test_find_close_random_small() {
        for size in (2..=40).step_by(2) {
            let (words, len) = random_balanced(size);
            let bp = JacobsonBP::new(words.clone(), len);
            for pos in 0..len {
                if words[pos / 64] & (1u64 << (pos % 64)) != 0 {
                    let expected = find_close_naive(&words, pos, len);
                    let actual = bp.find_close(pos).unwrap();
                    assert_eq!(
                        actual, expected,
                        "size={size}, pos={pos}: got {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_find_close_random_large() {
        for &size in &[128, 256, 512, 1024, 2048] {
            let (words, len) = random_balanced(size);
            let bp = JacobsonBP::new(words.clone(), len);
            for pos in 0..len {
                if words[pos / 64] & (1u64 << (pos % 64)) != 0 {
                    let expected = find_close_naive(&words, pos, len);
                    let actual = bp.find_close(pos).unwrap();
                    assert_eq!(
                        actual, expected,
                        "size={size}, pos={pos}: got {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_find_close_all_nested() {
        // ((((...))))  - 32 opens then 32 closes, fits in one word
        // word = 0x0000_0000_FFFF_FFFF
        let word = 0x0000_0000_FFFF_FFFFu64;
        let bp = JacobsonBP::new(vec![word], 64);
        for i in 0..32 {
            assert_eq!(bp.find_close(i), Some(63 - i), "pos={i}");
        }
    }

    #[test]
    fn test_find_close_3_words() {
        // 3 words: 192 bits. All opens in word 0, mixed in word 1, all closes
        // in word 2.
        // Word 0: 64 opens
        // Word 1: 32 closes then 32 opens (to close inner and reopen)
        // Word 2: 64 closes
        // Actually this won't be balanced easily. Let me use random_balanced.
        let (words, len) = random_balanced(192);
        let bp = JacobsonBP::new(words.clone(), len);
        for pos in 0..len {
            if words[pos / 64] & (1u64 << (pos % 64)) != 0 {
                let expected = find_close_naive(&words, pos, len);
                let actual = bp.find_close(pos).unwrap();
                assert_eq!(
                    actual, expected,
                    "3-word test, pos={pos}: got {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_count_far_open_close() {
        // All opens: count_far_open should count all of them
        assert_eq!(count_far_open(0xFFFF_FFFF_FFFF_FFFF, 64), 64);
        // All closes: count_far_close should count all of them
        assert_eq!(count_far_close(0, 64), 64);
        // "()" pairs: no far parens
        assert_eq!(count_far_open(0x5555_5555_5555_5555, 64), 0);
        assert_eq!(count_far_close(0x5555_5555_5555_5555, 64), 0);
    }

    #[test]
    fn test_empty() {
        let bp = JacobsonBP::new(vec![], 0);
        assert!(bp.is_empty());
        assert_eq!(bp.find_close(0), None);
    }

    #[test]
    fn test_single_pair() {
        // Just "()" = 2 bits
        let bp = JacobsonBP::new(vec![0b01], 2);
        assert_eq!(bp.len(), 2);
        assert_eq!(bp.find_close(0), Some(1));
        assert_eq!(bp.find_close(1), None); // close paren
    }
}
