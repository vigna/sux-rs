/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use num_primitive::{PrimitiveInteger, PrimitiveNumber, PrimitiveNumberAs};
use std::collections::HashMap;
use sux::func::codec::{Codec, Coder, Decoder, Huffman, HuffmanCoder, HuffmanDecoder, ZeroCodec};

fn freqs(pairs: &[(u64, usize)]) -> HashMap<u64, usize> {
    pairs.iter().copied().collect()
}

fn freqs_i8(pairs: &[(i8, usize)]) -> HashMap<i8, usize> {
    pairs.iter().copied().collect()
}

fn freqs_i32(pairs: &[(i32, usize)]) -> HashMap<i32, usize> {
    pairs.iter().copied().collect()
}

fn round_trip_generic<W: PrimitiveInteger + std::hash::Hash + std::fmt::Display>(
    coder: &HuffmanCoder<W>,
    decoder: &HuffmanDecoder<W>,
    symbol: W,
) where
    u64: PrimitiveNumberAs<W>,
{
    let w = coder.max_codeword_length();

    match coder.encode(symbol) {
        Some(cw) => {
            let len = coder.codeword_length(symbol);
            let mut value = 0u64;
            for k in 0..len {
                value |= ((cw >> k) & 1) << (w - 1 - k);
            }
            let decoded = decoder.decode(value);
            assert_eq!(decoded, Some(symbol), "round-trip for symbol {symbol}");
        }
        None => {
            let esc = coder.escape_codeword();
            let esc_len = coder.escape_codeword_length();
            let esym_len = coder.escaped_symbols_length();

            // Verify escape codeword is recognized
            let mut value = 0u64;
            for k in 0..esc_len {
                value |= ((esc >> k) & 1) << (w - 1 - k);
            }
            assert!(
                decoder.decode(value).is_none(),
                "escape should return None for {symbol}"
            );

            // Verify literal round-trip: the double bit-reversal in
            // CompVFunc's encoding places the original value in the
            // low esym_len bits of the stored window.
            if esym_len > 0 {
                let lit = u64::as_from(symbol.reverse_bits() >> (W::BITS - esym_len));
                let full_bits: u128 = esc as u128 | ((lit as u128) << esc_len);
                let total_len = esc_len + esym_len;
                let mut full_window = 0u128;
                for k in 0..total_len {
                    full_window |= ((full_bits >> k) & 1) << (total_len - 1 - k);
                }
                let mask = (1u128 << esym_len) - 1;
                assert_eq!(
                    W::as_from((full_window & mask) as u64),
                    symbol,
                    "escaped round-trip for symbol {symbol}"
                );
            }
        }
    }
}

fn round_trip(coder: &HuffmanCoder<u64>, decoder: &HuffmanDecoder<u64>, symbol: u64) {
    let w = coder.max_codeword_length();

    match coder.encode(symbol) {
        Some(cw) => {
            let len = coder.codeword_length(symbol);
            let mut value = 0u64;
            for k in 0..len {
                value |= ((cw >> k) & 1) << (w - 1 - k);
            }
            let decoded = decoder.decode(value);
            assert_eq!(decoded, Some(symbol), "round-trip for symbol {symbol}");
        }
        None => {
            let esc = coder.escape_codeword();
            let esc_len = coder.escape_codeword_length();
            let esym_len = coder.escaped_symbols_length();

            // Verify escape codeword is recognized
            let mut value = 0u64;
            for k in 0..esc_len {
                value |= ((esc >> k) & 1) << (w - 1 - k);
            }
            assert!(
                decoder.decode(value).is_none(),
                "escape should return None for {symbol}"
            );

            // Verify literal round-trip
            if esym_len > 0 {
                let lit = symbol.reverse_bits() >> (64 - esym_len);
                let full_bits: u128 = esc as u128 | ((lit as u128) << esc_len);
                let total_len = esc_len + esym_len;
                let mut full_window = 0u128;
                for k in 0..total_len {
                    full_window |= ((full_bits >> k) & 1) << (total_len - 1 - k);
                }
                let mask = (1u128 << esym_len) - 1;
                assert_eq!(
                    (full_window & mask) as u64,
                    symbol,
                    "escaped round-trip for symbol {symbol}"
                );
            }
        }
    }
}

#[test]
fn test_zero_codec() {
    let freqs: HashMap<u64, usize> = HashMap::new();
    let coder = <ZeroCodec as Codec<u64>>::build_coder(&ZeroCodec, &freqs);
    let decoder = <sux::func::codec::ZeroCoder as Coder<u64>>::into_decoder(coder);
    let out = decoder.decode(0xdead_beef);
    assert_eq!(out, Some(0));
}

#[test]
fn test_huffman_single_symbol() {
    let f = freqs(&[(42, 100)]);
    let coder = Huffman::new().build_coder(&f);
    // One real symbol → length 1 codeword "0".
    assert_eq!(coder.codeword_length(42), 1);
    let decoder = coder.clone().into_decoder();
    round_trip(&coder, &decoder, 42);
}

#[test]
fn test_huffman_two_symbols() {
    // Regression: with exactly 2 symbols the Moffat-Katajainen
    // second pass range `0..=size-3` would underflow to
    // `0..=usize::MAX`, crashing with index out of bounds.
    let f = freqs(&[(10, 7), (20, 3)]);
    let coder = Huffman::new().build_coder(&f);
    assert_eq!(coder.max_codeword_length(), 1);
    let decoder = coder.clone().into_decoder();
    round_trip(&coder, &decoder, 10);
    round_trip(&coder, &decoder, 20);
}

#[test]
fn test_huffman_three_symbols() {
    // Skewed frequencies: A is most frequent.
    let f = freqs(&[(0, 5), (1, 2), (2, 1)]);
    let coder = Huffman::new().build_coder(&f);
    let decoder = coder.clone().into_decoder();
    for &s in &[0u64, 1, 2] {
        round_trip(&coder, &decoder, s);
    }
}

#[test]
fn test_huffman_many_symbols() {
    let mut pairs = Vec::new();
    // Geometric-ish distribution.
    for i in 0..16u64 {
        pairs.push((i, 1usize << (16 - i.min(15))));
    }
    let f = freqs(&pairs);
    let coder = Huffman::new().build_coder(&f);
    let decoder = coder.clone().into_decoder();
    for (s, _) in pairs {
        round_trip(&coder, &decoder, s);
    }
}

#[test]
fn test_huffman_u128_large_values() {
    // u128 symbols well above the 2^64 range. With enough distinct
    // frequent symbols the codewords stay short and the "kept" set
    // carries the entire distribution — no escapes needed, so the
    // decoder never has to reconstruct a literal wider than a u64
    // window (which is the current read-path limit).
    let mut pairs: Vec<(u128, usize)> = Vec::new();
    let base: u128 = 1u128 << 100;
    for i in 0..8u128 {
        pairs.push((base + i, 1usize << (16 - i)));
    }
    let f: HashMap<u128, usize> = pairs.iter().copied().collect();
    let coder = <Huffman as Codec<u128>>::build_coder(&Huffman::new(), &f);
    let decoder = coder.clone().into_decoder();
    for (s, _) in &pairs {
        // Build the same `value` the read path would build for the
        // (unescaped) codeword, then decode.
        let w = coder.max_codeword_length();
        let len = coder.codeword_length(*s);
        let bits = coder.encode(*s).expect("symbol must be in the table");
        let mut value = 0u64;
        for k in 0..len {
            let bit = (bits >> k) & 1;
            value |= bit << (w - 1 - k);
        }
        let decoded = decoder.decode(value).expect("should not be escape");
        assert_eq!(decoded, *s, "u128 round-trip for {s:x}");
    }
}

#[test]
fn test_huffman_u128_large_values_unlimited() {
    let mut pairs: Vec<(u128, usize)> = Vec::new();
    let base: u128 = 1u128 << 100;
    for i in 0..8u128 {
        pairs.push((base + i, 1usize << (16 - i)));
    }
    let f: HashMap<u128, usize> = pairs.iter().copied().collect();
    let coder = <Huffman as Codec<u128>>::build_coder(&Huffman::unlimited(), &f);
    let decoder = coder.clone().into_decoder();
    for (s, _) in &pairs {
        let w = coder.max_codeword_length();
        let len = coder.codeword_length(*s);
        let bits = coder.encode(*s).expect("symbol must be in the table");
        let mut value = 0u64;
        for k in 0..len {
            let bit = (bits >> k) & 1;
            value |= bit << (w - 1 - k);
        }
        let decoded = decoder.decode(value).expect("should not be escape");
        assert_eq!(decoded, *s, "u128 round-trip for {s:x}");
    }
}

#[test]
fn test_huffman_length_limited() {
    // Sixteen symbols with very skewed frequencies; truncate the
    // table at four distinct lengths so the rest are escaped.
    let mut pairs = Vec::new();
    for i in 0..16u64 {
        pairs.push((100 + i, 1usize << (20 - i)));
    }
    let f = freqs(&pairs);
    let coder = Huffman::length_limited(4, 0.95).build_coder(&f);
    assert!(coder.escaped_symbols_length() > 0, "should have escapes");
    let decoder = coder.clone().into_decoder();
    for (s, _) in pairs {
        round_trip(&coder, &decoder, s);
    }
}

#[test]
fn test_huffman_i8_with_negatives() {
    // Signed i8 values including negatives: -5 (0xFB), -1 (0xFF),
    // 0, 42, 127 (i8::MAX — no longer reserved).
    let f = freqs_i8(&[(-5, 100), (-1, 50), (0, 30), (42, 20), (127, 10)]);
    let coder = <Huffman as Codec<i8>>::build_coder(&Huffman::new(), &f);
    let decoder = coder.clone().into_decoder();
    for &s in &[-5i8, -1, 0, 42, 127] {
        round_trip_generic(&coder, &decoder, s);
    }
}

#[test]
fn test_huffman_i8_with_escapes() {
    // Many i8 symbols with extreme skew + tight length limit forces
    // the rare ones (including negatives) into the escape path.
    let mut pairs: Vec<(i8, usize)> = Vec::new();
    for i in -64..64i8 {
        pairs.push((i, 1usize << (7 - (i.unsigned_abs().min(6)) as u32)));
    }
    let f = freqs_i8(&pairs);
    let coder = <Huffman as Codec<i8>>::build_coder(&Huffman::length_limited(3, 0.6), &f);
    assert!(coder.escaped_symbols_length() > 0, "should have escapes");
    let decoder = coder.clone().into_decoder();
    for (s, _) in pairs {
        round_trip_generic(&coder, &decoder, s);
    }
}

#[test]
fn test_huffman_i32_with_negatives() {
    // Signed i32 values with large negative numbers.
    let f = freqs_i32(&[
        (-1_000_000, 100),
        (-1, 80),
        (0, 60),
        (1, 40),
        (i32::MAX, 20),
        (i32::MIN, 10),
    ]);
    let coder = <Huffman as Codec<i32>>::build_coder(&Huffman::new(), &f);
    let decoder = coder.clone().into_decoder();
    for &s in &[-1_000_000i32, -1, 0, 1, i32::MAX, i32::MIN] {
        round_trip_generic(&coder, &decoder, s);
    }
}
