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
    let len = coder.codeword_length(symbol);
    let cw = coder.encode(symbol);

    let bits = match cw {
        Some(cw) => cw,
        None => {
            let esc = coder.escape();
            let esc_len = coder.escape_length();
            let esym_len = coder.escaped_symbol_length();
            let lit = symbol.reverse_bits() >> (W::BITS - esym_len);
            esc | (u64::as_from(lit) << esc_len)
        }
    };
    let mut value = 0u64;
    for k in 0..len {
        let bit = (bits >> k) & 1;
        value |= bit << (w - 1 - k);
    }

    match decoder.decode(value) {
        Some(decoded) => {
            assert_eq!(decoded, symbol, "round-trip for symbol {symbol}");
        }
        None => {
            let esc_len = decoder.escape_length();
            let esym_len = decoder.escaped_symbol_length();
            let mask = (1u64 << esym_len) - 1;
            let recovered = (value >> (w - esc_len - esym_len)) & mask;
            assert_eq!(
                W::as_from(recovered),
                symbol,
                "escaped round-trip for symbol {symbol}"
            );
        }
    }
}

fn round_trip(coder: &HuffmanCoder<u64>, decoder: &HuffmanDecoder<u64>, symbol: u64) {
    let w = coder.max_codeword_length();
    let len = coder.codeword_length(symbol);
    let cw = coder.encode(symbol);

    // Build the same `value` the read path would build: `len` bits
    // appended LSB-first into a window, then reads the high bits.
    let bits = match cw {
        Some(cw) => cw,
        None => {
            // Escape: append escape codeword, then literal symbol
            // (bit-reversed in `escaped_symbol_length` bits).
            let esc = coder.escape();
            let esc_len = coder.escape_length();
            let esym_len = coder.escaped_symbol_length();
            let lit = symbol.reverse_bits() >> (64 - esym_len);
            esc | (lit << esc_len)
        }
    };
    // Place into the high bits of a w-bit window: position 0 →
    // bit (w-1), position k → bit (w-1-k).
    let mut value = 0u64;
    for k in 0..len {
        let bit = (bits >> k) & 1;
        value |= bit << (w - 1 - k);
    }

    match decoder.decode(value) {
        Some(decoded) => {
            assert_eq!(decoded, symbol, "round-trip for symbol {symbol}");
        }
        None => {
            // Escape: read literal bits below the escape inside the
            // w-bit window.
            let esc_len = decoder.escape_length();
            let esym_len = decoder.escaped_symbol_length();
            let mask = (1u64 << esym_len) - 1;
            let recovered = (value >> (w - esc_len - esym_len)) & mask;
            assert_eq!(recovered, symbol, "escaped round-trip");
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
fn test_huffman_length_limited() {
    // Sixteen symbols with very skewed frequencies; truncate the
    // table at four distinct lengths so the rest are escaped.
    let mut pairs = Vec::new();
    for i in 0..16u64 {
        pairs.push((100 + i, 1usize << (20 - i)));
    }
    let f = freqs(&pairs);
    let coder = Huffman::length_limited(4, 0.95).build_coder(&f);
    assert!(coder.escaped_symbol_length() > 0, "should have escapes");
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
    assert!(coder.escaped_symbol_length() > 0, "should have escapes");
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
