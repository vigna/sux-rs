/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A pure Rust implementation of [SpookyHash](https://burtleburtle.net/bob/hash/spooky.html).
//!
//! Ported from <https://github.com/vigna/Sux4J/blob/master/c/mph.c>.

// The original is SC_CONST = 0xdeadbeefdeadbeef but we must
// be compatible with the original Java implementation in Sux4J.
pub const SC_CONST: u64 = 0x9e3779b97f4a7c13;

#[inline(always)]
#[must_use]
pub const fn spooky_short_mix(mut h: [u64; 4]) -> [u64; 4] {
    h[2] = h[2].rotate_left(50);
    h[2] = h[2].wrapping_add(h[3]);
    h[0] ^= h[2];
    h[3] = h[3].rotate_left(52);
    h[3] = h[3].wrapping_add(h[0]);
    h[1] ^= h[3];
    h[0] = h[0].rotate_left(30);
    h[0] = h[0].wrapping_add(h[1]);
    h[2] ^= h[0];
    h[1] = h[1].rotate_left(41);
    h[1] = h[1].wrapping_add(h[2]);
    h[3] ^= h[1];
    h[2] = h[2].rotate_left(54);
    h[2] = h[2].wrapping_add(h[3]);
    h[0] ^= h[2];
    h[3] = h[3].rotate_left(48);
    h[3] = h[3].wrapping_add(h[0]);
    h[1] ^= h[3];
    h[0] = h[0].rotate_left(38);
    h[0] = h[0].wrapping_add(h[1]);
    h[2] ^= h[0];
    h[1] = h[1].rotate_left(37);
    h[1] = h[1].wrapping_add(h[2]);
    h[3] ^= h[1];
    h[2] = h[2].rotate_left(62);
    h[2] = h[2].wrapping_add(h[3]);
    h[0] ^= h[2];
    h[3] = h[3].rotate_left(34);
    h[3] = h[3].wrapping_add(h[0]);
    h[1] ^= h[3];
    h[0] = h[0].rotate_left(5);
    h[0] = h[0].wrapping_add(h[1]);
    h[2] ^= h[0];
    h[1] = h[1].rotate_left(36);
    h[1] = h[1].wrapping_add(h[2]);
    h[3] ^= h[1];
    h
}

#[inline(always)]
#[must_use]
const fn spooky_short_end(mut h: [u64; 4]) -> [u64; 4] {
    h[3] ^= h[2];
    h[2] = h[2].rotate_left(15);
    h[3] = h[3].wrapping_add(h[2]);
    h[0] ^= h[3];
    h[3] = h[3].rotate_left(52);
    h[0] = h[0].wrapping_add(h[3]);
    h[1] ^= h[0];
    h[0] = h[0].rotate_left(26);
    h[1] = h[1].wrapping_add(h[0]);
    h[2] ^= h[1];
    h[1] = h[1].rotate_left(51);
    h[2] = h[2].wrapping_add(h[1]);
    h[3] ^= h[2];
    h[2] = h[2].rotate_left(28);
    h[3] = h[3].wrapping_add(h[2]);
    h[0] ^= h[3];
    h[3] = h[3].rotate_left(9);
    h[0] = h[0].wrapping_add(h[3]);
    h[1] ^= h[0];
    h[0] = h[0].rotate_left(47);
    h[1] = h[1].wrapping_add(h[0]);
    h[2] ^= h[1];
    h[1] = h[1].rotate_left(54);
    h[2] = h[2].wrapping_add(h[1]);
    h[3] ^= h[2];
    h[2] = h[2].rotate_left(32);
    h[3] = h[3].wrapping_add(h[2]);
    h[0] ^= h[3];
    h[3] = h[3].rotate_left(25);
    h[0] = h[0].wrapping_add(h[3]);
    h[1] ^= h[0];
    h[0] = h[0].rotate_left(63);
    h[1] = h[1].wrapping_add(h[0]);
    h
}

#[inline(always)]
#[must_use]
pub const fn spooky_short_rehash(signature: &[u64; 4], seed: u64) -> [u64; 4] {
    spooky_short_mix([
        seed,
        SC_CONST.wrapping_add(signature[0]),
        SC_CONST.wrapping_add(signature[1]),
        SC_CONST,
    ])
}

#[must_use]
#[inline]
pub fn spooky_short(data: &[u8], seed: u64) -> [u64; 4] {
    let mut h = [seed, seed, SC_CONST, SC_CONST];

    let iter = data.chunks_exact(32);
    let mut reminder = iter.remainder();

    for chunk in iter {
        // handle all complete sets of 32 bytes
        h[2] = h[2].wrapping_add(u64::from_le_bytes(chunk[0..8].try_into().unwrap()));
        h[3] = h[3].wrapping_add(u64::from_le_bytes(chunk[8..16].try_into().unwrap()));
        h = spooky_short_mix(h);
        h[0] = h[0].wrapping_add(u64::from_le_bytes(chunk[16..24].try_into().unwrap()));
        h[1] = h[1].wrapping_add(u64::from_le_bytes(chunk[24..32].try_into().unwrap()));
    }

    //Handle the case of 16+ remaining bytes.
    if reminder.len() >= 16 {
        h[2] = h[2].wrapping_add(u64::from_le_bytes(reminder[0..8].try_into().unwrap()));
        h[3] = h[3].wrapping_add(u64::from_le_bytes(reminder[8..16].try_into().unwrap()));
        h = spooky_short_mix(h);
        reminder = &reminder[16..];
    }

    // Handle the last 0..15 bytes, and its length
    // We copy it into a buffer filled with zeros so we can simplify the
    // code
    if reminder.is_empty() {
        h[2] = h[2].wrapping_add(SC_CONST);
        h[3] = h[3].wrapping_add(SC_CONST);
    } else {
        let mut buffer: [u8; 16] = [0; 16];
        buffer[..reminder.len()].copy_from_slice(reminder);
        h[2] = h[2].wrapping_add(u64::from_le_bytes(buffer[0..8].try_into().unwrap()));
        h[3] = h[3].wrapping_add(u64::from_le_bytes(buffer[8..16].try_into().unwrap()));
    }

    h[0] = h[0].wrapping_add((data.len() as u64) * 8);
    spooky_short_end(h)
}
