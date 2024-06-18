/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A pure Rust implementation of [SpookyHash::Short
//! V2](https://burtleburtle.net/bob/hash/spooky.html).
//!
//! We implement only the short version because we want to be able to precompute
//! the internal state of the hash function at regular intervals to be able to
//! hash every prefix in constant time. This feature much more complicated to
//! implement if the type of hash varies with the string length.
//!
//! We also need, in general, to access the entire 256-bit state of the hasher,
//! so we cannot use [std::hash::Hasher].
//!
//! Note that this implementation is identical to the original one, and
//! different from the one used in [Sux4J](https://sux.di.unimi.it/).

pub const SC_CONST: u64 = 0xdeadbeefdeadbeef;

#[inline(always)]
#[must_use]
const fn spooky_short_mix(mut h: [u64; 4]) -> [u64; 4] {
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

/// Rehash an internal state of SpookyHash using an additional seed.
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

/// Compute the 256-bit internal state of SpookyHash (short version)
/// for a reference to a slice of bytes.
///
/// The original implementation uses two 64-bit hash seeds: the only value
/// provided here is used for both seeds. The 128-bit standard SpookyHash
/// is given by the first two values of the returned array.
#[must_use]
#[inline]
pub fn spooky_short(data: impl AsRef<[u8]>, seed: u64) -> [u64; 4] {
    let data = data.as_ref();
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

    h[3] = h[3].wrapping_add(data.len().wrapping_shl(56) as u64);

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

    spooky_short_end(h)
}

#[test]

fn test() {
    let s = spooky_short("ciaociaociaociaoc", 0);
    assert_eq!(s[0], 0xfb9a067cf49b4b1c);
    assert_eq!(s[1], 0xd30b86ad7fb48d4);

    let s = spooky_short("ciaociaociaociaoc", 1);
    assert_eq!(s[0], 0x4b378d1bc317b08a);
    assert_eq!(s[1], 0x26087823be213893);

    let s = spooky_short("ciaociaociaociao", 0);
    assert_eq!(s[0], 0x4ff16aa850d481df);
    assert_eq!(s[1], 0xbc025187c0cb9eaf);

    let s = spooky_short("ciaociaociaocia", 0);
    assert_eq!(s[0], 0xf56ea3bd694d8c09);
    assert_eq!(s[1], 0xba8a7cfe1a359dd5);
}
