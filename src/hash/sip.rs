//! Implementation of SipHash from [Jean-Philippe Aumasson](https://www.aumasson.jp/) and Daniel J. Bernstein.
//!
//! SipHash is a general-purpose hashing function: it runs at a good
//! speed (competitive with Spooky and City) and permits strong _keyed_
//! hashing. This lets you key your hash tables from a strong RNG, such as
//! [`rand::os::OsRng`](https://docs.rs/rand/latest/rand/rngs/struct.OsRng.html).
//!
//! Although the SipHash algorithm is considered to be generally strong,
//! it is not intended for cryptographic purposes. As such, all
//! cryptographic uses of this implementation are _strongly discouraged
//!
//! # Reference
//! - <https://www.aumasson.jp/siphash/siphash.pdf>
//! - <https://131002.net/siphash/>
//! - <https://github.com/floodyberry/supercop/blob/master/crypto_auth/siphash24/sse41/siphash.c>
//! - <https://github.com/google/highwayhash/blob/master/highwayhash/sip_hash.h>
//! - <https://github.com/rust-lang/rust/blob/master/library/core/src/hash/sip.rs>
//!
//! # Reimplementation reasons
//! - The rust implementation is not const generic and is deprecated and has no simd optimzations
//! - The [most popular rust library](https://github.com/jedisct1/rust-siphash/tree/master)
//!  is just a port of rust implementation and has the same problems minus the deprecation

pub type SipHash64<const C: usize, const D: usize> = Sip64Scalar<C, D>;

/// Loads an integer of the desired type from a byte stream, in LE order. Uses
/// `copy_nonoverlapping` to let the compiler generate the most efficient way
/// to load it from a possibly unaligned address.
///
/// Unsafe because: unchecked indexing at `i..i+size_of(int_ty)`
macro_rules! load_int_le {
    ($buf:expr, $i:expr, $int_ty:ident) => {{
        debug_assert!($i + core::mem::size_of::<$int_ty>() <= $buf.len());
        let mut data = 0 as $int_ty;
        core::ptr::copy_nonoverlapping(
            $buf.as_ptr().add($i),
            &mut data as *mut _ as *mut u8,
            core::mem::size_of::<$int_ty>(),
        );
        data.to_le()
    }};
}

/// Loads a u64 using up to 7 bytes of a byte slice. It looks clumsy but the
/// `copy_nonoverlapping` calls that occur (via `load_int_le!`) all have fixed
/// sizes and avoid calling `memcpy`, which is good for speed.
///
/// Unsafe because: unchecked indexing at start..start+len
#[inline]
unsafe fn u8to64_le(buf: &[u8], start: usize, len: usize) -> u64 {
    debug_assert!(len < 8);
    let mut i = 0; // current byte index (from LSB) in the output u64
    let mut out = 0;
    if i + 3 < len {
        out = load_int_le!(buf, start + i, u32) as u64;
        i += 4;
    }
    if i + 1 < len {
        out |= (load_int_le!(buf, start + i, u16) as u64) << (i * 8);
        i += 2
    }
    if i < len {
        out |= (*buf.get_unchecked(start + i) as u64) << (i * 8);
        i += 1;
    }
    debug_assert_eq!(i, len);
    out
}

/// Porting of 64-bit SipHash from https://github.com/veorq/SipHash
#[derive(Debug, Clone, Copy)]
#[repr(align(16), C)] //alignement and C repr so the compiler can use simd instructions
pub struct Sip64Scalar<const C: usize, const D: usize> {
    // interleave the v values so that the compiler can optimize with simd
    v0: u64,
    v2: u64,
    v1: u64,
    v3: u64,
    k0: u64,
    k1: u64,
    /// precedent message
    m: u64,
    /// how many bytes we've processed
    length: usize,
    /// buffer of unprocessed bytes in little endian order
    tail: u64,
    /// how many bytes in tail are valid
    ntail: usize,
}

impl<const C: usize, const D: usize> core::default::Default for Sip64Scalar<C, D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, const D: usize> crate::hash::SeedableHasher for Sip64Scalar<C, D> {
    type SeedType = [u8; 16];

    /// Creates a new hasher with the given seed.
    #[inline(always)]
    fn new_with_seed(seed: Self::SeedType) -> Self {
        Self::new_with_key(seed)
    }
}

impl<const C: usize, const D: usize> Sip64Scalar<C, D> {
    #[inline]
    pub fn new() -> Self {
        // same default key as rust implementation
        Self::new_with_key([0; 16])
    }

    #[inline(always)]
    pub fn new_with_key(key: [u8; 16]) -> Self {
        Self::new_with_key_and_state(
            key,
            [
                0x736f6d6570736575,
                0x646f72616e646f6d,
                0x6c7967656e657261,
                0x7465646279746573,
            ],
        )
    }

    #[inline(always)]
    pub fn new_with_key_and_state(key: [u8; 16], state: [u64; 4]) -> Self {
        let mut res = Self {
            v0: state[0],
            v1: state[1],
            v2: state[2],
            v3: state[3],
            k0: u64::from_le_bytes(key[0..8].try_into().unwrap()),
            k1: u64::from_le_bytes(key[8..16].try_into().unwrap()),
            m: 0,
            length: 0,
            tail: 0,
            ntail: 0,
        };

        res.v3 ^= res.k1;
        res.v2 ^= res.k0;
        res.v1 ^= res.k1;
        res.v0 ^= res.k0;

        res
    }

    #[inline(always)]
    fn round(&mut self) {
        self.v0 = self.v0.wrapping_add(self.v1);
        self.v2 = self.v2.wrapping_add(self.v3);

        self.v1 = self.v1.rotate_left(13);
        self.v3 = self.v3.rotate_left(16);

        self.v1 ^= self.v0;
        self.v3 ^= self.v2;

        self.v0 = self.v0.rotate_left(32);

        self.v0 = self.v0.wrapping_add(self.v3);
        self.v2 = self.v2.wrapping_add(self.v1);

        self.v1 = self.v1.rotate_left(17);
        self.v3 = self.v3.rotate_left(21);

        self.v3 ^= self.v0;
        self.v1 ^= self.v2;

        self.v2 = self.v2.rotate_left(32);
    }

    // A specialized write function for values with size <= 8.
    //
    // The hashing of multi-byte integers depends on endianness. E.g.:
    // - little-endian: `write_u32(0xDDCCBBAA)` == `write([0xAA, 0xBB, 0xCC, 0xDD])`
    // - big-endian:    `write_u32(0xDDCCBBAA)` == `write([0xDD, 0xCC, 0xBB, 0xAA])`
    //
    // This function does the right thing for little-endian hardware. On
    // big-endian hardware `x` must be byte-swapped first to give the right
    // behaviour. After any byte-swapping, the input must be zero-extended to
    // 64-bits. The caller is responsible for the byte-swapping and
    // zero-extension.
    #[inline]
    pub fn short_write<T>(&mut self, _x: T, x: u64) {
        let size = core::mem::size_of::<T>();
        self.length += size;

        // The original number must be zero-extended, not sign-extended.
        debug_assert!(if size < 8 { x >> (8 * size) == 0 } else { true });

        // The number of bytes needed to fill `self.tail`.
        let needed = 8 - self.ntail;

        self.tail |= x << (8 * self.ntail);
        if size < needed {
            self.ntail += size;
            return;
        }

        // `self.tail` is full, process it.
        self.v3 ^= self.tail;
        for _ in 0..C {
            self.round();
        }
        self.v0 ^= self.tail;

        self.ntail = size - needed;
        self.tail = if needed < 8 { x >> (8 * needed) } else { 0 };
    }
}

impl<const C: usize, const D: usize> core::hash::Hasher for Sip64Scalar<C, D> {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        let length = msg.len();
        self.length += length;

        let mut needed = 0;

        if self.ntail != 0 {
            needed = 8 - self.ntail;
            self.tail |=
                unsafe { u8to64_le(msg, 0, core::cmp::min(length, needed)) } << (8 * self.ntail);
            if length < needed {
                self.ntail += length;
                return;
            } else {
                self.v3 ^= self.tail;
                for _ in 0..C {
                    self.round();
                }
                self.v0 ^= self.tail;
                self.ntail = 0;
            }
        }

        // Buffered tail is now flushed, process new input.
        let len = length - needed;
        let left = len & 0x7;

        let mut i = needed;
        while i < len - left {
            let mi = unsafe { load_int_le!(msg, i, u64) };

            self.v3 ^= mi;
            for _ in 0..C {
                self.round();
            }
            self.v0 ^= mi;

            i += 8;
        }

        self.tail = unsafe { u8to64_le(msg, i, left) };
        self.ntail = left;
    }

    #[inline]
    fn finish(&self) -> u64 {
        let mut state = *self;

        let b: u64 = ((self.length as u64 & 0xff) << 56) | self.tail;

        state.v3 ^= b;
        for _ in 0..C {
            state.round();
        }
        state.v0 ^= b;

        state.v2 ^= 0xff;
        for _ in 0..D {
            state.round();
        }

        state.v0 ^ state.v1 ^ state.v2 ^ state.v3
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.short_write(i, i.to_le() as u64);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.short_write(i, i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.short_write(i, i.to_le() as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.short_write(i, i.to_le());
    }
}

/// Porting of 128-bit SipHash from https://github.com/veorq/SipHash
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Sip128Scalar<const C: usize, const D: usize>(Sip64Scalar<C, D>);

impl<const C: usize, const D: usize> core::default::Default for Sip128Scalar<C, D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, const D: usize> Sip128Scalar<C, D> {
    #[inline]
    pub fn new() -> Self {
        Self(<Sip64Scalar<C, D>>::new())
    }

    #[inline(always)]
    pub fn new_with_key(key: [u8; 16]) -> Self {
        Self(<Sip64Scalar<C, D>>::new_with_key(key))
    }

    #[inline(always)]
    pub fn new_with_key_and_state(key: [u8; 16], state: [u64; 4]) -> Self {
        Self(<Sip64Scalar<C, D>>::new_with_key_and_state(key, state))
    }
}

impl<const C: usize, const D: usize> crate::hash::SeedableHasher for Sip128Scalar<C, D> {
    type SeedType = [u8; 16];

    /// Creates a new hasher with the given seed.
    #[inline(always)]
    fn new_with_seed(seed: Self::SeedType) -> Self {
        Self::new_with_key(seed)
    }
}

impl<const C: usize, const D: usize> crate::hash::Hasher for Sip128Scalar<C, D> {
    type HashType = u128;

    #[inline(always)]
    fn write(&mut self, msg: &[u8]) {
        self.0.write(msg)
    }

    #[inline]
    fn finish(&self) -> u128 {
        let mut state = *self;

        let b: u64 = ((self.0.length as u64 & 0xff) << 56) | self.0.tail;

        state.0.v3 ^= b;
        for _ in 0..C {
            state.0.round();
        }
        state.0.v0 ^= b;

        state.0.v2 ^= 0xff;
        for _ in 0..D {
            state.0.round();
        }

        let low = state.0.v0 ^ state.0.v1 ^ state.0.v2 ^ state.0.v3;

        state.0.v1 ^= 0xdd;
        for _ in 0..D {
            state.0.round();
        }

        let high = state.0.v0 ^ state.0.v1 ^ state.0.v2 ^ state.0.v3;

        ((high as u128) << 64) | (low as u128)
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0.write_usize(i);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.0.write_u8(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0.write_u32(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i);
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::Hasher;

    use super::*;
    #[test]
    fn test_siphasher() {
        let data = (0..255_u8).collect::<Vec<_>>();
        for i in 0..16 {
            #[allow(deprecated)]
            let mut sip =
                core::hash::SipHasher::new_with_keys(0x0706050403020100, 0x0f0e0d0c0b0a0908);
            sip.write(&data[..i]);
            let truth = sip.finish();

            let mut sip = <Sip64Scalar<2, 4>>::new_with_key([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ]);
            sip.write(&data[..i]);
            assert_eq!(sip.finish(), truth);
        }
    }
    #[test]
    fn test_default() {
        let data = (0..255_u8).collect::<Vec<_>>();
        for i in 0..16 {
            #[allow(deprecated)]
            let mut sip = std::collections::hash_map::DefaultHasher::new();
            sip.write(&data[..i]);
            let truth = sip.finish();

            let mut sip = <Sip64Scalar<1, 3>>::new();
            sip.write(&data[..i]);
            assert_eq!(sip.finish(), truth);
        }
    }
}
