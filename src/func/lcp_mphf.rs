/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! LCP-based monotone minimal perfect hash function.
//!
//! We provide two variants:
//!
//! - [`LcpMinPerfHashFuncStr`]

use crate::bits::BitFieldVec;
use crate::func::VBuilder;
use crate::func::VFunc;
use crate::func::shard_edge::{Fuse3NoShards, FuseLge3Shards, ShardEdge};
use crate::utils::*;
use anyhow::{Result, bail};
use dsi_progress_logger::ProgressLog;
use lender::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use rdst::RadixKey;
use std::borrow::Borrow;
use xxhash_rust::xxh3;

/// A bit-level prefix of an integer, used as key for the LCP-to-bucket
/// mapping.
///
/// Stores the original integer value and a bit length. The [`ToSig`]
/// implementation hashes only the top `bit_length` bits (by right-shifting
/// to discard the rest) followed by the bit length, directly on the stack
/// with no allocation.
#[derive(Debug, Clone, Copy, MemDbg, MemSize)]
#[mem_size_flat]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntBitPrefix<T: PrimitiveInteger> {
    /// The original integer value.
    value: T,
    /// Number of significant leading bits.
    bit_length: usize,
}

impl<T: PrimitiveInteger> IntBitPrefix<T> {
    /// Creates a new integer bit prefix.
    #[inline]
    pub fn new(value: T, bit_length: usize) -> Self {
        debug_assert!(bit_length <= T::BITS as usize);
        Self { value, bit_length }
    }

    /// Returns the top `bit_length` bits, right-aligned (trailing bits
    /// discarded).
    #[inline]
    fn significant_bits(&self) -> T {
        if self.bit_length == 0 {
            T::MIN // zero for unsigned types
        } else {
            self.value >> (T::BITS as usize - self.bit_length)
        }
    }
}

/// Returns a byte-slice view of a `PrimitiveInteger` value. TODO
#[inline]
fn as_bytes<T: PrimitiveInteger>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, size_of::<T>()) }
}

/// Packs significant bits and bit_length into a contiguous stack buffer
/// for one-shot xxh3 hashing. Returns the number of bytes written.
/// Buffer must be at least `size_of::<T>() + size_of::<usize>()` bytes.
#[inline]
fn pack_int_bit_prefix<T: PrimitiveInteger>(bp: &IntBitPrefix<T>, buf: &mut [u8]) -> usize {
    let sig = bp.significant_bits();
    let val = as_bytes(&sig);
    let len = bp.bit_length.to_ne_bytes();
    let n = val.len() + len.len();
    buf[..val.len()].copy_from_slice(val);
    buf[val.len()..n].copy_from_slice(&len);
    n
}

impl<T: PrimitiveInteger> ToSig<[u64; 2]> for IntBitPrefix<T> {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        // 24 bytes covers u128 (16) + usize (8) on 64-bit.
        let mut buf = [0u8; 24];
        let n = pack_int_bit_prefix(key.borrow(), &mut buf);
        let h = xxh3::xxh3_128_with_seed(&buf[..n], seed);
        [(h >> 64) as u64, h as u64]
    }
}

impl<T: PrimitiveInteger> ToSig<[u64; 1]> for IntBitPrefix<T> {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut buf = [0u8; 24];
        let n = pack_int_bit_prefix(key.borrow(), &mut buf);
        [xxh3::xxh3_64_with_seed(&buf[..n], seed)]
    }
}

/// Returns the number of leading bits shared by two integers of the same
/// type, compared MSB-first (big-endian bit order).
#[inline(always)]
fn lcp_bits<T: PrimitiveInteger>(a: T, b: T) -> usize {
    (a ^ b).leading_zeros() as usize
}

/// Computes the log2 of bucket size from *n* using the LCP-MPHF formula.
fn log2_bucket_size(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let ln_n = (n as f64).ln();
    let theoretical = 1.0 + 1.11 * f64::ln(2.0) + ln_n - (1.0 + ln_n).ln();
    let theoretical = theoretical.ceil().max(1.0) as usize;
    theoretical.next_power_of_two().ilog2() as usize
}

/// A monotone minimal perfect hash function for sorted integers, based on
/// longest common bit-prefixes (LCPs).
///
/// Given *n* integers of type `T` in ascending order, the structure maps
/// each integer to its rank (0 to *n* − 1). Querying an integer not in
/// the original set returns an arbitrary value (same contract as
/// [`VFunc`]).
///
/// The integers are divided into buckets of size 2^*k*. For each bucket,
/// the longest common bit-prefix shared by all consecutive pairs within
/// the bucket is computed (starting from the full bit width of the first
/// key). The rank of a key is then reconstructed as
/// `bucket * bucket_size + offset`, where the bucket is determined by the
/// bit-prefix and the offset is stored directly.
///
/// Internally, the structure contains two [`VFunc`]s:
/// - `offset_lcp_length`: maps each key (`T`) to a packed value encoding
///   the LCP bit-length and the offset within the bucket;
/// - `lcp2bucket`: maps each LCP bit-prefix ([`IntBitPrefix`]) to its
///   bucket index.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMinPerfHashFuncInt<
    T: PrimitiveInteger + ToSig<S>,
    S: Sig = [u64; 2],
    E: ShardEdge<S, 3> = FuseLge3Shards,
> {
    /// Number of keys.
    n: usize,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    offset_lcp_length: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    lcp2bucket: VFunc<IntBitPrefix<T>, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

impl<T: PrimitiveInteger + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> LcpMinPerfHashFuncInt<T, S, E> {
    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary
    /// (same contract as [`VFunc::get`]).
    #[inline]
    pub fn get(&self, key: T) -> usize
    where
        T: Copy,
    {
        let packed = self.offset_lcp_length.get(key);
        let lcp_bit_length = packed >> self.log2_bucket_size;
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        let prefix = IntBitPrefix::new(key, lcp_bit_length);
        let bucket = self.lcp2bucket.get(prefix);
        (bucket << self.log2_bucket_size) + offset
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

#[cfg(feature = "rayon")]
impl<T, S, E> LcpMinPerfHashFuncInt<T, S, E>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a new LCP-based monotone minimal perfect hash function for
    /// integers.
    ///
    /// The keys must be provided in strictly increasing order.
    ///
    /// # Arguments
    ///
    /// * `keys`: a lender yielding `&T` references in sorted order.
    ///
    /// * `n`: the number of keys.
    ///
    /// * `pl`: a progress logger.
    pub fn new(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if n == 0 {
            let empty_keys_t: Vec<T> = vec![];
            let empty_keys_bp: Vec<IntBitPrefix<T>> = vec![];
            let empty_vals: Vec<usize> = vec![];

            let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
                .try_build_func::<T, T>(
                    FromSlice::new(&empty_keys_t),
                    FromSlice::new(&empty_vals),
                    pl,
                )?;
            let lcp2bucket =
                VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                    .try_build_func::<IntBitPrefix<T>, IntBitPrefix<T>>(
                    FromSlice::new(&empty_keys_bp),
                    FromSlice::new(&empty_vals),
                    pl,
                )?;
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offset_lcp_length,
                lcp2bucket,
            });
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        // -- First pass: compute bit-level LCPs --

        let mut lcp_bit_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut bucket_first_keys: Vec<T> = Vec::with_capacity(num_buckets);

        let mut prev_key: Option<T> = None;
        let mut curr_lcp_bits: usize = 0;
        let mut i = 0usize;

        while let Some(key) = keys.next()? {
            let key: T = *key;

            if let Some(prev) = prev_key {
                if key <= prev {
                    bail!(
                        "Keys are not in strictly increasing order at \
                         position {i}: {prev:?} >= {key:?}"
                    );
                }
            }

            let offset = i & bucket_mask;

            if offset == 0 {
                // First key of a new bucket.
                if i > 0 {
                    lcp_bit_lengths.push(curr_lcp_bits);
                }
                bucket_first_keys.push(key);
                // Initialize to full bit width (integers have fixed
                // width, so no prefix-freeness issue).
                curr_lcp_bits = T::BITS as usize;
            } else {
                // Subsequent key: minimize LCP.
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits(key, prev_key.unwrap()));
            }

            prev_key = Some(key);
            i += 1;
        }

        assert_eq!(i, n, "Expected {n} keys but got {i}");
        lcp_bit_lengths.push(curr_lcp_bits);
        assert_eq!(lcp_bit_lengths.len(), num_buckets);

        // -- Build packed values --

        let packed_values: Vec<usize> = (0..n)
            .map(|idx| {
                let bucket = idx >> log2_bs;
                let offset = idx & bucket_mask;
                (lcp_bit_lengths[bucket] << log2_bs) | offset
            })
            .collect();

        // -- Build offset_lcp_length VFunc --

        let keys = keys.rewind()?;

        let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
            .expected_num_keys(n)
            .try_build_func::<T, T>(keys, FromSlice::new(&packed_values), pl)?;

        // -- Build lcp2bucket VFunc --

        let bit_prefixes: Vec<IntBitPrefix<T>> = (0..num_buckets)
            .map(|b| IntBitPrefix::new(bucket_first_keys[b], lcp_bit_lengths[b]))
            .collect();

        let bucket_indices: Vec<usize> = (0..num_buckets).collect();

        let lcp2bucket =
            VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                .expected_num_keys(num_buckets)
                .try_build_func::<IntBitPrefix<T>, IntBitPrefix<T>>(
                    FromSlice::new(&bit_prefixes),
                    FromSlice::new(&bucket_indices),
                    pl,
                )?;

        Ok(Self {
            n,
            log2_bucket_size: log2_bs,
            offset_lcp_length,
            lcp2bucket,
        })
    }
}

/// A bit-level prefix of a byte slice, used as key for the LCP-to-bucket
/// mapping.
///
/// Holds the bytes and a bit length. The [`ToSig`] implementation hashes
/// only the first `bit_length` bits, masking out unused bits in the last
/// partial byte.
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitPrefix {
    bytes: Vec<u8>,
    bit_length: usize,
}

impl BitPrefix {
    /// Creates a new bit prefix from a byte slice and a bit length.
    ///
    /// `bytes` must contain at least `ceil(bit_length / 8)` bytes.
    pub fn new(bytes: &[u8], bit_length: usize) -> Self {
        debug_assert!(bytes.len() * 8 >= bit_length);
        let needed = bit_length.div_ceil(8);
        Self {
            bytes: bytes[..needed].to_vec(),
            bit_length,
        }
    }
}

/// Feeds a bit prefix (given as raw bytes + bit length) into an [`Xxh3`]
/// hasher: the full bytes, the masked partial byte (if any), and the bit
/// length. Including the bit length distinguishes prefixes that differ
/// only by trailing zero bits.
#[inline]
fn hash_bit_prefix_raw(hasher: &mut xxh3::Xxh3, bytes: &[u8], bit_length: usize) {
    let full_bytes = bit_length / 8;
    let extra_bits = bit_length % 8;
    hasher.update(&bytes[..full_bytes]);
    if extra_bits > 0 {
        let mask = !((1u8 << (8 - extra_bits)) - 1);
        hasher.update(&[bytes[full_bytes] & mask]);
    }
    hasher.update(&bit_length.to_ne_bytes());
}

/// Computes a `[u64; 1]` signature from raw bytes and a bit length,
/// matching the [`BitPrefix`] `ToSig<[u64; 1]>` implementation but
/// without allocating a `BitPrefix`.
#[inline]
fn bit_prefix_sig(bytes: &[u8], bit_length: usize, seed: u64) -> [u64; 1] {
    let mut hasher = xxh3::Xxh3::with_seed(seed);
    hash_bit_prefix_raw(&mut hasher, bytes, bit_length);
    [hasher.digest()]
}

impl ToSig<[u64; 2]> for BitPrefix {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        let bp = key.borrow();
        let mut hasher = xxh3::Xxh3::with_seed(seed);
        hash_bit_prefix_raw(&mut hasher, &bp.bytes, bp.bit_length);
        let h = hasher.digest128();
        [(h >> 64) as u64, h as u64]
    }
}

impl ToSig<[u64; 1]> for BitPrefix {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let bp = key.borrow();
        let mut hasher = xxh3::Xxh3::with_seed(seed);
        hash_bit_prefix_raw(&mut hasher, &bp.bytes, bp.bit_length);
        [hasher.digest()]
    }
}

/// A monotone minimal perfect hash function for lexicographically sorted
/// strings, based on longest common prefixes (LCPs).
///
/// Given *n* strings in lexicographic order, the structure maps each string
/// to its rank (0 to *n* − 1). Querying a string not in the original set
/// returns an arbitrary value (same contract as [`VFunc`]).
///
/// The strings are divided into buckets of size 2^*k*. For each bucket,
/// the longest common bit-level prefix (LCP) shared by all consecutive
/// pairs within the bucket is computed (starting from the full first key).
/// A NUL sentinel byte is appended internally to ensure prefix-freeness.
/// The rank of a key is then reconstructed as
/// `bucket * bucket_size + offset`, where the bucket is determined by the
/// LCP and the offset is stored directly.
///
/// Internally, the structure contains two [`VFunc`]s:
/// - `offset_lcp_length`: maps each key (`str`) to a packed value
///   encoding the LCP bit-length and the offset within the bucket;
/// - `lcp2bucket`: maps each LCP bit-prefix ([`BitPrefix`]) to its bucket
///   index.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMinPerfHashFuncStr<S: Sig = [u64; 2], E: ShardEdge<S, 3> = FuseLge3Shards> {
    /// Number of keys.
    n: usize,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    offset_lcp_length: VFunc<str, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    lcp2bucket: VFunc<BitPrefix, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

impl<S: Sig, E: ShardEdge<S, 3>> LcpMinPerfHashFuncStr<S, E>
where
    str: ToSig<S>,
{
    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary
    /// (same contract as [`VFunc::get`]).
    #[inline]
    pub fn get(&self, key: &str) -> usize {
        let packed = self.offset_lcp_length.get(key);
        let lcp_bit_length = packed >> self.log2_bucket_size;
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        // Compute the lcp2bucket signature directly from the key bytes,
        // without allocating a BitPrefix. The LCP may extend into the
        // NUL sentinel (one byte past the key), so we need enough bytes.
        let key_bytes = key.as_bytes();
        let needed_bytes = lcp_bit_length.div_ceil(8);
        let seed = self.lcp2bucket.seed;
        let sig = if needed_bytes <= key_bytes.len() {
            bit_prefix_sig(key_bytes, lcp_bit_length, seed)
        } else {
            // LCP extends into the sentinel — use a small stack buffer.
            let mut buf = [0u8; 256];
            if key_bytes.len() <= 255 {
                buf[..key_bytes.len()].copy_from_slice(key_bytes);
                // buf[key_bytes.len()] is already 0 (the sentinel)
                bit_prefix_sig(&buf[..needed_bytes], lcp_bit_length, seed)
            } else {
                // Extremely long key — fall back to allocation.
                let mut extended = Vec::with_capacity(key_bytes.len() + 1);
                extended.extend_from_slice(key_bytes);
                extended.push(0);
                bit_prefix_sig(&extended, lcp_bit_length, seed)
            }
        };
        let bucket = self.lcp2bucket.get_by_sig(sig);
        (bucket << self.log2_bucket_size) + offset
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Returns the byte position of the first mismatch between two byte
/// slices, or the length of the shorter slice if one is a prefix of
/// the other. Uses `chunks_exact` to enable SIMD auto-vectorization.
#[inline]
fn mismatch(xs: &[u8], ys: &[u8]) -> usize {
    let off = std::iter::zip(xs.chunks_exact(128), ys.chunks_exact(128))
        .take_while(|(x, y)| x == y)
        .count()
        * 128;
    off + std::iter::zip(&xs[off..], &ys[off..])
        .take_while(|(x, y)| x == y)
        .count()
}

/// Returns the number of leading bits that are identical in two byte
/// slices, after appending a NUL sentinel byte to each. The sentinel
/// ensures prefix-freeness: two distinct strings always differ within
/// `min(a.len(), b.len()) * 8 + 8` bits.
fn lcp_bits_sentineled(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let pos = mismatch(&a[..min_len], &b[..min_len]);

    if pos < min_len {
        // Mismatch within the common part.
        pos * 8 + (a[pos] ^ b[pos]).leading_zeros() as usize
    } else if a.len() == b.len() {
        // Identical bytes — sentinels (both 0) also match.
        (min_len + 1) * 8
    } else {
        // Shorter string's sentinel (0) vs longer string's next byte.
        let next_byte = if a.len() > b.len() {
            a[min_len]
        } else {
            b[min_len]
        };
        if next_byte == 0 {
            (min_len + 1) * 8
        } else {
            min_len * 8 + next_byte.leading_zeros() as usize
        }
    }
}

#[cfg(feature = "rayon")]
impl<S, E> LcpMinPerfHashFuncStr<S, E>
where
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    str: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a new LCP-based monotone minimal perfect hash function.
    ///
    /// The keys must be provided in strictly increasing lexicographic
    /// order.
    ///
    /// # Arguments
    ///
    /// * `keys`: a lender yielding `&str` references in sorted order.
    ///
    /// * `n`: the number of keys.
    ///
    /// * `pl`: a progress logger.
    pub fn new(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend str>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if n == 0 {
            let empty_keys_str: Vec<&str> = vec![];
            let empty_keys_bp: Vec<BitPrefix> = vec![];
            let empty_vals: Vec<usize> = vec![];

            let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
                .try_build_func::<str, &str>(
                FromSlice::new(&empty_keys_str),
                FromSlice::new(&empty_vals),
                pl,
            )?;
            let lcp2bucket =
                VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                    .try_build_func::<BitPrefix, BitPrefix>(
                    FromSlice::new(&empty_keys_bp),
                    FromSlice::new(&empty_vals),
                    pl,
                )?;
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offset_lcp_length,
                lcp2bucket,
            });
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        // -- First pass: compute bit-level LCPs --
        //
        // For each bucket, the LCP is the minimum number of leading bits
        // shared by all consecutive pairs WITHIN the bucket. The initial
        // value is the bit-length of the first key (+ sentinel).

        let mut lcp_bit_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut bucket_first_strings: Vec<String> = Vec::with_capacity(num_buckets);

        let mut prev_key = String::new();
        let mut curr_lcp_bits: usize = 0;
        let mut i = 0usize;

        while let Some(key) = keys.next()? {
            let key: &str = key;

            if i > 0 && key <= prev_key.as_str() {
                bail!(
                    "Keys are not in strictly increasing lexicographic \
                     order at position {i}: {prev_key:?} >= {key:?}"
                );
            }

            let offset = i & bucket_mask;

            if offset == 0 {
                // First key of a new bucket.
                if i > 0 {
                    lcp_bit_lengths.push(curr_lcp_bits);
                }
                bucket_first_strings.push(key.to_owned());
                // Initialize to full key bit-length (including sentinel).
                curr_lcp_bits = (key.len() + 1) * 8;
            } else {
                // Subsequent key: minimize LCP.
                curr_lcp_bits =
                    curr_lcp_bits.min(lcp_bits_sentineled(key.as_bytes(), prev_key.as_bytes()));
            }

            prev_key.clear();
            prev_key.push_str(key);
            i += 1;
        }

        assert_eq!(i, n, "Expected {n} keys but got {i}");
        lcp_bit_lengths.push(curr_lcp_bits);
        assert_eq!(lcp_bit_lengths.len(), num_buckets);

        // -- Build packed values --

        let packed_values: Vec<usize> = (0..n)
            .map(|idx| {
                let bucket = idx >> log2_bs;
                let offset = idx & bucket_mask;
                (lcp_bit_lengths[bucket] << log2_bs) | offset
            })
            .collect();

        // -- Build offset_lcp_length VFunc --

        let keys = keys.rewind()?;

        let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
            .expected_num_keys(n)
            .try_build_func::<str, str>(keys, FromSlice::new(&packed_values), pl)?;

        // -- Build lcp2bucket VFunc --
        //
        // Each bucket's LCP key is a BitPrefix: the first
        // lcp_bit_lengths[b] bits of the sentineled first string.

        let sentineled_first_strings: Vec<Vec<u8>> = bucket_first_strings
            .iter()
            .map(|s| {
                let mut v = s.as_bytes().to_vec();
                v.push(0);
                v
            })
            .collect();

        let bit_prefixes: Vec<BitPrefix> = (0..num_buckets)
            .map(|b| BitPrefix::new(&sentineled_first_strings[b], lcp_bit_lengths[b]))
            .collect();

        let bucket_indices: Vec<usize> = (0..num_buckets).collect();

        let lcp2bucket =
            VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                .expected_num_keys(num_buckets)
                .try_build_func::<BitPrefix, BitPrefix>(
                    FromSlice::new(&bit_prefixes),
                    FromSlice::new(&bucket_indices),
                    pl,
                )?;

        Ok(Self {
            n,
            log2_bucket_size: log2_bs,
            offset_lcp_length,
            lcp2bucket,
        })
    }
}
