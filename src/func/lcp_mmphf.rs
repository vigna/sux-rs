/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! LCP-based monotone minimal perfect hash functions.
//!
//! We provide two variants: [`LcpMmphfInt`] works with any primitive integer
//! type, whereas [`LcpMmphf`] works with any byte-sequence key type
//! (`K: AsRef<[u8]>`). Type aliases [`LcpMmphfStr`] and [`LcpMmphfSliceU8`]
//! are provided for convenience.
//!
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
//! [Monotone minimal perfect hashing: Searching a sorted table with O(1)
//! accesses](https://dl.acm.org/doi/10.5555/1496770.1496856). In *Proceedings of
//! the 20th Annual ACM-SIAM Symposium On Discrete Mathematics (SODA)*, pages
//! 785−794, New York, 2009. ACM Press.

use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::shard_edge::{Fuse3NoShards, FuseLge3Shards, ShardEdge};
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use std::borrow::Borrow;
use xxhash_rust::xxh3;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    anyhow::{Result, bail},
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::convert::Infallible,
};

/// A 128-bit [random](https://www.random.org/) magic cookie appended to strings
/// to ensure prefix-freeness. Two distinct strings that are prefix-related will
/// diverge within these bytes. The probability of a real string containing this
/// exact sequence is 2⁻¹²⁸.
static MAGIC_COOKIE: [u8; 16] = [
    0xb6, 0x16, 0x1a, 0x72, 0xb1, 0xc4, 0x50, 0x11, 0x19, 0x02, 0xc6, 0xda, 0x23, 0x5b, 0xea, 0xdc,
];

/// A bit-level prefix of an integer, used as key for the LCP-to-bucket
/// mapping.
///
/// This type is public only because it appears in the signature of
/// [`LcpMmphfInt`].
///
/// Stores the original integer value and a bit length. The [`ToSig`]
/// implementation hashes only the top `bit_length` bits (by masking out
/// the bottom bits) followed by the bit length, directly on the stack
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

    /// Returns the value with the bottom `T::BITS - bit_length` bits
    /// masked out. When `bit_length == 0` (shift by `T::BITS` would
    /// overflow), returns `T::MIN`; this is harmless because the hash
    /// includes `bit_length` for disambiguation.
    ///
    /// Uses `T::MIN | T::MAX` as the all-ones value so that the mask
    /// is correct for both signed types (where `T::MAX` lacks the MSB)
    /// and unsigned types (where `T::MIN | T::MAX == T::MAX`).
    ///
    /// The `match` on `checked_shl` should compile to a branchless
    /// conditional move.
    #[inline]
    fn masked_value(&self) -> T {
        let all_ones = T::MIN | T::MAX;
        match all_ones.checked_shl(T::BITS - self.bit_length as u32) {
            Some(m) => self.value & m,
            None => T::MIN, // bit_length == 0
        }
    }
}

/// Packs significant bits and bit_length into a contiguous stack buffer
/// for one-shot xxh3 hashing. Returns the number of bytes written.
/// Buffer must be at least `size_of::<T>() + size_of::<usize>()` bytes.
#[inline]
fn pack_int_bit_prefix<T: PrimitiveInteger>(bp: &IntBitPrefix<T>, buf: &mut [u8]) -> usize {
    let val: T::Bytes = bp.masked_value().to_ne_bytes();
    let val = val.borrow() as &[u8];
    let len = bp.bit_length.to_ne_bytes();
    let n = val.len() + len.len();
    buf[..val.len()].copy_from_slice(val);
    buf[val.len()..n].copy_from_slice(&len);
    n
}

impl<T: PrimitiveInteger> ToSig<[u64; 1]> for IntBitPrefix<T> {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut buf = [0u8; 24];
        let n = pack_int_bit_prefix(key.borrow(), &mut buf);
        [xxh3::xxh3_64_with_seed(&buf[..n], seed)]
    }
}

// ── Helper functions (construction only) ─────────────────────────────

/// Returns the number of leading bits shared by two integers of the same
/// type, compared MSB-first (big-endian bit order).
#[cfg(feature = "rayon")]
#[inline(always)]
fn lcp_bits<T: PrimitiveInteger>(a: T, b: T) -> usize {
    (a ^ b).leading_zeros() as usize
}

/// Computes the log2 of bucket size from *n*.
#[cfg(feature = "rayon")]
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
/// The integers are divided into buckets. For each bucket, the longest common
/// bit-prefix shared by all consecutive pairs within the bucket is computed.
/// The rank of a key is then reconstructed as `bucket * bucket_size + offset`,
/// where the bucket is determined by the bit-prefix and the offset is stored
/// directly.
///
/// Internally, the structure contains two [`VFunc`]s:
/// - `offset_lcp_length`: maps each key (`T`) to a packed value encoding
///   the LCP bit-length and the offset within the bucket;
/// - `lcp2bucket`: maps each LCP bit-prefix ([`IntBitPrefix`]) to its
///   bucket index.
///
/// # Examples
///
/// The type annotation on the binding ensures that the default generic
/// parameters (`S = [u64; 2]`, `E = FuseLge3Shards`) are inferred:
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::LcpMmphfInt;
/// # use sux::utils::FromSlice;
/// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
///
/// let func: LcpMmphfInt<u64> =
///     LcpMmphfInt::new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, &key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMmphfInt<T: PrimitiveInteger, S = [u64; 2], E = FuseLge3Shards> {
    /// Number of keys.
    pub(crate) n: usize,
    /// Log2 of bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    pub(crate) offset_lcp_length: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket:
        VFunc<IntBitPrefix<T>, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

impl<T: PrimitiveInteger + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> LcpMmphfInt<T, S, E> {
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
        // XOR with T::MIN maps signed numeric order to bit-lexicographic
        // order by flipping the sign bit; for unsigned types T::MIN is 0,
        // so this is a no-op.
        let prefix = IntBitPrefix::new(key ^ T::MIN, lcp_bit_length);
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
impl<T, S, E> LcpMmphfInt<T, S, E>
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
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Like [`new`](Self::new), but uses the given [`VBuilder`] to
    /// configure the internal `offset_lcp_length` VFunc (e.g., for
    /// offline construction or thread control).
    pub fn new_with_builder(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::new_inner(keys, n, builder, false, pl).map(|(mmphf, _)| mmphf)
    }

    /// Internal constructor accepting a [`VBuilder`] and optionally
    /// returning the sig store.
    pub(crate) fn new_inner(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        keep_store: bool,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<(Self, Option<AnyShardStore<S, usize>>)> {
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
            return Ok((
                Self {
                    n: 0,
                    log2_bucket_size: 0,
                    offset_lcp_length,
                    lcp2bucket,
                },
                None,
            ));
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        pl.info(format_args!(
            "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
        ));

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

        // -- Build offset_lcp_length VFunc --

        pl.info(format_args!("Building key → (LCP length, offset) map..."));
        let keys = keys.rewind()?;

        let (offset_lcp_length, store) = builder
            .expected_num_keys(n)
            ._try_build_func::<T, T>(
                keys,
                FromIntoFallibleLenderFactory::new(|| {
                    Ok::<_, Infallible>(FromCloneableIntoIterator::new((0..n).map(|idx| {
                        (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                    })))
                })?,
                keep_store,
                pl,
            )?;

        // -- Build lcp2bucket VFunc --

        pl.info(format_args!("Building LCP prefix → bucket map ({num_buckets} buckets)..."));
        let lcp2bucket =
            VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                .expected_num_keys(num_buckets)
                .try_build_func::<IntBitPrefix<T>, IntBitPrefix<T>>(
                    FromIntoFallibleLenderFactory::new(|| {
                        Ok::<_, Infallible>(FromCloneableIntoIterator::new((0..num_buckets).map(
                            |b| {
                                IntBitPrefix::new(bucket_first_keys[b] ^ T::MIN, lcp_bit_lengths[b])
                            },
                        )))
                    })?,
                    FromIntoFallibleLenderFactory::new(|| {
                        Ok::<_, Infallible>(FromCloneableIntoIterator::new(0..num_buckets))
                    })?,
                    pl,
                )?;

        let total_bits =
            (offset_lcp_length.data.mem_size(SizeFlags::default()) +
             lcp2bucket.data.mem_size(SizeFlags::default())) * 8;
        pl.info(format_args!(
            "Actual bit cost per key: {:.2} ({total_bits} bits for {n} keys)",
            total_bits as f64 / n as f64
        ));

        Ok((
            Self {
                n,
                log2_bucket_size: log2_bs,
                offset_lcp_length,
                lcp2bucket,
            },
            store,
        ))
    }
}

/// A bit-level prefix of a byte slice, used as key for the LCP-to-bucket
/// mapping during construction.
///
/// This type is public only because it appears in the signature of
/// [`LcpMmphf`].
///
/// Holds an owned copy of the relevant bytes and a bit length. The
/// [`ToSig`] implementation hashes only the first `bit_length` bits,
/// masking out unused bits in the last partial byte.
///
/// This type is used only at construction time (to build the `lcp2bucket`
/// VFunc). At query time, signatures are computed directly from the key
/// bytes, avoiding any allocation.
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
    #[inline]
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
/// byte-sequence keys, based on longest common prefixes (LCPs).
///
/// Given *n* keys in lexicographic order, the structure maps each key to its
/// rank (0 to *n* − 1). The key type `K` must implement
/// [`AsRef<[u8]>`](core::convert::AsRef) so bytes can be extracted. Querying a
/// key not in the original set returns an arbitrary value (same contract as
/// [`VFunc`]).
///
/// # Implementation details
///
/// The keys are divided into buckets. For each bucket, the longest common
/// bit-level prefix (LCP) shared by all consecutive pairs within the bucket is
/// computed. A 128-bit magic cookie is appended internally to ensure
/// prefix-freeness. The rank of a key is then reconstructed as `bucket *
/// bucket_size + offset`, where the bucket is determined by the LCP and the
/// offset is stored directly.
///
/// Internally, the structure contains two [`VFunc`]s:
/// - `offset_lcp_length`: maps each key to a packed value encoding the
///   LCP bit-length and the offset within the bucket;
/// - `lcp2bucket`: maps each LCP bit-prefix ([`BitPrefix`]) to its bucket
///   index.
///
/// See [`LcpMmphfStr`] and [`LcpMmphfSliceU8`] for common instantiations.
///
/// # Examples
///
/// Build from sorted strings using the [`LcpMmphfStr`] alias. The type
/// annotation ensures that the default generic parameters are inferred:
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::LcpMmphfStr;
/// # use sux::utils::FromSlice;
/// let keys = vec![
///     "alpha".to_owned(),
///     "beta".to_owned(),
///     "delta".to_owned(),
///     "gamma".to_owned(),
/// ];
///
/// let func: LcpMmphfStr =
///     LcpMmphfStr::new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_str()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMmphf<K: ?Sized, S = [u64; 2], E = FuseLge3Shards> {
    /// Number of keys.
    pub(crate) n: usize,
    /// Log2 of bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    pub(crate) offset_lcp_length: VFunc<K, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket:
        VFunc<BitPrefix, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

/// A [`LcpMmphf`] for `str` keys.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::LcpMmphfStr;
/// # use sux::utils::FromSlice;
/// let keys = vec![
///     "alpha".to_owned(),
///     "beta".to_owned(),
///     "delta".to_owned(),
///     "gamma".to_owned(),
/// ];
///
/// let func: LcpMmphfStr =
///     LcpMmphfStr::new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_str()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type LcpMmphfStr<S = [u64; 2], E = FuseLge3Shards> = LcpMmphf<str, S, E>;

/// A [`LcpMmphf`] for `[u8]` keys.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::LcpMmphfSliceU8;
/// # use sux::utils::FromSlice;
/// let keys: Vec<Vec<u8>> = vec![
///     b"alpha".to_vec(),
///     b"beta".to_vec(),
///     b"delta".to_vec(),
///     b"gamma".to_vec(),
/// ];
///
/// let func: LcpMmphfSliceU8 = LcpMmphfSliceU8::new(
///     FromSlice::new(&keys),
///     keys.len(),
///     no_logging![],
/// )?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_slice()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type LcpMmphfSliceU8<S = [u64; 2], E = FuseLge3Shards> = LcpMmphf<[u8], S, E>;

impl<K: ?Sized + AsRef<[u8]> + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> LcpMmphf<K, S, E> {
    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary
    /// (same contract as [`VFunc::get`]).
    #[inline]
    pub fn get(&self, key: &K) -> usize {
        let packed = self.offset_lcp_length.get(key);
        let lcp_bit_length = packed >> self.log2_bucket_size;
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        // Compute the lcp2bucket signature by streaming the key bytes
        // and, if necessary, the magic cookie bytes into the hasher.
        // No allocation or copying needed.
        let key_bytes: &[u8] = key.as_ref();
        let seed = self.lcp2bucket.seed;

        let sig = if lcp_bit_length <= key_bytes.len() * 8 {
            // Fast path: LCP fits within the key bytes.
            bit_prefix_sig(key_bytes, lcp_bit_length, seed)
        } else {
            // Rare: LCP extends into the cookie. Stream key then cookie.
            let mut hasher = xxh3::Xxh3::with_seed(seed);
            hasher.update(key_bytes);
            let remaining_bits = lcp_bit_length - key_bytes.len() * 8;
            let cookie_full = remaining_bits / 8;
            hasher.update(&MAGIC_COOKIE[..cookie_full]);
            let extra_bits = remaining_bits % 8;
            if extra_bits > 0 {
                let mask = !((1u8 << (8 - extra_bits)) - 1);
                hasher.update(&[MAGIC_COOKIE[cookie_full] & mask]);
            }
            hasher.update(&lcp_bit_length.to_ne_bytes());
            [hasher.digest()]
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
#[cfg(feature = "rayon")]
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
/// slices, after conceptually appending [`MAGIC_COOKIE`] to each. The
/// cookie ensures prefix-freeness: two distinct strings that are
/// prefix-related will diverge within the cookie bytes.
///
/// The implementation first compares the string bytes (using vectorized
/// [`mismatch`]). If one string is a prefix of the other, it continues
/// comparing the shorter string's cookie extension against the longer
/// string's remaining bytes (and then its cookie extension).
///
/// The two strings must be distinct (the constructor enforces this).
#[cfg(feature = "rayon")]
fn lcp_bits_with_cookie(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let pos = mismatch(&a[..min_len], &b[..min_len]);

    if pos < min_len {
        // Mismatch within the common part — fast path.
        return pos * 8 + (a[pos] ^ b[pos]).leading_zeros() as usize;
    }

    // One string is a proper prefix of the other (they cannot be
    // identical because the constructor enforces strict ordering).
    let (longer, shorter_len) = if a.len() >= b.len() {
        (a, b.len())
    } else {
        (b, a.len())
    };

    debug_assert!(longer.len() > shorter_len);
    let extra = longer.len() - shorter_len;

    // Compare the shorter string's cookie extension (COOKIE[i])
    // against the longer string's continuation: first its actual
    // remaining bytes, then its own cookie extension (COOKIE[i - extra]).
    for (i, &cookie_byte) in MAGIC_COOKIE.iter().enumerate() {
        let longer_byte = if i < extra {
            longer[shorter_len + i]
        } else {
            MAGIC_COOKIE[i - extra]
        };
        if longer_byte != cookie_byte {
            return (shorter_len + i) * 8 + (longer_byte ^ cookie_byte).leading_zeros() as usize;
        }
    }

    unreachable!("the magic cookie guarantees prefix-freeness for distinct strings")
}

#[cfg(feature = "rayon")]
impl<K, S, E> LcpMmphf<K, S, E>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a new LCP-based monotone minimal perfect hash function for
    /// byte-sequence keys.
    ///
    /// The keys must be provided in strictly increasing lexicographic
    /// order (byte-level comparison).
    ///
    /// The lender may yield references to any type `B` that borrows as
    /// `K` (e.g., `&str` or `&String` for `K = str`; `&[u8]` or
    /// `&Vec<u8>` for `K = [u8]`).
    ///
    /// # Arguments
    ///
    /// * `keys`: a lender yielding key references in sorted order.
    ///
    /// * `n`: the number of keys.
    ///
    /// * `pl`: a progress logger.
    pub fn new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Like [`new`](Self::new), but uses the given [`VBuilder`] to
    /// configure the internal `offset_lcp_length` VFunc (e.g., for
    /// offline construction or thread control).
    pub fn new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::new_inner(keys, n, builder, false, pl).map(|(mmphf, _)| mmphf)
    }

    /// Internal constructor accepting a [`VBuilder`] and optionally
    /// returning the sig store.
    pub(crate) fn new_inner<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        keep_store: bool,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<(Self, Option<AnyShardStore<S, usize>>)> {
        if n == 0 {
            let empty_keys: Vec<&K> = vec![];
            let empty_keys_bp: Vec<BitPrefix> = vec![];
            let empty_vals: Vec<usize> = vec![];

            let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
                .try_build_func::<K, &K>(
                    FromSlice::new(&empty_keys),
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
            return Ok((
                Self {
                    n: 0,
                    log2_bucket_size: 0,
                    offset_lcp_length,
                    lcp2bucket,
                },
                None,
            ));
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        pl.info(format_args!(
            "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
        ));

        // -- First pass: compute bit-level LCPs --
        //
        // For each bucket, the LCP is the minimum number of leading bits
        // shared by all consecutive pairs WITHIN the bucket. The initial
        // value is the bit-length of the first key (+ cookie).

        let mut lcp_bit_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut bucket_first_keys: Vec<Vec<u8>> = Vec::with_capacity(num_buckets);

        let mut prev_key: Vec<u8> = Vec::new();
        let mut curr_lcp_bits: usize = 0;
        let mut i = 0usize;

        while let Some(key) = keys.next()? {
            let key_bytes: &[u8] = key.as_ref();

            if i > 0 && key_bytes <= prev_key.as_slice() {
                bail!(
                    "Keys are not in strictly increasing lexicographic \
                     order at position {i}"
                );
            }

            let offset = i & bucket_mask;

            if offset == 0 {
                // First key of a new bucket.
                if i > 0 {
                    lcp_bit_lengths.push(curr_lcp_bits);
                }
                bucket_first_keys.push(key_bytes.to_vec());
                // Initialize to full key bit-length (including cookie).
                curr_lcp_bits = (key_bytes.len() + MAGIC_COOKIE.len()) * 8;
            } else {
                // Subsequent key: minimize LCP.
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits_with_cookie(key_bytes, &prev_key));
            }

            prev_key.clear();
            prev_key.extend_from_slice(key_bytes);
            i += 1;
        }

        assert_eq!(i, n, "Expected {n} keys but got {i}");
        lcp_bit_lengths.push(curr_lcp_bits);
        assert_eq!(lcp_bit_lengths.len(), num_buckets);

        // -- Build offset_lcp_length VFunc --

        pl.info(format_args!("Building key → (LCP length, offset) map..."));
        let keys = keys.rewind()?;

        let (offset_lcp_length, store) = builder
            .expected_num_keys(n)
            ._try_build_func::<K, B>(
                keys,
                FromIntoFallibleLenderFactory::new(|| {
                    Ok::<_, Infallible>(FromCloneableIntoIterator::new((0..n).map(|idx| {
                        (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                    })))
                })?,
                keep_store,
                pl,
            )?;

        // -- Build lcp2bucket VFunc --
        //
        // Each bucket's LCP key is a BitPrefix: the first
        // lcp_bit_lengths[b] bits of the first key extended with the
        // magic cookie.

        pl.info(format_args!("Building LCP prefix → bucket map ({num_buckets} buckets)..."));
        let extended_first_keys: Vec<Vec<u8>> = bucket_first_keys
            .iter()
            .map(|k| {
                let mut v = Vec::with_capacity(k.len() + MAGIC_COOKIE.len());
                v.extend_from_slice(k);
                v.extend_from_slice(&MAGIC_COOKIE);
                v
            })
            .collect();

        let lcp2bucket =
            VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                .expected_num_keys(num_buckets)
                .try_build_func::<BitPrefix, BitPrefix>(
                    FromIntoFallibleLenderFactory::new(|| {
                        Ok::<_, Infallible>(FromCloneableIntoIterator::new(
                            (0..num_buckets).map(|b| {
                                BitPrefix::new(&extended_first_keys[b], lcp_bit_lengths[b])
                            }),
                        ))
                    })?,
                    FromIntoFallibleLenderFactory::new(|| {
                        Ok::<_, Infallible>(FromCloneableIntoIterator::new(0..num_buckets))
                    })?,
                    pl,
                )?;

        let total_bits =
            (offset_lcp_length.data.mem_size(SizeFlags::default()) +
             lcp2bucket.data.mem_size(SizeFlags::default())) * 8;
        pl.info(format_args!(
            "Actual bit cost per key: {:.2} ({total_bits} bits for {n} keys)",
            total_bits as f64 / n as f64
        ));

        Ok((
            Self {
                n,
                log2_bucket_size: log2_bs,
                offset_lcp_length,
                lcp2bucket,
            },
            store,
        ))
    }
}
