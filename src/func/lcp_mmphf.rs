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
//! These structures implement the [`TryIntoUnaligned`] trait, allowing them to
//! be converted into (usually faster) structures using unaligned access.
//!
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
//! [Monotone minimal perfect hashing: Searching a sorted table with O(1)
//! accesses](https://dl.acm.org/doi/10.5555/1496770.1496856). In *Proceedings of
//! the 20th Annual ACM-SIAM Symposium On Discrete Mathematics (SODA)*, pages
//! 785−794, New York, 2009. ACM Press.

use crate::bits::BitFieldVec;
use crate::bits::BitFieldVecU;
use crate::func::VFunc;
use crate::func::shard_edge::{Fuse3NoShards, FuseLge3Shards, ShardEdge};
use crate::traits::TryIntoUnaligned;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use std::borrow::Borrow;
use value_traits::slices::SliceByValue;
use xxhash_rust::xxh3;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    anyhow::{Result, bail},
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
};

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
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntBitPrefix<T> {
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
pub(crate) fn pack_int_bit_prefix<T: PrimitiveInteger>(
    bp: &IntBitPrefix<T>,
    buf: &mut [u8],
) -> usize {
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
pub(crate) fn lcp_bits<T: PrimitiveInteger>(a: T, b: T) -> usize {
    (a ^ b).leading_zeros() as usize
}

/// Computes the log2 of bucket size from *n*.
#[cfg(feature = "rayon")]
pub(crate) fn log2_bucket_size(n: usize) -> usize {
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
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
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
///     LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
pub struct LcpMmphfInt<
    T: PrimitiveInteger,
    D: SliceByValue = BitFieldVec<Box<[usize]>>,
    S = [u64; 2],
    E = FuseLge3Shards,
> {
    /// Number of keys.
    pub(crate) n: usize,
    /// Log2 of bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    pub(crate) offset_lcp_length: VFunc<T, D, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket: VFunc<IntBitPrefix<T>, D, [u64; 1], Fuse3NoShards>,
}

impl<T: PrimitiveInteger + ToSig<S>, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>>
    LcpMmphfInt<T, D, S, E>
{
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
}

impl<T: PrimitiveInteger, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>>
    LcpMmphfInt<T, D, S, E>
{
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
impl<T, S, E> LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a new LCP-based monotone minimal perfect hash function
    /// for integers using default [`VBuilder`] settings.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder). Use that
    /// method if you need to configure construction parameters such
    /// as offline mode, thread count, or sharding overhead.
    ///
    /// The keys must be provided in strictly increasing order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::LcpMmphfInt;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: LcpMmphfInt<u64> =
    ///     LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    ///
    /// for (i, &key) in keys.iter().enumerate() {
    ///     assert_eq!(func.get(key), i);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Creates a new LCP-based monotone minimal perfect hash function
    /// for integers using the given [`VBuilder`] configuration.
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    ///
    /// The keys must be provided in strictly increasing order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{LcpMmphfInt, VBuilder};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: LcpMmphfInt<u64> = LcpMmphfInt::try_new_with_builder(
    ///     FromSlice::new(&keys),
    ///     keys.len(),
    ///     VBuilder::default().offline(true),
    ///     no_logging![],
    /// )?;
    ///
    /// for (i, &key) in keys.iter().enumerate() {
    ///     assert_eq!(func.get(key), i);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new_with_builder(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_inner(keys, n, builder, true, pl).map(|(mmphf, _)| mmphf)
    }

    /// Internal constructor accepting a [`VBuilder`] and optionally
    /// returning the sig store.
    pub(crate) fn try_new_inner(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        drain_store: bool,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<(Self, Box<dyn ShardStore<S, usize> + Send + Sync>)> {
        if n == 0 {
            return Ok((
                Self {
                    n: 0,
                    log2_bucket_size: 0,
                    offset_lcp_length: VFunc::empty(),
                    lcp2bucket: VFunc::empty(),
                },
                Box::new(
                    crate::utils::sig_store::new_online::<S, usize>(0, 0, None)
                        .expect("empty online store")
                        .into_shard_store(0)
                        .expect("empty shard store"),
                ),
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

        let (offset_lcp_length, store) =
            builder
                .expected_num_keys(n)
                .try_build_func_and_store::<T, T, _>(
                    keys,
                    FromCloneableIntoIterator::new((0..n).map(|idx| {
                        (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                    })),
                    BitFieldVec::<Box<[usize]>>::new_unaligned,
                    drain_store,
                    pl,
                )?;

        // -- Build lcp2bucket VFunc --

        pl.info(format_args!(
            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
        ));
        let lcp2bucket = <VFunc<
            IntBitPrefix<T>,
            BitFieldVec<Box<[usize]>>,
            [u64; 1],
            Fuse3NoShards,
        >>::try_new_with_builder(
            FromCloneableIntoIterator::new(
                (0..num_buckets)
                    .map(|b| IntBitPrefix::new(bucket_first_keys[b] ^ T::MIN, lcp_bit_lengths[b])),
            ),
            FromCloneableIntoIterator::new(0..num_buckets),
            num_buckets,
            VBuilder::default(),
            pl,
        )?;

        let result = Self {
            n,
            log2_bucket_size: log2_bs,
            offset_lcp_length,
            lcp2bucket,
        };
        let total_bits = result.mem_size(SizeFlags::default()) * 8;
        pl.info(format_args!(
            "Actual bit cost per key: {:.2} ({total_bits} bits for {n} keys)",
            total_bits as f64 / n as f64
        ));

        Ok((result, store))
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
pub(crate) fn hash_bit_prefix_raw(hasher: &mut xxh3::Xxh3, bytes: &[u8], bit_length: usize) {
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
pub(crate) fn bit_prefix_sig(bytes: &[u8], bit_length: usize, seed: u64) -> [u64; 1] {
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
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Implementation details
///
/// The keys are divided into buckets. For each bucket, the longest common
/// bit-level prefix (LCP) shared by all consecutive pairs within the bucket is
/// computed. A virtual NUL byte is appended internally to ensure
/// prefix-freeness (keys must not contain ASCII NUL). The rank of a key is
/// then reconstructed as `bucket * bucket_size + offset`, where the bucket is
/// determined by the LCP and the offset is stored directly.
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
///     LcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
pub struct LcpMmphf<
    K: ?Sized,
    D: SliceByValue = BitFieldVec<Box<[usize]>>,
    S = [u64; 2],
    E = FuseLge3Shards,
> {
    /// Number of keys.
    pub(crate) n: usize,
    /// Log2 of bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    pub(crate) offset_lcp_length: VFunc<K, D, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket: VFunc<BitPrefix, D, [u64; 1], Fuse3NoShards>,
}

/// A [`LcpMmphf`] for `str` keys.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
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
///     LcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_str()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type LcpMmphfStr<D = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    LcpMmphf<str, D, S, E>;

/// A [`LcpMmphf`] for `[u8]` keys.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
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
/// let func: LcpMmphfSliceU8 = LcpMmphfSliceU8::try_new(
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
pub type LcpMmphfSliceU8<D = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    LcpMmphf<[u8], D, S, E>;

impl<K: ?Sized + AsRef<[u8]> + ToSig<S>, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>>
    LcpMmphf<K, D, S, E>
{
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
        // and, if necessary, the virtual NUL byte into the hasher.
        // No allocation or copying needed.
        let key_bytes: &[u8] = key.as_ref();
        let seed = self.lcp2bucket.seed;

        let sig = if lcp_bit_length <= key_bytes.len() * 8 {
            // Fast path: LCP fits within the key bytes.
            bit_prefix_sig(key_bytes, lcp_bit_length, seed)
        } else {
            // Rare: LCP extends into the virtual NUL (at most 8 extra bits).
            let mut hasher = xxh3::Xxh3::with_seed(seed);
            hasher.update(key_bytes);
            // The virtual NUL is 0x00, so we feed a zero byte.
            hasher.update(&[0u8]);
            hasher.update(&lcp_bit_length.to_ne_bytes());
            [hasher.digest()]
        };
        let bucket = self.lcp2bucket.get_by_sig(sig);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<K: ?Sized, D: SliceByValue, S: Sig, E: ShardEdge<S, 3>> LcpMmphf<K, D, S, E> {
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
pub(crate) fn mismatch(xs: &[u8], ys: &[u8]) -> usize {
    let off = std::iter::zip(xs.chunks_exact(128), ys.chunks_exact(128))
        .take_while(|(x, y)| x == y)
        .count()
        * 128;
    off + std::iter::zip(&xs[off..], &ys[off..])
        .take_while(|(x, y)| x == y)
        .count()
}

/// Returns the number of leading bits that are identical in two byte
/// slices, after conceptually appending a NUL byte to each. Since keys
/// must not contain NUL, two distinct prefix-related strings are
/// guaranteed to diverge at the NUL position.
///
/// The implementation first compares the string bytes (using vectorized
/// [`mismatch`]). If one string is a prefix of the other, the virtual
/// NUL (0x00) XOR the next byte of the longer string yields that byte
/// itself, and `leading_zeros` gives the additional shared bits.
///
/// The two strings must be distinct (the constructor enforces this).
#[cfg(feature = "rayon")]
pub(crate) fn lcp_bits_nul(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let pos = mismatch(&a[..min_len], &b[..min_len]);

    if pos < min_len {
        // Mismatch within the common part — fast path.
        return pos * 8 + (a[pos] ^ b[pos]).leading_zeros() as usize;
    }

    // One string is a proper prefix of the other (they cannot be
    // identical because the constructor enforces strict ordering).
    let longer = if a.len() >= b.len() { a } else { b };
    debug_assert!(longer.len() > min_len);

    // The virtual NUL after the shorter string diverges from the next
    // byte of the longer string (which is guaranteed non-NUL).
    min_len * 8 + longer[min_len].leading_zeros() as usize
}

#[cfg(feature = "rayon")]
impl<K, S, E> LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a new LCP-based monotone minimal perfect hash function
    /// for byte-sequence keys using default [`VBuilder`] settings.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder). Use that
    /// method if you need to configure construction parameters such
    /// as offline mode, thread count, or sharding overhead.
    ///
    /// The keys must be in strictly increasing lexicographic order.
    /// The lender may yield references to any type `B` that borrows
    /// as `K` (e.g., `&String` for `K = str`, `&Vec<u8>` for
    /// `K = [u8]`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::LcpMmphfStr;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: LcpMmphfStr =
    ///     LcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    ///
    /// for (i, &key) in keys.iter().enumerate() {
    ///     assert_eq!(func.get(key), i);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Creates a new LCP-based monotone minimal perfect hash function
    /// for byte-sequence keys using the given [`VBuilder`]
    /// configuration.
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    ///
    /// The keys must be in strictly increasing lexicographic order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{LcpMmphfStr, VBuilder};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: LcpMmphfStr = LcpMmphfStr::try_new_with_builder(
    ///     FromSlice::new(&keys),
    ///     keys.len(),
    ///     VBuilder::default().offline(true),
    ///     no_logging![],
    /// )?;
    ///
    /// for (i, &key) in keys.iter().enumerate() {
    ///     assert_eq!(func.get(key), i);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_inner(keys, n, builder, true, pl).map(|(mmphf, _)| mmphf)
    }

    /// Internal constructor accepting a [`VBuilder`] and optionally
    /// returning the sig store.
    pub(crate) fn try_new_inner<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        drain_store: bool,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<(Self, Box<dyn ShardStore<S, usize> + Send + Sync>)> {
        if n == 0 {
            return Ok((
                Self {
                    n: 0,
                    log2_bucket_size: 0,
                    offset_lcp_length: VFunc::empty(),
                    lcp2bucket: VFunc::empty(),
                },
                Box::new(
                    crate::utils::sig_store::new_online::<S, usize>(0, 0, None)
                        .expect("empty online store")
                        .into_shard_store(0)
                        .expect("empty shard store"),
                ),
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
        // value is the bit-length of the first key (+ virtual NUL).

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
                // Initialize to full key bit-length (including virtual NUL).
                curr_lcp_bits = (key_bytes.len() + 1) * 8;
            } else {
                // Subsequent key: minimize LCP.
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits_nul(key_bytes, &prev_key));
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

        let (offset_lcp_length, store) =
            builder
                .expected_num_keys(n)
                .try_build_func_and_store::<K, B, _>(
                    keys,
                    FromCloneableIntoIterator::new((0..n).map(|idx| {
                        (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                    })),
                    BitFieldVec::<Box<[usize]>>::new_unaligned,
                    drain_store,
                    pl,
                )?;

        // -- Build lcp2bucket VFunc --
        //
        // Each bucket's LCP key is a BitPrefix: the first
        // lcp_bit_lengths[b] bits of the first key extended with a
        // virtual NUL.

        pl.info(format_args!(
            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
        ));
        let extended_first_keys: Vec<Vec<u8>> = bucket_first_keys
            .iter()
            .map(|k| {
                let mut v = Vec::with_capacity(k.len() + 1);
                v.extend_from_slice(k);
                v.push(0x00);
                v
            })
            .collect();

        let lcp2bucket = <VFunc<
            BitPrefix,
            BitFieldVec<Box<[usize]>>,
            [u64; 1],
            Fuse3NoShards,
        >>::try_new_with_builder(
            FromCloneableIntoIterator::new(
                (0..num_buckets)
                    .map(|b| BitPrefix::new(&extended_first_keys[b], lcp_bit_lengths[b])),
            ),
            FromCloneableIntoIterator::new(0..num_buckets),
            num_buckets,
            VBuilder::default(),
            pl,
        )?;

        let result = Self {
            n,
            log2_bucket_size: log2_bs,
            offset_lcp_length,
            lcp2bucket,
        };
        let total_bits = result.mem_size(SizeFlags::default()) * 8;
        pl.info(format_args!(
            "Actual bit cost per key: {:.2} ({total_bits} bits for {n} keys)",
            total_bits as f64 / n as f64
        ));

        Ok((result, store))
    }
}

// ── Aligned ↔ Unaligned conversions ──────────────────────────────────

// -- LcpMmphfInt --

impl<T: PrimitiveInteger, S: Sig, E: ShardEdge<S, 3>>
    From<LcpMmphfInt<T, BitFieldVecU<Box<[usize]>>, S, E>>
    for LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>
{
    fn from(f: LcpMmphfInt<T, BitFieldVecU<Box<[usize]>>, S, E>) -> Self {
        LcpMmphfInt {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            offset_lcp_length: f.offset_lcp_length.into(),
            lcp2bucket: f.lcp2bucket.into(),
        }
    }
}

impl<T: PrimitiveInteger, S: Sig, E: ShardEdge<S, 3>> TryIntoUnaligned
    for LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>
{
    type Unaligned = LcpMmphfInt<T, BitFieldVecU<Box<[usize]>>, S, E>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(LcpMmphfInt {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            offset_lcp_length: self.offset_lcp_length.try_into_unaligned()?,
            lcp2bucket: self.lcp2bucket.try_into_unaligned()?,
        })
    }
}

// -- LcpMmphf --

impl<K: ?Sized, S: Sig, E: ShardEdge<S, 3>> From<LcpMmphf<K, BitFieldVecU<Box<[usize]>>, S, E>>
    for LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>
{
    fn from(f: LcpMmphf<K, BitFieldVecU<Box<[usize]>>, S, E>) -> Self {
        LcpMmphf {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            offset_lcp_length: f.offset_lcp_length.into(),
            lcp2bucket: f.lcp2bucket.into(),
        }
    }
}

impl<K: ?Sized, S: Sig, E: ShardEdge<S, 3>> TryIntoUnaligned
    for LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>
{
    type Unaligned = LcpMmphf<K, BitFieldVecU<Box<[usize]>>, S, E>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(LcpMmphf {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            offset_lcp_length: self.offset_lcp_length.try_into_unaligned()?,
            lcp2bucket: self.lcp2bucket.try_into_unaligned()?,
        })
    }
}
